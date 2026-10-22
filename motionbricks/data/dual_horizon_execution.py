"""Genuine long Root terrain hypotheses with native8-token near-term execution."""
import copy
from dataclasses import dataclass

import torch

from .local_terrain_observation import LocalTerrainObservation,trained_query_controls
from .native_mb_online_conditions import velocity_direction_local
from .native_mb_terrain_protocol import terrain_observation_config
from .pfnn_tuning import TunablePFNNBackend
from .root_path_split_execution import SplitRootPathExecution,VOLUME_KEYS


SCHEMA='motionbricks.dual_horizon_execution.v1'
NEAR_FRAMES=(4,12,20,27)
FAR_FRAMES={12:(32,36,40,43),16:(32,41,50,59)}


def policy(tokens):return dict(schema=SCHEMA,lookahead_tokens=tokens,execute_tokens=8)


def validate_policy(value):
    if not isinstance(value,dict) or value!=policy(value.get('lookahead_tokens')) or type(value['lookahead_tokens']) is not int or type(value['execute_tokens']) is not int or value['lookahead_tokens'] not in (8,12,16):
        raise ValueError('Expected the declared dual-horizon8/12/16preview and8execution policy')
    return dict(value)


class DualHorizonExecution(SplitRootPathExecution):
    @classmethod
    def from_runner(cls,runner):
        result=copy.copy(runner);result.__class__=cls;result.validation=[]
        return result

    def _stations(self,history,short_root,long_root,tokens):
        paths=[];centers=[];rights=[]
        for root,count,indices in ((short_root,8,NEAR_FRAMES),(long_root,tokens,FAR_FRAMES[tokens])):
            xz=self.inference.global_motion_rep.unnormalize(root)[:,:count*4,[0,2]]+history[:,:1,[0,2]]
            path=torch.cat((history[:,-1:,[0,2]],xz[:,4:count*4-4]),1)
            all_centers,all_right=self.observer._curve(path,path.shape[1],history)
            selected=history.new_tensor([frame-3 for frame in indices],dtype=torch.long)
            centers.append(all_centers.index_select(1,selected));rights.append(all_right.index_select(1,selected))
            paths.append(path)
        return (torch.cat(centers,1),torch.cat(rights,1)),paths

    def prepare_dual(self,history,goal,query,tokens):
        coarse=self._scene_inputs(history,goal,8,query)
        short=self.graphs[8][0](history,goal,query,*coarse)
        long=self.graphs[tokens][0](history,goal,query,*coarse)
        return self._prepare_observation(history,short,long,coarse,tokens)

    def _prepare_observation(self,history,short,long,coarse,tokens):
        stations,paths=self._stations(history,short['root'],long['root'],tokens)
        self.controller.frames=32
        volume,height,path=self.observer.sample(history,self.scene,short['root'],dict(zip(VOLUME_KEYS,coarse[8:])),
            LocalTerrainObservation(*coarse[:4]).as_model_inputs(),reuse_static_regions=True,corridor_stations=stations)
        return dict(short=short,long=long,coarse=coarse,volume=volume,height=height,path=path,
                    stations=stations,paths=paths,lookahead_tokens=tokens)

    def compiled_suffix(self,history,goal,query,prepared):
        short,coarse,height,volume=(prepared[k] for k in ('short','coarse','height','volume'))
        result=self.graphs[8][1](history,goal,query,short['count'],short['root'],short['local'],*coarse[:8],
            height.points,height.valid,height.world_query_xz,height.query_envelope_m,*(volume[k] for k in VOLUME_KEYS))
        config=terrain_observation_config(self.inference._root_model.args['observation_range_policy'])
        direction=velocity_direction_local(history[...,:5],goal,28/30)
        result.update(survey_points=coarse[4],survey_valid=coarse[5],survey_world_query_xz=coarse[6],
            actor_query_controls=trained_query_controls(query,direction,config))
        self.observer.path=prepared['path'];self.observer.height=height
        return result

    @torch.inference_mode()
    def predict_dual(self,history,goal,diagnostics,*,lookahead_tokens,num_tokens=8,query_controls=None,path_refinements=0):
        if num_tokens!=8 or path_refinements:raise ValueError('Dual-horizon execution remains8tokens with0refinement')
        if lookahead_tokens==8:return super().predict(history,goal,diagnostics,num_tokens=8,query_controls=query_controls)
        if lookahead_tokens not in FAR_FRAMES:raise ValueError('Unsupported preview horizon')
        query=history.new_tensor([[-1.,.6,1.,.4]]) if query_controls is None else query_controls
        prepared=self.prepare_dual(history,goal,query,lookahead_tokens)
        result=self.compiled_suffix(history,goal,query,prepared)
        motion,count=self._describe_prediction(history,goal,diagnostics,result,8)
        centers=prepared['stations'][0][0]
        speed=goal[0,:2].norm()
        heading=history[0,-1,3:5]
        direction=torch.where(speed>1e-5,goal[0,:2]/speed.clamp_min(1e-5),torch.stack((heading[1],heading[0])))
        displacement=centers-history[0,-1,[0,2]]
        near_end=prepared['paths'][0][0,-1]
        disagreement=(prepared['paths'][0]-prepared['paths'][1][:,:25]).norm(dim=-1)
        preview=self._root_world(history,dict(result,root=prepared['long']['root']))[:,4:lookahead_tokens*4-4,:3]
        diagnostics.update(planning_policy=policy(lookahead_tokens),terrain_sampling='dual_root_station_sets',
            root_neural_passes=2,pose_passes=4,execute_tokens=8,lookahead_tokens=lookahead_tokens,
            query_proposal_world_xz_m=prepared['path'][0].cpu().tolist(),
            preview_root_world_xz_m=prepared['paths'][1][0].cpu().tolist(),
            preview_root_world_xyz_m=preview[0].cpu().tolist(),
            station_source_frames=list(NEAR_FRAMES)+list(FAR_FRAMES[lookahead_tokens]),
            station_world_xz_m=centers.cpu().tolist(),station_forward_m=(displacement@direction).cpu().tolist(),
            far_from_short_endpoint_forward_m=((centers[4:]-near_end)@direction).cpu().tolist(),
            near_root_disagreement_max_m=float(disagreement.max()),coarse_snapshot='shared_original8')
        diagnostics['unified_sdf']['region_names']=['body','short_and_long_root_stations','actual_velocity']
        return motion,count


@dataclass(frozen=True)
class ExecutionView:
    engine:DualHorizonExecution
    lookahead_tokens:int

    neural_execution='dual-horizon-split-cuda'
    def __getattr__(self,name):return getattr(self.engine,name)
    def supports(self,controls):return controls is not None and controls.get('num_tokens')==8 and self.engine.supports(controls)
    def predict(self,*a,**kw):return self.engine.predict_dual(*a,lookahead_tokens=self.lookahead_tokens,**kw)


class DualHorizonBackend(TunablePFNNBackend):
    @classmethod
    def from_backend(cls,backend,engine):
        result=copy.copy(backend);result.__class__=cls
        result.history=backend.history.clone();result.inference_config=dict(backend.inference_config)
        result.runtime_controls=dict(backend.runtime_controls);result.dual_engine=engine
        result.planning_policy=policy(8);result.execution=ExecutionView(engine,8)
        return result

    def configure_runtime(self,values):
        if values['num_tokens']!=8 or values['inference_steps']!=4 or values['refresh_frames']!=4:
            raise ValueError('Candidate has8execution tokens,4Pose passes and4frame refresh')
        super().configure_runtime(values)

    def configure_policy(self,values):
        values=validate_policy(values)
        if values!=self.planning_policy:
            self.planning_policy=values;self.execution=ExecutionView(self.dual_engine,values['lookahead_tokens'])
            self._future_buffer=self._last_published_history=self._last_command=self._debug_future=None

    def _predict_tuned(self,*a,**kw):
        result=super()._predict_tuned(*a,**kw)
        self.last_generation.update(planning_policy=dict(self.planning_policy),execute_tokens=8,
            lookahead_tokens=self.planning_policy['lookahead_tokens'],predicted_frames=32,
            root_passes_per_replan=1 if self.planning_policy['lookahead_tokens']==8 else 2)
        return result

    def commit_prepared_plan(self,candidate):
        settings=validate_policy(candidate['generation']['planning_policy'])
        if candidate['generation']['runtime_controls']['num_tokens']!=8 or candidate['generation']['predicted_frames']!=32:
            raise ValueError('Candidate metadata must describe32actual motion frames')
        super().commit_prepared_plan(candidate)
        self.planning_policy=settings;self.execution=ExecutionView(self.dual_engine,settings['lookahead_tokens'])

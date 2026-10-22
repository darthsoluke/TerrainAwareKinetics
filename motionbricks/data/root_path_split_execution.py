"""Bounded horizon graphs sharing frozen weights and native scene queries."""

import torch

from .local_terrain_observation import LocalTerrainObservation,trained_query_controls
from .native_mb_online_conditions import velocity_direction_local
from .native_mb_online_live import online_command_goal,predict_planar_terrain
from .native_mb_terrain_protocol import terrain_observation_config
from .root_path_terrain import RootPathTerrainExecution
from .soma_cuda_graph import StaticCudaCall
from .soma_reusable_terrain_execution import ReusableTerrainExecution
from .soma_static_execution import trace_fixed_call,restore_native_transformers,use_dense_transformers
from .soma_terrain_execution import SETTINGS


VOLUME_KEYS=('points','distance','normal','valid','features')
HORIZONS=(8,12,16)


class _RootBoundary(Exception):
    def __init__(self,values):self.values=values


class SplitRootPathExecution(RootPathTerrainExecution):
    """Root once -> current scene query -> four Pose passes, for each horizon."""

    @torch.inference_mode()
    def __init__(self,inference,scene,skinner,seed,*,controller):
        self.graphs={}
        super().__init__(inference,scene,skinner,seed,controller=controller)
        self.flexible=True
        self.neural_execution='split-cuda-graphs'
        self.bind_scene(scene,seed)

    _scene_inputs=ReusableTerrainExecution._scene_inputs

    def supports(self,controls):
        return (controls is not None and controls.get('num_tokens') in HORIZONS
                and controls.get('inference_steps')==4 and controls.get('path_refinements')==0)

    @torch.jit.optimized_execution(False)
    def _build_graphs(self,tokens,history,goal,query):
        inference,controller=self.inference,self.controller
        original_root=inference._predict_root_trajectories
        original_decode=inference._vqvae_pose_model.forward_decoder
        captured={}
        root_transformer=inference._root_model.backbone_net._root_token_transformer_model
        dense_forward=None

        def decode(ids,*args,**kwargs):
            captured['tokens']=ids.clone()
            return original_decode(ids,*args,**kwargs)

        def run(history,goal,query,values):
            volume=dict(zip(VOLUME_KEYS,values[8:]))
            with controller.context(history,self.scene,volume_observation=volume):
                return predict_planar_terrain(inference,history,goal,self.scene,num_tokens=tokens,
                    config=SETTINGS,query_controls=query,
                    prepared_observations=(LocalTerrainObservation(*values[:4]),LocalTerrainObservation(*values[4:8])))

        def root_segment(history,goal,query,*values):
            def boundary(batch,config):raise _RootBoundary(original_root(batch,config))
            inference._predict_root_trajectories=boundary
            self.observer.deferred=True
            try:
                try:run(history,goal,query,values)
                except _RootBoundary as stop:
                    count,root,local=stop.values
                    return dict(count=count,root=root,local=local)
            finally:
                inference._predict_root_trajectories=original_root
                self.observer.deferred=False

        def pose_segment(history,goal,query,count,root,local,*values):
            coarse,height_values,volume_values=values[:8],values[8:12],values[12:]
            volume=dict(zip(VOLUME_KEYS,volume_values))
            height=LocalTerrainObservation(*height_values)

            def supplied_root(batch,config):
                controller.last_root=dict(pred_global_root_values=root)
                self.observer.height=height
                self.observer.context=controller._context
                return count,root,local

            inference._predict_root_trajectories=supplied_root
            try:motion,predicted_count=run(history,goal,query,(*coarse,*volume_values))
            finally:inference._predict_root_trajectories=original_root
            return dict(motion=motion,count=predicted_count,root=root,tokens=captured['tokens'],
                points=height.points,valid=height.valid,world_query_xz=height.world_query_xz,
                sdf_points=volume['points'],sdf_distance=volume['distance'],sdf_valid=volume['valid'])

        inference._vqvae_pose_model.forward_decoder=decode
        try:
            use_dense_transformers(inference)
            dense_forward=root_transformer.forward
            active_length=1+2*inference._root_model.backbone_net.get_num_frames_per_token()+tokens
            def packed_root(src,mask=None,src_key_padding_mask=None,is_causal=None):
                expected_mask=torch.arange(src.shape[1],device=src.device)[None]>=active_length
                if mask is not None or not torch.equal(src_key_padding_mask,expected_mask):
                    raise ValueError('Root horizon padding differs from its captured prefix')
                # Batch one has a fixed native NestedTensor prefix at each horizon.
                # Execute that unpadded sequence, then restore its original zero tail.
                active=dense_forward(src[:,:active_length],mask=None,src_key_padding_mask=None,is_causal=False)
                return torch.nn.functional.pad(active,(0,0,0,src.shape[1]-active_length))
            root_transformer.forward=packed_root
            coarse=self._scene_inputs(history,goal,tokens,query)
            root_call=StaticCudaCall(trace_fixed_call(root_segment,(history,goal,query,*coarse),shared_cuda_constants=True))
            root=root_call(history,goal,query,*coarse)
            volume,height,_=self._path_inputs(history,root['root'],coarse,tokens)
            inputs=(history,goal,query,root['count'],root['root'],root['local'],*coarse[:8],
                height.points,height.valid,height.world_query_xz,height.query_envelope_m,*(volume[k] for k in VOLUME_KEYS))
            pose_call=StaticCudaCall(trace_fixed_call(pose_segment,inputs,shared_cuda_constants=True))
            pose_call(*inputs)
            self.graphs[tokens]=(root_call,pose_call)
        finally:
            if dense_forward is not None:root_transformer.forward=dense_forward
            inference._predict_root_trajectories=original_root
            inference._vqvae_pose_model.forward_decoder=original_decode
            self.observer.deferred=False

    @torch.inference_mode()
    def bind_scene(self,scene,seed):
        super().bind_scene(scene,seed)
        if self.neural_execution=='native':return
        inference=self.inference
        original_decode=inference._vqvae_pose_model.forward_decoder
        captured={}
        def decode(ids,*args,**kwargs):
            captured['tokens']=ids.clone()
            return original_decode(ids,*args,**kwargs)
        self.validation=[]
        try:
            for tokens in HORIZONS:
                cases=[]
                inference._vqvae_pose_model.forward_decoder=decode
                restore_native_transformers(inference)
                for velocity,query_values in (([0.,0.],[-1.,.6,1.,.4]),([0.,3.],[4.2,.9,1.2,.3])):
                    goal=online_command_goal(seed[...,:5],velocity,0.,tokens,self.protocol_version)
                    query=seed.new_tensor([query_values])
                    with self.controller.context(seed,scene):
                        expected=RootPathTerrainExecution._prediction(self,seed,goal,num_tokens=tokens,query_controls=query)
                    expected['tokens']=captured['tokens'].clone()
                    assert self.controller.frames==tokens*4
                    cases.append((goal,query,expected))
                inference._vqvae_pose_model.forward_decoder=original_decode
                if tokens not in self.graphs:self._build_graphs(tokens,seed,cases[0][0],cases[0][1])
                for goal,query,expected in cases:
                    controls=torch.cat((seed.new_tensor([[tokens]]),query),-1)
                    actual=self._prediction(seed,goal,controls)
                    reference=inference.motion_rep.inverse(expected['motion'][:,:tokens*4],False,return_all=False)
                    result=inference.motion_rep.inverse(actual['motion'][:,:tokens*4],False,return_all=False)
                    error=float((result['posed_joints']-reference['posed_joints']).norm(dim=-1).max())
                    equal=torch.equal(actual['tokens'][:,:tokens],expected['tokens'][:,:tokens])
                    if not equal or error>3e-5:raise ValueError(f'Horizon{tokens} mismatch: tokens={equal}, FK={error}')
                    for key in ('points','world_query_xz','sdf_distance'):
                        try:torch.testing.assert_close(actual[key],expected[key],atol=1e-5,rtol=1e-5)
                        except AssertionError as error:
                            context=dict(tokens=tokens,goal=goal.cpu().tolist(),query=query.cpu().tolist(),key=key,
                                root_max_abs=float((actual['root']-expected['root']).abs().max()),
                                fk_max_m=float((result['posed_joints']-reference['posed_joints']).norm(dim=-1).max()))
                            raise ValueError(f'Observation parity {context}: {error}') from error
                    for key in ('valid','sdf_valid'):
                        torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
                    motion=actual['motion'][:,4:12].contiguous()
                    geometry=self._geometry(motion)
                    pose=inference.motion_rep.inverse(motion,False,return_all=False)
                    vertices=self.skinner.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
                    torch.testing.assert_close(self.skinner.lbs(geometry['rig']),vertices,atol=1e-5,rtol=1e-5)
                    self.validation.append(dict(num_tokens=tokens,condition_frames=tokens*4,usable_frames=tokens*4-8,
                        root_prefix_length=9+tokens,prefix_mask_checked_at_capture=True,
                        neural_execution=self.neural_execution,tokens_equal=equal,fk_max_error_m=error,
                        geometry_verified=True,full_skin_vertices=vertices.shape[-2],query_controls=query.cpu().tolist()))
            self.constant_storage=self.constant_storage_report()
        finally:
            inference._vqvae_pose_model.forward_decoder=original_decode
            self.observer.deferred=False
            use_dense_transformers(inference)

    def constant_storage_report(self):
        source={v.untyped_storage().data_ptr() for module in (self.inference,self.controller)
                for v in (*module.parameters(),*module.buffers()) if v.is_cuda}
        unique={};profiles=[]
        for tokens,calls in self.graphs.items():
            for stage,call in zip(('root','pose'),calls):
                current={}
                for node in call.function.graph.nodes():
                    if node.kind()=='prim::Constant' and 'value' in node.attributeNames() and node.kindOf('value')=='t':
                        value=node.t('value')
                        if value.is_cuda:
                            storage=value.untyped_storage();current[storage.data_ptr()]=storage.nbytes()
                unique.update(current)
                profiles.append(dict(num_tokens=tokens,stage=stage,constant_bytes=sum(current.values()),
                    shared_model_bytes=sum(size for ptr,size in current.items() if ptr in source)))
        return dict(profiles=profiles,unique_cuda_constant_bytes=sum(unique.values()),
            shared_model_bytes=sum(size for ptr,size in unique.items() if ptr in source),
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

    def _path_inputs(self,history,root,coarse,tokens):
        self.controller.frames=tokens*4
        return self.observer.sample(history,self.scene,root,dict(zip(VOLUME_KEYS,coarse[8:])),
            LocalTerrainObservation(*coarse[:4]).as_model_inputs(),reuse_static_regions=True)

    def _prediction(self,history,goal,controls=None,route=None,route_valid=None):
        tokens=8 if controls is None else int(controls[0,0].item())
        if tokens not in self.graphs:raise ValueError('Horizon graph was not prepared at startup')
        query=history.new_tensor([[-1.,.6,1.,.4]]) if controls is None else controls[:,1:]
        coarse=self._scene_inputs(history,goal,tokens,query)
        root_call,pose_call=self.graphs[tokens]
        root=root_call(history,goal,query,*coarse)
        volume,height,path=self._path_inputs(history,root['root'],coarse,tokens)
        result=pose_call(history,goal,query,root['count'],root['root'],root['local'],*coarse[:8],
            height.points,height.valid,height.world_query_xz,height.query_envelope_m,*(volume[k] for k in VOLUME_KEYS))
        config=terrain_observation_config(self.inference._root_model.args['observation_range_policy'])
        direction=velocity_direction_local(history[...,:5],goal,(4*tokens-4)/30)
        result.update(survey_points=coarse[4],survey_valid=coarse[5],survey_world_query_xz=coarse[6],
            actor_query_controls=trained_query_controls(query,direction,config))
        self.observer.height,self.observer.path=height,path
        return result

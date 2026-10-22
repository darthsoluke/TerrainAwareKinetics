"""Actual online v4 model output in the existing SOMA playback protocol."""

import copy
import time

import torch

from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.terrain_control import SoftCommandState,CONTROL_VERSION
from motionbricks.data.terrain_model_v3 import pack_scene_chunks
from motionbricks.data.terrain_decoder_pilot import vertical_origin


def joint_checkpoint_progress(checkpoint):
    contract=checkpoint['contract'];step=int(checkpoint['step'])
    if isinstance(contract.get('stair_training_stage'),dict):
        stage=contract['stair_training_stage']
        if (stage.get('schema')!='motionbricks.training_stage.v1' or step!=int(stage['optimizer_updates'])
                or step<1 or int(stage['actor_primitives'])<1):
            raise ValueError('Current stair training metadata is inconsistent')
        examples=int(stage['actor_primitives'])
        if checkpoint.get('exposure') and sum(int(v) for v in checkpoint['exposure'].values())!=examples:
            raise ValueError('Stair stage samples differ from retained exposure')
        return {'trained_updates':step,'optimizer_updates_in_stage':step,'primitive_examples':examples,
            'reference_batch':32,'reference_batch_equivalent_updates':examples/32,
            'source_autoencoder_examples':int(stage['source_autoencoder_examples']),
            'training_stage':stage['name'],'prior_training_reported_separately':True,
            'inherited_optimizer_updates':int(stage['inherited_optimizer_updates'])}
    if isinstance(contract.get('pure_parc_pilot'),dict):
        stage=contract['pure_parc_pilot'];exposure=checkpoint.get('exposure',{})
        window=sum(int(v) for k,v in exposure.items() if k.startswith('window:'))
        feedback=sum(int(v) for k,v in exposure.items() if k.startswith('episode:'))
        window_updates=min(step,int(stage['window_updates']))
        if (step<1 or step>int(stage['window_updates'])+int(stage['episode_updates'])
                or window!=int(stage['window_batch'])*window_updates
                or any(int(v)<0 for v in exposure.values())):
            raise ValueError('PARC fitting counters differ from retained exposure')
        examples=window+feedback
        return {'trained_updates':step,'optimizer_updates_in_stage':step,
            'primitive_examples':examples,'reference_batch':32,'reference_batch_equivalent_updates':examples/32,
            'training_stage':'pure_parc_stair_fit','prior_training_reported_separately':True,
            'source_autoencoder_examples':int(stage['window_batch'])*step,
            'source_window_examples':window,'generated_history_examples':feedback}
    if isinstance(contract.get('leg_head_fit'),dict):
        stage=contract['leg_head_fit'];examples=int(stage['actor_examples'])
        if step!=int(stage['actual_updates']):raise ValueError('Leg adaptation update count differs from its contract')
        return {'trained_updates':step,'optimizer_updates_in_stage':step,
            'primitive_examples':examples,'reference_batch':32,'reference_batch_equivalent_updates':examples/32,
            'training_stage':'leg_output_adaptation','prior_training_reported_separately':True,
            'source_autoencoder_examples':int(stage['source_autoencoder_examples']),
            'inherited_optimizer_updates':int(stage['inherited_optimizer_updates']),
            'inherited_fit_operations':int(stage['inherited_fit_operations'])}
    if isinstance(contract.get('joint_sdf'),dict) and 'global_batch' in contract['joint_sdf']:
        stage=contract['joint_sdf'];examples=step*stage['global_batch']
        return {'optimizer_updates_in_stage':step,'trained_updates':step,
            'primitive_examples':examples,'reference_batch':32,'reference_batch_equivalent_updates':examples/32,
            'training_stage':'joint_sdf_adaptation','prior_training_reported_separately':True}
    plan=contract.get('sample_budget')
    inherited=int(plan['inherited_reference_updates']) if plan else int((contract.get('initialization') or {}).get('step',0))
    result={'trained_updates':step+inherited,'optimizer_updates_in_stage':step}
    if plan:
        examples=plan['retained_examples']+(step-plan['anchor_update'])*plan['global_batch']
        result.update(primitive_examples=examples,reference_batch=plan['reference_batch'],
            reference_batch_equivalent_updates=examples/plan['reference_batch'])
    return result


class JointTerrainOnlineBackend(NativeTerrainOnlineBackend):
    def __init__(self,model,initial_history,scene):
        super().__init__(model.inference,initial_history,scene)
        self.model=model
        self.command_state=SoftCommandState()
        self.packed_scene=pack_scene_chunks(scene,self.history.device)
        self.committed_history_prefetch=False
        self.condition_contract.update(control_version=CONTROL_VERSION,refresh_frames=8)

    @torch.inference_mode()
    def predict_plan(self,velocity_xz,heading):
        velocity=torch.as_tensor(velocity_xz).reshape(-1)
        direction=torch.as_tensor(heading).reshape(-1)
        if velocity.shape!=(2,) or direction.shape!=(1,) or not (
                torch.isfinite(velocity).all() and torch.isfinite(direction).all()):
            raise ValueError('Expected one finite planar command')
        command=(tuple(velocity.tolist()),float(direction[0]))
        compatible=(self._last_published_history is not None
            and torch.allclose(self.history,self._last_published_history,atol=1e-5,rtol=0))
        if (command==self._last_command and compatible and self._future_buffer is not None
                and self._future_buffer.shape[1]>=self.PUBLISH_FRAMES):
            future=self._future_buffer[:,:self.PUBLISH_FRAMES].clone()
            self._future_buffer=self._future_buffer[:,self.PUBLISH_FRAMES:].clone()
            self._debug_future=self._advance_debug(self._debug_future,self.PUBLISH_FRAMES)
            publication=self._publication(future);self._last_published_history=publication['published_history']
            self.last_generation={**self.last_generation,'neural_generation':False,'reason':'model_buffer_sample',
                'remaining_frames':self._future_buffer.shape[1]}
            return {'planned_motion':future,'generation_seconds':0.,'debug':self._debug_future,**publication}
        settings=self.runtime_controls or {}
        task=self.command_state.intent(self.history,velocity_xz,heading,self.num_tokens)
        controls=self.history.new_tensor([[settings.get('lookahead_m',12.),settings.get('lateral_halfwidth_m',2.),
            settings.get('lookahead_gain',1.),settings.get('lookahead_base_m',.4)]]) if settings else None
        options={'num_tokens':self.num_tokens,'inference_steps':settings.get('inference_steps',4),
            'path_refinements':settings.get('path_refinements',1),'query_controls':controls}
        if self.history.is_cuda:torch.cuda.synchronize(self.history.device)
        started=time.perf_counter()
        if self.execution is not None:
            compiled=self.execution.predict(self.history,task,**options)
            result={'world':compiled['world'],'observation':{key:compiled[key] for key in ('points','valid','far_points','far_valid')}}
            execution_mode=compiled.get('execution_mode','cuda-graph')
        else:
            result=self.model(self.history,task,[self.scene],packed_scene=self.packed_scene,**options)
            execution_mode='native-eager'
        prediction=result['world']
        refresh=settings.get('refresh_frames',8)
        self._future_buffer=prediction[:,4:self.num_tokens*4-4][:,:refresh].clone()
        future=self._future_buffer[:,:8].clone();self._future_buffer=self._future_buffer[:,8:].clone()
        if not torch.isfinite(future).all():raise ValueError('Nonfinite online model output')
        observation=result['observation'];root=self.history[0,-1,:3]
        yaw=torch.atan2(self.history[0,-1,4],self.history[0,-1,3]);c,s=yaw.cos(),yaw.sin()
        def patch(points,valid,name):
            points=points[0]
            xyz=torch.stack((c*points[...,0]+s*points[...,2]+root[0],points[...,1]+root[1],
                -s*points[...,0]+c*points[...,2]+root[2]),-1)
            return {'name':name,'shape':list(points.shape[:2]),'world_xyz_m':xyz.reshape(-1,3).cpu().tolist(),
                'valid':valid[0].reshape(-1).cpu().tolist()}
        patches=[patch(observation['points'][:,i],observation['valid'][:,i],name)
                 for i,name in enumerate(('near','intent'))]
        for i,name in enumerate(('intent_far','predicted_route_far')[:observation['far_points'].shape[1]//9]):
            patches.append(patch(observation['far_points'][:,i*9:(i+1)*9],
                observation['far_valid'][:,i*9:(i+1)*9],name))
        if self.execution is not None:planned=compiled['root_world']
        else:
            planned=self.inference.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values']).clone()
            planned[...,[0,2]]+=self.history[:,:1,[0,2]]
            planned[...,1]+=vertical_origin(self.inference,self.history,observation)[:,None]
        debug={'version':'soma_motion_debug_v1','input_intent':{'start_world_xyz_m':root.cpu().tolist(),
            'endpoint_world_xz_m':task[0,:2].cpu().tolist(),'velocity_world_xz_mps':task[0,2:4].cpu().tolist(),
            'heading_world_rad':float(torch.atan2(task[0,5],task[0,4])),
            'horizon_seconds':(self.num_tokens*4-4)/30,'height_source':'not_conditioned','condition_kind':CONTROL_VERSION},
            'terrain_patches':patches,'observation_root_world_xyz_m':root.cpu().tolist(),
            'observation_source':'current_played_history',
            'root_stage_world_xyz_m':planned[0,4:self.num_tokens*4-4,:3].cpu().tolist(),
            'decoded_world_xyz_m':prediction[0,4:self.num_tokens*4-4,:3].cpu().tolist()}
        measured=(self.history[:,-1,[0,2]]-self.history[:,-2,[0,2]]).norm(dim=-1)*30
        speed=torch.maximum(task[:,2:4].norm(dim=-1),measured)
        debug.update(motion_observation_extent_m=float((speed*((self.num_tokens*4-4)/30)+.4).clamp(.8,3.4)),
            terrain_sampling='predicted_decoded_path_corridor' if options['path_refinements'] else 'input_direction',
            path_refinements=options['path_refinements'],runtime_controls=dict(settings) if settings else None)
        publication=self._publication(future)
        if self.history.is_cuda:torch.cuda.synchronize(self.history.device)
        elapsed=time.perf_counter()-started
        self._last_command=command;self._last_published_history=publication['published_history'];self._debug_future=debug
        self.last_generation={'neural_generation':True,'control_version':CONTROL_VERSION,
            'future_height_source':'model','future_pose_source':'model','predicted_frames':self.num_tokens*4,
            'remaining_frames':self._future_buffer.shape[1],'refresh_frames':refresh,'execution_mode':execution_mode,
            'inference_config':{'num_inference_step':options['inference_steps'],'pose_token_sampling_use_argmax':True}}
        if settings:self.last_generation['runtime_controls']=dict(settings)
        if self._future_buffer.shape[1]==0:self._future_buffer=None
        return {'planned_motion':future,'generation_seconds':elapsed,'debug':debug,**publication}

    def prepare_next_plan(self,*args,**kwargs):
        if self.committed_history_prefetch:
            if self.execution is not None:
                raise ValueError('Committed prefetch is verified for eager inference only')
            velocity,heading,observed,expected=args[:4]
            overlap=kwargs.get('overlap_frames')
            if (overlap not in (4,6,8) or expected.shape!=self.history.shape
                    or not torch.isfinite(expected).all()
                    or not torch.allclose(self.history,observed,atol=1e-5,rtol=0)
                    or self._last_published_history is None
                    or not torch.allclose(expected,self._last_published_history,atol=1e-5,rtol=0)):
                raise ValueError('Prefetch must follow the immutable published block')
            future_backend=copy.copy(self)
            future_backend.history=expected.detach().clone()
            future_backend.command_state=copy.deepcopy(self.command_state)
            result=future_backend.predict_plan(velocity,heading)
            remaining=future_backend._future_buffer
            if remaining is None:remaining=result['planned_motion'][:,:0].clone()
            return {**result,'remaining_motion':remaining,'command':future_backend._last_command,
                'expected_prior_history':expected.detach().clone(),
                'prepared_command_state':future_backend.command_state,
                'generation':dict(future_backend.last_generation,prefetch_policy='committed-model-history-v1'),
                'forecast_alignment_lag_frames':0}
        candidate=super().prepare_next_plan(*args,**kwargs)
        if not candidate.get('deferred') and self.runtime_controls:
            candidate['generation']['runtime_controls']=dict(self.runtime_controls)
        return candidate

    @torch.inference_mode()
    def commit_prepared_plan(self,candidate):
        generation=candidate.get('generation',{})
        if generation.get('prefetch_policy')!='committed-model-history-v1':
            return super().commit_prepared_plan(candidate)
        future=candidate.get('planned_motion');remaining=candidate.get('remaining_motion')
        expected=candidate.get('expected_prior_history');published=candidate.get('published_history')
        if (not self.committed_history_prefetch or not isinstance(future,torch.Tensor)
                or future.shape!=(1,8,self.history.shape[2]) or not torch.isfinite(future).all()
                or not isinstance(remaining,torch.Tensor) or remaining.ndim!=3
                or remaining.shape[0]!=1 or remaining.shape[2]!=self.history.shape[2]
                or not torch.isfinite(remaining).all() or not isinstance(expected,torch.Tensor)
                or expected.shape!=self.history.shape or not torch.allclose(self.history,expected,atol=1e-5,rtol=0)
                or not isinstance(published,torch.Tensor) or published.shape!=self.history.shape
                or not torch.isfinite(published).all()
                or not isinstance(candidate.get('prepared_command_state'),SoftCommandState)):
            raise ValueError('Committed prediction no longer follows actual playback')
        self._future_buffer=remaining.clone() if remaining.shape[1] else None
        self._last_published_history=published.clone()
        self._last_command=candidate['command'];self._debug_future=candidate.get('debug')
        self.last_generation=dict(generation)
        self.command_state=candidate['prepared_command_state']

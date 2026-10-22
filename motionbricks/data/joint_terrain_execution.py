"""Checked static execution of the same joint controller and official SOMA rig."""

import torch

from motionbricks.data.soma_cuda_graph import StaticCudaCall
from motionbricks.data.soma_static_execution import trace_fixed_call,use_dense_transformers,validate_observed_fk
from motionbricks.data.terrain_model_v3 import pack_scene_chunks
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.native_mb_observed_history import reencode_played_history


class JointTerrainExecution:
    @torch.inference_mode()
    def __init__(self,model,scene,skin,seed,*,flexible=False):
        if not seed.is_cuda or seed.shape[:2]!=(1,4):raise ValueError('Joint CUDA execution needs one four-frame CUDA seed')
        self.history_shape=seed.shape;self.model=model;self.scene=scene;self.flexible=flexible
        model.eval().requires_grad_(False)
        packed=pack_scene_chunks(scene,seed.device);self.packed_scene=packed
        captured={};native_decode=model.codec.forward_decoder;native_lbs=skin.lbs
        def decode(*args,**kwargs):
            captured['tokens']=args[0].clone();return native_decode(*args,**kwargs)
        model.codec.forward_decoder=decode
        def prediction(history,task,query_controls=None):
            result=model(history,task,[scene],num_tokens=8,packed_scene=packed,query_controls=query_controls)
            obs=result['observation']
            root=model.inference.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values']).clone()
            root[...,[0,2]]+=history[:,:1,[0,2]]
            root[...,1]+=vertical_origin(model.inference,history,obs)[:,None]
            return {'world':result['world'],'root_world':root,'tokens':captured['tokens'],
                **{key:obs[key] for key in ('points','valid','far_points','far_valid')}}
        def capture_rig(rig):
            captured['rig']=rig
            return rig.new_empty(0)
        def geometry(frames):
            pose=model.inference.global_motion_rep.inverse(frames,False,return_all=False,joint_positions_from='global_rot_data')
            skin.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
            checks={}
            history=reencode_played_history(frames[:,-4:],model.inference.global_motion_rep,tensor_checks=checks)
            return {'rig':captured['rig'],'joints':pose['posed_joints'][0],'history':history,**checks}
        def observe(frames):
            checks={}
            history=reencode_played_history(frames,model.inference.global_motion_rep,tensor_checks=checks)
            return {'history':history,**checks}
        try:
            state=SoftCommandState();history=seed.clone();cases=[]
            probes=[([0.,0.],0.,[12.,2.,1.,.4]),([0.,1.6],0.,[12.,2.,1.,.4]),
                ([1.6,0.],1.57,[12.,2.,1.,.4]),([0.,-1.6],0.,[12.,2.,1.,.4]),([0.,0.],0.,[12.,2.,1.,.4])]
            if flexible:probes+=[([0.,1.6],0.,[-1.,.3,2.,1.]),([0.,3.],0.,[20.,6.,4.,10.])]
            for velocity,yaw,query in probes:
                task=state.intent(history,velocity,yaw,8)
                inputs=(history.clone(),task.clone())+((history.new_tensor([query]),) if flexible else ())
                expected=prediction(*inputs)
                cases.append((inputs,{k:v.clone() for k,v in expected.items()}))
                history=reencode_played_history(expected['world'][:,8:12],model.inference.global_motion_rep)
            use_dense_transformers(model.inference)
            self._prediction=StaticCudaCall(trace_fixed_call(prediction,cases[0][0],cuda_constants=True))
            skin.lbs=capture_rig
            sample=cases[0][1]['world'][:,4:12].contiguous()
            self._geometry=StaticCudaCall(trace_fixed_call(geometry,(sample,),cuda_constants=True))
            self._observe=StaticCudaCall(trace_fixed_call(observe,(seed,),cuda_constants=True))
            self.validation=[]
            for inputs,expected in cases:
                history,task=inputs[:2]
                actual=self._prediction(*(value.contiguous() for value in inputs))
                if not torch.equal(actual['tokens'][:,:8],expected['tokens'][:,:8]):
                    raise ValueError('Static execution changed active native motion tokens')
                for key in ('world','root_world'):
                    torch.testing.assert_close(actual[key],expected[key],atol=1e-5,rtol=1e-5)
                # Tiny differences in a nearly stationary proposed route can
                # amplify under long-range extrapolation. Bound query error to
                # 0.1 mm (1000x below the 10 cm scene grid), while motion/token
                # checks stay unchanged and sampled-height changes still fail.
                for key in ('points','far_points'):
                    torch.testing.assert_close(actual[key],expected[key],atol=1e-4,rtol=0)
                for key in ('valid','far_valid'):
                    if not torch.equal(actual[key],expected[key]):raise ValueError('Static execution changed scene visibility')
                frames=actual['world'][:,4:12].contiguous()
                rigged=self._geometry(frames);validate_observed_fk(rigged)
                pose=model.inference.global_motion_rep.inverse(frames,False,return_all=False,joint_positions_from='global_rot_data')
                skin.lbs=native_lbs
                reference=skin.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
                torch.testing.assert_close(native_lbs(rigged['rig']),reference,atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(self._observe(history)['history'],observe(history)['history'],atol=1e-5,rtol=1e-5)
                skin.lbs=capture_rig
                self.validation.append({'active_tokens_equal':True,'world_max_feature_error':float((actual['world']-expected['world']).abs().max()),
                    'query_max_error_m':max(float((actual[key]-expected[key]).abs().max()) for key in ('points','far_points'))})
        finally:
            model.codec.forward_decoder=native_decode;skin.lbs=native_lbs

    @torch.inference_mode()
    def predict(self,history,task,*,num_tokens=8,inference_steps=4,path_refinements=1,query_controls=None):
        if history.shape!=self.history_shape or task.shape!=(1,6) or not torch.isfinite(task).all():
            raise ValueError('Static joint input does not match its checked control signature')
        if num_tokens==8 and inference_steps==4 and path_refinements==1 and (self.flexible or query_controls is None):
            inputs=(history.contiguous(),task.contiguous())
            if self.flexible:
                query=history.new_tensor([[12.,2.,1.,.4]]) if query_controls is None else query_controls
                inputs+=(query.contiguous(),)
            result=self._prediction(*inputs)
            result['execution_mode']='cuda-graph'
        else:
            result=self.model(history,task,[self.scene],num_tokens=num_tokens,packed_scene=self.packed_scene,
                query_controls=query_controls,inference_steps=inference_steps,path_refinements=path_refinements)
            root=self.model.inference.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values']).clone()
            root[...,[0,2]]+=history[:,:1,[0,2]]
            root[...,1]+=vertical_origin(self.model.inference,history,result['observation'])[:,None]
            result={'world':result['world'],'root_world':root,'execution_mode':'native-eager',
                **{key:result['observation'][key] for key in ('points','valid','far_points','far_valid')}}
        if not torch.isfinite(result['world']).all():raise ValueError('Nonfinite joint prediction')
        return result

    @torch.inference_mode()
    def geometry(self,frames):
        if frames.shape!=(1,8,self.history_shape[2]):raise ValueError('Only eight committed model frames are supported')
        result=self._geometry(frames.contiguous());validate_observed_fk(result);return result

    @torch.inference_mode()
    def observe(self,frames):
        if frames.shape!=self.history_shape:raise ValueError('Only four observed history frames are supported')
        result=self._observe(frames.contiguous());validate_observed_fk(result);return result['history']

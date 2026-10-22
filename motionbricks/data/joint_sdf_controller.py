"""Experimental 15-landmark SDF conditioning before native MB Transformers."""

from collections import OrderedDict

import torch
from torch import nn

from .terrain_model_v3 import JointTerrainController,multiscale_observation,retain_intent_observation,pack_scene_chunks
from .terrain_decoder_pilot import training_targets
from .joint_sdf_observation import HeightfieldPrismSDF,PackedPrismSDF,joint_sdf_features


SDF_VERSION='soma15-prism-sdf-pretransformer-v1'


class SDFCondition(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query=nn.Linear(width,128)
        self.attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.output=nn.Linear(128,width,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,query,memory,valid):
        available=valid.any(-1);mask=valid.clone();mask[:,0]|=~available
        update,_=self.attention(self.query(query),memory,memory,key_padding_mask=~mask,need_weights=False)
        return query+self.output(update)*available[:,None,None]


class JointSDFController(JointTerrainController):
    def __init__(self,inference,bundle):
        super().__init__(inference,bundle)
        self.sdf_encoder=nn.Sequential(nn.Linear(8,128),nn.GELU(),nn.Linear(128,128))
        self.sdf_joint_embedding=nn.Embedding(15,128)
        self.sdf_root=SDFCondition(self.root.args['n_embd'])
        self.sdf_pose=SDFCondition(self.pose._args['n_embd'])
        self.sdf_decoder=SDFCondition(self.codec.code_dim)
        self.sdf_enabled=True;self._sdf_memory=None;self._sdf_valid=None;self._sdf_fields=OrderedDict()
        self._sdf_packed_key=None
        self._sdf_handles=[]
        for module in (self.root._shared_transformer_model,self.root._root_token_transformer_model):
            if module is not None:self._sdf_handles.append(module.register_forward_pre_hook(self._root_sdf_hook))
        self._sdf_handles.append(self.pose._transformer_model.register_forward_pre_hook(self._pose_sdf_hook))
        self._sdf_handles.append(self.codec.decoder.register_forward_pre_hook(self._decoder_sdf_hook))
        previous=self.pose.forward
        def remember(*args,**kwargs):
            self._sdf_pose_request=(args,kwargs)
            return previous(*args,**kwargs)
        self.pose.forward=remember
        self.to(inference.device);self.eval()

    def freeze_base(self):
        self.requires_grad_(False)
        for name in ('sdf_encoder','sdf_joint_embedding','sdf_root','sdf_pose','sdf_decoder'):
            getattr(self,name).requires_grad_(True)

    def _condition(self,module,args,transpose=False):
        if not self.sdf_enabled or self._sdf_memory is None:return args
        value=args[0].transpose(1,2) if transpose else args[0]
        corrected=module(value,self._sdf_memory,self._sdf_valid)
        flat=module(value,self._sdf_plane,self._sdf_valid)
        value=value+(corrected-flat)
        return (value.transpose(1,2) if transpose else value,*args[1:])

    def _root_sdf_hook(self,_module,args):return self._condition(self.sdf_root,args)
    def _pose_sdf_hook(self,_module,args):return self._condition(self.sdf_pose,args)
    def _decoder_sdf_hook(self,_module,args):return self._condition(self.sdf_decoder,args,True)

    def _set_sdf(self,history,proposal,scenes,observation):
        fields=[]
        for scene in scenes:
            key=id(scene['heightfield_y_m'])
            if key not in self._sdf_fields:self._sdf_fields[key]=(scene,HeightfieldPrismSDF(scene))
            self._sdf_fields.move_to_end(key)
            fields.append(self._sdf_fields[key][1])
            if len(self._sdf_fields)>max(96,2*len(scenes)):self._sdf_fields.popitem(last=False)
        ground=history[:,-1,1]+observation['points'][:,0,4,4,1]
        key=tuple(id(scene['heightfield_y_m']) for scene in scenes)
        if self._sdf_packed_key!=key:
            self._sdf_packed=PackedPrismSDF(fields,scenes,history.device)
            self._sdf_packed_key=key
            self._sdf_packed_scenes=list(scenes)
        values,valid,plane=joint_sdf_features(self.inference.global_motion_rep,history,proposal,scenes,
            fields=fields,packed_fields=self._sdf_packed,ground_y=ground)
        identity=self.sdf_joint_embedding(torch.arange(15,device=history.device))[None,None]
        memory=self.sdf_encoder(values)+identity
        self._sdf_memory=torch.where(valid[...,None],memory,torch.zeros_like(memory)).flatten(1,2)
        plane_memory=self.sdf_encoder(plane)+identity
        self._sdf_plane=torch.where(valid[...,None],plane_memory,torch.zeros_like(plane_memory)).flatten(1,2)
        self._sdf_valid=valid.flatten(1,2)
        self.sdf_diagnostics={'version':SDF_VERSION,'points_per_frame':15,'frames':values.shape[1],
            'valid_fraction':float(valid.float().mean()),'future_source':'model_proposal' if proposal is not None else 'none'}

    def forward(self,history,intent,scenes,*,num_tokens=8,source=None,packed_scene=None,straight_through=False,scene_frame=None,
                inference_steps=4,path_refinements=1,query_controls=None,far_distance=12.,far_halfwidth=2.,
                history_gradients=False):
        self._history_gradients=bool(history_gradients)
        if not isinstance(inference_steps,int) or not 1<=inference_steps<=8:raise ValueError('Use 1..8 denoising passes')
        if not isinstance(path_refinements,int) or not 0<=path_refinements<=2:raise ValueError('Use 0..2 route refinements')
        if isinstance(scenes,dict):scenes=[scenes]
        packed=pack_scene_chunks(scenes,history.device) if packed_scene is None else packed_scene
        rep=self.inference.global_motion_rep
        physical_history=history if scene_frame is None else scene_frame.motion(rep,history,inverse=True)
        physical_intent=intent if scene_frame is None else scene_frame.task(intent,inverse=True)
        options=dict(packed=packed,query_controls=query_controls,far_distance=far_distance,far_halfwidth=far_halfwidth)
        initial=multiscale_observation(physical_history,physical_intent,scenes,num_tokens,**options)
        observation=initial;physical_proposal=None
        for _ in range(path_refinements):
            with torch.no_grad():
                self._set_sdf(physical_history,physical_proposal,scenes,observation)
                proposal=self._predict(history,intent,observation,num_tokens,False,inference_steps)
                physical_proposal=(proposal if scene_frame is None else scene_frame.motion(rep,proposal,inverse=True))[:,:num_tokens*4]
            route=physical_proposal[:,4:num_tokens*4-4,[0,2]].detach()
            refined=multiscale_observation(physical_history,physical_intent,scenes,num_tokens,route=route,**options)
            observation=retain_intent_observation(initial,refined)
        self._set_sdf(physical_history,physical_proposal,scenes,observation)
        world=self._predict(history,intent,observation,num_tokens,straight_through,inference_steps)
        result={'world':world[:,:num_tokens*4],'root_output':self.last_root,'pose_logits':self.last_logits,'observation':observation}
        if source is not None:
            result.update(target=training_targets(self.inference,source,history,observation,frames=num_tokens*4),
                local_prediction=self.last_local[:,:num_tokens*4])
        return result

    def native_losses(self,result,source):
        base=super().native_losses(result,source)
        tokens=self.codec.encode_into_idx(result['target']['local_motions'],fetch_overall_indices=False).detach()
        teacher=self.inference._pose_model
        batch={'groundtruth_pose_tokens':tokens,'local_motions':result['target']['local_motions']}
        keys=('focused_token_mask','masked_token_mask','incorrect_token_mask','correct_token_mask','mask_percentage','incorrect_percentage')
        batch.update(zip(keys,teacher._get_token_masks(batch)))
        corrupted=teacher._generate_tokens_for_training(batch)
        args,kwargs=self._sdf_pose_request
        padded=torch.full_like(args[0],self.pose.POSE_MASK_ID);padded[:,:tokens.shape[1]]=corrupted
        output=type(self.pose).forward(self.pose,padded,*args[1:],**kwargs)
        logits=output['pose_logits'][:,:tokens.shape[1]]
        auxiliary=teacher.loss(batch,{'pose_logits':logits})['loss'] if batch['focused_token_mask'].any() else logits.sum()*0
        base['native_total']=base['native_total']-base['token_ce']+auxiliary
        base['rollout_last_ce']=base['token_ce'];base['token_ce']=auxiliary
        base['sampled_mask_fraction']=float(batch['mask_percentage'].mean())
        return base

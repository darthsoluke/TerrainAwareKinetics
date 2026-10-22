"""Joint native MB learning with soft intent and separate near/far scene residuals.

Discrete token forward values remain hard codewords. The optional training
estimator passes gradients through their soft expectation; inference never uses it.
"""

from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config,INTENT_RANGE_POLICY
from motionbricks.data.native_mb_online_conditions import SOFT_INTENT_PROTOCOL,sparse_soft_intent_conditions
from motionbricks.data.native_mb_codec_height import build_codec_height_tokenizer,CODEC_HEIGHT_CONDITIONING
from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.neural_modules.local_terrain_encoder import LocalTerrainEncoder,TerrainCrossAttention
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel


SCHEMA='motionbricks.joint_terrain_controller.v4'


class FarSceneResidual(nn.Module):
    """Learn relief relative to the near surface, with a flat-reference residual.

    Spatial coordinates are normalized; physical range/reach time travel as
    explicit bounded features. A flat scene yields zero residual at any range,
    including after training, without constraining the generated motion.
    """
    def __init__(self,width,heads=4):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(7,32,3,padding=1),nn.GELU(),
                                   nn.Conv2d(32,width,3,padding=1),nn.GELU())
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True,bias=False)
        self.gate=nn.Parameter(torch.zeros(()))

    def tokens(self,observation):
        cache=observation.setdefault('_far_cache',{})
        key=(id(self),torch.is_grad_enabled())
        if key in cache:return cache[key]
        points=observation['far_points'].detach()
        valid=observation['far_valid'].detach() & torch.isfinite(points).all(-1)
        points=torch.nan_to_num(points)
        distance=points[...,[0,2]].norm(dim=-1)
        radius=distance.flatten(1).amax(1).clamp_min(.4)
        near_y=observation['points'][:,0,4,4,1].detach()
        delta=points[...,1]-near_y[:,None,None]
        low=observation.get('far_min_y',points[...,1]).detach()-near_y[:,None,None]
        travel=(observation['speed_mps']*observation['horizon_seconds']+.4).detach()
        shape=points.shape[:-1]
        channels=torch.stack((points[...,0]/radius[:,None,None],points[...,2]/radius[:,None,None],
            torch.asinh(delta),torch.asinh(low),valid.float(),torch.asinh(distance/travel[:,None,None]),
            torch.log1p(radius)[:,None,None].expand(shape)),-1)
        channels=torch.where(valid[...,None],channels,torch.zeros_like(channels)).permute(0,3,1,2)
        flat=channels.clone();flat[:,2:4]=0
        views=observation.get('far_views',1)
        b,c,h,w=channels.shape
        if h%views:raise ValueError('Far view grids must retain complete independent patches')
        channels=channels.reshape(b,c,views,h//views,w).permute(0,2,1,3,4).reshape(b*views,c,h//views,w)
        flat=flat.reshape(b,c,views,h//views,w).permute(0,2,1,3,4).reshape(b*views,c,h//views,w)
        encoded=self.encoder(channels)-self.encoder(flat)
        tokens=encoded.flatten(2).transpose(1,2).reshape(b,-1,encoded.shape[1])
        known=valid.flatten(1)
        tokens=torch.where(known[...,None],tokens,torch.zeros_like(tokens))
        cache[key]=(tokens,known)
        return tokens,known

    def forward(self,queries,observation):
        tokens,known=self.tokens(observation)
        any_known=known.any(-1)
        safe=known.clone();safe[:,0]|=~any_known
        update,_=self.attention(queries,tokens.to(queries),tokens.to(queries),
                               key_padding_mask=~safe,need_weights=False)
        # Bias-free value/output projections map zero relief to an exact zero
        # update, without a second attention pass or a motion-space clamp.
        return queries+self.gate*update*any_known[:,None,None]


def straight_through_codes(logits,hard,codebook):
    if logits.shape[-2:]!=codebook.shape[:2] or hard.shape[-1]!=codebook.shape[0]*codebook.shape[2]:
        raise ValueError('Token logits/codewords do not share the frozen vocabulary')
    soft=torch.einsum('bthk,hkd->bthd',logits.float().softmax(-1),codebook.detach().float()).flatten(-2)
    return hard+(soft-soft.detach()).to(hard)


def pack_scene_chunks(scenes,device,cell_budget=16_000_000):
    if isinstance(scenes,dict):scenes=[scenes]
    chunks=[];start=0
    while start<len(scenes):
        height=width=0;stop=start
        while stop<len(scenes):
            h,w=scenes[stop]['heightfield_y_m'].shape
            nh,nw=max(height,h),max(width,w)
            if stop>start and (stop-start+1)*nh*nw>cell_budget:break
            height,width=nh,nw;stop+=1
        selected=scenes[start:stop]
        packed=torch.full((len(selected),height,width),torch.nan,dtype=torch.float32)
        origins=[];spacing=[]
        for row,scene in enumerate(selected):
            hf=torch.as_tensor(scene['heightfield_y_m']).detach().cpu().float()
            if 'heightfield_valid' in scene:
                hf=torch.where(torch.as_tensor(scene['heightfield_valid']).cpu().bool(),hf,torch.nan)
            packed[row,:hf.shape[0],:hf.shape[1]]=hf
            origins.append(torch.as_tensor(scene['min_point_world_xyz_m']).cpu().float())
            spacing.append(torch.as_tensor(scene['dx_m']).cpu().float().reshape(()))
        chunks.append((start,start+len(selected),packed.to(device),
                       torch.stack(origins).to(device),torch.stack(spacing).to(device)))
        start=stop
    return chunks


def pool_far_observation(observation):
    """Keep thin walls and depressions instead of point-decimating a wide map."""
    points=observation.points[:,1];valid=observation.valid[:,1]
    known=F.adaptive_max_pool2d(valid[:,None].float(),(9,9))[:,0]>0
    heights=points[...,1][:,None]
    high=F.adaptive_max_pool2d(heights.masked_fill(~valid[:,None],-torch.inf),(9,9))[:,0]
    low=-F.adaptive_max_pool2d((-heights).masked_fill(~valid[:,None],-torch.inf),(9,9))[:,0]
    center=F.adaptive_avg_pool2d(points.permute(0,3,1,2),(9,9)).permute(0,2,3,1)
    center[...,1]=torch.where(known,high,0.)
    return center,known,torch.where(known,low,0.)


def retain_intent_observation(initial,refined):
    """Route refinement adds observations; it cannot erase an already seen hazard."""
    result=dict(refined)
    for key in ('far_points','far_valid','far_min_y'):
        result[key]=torch.cat((initial[key],refined[key]),dim=1)
    result['far_views']=2
    return result


def multiscale_observation(history,intent,scenes,tokens,*,far_distance=12.,far_halfwidth=2.,route=None,packed=None,query_controls=None):
    if isinstance(scenes,dict):scenes=[scenes]
    if len(scenes)!=len(history) or intent.shape!=(len(history),6):
        raise ValueError('Each causal history needs one scene and six soft-intent fields')
    config=terrain_observation_config(INTENT_RANGE_POLICY)
    values=[];seconds=(tokens*4-4)/30
    if packed is None:packed=pack_scene_chunks(scenes,history.device)
    for start,stop,heightfields,origins,spacing in packed:
        observed=history[start:stop].detach();task=intent[start:stop].detach()
        root=observed[:,-1,:5]
        velocity=task[:,2:4]
        measured=(observed[:,-1,[0,2]]-observed[:,-2,[0,2]])*30
        speed=torch.maximum(velocity.norm(dim=-1),measured.norm(dim=-1))
        direction=torch.where((velocity.norm(dim=-1)>1e-5)[:,None],velocity,measured)
        yaw=torch.atan2(root[:,4],root[:,3]);c,s=yaw.cos(),yaw.sin()
        local=torch.stack((c*direction[:,0]-s*direction[:,1],s*direction[:,0]+c*direction[:,1]),-1)*seconds
        controls=history.new_tensor([[-1.,.6,1.,.4]]).expand(stop-start,-1).clone()
        kwargs=dict(heightfield_y_m=heightfields,
            min_point_world_xyz_m=origins,dx_m=spacing,
            root_position_world_xyz_m=root[:,:3],root_heading_yaw_rad=yaw,
            control_direction_local_xz=local,config=config)
        proposal=route[start:stop].detach() if route is not None else None
        proposal_valid=torch.ones(proposal.shape[:-1],device=history.device,dtype=torch.bool) if proposal is not None else None
        # The near network keeps its trained metric geometry. Far inputs are
        # learned through their separate normalized scene encoder.
        controls[:,0]=(speed*seconds+.4).clamp(.8,3.4)
        near=extract_local_terrain_observation(**kwargs,query_controls=controls,
                query_path_world_xz=proposal,query_path_valid=proposal_valid)
        wide=history.new_tensor([[far_distance,far_halfwidth,1.,.4]]).expand(stop-start,-1).clone()
        if query_controls is not None:
            requested=query_controls[start:stop].to(history)
            automatic=(speed*seconds*requested[:,2]+requested[:,3]).clamp(.8,20.)
            wide=torch.stack((torch.where(requested[:,0]<0,automatic,requested[:,0]),requested[:,1],
                torch.ones_like(speed),torch.full_like(speed,.4)),1)
        dense_kwargs=dict(kwargs,config=replace(config,patch_height=201,patch_width=121))
        far=extract_local_terrain_observation(**dense_kwargs,query_controls=wide,
                query_path_world_xz=proposal,query_path_valid=proposal_valid)
        far_points,far_valid,far_min=pool_far_observation(far)
        values.append({'points':near.points,'valid':near.valid,'query_envelope_m':near.query_envelope_m,
            'far_points':far_points,'far_valid':far_valid,'far_min_y':far_min,
            'speed_mps':speed,'horizon_seconds':history.new_full((stop-start,),seconds)})
    return {key:torch.cat([v[key] for v in values],0).detach() for key in values[0]}


class JointTerrainController(nn.Module):
    def __init__(self,inference,bundle):
        super().__init__()
        object.__setattr__(self,'inference',inference)
        self.root=inference._root_model.backbone_net
        self.pose=inference._pose_model.backbone_net
        self._new_local_mask=not self.root._args.get('local_component_conditioning',False)
        if not self.root._args.get('local_component_conditioning',False):
            self.root._args['local_component_conditioning']=True
            self.root._proj_local_root_component_mask=nn.Linear(self.root._args['local_root_dim'],
                self.root._args['local_root_feat_dim'],bias=False).to(inference.device)
            nn.init.zeros_(self.root._proj_local_root_component_mask.weight)
        height=build_codec_height_tokenizer(inference.motion_rep,inference._vqvae_pose_model,
                                            OmegaConf.create(bundle['tokenizer_config'])).to(inference.device)
        self.codec=height.pose_net
        inference._vqvae_pose_model=self.codec
        inference._pose_model.supporting_nets['pose_net']=self.codec
        for model in (inference._root_model,inference._pose_model):
            model.args['online_conditioning']=SOFT_INTENT_PROTOCOL['version']
        inference._pose_model.args['codec_height_conditioning']=CODEC_HEIGHT_CONDITIONING
        inference._pose_model.args['codec_context_policy']='played_context4'
        self.control=nn.Sequential(nn.Linear(7,self.root.args['n_embd']),nn.GELU(),
                                   nn.Linear(self.root.args['n_embd'],self.root.args['n_embd']))
        nn.init.zeros_(self.control[-1].weight);nn.init.zeros_(self.control[-1].bias)
        self.root_far=FarSceneResidual(self.root.args['n_embd'])
        self.pose_far=FarSceneResidual(self.pose._args['n_embd'])
        self.decoder_near=LocalTerrainEncoder(self.codec.code_dim)
        self.decoder_attention=TerrainCrossAttention(self.codec.code_dim,4)
        self.decoder_far=FarSceneResidual(self.codec.code_dim)
        args=deepcopy(bundle['tokenizer_config']['model']['args'])
        object.__setattr__(self,'loss_model',MotionVQVAEModel(self.codec,None,inference.motion_rep,
                                                            optimizer=None,scheduler=None,args=args))
        self.codec.encoder.requires_grad_(False);self.codec.quantizer.requires_grad_(False)
        self.register_buffer('frozen_codebook',self.codec.get_codebook().detach().clone(),persistent=False)
        self._observation=None;self._control_embedding=None;self._st=False;self._pose_step=0;self._inference_steps=4
        self.last_root=None;self.last_logits=None
        self._handles=[]
        self._height_weights=[];self._warmup=False
        height_features=extract_feature_from_motion_rep(None,self.codec.motion_rep,
            self.codec.decoder_external_cond_feature_mode,fetch_feat_idx=True).tolist()
        height_position=height_features.index(int(self.codec.motion_rep.indices['global_root_y'][0]))
        for layer in range(self.codec.decoder._down_t+int(self.codec.decoder._COND_FUSION_LAST_LAYER)):
            linear=self.codec.decoder.external_cond_blocks[layer*2]
            mask=torch.zeros_like(linear.weight)
            frames=2**(self.codec.decoder._down_t-layer)
            columns=linear.out_features+torch.arange(frames,device=mask.device)*len(height_features)+height_position
            mask[:,columns]=1
            self._height_weights.append(linear.weight)
            self._handles.append(linear.weight.register_hook(
                lambda grad,mask=mask:grad*mask if self._warmup else grad))
        for transformer in (self.root._shared_transformer_model,self.root._root_token_transformer_model):
            if transformer is not None:self._handles.append(transformer.register_forward_pre_hook(self._control_hook))
        self._handles.append(self.root._terrain_cross_attention.register_forward_hook(self._root_scene_hook))
        self._handles.append(self.pose._terrain_cross_attention.register_forward_hook(self._pose_scene_hook))
        self._handles.append(self.root.register_forward_hook(self._root_output_hook))
        native_pose=self.pose.forward
        def pose_forward(*args,**kwargs):
            enabled=torch.is_grad_enabled() and self._pose_step==self._inference_steps-1
            self._pose_step+=1
            with torch.set_grad_enabled(enabled):result=native_pose(*args,**kwargs)
            self.last_logits=result['pose_logits']
            return result
        self.pose.forward=pose_forward
        native_dequantize=self.codec.quantizer.dequantize
        def dequantize(*args,**kwargs):
            hard=native_dequantize(*args,**kwargs)
            if self._st and torch.is_grad_enabled():
                return straight_through_codes(self.last_logits,hard,self.frozen_codebook)
            return hard
        self.codec.quantizer.dequantize=dequantize
        native_decode=self.codec.forward_decoder
        def decode(*args,**kwargs):
            result=native_decode(*args,**kwargs)
            self.last_local=result['recon_state']
            return result
        self.codec.forward_decoder=decode
        self._handles.append(self.codec.decoder.register_forward_pre_hook(self._decoder_hook))
        self.to(inference.device)
        self.eval()

    def _control_hook(self,_module,args):
        value=args[0]+self._control_embedding[:,None].to(args[0])
        return (value,*args[1:])

    def _root_scene_hook(self,_module,_args,output):return self.root_far(output,self._observation)
    def _pose_scene_hook(self,_module,_args,output):return self.pose_far(output,self._observation)
    def _root_output_hook(self,_module,_args,output):self.last_root=output

    def _decoder_hook(self,_module,args):
        latent=args[0].transpose(1,2)
        tokens,valid=self.decoder_near(self._observation,len(latent),latent.device,latent.dtype)
        latent=self.decoder_attention(latent,tokens,valid)
        latent=self.decoder_far(latent,self._observation)
        return (latent.transpose(1,2),*args[1:])

    def set_learning_phase(self,warmup):
        self._warmup=warmup
        self.root.requires_grad_(not warmup);self.pose.requires_grad_(not warmup)
        self.codec.decoder.requires_grad_(not warmup)
        if warmup:
            # Dense Root-Y columns are newly initialized; freezing them with the
            # pretrained decoder would prevent the new interface from learning.
            for weight in self._height_weights:weight.requires_grad_(True)
            if self._new_local_mask:self.root._proj_local_root_component_mask.requires_grad_(True)
        for module in (self.control,self.root_far,self.pose_far,self.decoder_near,self.decoder_attention,self.decoder_far):
            module.requires_grad_(True)
        self.codec.encoder.requires_grad_(False);self.codec.quantizer.requires_grad_(False)
        self.eval()

    def _predict(self,history,intent,observation,tokens,straight_through,inference_steps=4):
        inference=self.inference;batch=len(history)
        if history.shape[1]!=4 or intent.shape!=(batch,6) or not torch.isfinite(intent).all():
            raise ValueError('Only four factual history frames and finite planar intent are allowed')
        history_gradients=getattr(self,'_history_gradients',False)
        raw=history if history_gradients else history.detach()
        local=inference.motion_rep.dual_rep.global_to_local(raw,False,False,
            lengths=torch.full((batch,),4,device=raw.device,dtype=torch.long))
        root=extract_feature_from_motion_rep(raw,inference.global_motion_rep,'root')
        velocity=extract_feature_from_motion_rep(local,inference.local_motion_rep,'root')
        poses=extract_feature_from_motion_rep(local,inference.local_motion_rep,inference.EXTERNAL_POSE_FEATURE_MODE)
        conditions=sparse_soft_intent_conditions(root,velocity,poses)
        if history_gradients:
            # Same observed inputs and masks; opt-in training keeps their graph.
            conditions['global_root_values'][:,:4]=root
            conditions['local_root_values'][:,:3]=velocity[:,:3]
            conditions['local_poses'][:,:4]=poses
        command=intent if history_gradients else intent.detach()
        delta=command[:,:2]-root[:,-1,[0,2]]
        controls=torch.cat((torch.asinh(delta),torch.asinh(command[:,2:4]),
                            command[:,4:6],raw.new_full((batch,1),(tokens*4-4)/30)),1)
        self._control_embedding=self.control(controls)
        self._observation=observation;self._st=straight_through;self._pose_step=0;self._inference_steps=inference_steps
        inference.BATCH_SIZE=batch
        output,count=inference.predict(conditions['global_root_values'],conditions['has_global_root_values'],
            conditions['local_root_values'],conditions['has_local_root_values'],
            conditions['local_poses'],conditions['has_local_poses'],tokens,
            global_root_component_mask=conditions['global_root_component_mask'],
            local_root_component_mask=conditions['local_root_component_mask'],terrain_observation=observation,
            config={'num_inference_step':inference_steps,'pose_token_sampling_use_argmax':True,'cache_terrain_encoder':False})
        if not torch.equal(count.reshape(-1),count.new_full((batch,),tokens)):
            raise ValueError('Model changed the explicit prediction horizon')
        return output

    def forward(self,history,intent,scenes,*,num_tokens=8,far_distance=12.,far_halfwidth=2.,
                route_refinement=True,straight_through=False,source=None,packed_scene=None,scene_frame=None,
                query_controls=None,inference_steps=4,path_refinements=None):
        refinements=int(route_refinement) if path_refinements is None else path_refinements
        if not isinstance(inference_steps,int) or not 1<=inference_steps<=8:raise ValueError('Use 1..8 native denoising passes')
        if not isinstance(refinements,int) or not 0<=refinements<=2:raise ValueError('Use 0..2 predicted-route refinements')
        packed=pack_scene_chunks(scenes,history.device) if packed_scene is None else packed_scene
        sample_history=history if scene_frame is None else scene_frame.motion(self.inference.global_motion_rep,history,inverse=True)
        sample_intent=intent if scene_frame is None else scene_frame.task(intent,inverse=True)
        observation=multiscale_observation(sample_history,sample_intent,scenes,num_tokens,
                                           far_distance=far_distance,far_halfwidth=far_halfwidth,packed=packed,query_controls=query_controls)
        initial_observation=observation
        for _ in range(refinements):
            with torch.no_grad():proposal=self._predict(history,intent,observation,num_tokens,False,inference_steps)
            route=proposal[:,4:num_tokens*4-4,[0,2]].detach()
            if scene_frame is not None:route=scene_frame.points(route,inverse=True)
            refined=multiscale_observation(sample_history,sample_intent,scenes,num_tokens,far_distance=far_distance,
                                          far_halfwidth=far_halfwidth,route=route,packed=packed,query_controls=query_controls)
            observation=retain_intent_observation(initial_observation,refined)
        world=self._predict(history,intent,observation,num_tokens,straight_through,inference_steps)
        result={'world':world[:,:num_tokens*4],'root_output':self.last_root,
                'pose_logits':self.last_logits,'observation':observation}
        if source is not None:
            target=training_targets(self.inference,source,history,observation,frames=num_tokens*4)
            result.update(target=target,local_prediction=self.last_local[:,:num_tokens*4])
        return result

    def native_losses(self,result,source):
        frames=source.shape[1]-1
        target=result['target'];local=result['local_prediction']
        motion=self.loss_model.loss(target,{'recon_state':local,'l_commit':local.sum()*0.,'perplexity':0.},None)
        tokens=self.codec.encode_into_idx(target['local_motions'],fetch_overall_indices=False).detach()
        logits=result['pose_logits'][:,:tokens.shape[1]]
        ce=F.cross_entropy(logits[:,1:].reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1).long())
        root_target={'global_motions':target['global_motions'],'local_motions':target['local_motions'],
            'groundtruth_num_tokens':torch.full((len(local),1),frames//4,device=local.device,dtype=torch.long)}
        root=self.inference._root_model.loss(root_target,result['root_output'])
        return {'native_motion':motion['loss'],'native_root':root['loss'],'token_ce':ce,
                'native_total':motion['loss']+root['loss']+ce}

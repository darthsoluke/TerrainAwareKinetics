"""Bounded Root-height interface/geometry ablation using the deployed A100-B model.

Only the new dense-height columns in MB's existing decoder fusion layers learn.
Root/Pose and the encoder/codebook stay fixed; no output is projected or corrected.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from motionbricks.data.native_mb_codec_height import build_codec_height_tokenizer,CODEC_HEIGHT_CONDITIONING
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment
from motionbricks.data.terrain_response_loss import surface_training_losses
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.data.soma_skin_reference import load_official_skinner,move_skinner
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_feedback_ablation import evaluation_goal
from scripts.run_terrain_decoder_pilot import evaluate,frozen_state,save
from scripts.train_native_mb import sha256


def install_height_decoder(inference,bundle):
    model=build_codec_height_tokenizer(inference.motion_rep,inference._vqvae_pose_model,
                                       OmegaConf.create(bundle['tokenizer_config'])).cuda()
    inference._vqvae_pose_model=model.pose_net
    inference._pose_model.supporting_nets['pose_net']=model.pose_net
    inference._pose_model.args['codec_height_conditioning']=CODEC_HEIGHT_CONDITIONING
    pilot=DecoderExperiment(inference,bundle,terrain=False)
    pilot.codec.decoder.requires_grad_(False)
    mode=pilot.codec.decoder_external_cond_feature_mode
    features=np.asarray(extract_feature_from_motion_rep(None,pilot.codec.motion_rep,mode,fetch_feat_idx=True))
    height_index=int(pilot.codec.motion_rep.indices['global_root_y'][0])
    position=int(np.flatnonzero(features==height_index)[0])
    masks={}
    for i in range(pilot.codec.decoder._down_t+int(pilot.codec.decoder._COND_FUSION_LAST_LAYER)):
        block=pilot.codec.decoder.external_cond_blocks[i*2]
        block.weight.requires_grad_(True)
        mask=torch.zeros_like(block.weight)
        frames=2**(pilot.codec.decoder._down_t-i)
        for frame in range(frames):mask[:,block.out_features+frame*len(features)+position]=1
        block.weight.register_hook(lambda gradient,mask=mask:gradient*mask)
        masks[f'external_cond_blocks.{i*2}.weight']=mask.bool()
    return pilot,masks


def surface(pilot,world,skin,indices):
    rep=pilot.inference.global_motion_rep
    pose=rep.inverse(world,False,return_all=True,joint_positions_from='global_rot_data')
    y,xz=region_geometry_from_motion(pose['local_joint_rots'][0],world[0,:,:3],skin,indices)
    return torch.stack((xz[...,0],y,xz[...,1]),-1)


def gradient_norm(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--arm',choices=('height','geometry'),required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new immutable experiment directory')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(2333);np.random.seed(2333)
    inference,bundle=load_terrain_bundle(args.bundle,'cuda:0')
    pilot,masks=install_height_decoder(inference,bundle)
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    selected=[w for w in data['windows'] if w['split']=='train']
    frozen=frozen_state(pilot)
    before={k:v.clone() for k,v in pilot.codec.decoder.state_dict().items()}
    skin=move_skinner(load_official_skinner(inference.global_motion_rep.skeleton),torch.device('cuda:0'))
    skin.skeleton_skin.float().cuda()
    surfaces,*_=official_sole_surfaces(inference.global_motion_rep.skeleton)
    soles=np.concatenate([sole_region_indices(surfaces,side) for side in ('Left','Right')])
    body=np.linspace(0,len(skin.bind_vertices)-1,96).astype(np.int64)
    indices=np.concatenate((soles,body))
    params=[p for p in pilot.codec.decoder.parameters() if p.requires_grad]
    args.output.mkdir(parents=True)
    contract={'source_sha256':sha256(args.bundle),'windows_sha256':sha256(args.data),
        'arm':args.arm,'steps':args.steps,'batch':2,'seed':2333,'frozen_before':frozen,
        'trainable':'only newly initialized Root-Y columns in native decoder external fusion',
        'labels':'unchanged source world/clock; current history only as conditioning',
        'geometry':'official SOMA material vertices; PARC inverted-air-box collision approximation',
        'collision_surface_stride':4,'optimizer':'fresh Adam5e-5, clip0.5, bounded pilot, no long-run warmup',
        'loss_weights':'native 1; geometry gradient norms balanced on first six train samples',
        'live_changed':False,'quality_approved':False}
    save(args.output/'contract.json',contract)
    cache={}

    def forward(index):
        item=selected[index]
        scene=data['scenes'][item['scene']]
        world=item['world'].cuda()
        if index not in cache:
            with torch.no_grad():
                goal=evaluation_goal(inference,world,0,bundle['protocol']['version'])
                _,call=pilot.request(world[:,:4],goal,scene)
                truth=surface(pilot,world[:,4:32],skin,indices).detach()
                # Restrict expensive training geometry to a local target-side tile.
                hf=torch.as_tensor(scene['heightfield_y_m'])
                origin=torch.as_tensor(scene['min_point_world_xyz_m']).clone()
                dx=float(scene['dx_m']);root=world[0,3,[0,2]].cpu()
                cell=torch.floor((root-origin[[0,2]])/dx+.5).long()
                radius=int(np.ceil(4./dx))
                x0=max(0,int(cell[0])-radius);x1=min(hf.shape[1],int(cell[0])+radius+1)
                z0=max(0,int(cell[1])-radius);z1=min(hf.shape[0],int(cell[1])+radius+1)
                crop={'heightfield_y_m':hf[z0:z1,x0:x1].cuda(),
                      'min_point_world_xyz_m':origin.cuda()+world.new_tensor([x0*dx,0,z0*dx]),'dx_m':dx}
                if 'heightfield_valid' in scene:crop['heightfield_valid']=torch.as_tensor(scene['heightfield_valid'])[z0:z1,x0:x1].cuda()
                cache[index]=(call,truth,crop)
        call,truth,crop=cache[index]
        losses,local=pilot.loss(world[:,:33],world[:,:4],call)
        decoded=pilot.restore_world(local,world[:,:4],call['observation'])
        points=surface(pilot,decoded[:,4:32],skin,indices)
        geometry=surface_training_losses(points,truth,crop,len(soles),collision_stride=4)
        return losses,geometry

    # Calibration uses training data only; it is independent of validation success.
    norms=[]
    for index in np.linspace(0,len(selected)-1,6).astype(int):
        losses,geo=forward(int(index))
        norms.append({name:gradient_norm(loss,params) for name,loss in
                      [('native',losses['loss']),('collision',geo['collision_m2']),
                       ('support',geo['source_support_gap_m'])]})
    native=np.mean([row['native'] for row in norms])
    coefficients={name:float(native/max(np.mean([r[name] for r in norms]),1e-8)) for name in ('collision','support')}
    contract.update(calibration=norms,geometry_coefficients=coefficients)
    save(args.output/'contract.json',contract)
    optimizer=torch.optim.Adam(params,lr=5e-5)
    rng=np.random.default_rng(2333);rows=[];started=time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True);log={};draws=rng.integers(0,len(selected),2)
        for index in draws:
            losses,geo=forward(int(index))
            loss=losses['loss']
            if args.arm=='geometry':
                loss=loss+coefficients['collision']*geo['collision_m2']+coefficients['support']*geo['source_support_gap_m']
            if not torch.isfinite(loss):raise ValueError('Nonfinite native/geometry objective')
            (loss/2).backward()
            log={'loss':float(loss.detach()),'native':float(losses['loss'].detach()),
                 'collision_m2':float(geo['collision_m2'].detach()),
                 'source_support_gap_m':float(geo['source_support_gap_m'].detach()),
                 'unknown_points':geo['unknown_points'],'support_points':geo['support_points']}
        torch.nn.utils.clip_grad_norm_(params,.5);optimizer.step()
        if step%50==0 or step+1==args.steps:
            log.update(step=step+1,elapsed_seconds=time.perf_counter()-started)
            rows.append(log);print(json.dumps(log),flush=True)
            save(args.output/'training.json',rows)
    for name,value in pilot.codec.decoder.state_dict().items():
        keep=~masks[name] if name in masks else torch.ones_like(value,dtype=torch.bool)
        if not torch.equal(value[keep],before[name][keep]):raise ValueError('An original decoder weight changed')
    if frozen_state(pilot)!=frozen:raise ValueError('A frozen dependency changed')
    torch.save({'decoder':pilot.codec.decoder.cpu().state_dict(),'height_protocol':CODEC_HEIGHT_CONDITIONING,
                'contract':contract},args.output/'decoder.pt')
    pilot.codec.cuda()
    contract.update(status='trained',elapsed_seconds=time.perf_counter()-started)
    save(args.output/'contract.json',contract)
    evaluate(pilot,bundle,data,args.output/'review')
    contract['status']='evaluated';save(args.output/'contract.json',contract)
    pilot.close()


if __name__=='__main__':main()

"""Offline oracle interventions; never a runtime controller or quality approval."""
import argparse
from pathlib import Path
import json

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_native_mb import sha256
from scripts.evaluate_joint_sdf import save_motion


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    source=Path(saved['contract']['source'])
    if sha256(source)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source identity mismatch')
    inference,bundle=load_terrain_bundle(source,'cuda:0');model=JointSDFController(inference,bundle)
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False);model.eval()
    root=Path(saved['contract']['dataset']);rep=inference.global_motion_rep
    data=TerrainLearningDataset(root,build_native_representation(root/'stats'),'train')
    index=next(i for i,r in enumerate(data.records) if r['source_id']=='official_df3f1541e287fcb3')
    item=data.full_reference_item(index);scene=item['scene'];field=HeightfieldPrismSDF(scene)
    geometry=JointGeometryObjective(rep,'cuda:0');rows=[]
    for start in (0,24,40,48):
        window=item['world'][None,start:start+33].to('cuda:0');history=window[:,:4]
        controls=demonstration_controls(window,32)
        task=SoftCommandState().from_controls(history,controls,8)
        with capture_decoder_call(model.codec) as calls:
            result=model(history,task,[scene],source=window)
        call=calls[-1];tokens=model.codec.encode_into_idx(result['target']['local_motions'],fetch_overall_indices=False)
        expected_local=result['target']['local_motions']
        oracle_external=extract_feature_from_motion_rep(expected_local,model.codec.motion_rep,model.codec.decoder_external_cond_feature_mode)
        _,truth_points=geometry.geometry(window[:,:32])
        for mode in ('generated','replay_generated','oracle_tokens','oracle_root','oracle_tokens_and_root','oracle_native_length',
                     'oracle_native_autoencoder','oracle_continuous_latent'):
            if mode=='generated':world=result['world']
            else:
                pose_tokens=call['args'][0].clone();kwargs=dict(call['kwargs'])
                native_oracle=mode in ('oracle_native_length','oracle_native_autoencoder','oracle_continuous_latent')
                if mode in ('oracle_tokens','oracle_tokens_and_root') or native_oracle:pose_tokens[:,:tokens.shape[1]]=tokens
                if mode in ('oracle_root','oracle_tokens_and_root') or native_oracle:
                    external=kwargs['external_cond'].clone();external[:,:32]=oracle_external
                    kwargs['external_cond']=external
                if native_oracle:
                    pose_tokens=pose_tokens[:,:tokens.shape[1]]
                    for key in ('target_cond','has_target_cond','external_cond'):
                        if torch.is_tensor(kwargs.get(key)):kwargs[key]=kwargs[key][:,:32]
                    if torch.is_tensor(kwargs.get('token_mask')):kwargs['token_mask']=kwargs['token_mask'][:,:tokens.shape[1]]
                if mode=='oracle_native_autoencoder':
                    local=model.codec(expected_local,target_cond=kwargs['target_cond'],
                        has_target_cond=kwargs['has_target_cond'],external_cond=oracle_external)['recon_state']
                elif mode=='oracle_continuous_latent':
                    encoded=model.codec.encoder(model.codec.extract_feature(expected_local,model.codec.encoder_input_feature_mode).transpose(1,2))
                    conditioned=model.codec.extract_feature(kwargs['target_cond'],model.codec.decoder_target_cond_feature_mode)
                    local=model.codec.decoder(encoded,oracle_external,conditioned,kwargs['has_target_cond'],
                        token_mask=kwargs.get('token_mask')).transpose(1,2)
                else:local=model.codec.forward_decoder(pose_tokens,**kwargs)['recon_state'][:,:32]
                world=inference.motion_rep.dual_rep.local_to_global(local,True,False,
                    lengths=torch.full((1,),32,device=local.device,dtype=torch.long))
                offsets,heading,_=inference._extract_initial_root_info(history[...,:5])
                world=inference._reapply_initial_root_info({'pred_global_poses':world,
                    'reference_start_root_global_offsets':offsets,'reference_start_root_global_heading':heading})
                world=shift_global_motion(world,rep,vertical_origin(inference,history,result['observation']))
                if mode=='replay_generated':torch.testing.assert_close(world,result['world'],atol=2e-5,rtol=1e-5)
            joints,points=geometry.geometry(world)
            truth=window[:,:32];tj=rep.inverse(truth,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
            evaluated=points[0,4:28];gap,known=heightfield_gap(evaluated[:,:geometry.sole_count],scene)
            distance,_,valid=field.query(evaluated);vertical,occupied=heightfield_gap(evaluated,scene)
            penetration=(-torch.where(valid,distance,vertical)[occupied]).clamp_min(0)
            row={'mode':mode,'start_frame':start,'future_truth_used_only_for_oracle_diagnosis':mode.startswith('oracle_'),
                'decoder_replay_max_difference':float((world-result['world']).abs().max()) if mode=='replay_generated' else None,
                'root_rms_m':float((world[:,4:28,:3]-truth[:,4:28,:3]).square().sum(-1).mean().sqrt()),
                'relative_mpjpe_m':float(((joints-world[:,:,:3,None].transpose(-1,-2))-(tj-truth[:,:,:3,None].transpose(-1,-2)))[:,4:28].norm(dim=-1).mean()),
                'sole_vertical_p99_m':float(torch.quantile((-gap[known]).clamp_min(0),.99)),
                'sampled_body_penetration_p99_m':float(torch.quantile(penetration,.99)),
                'quality_approved':False}
            rows.append(row);print(json.dumps(row),flush=True)
            save_motion(args.output/f'window{start}-{mode}.npz',world[:,4:28],scene,rep,row)
        save_motion(args.output/f'window{start}-source.npz',window[:,4:28],scene,rep,
            {'role':'unchanged_reference','start_frame':start,'quality_approved':False})
    report={'checkpoint_sha256':sha256(args.checkpoint),'source_id':item['row']['source_id'],'rows':rows,
        'scope':'four source-history windows,24 future frames each; oracle inputs are diagnostic only, never live inference',
        'quality_approved':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())

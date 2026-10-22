"""Read-only Root/decoder/pose decomposition and predicted-query consistency."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF,JOINT_SDF_NAMES
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_model_v3 import (
    multiscale_observation, pack_scene_chunks, retain_intent_observation,
)
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.evaluate_joint_sdf import save_motion
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


class RefinementDiagnostic(JointSDFController):
    """Use the archived actor unchanged as the one-refinement reference."""

    def forward(self, history, intent, scenes, *, path_refinements=1):
        if path_refinements == 1:
            return super().forward(history, intent, scenes)
        if path_refinements != 2:
            raise ValueError('This diagnostic compares one versus two refinements')
        tokens = 8
        packed = pack_scene_chunks(scenes, history.device)
        initial = multiscale_observation(history, intent, scenes, tokens, packed=packed)
        observation = initial
        proposal = None
        for _ in range(path_refinements):
            self._set_sdf(history, proposal, scenes, observation)
            proposal = self._predict(history, intent, observation, tokens, False)
            route = proposal[:, 4:tokens * 4 - 4, [0, 2]].detach()
            refined = multiscale_observation(
                history, intent, scenes, tokens, route=route, packed=packed)
            observation = retain_intent_observation(initial, refined)
            proposal = proposal[:, :tokens * 4]
        self._set_sdf(history, proposal, scenes, observation)
        world = self._predict(history, intent, observation, tokens, False)
        return {'world': world[:, :tokens * 4], 'root_output': self.last_root,
                'pose_logits': self.last_logits, 'observation': observation}


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=torch.load(args.cases,map_location='cpu',weights_only=False)[0];scene=case['scene'];rows=[]
    for entry in args.runs:
        label,path=entry.split('=',1);saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        bundle_path=Path(saved['contract']['source'])
        if sha256(bundle_path)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source identity mismatch')
        inference,bundle=load_terrain_bundle(bundle_path,'cuda:0');rep=inference.global_motion_rep
        model=RefinementDiagnostic(inference,bundle);model.load_state_dict(saved['model_state'],strict=True)
        model.requires_grad_(False);model.eval();geometry=JointGeometryObjective(rep,'cuda:0');field=HeightfieldPrismSDF(scene)
        source=case['world'][None].cuda();joint_ids=[rep.skeleton.bone_order_names.index(n) for n in JOINT_SDF_NAMES]
        original_set_sdf=model._set_sdf;recorded={}
        def record(history,proposal,scenes,observation):
            recorded['proposal']=None if proposal is None else proposal.detach().clone()
            return original_set_sdf(history,proposal,scenes,observation)
        model._set_sdf=record
        for refinements in args.refinements:
            history=source[:,:4].clone();state=SoftCommandState();parts=[];plans=[];incoming=[];planning=[];decoding=[]
            mismatches=[];missed=assessed=0;failure=None;per_block=[]
            for block in range((source.shape[1]-4)//8):
                offset=block*8;task=state.intent(history,case['command_velocity'],case['command_heading'],8)
                try:result=model(history,task,[scene],path_refinements=refinements)
                except ValueError as error:failure=str(error);break
                actual=result['world'][:,4:12]
                plan=rep.unnormalize(result['root_output']['pred_global_root_values'])[:,4:12,:3].clone()
                plan[:,:,[0,2]]+=history[:,:1,[0,2]]
                plan[:,:,1]+=vertical_origin(inference,history,result['observation'])[:,None]
                target=source[:,offset+4:offset+12,:3];prior=source[:,offset+3:offset+4,:3]
                inherited=history[:,3:4,:3]-prior
                new_plan=plan-history[:,3:4,:3]-(target-prior)
                new_decode=actual[:,:,:3]-plan
                torch.testing.assert_close(actual[:,:,:3]-target,inherited+new_plan+new_decode,atol=2e-6,rtol=0)
                parts.append(actual);plans.append(plan);incoming.append(inherited.expand(-1,8,-1));planning.append(new_plan);decoding.append(new_decode)
                proposal=recorded['proposal']
                if proposal is not None:
                    predicted_joints=rep.inverse(proposal[:,4:12],False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][0,:,joint_ids]
                    actual_joints=rep.inverse(actual,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][0,:,joint_ids]
                    discrepancy=(predicted_joints-actual_joints).norm(dim=-1);mismatches.append(discrepancy)
                    pd,_,pk=field.query(predicted_joints);ad,_,ak=field.query(actual_joints);valid=pk&ak
                    missed+=int(((pd>=0)&(ad<-.02)&valid).sum());assessed+=int(valid.sum())
                    per_block.append({'block':block,'query_to_final_joint_mean_m':float(discrepancy.mean()),
                        'query_to_final_joint_max_m':float(discrepancy.max()),'missed_joint_penetrations':int(((pd>=0)&(ad<-.02)&valid).sum())})
                history=reencode_played_history(result['world'][:,8:12],rep)
            row={'arm':label,'refinements':refinements,'frames':sum(p.shape[1] for p in parts),
                'requested_frames':((source.shape[1]-4)//8)*8,'failure':failure,'quality_approved':False,
                'intervention':'model-generated scene re-query only; no motion correction or source future conditions'}
            if parts:
                motion=torch.cat(parts,1);truth=source[:,4:4+motion.shape[1]];plan=torch.cat(plans,1)
                _,points=geometry.geometry(motion);_,truth_points=geometry.geometry(truth)
                sole=points[:,:,:geometry.sole_count];truth_sole=truth_points[:,:,:geometry.sole_count]
                root_error=motion[:,:,:3]-truth[:,:,:3]
                relative_error=(sole-motion[:,:,:3,None].transpose(-1,-2))-(truth_sole-truth[:,:,:3,None].transpose(-1,-2))
                torch.testing.assert_close(sole-truth_sole,root_error[:,:,None]+relative_error,atol=2e-6,rtol=0)
                rms=lambda v:float(v.square().sum(-1).mean().sqrt())
                gap,known=heightfield_gap(sole[0],scene)
                row.update(root_plan_rms_m=rms(plan-truth[:,:,:3]),decoded_root_rms_m=rms(root_error),
                    root_plan_to_decode_rms_m=rms(motion[:,:,:3]-plan),
                    inherited_root_rms_m=rms(torch.cat(incoming,1)),new_root_plan_error_rms_m=rms(torch.cat(planning,1)),
                    new_decoder_root_error_rms_m=rms(torch.cat(decoding,1)),
                    foot_world_rms_m=rms(sole-truth_sole),foot_relative_rms_m=rms(relative_error),
                    sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0),.99)),
                    sole_vertical_max_m=float((-gap[known]).clamp_min(0).max()),
                    root_end_xz=motion[0,-1,[0,2]].tolist(),root_end_y=float(motion[0,-1,1]),
                    unknown_sole_fraction=float((~known).float().mean()))
                if mismatches:
                    discrepancy=torch.cat(mismatches)
                    row.update(query_to_final_joint_mean_m=float(discrepancy.mean()),query_to_final_joint_max_m=float(discrepancy.max()),
                        missed_joint_penetrations=missed,assessed_joint_samples=assessed)
                if refinements==1:
                    expected=Path(path).parent/'final/stairs-source_history_start.npz'
                    if expected.exists():
                        with np.load(expected,allow_pickle=False) as archive:reference=torch.from_numpy(archive['world']).to(motion)
                        torch.testing.assert_close(motion[0],reference,atol=3e-5,rtol=1e-5)
                        row['archived_inference_replay_max_difference']=float((motion[0]-reference).abs().max())
                save_motion(args.output/f'{label}-refine{refinements}.npz',motion,scene,rep,row)
            row['per_block']=per_block;rows.append(row);write_json(args.output/'progress.json',{'rows':rows})
            print(json.dumps({k:v for k,v in row.items() if k!='per_block'}),flush=True)
        del model,inference,bundle,geometry,saved
        gc.collect();torch.cuda.empty_cache()
    write_json(args.output/'complete.json',{'rows':rows,'quality_approved':False,
        'scope':'same training staircase, constant command; exact vector decompositions, not additive RMS or causal attribution'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refinements',nargs='+',type=int,default=[1,2]);main(p.parse_args())

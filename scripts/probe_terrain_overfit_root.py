"""Read-only Root-plan versus actual decoded-root attribution on overfit cases."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_terrain_overfit import select_cases
from scripts.train_joint_terrain_v3 import intents,geometry_precision,write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['schema']!='motionbricks.terrain_overfit_diagnostic.v1':raise ValueError('Only the diagnostic checkpoint is allowed')
    if sha256(args.source)!=saved['contract']['source_sha256']:raise ValueError('Source identity differs')
    infer,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(infer,bundle)
    model.load_state_dict(saved['model_state'],strict=True)
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    cases=select_cases(data)
    for case,expected in zip(cases,saved['contract']['cases']):
        if case['row']['source_id']!=expected['source_id'] or case['start_frame']!=expected['start_frame']:
            raise ValueError('Do not change the diagnostic references')
    report={'checkpoint_sha256':sha256(args.checkpoint),'cases':[],'intervention':None}
    for case in cases:
        world=case['world'][None].to('cuda:0');scene=case['scene'];state=SoftCommandState();history=world[:,:4]
        plans=[];decoded=[];history_errors=[];plan_steps=[];decode_steps=[]
        for offset in range(0,104,8):
            window=world[:,offset:offset+33]
            task=intents([case],window,32,'cuda:0',history,state)
            result=model(history,task,[scene],num_tokens=8)
            plan=infer.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values'])[:,:32,:3].clone()
            plan[:,:,[0,2]]+=history[:,:1,[0,2]]
            plan[:,:,1]+=vertical_origin(infer,history,result['observation'])[:,None]
            anchor=history[:,3:4,:3]
            source_steps=window[:,4:12,:3]-window[:,3:4,:3]
            history_errors.append(anchor-window[:,3:4,:3])
            plan_steps.append(plan[:,4:12]-anchor-source_steps)
            decode_steps.append(result['world'][:,4:12,:3]-plan[:,4:12])
            plans.append(plan[:,4:12]);decoded.append(result['world'][:,4:12,:3])
            with geometry_precision():history=reencode_played_history(result['world'][:,8:12],infer.global_motion_rep)
        plan=torch.cat(plans,1);actual=torch.cat(decoded,1);truth=world[:,4:108,:3]
        inherited=torch.cat([v.expand(-1,8,-1) for v in history_errors],1)
        planning=torch.cat(plan_steps,1);decoding=torch.cat(decode_steps,1)
        torch.testing.assert_close(actual-truth,inherited+planning+decoding,atol=1e-6,rtol=0)
        rmse=lambda v:float(v.square().sum(-1).mean().sqrt())
        report['cases'].append({'name':case['name'],'plan_xyz_rmse_m':rmse(plan-truth),
            'decoded_xyz_rmse_m':rmse(actual-truth),'plan_decode_disagreement_m':rmse(actual-plan),
            'plan_xz_rmse_m':rmse((plan-truth)[...,[0,2]]),
            'decoded_xz_rmse_m':rmse((actual-truth)[...,[0,2]]),
            'plan_decode_xz_disagreement_m':rmse((actual-plan)[...,[0,2]]),
            'inherited_root_error_m':rmse(inherited),
            'new_plan_displacement_error_m':rmse(planning),
            'new_plan_xz_displacement_error_m':rmse(planning[...,[0,2]]),
            'final_plan_error_sum_xyz_m':torch.stack([v[0,-1] for v in plan_steps]).sum(0).cpu().tolist(),
            'final_decoder_error_sum_xyz_m':torch.stack([v[0,-1] for v in decode_steps]).sum(0).cpu().tolist(),
            'final_root_error_xyz_m':(actual-truth)[0,-1].cpu().tolist(),
            'interpretation':'exact per-frame vector decomposition, not additive RMS or a causal intervention'})
    write_json(args.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

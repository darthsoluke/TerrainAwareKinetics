"""Test whether rolling controls can identify the fixed-world feedback targets."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_control import SoftCommandState,CONTROL_VERSION
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_terrain_overfit import select_cases
from scripts.train_joint_terrain_v3 import intents,write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['schema']!='motionbricks.terrain_overfit_diagnostic.v1':raise ValueError('Use the frozen diagnostic artifact')
    if sha256(args.source)!=saved['contract']['source_sha256']:raise ValueError('Source/checkpoint mismatch')
    infer,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(infer,bundle)
    model.load_state_dict(saved['model_state'],strict=True)
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    case=next(c for c in select_cases(data) if c['name']=='walk')
    if torch.count_nonzero(case['scene']['heightfield_y_m']):raise ValueError('The symmetry test requires actual flat terrain')
    source=case['world'][:33][None].to('cuda:0');captures=[]
    def capture(_module,arguments):
        captures.append([a.detach().clone() for a in arguments if torch.is_tensor(a)])
    hook=model.root.register_forward_pre_hook(capture)
    rows=[];baseline=None
    try:
        for shift in (0.,.1,1.):
            history=source[:,:4].clone();history[:,:,0]+=shift
            task=intents([case],source,32,'cuda:0',history,SoftCommandState())
            captures.clear()
            result=model(history,task,[case['scene']],source=source,num_tokens=8)
            inputs={'control_embedding':model._control_embedding.detach().clone(),
                **{k:result['observation'][k].detach().clone() for k in ('points','valid','far_points','far_valid')},
                **{f'root_argument_{i}':a for i,a in enumerate(captures[-1])}}
            raw_target=infer.global_motion_rep.unnormalize(result['target']['global_motions'])
            measured={'inputs':inputs,'target':raw_target.clone(),'world':result['world'].clone()}
            if baseline is None:baseline=measured
            differences={k:float((v.float()-baseline['inputs'][k].float()).abs().max()) for k,v in inputs.items()}
            target_delta=raw_target[:,4:28,:3]-baseline['target'][:,4:28,:3]
            output_delta=result['world'][:,4:28,:3]-baseline['world'][:,4:28,:3]
            expected=output_delta.new_tensor([shift,0,0])
            rows.append({'history_translation_x_m':shift,'neural_input_max_changes':differences,
                'target_xyz_delta_mean_m':target_delta.mean((0,1)).cpu().tolist(),
                'root_output_equivariance_error_m':float((output_delta-expected).abs().max()),
                'target_world_unchanged':True,'future_y_pose_conditioning':False})
    finally:hook.remove()
    report={'source_id':case['row']['source_id'],'checkpoint_sha256':sha256(args.checkpoint),
        'control_version':CONTROL_VERSION,
        'kind':'read-only flat-translation counterexample; no new motion or altered training data',
        'cases':rows}
    write_json(args.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

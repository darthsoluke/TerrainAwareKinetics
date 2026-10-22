"""Isolate lookahead, Root, token selection and final FK response; no live changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_online_live import predict_planar_terrain, online_command_goal
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment, training_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.review_parc_feedback_ablation import evaluation_goal
from scripts.train_native_mb import sha256


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed-scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    codec = inference._vqvae_pose_model
    rep = inference.global_motion_rep
    with np.load(args.seed_scene, allow_pickle=False) as values:
        history = torch.as_tensor(values['initial_history'], device='cuda:0')
    report = {'bundle_sha256': sha256(args.bundle), 'live_changed': False,
              'synthetic_scene_role': 'observation response probes, not motion targets',
              'lookahead': [], 'paired_oracles': []}
    references = {}
    for tokens in (8, 12, 16):
        for distance, height in [(10., 0.), (3., 1.2), (2., 1.2), (1., 1.2),
                                 (.4, 1.2), (1., .3), (1., .6), (1., 2.4)]:
            root = history[0, -1, :3]
            origin = root.new_tensor([float(root[0])-6, 0., float(root[2])-6])
            hf = root.new_zeros((121,121))
            z = origin[2] + torch.arange(121,device=root.device)*.1
            hf[z>=root[2]+distance] = height
            scene = {'heightfield_y_m':hf, 'min_point_world_xyz_m':origin, 'dx_m':.1}
            goal = online_command_goal(history[...,:5], [0,1.6], 0., tokens,
                                       bundle['protocol']['version'])
            diagnostics = {}
            with capture_decoder_call(codec) as calls:
                world, _ = predict_planar_terrain(inference,history,goal,scene,num_tokens=tokens,
                    config={'num_inference_step':4,'pose_token_sampling_use_argmax':True},
                    diagnostics=diagnostics)
            source_tokens = calls[0]['args'][0][:,:tokens]
            root_plan = torch.tensor(diagnostics['root_stage_world_xyz_m'])
            decoded = world[0,4:4*tokens-4,:3].cpu()
            ahead = diagnostics['terrain_patches'][1]
            points = torch.tensor(ahead['world_xyz_m'])
            reference = references.get(tokens)
            row = {'tokens':tokens,'wall_distance_m':distance,'wall_height_m':height,
                'observed_wall': bool((points[:,1]>.01).any()),
                'ahead_extent_m':float(points[:,2].max()-root[2].cpu()),
                'root_prefix_dy_m':float((root_plan[:8,1]-root[1].cpu()).mean()),
                'decoded_prefix_dy_m':float((decoded[:8,1]-root[1].cpu()).mean()),
                'decoded_prefix_dz_m':float((decoded[:8,2]-root[2].cpu()).mean()),
                'root_final_dy_m':float(root_plan[-1,1]-root[1].cpu()),
                'decoded_final_dy_m':float(decoded[-1,1]-root[1].cpu()),
                'decoded_final_dz_m':float(decoded[-1,2]-root[2].cpu())}
            if reference is None:
                references[tokens] = (source_tokens.clone(),root_plan.clone(),decoded.clone())
            else:
                row.update(token_change_fraction=float((source_tokens!=reference[0]).float().mean()),
                    root_response_mae_m=float((root_plan-reference[1]).abs().mean()),
                    decoded_response_mae_m=float((decoded-reference[2]).abs().mean()))
            report['lookahead'].append(row)
    pilot = DecoderExperiment(inference,bundle,terrain=False)
    data = torch.load(args.data,map_location='cpu',weights_only=False)
    for index,item in enumerate(w for w in data['windows'] if w['split']=='validation'):
        world = item['world'].cuda()
        current = world[:,:4]
        goal = evaluation_goal(inference,world,0,bundle['protocol']['version'])
        _, call = pilot.request(current,goal,data['scenes'][item['scene']])
        target = training_targets(inference,world[:,:33],current,call['observation'])
        oracle = pilot.codec.encode_into_idx(target['local_motions'],fetch_overall_indices=False)
        for token_source in ('predicted','source'):
            replay = dict(call,kwargs=dict(call['kwargs']))
            if token_source=='source':
                tokens = call['args'][0].clone();tokens[:,:8]=oracle
                replay['args']=(tokens,)
            decoded = pilot.restore_world(pilot.replay(replay),current,call['observation'])
            poses = rep.inverse(decoded[:,4:12],False,return_all=False,
                                joint_positions_from='global_rot_data')['posed_joints']
            truth = rep.inverse(world[:,4:12],False,return_all=False,
                                joint_positions_from='global_rot_data')['posed_joints']
            report['paired_oracles'].append({'window':index,'origin':item['origin'],
                'category':item['category'],'tokens':token_source,'offline_oracle':token_source=='source',
                'root_y_mae_m':float((decoded[:,4:12,1]-world[:,4:12,1]).abs().mean()),
                'fk_mpjpe_m':float((poses-truth).norm(dim=-1).mean())})
    pilot.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'lookahead':report['lookahead'], 'paired_cases':len(report['paired_oracles'])}),flush=True)


if __name__=='__main__':main()

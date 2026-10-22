"""Read-only source/terrain interventions on the final joint checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState, demonstration_controls
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import intents
from scripts.train_native_mb import sha256


@torch.inference_mode()
def main(checkpoint_path, output, families):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    contract = checkpoint['contract']
    inference, bundle = load_terrain_bundle(Path(contract['source']), 'cuda:0')
    model = JointTerrainController(inference, bundle).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    rep = inference.global_motion_rep
    data = TerrainLearningDataset(Path(contract['dataset']), build_native_representation(Path(contract['dataset'])/'stats'), 'train')
    geometry = JointGeometryObjective(rep, 'cuda:0')
    report = {'checkpoint_sha256': sha256(checkpoint_path), 'checkpoint_step': checkpoint['step'],
        'coefficients': checkpoint['coefficients'], 'cases': [], 'training': False,
        'far_gates': {key: float(getattr(model, key).gate) for key in ('root_far', 'pose_far', 'decoder_far')}}
    for family in families:
        eligible = sorted((i for i, row in enumerate(data.records) if row.get('terrain_family') == family and row['frames_30hz'] >= 129),
            key=lambda i: data.records[i]['source_id'])[:5]
        candidates = []
        for index in eligible:
            item = data.full_reference_item(index)
            world = item['world']
            if len(world) < 129:
                continue
            ascent = world[31:-97, 1]-world[3:-125, 1]
            start = int(ascent.argmax())
            candidates.append((float(ascent[start]), start, item))
        if not candidates:
            continue
        rise, start, item = max(candidates, key=lambda entry: entry[0])
        full = item['world'][None].to('cuda:0')
        source = full[:, start:start+33]
        history = source[:, :4]
        scene = item['scene']
        task = intents([item], source, 32, 'cuda:0', history, SoftCommandState())
        actual = model(history, task, [scene], source=source)
        plane_y = vertical_origin(inference, history, actual['observation']).item()
        plane = dict(scene, heightfield_y_m=torch.full_like(torch.as_tensor(scene['heightfield_y_m']), plane_y))
        removed = model(history, task, [plane])
        sj, sp = geometry.geometry(source[:, :32])
        pj, pp = geometry.geometry(actual['world'])
        fj, fp = geometry.geometry(removed['world'])
        relative = lambda joints, world: joints-world[:, :, None, :3]
        sr = relative(sj, source[:, :32])
        pr = relative(pj, actual['world'])
        fr = relative(fj, removed['world'])
        points = actual['observation']['far_points']
        valid = actual['observation']['far_valid']
        gap, known = heightfield_gap(pp[0, 4:28, :geometry.sole_count], scene)
        source_gap, source_known = heightfield_gap(sp[0, 4:28, :geometry.sole_count], scene)
        row = {'family': family, 'source_id': item['row']['source_id'], 'start_frame': start,
            'source_root_rise_28f_m': rise,
            'model_root_rise_28f_m': float(actual['world'][0,31,1]-history[0,3,1]),
            'flat_intervention_root_rise_28f_m': float(removed['world'][0,31,1]-history[0,3,1]),
            'near_valid_fraction': float(actual['observation']['valid'].float().mean()),
            'far_valid_fraction': float(valid.float().mean()),
            'observed_far_height_range_m': [float(points[...,1][valid].min()), float(points[...,1][valid].max())],
            'first8_root_rmse_m': float((actual['world'][:,4:12,:3]-source[:,4:12,:3]).square().mean().sqrt()),
            'first8_relative_mpjpe_m': float((pr[:,4:12]-sr[:,4:12]).norm(dim=-1).mean()),
            '24f_relative_mpjpe_m': float((pr[:,4:28]-sr[:,4:28]).norm(dim=-1).mean()),
            'terrain_intervention_root_y_rms_m': float((actual['world'][:,4:28,1]-removed['world'][:,4:28,1]).square().mean().sqrt()),
            'terrain_intervention_relative_mpjpe_m': float((pr[:,4:28]-fr[:,4:28]).norm(dim=-1).mean()),
            'source_sole_vertical_penetration_p99_m': float(torch.quantile((-source_gap[source_known]).clamp_min(0), .99)),
            'model_sole_vertical_penetration_p99_m': float(torch.quantile((-gap[known]).clamp_min(0), .99)),
            'rollouts': []}
        for mode in ('teacher', 'reference_feedback', 'rolling_feedback'):
            current = history.clone()
            state = SoftCommandState()
            parts = []
            error = None
            for offset in range(0,104,8):
                window = full[:,start+offset:start+offset+33]
                if mode == 'teacher':
                    current = window[:,:4]
                if mode == 'rolling_feedback':
                    control = demonstration_controls(window,32,category=item['row'].get('control_category'))
                    intent = state.from_controls(current,control,8)
                else:
                    intent = intents([item],window,32,'cuda:0',current,state)
                try:
                    prediction = model(current,intent,[scene])['world']
                except ValueError as exc:
                    error = str(exc)
                    break
                parts.append(prediction[:,4:12])
                current = reencode_played_history(prediction[:,8:12],rep)
            if not parts:
                row['rollouts'].append({'mode':mode,'frames':0,'failure':error})
                continue
            world = torch.cat(parts,1)
            truth = full[:,start+4:start+4+world.shape[1]]
            joints, surface = geometry.geometry(world)
            source_joints, _ = geometry.geometry(truth)
            gap, known = heightfield_gap(surface[0,:,:geometry.sole_count],scene)
            row['rollouts'].append({'mode':mode,'frames':world.shape[1],'failure':error,
                'root_rmse_m':float((world[...,:3]-truth[...,:3]).square().mean().sqrt()),
                'relative_mpjpe_m':float((relative(joints,world)-relative(source_joints,truth)).norm(dim=-1).mean()),
                'sole_vertical_penetration_p99_m':float(torch.quantile((-gap[known]).clamp_min(0),.99))})
        report['cases'].append(row)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    print(json.dumps({'far_gates':report['far_gates'],'coefficients':report['coefficients']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',nargs='+',default=['stairs','climbing','platform','boxes','mixed'])
    args=parser.parse_args()
    main(args.checkpoint,args.output,args.families)

"""Original Root timing versus source-clock stair intents; no time warping."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_native_mb_stair_sequence import all_landing_endpoints, generate_sequence
from scripts.review_native_mb_stairs import (
    _skin, digest, read_shard, render_example, select_windows, stats, support_report, transform_support,
)
from motionbricks.data.stair_sequence_metrics import stair_sequence_metrics


def acceleration_by_join(joints, joins, fps=30):
    acceleration = np.linalg.norm(np.diff(joints, n=2, axis=0) * fps ** 2, axis=-1)
    mask = np.isin(np.arange(1, len(joints) - 1), [f for j in joins for f in (j - 1, j)])
    return {'all_mps2': stats(acceleration), 'join_mps2': stats(acceleration[mask]),
        'interior_mps2': stats(acceleration[~mask]), 'join_center_frames': int(mask.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'root-run', 'pose-run', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lookahead', action='store_true', help='Also test original unplayed-context cursor semantics')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a fresh output directory')
    from scripts.train_native_mb import select_gpu
    gpu = select_gpu(args.gpu)
    import torch
    from scripts.evaluate_native_mb_pipeline import load_bundle
    from motionbricks.data.native_mb_smart_object import NativeSmartObject, rollout_authored_goals, rollout_fixed_lookahead
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    infer, model_dataset, bundle = load_bundle(args.root_run, args.pose_run, 'cuda')
    for filename in ('mean.npy', 'std.npy'):
        if digest(model_dataset / 'stats' / filename) != digest(args.dataset / 'stats' / filename):
            raise ValueError('Review normalization mismatch')
    rep, skeleton = infer.global_motion_rep, infer.global_motion_rep.skeleton
    skin_skeleton = build_native_representation().skeleton
    surfaces, _, _, mesh_hash = official_sole_surfaces(skin_skeleton)
    skinner = load_official_skinner(skin_skeleton)
    primitive = NativeSmartObject(infer)
    report = {'task': 'Original MB duration planning for authored stair interactions', 'bundle': bundle,
        'dataset_manifest_sha256': digest(args.dataset / 'native_manifest.json'),
        'gpu_at_launch': gpu, 'mesh_sha256': mesh_hash, 'quality_approved': False,
        'source_sha256': {Path(__file__).name: digest(__file__),
            'native_mb_smart_object.py': digest(Path(__file__).parents[1] / 'motionbricks/data/native_mb_smart_object.py')},
        'conditions': 'Same source-authored sparse poses/terminal derivatives; initial history only; all subsequent history generated',
        'postprocessing': 'none; native 30 Hz, no time warp or output correction',
        'interpretation': 'Timing varies. Do not compute paired frame MPJPE or use source-timed support masks for predicted timing. Geometry candidate speeds have prediction-dependent coverage and are not measured slip.',
        'lookahead_schedule': 'Optional fixed-source-clock late soft advance; unplayed context regenerated from frame zero, no blending. Not periodic preview replanning; intermediate goals are not hard guarantees.',
        'windows': []}
    args.output.mkdir(parents=True)
    seen = set()
    for index, selected in enumerate(select_windows(args.dataset)):
        if selected['shard'] in seen:
            continue
        seen.add(selected['shard'])
        row = {**selected, 'selection_window_start': selected['start'], 'start': 0}
        data = read_shard(args.dataset / row['shard'])
        frames = (len(data['global_features']) - 1) // 4 * 4
        evidence = support_report(args.dataset, row)
        endpoints, skipped = all_landing_endpoints(evidence['events'], 0,
            infer._args['min_tokens'], infer._args['max_tokens'], frames=frames)
        raw = torch.as_tensor(data['global_features'][:frames + 1], device='cuda')[None]
        with full_precision_evaluation(), torch.inference_mode():
            canonical = rep.change_first_heading(raw, 0., False, False)
            _, angles = rep.compute_root_pos_and_rot(raw, return_angle=True, return_quat=False)
            rotation = angle_to_Y_rotation_matrix(-angles[:, 0])[0].cpu().numpy()
            origin = raw[0, 0, :3].cpu().numpy().copy()
            origin[1] = 0
            patches = transform_support(evidence['patches'], rotation, origin)
            fixed, fixed_plans = generate_sequence(infer, canonical[:, :4],
                [(e, canonical[:, e - 3:e + 2].clone()) for e in endpoints],
                context_count=1, goal_count=1, iterations=1, official_inputs=True)
            predicted, predicted_plans = rollout_authored_goals(primitive, canonical[:, :4],
                [canonical[:, e - 3:e + 1].clone() for e in endpoints])
            features = {'source': canonical[:, :frames], 'fixed_timing': fixed, 'predicted_timing': predicted}
            if args.lookahead:
                lookahead, lookahead_plans = rollout_fixed_lookahead(infer, canonical[:, :4],
                    [(e, canonical[:, e - 3:e + 1].clone()) for e in endpoints])
                features['lookahead_fixed'] = lookahead
            motions = {key: rep.inverse(value, False, return_all=True) for key, value in features.items()}
        joints = {key: value['posed_joints'][0].cpu().numpy() for key, value in motions.items()}
        scores = {}
        plans = {'fixed_timing': fixed_plans, 'predicted_timing': predicted_plans}
        if args.lookahead:
            plans['lookahead_fixed'] = lookahead_plans
        for key, motion in motions.items():
            soles = _skin(motion, skeleton, surfaces, skinner)
            if key == 'source':
                for side in ('left', 'right'):
                    expected = (data[f'sole_vertices_{side}'][:frames] - origin) @ rotation.T
                    if abs(expected - soles[side]).max() > .003:
                        raise ValueError('Official source sole mismatch')
            own_endpoints = endpoints if key != 'predicted_timing' else [p['last_generated_frame'] for p in predicted_plans]
            mask = np.ones(len(joints[key]), dtype=bool)
            mask[:4] = False
            mask[own_endpoints] = False
            mask[np.asarray(own_endpoints) - 1] = False
            geometry = stair_sequence_metrics(soles, 30., patches, near_surface_tolerance_m=.02,
                closest_contact_tolerance_m=.02, scoring_frame_mask=mask,
                scoring_interval_mask=mask[:-1] & mask[1:])
            joins = [p['first_generated_frame'] for p in plans[key]] if key in plans else [4] + [e + 1 for e in endpoints[:-1]]
            targets = joints['source'][endpoints]
            reached = joints[key][own_endpoints]
            error = np.linalg.norm(reached - targets, axis=-1)
            scores[key] = {'frames': len(joints[key]), 'duration_seconds': (len(joints[key]) - 1) / 30,
                'goal_root_error_m': stats(error[:, skeleton.root_idx]),
                'goal_max_joint_error_m': stats(error.max(-1)),
                'acceleration': acceleration_by_join(joints[key], joins), 'geometry': geometry}
        name = f"{index:02d}_{row['direction']}"
        result = {'source': row, 'authored_source_endpoints': endpoints, 'skipped_events': skipped,
            'plans': plans, 'scores': scores, 'artifact': name, 'support_patches': patches}
        report['windows'].append(result)
        np.savez_compressed(args.output / (name + '.npz'), **joints, fps=np.float32(30))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'artifact': name, 'scores': {k: {m: n for m, n in v.items() if m != 'geometry'} for k, v in scores.items()}}), flush=True)
        if args.render:
            # Different durations are rendered separately, never resampled or padded.
            for key in features:
                render_example(args.output / (name + '_' + key), {key: joints[key]}, patches,
                    skeleton.joint_parents.cpu().numpy(), row['direction'],
                    condition_caption='authored sparse goals; native 30 Hz; ' + key)
    report['status'] = 'completed_not_quality_approved'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()

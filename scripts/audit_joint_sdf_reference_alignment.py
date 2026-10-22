"""Read-only reference-time matching: factual frames, unchanged actor and terrain."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from scripts.pilot_joint_sdf_full_context import command, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def choose_reference(costs, *, preferred, previous=None, max_advance=None):
    """Choose a real window using history-only costs; report rather than hide caps."""
    if costs.ndim != 1 or not len(costs) or not torch.isfinite(costs).all():
        raise ValueError('Finite costs for at least one factual window required')
    low = 0 if previous is None else previous
    high = len(costs) - 1 if max_advance is None or previous is None else min(len(costs) - 1, previous + max_advance)
    if not 0 <= low <= high < len(costs):
        raise ValueError('Invalid monotonic reference interval')
    selected = int(costs[low:high + 1].argmin()) + low
    if low <= preferred <= high and costs[preferred] <= costs[selected] + 1e-10:
        selected = preferred
    return selected


def target_metrics(source, history, index, direction):
    target = source[index:index + 33]
    if len(target) != 33:
        raise ValueError('Reference future cannot be padded')
    current = history[0, 3, :3]
    next_delta = target[4, :3] - current
    endpoint_delta = target[11, :3] - current
    factual_step = target[4, :3] - target[3, :3]
    return {'reference_start': index,
            'source_history_root_rms_m': float((history[0, :, :3] - target[:4, :3]).square().sum(-1).mean().sqrt()),
            'next_target_along_command_m': float(next_delta[[0, 2]].dot(direction)),
            'block_target_along_command_m': float(endpoint_delta[[0, 2]].dot(direction)),
            'next_target_step_norm_m': float(next_delta.norm()),
            'factual_source_step_norm_m': float(factual_step.norm()),
            'next_step_displacement_mismatch_m': float((next_delta - factual_step).norm()),
            'next_target_height_change_m': float(next_delta[1]),
            'target_frame_interval': [index + 4, index + 27],
            'published_frame_interval': [index + 4, index + 11]}


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, summaries = [], []
    for entry in args.runs:
        label, path = entry.split('=', 1)
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        source_path = Path(saved['contract']['source'])
        if sha256(source_path) != saved['contract']['source_bundle_sha256']:
            raise ValueError('Source bundle identity mismatch')
        inference, bundle = load_terrain_bundle(source_path, 'cuda:0')
        model = JointSDFController(inference, bundle)
        model.load_state_dict(saved['model_state'], strict=True)
        model.requires_grad_(False)
        model.eval()
        rep = inference.global_motion_rep
        cases = load_cases(args.cases, Path(saved['contract']['dataset']))
        data = TerrainLearningDataset(Path(saved['contract']['dataset']), rep, 'train')
        indices = {row['source_id']: i for i, row in enumerate(data.records)}
        for case in cases[:3]:
            source = case['world'].cuda()
            full = data.full_reference_item(indices[case['label']])['world']
            # Additional candidates are real contiguous source frames. They are
            # scored only offline; no chosen index feeds actor or command time.
            extended = full[case['start_frame']:case['start_frame'] + len(source) + 32].cuda()
            torch.testing.assert_close(extended[:len(source)], source, atol=0, rtol=0)
            joints = rep.inverse(extended[None], False, return_all=False,
                                 joint_positions_from='global_rot_data')['posed_joints'][0]
            starts = torch.arange(len(extended) - 32, device=source.device)
            root_candidates = extended[starts[:, None] + torch.arange(4, device=source.device), :3]
            pose_candidates = joints[starts[:, None] + torch.arange(4, device=source.device)]
            prepared_count = len(source) - 32
            history = source[None, :4].clone()
            state = SoftCommandState()
            previous = None
            chosen = {name: [] for name in ('fixed', 'nearest_root', 'nearest_pose', 'capped_pose', 'extended_pose')}
            actor_fail = None
            identity_errors = 0
            trajectory_rows = []
            histories = []
            for position in range(case['training_segments']):
                fixed = position * 8
                controls = command(case, fixed)
                task = state.from_controls(history, controls, 8)
                try:
                    result = model(history, task, [case['scene']])
                except ValueError as error:
                    actor_fail = str(error)
                    break
                actual_joints = rep.inverse(history, False, return_all=False,
                                            joint_positions_from='global_rot_data')['posed_joints'][0]
                root_cost = (root_candidates - history[0, :, :3]).square().mean((-1, -2))
                pose_cost = (pose_candidates - actual_joints).square().mean((-1, -2, -3))
                original_cost = (pose_candidates[:prepared_count] - joints[fixed:fixed + 4]).square().mean((-1, -2, -3))
                if choose_reference(original_cost, preferred=fixed) != fixed:
                    identity_errors += 1
                choices = {'fixed': fixed,
                           'nearest_root': choose_reference(root_cost[:prepared_count], preferred=fixed),
                           'nearest_pose': choose_reference(pose_cost[:prepared_count], preferred=fixed),
                           'capped_pose': choose_reference(pose_cost[:prepared_count], preferred=fixed,
                                                          previous=previous, max_advance=8),
                           'extended_pose': choose_reference(pose_cost, preferred=fixed)}
                # Identical source-history starts must anchor the constrained
                # path before monotonic updates. A moving spawn is not retimed.
                if position == 0 and choices['capped_pose'] != 0:
                    raise ValueError('Initial source identity was not preserved')
                previous = choices['capped_pose']
                direction = controls[0, :2] / controls[0, :2].norm()
                row = {'arm': label, 'source_id': case['label'], 'feedback_position': position,
                       'fixed_reference_start': fixed, 'matches': {}, 'quality_approved': False}
                for name, index in choices.items():
                    chosen[name].append(index)
                    row['matches'][name] = target_metrics(extended, history, index, direction)
                    row['matches'][name]['all_joint_history_rms_m'] = float((pose_candidates[index] - actual_joints).square().sum(-1).mean().sqrt())
                    row['matches'][name]['at_last_available_window'] = index == (len(extended) - 33 if name == 'extended_pose' else len(source) - 33)
                rows.append(row)
                trajectory_rows.append(row)
                histories.append(history[0].cpu().numpy())
                # Matching cannot affect generation: recompute after all source
                # scoring and assert actor output invariance on every block.
                repeated = model(history, task, [case['scene']])['world']
                torch.testing.assert_close(result['world'], repeated, atol=0, rtol=0)
                history = reencode_played_history(result['world'][:, 8:12], rep)
            summary = {'arm': label, 'source_id': case['label'], 'blocks': len(trajectory_rows),
                       'actor_failure': actor_fail, 'source_history_identity_errors': identity_errors,
                       'actor_replay_max_difference': 0., 'methods': {}}
            goal = case['evaluation_goal_source_frame'] - case['start_frame']
            for name, selection in chosen.items():
                coverage = np.zeros(len(extended), dtype=bool)
                for start in selection:
                    coverage[start + 4:start + 12] = True
                selected_rows = [row['matches'][name] for row in trajectory_rows]
                summary['methods'][name] = {
                    'indices': selection, 'backward_index_changes': int((np.diff(selection) < 0).sum()),
                    'index_jumps_over8': int((np.diff(selection) > 8).sum()),
                    'missing_published_reference_frames_to_goal': np.flatnonzero(~coverage[4:goal + 1]).__add__(4).tolist(),
                    'goal_reference_frame_covered': bool(coverage[goal]),
                    'next_target_behind_command_count': sum(row['next_target_along_command_m'] < 0 for row in selected_rows),
                    'next_step_mismatch_mean_m': float(np.mean([row['next_step_displacement_mismatch_m'] for row in selected_rows])),
                    'next_step_mismatch_max_m': max(row['next_step_displacement_mismatch_m'] for row in selected_rows),
                    'all_joint_history_rms_mean_m': float(np.mean([row['all_joint_history_rms_m'] for row in selected_rows]))}
            summaries.append(summary)
            np.savez_compressed(args.output / (label + '-' + case['label'] + '-histories.npz'),
                                histories=np.asarray(histories), fixed_offsets=np.arange(len(histories)) * 8)
            write_json(args.output / 'progress.json', {'rows': rows, 'summaries': summaries, 'quality_approved': False})
            print(json.dumps(summary), flush=True)
        del model, inference, bundle, saved, data
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / 'complete.json', {'rows': rows, 'summaries': summaries, 'quality_approved': False,
               'scope': 'Offline reference-window scoring of unchanged fixed-clock actor rollouts, no optimizer. '
                        'History matching uses world joints only; source futures never enter the actor. '
                        'Extended candidate pool is real source data, no translation, interpolation or padding. '
                        'Coverage refers to reference targets, not physically traversed actor geometry.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

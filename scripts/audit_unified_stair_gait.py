"""Separate Root displacement from articulated stair gait, without training."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_decoder_pilot import training_targets, vertical_origin
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_native_root_validation import source_root_intervention
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def gait_features(rep, world):
    joints = rep.inverse(world, False, return_all=False,
                         joint_positions_from='global_rot_data')['posed_joints']
    names = rep.skeleton.bone_order_names
    foot_ids = [names.index(side + 'Foot') for side in ('Left', 'Right')]
    leg_ids = [names.index(side + joint) for side in ('Left', 'Right') for joint in ('Leg', 'Shin', 'Foot')]
    relative = joints - world[..., None, :3]
    flexion = []
    for side in ('Left', 'Right'):
        leg = joints[..., [names.index(side + joint) for joint in ('Leg', 'Shin', 'Foot')], :]
        upper, lower = leg[..., 0, :] - leg[..., 1, :], leg[..., 2, :] - leg[..., 1, :]
        cosine = (upper * lower).sum(-1) / (upper.norm(dim=-1) * lower.norm(dim=-1)).clamp_min(1e-8)
        flexion.append(180 - torch.rad2deg(torch.acos(cosine.clamp(-1, 1))))
    knees = torch.stack(flexion, -1)
    return dict(joints=joints, legs=relative[..., leg_ids, :], feet=relative[..., foot_ids, :],
                knees=knees, world_feet=joints[..., foot_ids, :])


def summarize(rep, world, scene):
    values = gait_features(rep, world)
    feet, knees = values['feet'], values['knees']
    gap, known = heightfield_gap(values['world_feet'], scene)
    root_gap, root_known = heightfield_gap(world[..., :3], scene)
    return dict(root_y_min_m=float(world[..., 1].min()), root_y_max_m=float(world[..., 1].max()),
        root_y_range_m=float(world[..., 1].max() - world[..., 1].min()),
        pelvis_above_ground_mean_m=float(root_gap[root_known].mean()),
        knee_flexion_mean_deg=knees.mean((0, 1)).tolist(),
        knee_flexion_p95_deg=torch.quantile(knees.reshape(-1, 2), .95, dim=0).tolist(),
        feet_vertical_separation_p95_m=float(torch.quantile((feet[..., 0, 1] - feet[..., 1, 1]).abs(), .95)),
        relative_foot_y_range_m=(feet[..., 1].amax((0, 1)) - feet[..., 1].amin((0, 1))).tolist(),
        ankle_below_tread_fraction=float(((gap < 0) & known).float().mean()),
        ankle_below_tread_max_m=float(torch.where(known, (-gap).clamp_min(0), 0).max()),
        unknown_ankle_fraction=float((~known).float().mean()))


def difference(rep, left, right):
    a, b = gait_features(rep, left), gait_features(rep, right)
    return dict(root_y_max_difference_m=float((left[..., 1] - right[..., 1]).abs().max()),
        leg_relative_mpjpe_m=float((a['legs'] - b['legs']).norm(dim=-1).mean()),
        foot_relative_mpjpe_m=float((a['feet'] - b['feet']).norm(dim=-1).mean()),
        foot_relative_y_mae_m=float((a['feet'][..., 1] - b['feet'][..., 1]).abs().mean()),
        knee_flexion_mae_deg=float((a['knees'] - b['knees']).abs().mean()))


@contextmanager
def capture_tokens(codec):
    original = codec.forward_decoder
    recorded = []

    def capture(*args, **kwargs):
        recorded.append(args[0].detach().clone())
        return original(*args, **kwargs)

    codec.forward_decoder = capture
    try:
        yield recorded
    finally:
        codec.forward_decoder = original


@contextmanager
def source_token_intervention(inference, source_tokens):
    original = inference._predict_pose_tokens
    had_override = '_predict_pose_tokens' in inference.__dict__
    previous = inference.__dict__.get('_predict_pose_tokens')

    def replace(*args, **kwargs):
        tokens, pose, mask = original(*args, **kwargs)
        replaced = tokens.clone()
        replaced[:, :source_tokens.shape[1]] = source_tokens
        return replaced, pose, mask

    inference._predict_pose_tokens = replace
    try:
        yield
    finally:
        if had_override:
            inference._predict_pose_tokens = previous
        else:
            delattr(inference, '_predict_pose_tokens')


def generate(controller, history, command, scene, expected_tokens=None):
    with capture_tokens(controller.codec) as tokens:
        result = controller(history, command, scene)
    if len(tokens) != 1:
        raise ValueError(f'Expected one final codec decode, got{len(tokens)}')
    result['tokens'] = tokens[0][:, :8].clone()
    expected = result['pose_logits'][:, :8].argmax(-1) if expected_tokens is None else expected_tokens
    if not torch.equal(result['tokens'], expected):
        raise ValueError('Final Pose logits differ from the actual decoded tokens')
    return result


def token_difference(a, b):
    change = a['tokens'] != b['tokens']
    return dict(changed=int(change.sum()), total=change.numel(), fraction=float(change.float().mean()))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    rep = inference.global_motion_rep
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    with np.load(args.scene, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy()) for key in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    rows, series, source_rows = [], [], []
    for run in args.runs:
        saved = torch.load(run / 'candidate.pt', map_location='cpu', weights_only=False)
        if saved['schema'] != SCHEMA or saved['contract']['bundle_sha256'] != sha256(args.bundle):
            raise ValueError('Require matched adaptation-only checkpoints')
        controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        before = state_digest(controller, list(controller.state_dict()))
        report = json.loads((run / 'complete.json').read_text())
        stage = 'final'
        with np.load(run / stage / 'live_stairs/forward-1.6.npz', allow_pickle=False) as archive:
            recording = torch.from_numpy(archive['global_features'].copy()).to(args.device)
        metrics = next(row for row in report[stage]['live_stairs'] if row['speed_request_mps'] == 1.6)
        recorded_summary = {}
        for start, end, label in ((28, 76, 'flat_approach'), (100, 196, 'on_stairs')):
            recorded_summary[label] = summarize(rep, recording[None, start:end], scene)
        series.append(dict(run=run.name, actual_frames=len(recording), sections=recorded_summary))
        for frame in (72, 96, 120, 144, 168):
            history = reencode_played_history(recording[None, frame:frame + 4], rep)
            command = history.new_tensor([[0., 1.6, 1., 0.]])
            actual = generate(controller, history, command, scene)
            origin = vertical_origin(inference, history, actual['observation'])
            predicted = rep.unnormalize(actual['root_output']['pred_global_root_values'][:, :32])
            plan = next(r for r in metrics['plans'] if r['frame'] == frame)
            if not plan['neural_generation']:
                raise ValueError('Selected history must begin a recorded neural plan')
            error = abs(float(predicted[:, 4:28, 1].max() + origin) - plan['root_future_max_y_m'])
            if error > 3e-5:
                raise ValueError(f'Archived neural Root mismatch{error}')
            torch.testing.assert_close(actual['world'][:, 4:28], recording[None, frame + 4:frame + 28],
                                       atol=3e-5, rtol=1e-5)
            flat_scene = dict(scene, heightfield_y_m=torch.full_like(scene['heightfield_y_m'], float(origin)))
            flat = generate(controller, history, command, flat_scene)
            flat_origin = vertical_origin(inference, history, flat['observation'])
            torch.testing.assert_close(origin, flat_origin, atol=1e-6, rtol=0)
            with source_root_intervention(controller, actual['root_output']['pred_global_root_values'][:, :32]):
                fixed_root_flat = generate(controller, history, command, flat_scene)
            with source_root_intervention(controller, flat['root_output']['pred_global_root_values'][:, :32]):
                flat_root_actual = generate(controller, history, command, scene)
            replay = generate(controller, history, command, scene)
            torch.testing.assert_close(replay['world'], actual['world'], atol=0, rtol=0)
            modes = dict(actual=actual, all_flat=flat, same_root_flat_environment=fixed_root_flat,
                         flat_root_real_environment=flat_root_actual)
            row = dict(run=run.name, frame=frame, archived_root_replay_error_m=error, replay_error=0.,
                source_future_used=False, observed_root_world_xyz_m=history[0, 3, :3].tolist(),
                metrics={name: summarize(rep, value['world'][:, 4:28], scene) for name, value in modes.items()},
                vs_actual={name: dict(difference(rep, value['world'][:, 4:28], actual['world'][:, 4:28]),
                                      tokens=token_difference(value, actual)) for name, value in modes.items()
                           if name != 'actual'})
            rows.append(row)
            write_json(args.output / 'progress.json', dict(live_rows=rows, source_rows=source_rows, recordings=series))
        for case in cases:
            for start in (0, 24):
                if start + 33 > len(case['world']):
                    continue
                source = case['world'][None, start:start + 33].to(args.device)
                command = source_terminal_goal(source[..., :5])
                actual = generate(controller, source[:, :4], command, case['scene'])
                target = training_targets(inference, source, source[:, :4], actual['observation'])
                with source_root_intervention(controller, target['global_motions'][..., :5]):
                    oracle = generate(controller, source[:, :4], command, case['scene'])
                height_root = actual['root_output']['pred_global_root_values'][:, :32].clone()
                height_root[..., 1] = target['global_motions'][..., 1]
                with source_root_intervention(controller, height_root):
                    height_oracle = generate(controller, source[:, :4], command, case['scene'])
                source_tokens = controller.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False)
                with source_token_intervention(inference, source_tokens):
                    token_oracle = generate(controller, source[:, :4], command, case['scene'], source_tokens)
                    with source_root_intervention(controller, target['global_motions'][..., :5]):
                        both_oracle = generate(controller, source[:, :4], command, case['scene'], source_tokens)
                replay = generate(controller, source[:, :4], command, case['scene'])
                torch.testing.assert_close(replay['world'], actual['world'], atol=0, rtol=0)
                source_rows.append(dict(run=run.name, label=case['label'], split=case['native_split'], start=start,
                    source=summarize(rep, source[:, 4:28], case['scene']),
                    actual=summarize(rep, actual['world'][:, 4:28], case['scene']),
                    offline_source_root=summarize(rep, oracle['world'][:, 4:28], case['scene']),
                    actual_error=difference(rep, actual['world'][:, 4:28], source[:, 4:28]),
                    oracle_error=difference(rep, oracle['world'][:, 4:28], source[:, 4:28]),
                    height_only_oracle_error=difference(rep, height_oracle['world'][:, 4:28], source[:, 4:28]),
                    source_tokens_oracle_error=difference(rep, token_oracle['world'][:, 4:28], source[:, 4:28]),
                    source_root_and_tokens_oracle_error=difference(rep, both_oracle['world'][:, 4:28], source[:, 4:28]),
                    source_token_accuracy=float((actual['tokens'] == source_tokens).float().mean()),
                    token_changes=token_difference(actual, oracle), replay_error=0., oracle_is_offline_only=True))
        if state_digest(controller, list(controller.state_dict())) != before:
            raise ValueError('Read-only diagnosis altered weights')
        write_json(args.output / 'progress.json', dict(live_rows=rows, source_rows=source_rows, recordings=series))
    write_json(args.output / 'complete.json', dict(live_rows=rows, source_rows=source_rows, recordings=series,
        optimizer_updates=0, weights_unchanged=True, replay_error=0., runtime_changed=False,
        scope='Paired factual-history predictions. Root interventions are discarded offline diagnostics. '
              'Flat comparisons use the same history and command, not a clean flat demonstration. '
              'Ankle-center ground gaps are diagnostic, not full-skin collision certification. '
              'Source Root/height/tokens are factorial OFFLINE oracles only and never published. '
              'No gait classifier or gait-quality pass is inferred from knee flexion alone.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'scene', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise

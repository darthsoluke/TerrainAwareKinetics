"""Offline separation of Root planning and played-pose inputs to token prediction."""

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


@contextmanager
def capture_pose(infer):
    previous = infer._predict_pose_tokens
    calls = []

    def wrapped(batch, config, info):
        before = _snapshot(batch)
        result = previous(batch, config, info)
        calls.append(dict(batch=before, config=dict(config), result=_snapshot(result)))
        return result

    infer._predict_pose_tokens = wrapped
    try:
        yield calls
    finally:
        infer._predict_pose_tokens = previous


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('gait_audit_helpers', args.gait_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    path = Path(saved['contract']['source'])
    if sha256(path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    infer, bundle = load_terrain_bundle(path, 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    rep = infer.global_motion_rep
    pose_indices = extract_feature_from_motion_rep(None, infer.local_motion_rep,
        infer.INTERNAL_POSE_FEATURE_MODE, fetch_feat_idx=True).tolist()
    groups = {name: [pose_indices.index(int(index)) for index in infer.local_motion_rep.indices[key]]
              for name, key in (('hip_height', 'global_root_y'), ('positions', 'ric_data'), ('rotations', 'global_rot_data'))}
    groups['body_without_height'] = groups['positions'] + groups['rotations']
    groups['all'] = list(range(len(pose_indices)))
    if sorted(groups['hip_height'] + groups['body_without_height']) != groups['all']:
        raise ValueError('Pose condition components must cover every input exactly once')
    probes = ([(f'source_{name}_to_pose', False, indices) for name, indices in groups.items()]
              if args.pose_components else [('source_root_to_pose', True, []),
                  ('source_pose_to_pose', False, groups['all']),
                  ('source_root_and_pose_to_pose', True, groups['all'])])
    data_path = Path(saved['contract']['dataset'])
    data = TerrainLearningDataset(data_path, build_native_representation(data_path / 'stats'), 'train')
    case = next(c for c in torch.load(args.cases, map_location='cpu', weights_only=False)
                if c['row']['source_id'] == 'generated_8030ae7a2a7ad640')
    index = next(i for i, row in enumerate(data.records) if row['source_id'] == case['row']['source_id'])
    full = data.full_reference_item(index)
    source = full['world'][case['start_frame']:case['start_frame'] + 113]
    torch.testing.assert_close(source[:len(case['world'])], case['world'], atol=0, rtol=0)
    source = source[None].cuda()
    edge = case['edge_frame']
    velocity = (source[0, edge - 8, [0, 2]] - source[0, 3, [0, 2]]) / ((edge - 11) / 30)
    heading = torch.atan2(source[:, 3:edge - 8, 4].mean(1), source[:, 3:edge - 8, 3].mean(1))
    direction = velocity / velocity.norm()
    report = dict(checkpoint_sha256=sha256(args.checkpoint), gait_helper_sha256=sha256(args.gait_helper),
        source_id=case['row']['source_id'], optimizer_updates=0, quality_approved=False, modes={},
        pose_input_component_columns=groups, scope='All substitutions are isolated offline windows. Actual actor history, terrain/SDF memory, '
              'decoder pose-history conditions and outputs are never replaced in the live actor. '
              'Source Root at fixed source time is future truth, not a causal or feasible recovery label. '
              'Partial pose-component substitutions may be inconsistent bodies; they are input sensitivity probes only.')
    for mode in ('generated_history', 'source_history'):
        directory = args.output / mode
        directory.mkdir()
        history = source[:, :4].clone()
        state = SoftCommandState()
        rows = []
        names = ['actor'] + [name for name, _, _ in probes] + [name + '_and_decoder' for name, root, _ in probes if root]
        variants = {name: [] for name in names}
        for block in range(10):
            reference = source[:, block * 8:block * 8 + 33]
            if mode == 'source_history':
                history = reference[:, :4].clone()
            intent = state.intent(history, velocity, heading, 8)
            with capture_pose(infer) as poses, capture_decoder_call(model.codec) as decoders:
                result = model(history, intent, [case['scene']])
            if len(poses) != 2 or len(decoders) != 2:
                raise ValueError('Unexpected normal actor call count')
            actual = result['world'].clone()
            call = poses[-1]
            decoder = decoders[-1]
            model._pose_step = 0
            replay_tokens = infer._predict_pose_tokens(_snapshot(call['batch']), call['config'], {})[0]
            torch.testing.assert_close(replay_tokens, call['result'][0], atol=0, rtol=0)
            local = model.codec.forward_decoder(replay_tokens, **decoder['kwargs'])['recon_state']
            replay = helper.restore(infer, local, history, result['observation'])
            torch.testing.assert_close(replay, actual, atol=1e-5, rtol=1e-5)
            target = training_targets(infer, reference, history, result['observation'])
            truth_tokens = model.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False)
            truth_global = extract_feature_from_motion_rep(target['global_motions'], rep, 'root')
            truth_local = extract_feature_from_motion_rep(target['local_motions'], infer.local_motion_rep, 'root')
            truth_pose = extract_feature_from_motion_rep(target['local_motions'], infer.local_motion_rep,
                                                         infer.INTERNAL_POSE_FEATURE_MODE)
            truth_external = extract_feature_from_motion_rep(target['local_motions'], model.codec.motion_rep,
                                                               model.codec.decoder_external_cond_feature_mode)
            source_past_error = float((truth_pose[:, :4] - call['batch']['local_poses'][:, :4]).abs().max())
            if mode == 'source_history':
                torch.testing.assert_close(truth_pose[:, :4], call['batch']['local_poses'][:, :4], atol=1e-5, rtol=1e-5)
            row = dict(block=block, replay_max_difference=float((replay - actual).abs().max()),
                       source_past_pose_input_max_difference=source_past_error,
                       actor_minus_source_history_y_m=(history[0, :, 1] - reference[0, :4, 1]).tolist(), variants={})
            worlds = dict(actor=actual)
            tokens_by_arm = dict(actor=replay_tokens)
            baseline_batch = _snapshot(call['batch'])
            baseline_decoder = _snapshot(decoder['kwargs'])
            for arm, root_change, pose_columns in probes:
                batch = _snapshot(call['batch'])
                if root_change:
                    batch['pred_global_root_values'][:, :32] = truth_global
                    batch['pred_local_root_values'][:, :32] = truth_local
                if pose_columns:
                    if batch['has_local_poses'][:, 4:].any() or not batch['has_local_poses'][:, :4].all():
                        raise ValueError('Only the four-frame Pose history may be substituted')
                    batch['local_poses'][:, :4, pose_columns] = truth_pose[:, :4, pose_columns]
                model._pose_step = 0
                tokens, _, _ = infer._predict_pose_tokens(batch, call['config'], {})
                tokens_by_arm[arm] = tokens
                local = model.codec.forward_decoder(tokens, **decoder['kwargs'])['recon_state']
                worlds[arm] = helper.restore(infer, local, history, result['observation'])
                if root_change:
                    decoder_args = _snapshot(decoder['kwargs'])
                    decoder_args['external_cond'][:, :32] = truth_external
                    label = arm + '_and_decoder'
                    tokens_by_arm[label] = tokens
                    local = model.codec.forward_decoder(tokens, **decoder_args)['recon_state']
                    worlds[label] = helper.restore(infer, local, history, result['observation'])
            for key in ('pred_global_root_values', 'pred_local_root_values', 'local_poses'):
                torch.testing.assert_close(call['batch'][key], baseline_batch[key], atol=0, rtol=0)
            for key in ('external_cond', 'target_cond', 'has_target_cond', 'token_mask'):
                torch.testing.assert_close(decoder['kwargs'][key], baseline_decoder[key], atol=0, rtol=0)
            model._pose_step = 0
            after_tokens = infer._predict_pose_tokens(_snapshot(call['batch']), call['config'], {})[0]
            torch.testing.assert_close(after_tokens, replay_tokens, atol=0, rtol=0)
            ref_joints = rep.inverse(reference[:, 4:12], False, return_all=False,
                                     joint_positions_from='global_rot_data')['posed_joints']
            for arm, world in worlds.items():
                played = world[:, 4:12]
                joints = rep.inverse(played, False, return_all=False,
                                     joint_positions_from='global_rot_data')['posed_joints']
                relative = (joints - played[:, :, None, :3]) - (ref_joints - reference[:, 4:12, None, :3])
                row['variants'][arm] = dict(relative_mpjpe_m=float(relative.norm(dim=-1).mean()),
                    token_accuracy=float((tokens_by_arm[arm][:, :8] == truth_tokens).float().mean()),
                    token_change_fraction=float((tokens_by_arm[arm][:, :8] != replay_tokens[:, :8]).float().mean()))
                variants[arm].append(played.clone())
            rows.append(row)
            history = reencode_played_history(actual[:, 8:12], rep)
        summary = dict(rows=rows, variants={})
        for arm, parts in variants.items():
            world = torch.cat(parts, 1)
            summary['variants'][arm] = dict(gait=helper.gait(rep, world, direction),
                relative_mpjpe_m=float(np.mean([r['variants'][arm]['relative_mpjpe_m'] for r in rows])),
                token_accuracy=float(np.mean([r['variants'][arm]['token_accuracy'] for r in rows])))
            if mode == 'generated_history' and arm == 'actor':
                with np.load(args.recording, allow_pickle=False) as archive:
                    expected = torch.from_numpy(archive['world'].copy()).to(world)[None]
                torch.testing.assert_close(world, expected, atol=3e-5, rtol=1e-5)
                summary['recording_replay_max_difference'] = float((world - expected).abs().max())
            save_motion(directory / (arm + '.npz'), world, case['scene'], rep,
                dict(role='continuous_actor' if mode == 'generated_history' and arm == 'actor' else 'offline_condition_diagnostic',
                     frames=80, quality_approved=False, oracle_future_root=arm != 'actor' and 'root' in arm,
                     decoder_history='unchanged factual actor history', diagnostic_only=arm != 'actor' or mode != 'generated_history'))
        report['modes'][mode] = summary
        write_json(args.output / 'progress.json', report)
        print(json.dumps(dict(mode=mode, status='complete')), flush=True)
    write_json(args.output / 'complete.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--gait-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pose-components', action='store_true')
    main(parser.parse_args())

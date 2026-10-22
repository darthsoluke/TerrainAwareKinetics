"""Audit factual rotation encoding and native stochastic recovery proposals offline."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_distance_volume import swept_surface_points
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rotation_error_degrees(left, right):
    relative = left.double().transpose(-1, -2) @ right.double()
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]), -1)
    return torch.rad2deg(torch.atan2(skew.norm(dim=-1) / 2,
                                  (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2))


@torch.no_grad()
def encoding_audit(rep, raw):
    first_info, checks = {}, {}
    observed = reencode_played_history(raw, rep, first_info, checks)
    repeated = reencode_played_history(observed, rep)
    before = rep.inverse(raw, False, return_all=False, return_quat=False, joint_positions_from='global_rot_data')
    after = rep.inverse(observed, False, return_all=False, return_quat=False, joint_positions_from='global_rot_data')
    indices = rep.indices['global_rot_data']
    raw6 = raw[..., indices].reshape(1, 4, -1, 6)
    obs6 = observed[..., indices].reshape_as(raw6)
    def ortho(value):
        a, b = value[..., :3], value[..., 3:]
        return dict(norm_error_max=float(torch.stack(((a.norm(dim=-1) - 1).abs(),
            (b.norm(dim=-1) - 1).abs())).max()), dot_abs_max=float((a * b).sum(-1).abs().max()))
    angle = rotation_error_degrees(before['global_joint_rots'], after['global_joint_rots'])
    # Root XYZ and FK rotations, not the decoder's redundant raw RIC predictions,
    # are the authoritative rendered state. No re-encoded pose is installed here.
    return observed, dict(**first_info, raw_rotation6d=ortho(raw6), observed_rotation6d=ortho(obs6),
        rotation6d_change_max=float((raw6 - obs6).abs().max()),
        physical_rotation_error_max_deg=float(angle.max()),
        repeated_reencoding_features_max=float((observed - repeated).abs().max()),
        physical_position_error_max_m=float(checks['position_error']))


@torch.no_grad()
def candidate_metrics(rep, geometry, scene, field, exact, history, future, direction):
    whole = torch.cat((history, future), 1)
    pose = rep.inverse(whole, False, return_all=True, return_quat=False, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]
    velocity = torch.diff(joints, dim=0) * 30
    angular = rotation_error_degrees(pose['global_joint_rots'][0, 3], pose['global_joint_rots'][0, 4])
    skins = geometry.skin.skin(pose['local_joint_rots'][0, 3:],
        whole[0, 3:, None, :3].expand(-1, 30, -1), rot_is_global=False)
    swept = swept_surface_points(skins, 8)
    penetration, known, uncertain = exact(swept, scene, field)
    names = rep.skeleton.bone_order_names
    foreaft = (joints[4:, names.index('LeftFoot'), [0, 2]] - joints[4:, names.index('RightFoot'), [0, 2]]) @ direction
    root_delta = future[0, :, [0, 2]] - history[0, 3, [0, 2]]
    return dict(inherited_skin_peak_m=float(penetration[0].max()),
        boundary_skin_peak_m=float(penetration[:9].max()),
        future_skin_peak_m=float(penetration[8:].max()),
        future_skin_p99_m=float(torch.quantile(penetration[8:][known[8:]], .99)),
        unknown_samples=int((~known).sum()), uncertain_inside_samples=uncertain,
        forward_progress_m=float(root_delta[-1] @ direction),
        max_cross_track_m=float((root_delta @ torch.stack((-direction[1], direction[0]))).abs().max()),
        boundary_joint_speed_max_mps=float(velocity[3].norm(dim=-1).max()),
        boundary_joint_speed_mean_mps=float(velocity[3].norm(dim=-1).mean()),
        boundary_acceleration_max_mps2=float((velocity[3] - velocity[2]).norm(dim=-1).max() * 30),
        boundary_rotation_step_max_deg=float(angular.max()),
        close_feet_fraction=float((foreaft.abs() < .1).float().mean()),
        foreaft_rms_m=float(foreaft.square().mean().sqrt()))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pose_helpers = module(args.pose_helper, 'pose_recovery_helpers')
    gait = module(args.gait_helper, 'gait_recovery_helpers')
    recovery = module(args.recovery_helper, 'recovery_geometry_helpers')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    path = Path(saved['contract']['source'])
    if sha256(path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source identity mismatch')
    infer, bundle = load_terrain_bundle(path, 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    rep = infer.global_motion_rep
    geometry = JointGeometryObjective(rep, 'cuda:0')
    data_path = Path(saved['contract']['dataset'])
    data = TerrainLearningDataset(data_path, build_native_representation(data_path / 'stats'), 'train')
    case = next(c for c in torch.load(args.cases, map_location='cpu', weights_only=False)
                if c['row']['source_id'] == 'generated_8030ae7a2a7ad640')
    index = next(i for i, row in enumerate(data.records) if row['source_id'] == case['row']['source_id'])
    source = data.full_reference_item(index)['world'][case['start_frame']:case['start_frame'] + 113]
    torch.testing.assert_close(source[:len(case['world'])], case['world'], atol=0, rtol=0)
    source = source[None].cuda()
    edge = case['edge_frame']
    velocity = (source[0, edge - 8, [0, 2]] - source[0, 3, [0, 2]]) / ((edge - 11) / 30)
    heading = torch.atan2(source[:, 3:edge - 8, 4].mean(1), source[:, 3:edge - 8, 3].mean(1))
    direction = velocity / velocity.norm()
    scene, field = case['scene'], HeightfieldPrismSDF(case['scene'])
    report = dict(optimizer_updates=0, quality_approved=False, admitted_labels=0,
        checkpoint_sha256=sha256(args.checkpoint), blocks=[],
        planned_blocks=args.blocks, samples_per_block=args.samples,
        geometry_screen=dict(max_future_skin_peak_m=.02, unknown=0, positive_forward_progress=True),
        scope='Causal native MaskGIT stochastic proposals use observed history/command/terrain and predicted Root only. '
              'Fixed argmax actor supplies factual history; no random candidate is fed back. '
              'Each candidate is retained and measured without projection, IK or altered Root. '
              'Passing local geometry alone is NOT recovery-label or deployment approval. Full32 prediction and8-frame output saved.')
    history, state, actor_parts = source[:, :4].clone(), SoftCommandState(), []
    for block in range(10):
        if block in args.blocks:
            _, source_encoding = encoding_audit(rep, source[:, block * 8:block * 8 + 4])
        with pose_helpers.capture_pose(infer) as poses, capture_decoder_call(model.codec) as decoders:
            result = model(history, state.intent(history, velocity, heading, 8), [scene])
        actual = result['world'].clone()
        actor_parts.append(actual[:, 4:12])
        if block in args.blocks:
            directory = args.output / f'block-{block}'
            directory.mkdir()
            pose_call, decoder = poses[-1], decoders[-1]
            pose_before = pose_helpers._snapshot(pose_call['batch'])
            decoder_before = pose_helpers._snapshot(decoder['kwargs'])
            row = dict(block=block, source_encoding=source_encoding,
                actor_encoding=incoming_encoding if block else source_encoding, candidates=[])
            for candidate in range(args.samples + 1):
                config = dict(pose_call['config'], pose_token_sampling_use_argmax=candidate == 0)
                seed = 90210 + block * 100 + candidate
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    model._pose_step = 0
                    tokens, _, _ = infer._predict_pose_tokens(pose_helpers._snapshot(pose_call['batch']), config, {})
                if candidate == 0:
                    torch.testing.assert_close(tokens, pose_call['result'][0], atol=0, rtol=0)
                if candidate == 1:
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(seed)
                        model._pose_step = 0
                        repeat, _, _ = infer._predict_pose_tokens(pose_helpers._snapshot(pose_call['batch']), config, {})
                    torch.testing.assert_close(repeat, tokens, atol=0, rtol=0)
                local = model.codec.forward_decoder(tokens, **decoder['kwargs'])['recon_state']
                world = gait.restore(infer, local, history, result['observation'])
                if candidate == 0:
                    torch.testing.assert_close(world, actual, atol=1e-5, rtol=1e-5)
                measurements = candidate_metrics(rep, geometry, scene, field, recovery.exact_penetration,
                    history, world[:, 4:12], direction)
                entry = dict(candidate=candidate, seed=seed, argmax=candidate == 0,
                    token_change_fraction=float((tokens[:, :8] != pose_call['result'][0][:, :8]).float().mean()),
                    geometry_only_screen=measurements['future_skin_peak_m'] <= .02 and measurements['unknown_samples'] == 0
                        and measurements['uncertain_inside_samples'] == 0 and measurements['forward_progress_m'] > 0,
                    **measurements)
                row['candidates'].append(entry)
                save_motion(directory / f'candidate-{candidate:02}.npz', world[:, 4:12], scene, rep,
                    dict(entry, role='offline_causal_candidate', quality_approved=False))
                np.savez_compressed(directory / f'candidate-{candidate:02}-full.npz',
                    world=world[0].cpu().numpy(), tokens=tokens.cpu().numpy(), history=history[0].cpu().numpy())
            for key in ('pred_global_root_values', 'pred_local_root_values', 'local_poses'):
                torch.testing.assert_close(pose_call['batch'][key], pose_before[key], atol=0, rtol=0)
            for key in decoder['kwargs']:
                if torch.is_tensor(decoder['kwargs'][key]):
                    torch.testing.assert_close(decoder['kwargs'][key], decoder_before[key], atol=0, rtol=0)
            model._pose_step = 0
            after = infer._predict_pose_tokens(pose_helpers._snapshot(pose_call['batch']), pose_call['config'], {})[0]
            torch.testing.assert_close(after, pose_call['result'][0], atol=0, rtol=0)
            row['geometry_only_passes'] = sum(c['geometry_only_screen'] for c in row['candidates'])
            report['blocks'].append(row)
            write_json(args.output / 'progress.json', report)
            print(json.dumps(dict(block=block, geometry_passes=row['geometry_only_passes'], candidates=len(row['candidates']))), flush=True)
        history, incoming_encoding = encoding_audit(rep, actual[:, 8:12])
    with np.load(args.recording, allow_pickle=False) as archive:
        expected = torch.from_numpy(archive['world'].copy()).to(source)[None]
    actor = torch.cat(actor_parts, 1)
    torch.testing.assert_close(actor, expected, atol=3e-5, rtol=1e-5)
    report['actor_recording_replay_max_difference'] = float((actor - expected).abs().max())
    write_json(args.output / 'complete.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'cases', 'recording', 'pose-helper', 'gait-helper', 'recovery-helper', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--blocks', type=int, nargs='+', default=[0, 3, 6, 8])
    parser.add_argument('--samples', type=int, default=16)
    main(parser.parse_args())

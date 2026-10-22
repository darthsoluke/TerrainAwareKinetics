"""Read-only native Root representation and downstream intervention audit.

Source Root substitution is an offline diagnostic, never an actor or training
target change. No optimizer is constructed and no checkpoint is written.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def source_root_intervention(controller, normalized_root):
    if normalized_root.shape != (1, 32, 5):
        raise ValueError('Offline intervention requires exactly 32 source Root frames')

    def replace(_module, _args, output):
        result = dict(output)
        root = output['pred_global_root_values'].clone()
        root[:, :32] = normalized_root
        result['pred_global_root_values'] = root
        return result

    # The terrain decoder's Root-height observer must see the same intervention.
    handle = controller.root.register_forward_hook(replace, prepend=True)
    try:
        yield
    finally:
        handle.remove()


def rms(value):
    return float(value.square().mean().sqrt())


@torch.no_grad()
def representation_checks(inference, case, device):
    rep, dual = inference.global_motion_rep, inference.motion_rep.dual_rep
    original = case['world'][None].to(device)
    rows = []
    for yaw in (0., np.pi / 2, np.pi, -np.pi / 2):
        source = rep.change_first_heading(original.clone(), yaw, False, False)
        lengths = torch.full((1,), source.shape[1], device=device, dtype=torch.long)
        local = dual.global_to_local(source, False, True, lengths=lengths)
        restored = dual.local_to_global(local, True, False, lengths=lengths)
        offsets, heading, _ = inference._extract_initial_root_info(source[..., :5])
        restored = inference._reapply_initial_root_info(dict(pred_global_poses=restored,
            reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
        sj = rep.inverse(source, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        rj = rep.inverse(restored, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        row = dict(yaw_rad=yaw, frames=source.shape[1],
            root_position_max_m=float((restored[..., :3] - source[..., :3]).abs().max()),
            root_height_max_m=float((restored[..., 1] - source[..., 1]).abs().max()),
            body_feature_max=float((restored[..., 5:] - source[..., 5:]).abs().max()),
            fk_joint_max_m=float((rj - sj).norm(dim=-1).max()),
            all_feature_max=float((restored - source).abs().max()))
        torch.testing.assert_close(restored, source, atol=4e-5, rtol=0)
        rows.append(row)
    rotation = rep.slice(original, 'global_rot_data').reshape(1, -1, rep.num_joints, 6)
    return dict(label=case['label'], source_frames=original.shape[1],
        source_height_range_m=float(original[..., 1].max() - original[..., 1].min()),
        body_rotation_shape=list(rotation.shape), checks=rows)


@torch.no_grad()
def prediction_metrics(controller, prediction, source, target):
    rep = controller.inference.global_motion_rep
    root = rep.unnormalize(prediction['root_output']['pred_global_root_values'][:, :32])
    target_root = rep.unnormalize(target['global_motions'])[..., :5]
    part = slice(4, 28)
    root_error = root[:, part, :3] - target_root[:, part, :3]
    world = prediction['world']
    pj = rep.inverse(world[:, part], False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    sj = rep.inverse(source[:, part], False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    relative = (pj - world[:, part, None, :3]) - (sj - source[:, part, None, :3])
    return dict(root_xyz_component_rms_m=rms(root_error), root_y_rms_m=rms(root_error[..., 1]),
        root_xz_component_rms_m=rms(root_error[..., [0, 2]]),
        root_y_max_m=float(root_error[..., 1].abs().max()),
        decoded_root_y_rms_m=rms(world[:, part, 1] - source[:, part, 1]),
        decoded_root_xyz_component_rms_m=rms(world[:, part, :3] - source[:, part, :3]),
        world_mpjpe_m=float((pj - sj).norm(dim=-1).mean()),
        root_relative_mpjpe_m=float(relative.norm(dim=-1).mean()),
        predicted_root_height_change_m=float(root[0, 27, 1] - root[0, 3, 1]),
        source_root_height_change_m=float(target_root[0, 27, 1] - target_root[0, 3, 1]))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(14291)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hashes = {key: sha256(getattr(args, key)) for key in ('bundle', 'candidate', 'cases')}
    if hashes['bundle'] != '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d':
        raise ValueError('Wrong approved baseline bundle')
    if hashes['candidate'] != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
        raise ValueError('Wrong current live candidate')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    native_keys = list(inference.state_dict())
    native_before = state_digest(inference, native_keys)
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=False)
    if candidate['schema'] != SCHEMA or hashes['cases'] != candidate['contract']['cases_sha256']:
        raise ValueError('Case contract differs from the live candidate training/review sources')
    representation = [representation_checks(inference, case, args.device) for case in cases]
    write_json(args.output / 'representation.json', representation)
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(candidate['adaptation'], strict=True)
    adaptation_keys = list(controller.adaptation.state_dict())
    adaptation_before = state_digest(controller.adaptation, adaptation_keys)
    rows, constraints = [], []

    def inspect_inputs(_module, inputs, kwargs):
        global_mask = kwargs['global_root_component_mask']
        local_mask = kwargs['local_root_component_mask']
        if global_mask[:, 4:, 1].any() or local_mask[:, 4:, 3].any() or inputs[5][:, 4:].any():
            raise ValueError('Normal actor has an unauthorized future height/pose constraint')
        constraints.append(dict(future_global_height_known=False, future_local_height_known=False,
                                future_pose_known=False))

    input_handle = controller.root.register_forward_pre_hook(inspect_inputs, with_kwargs=True)
    try:
        with torch.no_grad():
            for case in cases:
                command = case['exact_constant_control'].to(args.device)
                for start in range(0, len(case['world']) - 32, 24):
                    source = case['world'][None, start:start + 33].to(args.device)
                    history = source[:, :4]
                    controller.enabled = True
                    current = controller(history, command, case['scene'])
                    target = training_targets(inference, source, history, current['observation'])
                    current_metrics = prediction_metrics(controller, current, source, target)
                    # Both arms use the same factual history, command and observation.
                    controller.enabled = False
                    baseline = controller(history, command, case['scene'])
                    before = prediction_metrics(controller, baseline, source, target)
                    controller.enabled = True
                    with source_root_intervention(controller, target['global_motions'][..., :5]):
                        oracle = controller(history, command, case['scene'])
                        oracle_metrics = prediction_metrics(controller, oracle, source, target)
                    if oracle_metrics['root_xyz_component_rms_m'] > 1e-6:
                        raise ValueError('Offline Root substitution was not exact')
                    replay = controller(history, command, case['scene'])
                    parity = float((replay['world'] - current['world']).abs().max())
                    torch.testing.assert_close(replay['world'], current['world'], atol=0, rtol=0)
                    row = dict(label=case['label'], split=case['native_split'], source_start=start,
                        published_frames=24, baseline=before, current=current_metrics,
                        offline_source_root=oracle_metrics, replay_max_feature_error=parity)
                    rows.append(row)
                    write_json(args.output / 'progress.json', dict(completed_windows=len(rows), rows=rows))
                    print(f"{case['label']} start={start} root_y={current_metrics['root_y_rms_m']:.5f} "
                          f"oracle_mpjpe={oracle_metrics['world_mpjpe_m']:.5f}", flush=True)
        if state_digest(inference, native_keys) != native_before:
            raise ValueError('Native weights changed during read-only diagnostic')
        if state_digest(controller.adaptation, adaptation_keys) != adaptation_before:
            raise ValueError('Adaptation weights changed during read-only diagnostic')
        summary = {}
        for name in ('baseline', 'current', 'offline_source_root'):
            summary[name] = {key: float(np.mean([row[name][key] for row in rows]))
                             for key in rows[0][name]}
        write_json(args.output / 'complete.json', dict(status='complete', hashes=hashes,
            cases=len(cases), windows=len(rows), representation=representation, rows=rows, summary=summary,
            root_representation=dict(global_channels=['x', 'y', 'z', 'cos_yaw', 'sin_yaw'],
                local_channels=['yaw_velocity', 'vx', 'vz', 'height'],
                body_joints=inference.global_motion_rep.num_joints, body_rotation='6D for all joints, including pelvis'),
            actor_constraint_checks=len(constraints), source_future_in_normal_actor=False,
            oracle_is_offline_only=True, optimizer_updates=0, weights_unchanged=True,
            live_endpoint_modified=False, traversal_or_generalization_approved=False,
            limitation='Teacher-history windows and exact source Root are diagnostic, not closed-loop WASD certification.'))
    finally:
        input_handle.remove()
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'candidate', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise

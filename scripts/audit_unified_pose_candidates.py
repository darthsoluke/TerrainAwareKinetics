"""Offline native Pose sampling from factual live histories; never a runtime search."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_pose_training import load_pose_checkpoint
from scripts.audit_unified_stair_gait import capture_tokens, gait_features, summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def native_sampling(controller, enabled):
    original = controller._native_predict

    def predict(*args, **kwargs):
        config = dict(kwargs.get('config', {}), pose_token_sampling_use_argmax=not enabled)
        return original(*args, **dict(kwargs, config=config))

    controller._native_predict = predict
    try:
        yield
    finally:
        controller._native_predict = original


@torch.no_grad()
def main(args):
    if not 1 <= args.samples <= 16:
        raise ValueError('Read-only probe is bounded to16native samples per factual history')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    rep = inference.global_motion_rep
    with np.load(args.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    with np.load(args.recording, allow_pickle=False) as data:
        recording = torch.from_numpy(data['global_features'].copy()).to(args.device)
    geometry = JointGeometryObjective(rep, args.device)
    rows, histories, root_reference, deterministic_reference = [], [], {}, {}
    for label, path in [('before', args.initial), ('output', args.candidate)]:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        stage = saved['contract'].get('pose_condition_stage', 'input')
        controller = UnifiedTerrainController(inference, bundle, pose_condition_stage=stage)
        if label == 'before':
            if saved['contract']['bundle_sha256'] != sha256(args.bundle):
                raise ValueError('Initial bundle mismatch')
            controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        else:
            load_pose_checkpoint(controller, saved, sha256(args.bundle))
        controller.eval().requires_grad_(False)
        before_hash = state_digest(controller, list(controller.state_dict()))
        for frame in (48, 72, 96):
            history = reencode_played_history(recording[None, frame:frame + 4], rep)
            if history.shape[1] != 4:
                raise ValueError('Missing actual history')
            history_joints = gait_features(rep, history)['joints']
            command = history.new_tensor([[0., 1.6, 1., 0.]])
            if label == 'before':
                histories.append(dict(frame=frame, skin=exact_skin_metrics(geometry, history, scene),
                    root_xyz_m=history[0, -1, :3].tolist()))
            for sample in range(-1, args.samples):
                seed = 16114 + max(sample, 0)
                with torch.random.fork_rng(devices=[history.device.index]), native_sampling(controller, sample >= 0):
                    torch.manual_seed(seed)
                    with capture_tokens(controller.codec) as tokens:
                        result = controller(history, command, scene)
                if len(tokens) != 1:
                    raise ValueError('Expected one decode per native generation')
                root = result['root_output']['pred_global_root_values'].detach().clone()
                if label == 'before' and sample == -1:
                    root_reference[frame] = root
                    deterministic_reference[frame] = result['world'].clone()
                    torch.testing.assert_close(result['world'][:, 4:28],
                        recording[None, frame + 4:frame + 28], atol=3e-5, rtol=1e-5)
                torch.testing.assert_close(root, root_reference[frame], atol=0, rtol=0)
                future = result['world'][:, 4:28]
                joints = gait_features(rep, future)['joints']
                velocity_before = (history_joints[:, -1] - history_joints[:, -2]) * 30
                velocity_next = (joints[:, 0] - history_joints[:, -1]) * 30
                played = torch.cat((history, future), 1)
                metrics = exact_skin_metrics(geometry, future, scene)
                transition = exact_skin_metrics(geometry, torch.cat((history[:, -1:], future[:, :1]), 1), scene)
                name = f'{label}-frame{frame}-' + ('argmax' if sample < 0 else f'sample{sample:02d}') + '.npz'
                np.savez_compressed(args.output / name, global_features=played[0].cpu().numpy(),
                    tokens=tokens[0][:, :8].cpu().numpy(), fps=np.int32(30))
                row = dict(arm=label, pose_stage=stage, frame=frame, sample=sample, seed=seed,
                    sampling='argmax' if sample < 0 else 'native_gumbel_temperature1', recording=name,
                    actual_frames=28, future_frames=24, root_condition_difference=0.,
                    future_skin=metrics, boundary_swept_skin=transition,
                    root_advance_z_m=float(future[0, -1, 2] - history[0, -1, 2]),
                    root_lateral_m=float((future[..., 0] - history[:, -1:, 0]).abs().max()),
                    first_frame_joint_step_max_m=float((joints[:, 0] - history_joints[:, -1]).norm(dim=-1).max()),
                    boundary_joint_velocity_change_rms_mps=float((velocity_next - velocity_before).square().mean().sqrt()),
                    gait=summarize(rep, future, scene),
                    future_geometry_only_pass=bool(metrics['body_prism_max_m'] <= .02
                        and metrics['unknown_body_fraction'] == 0 and metrics['uncertain_inside_samples'] == 0),
                    recovery_approved=False)
                rows.append(row)
                write_json(args.output / 'progress.json', dict(rows=rows, histories=histories))
            replay = controller(history, command, scene)
            if label == 'before':
                torch.testing.assert_close(replay['world'], deterministic_reference[frame], atol=0, rtol=0)
        if state_digest(controller, list(controller.state_dict())) != before_hash:
            raise ValueError('Read-only probe changed weights')
        controller.close()
    write_json(args.output / 'complete.json', dict(rows=rows, histories=histories,
        initial_sha256=sha256(args.initial), candidate_sha256=sha256(args.candidate),
        recording_sha256=sha256(args.recording), scene_sha256=sha256(args.scene),
        source_future_used=False, optimizer_updates=0, weights_unchanged=True,
        runtime_changed=False, all_candidates_saved=True, quality_approved=False,
        limitation='Finite native samples do not prove impossibility. Geometry-only pass is not '
            'support/continuity/full-route acceptance. History and new-frame metrics are separate. '
            'These self-generated candidates are not expert truth and are never selected at runtime.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'candidate', 'scene', 'recording', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise

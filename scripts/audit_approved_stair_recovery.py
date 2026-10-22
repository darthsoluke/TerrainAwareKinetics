"""Test the preserved stair model on current actual histories; never use as truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.locomotion_command_adapter import install_checkpoint_command
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.audit_unified_stair_gait import summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_unified_goal_alignment import load_world
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(a.checkpoint) != 'ae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934':
        raise ValueError('Require the user-approved preserved stair model')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if saved['contract']['source_bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Teacher baseline mismatch')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    model = JointSDFController(inference, bundle)
    install_checkpoint_command(model, saved)
    model.load_state_dict(saved['model_state'], strict=True)
    model.eval().requires_grad_(False)
    digest = state_digest(model, list(model.state_dict()))
    scene, seed, placed, spawn = load_world(a, inference.global_motion_rep, a.device)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    with np.load(a.recording, allow_pickle=False) as data:
        recording = torch.from_numpy(data['global_features'].copy()).to(a.device)
    rows = []
    for frame in (72, 96, 120):
        history = reencode_played_history(recording[None, frame:frame + 4], inference.global_motion_rep)
        backend = JointTerrainOnlineBackend(model, history, scene)
        result = backend.predict_plan([0., 1.6], 0.)
        motion = result['planned_motion']
        skin = exact_skin_metrics(geometry, torch.cat((history, motion), 1), scene)
        rows.append(dict(frame=frame, generated_frames=motion.shape[1], skin=skin,
            root_boundary_step_m=float((motion[0, 0, :3] - history[0, -1, :3]).norm()),
            gait=summarize(inference.global_motion_rep, motion, scene), not_used_as_training_label=True))
    actor = JointTerrainOnlineBackend(model, placed, scene)
    parts, failure = [placed], None
    for _ in range(100):
        try:
            result = actor.predict_plan([0., 1.6], 0.)
        except (ValueError, RuntimeError) as error:
            failure = repr(error); break
        parts.append(result['planned_motion'].clone())
        actor.history = result['published_history'].clone()
        if float(actor.history[0, -1, 2]) >= -100:
            break
    world = torch.cat(parts, 1)
    np.savez_compressed(a.output / 'rollout.npz', global_features=world[0].cpu().numpy(), fps=np.int32(30))
    skin = exact_skin_metrics(geometry, world, scene)
    if state_digest(model, list(model.state_dict())) != digest:
        raise ValueError('Read-only model changed weights')
    write_json(a.output / 'complete.json', dict(rows=rows, frames=world.shape[1], failure=failure,
        root_z_final_m=float(world[0, -1, 2]), reached_post_stair_goal=bool(world[0, -1, 2] >= -100),
        max_lateral_m=float((world[..., 0] + 20).abs().max()), skin=skin, optimizer_updates=0,
        weights_unchanged=True, runtime_changed=False, recovery_teacher_approved=False,
        note='Preserved whole-version comparison uses its own soft-intent/default8-frame pipeline, '
             'not an isolated causal ablation or a replacement for current WASD.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'checkpoint', 'scene', 'scene-receipt', 'recording', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    try:
        main(a)
    except Exception as error:
        if a.output.exists():
            write_json(a.output / 'failure.json', dict(error=repr(error)))
        raise

"""Closed-loop downhill WASD requests from an authored measured top-platform spawn."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE
from motionbricks.data.unified_pose_training import load_pose_checkpoint
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    native_pose = {k: v.detach().clone() for k, v in controller.pose.state_dict().items()}
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    with np.load(args.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
        seed = torch.from_numpy(data['initial_history'].copy()).to(args.device)
    rep = inference.global_motion_rep
    placed = rep.change_first_heading(seed, np.pi, False, False)
    placed[..., [0, 2]] += placed.new_tensor([-20., -100.]) - placed[:, -1:, [0, 2]].clone()
    original_gap, valid_seed = heightfield_gap(seed[..., :3], scene)
    placed_gap, valid_top = heightfield_gap(placed[..., :3], scene)
    if not valid_seed.all() or not valid_top.all():
        raise ValueError('Respawn terrain must be measured')
    floor = placed[..., 1] - placed_gap
    original_floor = seed[..., 1] - original_gap
    if float(floor.max() - floor.min()) > 1e-6 or float(original_floor.max() - original_floor.min()) > 1e-6:
        raise ValueError('Require flat spawn areas for an unchanged flat initial pose')
    placed = shift_global_motion(placed, rep, (floor[:, -1] - original_floor[:, -1]))
    geometry = JointGeometryObjective(rep, args.device)
    rows = []
    arms = [('mask', args.mask)] if args.mask else [('before', None), ('adapter', args.adapter), ('capacity', args.capacity)]
    for arm, path in arms:
        controller.pose.load_state_dict(native_pose, strict=True)
        controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
        if path is not None:
            load_pose_checkpoint(controller, torch.load(path, map_location='cpu', weights_only=False), sha256(args.bundle))
        for speed in (.8, 1.6):
            actor = UnifiedTerrainBackend(controller, placed, scene)
            actor.configure_runtime(dict(PROFILE))
            parts, failure = [placed.clone()], None
            for _ in range(60):
                try:
                    result = actor.predict_plan([0., -speed], np.pi)
                except (ValueError, RuntimeError) as error:
                    failure = repr(error); break
                parts.append(result['planned_motion'].clone())
                actor.history = result['published_history'].clone()
                if float(actor.history[0, -1, 2]) <= -112:
                    break
            world = torch.cat(parts, 1)
            metrics = exact_skin_metrics(geometry, world, scene)
            reached = float(world[0, -1, 2]) <= -112
            lateral = float((world[..., 0] + 20).abs().max())
            filename = f'down-{speed:g}-{arm}.npz'
            np.savez_compressed(args.output / filename, global_features=world[0].cpu().numpy(), fps=np.int32(30))
            rows.append(dict(arm=arm, speed_request_mps=speed, actual_frames=world.shape[1], recording=filename,
                failure=failure, reached_after_stair_goal=reached, max_lateral_m=lateral, metrics=metrics,
                traversal_screen_pass=bool(reached and failure is None and lateral <= .25
                    and metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0
                    and metrics['uncertain_inside_samples'] == 0)))
            write_json(args.output / 'progress.json', dict(rows=rows))
    write_json(args.output / 'complete.json', dict(rows=rows, spawn_xz=[-20., -100.], goal_z=-112.,
        heading_rad=float(np.pi), spawn_floor_y_m=float(floor[0, -1]), quality_approved=False,
        pose_source='unchanged flat initial history rigidly placed once at measured top platform',
        runtime_correction=False, no_padding=True, optimizer_updates=0))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'adapter', 'capacity', 'scene', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mask', type=Path)
    main(p.parse_args())

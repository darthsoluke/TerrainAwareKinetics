"""Read-only trained flat-regression review with the original playback contract."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.unified_terrain import UnifiedTerrainBackend, UnifiedTerrainController, PROFILE, SCHEMA
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def stats(rep, world):
    tail = world[:, -120:]
    pose = rep.inverse(tail, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
    ids = [rep.skeleton.bone_order_names.index(name) for name in ('LeftFoot', 'RightFoot')]
    offset = pose[:, :, ids]-tail[:, :, None, :3]
    yaw = torch.atan2(tail[..., 4], tail[..., 3])
    forward = offset[..., 0]*yaw.sin()[..., None]+offset[..., 2]*yaw.cos()[..., None]
    spread = torch.quantile(forward, .95, dim=1)-torch.quantile(forward, .05, dim=1)
    speed = torch.diff(tail[:, -31:, [0, 2]], dim=1).norm(dim=-1)*30
    return dict(last_second_mean_root_speed_mps=float(speed.mean()),
                last4s_body_relative_foot_foreaft_p90_m=spread[0].cpu().tolist(),
                root_displacement_m=float((world[0, -1, [0, 2]]-world[0, 0, [0, 2]]).norm()),
                finite=bool(torch.isfinite(world).all()))


@torch.no_grad()
def runs(inference, controller, seed, scene, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for yaw in (0., np.pi/2, np.pi, -np.pi/2):
        for speed in (1.6, 5.):
            for idle in (0, 60):
                cases.append((f'{yaw:.6f}-{speed:g}-{idle}', yaw,
                              [(yaw, 0., idle), (yaw, speed, 30)]))
    cases.append(('continuous-wasd-release', 0., [(0., 1.6, 6), (0., 0., 6),
        (np.pi/2, 1.6, 6), (-np.pi/2, 1.6, 6), (np.pi, 1.6, 6), (0., 1.6, 6)]))
    rows = []
    for label, first_yaw, schedule in cases:
        history = inference.global_motion_rep.change_first_heading(seed.clone(), first_yaw, False, False)
        backend = (NativeTerrainOnlineBackend(inference, history, scene) if controller is None
                   else UnifiedTerrainBackend(controller, history, scene))
        backend.configure_runtime(dict(PROFILE))
        pieces = [history]
        failure = None
        for yaw, speed, blocks in schedule:
            for _ in range(blocks):
                try:
                    result = backend.predict_plan([np.sin(yaw)*speed, np.cos(yaw)*speed], yaw)
                    pieces.append(result['planned_motion'].clone())
                    backend.history = result['published_history'].clone()
                except (ValueError, RuntimeError) as error:
                    failure = repr(error)
                    break
            if failure:
                break
        world = torch.cat(pieces, 1)
        metrics = stats(inference.global_motion_rep, world)
        rows.append(dict(case=label, actual_frames=world.shape[1],
                         requested_frames=4+sum(item[2] for item in schedule)*8, failure=failure, metrics=metrics))
        np.savez_compressed(output/(label+'.npz'), global_features=world[0].cpu().numpy(), fps=np.int32(30))
        write_json(output/'progress.json', dict(rows=rows))
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.candidate, map_location='cpu', weights_only=False)
    if saved['schema'] != SCHEMA or saved['contract']['bundle_sha256'] != sha256(args.bundle):
        raise ValueError('Candidate schema or baseline identity mismatch')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    with np.load(args.seed_scene, allow_pickle=False) as archive:
        seed = torch.from_numpy(archive['initial_history'].copy()).to(args.device)
    seed = inference.global_motion_rep.change_first_heading(seed, 0., False, False)
    seed[..., [0, 2]] -= seed[:, -1:, [0, 2]].clone()
    scene = dict(heightfield_y_m=torch.zeros(201, 201), min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
    write_json(args.output/'state.json', dict(status='baseline'))
    before = runs(inference, None, seed, scene, args.output/'before')
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    write_json(args.output/'state.json', dict(status='candidate'))
    after = runs(inference, controller, seed, scene, args.output/'after')
    write_json(args.output/'complete.json', dict(before=before, after=after,
        candidate_sha256=sha256(args.candidate), optimizer_updates=0, no_padding=True,
        quality_approved=False, note='Kinematic flat-regression diagnostics, not force/contact or WASD acceptance.'))
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'candidate', 'seed-scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise

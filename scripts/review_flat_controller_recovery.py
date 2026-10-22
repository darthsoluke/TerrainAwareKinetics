"""Compare the preserved Oct11 native controller with later learned controls."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.locomotion_command_adapter import install_checkpoint_command
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_runtime_controls import RUNTIME_DEFAULTS, JOINT_RUNTIME_DEFAULTS
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import gait_metrics
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def foot_diagnostics(rep, motion, direction):
    joints = rep.inverse(motion, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    ids = [rep.skeleton.bone_order_names.index(side+'Foot') for side in ('Left', 'Right')]
    relative = joints[:, ids]-motion[0, :, None, :3]
    foreaft = relative[..., [0, 2]]@direction
    rows = {}
    for index, side in enumerate(('left', 'right')):
        tail = foreaft[-120:, index]
        energy = torch.fft.rfft(tail-tail.mean()).abs().square()
        hz = torch.fft.rfftfreq(len(tail), d=1/30).to(energy)
        rows[side] = dict(foreaft_p95_minus_p05_m=float(torch.quantile(tail, .95)-torch.quantile(tail, .05)),
            spectral_energy_above5hz_fraction=float(energy[hz>5].sum()/energy[hz>0].sum().clamp_min(1e-12)))
    return dict(last4_seconds=rows, scope='Unmodified body-relative foot motion; spectrum is diagnostic,not a physical contact or quality gate')


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.bundle) != '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d':
        raise ValueError('Expected preserved Oct11 A100-B source bundle')
    if sha256(args.seed_scene) != 'e19876a349c54ba4a2c663b358f9e33bc0bab2e115f302e91ff3f63b7f819ffa':
        raise ValueError('Expected original world entry history artifact')
    args.output.mkdir(parents=True, exist_ok=False)
    for profile in args.profiles:
        inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
        native = profile.startswith('old')
        tokens = int(profile.removeprefix('old').removeprefix('current'))
        model = None
        if not native:
            saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
            model = JointSDFController(inference, bundle)
            install_checkpoint_command(model, saved)
            model.load_state_dict(saved['model_state'], strict=True)
            model.requires_grad_(False); model.eval()
        rep = inference.global_motion_rep
        geometry = JointGeometryObjective(rep, 'cuda:0')
        with np.load(args.seed_scene, allow_pickle=False) as archive:
            seed = torch.from_numpy(archive['initial_history'].copy()).cuda()
        seed = rep.change_first_heading(seed, 0., False, False)
        seed[..., [0, 2]] -= seed[:, -1:, [0, 2]].clone()
        scene = dict(heightfield_y_m=torch.zeros(201, 201), min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
        folder = args.output/profile
        folder.mkdir()
        settings = dict(RUNTIME_DEFAULTS if native else JOINT_RUNTIME_DEFAULTS, num_tokens=tokens)
        contract = dict(profile=profile, bundle_sha256=sha256(args.bundle),
            checkpoint_sha256=None if native else sha256(args.checkpoint), settings=settings,
            original_conditioning=inference._root_model.args.get('online_conditioning'),
            direct_backend='native' if native else 'joint', source_future_in_actor=False, optimizer_updates=0,
            postprocessing=None, original_interactive_defaults=(profile=='old12'), quality_approved=False)
        write_json(folder/'contract.json', contract)
        rows = []
        for idle_blocks in (0, 60):
            for yaw in (0., np.pi):
                for speed in (1.6, 5.):
                    history = rep.change_first_heading(seed.clone(), yaw, False, False)
                    backend = NativeTerrainOnlineBackend(inference, history, scene) if native else JointTerrainOnlineBackend(model, history, scene)
                    backend.configure_runtime(settings)
                    axis = history.new_tensor([np.sin(yaw), np.cos(yaw)])
                    initial = history[0, 3, [0, 2]].clone()
                    pieces, generation_seconds = [], []
                    for block in range(idle_blocks+30):
                        if block == idle_blocks:
                            moving_start = backend.history[0, 3, [0, 2]].clone()
                        velocity = axis*(0. if block<idle_blocks else speed)
                        result = backend.predict_plan(velocity, yaw)
                        pieces.append(result['planned_motion'].clone())
                        generation_seconds.append(result['generation_seconds'])
                        backend.history = result['published_history'].clone()
                    full = torch.cat(pieces, 1)
                    moving = full[:, idle_blocks*8:]
                    delta = moving[0, -1, [0, 2]]-moving_start
                    velocity = torch.diff(moving[0, :, [0, 2]], dim=0)*30
                    name = f'idle{idle_blocks}-yaw{round(yaw/np.pi*180)}-speed{speed:g}'
                    row = dict(name=name, idle_seconds=idle_blocks*8/30, yaw=yaw, request_speed=speed,
                        frames=full.shape[1], movement_frames=240, idle_drift_m=float((moving_start-initial).norm()),
                        forward_m=float(delta@axis), side_m=float(delta@torch.stack((-axis[1], axis[0]))),
                        speed_last_second=float((velocity[-30:]@axis).mean()),
                        **gait_metrics(rep, moving, axis), **exact_skin_metrics(geometry, moving, scene),
                        foot_diagnostics=foot_diagnostics(rep, moving, axis),
                        median_generation_seconds=float(np.median(generation_seconds)))
                    rows.append(row)
                    save_motion(folder/(name+'.npz'), full, scene, rep, row)
                    write_json(folder/'progress.json', dict(rows=rows))
                    print(profile, name, row['forward_m'], row['close_feet_fraction'], flush=True)
                    del backend
        write_json(folder/'complete.json', dict(contract=contract, rows=rows, quality_approved=False))
        del model, geometry, inference, bundle
        torch.cuda.empty_cache()
    write_json(args.output/'complete.json', dict(profiles=args.profiles, optimizer_updates=0, quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'seed-scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profiles', nargs='+', choices=('old8', 'old12', 'current8', 'current12'), required=True)
    main(parser.parse_args())

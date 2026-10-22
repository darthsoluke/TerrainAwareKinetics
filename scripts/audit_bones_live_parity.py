"""Compare deployed online execution to direct actor and test idle-seed motion."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.pilot_joint_sdf_contact_preserve import gait_metrics


@torch.inference_mode()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    infer, bundle = load_terrain_bundle(args.bundle, args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['contract']['source_bundle_sha256'] != sha256(args.bundle):
        raise ValueError('Checkpoint dependency mismatch')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False); model.eval()
    with np.load(args.scene, allow_pickle=False) as values:
        initial = torch.from_numpy(values['initial_history'].copy()).to(args.device)
    rep = infer.global_motion_rep
    initial = rep.change_first_heading(initial, 0., False, False)
    initial[..., [0, 2]] -= initial[:, -1:, [0, 2]].clone()
    flat = dict(heightfield_y_m=torch.zeros(201, 201),
        min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
    backend = JointTerrainOnlineBackend(model, initial, flat)
    backend.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
    state = SoftCommandState()
    direct_history = initial.clone()
    parity = []
    for speed in (1.6, 1.6, 0., 0., 1.6):
        task = state.intent(direct_history, [0., speed], 0., 8)
        direct = model(direct_history, task, [flat])
        world = direct['world'].clone()
        observation = {k:v.clone() for k,v in direct['observation'].items() if isinstance(v, torch.Tensor)}
        online = backend.predict_plan([0., speed], 0.)
        published = reencode_played_history(world[:, 8:12], rep)
        row = dict(speed=speed, world_error=float((world[:, 4:12]-online['planned_motion']).abs().max()),
            history_error=float((published-online['published_history']).abs().max()),
            stop_goal_error=float((state.stop_goal-backend.command_state.stop_goal).abs().max()))
        torch.testing.assert_close(world[:, 4:12], online['planned_motion'], atol=3e-5, rtol=1e-5)
        parity.append(row)
        backend.history = online['published_history'].clone()
        direct_history = published
    # Committed prefetch uses the exact immutable next block, not an invented prefix.
    serial = JointTerrainOnlineBackend(model, initial, flat)
    queued = JointTerrainOnlineBackend(model, initial, flat)
    for b in (serial, queued):
        b.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS)); b.committed_history_prefetch = True
    queue_rows = []
    first = queued.predict_plan([0., 1.6], 0.)
    previous = first['published_history']
    expected = serial.predict_plan([0., 1.6], 0.)
    serial.history = expected['published_history']
    for speed in (1.6, 0., 0., 1.6):
        candidate = queued.prepare_next_plan([0., speed], 0., queued.history.clone(), previous.clone(), overlap_frames=8)
        queued.history = previous.clone()
        queued.commit_prepared_plan(candidate)
        direct = serial.predict_plan([0., speed], 0.)
        error = float((candidate['planned_motion']-direct['planned_motion']).abs().max())
        torch.testing.assert_close(candidate['planned_motion'], direct['planned_motion'], atol=3e-5, rtol=1e-5)
        queue_rows.append(dict(speed=speed, max_error=error))
        previous = candidate['published_history']; serial.history = direct['published_history']
    write_json(args.output/'parity.json', dict(direct_vs_backend=parity, serial_vs_prefetch=queue_rows))
    cases = [(yaw, speed) for yaw in (0., torch.pi/2, torch.pi, -torch.pi/2) for speed in (1.6, 5.)]
    histories = torch.cat([rep.change_first_heading(initial.clone(), yaw, False, False) for yaw,speed in cases])
    axes = histories.new_tensor([[np.sin(yaw), np.cos(yaw)] for yaw,speed in cases])
    controls = torch.cat((axes*histories.new_tensor([s for y,s in cases])[:, None],
        histories.new_tensor([[np.cos(y),np.sin(y)] for y,s in cases])), -1)
    state = SoftCommandState(); parts = []
    idle_start = histories[:, 3, [0, 2]].clone()
    for block in range(args.idle_blocks):
        idle = controls.clone(); idle[:, :2] = 0
        result = model(histories, state.from_controls(histories, idle, 8), [flat]*len(cases))['world']
        histories = reencode_played_history(result[:, 8:12], rep)
    start = histories[:, 3, [0, 2]].clone()
    for block in range(30):
        task = state.from_controls(histories, controls, 8)
        result = model(histories, task, [flat]*len(cases))['world']
        parts.append(result[:, 4:12].clone())
        histories = reencode_played_history(result[:, 8:12], rep)
    motion = torch.cat(parts, 1)
    rows = []
    for i,(yaw,speed) in enumerate(cases):
        velocity = torch.diff(motion[i, :, [0, 2]], dim=0)*30
        forward = velocity@axes[i]
        delta = motion[i, -1, [0,2]]-start[i]
        rows.append(dict(yaw=yaw, request_speed=speed, frames=motion.shape[1],
            idle_seconds=args.idle_blocks*8/30, idle_drift_m=float((start[i]-idle_start[i]).norm()),
            forward_m=float(delta@axes[i]), side_m=float(delta@torch.stack((-axes[i,1],axes[i,0]))),
            speed_last_second=float(forward[-30:].mean()),
            low_progress_fraction=float((forward < .2).float().mean()),
            root_y_min=float(motion[i,:,1].min()), root_y_max=float(motion[i,:,1].max()),
            **gait_metrics(rep,motion[i:i+1],axes[i])))
    np.savez_compressed(args.output/'flat-rollouts.npz', world=motion.cpu().numpy())
    write_json(args.output/'complete.json', dict(direct_vs_backend=parity, serial_vs_prefetch=queue_rows,
        flat_idle_seed=rows, checkpoint_sha256=sha256(args.checkpoint), optimizer_updates=0,
        future_source_used=False, postprocessing=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','scene','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--idle-blocks', type=int, default=0)
    main(parser.parse_args())

"""Matched frozen-model rollouts isolating the decoded context integration origin."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scripts.soma_pfnn_maps_server import PFNNMapRuntime
from motionbricks.data.reactive_kinematic import ReactiveNativeEngine


def replay_analog(args, runtime, engine):
    from motionbricks.data.reactive_kinematic import ReactivePlayback

    events = [json.loads(line) for line in args.journal.read_text().splitlines()]
    seed = np.asarray(events[0]['seed'], dtype=np.float32)
    inputs = {}
    for event in events:
        if event['kind'] == 'intent':
            inputs.setdefault(event['frame'], []).append(event)
    frames = max(event['frame'] for event in events if event['kind'] == 'played')
    spec = importlib.util.spec_from_file_location('saved_reactive_v3', args.old_playback)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = dict(journal=str(args.journal), frames=frames, optimizer_updates=0,
        clock='recorded input frame; synchronous inference; fixed three-frame lead', cases=[])
    variants = [('original', module.ReactivePlayback, False, 4),
        ('cadence', ReactivePlayback, False, 4), ('cadence_origin', ReactivePlayback, True, 4)]
    if args.longer:
        class LongerPlayback(ReactivePlayback):
            def configure(self, step_frames):
                self.step_frames = step_frames
        variants = [(f'cadence_{step}', LongerPlayback, False, step) for step in (8, 24)]
    for name, playback_type, anchored, step in variants:
        state = playback_type(seed, step_frames=step)
        state.started = True
        geometry, generated, roots, joints, commands, plan_ids = {}, 0, [], [], [], []
        for frame in range(frames):
            for event in inputs.get(frame, []):
                state.receive_intent(dict(seq=event['seq'], velocity_xz=event['velocity'],
                    heading=event['heading'], client_input_ms=event['client_input_ms']))
            if state.due():
                request = state.capture()
                history = torch.tensor(request.history.copy(), device=runtime.inference.device)[None]
                result = engine.path(history, history.new_tensor([[*request.velocity, request.heading]]))
                motion, joint = result['motion'][0, 4:28].clone(), result['joints'].clone()
                if anchored:
                    delta = result['history'][0, 3, [0, 2]] - result['motion'][0, 3, [0, 2]]
                    motion[:, [0, 2]] += delta
                    joint[..., [0, 2]] += delta
                plan = state.publish(request, motion.cpu().numpy())
                state.accept(1, plan, request.origin_frame + 1)
                geometry[plan] = (request.origin_frame, joint.cpu().numpy())
                generated += 1
            plan, sample = state.scheduled_frame(frame+1)
            origin, joint = geometry[plan]
            roots.append(sample[:3].copy())
            joints.append(joint[frame-origin].copy())
            commands.append([*state.command['velocity'], state.command['heading']])
            plan_ids.append(plan)
            state.acknowledge(1, frame+1, plan)
            geometry = {key:value for key,value in geometry.items() if key in state.plans}
        roots, joints, commands = np.asarray(roots), np.asarray(joints), np.asarray(commands)
        velocity = np.diff(np.concatenate((seed[-1:, :3], roots)), axis=0)*30
        speed = np.linalg.norm(commands[:, :2], axis=-1)
        direction = commands[:, :2]/np.maximum(speed[:, None], 1e-9)
        along = (velocity[:, [0, 2]]*direction).sum(-1)
        moving = speed > .1
        body_step = np.linalg.norm(np.diff(joints-roots[:, None], axis=0), axis=-1).max(-1)
        changes = np.r_[True, np.diff(plan_ids) != 0]
        lengths = np.diff(np.r_[np.where(changes)[0], frames])
        item = dict(name=name, generations=generated, one_frame_plans=int((lengths==1).sum()),
            played_plans=len(lengths), root_acceleration_p95=float(np.percentile(np.linalg.norm(np.diff(velocity, axis=0)*30, axis=-1), 95)),
            body_step_max_joint_p95_m=float(np.percentile(body_step, 95)),
            progress_mean_mps=float(along[moving].mean()), requested_mean_mps=float(speed[moving].mean()),
            velocity_error_mean_mps=float(np.linalg.norm(velocity[moving][:, [0, 2]]-commands[moving, :2], axis=-1).mean()),
            root_y_range_m=[float(roots[:, 1].min()), float(roots[:, 1].max())])
        np.savez_compressed(args.output/f'{name}.npz', roots=roots, joints=joints, commands=commands,
            plan_ids=plan_ids, velocity=velocity)
        report['cases'].append(item)
        (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(item), flush=True)


def metrics(motion, joints, phases, seed):
    xyz = motion[:, :3]
    velocity = np.diff(np.concatenate((seed[-1:, :3], xyz)), axis=0) * 30
    acceleration = np.diff(velocity, axis=0) * 30
    body = joints - xyz[:, None]
    body_delta = np.linalg.norm(np.diff(body, axis=0), axis=-1)
    output = dict(root_acceleration_p95=float(np.percentile(np.linalg.norm(acceleration, axis=-1), 95)),
        planar_acceleration_p95=float(np.percentile(np.linalg.norm(acceleration[:, [0, 2]], axis=-1), 95)),
        body_step_p95_m=float(np.percentile(body_delta, 95)),
        body_step_max_m=float(body_delta.max()))
    start = 0
    for name, frames, control in phases:
        stop = start + frames
        settled = slice(start + min(30, frames // 2), stop)
        actual = velocity[settled][:, [0, 2]]
        target = np.asarray(control[:2])
        output[name] = dict(mean_velocity_xz=actual.mean(0).tolist(),
            speed_mean=float(np.linalg.norm(actual, axis=-1).mean()),
            command_error_mean=float(np.linalg.norm(actual - target, axis=-1).mean()),
            displacement_xz=(xyz[stop-1, [0, 2]] - (seed[-1, [0, 2]] if start == 0 else xyz[start-1, [0, 2]])).tolist(),
            root_y_range=[float(xyz[start:stop, 1].min()), float(xyz[start:stop, 1].max())])
        start = stop
    return output


@torch.inference_mode()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = PFNNMapRuntime(args.bundle, args.catalog, 'cuda:0', 'root-path-live', args.map,
        planning_policy='dual', dual_proof=args.dual_proof)
    engine = ReactiveNativeEngine(runtime, json.loads(args.scene_proofs.read_text())['scene_query_proofs'])
    if args.flat_artifact:
        runtime._active._spatial_index = 'native'
        runtime._active.switch_artifact(args.flat_artifact)
        runtime.scene_revision += 1
        runtime.current_map = 'flat64-probe'
        engine._install_scene_query = lambda: (None, dict(backend='baseline_exact_scene_field'))
    proof = engine.prepare()
    if args.journal:
        replay_analog(args, runtime, engine)
        runtime.close()
        return
    seed = runtime.terrain_initial_history.clone()
    phases = [('forward', 180, (0., 1.2, 0.)), ('stop', 120, (0., 0., 0.)),
        ('strafe', 120, (1.2, 0., 0.)), ('reverse', 120, (0., -1.2, 0.)),
        ('final_stop', 60, (0., 0., 0.))]
    report = dict(checkpoint=runtime.catalog['checkpoint_sha256'], optimizer_updates=0,
        map=runtime.current_map, phases=phases, proof=proof, cases=[])
    for anchored in (False, True):
        for step in args.steps:
            history = seed.clone()
            motion_chunks, joint_chunks, offsets, timings, contexts, forecasts = [], [], [], [], [], []
            for name, frame_count, control in phases:
                remaining = frame_count
                while remaining:
                    count = min(step, remaining)
                    started = time.perf_counter()
                    result = engine.path(history, history.new_tensor([control]))
                    motion = result['motion'].clone()
                    delta = result['history'][:, 3, [0, 2]] - motion[:, 3, [0, 2]]
                    if anchored:
                        motion[..., [0, 2]] += delta[:, None]
                    geometry = engine.geometry(motion[:, 4:28].contiguous())
                    future = motion[:, 4:4+count].clone()
                    motion_chunks.append(future[0].cpu().numpy())
                    joint_chunks.append(geometry['joints'][:count].cpu().numpy())
                    offsets.append(delta[0].cpu().numpy())
                    timings.append((time.perf_counter()-started)*1000)
                    contexts.append(history[0].cpu().numpy())
                    forecasts.append(motion[0, :32].cpu().numpy())
                    history = torch.cat((history, future), 1)[:, -4:].contiguous()
                    remaining -= count
            motion, joints = np.concatenate(motion_chunks), np.concatenate(joint_chunks)
            tag = f"{'anchored' if anchored else 'original'}_{step}"
            np.savez_compressed(args.output / f'{tag}.npz', motion=motion, joints=joints,
                contexts=np.asarray(contexts), forecasts=np.asarray(forecasts), offsets=np.asarray(offsets))
            item = dict(name=tag, step=step, anchored=anchored,
                compute_p50_ms=float(np.median(timings)), compute_p95_ms=float(np.percentile(timings, 95)),
                context_offset_p95_m=float(np.percentile(np.linalg.norm(offsets, axis=-1), 95)),
                **metrics(motion, joints, phases, seed[0].cpu().numpy()))
            report['cases'].append(item)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--dual-proof', type=Path, required=True)
    parser.add_argument('--scene-proofs', type=Path, required=True)
    parser.add_argument('--map', default='urban')
    parser.add_argument('--flat-artifact', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--old-playback', type=Path)
    parser.add_argument('--longer', action='store_true')
    main(parser.parse_args())

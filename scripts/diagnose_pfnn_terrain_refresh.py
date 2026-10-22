"""Bounded, zero-update comparison of terrain observation age on PFNN003."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainBackend, PROFILE
from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
from scripts.train_native_mb import sha256


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


@torch.inference_mode()
def measure(runtime, world, rows):
    field = runtime.unified_controller.scene_field(runtime.terrain_scene)
    depths, frames, unknown, uncertain = [], [], 0, 0
    joints = runtime.inference.motion_rep.inverse(world, False, return_all=False)['posed_joints'][0]
    previous = None
    for start in range(3, world.shape[1], 8):
        part = world[:, start:start+8]
        pose = runtime.inference.motion_rep.inverse(part, False, return_all=False)
        body = runtime.skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
        original = body
        if previous is not None:body = torch.cat((previous, body), 0)
        previous = original[-1:]
        points = swept_surface_points(body, 8)
        if start+8 < world.shape[1]:points = points[:-1]
        gap, known = heightfield_gap(points, runtime.terrain_scene)
        inside = known & (gap < 0)
        depth = torch.zeros_like(gap)
        if inside.any():
            distance, _, valid = field.query(points[inside])
            depth[inside] = torch.where(valid, (-distance).clamp_min(0), -gap[inside])
            uncertain += int((~valid).sum())
        unknown += int((~known).sum())
        depths.append(depth[known].cpu().numpy())
        frames.extend(depth.max(-1).values.cpu().tolist())
    values = np.concatenate(depths)
    velocity = torch.diff(joints, dim=0).norm(dim=-1)*30
    acceleration = torch.diff(joints, n=2, dim=0).norm(dim=-1)*900
    boundaries = [row['first_frame'] for row in rows if row['neural_generation']]
    root = world[0, :, :3]
    return dict(body_prism_max_m=float(values.max()), body_prism_p99_m=float(np.quantile(values, .99)),
        fraction_over_2cm=float((values > .02).mean()), unknown_samples=unknown,
        uncertain_inside_samples=uncertain, full_vertices=runtime.vertex_count, swept_substeps=8,
        frame_peak_depths_m=frames, root_start=root[3].cpu().tolist(), root_end=root[-1].cpu().tolist(),
        travelled_xz_m=float(torch.diff(root[3:, [0, 2]], dim=0).norm(dim=-1).sum()),
        boundary_joint_speed_max_mps=max(float(velocity[index-1].max()) for index in boundaries),
        boundary_joint_acceleration_max_mps2=max(float(acceleration[index-2].max()) for index in boundaries))


@torch.inference_mode()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [('idle', 0., 0., 6), ('forward_walk', 1.6, 0., 9), ('forward_fast', 4., 0., 9),
             ('left_walk', 1.6, -math.pi/2, 9), ('left_fast', 4., -math.pi/2, 9)]
    save(args.output/'plan.json', dict(cases=cases, refresh_frames=[24, 8], selected_before_rollout=True,
        hypothesis='Fresher factual-history terrain queries reduce rapid-entry penetration.',
        original_profile=dict(PROFILE), alternative_scope='Offline only; not approved for the live artifact.',
        no_optimizer=True, no_ik=True, no_output_correction=True, all_failures_retained=True))
    runtime = FullCorpusAcceptanceRuntime(args.bundle, args.artifact, acceptance_test=True,
        execution='verified-cuda', spatial_index='bvh', reusable_scenes=True)
    before = tensor_digest(runtime.unified_controller.state_dict())
    results = []
    try:
        for name, speed, heading, blocks in cases:
            for refresh in (24, 8):
                torch.manual_seed(2333)
                backend = UnifiedTerrainBackend(runtime.unified_controller,
                    runtime.terrain_initial_history.clone(), runtime.terrain_scene)
                backend.configure_runtime(dict(PROFILE, refresh_frames=refresh))
                parts, rows = [backend.history.clone()], []
                failure = None
                for block in range(blocks):
                    started = time.perf_counter()
                    try:
                        result = backend.predict_plan([speed*math.sin(heading), speed*math.cos(heading)], heading)
                        motion = result['planned_motion']
                        parts.append(motion.clone())
                        backend.history = result['published_history'].clone()
                        rows.append(dict(block=block, first_frame=4+block*8,
                            neural_generation=backend.last_generation['neural_generation'],
                            observation_lag_frames=result['debug'].get('observation_lag_frames', 0),
                            wall_ms=(time.perf_counter()-started)*1000))
                    except Exception as error:
                        failure = repr(error)
                        break
                world = torch.cat(parts, 1)
                torch.save(dict(world=world.cpu(), commands=rows, source='generated_feedback',
                    refresh_frames=refresh), args.output/f'{name}-{refresh}.pt')
                metrics = measure(runtime, world, rows) if rows else None
                row = dict(case=name, requested_speed_mps=speed, refresh_frames=refresh,
                    frames=world.shape[1]-4, failure=failure, packets=rows, metrics=metrics)
                results.append(row)
                save(args.output/f'{name}-{refresh}.json', row)
                save(args.output/'progress.json', dict(completed=len(results),total=len(cases)*2))
                print(json.dumps(dict(case=name,refresh=refresh,failure=failure,
                    max_penetration_m=metrics['body_prism_max_m'] if metrics else None)),flush=True)
        after = tensor_digest(runtime.unified_controller.state_dict())
        if before != after:raise ValueError('Diagnostic changed model weights')
        save(args.output/'complete.json', dict(completed=True,rows=results,weights_unchanged=True,
            model_state_sha256=before,script_sha256=sha256(Path(__file__)),
            model_sha256=runtime.acceptance_receipt['checkpoint_sha256'],
            scene_sha256=runtime.acceptance_receipt['scene_sha256'],actual_optimizer_updates=0,
            quality_approved=False,delivery_approved=False))
    finally:runtime.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','artifact','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.is_dir():save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=0))
        raise

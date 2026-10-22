"""Read-only full flat counterfactuals and four travel-direction probes."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model, control, gait_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    rows = []
    for spec in args.runs:
        arm, filename = spec.split('=', 1)
        checkpoint = Path(filename)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        model, _ = load_model(saved)
        rep = model.inference.global_motion_rep
        directory = args.output / arm
        directory.mkdir()
        for case in cases:
            initial = case['world'][None, :4].cuda()
            request = control(case, 0).cuda()
            velocities = dict(forward=request[:, :2], reverse=-request[:, :2],
                              left=torch.stack((-request[:, 1], request[:, 0]), -1),
                              right=torch.stack((request[:, 1], -request[:, 0]), -1))
            directions = []
            for name, velocity in velocities.items():
                asked = torch.cat((velocity, request[:, 2:]), -1)
                intent = SoftCommandState().from_controls(initial, asked, 8)
                result = model(initial, intent, [case['scene']])['world']
                axis = velocity[0] / velocity[0].norm()
                first = result[0, 11, [0, 2]] - initial[0, 3, [0, 2]]
                forecast = result[0, 31, [0, 2]] - initial[0, 3, [0, 2]]
                directions.append(dict(request=name, requested_velocity_mps=velocity[0].tolist(),
                    published_displacement_xz_m=first.tolist(), forecast_displacement_xz_m=forecast.tolist(),
                    published_along_request_m=float(first @ axis), forecast_along_request_m=float(forecast @ axis)))
            gap, known = heightfield_gap(initial[0, 3:4, :3], case['scene'])
            if not known.all():
                raise ValueError('Factual initial floor is unknown')
            floor = float(initial[0, 3, 1] - gap[0])
            flat = dict(heightfield_y_m=torch.full((201, 201), floor), dx_m=torch.tensor(.5),
                        min_point_world_xyz_m=torch.tensor([float(initial[0, 3, 0]) - 50, 0., float(initial[0, 3, 2]) - 50]))
            trajectories = {}
            for terrain, scene in [('actual', case['scene']), ('flat', flat)]:
                history, state, parts, failure = initial.clone(), SoftCommandState(), [], None
                for block in range(case['training_segments']):
                    try:
                        result = model(history, state.from_controls(history, request, 8), [scene])['world']
                    except ValueError as error:
                        failure = str(error)
                        break
                    parts.append(result[:, 4:12])
                    history = reencode_played_history(result[:, 8:12], rep)
                metric = dict(frames=len(parts) * 8, failure=failure, ground_truth_future_used=False,
                              scene=terrain, postprocessing=None)
                if parts:
                    world = torch.cat(parts, 1)
                    metric.update(root_net_y_m=float(world[0, -1, 1] - initial[0, 3, 1]),
                                  root_y_range_m=float(world[0, :, 1].max() - world[0, :, 1].min()))
                    metric.update(gait_metrics(rep, world, request[0, :2] / request[0, :2].norm()))
                    save_motion(directory / (case['label'] + '-' + terrain + '.npz'), world, scene, rep, metric)
                trajectories[terrain] = metric
            row = dict(arm=arm, label=case['label'], entry=case['entry'], pilot_split=case['pilot_split'],
                checkpoint_sha256=sha256(checkpoint), directions=directions, trajectories=trajectories,
                interpretation='Identical factual initial body and request; subsequent histories independently generated. '
                               'Synthetic flat is diagnostic, never a fabricated source training target. '
                               'Direction forecasts are not a complete turn/control acceptance.')
            rows.append(row)
            write_json(args.output / 'progress.json', dict(rows=rows, optimizer_updates=0))
            print(json.dumps(row), flush=True)
        del model, saved
        torch.cuda.empty_cache()
    write_json(args.output / 'complete.json', dict(rows=rows, optimizer_updates=0,
        actor_outputs_modified=False, future_ground_truth_in_actor=False, quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    for name in ('cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())

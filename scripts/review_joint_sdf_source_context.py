"""Inspect existing long-route stair sources and unchanged actor rollouts."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState, demonstration_controls
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.joint_sdf_controller import JointSDFController
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    bundle_path = Path(saved['contract']['source'])
    if sha256(bundle_path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source/checkpoint identity mismatch')
    inference, bundle = load_terrain_bundle(bundle_path, 'cuda:0')
    model = JointSDFController(inference, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.eval()
    model.requires_grad_(False)
    dataset = Path(saved['contract']['dataset'])
    data = TerrainLearningDataset(dataset, build_native_representation(dataset / 'stats'), 'train')
    audit = json.loads(args.audit.read_text())
    indexed = {row['source_id']: index for index, row in enumerate(data.records)}
    rep = model.inference.global_motion_rep
    geometry = JointGeometryObjective(rep, 'cuda:0')
    results = []
    for source_id in args.sources:
        candidate = next(row for row in audit['rows'] if row['source_id'] == source_id)['candidates'][0]
        item = data.full_reference_item(indexed[source_id])
        begin, end = candidate['start'], candidate['available_end_exclusive']
        source = item['world'][None, begin:end].cuda()
        scene = item['scene']
        edge = candidate['first_riser'] - begin
        velocity = (source[:, edge - 8, [0, 2]] - source[:, 3, [0, 2]]) / ((edge - 11) / 30)
        yaw = torch.atan2(source[:, 3:edge - 8, 4].mean(1), source[:, 3:edge - 8, 3].mean(1))
        blocks = (len(source[0]) - 4) // 8
        source_played = source[:, 4:4 + blocks * 8]
        source_meta = dict(role='unchanged_reference', requested_frames=blocks * 8,
                           source_id=source_id, source_interval=[begin + 4, begin + 4 + blocks * 8],
                           candidate=candidate, quality_approved=False)
        source_metrics = geometry.full_surface_metrics(source_played, scene, source, source_id)
        source_meta.update(source_metrics)
        source_dir = args.output / 'source'
        source_dir.mkdir(exist_ok=True)
        save_motion(source_dir / (source_id + '-source.npz'), source_played, scene, rep, source_meta)
        results.append(dict(source_meta, mode='source'))
        for mode in ('constant_approach', 'source_velocity_schedule'):
            history = source[:, :4].clone()
            state = SoftCommandState()
            parts = []
            failure = None
            for block in range(blocks):
                if mode == 'constant_approach':
                    task = state.intent(history, velocity, yaw, 8)
                else:
                    # Offline weak planar labels, with only actually recorded
                    # endpoints; no future Y or pose is passed to the actor.
                    offset = block * 8
                    window = source[:, offset:min(offset + 33, source.shape[1])]
                    controls = demonstration_controls(window, window.shape[1] - 1)
                    task = state.from_controls(history, controls, 8)
                try:
                    generated = model(history, task, [scene])['world']
                except ValueError as error:
                    failure = str(error)
                    break
                parts.append(generated[:, 4:12])
                history = reencode_played_history(generated[:, 8:12], rep)
            row = dict(source_id=source_id, mode=mode, candidate=candidate, failure=failure,
                       requested_frames=blocks * 8, frames=sum(p.shape[1] for p in parts),
                       quality_approved=False, constant_command=mode == 'constant_approach',
                       command_velocity_mps=velocity[0].tolist())
            if parts:
                world = torch.cat(parts, 1)
                _, points = geometry.geometry(world)
                gap, known = heightfield_gap(points[0, :, :geometry.sole_count], scene)
                row.update(geometry.full_surface_metrics(world, scene, source, source_id))
                direction = velocity[0] / velocity[0].norm()
                delta = world[0, -1, :3] - source[0, 3, :3]
                reference = source[0, 3 + world.shape[1], :3] - source[0, 3, :3]
                row.update(sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0), .99)),
                           sole_vertical_max_m=float((-gap[known]).clamp_min(0).max()),
                           progress_m=float(delta[[0, 2]].dot(direction)),
                           source_progress_m=float(reference[[0, 2]].dot(direction)),
                           root_rise_m=float(delta[1]), source_root_rise_m=float(reference[1]))
                directory = args.output / mode
                directory.mkdir(exist_ok=True)
                save_motion(directory / (source_id + '.npz'), world, scene, rep, row)
            results.append(row)
            write_json(args.output / 'progress.json', {'rows': results, 'quality_approved': False})
            print(json.dumps(row), flush=True)
    write_json(args.output / 'complete.json', {'rows': results, 'quality_approved': False,
               'scope': 'Read-only candidate audit; source skin and unchanged SDF-only A. '
                        'Schedule is offline weak planar supervision, not held-WASD success.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

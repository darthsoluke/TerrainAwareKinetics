"""Prepare real stair passages with enough tail for every evaluated feedback step."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.lafan_soma_retarget import Y_TO_Z
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def recorded_direction(data, row, frames):
    if row.get('source_intent', {}).get('kind') != 'recorded_intent_sidecar':
        raise ValueError('Context candidate must retain its recorded direction sidecar')
    path = (data.base.dataset_dir / row['source_intent_sidecar']).resolve()
    if data.base.dataset_dir.resolve() not in path.parents:
        raise ValueError('Intent sidecar escapes the dataset')
    if sha256(path) != row['source_intent_sidecar_sha256']:
        raise ValueError('Intent SHA256 changed')
    with np.load(path, allow_pickle=False) as archive:
        values = archive['command_world_xy'].copy()
        indices = archive['source_frame_indices'].copy()
        fps = float(archive['fps'])
    if values.shape != (frames, 2) or indices.shape != (frames,) or fps != 30 or (np.diff(indices) <= 0).any():
        raise ValueError('Recorded direction/source clock mismatch')
    digest = hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()
    if row.get('source_frame_indices_sha256') and digest != row['source_frame_indices_sha256']:
        raise ValueError('Source frame-index hash changed')
    # The incremental PARC writer stores the source clock in contacts/intent,
    # not in the encoded motion shard; all three hashes bind the same row.
    contacts = data.base.dataset_dir / row['source_contact_sidecar']
    shard = data.base.dataset_dir / row['shard']
    if sha256(contacts) != row['source_contact_sidecar_sha256'] or sha256(shard) != row['converted_shard_sha256']:
        raise ValueError('Contact or encoded motion identity changed')
    with np.load(contacts, allow_pickle=False) as archive:
        if not np.array_equal(indices, archive['source_frame_indices']):
            raise ValueError('Recorded direction and contact source-frame indices differ')
    with np.load(shard, allow_pickle=False) as archive:
        if len(archive['global_features']) != frames or float(archive['fps']) != 30:
            raise ValueError('Encoded motion and sidecar lengths/fps differ')
    transformed = np.column_stack((values, np.zeros(frames))) @ Y_TO_Z
    result = torch.from_numpy(transformed[:, [0, 2]].astype(np.float32))
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(frames), atol=2e-5, rtol=0)
    return result


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rep = build_native_representation(args.dataset / 'stats').to('cuda:0')
    data = TerrainLearningDataset(args.dataset, rep, 'train')
    geometry = JointGeometryObjective(rep, 'cuda:0')
    audit = json.loads(args.audit.read_text())
    index = {row['source_id']: i for i, row in enumerate(data.records)}
    cases, reports = [], []
    for source_id in args.sources:
        selected = next(row for row in audit['rows'] if row['source_id'] == source_id)['candidates'][0]
        item = data.full_reference_item(index[source_id])
        begin = selected['start']
        goal_frame = selected['plateau_start'] + 15
        # Each published block is [offset+4, offset+11]; its original MB target
        # requires all 33 real frames, including the interval endpoint.
        last_offset = 8 * math.ceil(max(0, goal_frame - begin - 11) / 8)
        end = begin + last_offset + 33
        if end > len(item['world']):
            raise ValueError('Real continuation is missing; never pad or replay the tail')
        source = item['world'][begin:end].clone()
        scene = item['scene']
        gap, known = heightfield_gap(source[:, :3], scene)
        if not known.all():
            raise ValueError('Source continuation leaves measured terrain')
        edge = selected['first_riser'] - begin
        velocity = (source[edge - 8, [0, 2]] - source[3, [0, 2]]) / ((edge - 11) / 30)
        yaw = float(torch.atan2(source[3:edge - 8, 4].mean(), source[3:edge - 8, 3].mean()))
        recorded = recorded_direction(data, item['row'], len(item['world']))[begin:end]
        change = torch.rad2deg(torch.acos((recorded * recorded[3]).sum(-1).clamp(-1, 1)))
        outcome_velocity = (source[1:, [0, 2]] - source[:-1, [0, 2]]) * 30
        source_world = source[None].cuda()
        pose = rep.inverse(source_world, False, return_all=True, joint_positions_from='global_rot_data')
        skin = geometry.skin.skin(pose['local_joint_rots'][0], source_world[0, :, :3, None].transpose(-1, -2).expand(-1, 30, -1),
                                  rot_is_global=False)
        field = HeightfieldPrismSDF(scene)
        peak = 0.
        uncertain = 0
        fractions = []
        for offset in range(0, len(skin), 8):
            part = skin[offset:min(offset + 9, len(skin))]
            if len(part) < 2:
                continue
            points = swept_surface_points(part, 8)
            vertical, occupied = heightfield_gap(points, scene)
            if not occupied.all():
                raise ValueError('Full source skin leaves measured terrain')
            inside = vertical < 0
            penetration = torch.zeros_like(vertical)
            if inside.any():
                distance, _, valid = field.query(points[inside])
                uncertain += int((~valid).sum())
                penetration[inside] = torch.where(valid, (-distance).clamp_min(0), -vertical[inside])
            peak = max(peak, float(penetration.max()))
            fractions.append((int((penetration > .02).sum()), penetration.numel()))
        case = dict(item, world=source, name='stairs', label=source_id, start_frame=begin, edge_frame=edge,
                    command_velocity=velocity.tolist(), command_heading=yaw,
                    evaluation_goal_world_xz=source[goal_frame - begin, [0, 2]].tolist(),
                    evaluation_goal_source_frame=goal_frame,
                    training_segments=1 + last_offset // 8)
        cases.append(case)
        row = {'source_id': source_id, 'real_source_interval': [begin, end], 'frames': len(source),
               'previous_available_end_exclusive': selected['available_end_exclusive'],
               'additional_real_tail_frames': end - selected['available_end_exclusive'],
               'goal_source_frame': goal_frame, 'segments': case['training_segments'],
               'last_history_frames': [begin + last_offset, begin + last_offset + 3],
               'last_published_frames': [begin + last_offset + 4, begin + last_offset + 11],
               'full_context_exact_skin_peak_m': peak, 'uncertain_inside_samples': uncertain,
               'skin_samples_over2cm_fraction': sum(a for a, _ in fractions) / sum(n for _, n in fractions),
               'recorded_direction_norm': 'unit, not metres per second',
               'recorded_direction_change_max_deg': float(change.max()),
               'weak_requested_speed_mps': float(velocity.norm()),
               'outcome_speed_approach_median_mps': float(outcome_velocity[:edge - 8].norm(dim=-1).median()),
               'outcome_speed_ascent_median_mps': float(outcome_velocity[edge:selected['plateau_start'] - begin].norm(dim=-1).median()),
               'command_provenance': 'same constant weak cruise label inferred from the real flat approach; not recorded speed',
               'quality_approved': False}
        reports.append(row)
        save_motion(args.output / (source_id + '-source.npz'), source_world, scene, rep,
                    dict(row, role='unchanged_reference'))
    original = torch.load(args.original_cases, map_location='cpu', weights_only=False)
    for case in original[1:]:
        cases.append(dict(case, label=case['name'], training_segments=13))
    torch.save(cases, args.output / 'cases.pt')
    write_json(args.output / 'complete.json', {'rows': reports, 'cases_sha256': sha256(args.output / 'cases.pt'),
               'original_failed_staircase_retained_as_evaluation': str(args.original_cases),
               'training_launched': False, 'quality_approved': False,
               'planned_comparison': 'Same initialization, examples, source windows, commands and losses; '
                                     'source-history versus full generated-history feedback. No backbone '
                                     'expansion or loss reweighting. Review source visuals before adoption.'})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--original-cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

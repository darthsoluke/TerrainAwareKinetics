"""Compare volume and exact-prism penetration on identical recorded skin samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rep = build_native_representation(args.dataset / 'stats').to('cuda:0')
    data = TerrainLearningDataset(args.dataset, rep, 'train')
    geometry = JointGeometryObjective(rep, 'cuda:0')
    report = json.loads((args.review / 'complete.json').read_text())
    indexed = {row['source_id']: index for index, row in enumerate(data.records)}
    rows = []
    cached = {}
    for previous_row in report['rows']:
        source_id, mode = previous_row['source_id'], previous_row['mode']
        if source_id not in cached:
            item = data.full_reference_item(indexed[source_id])
            candidate = previous_row['candidate']
            source = item['world'][None, candidate['start']:candidate['available_end_exclusive']].cuda()
            scene = item['scene']
            cached[source_id] = (scene, HeightfieldPrismSDF(scene), geometry.volume(scene, source, source_id))
        scene, field, volume = cached[source_id]
        file = args.review / mode / (source_id + ('-source' if mode == 'source' else '') + '.npz')
        with np.load(file, allow_pickle=False) as archive:
            rotations = torch.from_numpy(archive['local_rot_mats'].copy()).cuda()
            roots = torch.from_numpy(archive['roots'].copy()).cuda()
        exact_values, approximate_values = [], []
        previous = None
        total = unknown = uncertain_inside = 0
        first_validation = True
        peak = {'penetration_m': 0., 'position_xyz_m': None}
        for start in range(0, len(roots), 16):
            count = len(roots[start:start + 16])
            points = geometry.skin.skin(rotations[start:start + 16],
                                        roots[start:start + 16, None].expand(-1, 30, -1), rot_is_global=False)
            if previous is not None:
                points = torch.cat((previous, points), 0)
            previous = points[-1:]
            swept = swept_surface_points(points, 8)
            if start + count < len(roots):
                swept = swept[:-1]
            gap, known = heightfield_gap(swept, scene)
            approximate, approximate_known = volume.query(swept)
            approximate = torch.where(approximate_known, approximate, gap)
            # Outside the measured solid, penetration is exactly zero. Only
            # interior samples need a nearest-air-prism search for its depth.
            interior = (gap < 0) & known
            penetration = torch.zeros_like(gap)
            selected = swept[interior]
            if len(selected):
                distance, _, certain = field.query(selected)
                penetration[interior] = torch.where(certain, (-distance).clamp_min(0), -gap[interior])
                uncertain_inside += int((~certain).sum())
                local_peak, index = penetration[interior].max(0)
                if float(local_peak) > peak['penetration_m']:
                    peak = {'penetration_m': float(local_peak), 'position_xyz_m': selected[int(index)].tolist(),
                            'certain': bool(certain[int(index)]), 'chunk_start_frame': start}
            if first_validation:
                sample = swept.reshape(-1, 3)[::997]
                full, _, valid = field.query(sample)
                expected = penetration.reshape(-1)[::997]
                torch.testing.assert_close((-full[valid]).clamp_min(0), expected[valid], atol=1e-6, rtol=0)
                first_validation = False
            exact_values.append(penetration[known].cpu().numpy())
            approximate_values.append((-approximate[known]).clamp_min(0).cpu().numpy())
            total += known.numel()
            unknown += int((~known).sum())
        exact = np.concatenate(exact_values)
        approximate = np.concatenate(approximate_values)
        approximate_p99 = float(np.quantile(approximate, .99))
        approximate_max = float(approximate.max())
        if abs(approximate_p99 - previous_row['body_penetration_p99_m']) > 1e-5 or abs(
                approximate_max - previous_row['body_penetration_max_m']) > 1e-5:
            raise ValueError('Identical recorded-volume measurement failed replay parity')
        row = {'source_id': source_id, 'mode': mode, 'frames': len(roots),
               'vertices_per_frame': len(geometry.skin.bind_vertices), 'swept_substeps': 8,
               'assessed_samples': len(exact), 'unknown_samples': unknown,
               'unknown_fraction': unknown / max(1, total), 'uncertain_inside_samples': uncertain_inside,
               'exact_prism_p99_m': float(np.quantile(exact, .99)), 'exact_prism_max_m': float(exact.max()),
               'volume_p99_m': approximate_p99, 'volume_max_m': approximate_max,
               'volume_false_positive_over_2cm_fraction': float(((approximate > .02) & (exact == 0)).mean()),
               'volume_missed_over_2cm_fraction': float(((exact > .02) & (approximate == 0)).mean()),
               'peak': peak, 'volume_replay_parity': True, 'quality_approved': False}
        rows.append(row)
        write_json(args.output / 'progress.json', {'rows': rows, 'quality_approved': False})
        print(json.dumps(row), flush=True)
    write_json(args.output / 'complete.json', {'rows': rows, 'quality_approved': False,
               'scope': 'Exact distances to the observed piecewise-constant heightfield solid, not the original '
                        'PARC triangle scene. Linear skin-point sweeps at eight substeps are sampled, not '
                        'continuous collision detection. Unknown nearest boundaries use reported fallback.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

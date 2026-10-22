"""Audit real earlier approach frames; no training, synthetic prefix or output correction."""

import argparse
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rep = build_native_representation(args.dataset / 'stats')
    dataset = TerrainLearningDataset(args.dataset, rep, 'train')
    by_id = {row['source_id']: i for i, row in enumerate(dataset.records)}
    geometry = JointGeometryObjective(rep, 'cpu')
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    results = []
    for case in cases:
        if case['native_split'] != 'train':
            continue
        source = dataset.full_reference_item(by_id[case['row']['source_id']])
        old_start = case['start_frame']
        retained = source['world'][old_start:old_start + len(case['world'])]
        difference = float((retained - case['world']).abs().max())
        torch.testing.assert_close(retained, case['world'], atol=0, rtol=0)
        new_start = max(0, old_start - 24)
        extended = source['world'][new_start:old_start + len(case['world'])].clone()
        prefix_frames = old_start - new_start
        gap, known = heightfield_gap(extended[:, :3], source['scene'])
        ground = extended[:, 1] - gap
        old_floor = ground[prefix_frames:prefix_frames + 4].mean()
        near_flat_prefix = bool((ground[:prefix_frames + 4] - old_floor).abs().max() <= .03)
        metrics = exact_skin_metrics(geometry, extended[None], source['scene'])
        eligible = bool(prefix_frames == 24 and known.all() and near_flat_prefix
            and metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0
            and metrics['uncertain_inside_samples'] == 0)
        result = dict(label=case['label'], old_start=old_start, new_start=new_start,
            added_real_prefix_frames=prefix_frames, original_overlap_max_error=difference,
            source_id=case['row']['source_id'], source_original_sha256=case['row']['source_original_sha256'],
            metrics=metrics, prefix_ground_range_m=float(ground[:prefix_frames + 4].max() - ground[:prefix_frames + 4].min()),
            prefix_matches_entry_floor=near_flat_prefix, eligible_for_flat_approach_extension=eligible,
            independent_sequences_added=0, synthetic_frames=0, training_started=False)
        results.append(result)
        write_json(args.output / 'progress.json', dict(rows=results))
    write_json(args.output / 'complete.json', dict(rows=results, optimizer_updates=0,
        source_cases_sha256=sha256(args.cases), dataset_manifest_sha256=sha256(args.dataset / 'native_manifest.json'),
        source_frames_modified=False, cases_replaced=False, quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cases', 'dataset', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise

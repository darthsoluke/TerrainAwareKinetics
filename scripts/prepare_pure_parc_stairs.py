"""Screen a small PARC-only stair pilot without altering recorded bodies or terrain."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.parc_stair_passages import regular_passages, validate_pure_cases
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import gait_metrics
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    expected = 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf'
    if sha256(args.checkpoint) != expected:
        raise ValueError('Preserve the approved orange parent')
    dataset = Path(saved['contract']['dataset'])
    rep = build_native_representation(dataset / 'stats')
    data = TerrainLearningDataset(dataset, rep, 'train')
    geometry = JointGeometryObjective(rep, 'cuda:0')
    audit, selected, used = [], [], set()
    counts = dict(up=0, down=0)
    select_passages = importlib.import_module(args.selector_module).regular_passages
    for index, row in enumerate(data.records):
        if not row.get('origin', '').startswith('parc') or row['source_id'] in used:
            continue
        item = data.full_reference_item(index)
        for entry in ('up', 'down'):
            if counts[entry] >= 3:
                continue
            for part in select_passages(item['world'], item['scene'], entry):
                world = item['world'][part['start']:part['end']].clone()
                metric = exact_skin_metrics(geometry, world[None].cuda(), item['scene'])
                eligible = metric['body_prism_max_m'] <= .02 and not metric['unknown_body_fraction'] and not metric['uncertain_inside_samples']
                report = dict(source_id=row['source_id'], origin=row['origin'], interval=part,
                              admitted=bool(eligible), metrics=metric)
                audit.append(report)
                write_json(args.output / 'progress.json', dict(scanned_index=index, source_audit=audit, counts=counts))
                if not eligible:
                    continue
                edge = part['edge'] - part['start']
                velocity = (world[edge - 8, [0, 2]] - world[3, [0, 2]]) / ((edge - 11) / 30)
                if velocity.norm() < .2:
                    report['admitted'] = False
                    report['reason'] = 'No meaningful continuous approach command'
                    continue
                yaw = torch.atan2(world[3:edge - 8, 4].mean(), world[3:edge - 8, 3].mean())
                control = torch.cat((velocity, yaw.cos()[None], yaw.sin()[None]))[None]
                axis = velocity.cuda() / velocity.norm().cuda()
                report['source_gait'] = gait_metrics(rep, world[None].cuda(), axis)
                case = dict(item, world=world, name='stairs', label=row['source_id'] + '-' + entry,
                            entry=entry, pilot_split='train' if counts[entry] < 2 else 'heldout',
                            start_frame=part['start'], edge_frame=edge,
                            evaluation_goal_source_frame=part['goal'],
                            training_segments=(len(world) - 33) // 8 + 1,
                            exact_constant_control=control, source_screen=metric)
                selected.append(case)
                counts[entry] += 1
                used.add(row['source_id'])
                print(json.dumps(dict(label=case['label'], split=case['pilot_split'], frames=len(world),
                                      source_max_m=metric['body_prism_max_m'])), flush=True)
                break
            if row['source_id'] in used:
                break
        if min(counts.values()) == 3:
            break
    write_json(args.output / 'source_screen.json', dict(rows=audit, counts=counts, source_frames_modified=False))
    validate_pure_cases(selected)
    torch.save(selected, args.output / 'cases.pt')
    write_json(args.output / 'complete.json', dict(schema='motionbricks.pure_parc_stairs.v1',
        cases_sha256=sha256(args.output / 'cases.pt'), parent_sha256=expected,
        source_bundle_sha256=saved['contract']['source_bundle_sha256'],
        dataset_manifest_sha256=sha256(dataset / 'native_manifest.json'),
        train_sources=[c['row']['source_id'] for c in selected if c['pilot_split'] == 'train'],
        heldout_sources=[c['row']['source_id'] for c in selected if c['pilot_split'] == 'heldout'],
        control_semantics='fixed weak approach speed and heading; not recorded WASD; no STOP labels',
        selector_module=args.selector_module,
        holdout_scope='different source IDs for this pilot; not unseen by inherited full-budget training',
        source_screen='18056 skin vertices, 8 substeps, exact negative prism distance, max<=2cm, no unknown/uncertain',
        source_frames_modified=False, quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--selector-module', default='motionbricks.data.parc_stair_passages')
    main(parser.parse_args())

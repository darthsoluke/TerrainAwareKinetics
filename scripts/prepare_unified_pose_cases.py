"""Select real complete stair entries and BONES clips without split leakage."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def stair_intervals(world, scene):
    gap, known = heightfield_gap(world[:, :3], scene)
    level = world[:, 1] - gap
    changes = torch.diff(level)
    result = []
    for sign, entry in ((1, 'up'), (-1, 'down')):
        edges = torch.nonzero((changes * sign > .05) & known[:-1] & known[1:]).flatten() + 1
        for first in edges.tolist():
            start = first - 40
            if start < 0 or float(level[start:first].max() - level[start:first].min()) > .03:
                continue
            for end in range(first + 8, min(first + 160, len(world) - 40)):
                count = int(((edges >= first) & (edges <= end)).sum())
                if count < 3 or float((level[end] - level[start]) * sign) < .3:
                    continue
                if float(level[end:end + 24].max() - level[end:end + 24].min()) > .03:
                    continue
                if (changes[first - 1:end] * sign < -.03).any():
                    continue
                length = 33 + 24 * math.ceil((end + 24 - start - 33) / 24)
                if start + length > len(world) or not known[start:start + length].all():
                    continue
                delta = world[start + length - 6, [0, 2]] - world[start + 3, [0, 2]]
                axis = delta / delta.norm().clamp_min(1e-6)
                side = torch.stack((axis[1], -axis[0]))
                lateral = float(((world[start:start + length, [0, 2]] - world[start + 3, [0, 2]]) @ side).abs().max())
                if lateral > .35:
                    continue
                result.append(dict(start=start, length=length, entry=entry, first_riser=first,
                                   plateau=end, steps=count, lateral_m=lateral))
                break
            if result and result[-1]['entry'] == entry:
                break
    return result


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rep = build_native_representation(args.parc / 'stats')
    datasets = [TerrainLearningDataset(path, rep, 'train') for path in [args.parc, *args.additional_parc]]
    bones = TerrainLearningDataset(args.bones, rep, 'train')
    geometry = JointGeometryObjective(rep, args.device)
    reviews = torch.load(args.review_cases, map_location='cpu', weights_only=False)
    reserved = {c['row']['source_group'] for c in reviews if c['native_split'] == 'review_only'}
    for extra in args.reserve_cases:
        for case in torch.load(extra, map_location='cpu', weights_only=False):
            if case['row']['origin'].startswith('parc'):
                reserved.add(case['row']['source_group'])
    selected, records, used_ids, used_groups = [], [], set(), set()
    counts = Counter()
    source_audit = {} if args.source_audit is None else {
        r['source_id']: r['ready_for_source_window_selection'] for r in json.loads(args.source_audit.read_text())['rows']}
    ordered = [(data, i) for data in datasets for i in sorted(range(len(data)),
               key=lambda i: (data.records[i].get('terrain_family') != 'stairs', data.records[i]['source_id']))]
    inspected_ids = set()
    for data, index in ordered:
        record = data.records[index]
        if not record['origin'].startswith('parc'):
            continue
        if record['source_id'] in inspected_ids:
            continue
        inspected_ids.add(record['source_id'])
        if record['source_id'] in source_audit and not source_audit[record['source_id']]:
            records.append(dict(source_id=record['source_id'], disposition='full_new_source_audit_rejected'))
            continue
        if record['source_group'] in reserved:
            records.append(dict(source_id=record['source_id'], disposition='reserved_review_family'))
            continue
        if record['source_group'] in used_groups:
            continue
        item = data.full_reference_item(index)
        intervals = stair_intervals(item['world'], item['scene'])
        for interval in intervals:
            if counts[interval['entry']] >= 6:
                continue
            start, length = interval['start'], interval['length']
            world = item['world'][start:start + length].clone()
            metrics = exact_skin_metrics(geometry, world[None].to(args.device), item['scene'])
            good = metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0 and metrics['uncertain_inside_samples'] == 0
            row = dict(source_id=record['source_id'], group=record['source_group'], split=record['split'],
                       interval=interval, metrics=metrics, disposition='selected' if good else 'source_skin_rejected')
            records.append(row)
            if good:
                count = (length - 9) // 24
                velocity = (world[-6, [0, 2]] - world[3, [0, 2]]) / ((length - 9) / 30)
                selected.append(dict(name='stairs', label=record['source_id'] + '-' + interval['entry'],
                    row=record, scene=item['scene'], world=world, start_frame=start, native_split='train',
                    entry=interval['entry'], native_segments=count, interval=interval,
                    exact_constant_control=torch.cat((velocity, world[3, 3:5]))[None], source_screen=metrics))
                counts[interval['entry']] += 1
                used_ids.add(record['source_id']); used_groups.add(record['source_group'])
                break
        write_json(args.output / 'progress.json', dict(counts=dict(counts), records=records))
        if counts['up'] == 6 and counts['down'] == 6:
            break
    flat_groups = set()
    for category in ('idle', 'walk', 'run', 'turn', 'start', 'walk'):
        matched = False
        candidates = [(i, r) for i, r in enumerate(bones.records) if r['origin'] == 'bones_seed'
                      and r['source_group'] not in flat_groups
                      and (r.get('control_category') == category or category in r.get('filename', '').lower())]
        for index, record in candidates:
            item = bones.full_reference_item(index)
            world = item['world'][:129].clone()
            if len(world) < 81:
                continue
            hf = item['scene']['heightfield_y_m']
            if not torch.isfinite(hf).all() or not hf.eq(hf[0, 0]).all():
                continue
            metrics = exact_skin_metrics(geometry, world[None].to(args.device), item['scene'])
            _, surface = geometry.geometry(world[None].to(args.device))
            gap = (surface[0, :, :geometry.sole_count, 1] - float(hf[0, 0])).abs().amin(-1)
            good = metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0 and metrics['uncertain_inside_samples'] == 0 and float(torch.quantile(gap, .95)) <= .05
            records.append(dict(source_id=record['source_id'], category=category, metrics=metrics,
                                disposition='selected' if good else 'flat_source_rejected'))
            if not good:
                continue
            velocity = (world[-6, [0, 2]] - world[3, [0, 2]]) / ((len(world) - 9) / 30)
            selected.append(dict(name=category, label=record['source_id'], row=record, world=world,
                scene=item['scene'], start_frame=0, native_split='train', native_segments=(len(world)-9)//24,
                exact_constant_control=torch.cat((velocity, world[3, 3:5]))[None], source_screen=metrics))
            flat_groups.add(record['source_group']); counts['bones'] += 1
            matched = True
            break
        if not matched:
            records.append(dict(category=category, disposition='category_shortfall'))
        write_json(args.output / 'progress.json', dict(counts=dict(counts), records=records))
    torch.save(selected, args.output / 'cases.pt')
    write_json(args.output / 'complete.json', dict(counts=dict(counts), records=records,
        reserved_groups=sorted(reserved), selected_ids=sorted(used_ids), cases_sha256=sha256(args.output / 'cases.pt'),
        parc_manifest_sha256=sha256(args.parc / 'native_manifest.json'),
        bones_manifest_sha256=sha256(args.bones / 'native_manifest.json'),
        additional_manifest_sha256={str(p): sha256(p / 'native_manifest.json') for p in args.additional_parc},
        source_audit_sha256=sha256(args.source_audit) if args.source_audit else None,
        requested=dict(up=6, down=6, bones=6), target_counts_met=dict(counts) == dict(up=6, down=6, bones=6),
        source_frames_modified=False, synthetic_frames=0, quality_approved=False))
    print(dict(counts), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parc', 'bones', 'review-cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--reserve-cases', type=Path, nargs='*', default=[])
    p.add_argument('--additional-parc', type=Path, nargs='*', default=[])
    p.add_argument('--source-audit', type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    main(args)

"""Find factual stair routes with real lead-in and post-ascent feedback context."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    data = TerrainLearningDataset(args.dataset, build_native_representation(args.dataset / 'stats'), 'train')
    rows = []
    for index, record in enumerate(data.records):
        if not record['origin'].startswith('parc'):
            continue
        item = data.full_reference_item(index)
        world = item['world']
        gap, known = heightfield_gap(world[:, :3], item['scene'])
        ground = world[:, 1] - gap
        rises = torch.nonzero((ground[1:] - ground[:-1] > .05) & known[1:] & known[:-1]).flatten() + 1
        candidates = []
        for first in rises.tolist():
            start = first - 20
            if start < 0 or not known[start:first].all():
                continue
            if float(ground[start:first].max() - ground[start:first].min()) > .03:
                continue
            delta = world[first - 8, [0, 2]] - world[start + 3, [0, 2]]
            if delta.norm() < .06:
                continue
            direction = delta / delta.norm()
            side = torch.stack((direction[1], -direction[0]))
            limit = min(len(world), start + 257)
            # A 16-frame measured plateau after at least two rises defines an
            # ascent end. This screen does not certify the skin or controller.
            for end in range(first + 8, limit - 32):
                if int(((rises >= first) & (rises <= end)).sum()) < 2:
                    continue
                if float(ground[end] - ground[start]) < .2:
                    continue
                if not known[start:end + 33].all():
                    continue
                if float(ground[end:end + 16].max() - ground[end:end + 16].min()) > .03:
                    continue
                lateral = (world[start:end + 33, [0, 2]] - world[start, [0, 2]]) @ side
                if float(lateral.abs().max()) > .35:
                    continue
                candidates.append({'start': start, 'first_riser': first, 'plateau_start': end,
                                   'available_end_exclusive': end + 33,
                                   'rise_m': float(ground[end] - ground[start]),
                                   'risers': int(((rises >= first) & (rises <= end)).sum()),
                                   'lateral_max_m': float(lateral.abs().max())})
                break
            if candidates:
                break
        rows.append({'source_id': record['source_id'], 'family': record.get('terrain_family'),
                     'frames': len(world), 'rising_edges': len(rises), 'candidates': candidates})
    report = {'scope': 'All training PARC full references; real data only, no augmentation or relabeling',
              'rows': rows, 'candidate_records': sum(bool(r['candidates']) for r in rows),
              'limitations': 'Root-support geometry screen only. Candidates still require full source skin '
                             'inspection and unchanged actor evaluation before a training experiment.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'records': len(rows), 'candidates': [r for r in rows if r['candidates']]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

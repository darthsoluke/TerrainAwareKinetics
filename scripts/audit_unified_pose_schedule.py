"""Compare actual archived exposure against independent per-source phase rotation."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scripts.pilot_unified_pose_capacity import build_schedule
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    archived = json.loads(a.schedule.read_text())
    legacy = build_schedule(cases, len(archived), 'global-step-phase-v1')
    if legacy != archived:
        raise ValueError('Cannot reproduce the actual archived schedule')
    fixed = build_schedule(cases, len(archived))
    if [[(d['case'], d['mask_seed']) for d in batch] for batch in fixed] != [[(d['case'], d['mask_seed']) for d in batch] for batch in archived]:
        raise ValueError('Window correction also changed source IDs or masking RNG')
    rows = []
    for index, case in enumerate(cases):
        thirds = np.array_split(np.arange(len(case['world']) - 32), 3)
        window_phase = {int(start): phase for phase, values in enumerate(thirds) for start in values}
        row = dict(label=case['label'], entry=case.get('entry', 'flat'), real_frames=len(case['world']))
        for name, schedule in (('archived', archived), ('corrected', fixed)):
            starts = [d['start'] for batch in schedule for d in batch if d['case'] == index]
            phases = Counter(window_phase[start] for start in starts)
            coverage = sorted({frame for start in starts for frame in range(start + 4, start + 32)})
            interval = case.get('interval')
            first = None if interval is None else interval['first_riser'] - interval['start']
            row[name] = dict(exposures=len(starts), phase_counts=[phases[p] for p in range(3)],
                starts=sorted(set(starts)), supervised_source_frames=coverage,
                first_riser_frame=first, first_riser_exposures=None if first is None else sum(start + 4 <= first < start + 32 for start in starts))
        if min(row['corrected']['phase_counts']) == 0:
            raise ValueError('A source still lacks phase coverage')
        rows.append(row)
    write_json(a.output / 'corrected_schedule.json', fixed)
    write_json(a.output / 'complete.json', dict(rows=rows, actual_schedule_reproduced=True,
        source_identity_order_equal=True, mask_seed_order_equal=True,
        cases_sha256=sha256(a.cases), archived_schedule_sha256=sha256(a.schedule),
        updates=len(archived), primitive_exposures=sum(map(len, archived)), optimizer_updates=0,
        source_frames_modified=False, limitation='Real-time thirds are coverage strata, not annotated gait phases. '
        'This proves a sampling coverage defect, not that correcting it solves closed-loop stair motion.'))
    print(json.dumps([dict(label=r['label'], before=r['archived']['phase_counts'], after=r['corrected']['phase_counts']) for r in rows]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cases', 'schedule', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())

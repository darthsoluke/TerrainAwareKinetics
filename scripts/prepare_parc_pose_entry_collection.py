"""Prepare eight new source geometries; dispatch remains explicitly bounded."""
import argparse
import json
from pathlib import Path

import build_parc_stair_course as base
from build_parc_pose_entry_course import build_pose_entry_course
from run_parc_rugged_collection import digest, write

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
base.PARC_REPO = a.base / 'parc_runtime_v1/wide_v1/out/vendor/PARC'
jobs = []
for index in range(8):
    _, nodes, course = build_pose_entry_course(index)
    jobs.append(dict(layout=index, family='stairs', split='train', seed=2026101480 + index,
        terrain_sha256=course['terrain_sha256'], route_nodes=len(nodes), course=course, candidates=4))
if len({row['terrain_sha256'] for row in jobs}) != 8:
    raise ValueError('Duplicate source geometries')
protected = a.base / 'model_terrain_v3_20261011/parc_rugged32_20261013_v2/unified_pilot_20261014_v1/candidate.pt'
if digest(protected) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
    raise ValueError('Preserved controller identity changed')
write(a.output / 'plan.json', dict(schema='motionbricks.parc_pose_entry32.v1', jobs=jobs,
    total_candidates=32, initial_dispatch_candidates=4, approved_model=str(protected),
    approved_model_sha256=digest(protected), training_started=False, postprocessing=None,
    split_unit='whole geometry, every candidate from it remains train; existing review families excluded',
    note='Only layout0is initially dispatched. Its result must be reviewed before the remaining7layouts.'))
print(json.dumps(dict(prepared_layouts=8, possible_candidates=32, initial_dispatch=4)))

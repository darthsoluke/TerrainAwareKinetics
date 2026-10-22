"""Finish the declared32-source collection; never start model training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.build_parc_only_dataset import build_dataset
from scripts.prepare_parc_only_catalog import bind_geometry_components
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256, verify_dataset


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((a.collection / 'plan.json').read_text())
    plan_hash = sha256(a.collection / 'plan.json')
    while True:
        workers = [json.loads((a.collection / f'worker_{gpu}.json').read_text()) for gpu in a.worker_gpus]
        assignment = [layout for w in workers for layout in w['layouts']]
        if sorted(assignment) != list(range(8)):
            raise ValueError('Workers must cover all eight layouts exactly once')
        if any(w['plan_sha256'] != plan_hash for w in workers):
            raise ValueError('Source collection plan changed')
        if any(w['status'] == 'failed' for w in workers):
            raise ValueError('Source collection failed; retain all partial outputs')
        write_json(a.output / 'state.json', dict(status='waiting_for_collection', model_training_started=False,
            workers=[dict(gpu=w['gpu'], status=w['status'], jobs=[dict(layout=j['layout'], status=j['status']) for j in w['jobs']]) for w in workers]))
        if all(w['status'] == 'collection_complete' for w in workers):
            break
        time.sleep(15)
    rows, inventory = [], []
    for job in plan['jobs']:
        folder = a.collection / f'layout_{job["layout"]:02d}'
        generated = json.loads((folder / 'generation/generation.json').read_text())
        if generated['seed'] != job['seed'] or len(generated['candidates']) != 4:
            raise ValueError('Generation identity or candidate count changed')
        inventory.extend(dict(layout=job['layout'], candidate_id=r['candidate_id'], frames=r['frames'],
            route_covered=r['geometric_route_covered'], sha256=r['sha256']) for r in generated['candidates'])
        catalog = folder / 'admission/catalog.json'
        if catalog.exists():
            rows.extend(json.loads(catalog.read_text())['rows'])
    if len(inventory) != 32:
        raise ValueError('Preserve every one of the32declared candidates')
    write_json(a.output / 'candidate_inventory.json', dict(rows=inventory, failures_preserved=True))
    catalog = a.output / 'catalog.json'
    write_json(catalog, dict(schema='parc.only_teacher_catalog.v1', rows=rows,
        geometry_split_binding=bind_geometry_components(rows), quality_approved=False))
    if sha256(plan['approved_model']) != plan['approved_model_sha256']:
        raise ValueError('Preserved live model changed')
    write_json(a.output / 'state.json', dict(status='building_native', model_training_started=False))
    dataset = a.output / 'parc_native'
    manifest = build_dataset(catalog, dataset, incremental_base=a.anchor)
    verify_dataset(dataset, manifest)
    write_json(a.output / 'complete.json', dict(status='native_dataset_ready_exact_audit_pending',
        model_training_started=False, optimizer_updates=0, candidate_count=32,
        admitted=sum(r['disposition'] == 'admitted' for r in rows), catalog_sha256=sha256(catalog),
        native_manifest_sha256=sha256(dataset / 'native_manifest.json'), original_frames_preserved=True,
        scene_speed_bound=None, note='Source speed/time is not clipped or retimed. Full exact skin/route audit still required.'))
    write_json(a.output / 'state.json', dict(status='native_dataset_ready_exact_audit_pending', model_training_started=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('collection', 'anchor', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--worker-gpus', type=int, nargs='+', default=[0, 2])
    a = p.parse_args()
    try:
        main(a)
    except Exception as error:
        if a.output.exists():
            write_json(a.output / 'failure.json', dict(error=repr(error), model_training_started=False))
        raise

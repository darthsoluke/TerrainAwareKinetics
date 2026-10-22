"""Bind full PARC physics outputs to SOMA preflight; never approve quality."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(export, output):
    manifest_path = export / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'exported_all_reported_physics_recordings':
        raise ValueError('Physical export is incomplete')
    selected, excluded = [], []
    for row in manifest['recordings']:
        if not row['whole_motion'] or row['start_fraction'] != 0.:
            excluded.append({'candidate_key': row['candidate_key'], 'reason': 'partial_recording'})
            continue
        source = (export / row['converted_path']).resolve()
        source.relative_to(export.resolve())
        if digest(source) != row['converted_sha256'] or row['recorded_terrain_error_m'] > 1e-5:
            raise ValueError('Physical source/terrain pairing changed')
        item = {'behavior': row['candidate_key'], 'source_group': row['source_group'],
                'source': {'local_file': str(source), 'sha256': digest(source)},
                'recorded_state': 'simulated', 'physics_manifest_sha256': digest(manifest_path),
                'training_approved': False, 'intent_status': row.get('intent_status', 'unavailable')}
        if 'intent_path' in row:
            intent = (export / row['intent_path']).resolve()
            intent.relative_to(export.resolve())
            if digest(intent) != row['intent_sha256']:
                raise ValueError('Intent changed')
            item.update(intent_path=str(intent), intent_sha256=digest(intent))
        selected.append(item)
    if not selected:
        raise ValueError('No full recordings for retarget preflight')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump({'selected_items': selected, 'excluded': excluded, 'training_approved': False}, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.export.resolve(), args.output.resolve())

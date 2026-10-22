"""Bind official PARC members and measured generated exports to SOMA retargets."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def official_rows(audits):
    rows, seen = [], set()
    for audit_path in audits:
        audit_path = Path(audit_path).resolve()
        audit = load(audit_path)
        revision = audit.get('revision')
        archive_hash = audit.get('archive', {}).get('lfs', {}).get('oid')
        if not isinstance(revision, str) or len(revision) != 40 or not archive_hash:
            raise ValueError('Official audit must bind the HF revision and archive LFS SHA256')
        for sample in audit.get('samples', []):
            source = Path(sample['local_file']).resolve()
            if digest(source) != sample['sha256']:
                raise ValueError(f'Official PARC member changed: {source}')
            if sample['sha256'] in seen:
                continue
            seen.add(sample['sha256'])
            member = sample['member']
            base = sample.get('base_id', 'official:' + Path(member).stem.replace('_flipped', '').replace('_dm', ''))
            group = sample.get('source_group', base)
            source_id = 'official_' + sample['sha256'][:16]
            rows.append({'source_id': source_id, 'origin': 'parc_official',
                'base_id': base, 'source_group': group,
                'terrain_family': sample.get('terrain_family', Path(member).parent.as_posix()), 'terrain_seed': group,
                'split': 'train', 'upstream_identity': f'mxucg/PARC@{revision}:{member}',
                'upstream_manifest_path': str(audit_path), 'upstream_manifest_sha256': digest(audit_path),
                'source_path': str(source), 'source_sha256': sample['sha256'],
                'official_revision': revision, 'official_archive_sha256': archive_hash,
                'official_member': member})
    return rows


def generated_recording_identity(report_sha256, candidate_key):
    return f'PARC-physics-recording:{report_sha256}:{candidate_key}'


def generated_rows(exports):
    rows = []
    for directory in exports:
        directory = Path(directory).resolve()
        manifest_path = directory / 'manifest.json'
        manifest = load(manifest_path)
        if manifest.get('status') != 'exported_all_reported_physics_recordings':
            raise ValueError('Generated sources must be official converted physics recordings')
        report = directory.parent / 'physics/report.json'
        if not report.is_file() or digest(report) != manifest['run_report_sha256']:
            raise ValueError('Generated export does not bind its physics report')
        for item in manifest['recordings']:
            source = (directory / item['converted_path']).resolve()
            if directory not in source.parents or digest(source) != item['converted_sha256']:
                raise ValueError('Generated converted source changed or escaped export directory')
            key = item['candidate_key']
            source_id = 'generated_' + hashlib.sha256((str(directory) + ':' + key).encode()).hexdigest()[:16]
            group = item.get('source_group', str(directory.parent) + ':' + key)
            rows.append({'source_id': source_id, 'origin': 'parc_generated',
                'base_id': 'generated:' + group, 'source_group': 'generated:' + group,
                'terrain_family': group, 'terrain_seed': group, 'split': 'train',
                'upstream_identity': generated_recording_identity(manifest['run_report_sha256'],key),
                'upstream_manifest_path': str(manifest_path), 'upstream_manifest_sha256': digest(manifest_path),
                'source_path': str(source), 'source_sha256': item['converted_sha256'],
                'physics_report_path': str(report), 'physics_report_sha256': digest(report),
                'export_manifest_path': str(manifest_path), 'export_manifest_sha256': digest(manifest_path),
                'physics_candidate_key': key,
                'whole_motion': item.get('whole_motion') is True,
                'intent_path': str(directory / item['intent_path']) if item.get('intent_path') else None,
                'intent_sha256': item.get('intent_sha256')})
    return rows


def group_splits(rows):
    """Group-stratified holdouts selected before training, without predicted scores."""
    groups = {}
    for row in rows:
        if row.get('disposition') == 'excluded':
            continue
        family = row.get('official_member', '').split('/')
        family = family[1] if row['origin'] == 'parc_official' and len(family) > 1 else 'generated'
        group = row['source_group']
        if group in groups and family not in groups[group] and row.get('geometry_component') != group:
            raise ValueError('Source group belongs to multiple release families')
        groups.setdefault(group, set()).add(family)
    # Components sharing generated scenes stay in train, including matching official rows.
    families = {group: ('generated' if 'generated' in values else '|'.join(sorted(values)))
                for group, values in groups.items()}
    official_groups = [group for group, family in families.items() if family != 'generated']
    if len(official_groups) < 3:
        raise ValueError('Need three usable official groups; cannot fabricate held-out clips')
    owners = {group: 'train' for group in groups}
    strata = sorted(set(families.values()) - {'generated'})
    for family in strata:
        available = sorted([group for group, value in families.items() if value == family],
            key=lambda group: hashlib.sha256(('parc-group-split-v1:' + group).encode()).hexdigest())
        if len(available) >= 3:
            count = max(1, math.floor(len(available) * .1))
            owners.update({group: 'validation' for group in available[:count]})
            owners.update({group: 'test' for group in available[count:2 * count]})
    # Small engineering cohorts may lack a three-group family; retain real holdouts.
    ordered = sorted(official_groups,
        key=lambda group: hashlib.sha256(('parc-group-split-v1:' + group).encode()).hexdigest())
    for split in ('validation', 'test'):
        if split not in owners.values():
            group = next(group for group in ordered if owners[group] == 'train')
            owners[group] = split
    return owners


def bind_geometry_components(rows):
    """Keep exact local heightfield copies and source variants in one split."""
    admitted = [row for row in rows if row.get('disposition') == 'admitted']
    parents = {row['source_group']: row['source_group'] for row in admitted}

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    geometry_owner = {}
    for row in admitted:
        with np.load(row['retarget_path'], allow_pickle=False) as shard:
            hf = np.asarray(shard['hf'], dtype='<f4')
            dx = np.asarray(shard['dx'], dtype='<f8')
        if hf.ndim != 2 or dx.size != 1 or not np.isfinite(hf).all() or not np.isfinite(dx).all():
            raise ValueError('Invalid heightfield geometry during split binding')
        digest_value = hashlib.sha256()
        for array in (np.asarray(hf.shape, dtype='<i8'), hf, dx):
            digest_value.update(array.tobytes(order='C'))
        identity = digest_value.hexdigest()
        row['terrain_content_sha256'] = identity
        group = row['source_group']
        if identity in geometry_owner:
            first, second = find(geometry_owner[identity]), find(group)
            parents[max(first, second)] = min(first, second)
        else:
            geometry_owner[identity] = group
    components = {}
    for group in parents:
        components.setdefault(find(group), []).append(group)
    identifiers = {owner: 'parc-geometry:' + hashlib.sha256(
        '\0'.join(sorted(members)).encode()).hexdigest() for owner, members in components.items()}
    for row in admitted:
        old_group = row['source_group']
        row['original_source_group'] = old_group
        row['geometry_component'] = identifiers[find(old_group)]
        row['source_group'] = row['geometry_component']
        row['terrain_seed'] = row['geometry_component']
    return {'admitted_rows': len(admitted), 'exact_heightfields': len(geometry_owner),
            'source_groups_before': len(parents), 'split_components_after': len(components),
            'definition': 'exact float32 heightfield shape/values and float64 cell spacing; horizontal origin ignored for translation-invariant local observations; not a similarity or unseen-terrain certificate'}


def prepare(audits, exports, output, *, reuse_retarget=None, prepare_retargets=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = official_rows(audits) + generated_rows(exports)
    if not rows:
        raise ValueError('No PARC sources')
    # Hold out whole source groups, never random frames or augmented variants.
    groups = sorted({row['source_group'] for row in rows})
    if len(groups) < 4:
        raise ValueError('Need at least four independent groups for train/validation/test')
    official_groups = sorted({row['source_group'] for row in rows if row['origin'] == 'parc_official'})
    if len(official_groups) < 3:
        raise ValueError('Need at least three independent official groups')
    split_owner = group_splits(rows)
    for row in rows:
        row['split'] = split_owner.get(row['source_group'], 'train')
    existing = {}
    for directory in reuse_retarget or []:
        for item in load(Path(directory) / 'report.json')['items']:
            if item.get('execution_status') == 'processed':
                existing[item['source']['sha256']] = item
    selected = []
    for row in rows:
        if row.get('whole_motion') is False:
            row.update(disposition='excluded', exclusion_reason='partial physics recording')
            row.pop('split', None)
            continue
        old = existing.get(row['source_sha256'])
        if old:
            shard = Path(old['shard']).resolve()
            report = shard.parent / 'report.json'
            row.update(retarget_path=str(shard), retarget_sha256=old['shard_sha256'],
                report_path=str(report), report_sha256=digest(report))
        else:
            selected.append({'behavior': row['source_id'], 'source_group': row['source_group'],
                'source': {'local_file': row['source_path'], 'sha256': row['source_sha256']}})
    save(output / 'selection.json', {'selected_items': selected})
    save(output / 'catalog.pending.json', {'schema': 'parc.only_teacher_catalog.v1', 'rows': rows})
    if selected and prepare_retargets:
        completed = subprocess.run([sys.executable, '-u', '-m', 'scripts.prepare_parc_soma',
            '--selection', str(output / 'selection.json'), '--output', str(output / 'retarget')],
            cwd=ROOT, check=False)
        if completed.returncode and not (output / 'retarget/report.json').is_file():
            raise RuntimeError('Retarget process failed without a per-source report')
    prepared_path = output / 'retarget/report.json'
    if prepared_path.is_file():
        prepared = {row['behavior']: row for row in load(prepared_path)['items']}
        for row in rows:
            if 'retarget_path' in row or row.get('disposition') == 'excluded':
                continue
            item = prepared.get(row['source_id'], {})
            if item.get('execution_status') == 'processed':
                shard = Path(item['shard']).resolve()
                report = shard.parent / 'report.json'
                row.update(retarget_path=str(shard), retarget_sha256=item['shard_sha256'],
                    report_path=str(report), report_sha256=digest(report))
            else:
                row.update(disposition='excluded', exclusion_reason='retarget did not process: ' + str(item.get('error', item.get('execution_status'))))
                row.pop('split', None)
    for row in rows:
        if row.get('disposition') == 'excluded':
            continue
        if 'retarget_path' not in row:
            row.update(disposition='excluded', exclusion_reason='retarget not yet available')
            row.pop('split', None)
        elif load(row['report_path']).get('screen_status') != 'pass':
            row.update(disposition='excluded', exclusion_reason='SOMA geometry/continuity screen failed')
            row.pop('split', None)
        else:
            report = load(row['report_path'])
            if report.get('frames', 0) < 65:
                row.update(disposition='excluded', exclusion_reason='shorter than native 65-frame training contract')
                row.pop('split', None)
            else:
                row['disposition'] = 'admitted'
    geometry_binding = bind_geometry_components(rows)
    final_owner = group_splits(rows)
    for row in rows:
        if row['disposition'] == 'admitted':
            row['split'] = final_owner.get(row['source_group'], 'train')
    catalog = {'schema': 'parc.only_teacher_catalog.v1', 'rows': rows,
        'geometry_split_binding': geometry_binding,
        'split_policy': 'exact heightfield copies and source variants form components; SHA256 ranked whole components within release-family strata, approximately 80/10/10; components containing generated rows kept in train; two real official holdouts; assigned before model training',
        'purpose': 'paired geometric teacher for native MB training; not a dynamics certificate',
        'quality_approved': False}
    save(output / 'catalog.json', catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-audit', type=Path, nargs='+', required=True)
    parser.add_argument('--generated-export', type=Path, nargs='+', required=True)
    parser.add_argument('--reuse-retarget', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-retargets', action='store_true')
    args = parser.parse_args()
    result = prepare(args.official_audit, args.generated_export, args.output,
        reuse_retarget=args.reuse_retarget, prepare_retargets=args.prepare_retargets)
    print(json.dumps({'rows': len(result['rows']), 'admitted': sum(row['disposition'] == 'admitted'
        for row in result['rows']), 'catalog': str(args.output / 'catalog.json')}))


if __name__ == '__main__':
    main()

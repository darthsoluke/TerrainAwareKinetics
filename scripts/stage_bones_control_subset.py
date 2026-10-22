"""Stage existing native BONES clips for mixed-data control adaptation.

No motion is synthesized, no split reassigned, and no plane is fitted here.
The original fixed-plane geometry still needs region-based contact validation.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select(records, per_category=24):
    selected, counts = [], Counter()
    for row in sorted(records, key=lambda item: hashlib.sha256(item['filename'].encode()).hexdigest()):
        name = row['filename'].lower()
        category = ('idle' if row['behavior'] == 'idle' else 'stop' if '_stop_' in name
                    else 'start' if '_start_' in name else row['behavior'])
        key = (row['split'], category)
        cap = per_category if row['split'] == 'train' else per_category // 2
        if counts[key] < cap:
            selected.append(dict(row, control_category=category))
            counts[key] += 1
    if not all(counts['train', name] for name in ('idle', 'start', 'stop')):
        raise ValueError('The source has no train idle/start/stop coverage')
    return selected, {f'{split}/{category}': count for (split, category), count in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-records', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new staging directory')
    accepted = args.dataset / 'accepted.json'
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    if digest(accepted) != manifest['accepted_json_sha256']:
        raise ValueError('Source accepted manifest changed')
    records = json.loads(accepted.read_text())
    selected, counts = select(records, per_category=len(records) if args.all_records else 24)
    total = 0
    for row in selected:
        source = (args.dataset / row['shard']).resolve()
        source.relative_to(args.dataset.resolve())
        if digest(source) != row['converted_shard_sha256']:
            raise ValueError(f"Source shard changed: {row['shard']}")
        total += source.stat().st_size
    if total > 2 * 1024**3:
        raise ValueError('The bounded first control subset exceeds 2 GiB')
    args.output.mkdir(parents=True)
    report = {'status': 'staged_not_training_admitted', 'source': str(args.dataset),
              'source_manifest_sha256': digest(args.dataset / 'native_manifest.json'),
              'motion_format': 'existing native SOMA30 30 Hz global features',
              'counts': counts, 'uncompressed_shard_bytes': total,
              'split_policy': 'original recording/augmentation family assignments retained',
              'geometry_status': 'legacy fixed plane; region contact validation required', 'records': selected}
    selection = args.output / 'selection.json'
    selection.write_text(json.dumps(report, indent=2), encoding='utf-8')
    archive = args.output / 'bones-control-subset.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(selection, arcname='selection.json')
        for row in selected:
            bundle.add(args.dataset / row['shard'], arcname=row['shard'])
        bundle.add(args.dataset / 'native_manifest.json', arcname='source_native_manifest.json')
        bundle.add(accepted, arcname='source_accepted.json')
    receipt = {'counts': counts, 'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive),
               'status': report['status'], 'public_network_transfer': False}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

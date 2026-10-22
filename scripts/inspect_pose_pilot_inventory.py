"""Read only metadata for selecting a split-preserving Pose pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch


def main(args):
    for path in args.manifests:
        data = json.loads(path.read_text())
        print('MANIFEST', path, 'KEYS', list(data) if isinstance(data, dict) else 'list')
        rows = data if isinstance(data, list) else next((data[k] for k in ('records', 'items', 'clips', 'rows') if isinstance(data.get(k), list)), [])
        print('COUNTS', dict(Counter((r.get('split', '?'), r.get('origin', '?'),
                                     r.get('terrain_family', r.get('control_category', '?'))) for r in rows)))
        if rows:
            print('EXAMPLE', json.dumps(rows[0]))
    for path in args.cases:
        rows = torch.load(path, map_location='cpu', weights_only=False)
        print('CASES', path, 'COUNT', len(rows))
        print(json.dumps([{k: c.get(k) for k in ('label', 'name', 'entry', 'native_split', 'start_frame')}
                          | {'source_id': c['row'].get('source_id'), 'group': c['row'].get('source_group'),
                             'split': c['row'].get('split'), 'origin': c['row'].get('origin'),
                             'frames': len(c['world'])} for c in rows]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifests', type=Path, nargs='*', default=[])
    p.add_argument('--cases', type=Path, nargs='*', default=[])
    main(p.parse_args())

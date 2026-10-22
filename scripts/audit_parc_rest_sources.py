"""Find real rest intervals in the pinned PARC release, without inventing labels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np

from motionbricks.data.parc_paired_source import _decode_section, _restricted_load
from scripts.prepare_parc_release_subset import OFFICIAL_FAMILIES, verify_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identity = verify_archive(args.archive)
    counts, candidates, errors = Counter(), [], []
    with zipfile.ZipFile(args.archive) as archive:
        names = [name for name in archive.namelist() if name.endswith('.pkl')
                 and any('/' + family + '/' in '/' + name for family in OFFICIAL_FAMILIES)]
        for index, name in enumerate(names):
            try:
                source = _restricted_load(archive.read(name))
                motion = _decode_section(source, 'motion_data')
                root = np.asarray(motion['root_pos'])
                contacts = np.asarray(motion['body_contacts'])
                fps = float(motion['fps'])
                window = round(2 * fps) + 1
                stride = max(1, round(fps / 4))
                counts['sources'] += 1
                best = None
                for start in range(1, len(root) - window + 1, stride):
                    part = root[start:start+window]
                    excursion = float(np.linalg.norm(part-part[0], axis=-1).max())
                    speed = np.linalg.norm(np.diff(part, axis=0), axis=-1) * fps
                    both_feet = (contacts[start:start+window, [11, 14]] > .5).all(-1)
                    counts['windows'] += 1
                    if excursion > .05 or np.percentile(speed, 95) > .15 or both_feet.mean() < .9:
                        continue
                    counts['rest_candidates'] += 1
                    row = {'member': name, 'start': start, 'frames': window, 'fps': fps,
                           'root_excursion_m': excursion, 'root_speed_p95_mps': float(np.percentile(speed, 95)),
                           'both_feet_source_contact_fraction': float(both_feet.mean())}
                    if best is None or excursion < best['root_excursion_m']:
                        best = row
                if best:
                    candidates.append(best)
            except (ValueError, KeyError, TypeError) as error:
                errors.append({'member': name, 'error': str(error)})
            if index % 500 == 0:
                print(json.dumps({'processed': index, 'total': len(names), **counts}), flush=True)
    report = {'archive': identity, 'counts': dict(counts), 'errors': errors,
              'candidate_sources': sorted(candidates, key=lambda row: row['root_excursion_m']),
              'status': 'candidate_search_only_not_training_admission',
              'next_required': 'inherited source/geometry split, SOMA retarget and sole geometry validation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'counts': counts, 'candidate_sources': len(candidates), 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    main()

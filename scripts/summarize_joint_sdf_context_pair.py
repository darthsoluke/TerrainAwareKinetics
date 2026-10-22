"""Compare matched full-context pilots without changing any quality criteria."""

import argparse
import json
from pathlib import Path

import numpy as np


def assert_same(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Matched report keys differ')
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('Matched report lengths differ')
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, float):
        if not np.isclose(a, b, atol=1e-5, rtol=1e-5):
            raise ValueError('Matched baseline numeric values differ')
    elif a != b:
        raise ValueError('Matched baseline metadata differs')


def main(args):
    reports = {}
    for arm in ('source', 'feedback'):
        path = args.root / ('context_' + arm + '_v1')
        reports[arm] = json.loads((path / 'complete.json').read_text())
        if reports[arm]['actual_updates'] != 300 or reports[arm]['primitive_examples'] != 1200:
            raise ValueError('Unexpected pilot budget')
    assert_same(reports['source']['baseline'], reports['feedback']['baseline'])
    assert_same(reports['source']['exposure'], reports['feedback']['exposure'])
    if reports['source']['schedule_sha256'] != reports['feedback']['schedule_sha256']:
        raise ValueError('Source windows/cursors/commands were not identical')
    with np.load(args.root / 'context_source_v1/initial-forward.npz', allow_pickle=False) as a:
        with np.load(args.root / 'context_feedback_v1/initial-forward.npz', allow_pickle=False) as b:
            difference = float(np.abs(a['world'] - b['world']).max())
            np.testing.assert_allclose(a['world'], b['world'], atol=1e-5, rtol=1e-5)
    rows = []
    for arm, report in reports.items():
        for before, after in zip(report['baseline'], report['final']):
            if (before['name'], before['variant']) != (after['name'], after['variant']):
                raise ValueError('Evaluation ordering changed')
            rows.append({'arm': arm, 'case': after['name'], 'variant': after['variant'],
                         'baseline_passed': before['passed'], 'final_passed': after['passed'],
                         'goal_reached': after['goal_reached'], 'failure': after['failure'],
                         **{key: {'before': before.get(key), 'after': after.get(key)} for key in
                            ('sole_vertical_p99_m', 'body_prism_p99_m', 'body_prism_max_m',
                             'cross_track_m', 'root_height_error_m', 'root_drift_m', 'frames')}})
    official = {}
    for arm in reports:
        report = json.loads((args.root / ('context_' + arm + '_v1/official_start_review/complete.json')).read_text())
        official[arm] = report['rows']
    result = {'initial_forward_max_difference': difference, 'baseline_parity': True,
              'source_window_command_and_exposure_parity': True, 'quality_approved': False,
              'summary': {arm: {'new_passed': sum(row['passed'] for row in report['final']),
                                'new_cases': len(report['final']),
                                'new_stair_passed': sum(row['passed'] for row in report['final'] if row['variant'] != 'flat'),
                                'new_stair_goal_reached': sum(row['goal_reached'] for row in report['final'] if row['variant'] != 'flat'),
                                'original_stair_passed': sum(row['passed'] for row in official[arm]),
                                'original_stair_cases': len(official[arm])} for arm, report in reports.items()},
              'rows': rows, 'official_stair_rows': official,
              'limits': 'New-case pass also requires full-skin peak <=2cm. Original retained screen is '
                        'unchanged and reports its sampled-body peak separately. Neither count alone '
                        'establishes broad generalization or live WASD quality.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'official_stair_rows')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

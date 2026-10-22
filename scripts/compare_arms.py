"""Report completed endpoints without hardcoded baseline measurements."""

import argparse
import json
from pathlib import Path


def read_arms(root, names):
    rows = {}
    for name in names:
        directory = root / ('arm_' + name)
        payload = {key: json.loads((directory / filename).read_text()) for key, filename in
                   (('tf', 'tf_strat.json'), ('cl', 'cl_strat.json'), ('result', 'results.json'))}
        payload['cl'] = payload['cl']['summary']
        result = payload['result']
        if (result['updates'] != result['plan']['updates']
                or result['window_exposures'] != result['updates'] * result['plan']['batch']):
            raise ValueError('Incomplete or inconsistent training endpoint: ' + name)
        rows[name] = payload
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['A', 'B', 'C'])
    parser.add_argument('--fixed-review', type=Path, help='Source-mask report including its persistence baseline')
    args = parser.parse_args()
    rows = read_arms(args.root, args.arms)
    print('Original 17-frame audit: self-predicted support masks, median of window quantiles')
    print('arm stance_cm swing_cm body_cm selfmask_speed_cmps scored_points params validation_loss_initial_final')
    for name, row in rows.items():
        tf, cl, result = row['tf'], row['cl'], row['result']
        print(f"{name} {tf['feet_stance_p50_m']*100:.3f} {tf['feet_swing_p50_m']*100:.3f} "
              f"{tf['body_error_p50_m']*100:.3f} {cl['rollout_stance_median']*100:.3f} "
              f"{cl['stance_frames']} {result['trainable_parameters']} "
              f"{result['initial']['total']:.6f}->{result['final']['total']:.6f}")
    if args.fixed_review is not None:
        report = json.loads(args.fixed_review.read_text())
        print('\nFixed source-support intervals; 14 valid-command frames; pooled quantiles')
        print('arm stance_cm swing_cm body_cm source_mask_speed_cmps scored_points')
        baseline = report['persistence']
        print('persistence ' + ' '.join(f"{baseline[k]['p50']*100:.3f}" for k in
                                      ('stance_error', 'swing_error', 'body_error')))
        for name in args.arms:
            arm = report['arms'][name]
            tf, cl = arm['modes']['teacher']['summary'], arm['modes']['closed']['summary']
            print(name, *(f"{tf[k]['p50']*100:.3f}" for k in ('stance_error', 'swing_error', 'body_error')),
                  f"{cl['source_mask_speed']['p50']*100:.3f}", cl['source_mask_speed']['count'])
        print('Original windows:', report['original_windows'], 'families:', len(report['families']))


if __name__ == '__main__':
    main()

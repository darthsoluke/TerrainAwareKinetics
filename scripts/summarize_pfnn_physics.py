"""Final verdict table: reference distribution against rollout conditions.

Two thresholds make the comparison interpretable. Differences below the null W1
(same condition, split halves) are not detectable; differences below the
contact-shift calibration are smaller than a two-frame label error.
"""
import argparse, json, pathlib
import numpy as np
from scipy.stats import wasserstein_distance


def stats(values):
    return dict(n=int(values.size),
                p10=100 * float(np.percentile(values, 10)),
                p50=100 * float(np.median(values)),
                p90=100 * float(np.percentile(values, 90)),
                mean=100 * float(values.mean()),
                over5=100 * float((values > 0.05).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--reference-key', default='0')
    parser.add_argument('--conditions', nargs='*', default=[])
    parser.add_argument('--null-w1', type=float, default=3.30)
    parser.add_argument('--shift-w1', type=float, default=7.46)
    args = parser.parse_args()
    ref = np.asarray(np.load(args.reference)[args.reference_key], dtype=float)
    rows = [('reference', ref, 0.0)]
    for item in args.conditions:
        label, _, path = item.partition('=')
        path, _, wanted = path.partition('::')
        archive = np.load(path)
        if wanted:
            key_name = wanted
        else:
            key_name = 'rollout' if 'rollout' in archive.files else archive.files[0]
        values = np.asarray(archive[key_name], dtype=float)
        rows.append((label, values, 100 * wasserstein_distance(ref, values)))
    print(f"{'condition':<14}{'n':>7}{'p10':>8}{'p50':>8}{'p90':>8}{'mean':>8}{'W1':>9}  verdict")
    for label, values, distance in rows:
        s = stats(values)
        if label == 'reference':
            verdict = 'baseline'
        elif distance < args.null_w1:
            verdict = 'not detectable'
        elif distance < args.shift_w1:
            verdict = 'detectable, below label jitter'
        else:
            verdict = 'comparable to 2-frame label error'
        print(f"{label:<14}{s['n']:>7}{s['p10']:>8.2f}{s['p50']:>8.2f}{s['p90']:>8.2f}"
              f"{s['mean']:>8.2f}{distance:>9.3f}  {verdict}")
    print()
    print(f'thresholds: null W1 < {args.null_w1} cm/s not detectable; '
          f'< {args.shift_w1} cm/s smaller than a two-frame contact-label error')


if __name__ == '__main__':
    main()

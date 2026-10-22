"""Compare stance-slip distributions between conditions.

Wasserstein-1 on the pooled sample, in cm/s, plus quantile gaps. Distances are
only interpretable against a calibrated scale, so this script also reports the
known-strength contact-shift conditions when they are present in the archive.
"""
import argparse, pathlib
import numpy as np
from scipy.stats import wasserstein_distance


def describe(values):
    if values.size == 0:
        return dict(n=0)
    return dict(n=int(values.size),
                p10=float(np.percentile(values, 10)),
                p50=float(np.median(values)),
                p90=float(np.percentile(values, 90)),
                mean=float(values.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--baseline', default='0')
    args = parser.parse_args()
    data = np.load(args.archive, allow_pickle=True)
    names = list(data.files)
    print('conditions:', names)
    print()
    for name in names:
        values = np.asarray(data[name], dtype=float)
        d = describe(values)
        if d['n'] == 0:
            print(f'{name:>10}  empty')
            continue
        print(f"{name:>10}  n={d['n']:5d}  p10={d['p10']*100:6.2f}  p50={d['p50']*100:6.2f}  "
              f"p90={d['p90']*100:7.2f}  mean={d['mean']*100:6.2f}  cm/s")
    if args.baseline not in names:
        return
    base = np.asarray(data[args.baseline], dtype=float)
    print()
    print(f'Wasserstein-1 against {args.baseline} (cm/s):')
    for name in names:
        if name == args.baseline:
            continue
        other = np.asarray(data[name], dtype=float)
        if other.size == 0 or base.size == 0:
            continue
        print(f'  {name:>10}  {100 * wasserstein_distance(base, other):8.3f}')


if __name__ == '__main__':
    main()

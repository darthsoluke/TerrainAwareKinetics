"""Plot paired gait diagnostics; oracle results are never deployment results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    data = json.loads(args.report.read_text(encoding='utf-8-sig'))
    run = 'unified_approach_fit_20261014_v1'
    live = [r for r in data['live_rows'] if r['run'] == run]
    source = [r for r in data['source_rows'] if r['run'] == run]
    if len(live) != 5 or len(source) != 12 or data['optimizer_updates'] != 0:
        raise ValueError('Expected the complete paired read-only gait diagnostic')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    knee = [r['vs_actual']['same_root_flat_environment']['knee_flexion_mae_deg'] for r in live]
    axes[0].bar(np.arange(len(live)), knee, color='#3576a8', width=.6)
    axes[0].set_xticks(np.arange(len(live)), [r['frame'] for r in live])
    axes[0].set(title='Fixed Root: change only the environment', xlabel='Live neural planning frame',
                ylabel='Mean knee-flexion difference (degrees)', ylim=(0, 1.4))
    for i, value in enumerate(knee):
        axes[0].text(i, value + .035, f'{value:.2f}', ha='center')
    keys = ('actual_error', 'height_only_oracle_error', 'oracle_error',
            'source_tokens_oracle_error', 'source_root_and_tokens_oracle_error')
    labels = ('Actor', 'Source\nheight only', 'Source\nfull Root', 'Source\ntokens', 'Source Root\n+ tokens')
    values = [100 * np.mean([r[k]['leg_relative_mpjpe_m'] for r in source]) for k in keys]
    axes[1].bar(np.arange(len(keys)), values, color=['#b5513f', '#7e858b', '#7e858b', '#378370', '#378370'], width=.65)
    axes[1].set_xticks(np.arange(len(keys)), labels)
    axes[1].set(title='12 real-source windows: offline interventions',
                ylabel='Mean Root-relative leg-joint error (cm)', ylim=(0, 10))
    for i, value in enumerate(values):
        axes[1].text(i, value + .15, f'{value:.2f}', ha='center')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('The orange candidate does not reliably turn terrain information into stair footwork', fontsize=13)
    fig.supxlabel('Different diagnostic contexts across panels. Source truth is offline only; no output correction or new training.', fontsize=9)
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

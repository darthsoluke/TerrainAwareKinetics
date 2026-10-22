"""Summarize completed bounded Pose experiments without implying quality passes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    files = [('A: adapter', 'unified_pose_adapter_20261014_v1-report.json'),
             ('B: Pose capacity', 'unified_pose_capacity_20261014_v1-report.json'),
             ('C: 25% full mask', 'unified_pose_mask_20261014_v2-report.json')]
    reports = [(name, json.loads((a.directory / filename).read_text(encoding='utf-8-sig'))) for name, filename in files]
    labels = ['Before', *[name for name, _ in reports]]
    rows = [reports[0][1]['baseline']['live_stairs'], *[r['final']['live_stairs'] for _, r in reports]]
    peaks = np.array([[r['metrics']['body_prism_max_m'] * 100 for r in values] for values in rows])
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    x = np.arange(4)
    for i, (speed, color) in enumerate([('.8 m/s request', '#267b94'), ('1.6 m/s request', '#bc5945')]):
        ax.bar(x + (i - .5) * .35, peaks[:, i], width=.35, color=color, label=speed)
        for j, value in enumerate(peaks[:, i]):
            ax.text(x[j] + (i - .5) * .35, value + .7, f'{value:.1f}', ha='center', fontsize=10)
    ax.axhline(2, color='#444', linestyle='--', linewidth=1, label='Existing 2 cm peak screen')
    ax.set_xticks(x, labels)
    ax.set(ylabel='Maximum full-skin penetration (cm)', ylim=(0, 57),
           title='Pose training completed; all live ascent candidates still fail geometry')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.10), ncol=3, fontsize=9, frameon=False)
    fig.supxlabel('Generated pace is unrestricted. All runs reach the goal; peak metrics use 18,056 vertices and 8 substeps.', fontsize=9)
    for ext in ('.png', '.pdf'):
        fig.savefig(a.output.with_suffix(ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())

"""Plot the recorded, FK-reencoded SDF intervention audit without motion edits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    if report.get('live_history') != 'official FK reencoded, matching publication':
        raise ValueError('Use the runtime-matched history audit')
    rows = [r for r in report['rows'] if r['scope'] == 'fixed_live_history'
            and r['checkpoint'] == 'unified_pilot_20261014_v1']
    frames = [r['frame'] for r in rows]
    x = np.arange(len(rows))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(x, [r['sdf_plane_delta_max_m'] for r in rows], 'o-', color='#326899')
    axes[0].set(title='SDF sees non-flat geometry', ylabel='Maximum distance difference (m)')
    axes[1].plot(x, [r['memory_plane_delta_rms'] for r in rows], 'o-', color='#217c66')
    axes[1].set(title='Encoded features also change', ylabel='Memory difference RMS (model units)')
    sdf = [r['root_y_sdf_effect_max_m'] * 1000 for r in rows]
    all_terrain = [r['root_y_all_terrain_effect_max_m'] * 1000 for r in rows]
    axes[2].bar(x - .18, sdf, width=.36, label='New SDF only', color='#326899')
    axes[2].bar(x + .18, all_terrain, width=.36, label='All terrain inputs', color='#b85336')
    axes[2].set(title='Root reacts weakly to new SDF', ylabel='Root Y effect (mm, log scale)', yscale='log')
    axes[2].legend(loc='upper left', fontsize=8)
    for ax in axes:
        ax.set_xticks(x, frames)
        ax.set_xlabel('Logical playback frame')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Current 28941: fixed actual history and command; environment interventions only', fontsize=13)
    fig.supxlabel('Counterfactuals are discarded. Nonlinear effects are not additive; this is not a motion-quality pass.', fontsize=9)
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

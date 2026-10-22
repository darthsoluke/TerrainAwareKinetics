"""Plot the read-only native terrain diagnostic reports."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    report = json.loads((args.run/'complete.json').read_text(encoding='utf-8-sig'))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), layout='constrained')
    fig.suptitle('Native terrain model: coverage and inference diagnostics', fontsize=17)
    coverage = report['coverage']
    groups = coverage['groups']
    values = [[row['speed_mps'] for row in coverage['rows'] if row['label'] == group['label']] for group in groups]
    names = ['up', 'down', 'idle', 'walk crop', 'jog', 'turn', 'side start', 'back walk']
    ax = axes[0, 0]
    boxes = ax.boxplot(values, patch_artist=True, widths=.55)
    for index, box in enumerate(boxes['boxes']):
        box.set_facecolor('#d78850' if index < 2 else '#5ba5a1')
    rng = np.random.default_rng(17121)
    for index, points in enumerate(values):
        ax.scatter(index+1+rng.uniform(-.12, .12, len(points)), points, s=15, color='#30393c', alpha=.65, zorder=3)
    ax.set_xticks(np.arange(1, 9), names, rotation=20, ha='right')
    ax.set_ylabel('Terminal command speed (m/s)')
    ax.set_title('A. Exact 88 training windows; 8 source clips', loc='left')
    ax.set_ylim(-.1, 4.1)
    ax.grid(axis='y', alpha=.2)

    rows = [row for row in report['conditions']['rows'] if row['entry'] in ('up', 'down')]
    ax = axes[0, 1]
    positions = np.arange(len(rows))
    labels = [('U' if row['entry'] == 'up' else 'D')+str(row['start']) for row in rows]
    ax.plot(positions, [100*row['geometry_comparison']['planner_height_max_difference_m'] for row in rows],
        marker='o', ms=4, color='#c87539', label='Root planner')
    ax.plot(positions, [100*row['geometry_comparison']['decoded_height_max_difference_m'] for row in rows],
        marker='s', ms=4, color='#237f7b', label='Final decoded Root')
    ax.set_xticks(positions, labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Maximum height difference (cm)')
    ax.set_title('B. Same history / command: stairs vs plane', loc='left')
    ax.axvline(7.5, color='#aaa', ls=':', lw=1)
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    arms = ['legacy_pose_off', 'volume_pose_off', 'both_pose_off']
    counts = [sum(row['pose_ablations'][arm]['changed_tokens'] for row in rows) for arm in arms]
    total = sum(row['pose_ablations'][arms[0]]['total_tokens'] for row in rows)
    percent = np.array(counts)/total*100
    bars = ax.bar(np.arange(3), percent, color=['#c87539', '#237f7b', '#707a7d'], width=.55)
    for bar, count, value in zip(bars, counts, percent):
        ax.text(bar.get_x()+bar.get_width()/2, value+.25, f'{count}/{total}\n({value:.2f}%)', ha='center', fontsize=10)
    ax.set_xticks(np.arange(3), ['Legacy Pose off', 'Volume Pose off', 'Both Pose off'])
    ax.set_ylabel('Changed categorical tokens (%)')
    ax.set_ylim(0, max(percent)+3)
    ax.set_title('C. Pose-only ablation; predicted Root unchanged', loc='left')
    ax.grid(axis='y', alpha=.2)

    ax = axes[1, 1]
    routes = report['routes']['rows']
    positions = np.arange(len(routes))
    labels, counters = [], {'up': 0, 'down': 0}
    for row in routes:
        entry = row['entry']
        labels.append(('U' if entry == 'up' else 'D')+str(counters[entry])+('*' if row['latest_bank_trained_source'] else ''))
        counters[entry] += 1
    for index, (mode, name, color) in enumerate([('source_start', 'Source start', '#c87539'), ('idle_start', 'Idle start', '#237f7b')]):
        peak = [100*next(value for value in row['rollouts'] if value['mode'] == mode)['skin']['body_prism_max_m'] for row in routes]
        ax.bar(positions+(index-.5)*.36, peak, width=.36, label=name, color=color)
    ax.axhline(2, color='#9c3b40', ls='--', lw=1, label='2 cm geometry gate')
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Full-skin peak penetration (cm)')
    ax.set_title('D. Continuous constant-command routes: 0/20 pass', loc='left')
    ax.set_xlabel('* Source included in the latest 88-window bank')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.run/'diagnostics.png', dpi=160)
    fig.savefig(args.run/'diagnostics.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())

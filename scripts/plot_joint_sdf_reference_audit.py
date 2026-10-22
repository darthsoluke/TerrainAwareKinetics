"""Plot offline target correspondence, explicitly separate from actor quality."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    report = json.loads(args.report.read_text())
    rows = [row for row in report['rows'] if row['arm'] == 'feedback'
            and row['source_id'] == 'generated_8030ae7a2a7ad640']
    if len(rows) != 9:
        raise ValueError('Expected all nine factual feedback blocks')
    x = [row['feedback_position'] for row in rows]
    colors = {'fixed': '#2459a6', 'nearest_pose': '#b44831', 'capped_pose': '#167a68'}
    labels = {'fixed': 'Fixed source time', 'nearest_pose': 'Nearest full-pose history',
              'capped_pose': 'Monotonic, advance <=8 frames'}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.8), sharex=True, layout='constrained')
    fig.suptitle('Reference matching does not yet provide feasible recovery targets', fontsize=14)
    for name in colors:
        axes[0].plot(x, [row['matches'][name]['reference_start'] for row in rows], marker='o',
                     markersize=4, color=colors[name], label=labels[name])
        axes[1].plot(x, [100 * row['matches'][name]['next_target_along_command_m'] for row in rows],
                     marker='o', markersize=4, color=colors[name])
    axes[0].set_ylabel('Selected reference window start (frame)')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    axes[0].set_title('Unchanged actor rollout: feedback checkpoint, route 8030ae7a2a7ad640', fontsize=10)
    axes[1].axhline(0, color='#555555', linewidth=.8)
    axes[1].set_ylabel('Next source target along forward input (cm)')
    axes[1].set_xlabel('Generated feedback block')
    axes[1].set_xticks(x)
    final_target = 100 * rows[-1]['matches']['fixed']['next_target_along_command_m']
    axes[1].annotate(f'Fixed-time target: {-final_target:.1f} cm behind', xy=(x[-1], final_target),
                     xytext=(3.1, -56), fontsize=9,
                     arrowprops={'arrowstyle': '->', 'color': colors['fixed']}, color=colors['fixed'])
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Offline label analysis only. No retiming, correction, training or deployment of the actor.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

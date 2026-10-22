"""Plot the reported original and generated gait at their unchanged30Hz clocks."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    report = json.loads(args.report.read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
    colors = {'source': '#225ca7', 'generated': '#b34b30'}
    tail = {}
    for label in colors:
        series = report['series'][label]
        times = np.arange(len(series['root_y_m'])) / 30
        foreaft = np.asarray(series['foot_foreaft_relative_m'])
        knees = np.asarray(series['knee_flexion_deg'])
        separation = foreaft[:, 0] - foreaft[:, 1]
        axes[0].plot(times, series['forward_speed_mps'], color=colors[label], label=label)
        axes[1].plot(times, 100 * separation, color=colors[label])
        tail[label] = {'last_frames': 32, 'foot_foreaft_separation_rms_m': float(np.sqrt(np.mean(separation[-32:] ** 2))),
                       'close_foreaft_feet_under10cm_fraction': float(np.mean(np.abs(separation[-32:]) < .1)),
                       'mean_knee_flexion_deg': float(knees[-32:].mean())}
    seam_times = np.asarray([row['first_played_frame'] for row in report['blocks']]) / 30
    offsets = 100 * np.asarray([row['past_offset_along_command_m'] for row in report['blocks']])
    axes[2].bar(seam_times, offsets, width=.035, color='#7951a6', label='Past-velocity offset carried into first new frame')
    for axis in axes:
        for t in seam_times:
            axis.axvline(t, color='#777777', linestyle=':', linewidth=.7, alpha=.35)
        axis.axhline(0, color='#888888', linewidth=.6)
        axis.grid(axis='y', alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, ncols=2)
    axes[0].set_ylabel('Forward Root speed (m/s)')
    axes[1].set_ylabel('Left minus right foot\nfore-aft position (cm)')
    axes[2].set_ylabel('Past integration offset (cm)')
    axes[2].set_xlabel('Original recording time (seconds)')
    axes[2].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Reported close-leg hopping: gait timing and eight-frame boundaries', fontsize=13)
    fig.supxlabel('Vertical dotted lines are model replans. No output correction or retiming is applied.', fontsize=9)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({'tail32_recording_metrics': tail,
        'scope': 'Same last32 recorded frames, not a spatially matched gait phase or contact-force estimate.'}, indent=2))
    print(json.dumps(tail), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

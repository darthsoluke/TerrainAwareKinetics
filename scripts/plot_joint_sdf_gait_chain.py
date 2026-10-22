"""Plot factual gait measurements and explicitly offline module interventions."""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.report, encoding='utf-8') as stream:
    case = json.load(stream)['results'][0]
generated = case['modes']['generated_history']['variants']
teacher = case['modes']['source_history']['variants']
items = [
    ('Source playback', case['source'], '#3469ab'),
    ('Continuous actor', generated['actor']['gait'], '#ba463b'),
    ('Offline: source history', teacher['actor']['gait'], '#46864b'),
    ('Offline: source tokens', generated['source_tokens']['gait'], '#8052a1'),
    ('Offline: source Root', generated['source_root']['gait'], '#ad7626'),
]
fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained', gridspec_kw={'height_ratios': [1.3, 1]})
for label, gait, color in items[:3]:
    feet = np.asarray(gait['foreaft_m'])
    axes[0].plot(np.arange(len(feet)) / 30, 100 * (feet[:, 0] - feet[:, 1]), label=label, color=color)
axes[0].axhspan(-10, 10, color='#777777', alpha=.08)
axes[0].axvspan(48 / 30, 80 / 30, color='#777777', alpha=.08)
axes[0].axhline(0, color='#999999', linewidth=.5)
axes[0].set(xlabel='Time (s)', ylabel='Left minus right foot, fore-aft (cm)',
            title='The alternating gait shrinks under continuous generated history')
axes[0].legend(frameon=False, fontsize=9, ncols=3)
percentages = [100 * values['last32']['close_feet_fraction'] for _, values, _ in items]
axes[1].barh([label for label, _, _ in items], percentages, color=[color for _, _, color in items], height=.6)
for row, value in enumerate(percentages):
    axes[1].text(value + 1, row, f'{value:.1f}%', va='center', fontsize=10)
axes[1].invert_yaxis()
axes[1].set(xlim=(0, 112), xlabel='Last32 frames with fore-aft foot separation <10cm (%)',
            title='Keep actor history fixed: source tokens reduce collapse; source Root alone does not')
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('Reported stair clip: exact replay and offline causal probes', fontsize=14)
fig.supxlabel('Offline arms are independent windows, not deployed controllers. Close feet is not a flight/contact or quality label.', fontsize=9)
fig.savefig(args.output, dpi=170)
fig.savefig(args.output.rsplit('.', 1)[0] + '.pdf')

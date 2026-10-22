"""Visualize all raw candidate outcomes without hiding failed stop intervals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    report=json.loads(args.report.read_text())
    rows=report['rows'];events=[s for row in rows for s in row['stops']]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    x=np.arange(8)
    traversed=[sum(r['complete_route'] for r in rows if r['layout']==i) for i in x]
    axes[0].bar(x,traversed,color='#368478',label='Complete route')
    axes[0].bar(x,8-np.asarray(traversed),bottom=traversed,color='#cbd0d3',label='Route incomplete')
    for i,n in enumerate(traversed):axes[0].text(i,n+.12,str(n),ha='center',fontsize=10)
    axes[0].set(xticks=x,ylim=(0,9),xlabel='Independent stair layout',ylabel='Candidates (8 per layout)',
        title=f"{report['complete_routes']}/64 complete routes; 0 complete stop/resume passes")
    axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=9)
    axes[1].axvspan(0,.15,color='#368478',alpha=.08)
    axes[1].axvline(.15,color='#368478',linestyle='--',linewidth=1)
    axes[1].axhline(.08,color='#368478',linestyle='--',linewidth=1)
    for inside,color,marker,label in [(False,'#8b9297','x','Outside requested stair flight'),
                                      (True,'#c77736','o','Inside requested stair flight')]:
        chosen=[s for s in events if s['stopped_inside_flight']==inside]
        axes[1].scatter([s['final_half_second_speed_p90_mps'] for s in chosen],
            [s['final_half_second_drift_m'] for s in chosen],s=25,alpha=.72,c=color,marker=marker,label=label)
    axes[1].set(xlim=(0,None),ylim=(0,None),xlabel='Final 0.5 s root speed P90 (m/s)',
        ylabel='Final 0.5 s planar drift (m)',title=f"All {len(events)} recorded stop requests")
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('PARC continuous stairs: native command-response screen',fontsize=15)
    fig.savefig(args.output,dpi=160)
    fig.savefig(args.output.with_suffix('.pdf'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

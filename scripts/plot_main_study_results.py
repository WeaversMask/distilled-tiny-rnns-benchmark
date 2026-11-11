"""Plot the frozen primary contrast from a completed saved main-study summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.summary.resolve().parent):
        raise ValueError('Plot output must be outside the input results folder')
    summary=json.loads(args.summary.read_text())
    if summary['status']!='complete' or summary['pending_populations']:
        raise ValueError('Every planned population must complete before plotting')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    kinds=[('parametric_model_free','Model-free','#2563a6'),('reward_cue','Reward cue','#a5479a')]
    for y,(kind,label,color) in enumerate(kinds):
        rows=summary['kinds'][kind]['per_seed']
        for ax,key in zip(axes,('prediction_gain','behavioral_distortion')):
            values=np.array([r['groups']['rare']['primary']['population'][key] for r in rows])
            ax.scatter(values,y+np.linspace(-.08,.08,len(rows)),color=color,s=38,zorder=3)
            ax.scatter(values.mean(),y+.22,color=color,marker='D',edgecolors='black',s=65,zorder=4)
    for ax in axes:
        ax.axvline(0,color='#59636d',lw=1)
        ax.grid(axis='x',alpha=.18)
        ax.set_yticks([0,1],['Model-free','Reward cue'])
        ax.set_ylim(-.28,1.5)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[0].set(xlabel='Solo − distilled test NLL (nats)\nPositive = better prediction',title='Prediction gain')
    axes[1].set(xlabel='Distilled − solo assay RMS error\nNegative = better behavior in the assay',title='Behavioral distortion')
    fig.suptitle('First main-study tranche · rare groups · population checkpoints',fontsize=12)
    fig.text(.5,-.045,'Dots show the three population means per rare kind. Diamonds show their average.',ha='center',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()

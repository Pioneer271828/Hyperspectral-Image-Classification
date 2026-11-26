"""Static raw-seed plots for the approved dual-scene package; no fitting."""
import argparse
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    if sys.flags.optimize:
        raise RuntimeError('Refuse optimized Python')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    colors = ['#327da8', '#d98a2d', '#4c937a']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for row, scene in enumerate(['paviaU', 'xuzhou']):
        with (args.base/scene/'results/paired-endpoints.csv').open() as stream:
            pairs = [r for r in csv.DictReader(stream) if r['candidate']=='content' and r['baseline']=='mean'
                     and r['head']=='fused' and r['endpoint']=='selected']
        for col, band in enumerate(['full', 'even_retained_positions']):
            data = {int(r['seed']): r for r in pairs if r['band']==band}
            values = [float(data[s]['nll_delta']) for s in (64, 512, 2048)]
            axes[row, col].bar(['64', '512', '2048'], values, color=colors)
            axes[row, col].axhline(0, color='#444444', lw=1)
            axes[row, col].set_title(f'{scene} / {band}')
            axes[row, col].set_ylabel('Fused NLL difference')
            axes[row, col].set_xlabel('Initialization / order seed')
            axes[row, col].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    fig.suptitle('Factor1 scale16: content minus mean, separately trained matched controls')
    fig.savefig(args.out/'primary-pairs.png', dpi=170)
    plt.close(fig)
    with (args.base/'wp12/graph-response.csv').open() as stream:
        data = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, scene in zip(axes, ['paviaU', 'xuzhou']):
        for i, seed in enumerate([64, 512, 2048]):
            values = [float(r['relative_l2_delta']) for r in data if r['scene']==scene
                      and int(r['seed'])==seed and r['endpoint']=='selected']
            # All fixed original views: an empirical distribution, no stochastic sampling.
            values = np.sort(values)
            ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=colors[i], label=f'{seed}: n={len(values)}')
        ax.set_title(scene)
        ax.set_xlabel('Near/wide actual broadcast-state relative L2 difference')
        ax.set_ylabel('Fraction of fixed validation views')
        ax.set_ylim(0, 1.02)
        ax.legend(title='Model seed')
    fig.suptitle('Same content checkpoint, fixed input: response strength across all views')
    fig.savefig(args.out/'graph-response.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Export editable, explicitly development-only recall and measured-cost figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.fonttype']='none'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cases',nargs=3,default=[f'transfer-s{s}-span03' for s in (2036,2046,2056)])
    a=p.parse_args();source=Path(a.analysis);all_cases=json.loads(source.read_text())['cases']
    cases=[all_cases[name] for name in a.cases]
    out=Path(a.out);out.mkdir()
    arms=list(cases[0]['metrics']);views=list(cases[0]['metrics']['content'])
    rows=[]
    for case in cases:
        for arm in arms:
            for view in views:
                metric=case['metrics'][arm][view]
                for i,recall in enumerate(metric['recall']):
                    rows.append({'seed':case['frozen_model_seed'],'arm':arm,'view':view,'class_id':i+1,
                                 'recall':recall,'support':metric['support'][i],
                                 'pipeline_p50_ms_per_20_request_batch':case['timing']['latency_p50_ms'][arm]})
    with (out/'figure-data.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    delta=np.mean([[[100*(x-y) for x,y in zip(c['metrics']['content_coverage'][v]['recall'],c['metrics']['content'][v]['recall'])]
                     for v in views] for c in cases],axis=0)
    fig,ax=plt.subplots(figsize=(10.6,4.5),layout='constrained')
    image=ax.imshow(delta,cmap='RdBu',vmin=-50,vmax=50,aspect='auto')
    ax.set_xticks(range(9),[str(x) for x in range(1,10)]);ax.set_xlabel('Numeric class ID; semantic remapping unresolved')
    ax.set_yticks(range(4),['Native436','Stride2 (218)','Stride4 (109)','Approx. center cut (122)'])
    for y in range(4):
        for x in range(9):ax.text(x,y,f'{delta[y,x]:+.1f}',ha='center',va='center',fontsize=10,
                                  color='white' if abs(delta[y,x])>=30 else 'black')
    ax.set_title('Coverage minus content: paired recall difference (percentage points)\nDevelopment only; same 45 labels/view; mean of 3 original training seeds',pad=12)
    fig.colorbar(image,ax=ax,label='Recall difference (pp)',shrink=.8)
    for suffix in ('svg','png'):fig.savefig(out/f'development-recall-delta01.{suffix}',dpi=160)
    plt.close(fig)
    labels=['Light','Full','Content','Content +\ncoverage','Permuted\ncoverage','Matched\nrandom']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained',gridspec_kw={'width_ratios':[2,1]})
    x=np.arange(6);width=.24;colors=['#22658c','#439e8f','#87bccc']
    for i,(seed,c) in enumerate(zip((2036,2046,2056),cases)):
        axes[0].bar(x+(i-1)*width,[c['timing']['latency_p50_ms'][arm] for arm in arms],width,label=str(seed),color=colors[i])
    axes[0].set_xticks(x,labels,fontsize=9);axes[0].set_ylabel('Median batch pipeline elapsed (ms)')
    axes[0].set_title('Actual CPU elapsed: 20 requests / batch');axes[0].legend(title='Frozen run seed',frameon=False)
    axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    axes[1].bar(x,[0,20,10,10,10,10],color='#22658c')
    axes[1].set_xticks(x,labels,rotation=45,ha='right',fontsize=8);axes[1].set_yticks([0,10,20])
    axes[1].set_ylabel('Actual graph builds / 20 requests');axes[1].set_title('Heavy work actually skipped')
    fig.suptitle('Development only: 6 rotated-order timing rounds; shared CPU; setup/loading reported separately',fontsize=11)
    for suffix in ('svg','png'):fig.savefig(out/f'development-cost01.{suffix}',dpi=160)
    plt.close(fig)
    receipt={'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_code':'plot_results.py',
             'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
             'new_neural_forwards':0,'new_optimizer_updates':0,'confidence_intervals':False,
             'interpretation':'Descriptive development figures; model seeds share labels/split. PNG for preview; SVG for editing.'}
    with (out/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':'EXPORTED','figure_files':4,'data_rows':len(rows),'optimizer_updates':0}))


if __name__=='__main__':main()

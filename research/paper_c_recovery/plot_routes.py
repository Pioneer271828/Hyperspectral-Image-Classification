"""Editable aggregate figures, explicitly limited to three-sweep development."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    data=json.loads(Path(a.collection).read_text());out=Path(a.out);out.mkdir()
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':10})
    directions=[('paviaU','whu-honghu'),('xuzhou','paviaU'),('xuzhou','whu-honghu'),('whu-honghu','paviaU'),('whu-honghu','xuzhou')]
    short={'paviaU':'PaviaU','xuzhou':'Xuzhou','whu-honghu':'HongHu'};labels=[short[s]+' → '+short[t] for s,t in directions]
    records=[];fig,ax=plt.subplots(figsize=(9.2,4.8));palette=['#245b92','#df8c32','#489b87']
    for i,(source,target) in enumerate(directions):
        rows=sorted([r for r in data['rows'] if r['source']==source and r['target']==target and r['arm']=='full'],key=lambda r:r['seed'])
        if len(rows)!=3:raise ValueError('Three paired model seeds per direction required')
        values=[r['full_transfer_minus_scratch_OA_pp'] for r in rows]
        for j,r in enumerate(rows):
            ax.scatter(i+(j-1)*.12,values[j],s=45,color=palette[j],label=str(r['seed']) if i==0 else None,zorder=3)
            records.append({'direction':labels[i],'kind':'transfer_minus_scratch_OA_pp','seed':r['seed'],'value':values[j]})
        ax.scatter(i,float(np.mean(values)),marker='D',s=38,color='#1f2730',zorder=4,label='mean' if i==0 else None)
    ax.axhline(0,color='#66717d',lw=1,ls='--');ax.set_xticks(range(5),labels,rotation=12,ha='right')
    ax.set_ylabel('Selected native OA difference (percentage points)');ax.set_title('Source3 → target3 versus target-scratch3')
    ax.grid(axis='y',alpha=.2);ax.legend(ncol=4,frameon=False,loc='best')
    fig.text(.09,.025,'Engineering readout on reused K5/V5 roles; source cost is additional. No convergence or independent-confirmation claim.',fontsize=8,color='#4c5664')
    fig.subplots_adjust(bottom=.25,left=.10,right=.98,top=.88)
    for ext in ('png','svg'):fig.savefig(out/('paired-engineering-transfer.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7));heavy=[];latency=[]
    for source,target in directions:
        per=[];ratio=[]
        for seed in (64,512,2048):
            rows={r['arm']:r for r in data['rows'] if r['source']==source and r['target']==target and r['seed']==seed}
            per.append(rows['content_coverage']['full_calls']/rows['full']['full_calls'])
            ratio.append(rows['content_coverage']['pipeline_p50_ms']/rows['full']['pipeline_p50_ms'])
        heavy.append(float(np.mean(per)));latency.append(float(np.mean(ratio)))
        for kind,values in (('heavy_call_ratio',per),('paired_resident_p50_ratio',ratio)):
            for seed,value in zip((64,512,2048),values):records.append({'direction':short[source]+' → '+short[target],'kind':kind,'seed':seed,'value':value})
    for ax,values,title in zip(axes,(heavy,latency),('Paid graph calls / static full','Resident batch p50 / static full')):
        ax.bar(range(5),values,color='#4c7c9f',width=.62);ax.axhline(1,color='#65717c',ls='--',lw=1)
        wrapped=[short[s]+'\n→ '+short[t] for s,t in directions]
        ax.set_xticks(range(5),wrapped,ha='center');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(1.2,max(values)*1.15))
    fig.suptitle('Content + approximate-span coverage: call budget and observed cost',fontsize=12)
    fig.text(.07,.06,'Four approximate spectral views; VNIR range cuts can be no-ops. Full-call quota = 50%.',fontsize=8,color='#4c5664')
    fig.text(.07,.025,'Means of three paired model-seed ratios; shared CPU, six rotated rounds. Cold setup/training/audit/backup excluded.',fontsize=8,color='#4c5664')
    fig.subplots_adjust(bottom=.29,left=.07,right=.98,top=.82,wspace=.26)
    for ext in ('png','svg'):fig.savefig(out/('calls-and-resident-cost.'+ext),dpi=180)
    plt.close(fig)
    with (out/'figure-data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('direction','kind','seed','value'));w.writeheader();w.writerows(records)
    print(json.dumps({'figures':2,'data_rows':len(records),'visual_QA':'PENDING; inspect both PNGs before delivery'}))


if __name__=='__main__':main()

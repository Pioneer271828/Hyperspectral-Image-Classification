"""Paper-facing fixed tables/figures from the independently recounted B supplement."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def mdtable(data,columns):
    return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|',
        *['|'+'|'.join(str(row[column]) for column in columns)+'|' for row in data]])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wp12',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir();receipt=json.loads((a.wp12/'receipt.json').read_text())
    pairs=rows(a.wp12/'paired-comparisons.csv')
    selected=[r for r in pairs if r['endpoint']=='selected' and r['head']=='fused' and r['contrast']=='exact-summary' and r['policy']=='both.joint']
    bands=['shift_a','shift_b','interior_a','interior_b'];colors={64:'#1b7fa4',512:'#dc7c26',2048:'#548c59'}
    fig,axes=plt.subplots(2,2,figsize=(12,7.6))
    for i,family in enumerate(('legacy','balanced')):
        for j,classes in enumerate((9,22)):
            ax=axes[i,j];ax.axhline(0,color='#82919c',lw=1)
            for seed in (64,512,2048):
                data={r['band']:r for r in selected if r['family']==family and int(r['classes'])==classes and int(r['seed'])==seed}
                offset={64:-.12,512:0,2048:.12}[seed]
                ax.scatter(np.arange(4)+offset,[float(data[b]['nll_delta'])*1000 for b in bands],s=38,color=colors[seed],label=str(seed))
            ax.set_title(f'{family.capitalize()} encoders · {classes} classes',fontsize=12)
            ax.set_xticks(range(4),['Shift A','Shift B','Interior A','Interior B'])
            ax.set_ylabel('Fused NLL (exact − summary) × 10³')
            ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,1].legend(title='Model seed',frameon=False)
    fig.suptitle('Paired 24-step synthetic controls; no real-condition efficacy claim',fontsize=15,y=.98)
    fig.text(.5,.015,'Each view reuses the same validation locations. Seed and view rows are paired; no spatial confidence interval.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ('png','pdf'):fig.savefig(a.out/('paired-position-nll.'+ext),dpi=170)
    plt.close(fig)
    params=rows(a.wp12/'parameter-participation.csv')
    groups={'shared position projection':'.exact_mask.','old summary projection':'.indices.',
            'content selector':None,'metadata head':None,'graph feature norm':'learnable_graph.feature_norm.'}
    def group(name):
        if '.exact_mask.' in name:return 'shared position projection'
        if '.indices.' in name:return 'old summary projection'
        if 'content_selector' in name or 'range_content' in name:return 'content selector'
        if 'condition_selector' in name or 'range_metadata' in name:return 'metadata head'
        if 'learnable_graph.feature_norm.' in name:return 'graph feature norm'
        return None
    totals=defaultdict(lambda:defaultdict(int));allocated={}
    for r in params:
        if r['family']!='balanced':continue
        g=group(r['tensor'])
        if g:
            totals[r['case']][g]+=int(r['final_changed'])
            if r['case']=='c9-s64-balanced_summary-both.mean':allocated[g]=allocated.get(g,0)+int(r['allocated'])
    fig,ax=plt.subplots(figsize=(12,5.6));palette=['#277da8','#9aa9b7','#369f82','#e38a2c','#9a77be']
    for j,representation in enumerate(('balanced_summary','balanced_exact')):
        xs=np.arange(4)+(j-.5)*.32;bottom=np.zeros(4)
        for g,color in zip(groups,palette):
            values=[]
            for mode in ('mean','content','metadata','joint'):
                cases=[v[g] for key,v in totals.items() if '-'+representation+'-both.'+mode in key]
                values.append(statistics.mean(cases))
            ax.bar(xs,values,width=.3,bottom=bottom,color=color,hatch=None if j==0 else '//',label=g if j==0 else None)
            bottom+=values
    ax.axhline(sum(allocated.values()),ls='--',lw=1,color='#52677a')
    ax.set_xticks(range(4),['mean','content','metadata','joint']);ax.set_ylabel('Elements changed from initial to final state')
    ax.set_title('Balanced common projection: measured parameter participation',fontsize=15)
    ax.text(.01,.93,'Plain: summary; hatched: exact. Means across 9/22 classes × three model seeds.',transform=ax.transAxes,fontsize=10)
    ax.legend(frameon=False,loc='center left',bbox_to_anchor=(1, .5));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.out/('balanced-participation.'+ext),dpi=170)
    plt.close(fig)
    aggregate=[]
    for family in ('legacy','balanced'):
        for classes in (9,22):
            data=[r for r in selected if r['family']==family and int(r['classes'])==classes]
            aggregate.append(dict(family=family,classes=classes,paired_seed_view_rows=len(data),
                lower_NLL=sum(float(r['nll_delta'])<0 for r in data),
                min_delta=f"{min(float(r['nll_delta']) for r in data):.8g}",max_delta=f"{max(float(r['nll_delta']) for r in data):.8g}",
                class_protected=sum(r['protected']=='True' for r in data)))
    with (a.out/'paired-summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    text='''# WP07 exact processed-position integration — engineering and development supplement

The package separates two questions: translating a retained-column set at equal count/span (endpoints move), and changing interior columns at fixed count/endpoints/span. A planted pair also matches all eight FP32 index statistics. Full mask inputs identify the retained processed-column set; a learned32-dimensional encoder is not guaranteed globally injective. No new real view or real optimizer update was executed.

The original conditional CNN/graph candidates, native lifecycle and B phase executor are reused. Summary8/exact-mask use different learned input widths; that original engineering matrix is retained. Before inspecting paired scores, a second matrix was preregistered with a common deterministic orthogonal rotation, L2 normalization and the very same437-to32 learned projection. Its eight-statistic input is padded; both variants share the same allocation, complete initialization, candidate computation and budget. Input information rank and optimizer geometry remain different. Fixed rotation buffers are explicitly pinned and checked on restore; they are not trainable parameters or a second routing system.

Two matrices contain96 interface cases, each with24 continuous and12+12 split updates, completed0 re-entry, and first minimum fused validation NLL at0/12/24. Main synthetic updates total4608. All saved routes and point auditors pass. Within each family mean/content representations are predictor aliases; after excluding only the known fixed rotation buffers, their cross-family trained tensors, shared/active Adam state and RNG also match. Dormant condition projections may have different zero-moment Adam slot inventories because different encoder paths executed. Full optimizer dictionaries are therefore reported separately and are not universally bitwise identical. Thus there are10 distinct policy variants per class/seed across the two families, not16 independent scientific arms. All repetitions share the released synthetic geometry; per head/view denominators are20 point exposures for9 classes and46 for22 classes, with class supports2/4 (9 classes) or2/3 (22 classes) depending on planted points. The four spectral views reuse the same locations.

## Exact − summary, joint policy, selected fused head

'''+mdtable(aggregate,list(aggregate[0]))+'''

Negative values favor the exact descriptor in this synthetic fixture. These paired short-budget values are engineering diagnostics. They do not establish real processed-column benefit, physical bandwidth, sensor generalization or independent spatial accuracy. Same-content descriptor swaps, summary collision negatives, mean/content negatives, missing fallback and CNN/graph isolation are in the full table; both Q/A index and value identities remain fixed under descriptor-only changes. Derived synthetic inputs have their own geometry caches.

The B WP12 consumer independently reconstructs2304 synthetic and60 old-real-stress confusion matrices and NLL from saved original-point probabilities. Five prior B WP12 CSVs remain byte-identical. The central production registry is unchanged; ingestion remains the integration owner's action. Effective changed parameters and gradient element-step counts are supplied separately from allocated parameters. Fixed rotation buffers add381,938 FP32 elements (1,527,752bytes) to a dual-selector model and are reported as buffers, never optimized parameters. All candidate paths still execute.

## Real source and role evidence

The PaviaU bank contains127 devtrain /41 devval normalized-FP32 blocks,30/17 labelled blocks; Xuzhou has73/5 raw-FP32 blocks,31/5 labelled blocks. Each role releases45 points,9 classes with5 each, original support512/validation3031. Current bank pins, publication chains, loader metadata, train extrema and actual existing-view hashes are retained. These are103/436 processed columns, not authenticated original instrument coordinates. Approximate wavelengths and absent FWHM/GSD do not become physical evidence. A loader comment about removed Pavia bands is not an authenticated inverse mapping from these bank columns.

Eight existing full/even views pass provider access, historical preflight, exact input/released-point re-entry and frozen Q/near/wide identity checks (16 payload reads). Twenty requests for new positions, odd or an unadmitted role fail before payload access. Generic sampling API capability is not protocol admission. All new controlled real views remain metadata-only proposals. A registered read-only provider cannot authorize optimizer updates.

One existing Xuzhou seed512 fine16 graph-content checkpoint was read on its original physical GPU3. All-missing and neutral exact-mask insertion preserve270 historical probability vectors bitwise; old tensors/file and CPU/CUDA RNG remain unchanged. New heads are untrained. Descriptor-only changes preserve input, points, frozen geometry and CNN predictions. Unknown approximate wavelengths/absent GSD are exact negatives. Its60 endpoint matrices are stress diagnostics, not trained-condition efficacy. Separate score-free cost samples use the same old predictor and existing full/even inputs. GPU measurements share the device with other jobs; no universal speed ranking or kernel-time claim follows.

## Next real matrix is a draft

For Xuzhou, a52/218 count construction (52 in PaviaU,218 in Xuzhou) extends the planted internal collision with a common suffix. Xuzhou's shared_a and interior_b both retain218 columns with endpoints0/434; shift_b is shared_a+1 with endpoints1/435 and equal span434. All eight-statistic summaries collide for the internal pair; all four representations distinguish the shift, and both exact representations distinguish the internal pair before any new-view score is read. These are processed-index interventions, not new sensors. The positions/hashes and both dataset qualifications are supplied; no odd view is adopted.

The minimal Xuzhou proposal shares baseline_a once across both questions: content, joint-balanced-summary and joint-balanced-exact × three model seeds =9 models;31 released labelled blocks × three views =93 updates/model,837 total. Prefix46+resume47+completed0, validation0/46/93, original three-head loss/Adam1e−4 and same complete initial state/candidates. The planned whole-chain GPU3 range is15–40minutes plus contention, not measured training speed. Exact837-budget approval, explicit admission of all three derived views, fresh hash-pinned source/statistics/role/geometry preflight, real-provider optimizer/recovery and physical arithmetic qualification remain gates. Current read-only code rejects training. A2-percentage-point per-class recall floor on5 points allows no class loss; lower fused NLL must hold across every model seed/view for the primary exact-vs-content contrast. Three model seeds share one released support/validation set and do not create independent spatial confirmation.

C owns actual device/header/byte admission and D owns the corresponding semantic evidence. The read-only descriptor schema and code pin are provided separately. B does not duplicate C's inventory or treat its synthetic/controlled masks as genuine-device evidence. Source/runtime identities, failures, actual elapsed work, costs and verified incremental backup are in HANDOFF.json.
'''
    (a.out/'PAPER-D-NOTES.md').write_text(text)
    (a.out/'REPORT.md').write_text('# WP07 B 精确带位接入与可辨性\n\n工程和开发诊断完成：96接口例、4608主合成更新，连续/分段/完成重入和原点独审通过；真实更新0，新真实视图0。八统计碰撞、精确掩码及共同投影容量控制分开验收。真实full/even provider及冻结几何/重入通过；新视图仍未准入，不能由底层切片能力推断授权。\n\n旧徐州权重的零更新中性读出与历史概率逐位一致；新条件头未训练，不能报告真实精度收益。下一矩阵837更新仅draft，明确三视图、两问题共享基准、原45/45释放点和放行条件。WP12为私有补充，中央登记未修改。\n\n完整方法、表图及限制见PAPER-D-NOTES.md，来源/命令/失败/耗时和增量恢复回执见HANDOFF.json。\n')
    print(json.dumps({'status':'passed','aggregate':aggregate,'WP12_receipt':receipt['executor']}),flush=True)


if __name__=='__main__':main()

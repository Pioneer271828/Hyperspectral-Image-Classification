"""Create paper-ready figures and a complete readout from anonymous point tables."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(root, name):
    with (root/name).open(newline='') as f:return list(csv.DictReader(f))


def mdtable(data, keys):
    def value(v):
        if isinstance(v,float):return f'{v:.9g}'
        return str(v)
    return '\n'.join(['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']+
        ['|'+'|'.join(value(r[k]) for k in keys)+'|' for r in data])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir();start=time.perf_counter()
    endpoints=rows(a.analysis,'endpoint-decomposition.csv');points=rows(a.analysis,'points.csv')
    proxies=rows(a.analysis,'point-graph-proxy.csv');bands=rows(a.analysis,'band-metadata.csv')
    classes=rows(a.analysis,'class-decomposition.csv');graphs=rows(a.analysis,'paired-band-graphs.csv')
    gradients=rows(a.analysis,'gate-gradients.csv');parameters=rows(a.analysis,'parameter-changes.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),bbox_inches='tight');fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True)
    fused=[r for r in endpoints if r['head']=='fused']
    for scene,ax in zip(('paviaU','xuzhou'),axes):
        rr=[r for r in fused if r['scene']==scene]; y=np.arange(len(rr))
        ax.barh(y-.16,[1e3*float(r['coefficient_delta']) for r in rr],.30,label='Coefficient switch (A−B)',color='#247ba0')
        ax.barh(y+.16,[1e3*float(r['training_residual_delta']) for r in rr],.30,label='Training residual (B−C)',color='#ed8b36')
        ax.scatter([1e3*float(r['total_delta']) for r in rr],y,s=34,c='#172a3a',label='Total (A−C)',zorder=4)
        ax.axvline(0,color='#7c8992',lw=.8);ax.set_yticks(y,[r['seed']+' / '+('full' if r['band']=='full' else 'retained') for r in rr])
        ax.invert_yaxis();ax.set_title(('PaviaU' if scene=='paviaU' else 'Xuzhou')+' · fused head');ax.set_xlabel('Mean paired NLL difference × 10³ (lower is better)');ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,fontsize=9,frameon=False)
    fig.suptitle('Frozen model attribution: one reference order, 45 points per row',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'fused-nll-decomposition')
    flips=[r for r in points if r['transition'] in ('wrong->correct','correct->wrong')]
    assert len(flips)==3
    fig,axes=plt.subplots(1,3,figsize=(12,4.0))
    for ax,r in zip(axes,flips):
        names=['C · mean-trained','B · equal coeff.','A · learned coeff.']
        margins=[1e3*float(r['true_class_margin_'+n]) for n in ('trained_mean','content_weights_mean_coeff','content')]
        ax.barh(range(3),margins,color=['#8796a5','#247ba0','#ed8b36']);ax.axvline(0,color='#334c60',lw=.8)
        ax.set_yticks(range(3),names,fontsize=8);ax.invert_yaxis();ax.set_xlabel('True-class margin × 10³')
        lo,hi=min(0,min(margins)),max(0,max(margins));span=hi-lo;ax.set_xlim(lo-.25*span,hi+.25*span)
        ax.set_title('Xuzhou '+r['seed']+' · '+r['head']+' · class '+r['class_id']+'\n'+('full' if r['band']=='full' else 'retained'))
        for i,y in enumerate(margins):ax.annotate(f'{y:.4f}',(y,i),xytext=(5 if y>=0 else -5,0),textcoords='offset points',ha='left' if y>=0 else 'right',va='center',fontsize=8)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('All correctness flips in 1,620 paired point rows (two panels share one class-6 point)',fontsize=12)
    fig.tight_layout();save(fig,'xuzhou-point-margins')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,sharey=True)
    for scene,ax in zip(('paviaU','xuzhou'),axes):
        for band,marker,color in [('full','o','#247ba0'),('even_retained_positions','x','#ed8b36')]:
            rr=[r for r in proxies if r['scene']==scene and r['band']==band]
            ax.scatter([float(r['mean_inverse_HHI'])/float(r['native_cells']) for r in rr],
                [float(r['content_inverse_HHI'])/float(r['native_cells']) for r in rr],s=15,marker=marker,color=color,alpha=.6,label='full' if band=='full' else 'retained')
        ax.plot([0,1],[0,1],color='#8796a5',lw=.8);ax.set_xlim(.12,.9);ax.set_ylim(.12,.9)
        ax.set_title('PaviaU' if scene=='paviaU' else 'Xuzhou');ax.set_xlabel('Mean-mixture inverse HHI / native cells');ax.grid(alpha=.15)
    axes[0].set_ylabel('Content-mixture inverse HHI / native cells');axes[1].legend(frameon=False)
    fig.suptitle('Weight concentration can change while positive union support stays identical',fontsize=12)
    fig.tight_layout();save(fig,'proxy-concentration')
    distribution=[]
    for endpoint in endpoints:
        rr=[r for r in points if all(r[k]==endpoint[k] for k in ('scene','seed','head','band'))]
        vals=[float(r['total_delta']) for r in rr]
        distribution.append({**{k:endpoint[k] for k in ('scene','seed','head','band')},'points':len(rr),
            'NLL_improved_points':sum(v<0 for v in vals),'NLL_worsened_points':sum(v>0 for v in vals),'NLL_equal_points':sum(v==0 for v in vals),
            'negative_delta_sum':math.fsum(v for v in vals if v<0),'positive_delta_sum':math.fsum(v for v in vals if v>0),
            'median_delta':statistics.median(vals),'minimum_delta':min(vals),'maximum_delta':max(vals)})
    with (a.out/'point-delta-distribution.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(distribution[0]));w.writeheader();w.writerows(distribution)
    for r in fused:
        for k in ('total_delta','coefficient_delta','training_residual_delta'):r[k]=float(r[k])
    flipread=[]
    for r in flips:
        flipread.append({**{k:r[k] for k in ('scene','seed','head','band','point_id','class_id','pred_trained_mean','pred_content_weights_mean_coeff','pred_content')},
            **{k:float(r[k]) for k in ('total_delta','coefficient_delta','training_residual_delta','true_class_margin_trained_mean','true_class_margin_content_weights_mean_coeff','true_class_margin_content')}})
    assert all(int(r['coefficient_correctness_changes'])==0 for r in endpoints)
    parameter_ranges={}
    for tensor in sorted({r['tensor'] for r in parameters}):
        vals=[float(r['delta_l2']) for r in parameters if r['tensor']==tensor];parameter_ranges[tensor]={'min_delta_l2':min(vals),'max_delta_l2':max(vals)}
    facts={'endpoint_rows':len(endpoints),'class_rows':len(classes),'point_rows':len(points),'unique_points':len({(r['scene'],r['point_id']) for r in points}),
        'correctness_flip_rows':len(flips),'unique_correctness_flip_points':len({r['point_id'] for r in flips}),
        'coefficient_argmax_changes_all_heads_bands_seeds':sum(int(r['coefficient_argmax_changes']) for r in endpoints),
        'coefficient_correctness_changes_all_heads_bands_seeds':sum(int(r['coefficient_correctness_changes']) for r in endpoints),
        'PaviaU_trained_pair_argmax_changes':sum(int(r['trained_pair_argmax_changes']) for r in endpoints if r['scene']=='paviaU'),
        'Xuzhou_trained_pair_argmax_changes':sum(int(r['trained_pair_argmax_changes']) for r in endpoints if r['scene']=='xuzhou'),
        'paired_band_labelled_views':{s:sum(r['scene']==s for r in graphs) for s in ('paviaU','xuzhou')},
        'paired_band_Q_near_wide_support_identical_every_labelled_view':all(r[k]=='True' for r in graphs for k in r if k.endswith('support')),
        'paired_band_Q_near_wide_values_different_every_labelled_view':all(r[k]=='False' for r in graphs for k in r if k.endswith('values')),
        'gate_positive_gradient_steps':sum(int(r['positive_gradient_steps']) for r in gradients),
        'gate_changed_steps':sum(int(r['changed_steps']) for r in gradients),'parameter_delta_l2_ranges':parameter_ranges,
        'all_positive_mixtures_have_identical_union_support':all(r['same_positive_union_support']=='True' for r in proxies),
        'proxy_rows':len(proxies),'same_view_shared_gate_not_point_gate':True,'no_significance_tests':True}
    (a.out/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    note='''# Frozen scale16 point attribution — Paper D / WP12

This supplement uses the previously completed development protocol: three model seeds per scene, one shared spatial release per scene, 45 validation points per band and five points per class. It adds no training, labels, query access or model selection. The selected endpoint equals the last endpoint; duplicates are omitted. Anonymous point IDs pair the same released points across model seeds, heads and bands. PaviaU and Xuzhou classes have different semantics.

For each point, A denotes the content-trained model with learned coefficients, B the same weights with equal coefficients, and C the separately mean-trained model. We reconstruct NLL(A)−NLL(C)=[NLL(A)−NLL(B)]+[NLL(B)−NLL(C)]. This is an exact accounting identity for a specified reference order. The second term includes all remaining differences in the learned model, including upstream features and graph feature normalization; it is not a unique causal contribution of the gate. All 36 endpoint NLL values reconstruct the saved original and intervention metrics within 10⁻¹².

## Fused-head decomposition

Negative differences favor A. Every row contains 45 points. Units are natural-log NLL, not percentage points.

'''+mdtable(fused,['scene','seed','band','total_delta','coefficient_delta','training_residual_delta'])+'''

PaviaU has lower fused NLL for all three seeds in both bands, but no correctness changes in any of the three heads; nine paired rows change between incorrect class predictions. For seed 2048 the coefficient-switch term is adverse, and the training residual offsets it. Xuzhou seed 2048 is adverse in both terms. Thus neither a consistent independent coefficient benefit nor a confirmed accuracy gain follows from the aggregate training contrast.

## Every correctness flip

'''+mdtable(flipread,['scene','seed','head','band','point_id','class_id','pred_trained_mean','pred_content_weights_mean_coeff','pred_content','total_delta','coefficient_delta','training_residual_delta'])+'''

There are only two distinct correctness-flip points. Xuzhou seed 64 loses one full-band CNN class-5 point (5→2); its coefficient term is exactly zero because that head is invariant to the downstream graph mixing coefficients. Xuzhou seed 512 gains the same fused class-6 point in both bands (2→6). At this point A and B both predict 6, and the coefficient switch slightly increases true-class NLL; the remaining training difference supplies the improvement. Across all 1,620 paired rows, switching A to equal coefficients changes 12 argmax predictions (ten PaviaU, two Xuzhou), all between incorrect classes; no point changes correctness. These findings locate the earlier protection failure and apparent gain without claiming a unique mechanism for either training trajectory.

## Band identity and local graph-input proxy

'''+mdtable(bands,['scene','band','actual_band_count','declared_wavelength_min','declared_wavelength_max','declared_wavelength_span','wavelength_quality','FWHM_present'])+'''

The retained conditions contain 52/103 PaviaU bands and 218/436 Xuzhou bands. PaviaU retains both declared spectral endpoints; Xuzhou loses only the last declared center. These are index-subset conditions, not half physical bandwidth. All wavelengths are approximate, FWHM is absent and original sensor mappings are unauthenticated.

The 22 labelled validation views (17 PaviaU, five Xuzhou) retain identical positive Q/near-A/wide-A support across bands, while all corresponding weights differ. The graph inputs therefore change numerically across bands despite equal topology. This paired observation does not isolate a wavelength effect from the simultaneously changed spectra, normalization and assignment weights.

At the 90 unique released points we evaluate scalar rows of Q A² R, with R=Qᵀ/clamp(column mass,10⁻⁶), using float64 analysis of frozen float32 matrices. The 540 point/seed/band rows distinguish strictly positive support counts, self weights and inverse HHI (mass²/squared-mass sum). Near positive support ranges from 400 to 784 native cells; wide and both positive mixtures cover 784 cells at these points. Mean and content mixtures have identical positive union support in all rows, while their concentration can change. Inverse HHI is a weight-concentration measure, not nonzero coverage. This proxy describes the graph-input pixel-feature path and excludes nonlinear message transforms, the upstream CNN, shared normalization and global-gate dependencies; it is not the classifier Jacobian or full raw-image receptive field.

The existing actual two-layer branch-response summaries are joined by view without recomputing the response batch. A view shares one pair of gate coefficients across its points; these coefficients are neither point-specific nor class-specific. Per-view actual response summaries do not measure the actual local response at each released point. The existing three fixed-source finite-difference probes remain limited to their original source positions and tolerances.

## Participation and limits

The saved training histories contain 366 positive-gradient and changed gate steps (180 PaviaU, 186 Xuzhou). Each content model also changes the existing graph feature-normalization weight and bias. This establishes participation under the completed budget, not validation benefit. Parameter deltas and all per-class, point, view and paired-band tables are supplied; point-delta-distribution.csv includes favorable and adverse NLL counts without selecting points by score. Point rows sharing a view, band pairs and three model seeds sharing one release are not independent spatial replicates. No significance or sensor-generalization claim is made. No tuning or new training follows from this supplement without a separately reviewed gap and protocol.
'''
    (a.out/'PAPER-D-NOTES.md').write_text(note)
    report='''# WP07 B 逐点归因补充

已完成六个冻结content模型的均值系数逐点补采，零优化；1,620行点、324行逐类、36行端点均可重建旧指标，误差不超过1e−12。selected与last相同，未重复计数。

PaviaU三头均无正确性变化，9行是在错误类别间改变argmax；fused NLL改善不能解释为精度提升，2048种子的系数项反向、由训练残差抵消。徐州仅两个不同点改变正确性：64/full/CNN类5从5误判为2（系数项0）；512/fused类6同一点在两带从2改为6，均值系数也预测6，学习系数略增该点NLL。六模型三头两带共1,620行在同权重切换系数时有12行argmax改变（PaviaU10/徐州2），均在错误类别之间，正确性未改变。

逐点分解是指定参考顺序的恒等式，训练残差不是门的唯一因果贡献。完整数字、英文可复用方法/结果/限制及三张图见PAPER-D-NOTES.md与同目录文件。不得把视图共享系数称为逐点/逐类门，不得把QA²R集中度称为完整非线性局部响应。

波段数为103→52与436→218；PaviaU声明波长跨度不变，徐州仅末端少一个声明中心，故不是物理带宽减半。22个含释放点视图两带Q/A正支持相同但权重均变化。540行逐点结构代理里两种正混合支持相同，变化的是权重集中度；所有波长均近似且无FWHM。

原始tile/ordinal/native索引映射与补采原始概率仅在授权远程私有目录；交付只含匿名点/视图ID及派生表。新训练0、新标签0、query/原始GT访问0。工程与备份回执、完整源码身份和真实耗时见HANDOFF.json。
'''
    (a.out/'REPORT.md').write_text(report)
    summary={'status':'passed','facts':facts,'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.perf_counter()-start,'input_summary_sha256':hashlib.sha256((a.analysis/'summary.json').read_bytes()).hexdigest()}
    (a.out/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()

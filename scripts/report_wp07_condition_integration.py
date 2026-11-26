"""Read-only report and figures for the WP07 conditional integration supplement."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(folder,name):
    with (folder/name).open(newline='') as f:return list(csv.DictReader(f))


def mdtable(data,keys):
    return '\n'.join(['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']+
        ['|'+'|'.join(f'{r[k]:.8g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+'|' for r in data])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wp12',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir();receipt=json.loads((a.wp12/'receipt.json').read_text());pairs=rows(a.wp12,'synthetic-pairs.csv')
    participation=rows(a.wp12,'parameter-participation.csv');costs=rows(a.wp12,'synthetic-resident-cost.csv');real=rows(a.wp12,'real-untrained-condition-endpoints.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),bbox_inches='tight');fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    modes=['mean','content','metadata','joint'];groups=['content_selector','verified_physical_encoder','index_encoder','condition_head','graph_feature_norm']
    colors=['#227ba0','#8898aa','#24a789','#ea8935','#a27fc1'];fig,ax=plt.subplots(figsize=(9,5));bottom=np.zeros(4);param_summary=[]
    per_case=defaultdict(lambda:defaultdict(int));allocated=defaultdict(lambda:defaultdict(int))
    for r in participation:
        if r['policy'].startswith('both.'):
            per_case[r['case']][r['group']]+=int(r['final_changed_elements']);allocated[r['case']][r['group']]+=int(r['allocated_elements'])
    for group,color in zip(groups,colors):
        vals=[]
        for mode in modes:
            members=[(c,x) for c,x in per_case.items() if c.endswith('.'+mode)];data=[x[group] for c,x in members]
            assert len(data)==6;avg=statistics.mean(data);vals.append(avg)
            param_summary.append(dict(mode=mode,group=group,allocated=allocated[members[0][0]][group],changed_min=min(data),changed_max=max(data),changed_mean=avg))
        ax.bar(modes,vals,bottom=bottom,label=group.replace('_',' '),color=color);bottom+=np.asarray(vals)
    ax.axhline(3040,color='#536779',ls='--',lw=.8);ax.text(3.45,3075,'3,040 allocated in these groups',ha='right',fontsize=8)
    ax.set_ylim(0,3450);ax.set_ylabel('Elements changed from initial to final state');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Equal total capacity; measured selector/normalization participation',fontsize=13)
    ax.set_title('Both selectors · six cases per policy (9/22 classes × three model seeds)',fontsize=10)
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8);fig.tight_layout();save(fig,'parameter-participation')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True);seedcolors={64:'#247ba0',512:'#ed8b36',2048:'#769461'}
    for c,ax in zip((9,22),axes):
        rr=[r for r in pairs if int(r['classes'])==c and r['candidate']=='joint' and r['baseline']=='content' and r['head']=='fused']
        for j,scope in enumerate(('both','cnn','graph')):
            for i,seed in enumerate((64,512,2048)):
                values={r['band']:1e3*float(r['nll_delta']) for r in rr if r['scope']==scope and int(r['seed'])==seed}
                x=j+(i-1)*.16
                ax.plot([x-.025,x+.025],[values['full'],values['even_retained_positions']],color=seedcolors[seed],alpha=.35,lw=.9)
                ax.scatter(x-.025,values['full'],color=seedcolors[seed],s=35,label=str(seed) if j==0 else None)
                ax.scatter(x+.025,values['even_retained_positions'],color=seedcolors[seed],s=35,marker='x')
        ax.axhline(0,color='#8795a5',lw=.8);ax.set_xticks([0,1,2],['Both','CNN selection','Graph selection']);ax.set_title(str(c)+' classes · synthetic only');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Paired fused NLL (joint − content) × 10³');axes[1].legend(title='Model seed',frameon=False,fontsize=8)
    fig.suptitle('The 12-step engineering fixture does not establish consistent joint-condition gain',fontsize=12)
    fig.text(.5,.01,'Circles: full; crosses: retained. Each pair shares the same validation population; no spatial confidence interval.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,'synthetic-paired-nll')
    syn_summary=[]
    for c in (9,22):
        for scope in ('both','cnn','graph'):
            rr=[r for r in pairs if int(r['classes'])==c and r['scope']==scope and r['candidate']=='joint' and r['baseline']=='content' and r['head']=='fused']
            syn_summary.append(dict(classes=c,scope=scope,seed_band_rows=len(rr),NLL_better=sum(float(r['nll_delta'])<0 for r in rr),
                NLL_min=min(float(r['nll_delta']) for r in rr),NLL_max=max(float(r['nll_delta']) for r in rr),protected=sum(r['all_classes_protected']=='True' for r in rr)))
    real_summary=[]
    for regime in ('fine16_graph','native_factor2_both'):
        for scope in ('cnn','graph','both'):
            rr=[r for r in real if r['regime']==regime and r['scope']==scope and r['condition']=='matched' and r['neutral']=='False' and r['head']=='fused']
            if rr:real_summary.append(dict(regime=regime,scope=scope,rows=len(rr),NLL_min=min(float(r['nll_delta']) for r in rr),NLL_max=max(float(r['nll_delta']) for r in rr),
                OA_changed_rows=sum(float(r['oa_delta'])!=0 for r in rr),condition_parameters_trained=False))
    cost_summary=[]
    for c in (9,22):
        for mode in modes:
            rr=[r for r in costs if int(r['classes'])==c and r['policy']=='both.'+mode]
            cost_summary.append(dict(classes=c,mode=mode,parameters=rr[0]['allocated_parameters'],
                p50_min_ms=min(float(r['p50_ms']) for r in rr),p50_max_ms=max(float(r['p50_ms']) for r in rr),
                candidates='3CNN+2graph',device='CPU one thread'))
    facts=dict(synthetic_cases=72,unique_policy_pairs_per_class_seed=10,synthetic_updates=1728,new_real_updates=0,
        source_condition_cells=4,condition_design_rank=4,condition_design_columns_with_intercept=9,
        synthetic_joint_content=syn_summary,real_untrained_stress=real_summary,
        parameter_summary=param_summary,CPU_cost_summary=cost_summary,
        independent_confusions=receipt['independent_fixture_confusion_matrices']+receipt['independent_real_confusion_matrices'],
        real_trained_condition_gain=None,physical_sensor_gain=None,central_WP12_registry_modified=False)
    (a.out/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    note='''# WP07 observed-index conditional multiscale integration

This supplement extends the existing shared-kernel dilation 1/2/3 and same-Q radius 1/2 implementation with eight processed-band-index statistics. Approximate wavelengths remain unavailable to the verified physical encoder; no FWHM, GSD or original sensor mapping is invented. Unknown wavelengths retain the index path, while all-missing metadata reduces metadata selection to equal averaging and joint selection to the same trained content selector. No real optimizer update is authorized by this package.

The real preflight manifests contain 1,476 scene/role/factor/band view rows but only four distinct condition cells. Processed-band count uniquely identifies scene and full/retained view in this inventory. The nine-column descriptor design with intercept has rank four. There is no same-scene, same-count alternate position set and no nontrivial within-scene/within-band descriptor permutation. Thus these data cannot identify an isolated band-position or physical-scale effect. Actual forward-captured band declarations independently agree with the preflight positions; wavelengths are approximate and FWHM is absent.

The engineering matrix contains 72 interface cases: 9/22 classes × three model seeds × four selection policies × three selector locations. Each uses the same complete initial state and total parameter count within class/seed, the same 3-CNN/2-graph candidate pool, loss, exposures and 12-step budget. The other branch keeps equal averaging in CNN-only/graph-only selection tests; neither classifier branch is disabled. The three mean scopes are equivalent policy pairs and serve as negative wiring controls, leaving ten distinct policy pairs per class/seed. They are not independent scientific arms. Continuous 12-step and 5+7-step routes, selected/last replay and completed zero-step re-entry pass in all 72 cases: 1,728 synthetic updates in total. The first case's 24 updates were reused after export-only import failures. No completed training stage was repeated.

An independent WP12-facing consumer recounts 864 synthetic and 300 real-stress confusion matrices from saved original-point probabilities, with NLL differences at most 10⁻¹². Five prior B WP12 CSV tables remain byte-identical. The append-only B supplement is ready for the integration owner; the central production registry is unchanged. All rows retain synthetic-engineering versus untrained real-stress scope tags.

## Synthetic joint − content (fused head)

'''+mdtable(syn_summary,['classes','scope','seed_band_rows','NLL_better','NLL_min','NLL_max','protected'])+'''

The fixture establishes end-to-end consumption, recovery and falsifiable controls. The completed short budget does not establish consistent joint-condition benefit. Model seeds share the same synthetic design; the two band views are paired observations, not independent spatial replicates. All three classifier heads and all class counts are provided.

## Parameter participation and measured cost

'''+mdtable(param_summary,['mode','group','allocated','changed_min','changed_max'])+'''

Allocated elements and elements whose final values changed are distinct. Additional per-tensor tables retain actual nonzero-gradient element-step counts; those sums are not unique parameter counts. Frozen assignment/edge scorers remain unchanged. Source shared tensors also train; the figure shows the selector/normalization groups, with all groups available in the table.

'''+mdtable(cost_summary,['classes','mode','parameters','p50_min_ms','p50_max_ms','candidates','device'])+'''

Each band cost uses two warm-ups and five resident forward samples, including preparation, frozen-Q/A application, diagnostics and policy-hook overhead. Candidate execution is retained even for masked selectors. CPU contention, differing native input shapes and control-hook costs preclude a universal speed ranking. The seven real-checkpoint GPU measurements use each checkpoint's original physical GPU3 and report resident neutral-graft latency plus PyTorch allocated peak memory; they are not isolated kernel times or whole-system memory.

## Existing real checkpoints: untrained-condition stress only

'''+mdtable(real_summary,['regime','scope','rows','NLL_min','NLL_max','OA_changed_rows','condition_parameters_trained'])+'''

Six fine16 graph-content checkpoints and the original-GPU Xuzhou seed512 factor2 dual-content checkpoint retain every old tensor. Newly inserted metadata heads have fixed-seed, untrained weights, so their nonzero effects demonstrate consumption or sensitivity only. All-missing and explicit zero-head insertion reproduce historical point probabilities bitwise. Unknown approximate wavelengths and absent GSD are exact negative controls. CNN-only condition perturbations preserve graph-head probabilities; graph-only perturbations preserve CNN-head probabilities. The old factor2 graph supports remain degenerate, so its roughly 10⁻⁹ graph-only NLL differences must not be interpreted as graph-range information gain. The package also verifies both adjacency index/value sets under condition changes before classifier heads, and CPU/CUDA RNG isolation of the final initializer. A CPU reconstruction matches the six archived fixture initial states and all seven previously probed model tensor states; the RNG fix does not alter their initialized parameter values.

## Next real pairing — draft, not released

The machine-readable draft requests Xuzhou even versus odd processed positions, 218/436 bands in each view, after a new approved preflight. Original K5/V5 roles remain 45/45 points, nine classes with five points per class; 31 labelled training blocks and five validation blocks. Four common-capacity policies × three model seeds × 62 steps would cost at most 744 real updates, split 31+31 with zero-step completion. Selection is first minimum fused V5 NLL at 0/31/62, not independent confirmation. The primary contrast is joint−content with lower fused NLL in every seed and both views plus a −0.02 recall floor for every head/class. One lost class point is 20 percentage points, so this guard permits no class loss. GPU3 end-to-end estimate is 10–25 minutes plus contention, explicitly including registration, validation, independent audit and checkpoint IO.

The odd view is not currently published or executed. Release requires separate main authorization, hash-pinned even/odd materialization, pre-score condition separability and graph-support checks, complete initial-state/budget matching, a reviewed real-provider integration and numerical/recovery qualification. The present synthetic adapter rejects real providers. This draft does not establish sensor/GSD eligibility or permit query/raw-GT use; independent spatial and physical-condition confirmation remain missing.
'''
    (a.out/'PAPER-D-NOTES.md').write_text(note)
    chinese='''# WP07 B 条件多尺度集成与可辨性验收

已完成共同候选/共同容量的条件多尺度工程主链：72个接口案例、1728合成更新，连续/分段/完成重入及原点独审全部通过；真实训练0。三种mean scope是同策略负例，不当独立科学臂；每类/种子有10个不同策略对。

实际1476条预检视图只有4个场景×带条件格，带数可完全识别场景与full/retained，9列含截距设计秩4；无同场景同带数的独立带位变化，物理波长/GSD也无可信变化。索引条件可接通，但不能从这些格子确认位置/覆盖/GSD独立因果。

原CNN共享核dilation1/2/3、同Q半径1/2与物理缺失回退直接复用，仅补索引编码及控制适配。四模式总容量相等不代表有效参与相同；完整逐张量最终改变数与逐步非零梯度数另附。12步夹具joint−content没有一致改善，工程通过不写为精度收益。

七个既有真实检查点的条件头均未在新索引上训练；零头/缺失插入精确保留历史点概率，非零随机头仅说明消费/敏感性。未知波长/无GSD为精确负例，分支隔离通过。粗图范围退化仍保留。864+300份混淆矩阵及NLL独立复算通过，旧B WP12五表字节不变；中央WP12登记未修改。

下一次最小真实配对仅为draft：徐州even/odd各218带，3种子×4模式×62=744更新上限，V5九类每类5；预计GPU3整链10—25分钟。odd尚未发布，授权、预检、来源/几何、真实provider/恢复与设备放行均须另审，本包真实执行guard仍拒绝。详见next-real-pairing-draft.json和英文方法/表图。

源身份、命令、失败、真实耗时及去重备份见HANDOFF.json；不得将合成接口验收或随机条件头当作已训练真实条件收益。
'''
    (a.out/'REPORT.md').write_text(chinese)
    (a.out/'receipt.json').write_text(json.dumps({'status':'passed','source_wp12_receipt_sha256':hashlib.sha256((a.wp12/'receipt.json').read_bytes()).hexdigest(),
        'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':['parameter-participation','synthetic-paired-nll']},indent=2)+'\n')
    print(json.dumps({'status':'passed','synthetic_summary':syn_summary,'real_stress':real_summary}))


if __name__=='__main__':main()

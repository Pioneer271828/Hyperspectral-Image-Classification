"""WP07 condition integration: existing phase executor, no real optimizer authorization."""
import argparse
import csv
from collections import defaultdict, Counter
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from research_runtime import code_snapshot,require_frozen_code
from transfer import wp07_condition_run as integration
from transfer import wp07_observed_conditions as conditions
from transfer import wp07_fine16_training as phases
from transfer import wp07_graph_range as graph
from transfer import registered_development as dev
from transfer.development_bank import _atomic_new
from transfer.native_scale_campaign import read,packet
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact
from scripts.run_native_released_points import arithmetic


def save(path,value):
    with BytesIO() as b:torch.save(value,b);_atomic_new(path,b.getvalue())
    return dev._pin(path)


def table(path,rows):
    graph.require(rows,'Empty table')
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inventory(core,out,code):
    """Only frozen preflight JSON descriptors; do not materialize images or access labels."""
    out.mkdir();rows=[];details=[];signatures=defaultdict(set);inputs=[]
    for scene in ('paviaU','xuzhou'):
        planpath=core/scene/'pair/plan.json';plan=json.loads(planpath.read_text());inputs.append(dev._pin(planpath))
        p=plan['preflight'];receipt=json.loads(Path(p['path']).read_text());graph.require(hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256'],'Changed preflight');inputs.append(p)
        graph.require(not receipt['query_accessed'] and not receipt['raw_gt_loaded'],'Preflight authorization differs')
        fullsets={tuple(v['source']['positions']) for v in receipt['views'] if v['band']=='full'}
        graph.require(len(fullsets)==1,'Full positions not uniform');full=next(iter(fullsets));master=len(full)
        graph.require(full==tuple(range(master)),'Processed-band parent positions are not contiguous')
        groups=defaultdict(list)
        for v in receipt['views']:groups[v['band'],v['role'],v['factor']].append(v)
        for (band,role,factor),views in sorted(groups.items()):
            positions={tuple(v['source']['positions']) for v in views};graph.require(len(positions)==1,'Within-condition position variation')
            pos=next(iter(positions));features=conditions.index_features(pos,master);signatures[(scene,len(pos))].add(features)
            rows.append(dict(scene=scene,band=band,role=role,factor=factor,views=len(views),master_processed_bands=master,
                retained_processed_bands=len(pos),position_set_sha256=hashlib.sha256(json.dumps(pos).encode()).hexdigest(),
                **dict(zip(conditions.FEATURES,features)),verified_wavelength_conditions=0,verified_GSD_conditions=0,
                donor_permutation_within_scene_band='identity_only'))
        details.append(dict(scene=scene,views=len(receipt['views']),unique_observed_index_conditions=2,master_processed_bands=master,
            verified_physical_wavelengths=False,verified_GSD=False,train_labels_read=0,validation_labels_read=0,images_decoded=0))
    unique={(r['scene'],r['band']):[r[k] for k in conditions.FEATURES] for r in rows}
    matrix=torch.tensor([[1.]+v for v in unique.values()],dtype=torch.float64)
    rank=int(torch.linalg.matrix_rank(matrix));counts=defaultdict(set)
    for r in rows:counts[r['retained_processed_bands']].add((r['scene'],r['band']))
    require_one=all(len(v)==1 for v in counts.values())
    graph.require(all(len(v)==1 for v in signatures.values()),'There is independent same-count position variation; update identifiability assessment')
    table(out/'condition-identifiability.csv',rows)
    result=dict(status='passed',producer=code['commit'],program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=inputs,scenes=details,observed_condition_cells=len(unique),design_columns_with_intercept=9,design_rank=rank,
        processed_band_count_identifies_scene_and_view_in_this_manifest=require_one,
        same_scene_same_count_distinct_position_sets=0,within_scene_band_nontrivial_permutations=0,
        causal_position_effect_at_fixed_count_identifiable=False,physical_scale_effect_identifiable=False,
        index_condition_is_scene_view_confounded=True,new_labels=0,images_decoded=0,optimizer_updates=0,
        limits='Index count/position summaries can identify the scene/view; no independent same-count placement, authenticated wavelengths/FWHM/GSD or spatial release. Same-content full/retained donor swaps are stress tests, not causal coverage or sensor validation.')
    dev._json_new(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='inputs'}),flush=True)


def fixtures(out,code):
    graph.require(not out.exists(),'A fresh fixture campaign is required; completed stages are immutable')
    out.mkdir();started=time.perf_counter()
    header=dev._json_new(out/'registration.json',dict(producer=code['commit'],source_hash=code['source_hash'],
        classes=[9,22],seeds=[64,512,2048],cases=list(integration.CASES),cases_total=72,updates_per_route=12,
        routes=['full','prefix+resume'],maximum_synthetic_optimizer_updates=1728,maximum_real_optimizer_updates=0,
        checkpoint_clock=[0,6,12],prefix=5,selection='first_min_released_point_fused_nll',
        same_complete_initial_state_within_seed=True,common_candidates={'cnn':[1,2,3],'graph':[1,2]},
        purpose='engineering and falsifiable controls; no confirmation accuracy claim'))
    summaries=[];endpoints=[];perclass=[];participation=[];costs=[];controls=[];point_rows=[]
    with arithmetic(out,None) as execution:
        for classes in (9,22):
            provider=integration.FixtureProvider(classes);entries=integration.cache(provider,code['commit'])
            cachepin=save(out/f'fixture-{classes}-geometry.pt',{'entries':entries})
            initial_by_seed={}
            for seed in (64,512,2048):
                for arm in integration.CASES:
                    case=f'c{classes}-s{seed}-{arm}';dest=out/case;dest.mkdir()
                    spec=integration.contract(provider,seed,arm,code['commit'],execution.numeric,cachepin,{'synthetic':True,'registration':header})
                    specpin=dev._json_new(dest/'registration.json',spec);initial_by_seed.setdefault(seed,set()).add(spec['initialization']['state'])
                    states={};stages=[];t=time.perf_counter()
                    for phase in ('full','prefix','resume','completed'):
                        prior=states.get({'resume':'prefix','completed':'resume'}.get(phase));path=dest/(phase+'.pt')
                        dev._json_new(dest/(phase+'.intent.json'),dict(registration=specpin,phase=phase,maximum_updates={'full':12,'prefix':5,'resume':7,'completed':0}[phase]))
                        stage=phases.execute(spec,phase,path,provider,entries,code['commit'],execution,prior,run_factory=integration.ConditionRun)
                        saved=dev._json_new(dest/(phase+'.json'),dict(**stage,checkpoint=dev._pin(path),registration=specpin))
                        states[phase]=packet(read(saved,out)['checkpoint'],out);stages.append(stage)
                    graph.require(_exact(states['full'],states['resume']) and _exact(states['full'],states['completed']),'Continuous/split/completed packets differ')
                    graph.require(_exact(states['full']['history'][:5],states['prefix']['history']),'Prefix path differs')
                    audit=phases.audit(spec,states['full'],provider,entries,code['commit'],execution,cost=False,run_factory=integration.ConditionRun,point_context=('profile',))
                    auditpin=dev._json_new(dest/'independent.json',audit)
                    replay=integration.ConditionRun(spec,provider,entries,code['commit'],execution);replay.restore(states['full'],12)
                    replay.model.load_state_dict(states['full']['selected']['state']);replay.model.eval();before=replay.pack()
                    same_reference=None
                    with torch.no_grad():
                        for control in ('matched','missing','swap_band','unknown_wavelength','missing_gsd','permuted_profile'):
                            with provider.intervention(control):observed=replay.evaluate(audit['selected_step'],detailed=True)
                            if control=='matched':same_reference=observed
                            graph.require([(r['observation'],r['released'],r['graph']['range_identity']) for r in observed['rows']]==
                                [(r['observation'],r['released'],r['graph']['range_identity']) for r in same_reference['rows']],'Control changed input/released points/Q/A')
                            if arm.endswith(('.mean','.content')):
                                graph.require([r['heads'] for r in observed['rows']]==[r['heads'] for r in same_reference['rows']],'Condition-independent policy changed')
                            for head,bands in __import__('transfer.native_released_campaign',fromlist=['scoring']).scoring(observed,classes).items():
                                for band,m in bands.items():controls.append(dict(case=case,classes=classes,seed=seed,policy=arm,condition=control,head=head,band=band,**{k:m[k] for k in ('count','oa','aa','nll')}))
                        graph.require(_exact(before,replay.pack()),'Read-only condition controls changed model/optimizer/RNG')
                        # Same trained weights, all-missing reference must reduce to the content/mean policy.
                        missing_policy='content' if arm.endswith(('.content','.joint')) else 'mean'
                        scope=arm.split('.')[0];fallback=scope+'.'+missing_policy
                        with provider.intervention('missing'):missing=replay.evaluate(audit['selected_step'],detailed=True)
                        with replay.policy_intervention(fallback):fallback_value=replay.evaluate(audit['selected_step'],detailed=True)
                        graph.require([r['heads'] for r in missing['rows']]==[r['heads'] for r in fallback_value['rows']],'Missing-condition same-weight fallback differs')
                    for endpoint,data in audit['endpoints'].items():
                        for head,bands in data['metrics'].items():
                            for band,m in bands.items():
                                common=dict(case=case,classes=classes,seed=seed,policy=arm,endpoint=endpoint,head=head,band=band)
                                endpoints.append(dict(**common,**{k:m[k] for k in ('count','oa','aa','nll')}))
                                for j,c in enumerate(m['per_class']):perclass.append(dict(**common,class_id=c['class_id'],correct=m['confusion_matrix'][j][j],support=c['support'],recall=c['recall']))
                        for r in data['original_point_predictions']:
                            point_rows.append(dict(case=case,classes=classes,seed=seed,policy=arm,endpoint=endpoint,
                                point_id=digest((classes,r['tile'],r['released_ordinal']))[:20],head=r['head'],band=r['band'],class_id=r['class_id'],
                                profile=r['profile'],
                                p_true=r['probabilities'][r['class_id']-1],prediction=max(range(classes),key=r['probabilities'].__getitem__)+1))
                    grouped=defaultdict(list)
                    for event in states['full']['history']:
                        for p in event['parameter_participation']:grouped[p['tensor']].append(p)
                    for name,ps in grouped.items():participation.append(dict(case=case,classes=classes,seed=seed,policy=arm,tensor=name,numel=ps[0]['numel'],
                        steps_gradient_present=sum(p['gradient_present'] for p in ps),steps_nonzero_gradient=sum(p['nonzero_gradient_elements']>0 for p in ps),
                        steps_changed=sum(p['changed_elements']>0 for p in ps),nonzero_gradient_elements_step_sum=sum(p['nonzero_gradient_elements'] for p in ps)))
                    if seed==64:
                        for band in ('full','even_retained_positions'):
                            event=next(e for e in spec['validation'] if e['band']==band and e['profile']=='a')
                            actual,_,_=replay._materialize('devval',event);samples=[]
                            with torch.no_grad():
                                for i in range(7):
                                    t0=time.perf_counter();output,trace,_=replay._forward(actual,'devval','matched');dt=(time.perf_counter()-t0)*1000
                                    if i>=2:samples.append(dt)
                            costs.append(dict(case=case,classes=classes,policy=arm,band=band,device='CPU_one_thread',warmups=2,samples=5,
                                p50_ms=statistics.median(samples),p95_ms=float(np.percentile(samples,95)),allocated_parameters=spec['parameters'],
                                cnn_candidates=3,graph_candidates=2,near_nnz=trace['range_nnz'][0],wide_nnz=trace['range_nnz'][1],
                                scope='resident model + host view preparation/forward/frozen-QA/diagnostics/policy hooks; not CUDA kernel time'))
                    summaries.append(dict(case=case,registration=specpin,audit=auditpin,parameters=spec['parameters'],initial_state=spec['initialization']['state'],
                        continuous_split_completed_exact=True,selected_step=audit['selected_step'],synthetic_updates=24,real_updates=0,
                        read_only_condition_controls=6,model_optimizer_RNG_unchanged=True,elapsed_seconds=time.perf_counter()-t))
                    print(json.dumps({'case':case,'status':'passed','new_synthetic_updates':24}),flush=True)
            graph.require(all(len(v)==1 for v in initial_by_seed.values()),'Complete initial state changed across policies/scopes')
    for name,data in [('endpoints.csv',endpoints),('per-class.csv',perclass),('parameter-participation.csv',participation),
            ('resident-cost.csv',costs),('condition-controls.csv',controls),('points.csv',point_rows)]:table(out/name,data)
    summary=dict(status='passed',producer=code['commit'],source_hash=code['source_hash'],cases=summaries,
        cases_total=len(summaries),synthetic_optimizer_updates=sum(r['synthetic_updates'] for r in summaries),real_optimizer_updates=0,
        table_rows={n:len(r) for n,r in [('endpoints',endpoints),('per_class',perclass),('participation',participation),('cost',costs),('controls',controls),('points',point_rows)]},
        exact_full_split_completed_cases=len(summaries),read_only_controls_cases=len(summaries),
        effect_claim='engineering only; no significance or real-data condition efficacy',elapsed_seconds=time.perf_counter()-started)
    dev._json_new(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='cases'}),flush=True)


def main():
    graph.require_checks();torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True)
    sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('inventory');q.add_argument('--core',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('fixtures');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args();code=code_snapshot();require_frozen_code(code);graph.require(code['commit']==a.expected_commit,'Wrong clean analysis source')
    graph.require(Path(__file__).resolve().parents[1] not in a.out.resolve().parents,'Outputs must be outside checkout')
    if a.mode=='inventory':inventory(a.core,a.out,code)
    else:fixtures(a.out,code)


if __name__=='__main__':main()

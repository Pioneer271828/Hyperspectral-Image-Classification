"""Authenticate the complete condition supplement for WP12 without changing its existing tables."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.classification_metrics import classification_metrics
from research_runtime import code_snapshot,require_frozen_code
from transfer import registered_development as dev
from transfer import wp07_condition_run as engineering
from transfer import wp07_observed_conditions as cond
from transfer import wp07_fine16_training as frozen
from transfer.native_scale_campaign import read,packet
from transfer.multiscale_training import MultiscaleTrial
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact
from scripts.run_wp07_condition_integration import table


def pin(path):return dev._pin(Path(path).resolve(strict=True))


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','fixtures','inventory','real','base-wp12','core','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-commit',required=True);a=p.parse_args();torch.set_num_threads(1)
    graph=__import__('transfer.wp07_graph_range',fromlist=['require']);require=graph.require;graph.require_checks()
    code=code_snapshot();require_frozen_code(code);require(code['commit']==a.expected_commit,'Wrong evidence reader')
    require(a.root in a.out.parents and not a.out.exists(),'Fresh external supplement required')
    fixture=read(pin(a.fixtures/'summary.json'),a.root);inventory=read(pin(a.inventory/'summary.json'),a.root);real=read(pin(a.real/'summary.json'),a.root)
    require(fixture['status']==real['status']==inventory['status']=='passed' and fixture['cases_total']==72
        and fixture['synthetic_optimizer_updates']==1728 and fixture['real_optimizer_updates']==real['optimizer_updates']==0
        and real['models']==7 and real['condition_parameters_trained'] is False,'Partial matrix or promoted real scope')
    a.out.mkdir();baseline=a.out/'unchanged-base';baseline.mkdir();basepins={}
    for path in sorted(a.base_wp12.glob('*.csv')):
        target=baseline/path.name;shutil.copyfile(path,target)
        require(pin(path)['sha256']==pin(target)['sha256'],'Existing WP12 table bytes changed');basepins[path.name]=pin(target)
    require(len(basepins)>=5,'Missing prior B WP12 base tables')
    expected={(c,s,mode) for c in (9,22) for s in (64,512,2048) for mode in engineering.CASES};seen=set()
    metrics=[];recalls=[];participation=[];initial_states={};caseinitial={};phase_updates=0;sourcepins=[]
    with torch.no_grad():
        for case in fixture['cases']:
            spec=read(case['registration'],a.root);audit=read(case['audit'],a.root);c,s,mode=spec['classes'],spec['seed'],spec['arm'];seen.add((c,s,mode))
            require((c,s,mode) in expected and spec['producer']==fixture['producer'] and spec['steps']==12 and spec['prefix']==5
                and spec['events']==[0,6,12] and spec['synthetic'],'Foreign fixture/selection protocol')
            dest=Path(case['registration']['path']).parent;states={}
            for phase,amount in [('full',12),('prefix',5),('resume',7),('completed',0)]:
                r=read(pin(dest/(phase+'.json')),a.root)
                require(r['registration']==case['registration'] and r['updates']==amount and r['phase']==phase,'Phase/registration mismatch')
                states[phase]=packet(r['checkpoint'],a.root);phase_updates+=amount
            require(_exact(states['full'],states['resume']) and _exact(states['full'],states['completed'])
                and _exact(states['full']['history'][:5],states['prefix']['history']),'Independent full/split/completed verification failed')
            require(states['full']['selected']['step']==min(states['full']['validation'],key=lambda r:r['fused_nll'])['step']==audit['selected_step'],
                'First-min selected endpoint differs')
            if (c,s) not in initial_states:
                provider=engineering.FixtureProvider(c);model=cond.initialize(provider.baseline(s),s+7321)
                require(state_hash(model)==spec['initialization']['state'],'CPU-only initializer fix changed the archived production initial tensors')
                initial_states[c,s]={n:v.clone() for n,v in model.state_dict().items()}
            caseinitial.setdefault((c,s),set()).add(spec['initialization']['state'])
            final=states['full']['model'];history=defaultdict(list)
            for step in states['full']['history']:
                for part in step['parameter_participation']:history[part['tensor']].append(part)
            for name,parts in history.items():
                changed=int(torch.count_nonzero(final[name]-initial_states[c,s][name]))
                group=('index_encoder' if '.indices.' in name else 'verified_physical_encoder' if '.physical.' in name else
                    'condition_head' if any(x in name for x in ('condition_selector.','range_metadata.')) else
                    'content_selector' if any(x in name for x in ('content_selector.','range_content.')) else
                    'graph_feature_norm' if 'learnable_graph.feature_norm.' in name else
                    'frozen_assignment_edge_scorer' if any(x in name for x in ('learnable_graph.pixel_query.','learnable_graph.seed_key.','learnable_graph.graph_query.','learnable_graph.graph_key.')) else 'shared_source')
                participation.append(dict(case=case['case'],classes=c,seed=s,policy=mode,tensor=name,group=group,
                    allocated_elements=parts[0]['numel'],final_changed_elements=changed,
                    steps_with_nonzero_gradient=sum(r['nonzero_gradient_elements']>0 for r in parts),
                    nonzero_gradient_element_step_sum=sum(r['nonzero_gradient_elements'] for r in parts)))
            for endpoint,data in audit['endpoints'].items():
                by=defaultdict(list)
                for r in data['original_point_predictions']:
                    require('profile' in r,'Synthetic profile identity missing');by[r['head'],r['band']].append(r)
                for (head,band),points in by.items():
                    labels=[r['class_id'] for r in points];predictions=[max(range(c),key=r['probabilities'].__getitem__)+1 for r in points]
                    reconstructed=classification_metrics(labels,predictions,list(range(1,c+1)))
                    saved=data['metrics'][head][band]
                    matrix=[[0]*c for _ in range(c)]
                    for t,y in zip(labels,predictions):matrix[t-1][y-1]+=1
                    require(reconstructed['confusion_matrix']==matrix and abs(reconstructed['oa']-saved['oa'])<=1e-15
                        and abs(reconstructed['aa']-saved['aa'])<=1e-15,'Existing WP12 classification reader disagrees')
                    nll=math.fsum(-math.log(max(r['probabilities'][r['class_id']-1],1e-8)) for r in points)/len(points)
                    require(matrix==saved['confusion_matrix'] and abs(nll-saved['nll'])<=1e-12 and saved['count']==len(points),
                        'Independent original-point confusion/NLL differs')
                    common=dict(case=case['case'],classes=c,seed=s,policy=mode,endpoint=endpoint,head=head,band=band,
                        evidence_scope='synthetic_engineering',condition_parameters_trained=True,checkpoint_producer=spec['producer'])
                    metrics.append(dict(**common,count=len(points),oa=saved['oa'],aa=saved['aa'],nll=nll))
                    for i in range(c):
                        support=sum(matrix[i]);recalls.append(dict(**common,class_id=i+1,correct=matrix[i][i],support=support,recall=matrix[i][i]/support if support else None))
            sourcepins.extend([case['registration'],case['audit']])
    require(seen==expected and phase_updates==1728 and all(len(v)==1 for v in caseinitial.values()),'Missing cases, changed initial state or spent budget differs')
    # Mean scopes are explicit same-policy negative wiring controls, not independent scientific arms.
    for c in (9,22):
        for s in (64,512,2048):
            groups=[next(r for r in fixture['cases'] if r['case']==f'c{c}-s{s}-{scope}.mean') for scope in ('both','cnn','graph')]
            data=[packet(read(pin(Path(r['registration']['path']).parent/'completed.json'),a.root)['checkpoint'],a.root) for r in groups]
            require(all(_exact(data[0][k],v[k]) for v in data[1:] for k in ('model','optimizer','rng')),'Mean-scope negative controls differ')
    index={(r['classes'],r['seed'],r['policy'],r['endpoint'],r['head'],r['band']):r for r in metrics};pairs=[]
    classindex={(r['classes'],r['seed'],r['policy'],r['endpoint'],r['head'],r['band'],r['class_id']):r for r in recalls}
    for c in (9,22):
        for s in (64,512,2048):
            for scope in cond.SCOPES:
                for candidate,baseline_mode in [('joint','content'),('metadata','mean'),('content','mean'),('joint','mean')]:
                    for head in ('fused','graph','cnn'):
                        for band in ('full','even_retained_positions'):
                            x,y=(index[c,s,scope+'.'+m,'selected',head,band] for m in (candidate,baseline_mode))
                            protected=all(classindex[c,s,scope+'.'+candidate,'selected',head,band,cl]['recall']-
                                classindex[c,s,scope+'.'+baseline_mode,'selected',head,band,cl]['recall']>=-.02 for cl in range(1,c+1))
                            pairs.append(dict(classes=c,seed=s,scope=scope,candidate=candidate,baseline=baseline_mode,head=head,band=band,
                                count=x['count'],oa_delta=x['oa']-y['oa'],nll_delta=x['nll']-y['nll'],all_classes_protected=protected,
                                evidence_scope='synthetic_engineering_not_real_accuracy'))
    # Reconstruct each real stress table from actual saved point probabilities; no classifier forward.
    real_metrics=[];real_classes=[];real_initial_hashes=[]
    for item in real['cases']:
        data=read(item,a.root);require(data['optimizer_updates']==0 and not data['condition_parameters_trained'],'Promoted real condition-training claim')
        for record in data['records']:
            group=defaultdict(list)
            for r in record['points']:group[r['head'],r['band']].append(r)
            for (head,band),points in group.items():
                require(len(points)==45,'Real development denominator differs');cm=[[0]*9 for _ in range(9)]
                for r in points:cm[r['class_id']-1][max(range(9),key=r['probabilities'].__getitem__)]+=1
                nll=math.fsum(-math.log(max(r['probabilities'][r['class_id']-1],1e-8)) for r in points)/45
                m=record['metrics'][head][band]
                require(cm==m['confusion_matrix'] and abs(nll-m['nll'])<=1e-12,'Real stress recount differs')
        # CPU re-create the metadata initializer: no new scoring and no dependency on GPU RNG reseeding.
        checkpoint=data['checkpoint'];spec=read(pin(Path(checkpoint['path']).parent/'registration.json'),a.root)
        basename=Path(item['path']).stem;scene='paviaU' if '-paviaU-' in basename else 'xuzhou';native=basename.startswith('native_factor2')
        if native:preflight=spec['preflight']
        else:preflight=read(pin(a.core/scene/'pair/plan.json'),a.root)['preflight']
        provider=frozen.BankProvider(a.root,read(preflight,a.root),spec['seed'])
        model=MultiscaleTrial(cnn_multiscale='content' if native else 'fixed',graph_multiscale='content').initialize(provider.baseline(spec['seed']),spec['module_seed'])
        state=packet(checkpoint,a.root);model.load_state_dict(state['selected']['state'],strict=True)
        cond.install_encoders(model,data['metadata_initialization_seed'])
        require(state_hash(model)==data['random_adapter_initial_state'],'RNG-isolated initializer changed a real stress model tensor')
        real_initial_hashes.append(dict(case=basename,state=state_hash(model),exact=True,classifier_forwards=0))
    real_metrics=rows(a.real/'endpoints.csv');real_classes=rows(a.real/'per-class.csv')
    for name,data in [('synthetic-endpoints.csv',metrics),('synthetic-classes.csv',recalls),('synthetic-pairs.csv',pairs),
            ('parameter-participation.csv',participation),('real-untrained-condition-endpoints.csv',real_metrics),('real-untrained-condition-classes.csv',real_classes)]:table(a.out/name,data)
    for src,name in [(a.fixtures/'resident-cost.csv','synthetic-resident-cost.csv'),(a.real/'resident-cost.csv','real-neutral-graft-cost.csv'),
            (a.inventory/'condition-identifiability.csv','condition-identifiability.csv')]:shutil.copyfile(src,a.out/name)
    outputs={p.name:pin(p) for p in sorted(a.out.glob('*.csv'))}
    draft=dev._json_new(a.out/'next-real-pairing-draft.json',dict(schema='wp07_index_position_equal_count_draft_v1',
        status='draft_not_authorized_not_executed',execution_authorized=False,ready_for_release=False,
        question='Processed-band position-conditioned development fit at equal retained count; no physical sensor or GSD claim',
        scene='xuzhou',parent_processed_bands=436,conditions={'even':list(range(0,436,2)),'odd':list(range(1,436,2))},
        retained_bands_each=218,odd_condition_currently_published=False,
        support_seed=512,validation_seed=3031,classes=9,train_points=45,validation_points=45,points_per_class_per_role=5,
        train_blocks=73,labelled_train_blocks=31,validation_blocks=5,new_labels=0,query_authorized=False,raw_GT_authorized=False,
        seeds=[64,512,2048],policies=['mean','content','metadata','joint'],selector_scope='both',
        model_cases=12,steps_per_case=62,split=[31,31,0],total_real_updates_requested_only_as_draft=744,
        training_exposure='Each of 31 original labelled train blocks once per even/odd condition; identical order and initial tensors within seed',
        candidates={'CNN_shared_kernel_dilations':[1,2,3],'graph_same_Q_radii':[1,2],'graph_scale':16},
        loss='unchanged original three heads',optimizer='Adam 1e-4',dropout=0,
        selection={'events':[0,31,62],'rule':'first minimum fused released-point NLL','scope':'same V5 development; not independent confirmation'},
        primary='joint-content: lower fused NLL in all three model seeds and both equal-count conditions; every head/class recall delta >= -0.02',
        secondary=['metadata-mean','content-mean','joint-mean','same-content even/odd selector descriptor swap','all-missing fallback'],
        precision_limit='One class error is 1/5=20 percentage points; no class loss is allowed by the 2-point recall guard',
        estimated_GPU3_wall_minutes=[10,25],estimate_basis='Original 62-step real protocol plus measured candidate/condition overhead; includes registration, three validations, independent audit and checkpoint IO; contention separate',
        physical_gpu='GPU-2cbff259-d7db-32cf-b267-8f296e34b1d5',
        release_requirements=['Main separately authorizes exactly this new real budget and odd-band materialization',
            'New hash-pinned odd/even preflight, metadata and frozen-QA manifest; no score-driven choice of positions',
            'Unlabelled same-count descriptor separability and distinct graph ranges pass before labels are evaluated',
            'Same-capacity complete initial state and aligned source/loss/exposure/selection contracts pinned',
            'Reviewed real-provider adapter and original phase/controller integration pass recovery and negative qualification',
            'Physical GPU lease/arithmetic qualification and backup custody checked; rmvision switch needs reviewed relocation and numerical qualification'],
        current_execution_guard='ConditionRun.contract rejects provider.synthetic=False; this JSON is not an executable release',
        remaining_unidentified=['Authenticated original sensor mapping, FWHM, actual GSD and sensor-held-out conditions',
            'Independent spatial confirmation; three model seeds still share one released support/validation set']))
    receipt=dict(schema='wp12_wp07_observed_index_condition_supplement_v1',status='accepted_engineering_and_development_diagnostic',
        executor=code['commit'],source_hash=code['source_hash'],inputs={'fixture':pin(a.fixtures/'summary.json'),'inventory':pin(a.inventory/'summary.json'),'real':pin(a.real/'summary.json')},
        next_real_pairing_draft=draft,
        unchanged_prior_B_WP12_tables=basepins,artifacts=outputs,cases=72,distinct_policy_pairs_per_seed=10,
        mean_scope_aliases_are_negative_wiring_controls=True,synthetic_optimizer_updates=1728,new_real_optimizer_updates=0,
        independent_fixture_confusion_matrices=len(metrics),independent_real_confusion_matrices=len(real_metrics),NLL_tolerance=1e-12,
        CPU_initializer_matches_all_six_archived_fixture_initial_states=True,CPU_initializer_matches_seven_prior_real_stress_models=real_initial_hashes,
        condition_identifiability=inventory['limits'],real_condition_parameters_trained=False,
        physical_sensor_gain=None,confirmation_accuracy_gain=None,global_WP12_production_registry_modified=False,
        scope_limit='Private append-only B supplement; existing classification reader reused, baseline table bytes retained; central WP12 registration remains integration-owner work',
        raw_GT_opened=False,query_accessed=False)
    dev._json_new(a.out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','artifacts','unchanged_prior_B_WP12_tables','CPU_initializer_matches_seven_prior_real_stress_models')}),flush=True)


if __name__=='__main__':main()

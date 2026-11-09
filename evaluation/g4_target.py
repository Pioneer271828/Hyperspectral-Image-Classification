"""All-model query barrier and metrics for the frozen G4 target study."""
import json,math
from pathlib import Path
import numpy as np
import torch
from joint_baseline import NAMES
from joint_training import schedule_for,_tree_fingerprint,PREPARATION_HASH
from loadData.dataset_registry import file_hash,DATASETS
from model.research_config import digest
from target_initialization import METHODS
from target_study import EXPECTED,select_case,validate_source_case,TargetStudyRuntime
from target_data_context import load_target_context
from target_training import features
from evaluation.classification_metrics import classification_metrics
from evaluation.branch_diagnostics import branch_diagnostics
from research_runtime import external_directory,write_json
from loadData.target_label_protocol import final_query
from source_data_context import bounded

TRAINING_COMMIT='85801647e4824ea4adadebbde3394a38f75cb62c'
HEADS=('fused','graph','cnn')
POLICY='g4_all27_frozen_query_v1'


def read_json(path):return json.loads(Path(path).read_text())


def check_files(pins):
    if any(file_hash(p)!=h for p,h in pins.items()):raise ValueError('Frozen training/input artifact changed')


def verify_training(directory,launch_sha256,acceptance_sha256,verification_source):
    """No GT reader here: every model must pass before any label release."""
    root=Path(directory).resolve(strict=True);lp=root/'launch.json';ap=root/'training-acceptance.json'
    if file_hash(lp)!=launch_sha256 or file_hash(ap)!=acceptance_sha256:raise ValueError('Pinned full training acceptance/launch differs')
    launch=read_json(lp);accepted=read_json(ap);producer=launch['code']
    if (producer.get('commit')!=TRAINING_COMMIT or producer.get('dirty') or not producer.get('files')
            or producer.get('source_hash')!=digest(producer['files']) or verification_source.get('dirty')
            or verification_source.get('source_hash')!=digest(verification_source['files'])
            or any(verification_source['files'].get(k)!=v for k,v in producer['files'].items())):
        raise ValueError('Expected unchanged full training implementation and additive verifier')
    if (accepted.get('status')!='passed' or accepted.get('source')!=producer
            or accepted.get('launch_sha256')!=launch_sha256 or accepted.get('optimizer_updates')!=40500
            or accepted.get('query_labels_accessed') is not False or accepted.get('test_scoring') is not False
            or len(accepted.get('models',[]))!=27 or len({m['name'] for m in accepted['models']})!=27
            or Path(launch['out_dir']).resolve()!=root):raise ValueError('All27 frozen accepted models required')
    models={m['name']:m for m in accepted['models']};pins={str(lp):launch_sha256,str(ap):acceptance_sha256};rows=[]
    for case in launch['cases']:
        select_case(launch,producer,case['target'],case['seed'],case['method'])
        if case['name'] not in models:raise ValueError('Frozen model coverage incomplete')
        frozen=models[case['name']];folder=Path(case['out']);rp=folder/'result.json';review=root/'reviews'/(case['name']+'.json')
        if (file_hash(rp)!=frozen['result_sha256'] or file_hash(review)!=frozen['review_sha256']
                or file_hash(folder/'primary.pt')!=frozen['primary_sha256']
                or file_hash(folder/'primary.pt.json')!=frozen['sidecar_sha256']):raise ValueError('Frozen selected model/result changed')
        result=read_json(rp);identity=result['identity'];settings=case['settings']
        if (result.get('status')!='completed' or result.get('optimizer_steps')!=1500 or result.get('start_code')!=producer
                or result.get('settings')!=settings or result.get('test_scoring') is not False
                or result.get('selected_step')!=frozen['selected_step'] or result.get('per_dataset_updates')!={case['target']:1500}
                or identity.get('settings')!=settings or identity.get('settings_hash')!=digest(settings)
                or identity.get('consumer_source_hash')!=producer['source_hash']
                or identity.get('prepared')!=case['provenance'] or identity.get('execution_device')!='cuda'
                or result.get('schedule_sha256')!=digest(schedule_for(settings,task_names=(case['target'],)))):
            raise ValueError('Completed target budget/settings/producer identity differs')
        required={'identity.json','progress.json','primary.pt','primary.pt.json','last.pt','last.pt.json',
                  'selected-probabilities.pt','trace.json','history.json'}
        artifacts=result['artifacts']
        if not required<=set(artifacts):raise ValueError('Completed artifacts incomplete')
        for name,sha in artifacts.items():
            if Path(name).name!=name:raise ValueError('Unsafe artifact name')
            pins[str(folder/name)]=sha
        pins[str(rp)]=frozen['result_sha256'];pins[str(review)]=frozen['review_sha256']
        trace=read_json(folder/'trace.json');history=read_json(folder/'history.json');progress=read_json(folder/'progress.json')
        if (len(trace)!=1500 or [x['step'] for x in trace]!=list(range(1,1501))
                or [x['dataset'] for x in trace]!=[case['target']]*1500
                or [x['step'] for x in history]!=list(range(1,1472,30))+[1500]
                or any(not math.isfinite(x['val_loss']) or x['selection_score']!=-x['val_loss'] for x in history)
                or min(history,key=lambda x:x['val_loss'])['step']!=result['selected_step']
                or progress['status']!='completed' or progress['cursor']!=1500
                or Path(progress['state_file']).name!=progress['state_file']
                or artifacts.get(progress['state_file'])!=progress['state_sha256']):
            raise ValueError('Full target trajectory/selection/state publication differs')
        review_value=read_json(review)
        if (review_value['status']!='completed' or review_value['updates']!=1500
                or review_value['selected_step']!=result['selected_step']
                or review_value['artifacts']!={**artifacts,'result.json':frozen['result_sha256']}
                or review_value['completed_resume_updates']!=0):raise ValueError('Post-training review differs')
        # Authenticated source result plus sanitized target loading happen in the
        # native audit too. These pins block mutations before any GT decoding.
        pins.update(case['provenance']['protected_files'])
        rows.append({'case':case,'result':result,'frozen':frozen})
    check_files(pins)
    return {'launch':launch,'acceptance':accepted,'rows':rows,'protected_files':pins,'query_labels_read':False}


def load_case(row,*,device):
    case=row['case'];validate_source_case(case);i=case['inputs']
    items,provenance,guard=load_target_context(case['target'],i['observation'],i['record'],i['record_sha256'],
        i['family'],i['family_sha256'],per_class=10,support_seed=64)
    runtime=TargetStudyRuntime(case['settings'],items,case['out'],source_spec=case['source_spec'],
        initialization_state_sha256=case['initial_state_sha256'])
    folder=Path(case['out']);step=row['result']['selected_step']
    model=runtime.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{step:06d}.pt',device=device)
    meta=model.training_config
    if (meta['training_code']!=row['result']['start_code'] or meta['provenance']!=case['provenance']
            or meta['checkpoint_optimizer_step']!=step):raise ValueError('Selected model training metadata differs')
    guard();return model,runtime,items


def independent_trainval(values,item,expected):
    """NumPy oracle using released labels; no query labels accepted."""
    count=recalls=0
    for phase in ('train','val'):
        labels=item[phase+'_labels'].cpu().numpy();classes=item['class_num']
        for head in HEADS:
            value=values[phase][head].cpu().numpy();prediction=value.argmax(1)
            cm=np.zeros((classes,classes),np.int64);np.add.at(cm,(labels,prediction),1)
            recall=cm.diagonal()/cm.sum(1)
            score=[cm.trace()/cm.sum(),recall.mean(),-np.log(np.clip(value[np.arange(len(labels)),labels],1e-8,1)).mean()]
            record=expected[phase][head]
            if cm.tolist()!=record['confusion']:raise ValueError('Saved train/val confusion differs')
            np.testing.assert_allclose(record['recall'],recall,rtol=0,atol=1e-12)
            np.testing.assert_allclose([record['oa'],record['aa'],record['nll']],score,rtol=1e-5,atol=1e-6)
            count+=1;recalls+=classes
    return {'head_metrics':count,'class_recalls':recalls}


def audit_native_case(row,*,device):
    model,runtime,items=load_case(row,device=device);case=row['case'];folder=Path(case['out']);target=case['target']
    progress=read_json(folder/'progress.json');state=torch.load(folder/progress['state_file'],map_location='cpu',weights_only=True)
    if (state['cursor']!=1500 or state['identity']!=row['result']['identity'] or state['start_code']!=row['result']['start_code']
            or state['trace']!=read_json(folder/'trace.json') or state['history']!=read_json(folder/'history.json')
            or state['best_step']!=row['result']['selected_step']
            or state['best_score']!=-min(h['val_loss'] for h in state['history'])
            or any(_tree_fingerprint(state[k])!=state['state_fingerprints'][k] for k in ('model','optimizer','scheduler'))
            or _tree_fingerprint(model.state_dict())!=_tree_fingerprint(state['best_model'])):
        raise ValueError('Authoritative training state differs from frozen selected model')
    raw_last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    if _tree_fingerprint(raw_last)!=_tree_fingerprint(state['model']):raise ValueError('Saved last state differs')
    stored=torch.load(folder/'selected-probabilities.pt',map_location='cpu',weights_only=True)
    if _tree_fingerprint(stored)!=_tree_fingerprint(state['best_probabilities']):raise ValueError('Saved selected probability record differs')
    _,values=runtime.validation_snapshot(model,items);maximum=0.
    for phase in ('train','val'):
        for head in HEADS:
            old,new=stored[target][phase][head],values[target][phase][head]
            torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-6)
            if not torch.equal(old.argmax(1),new.argmax(1)):raise ValueError('Native selected train/val classification changed')
            maximum=max(maximum,float((new-old).abs().max()))
    checks=independent_trainval(stored[target],items[target],row['result']['selected_validation']['metrics'][target])
    del model,runtime,items,state,raw_last,stored,values
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return {'name':case['name'],'status':'passed','probability_max_abs_error':maximum,**checks,'query_labels_read':False}


def predict_query(model,item,indices):
    model.eval()
    if any(m.training for m in model.modules()):raise ValueError('Query inference must be fully eval')
    device=next(model.parameters()).device;ids=torch.as_tensor(indices,dtype=torch.long,device=device)
    if ids.ndim!=1 or not len(ids) or len(torch.unique(ids))!=len(ids):raise ValueError('Distinct query rows required')
    with torch.no_grad():
        vectors=features(model,item['name'],item)
        if int(ids.min())<0 or int(ids.max())>=vectors[0].shape[0]:raise ValueError('Query row outside observation')
        probabilities=model.predict_features(item['name'],*(v[ids] for v in vectors))
    result={h:p.detach().cpu() for h,p in zip(HEADS,probabilities)}
    for p in result.values():
        if (not torch.isfinite(p).all() or (p<0).any() or (p>1).any()
                or not torch.allclose(p.sum(1),torch.ones_like(p[:,0]),rtol=0,atol=2e-6)):
            raise ValueError('Invalid query probabilities')
    return result


def query_metrics(labels,probabilities):
    labels=np.asarray(labels);classes=next(iter(probabilities.values())).shape[1];ids=list(range(1,classes+1))
    if (labels.ndim!=1 or not len(labels) or labels.dtype.kind not in 'iu'
            or (labels<1).any() or (labels>classes).any() or set(probabilities)!=set(HEADS)):
        raise ValueError('Foreground integer query labels and three heads required')
    metrics={};predictions={}
    for head in HEADS:
        p=np.asarray(probabilities[head],dtype=np.float64)
        if (p.shape!=(len(labels),classes) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
                or not np.allclose(p.sum(1),1,rtol=0,atol=2e-6)):raise ValueError('Malformed query probability matrix')
        pred=p.argmax(1)+1;predictions[head]=pred
        metric=classification_metrics(labels,pred,ids)
        if not metric['full_class_coverage']:raise ValueError('Frozen query must cover every target class')
        metric['nll']=float(-np.log(np.clip(p[np.arange(len(labels)),labels.astype(int)-1],1e-8,1)).mean())
        metrics[head]=metric
    branch=branch_diagnostics(labels,predictions['cnn'],predictions['graph'],predictions['fused'],ids)
    branch['per_class']={str(c):branch_diagnostics(labels,predictions['cnn'],predictions['graph'],predictions['fused'],ids,mask=labels==c) for c in ids}
    return metrics,branch


def paired_summary(rows):
    lookup={(r['target'],r['seed'],r['method']):r for r in rows}
    expected={(t,s,m) for t in NAMES for s in (64,512,2048) for m in METHODS}
    if len(rows)!=27 or set(lookup)!=expected:raise ValueError('All27 paired query rows required')
    for row in rows:
        classes=list(range(1,DATASETS[row['target']][2]+1));metric=row['metrics']['fused']
        if (metric['class_ids']!=classes or [c['class_id'] for c in metric['per_class']]!=classes
                or any(c['recall'] is None for c in metric['per_class'])
                or not all(math.isfinite(metric[k]) for k in ('oa','aa','kappa','nll'))):
            raise ValueError('Complete finite target metrics required')
    def stats(values):return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),'values':list(map(float,values))}
    by_target={};macro={};seeds=(64,512,2048)
    for method in METHODS:
        by_target[method]={t:{k:stats([lookup[t,s,method]['metrics']['fused'][k] for s in seeds]) for k in ('oa','aa','kappa','nll')} for t in NAMES}
        macro[method]={k:stats([float(np.mean([lookup[t,s,method]['metrics']['fused'][k] for t in NAMES])) for s in seeds]) for k in ('oa','aa','kappa','nll')}
    contrasts={}
    for method in METHODS[1:]:
        delta=[macro[method]['oa']['values'][i]-macro[METHODS[0]]['oa']['values'][i] for i in range(3)]
        aa={t:by_target[method][t]['aa']['mean']-by_target[METHODS[0]][t]['aa']['mean'] for t in NAMES}
        hong=by_target[method]['whu-honghu']['oa']['mean']-by_target[METHODS[0]]['whu-honghu']['oa']['mean']
        recalls={}
        for t in NAMES:
            count=len(lookup[t,64,method]['metrics']['fused']['per_class'])
            recalls[t]=[float(np.mean([lookup[t,s,method]['metrics']['fused']['per_class'][i]['recall']-
                lookup[t,s,METHODS[0]]['metrics']['fused']['per_class'][i]['recall'] for s in seeds])) for i in range(count)]
        gates={'positive_paired_macro_mean':float(np.mean(delta))>0,'all3_macro_positive':all(d>0 for d in delta),
            'all_target_aa_protection':all(d>=-.01-1e-12 for d in aa.values()),'honghu_oa_protection':hong>=-.01-1e-12,
            'all_class_recall_protection':all(d>=-.02-1e-12 for ds in recalls.values() for d in ds)}
        contrasts[method]={'paired_macro_oa_delta':stats(delta),'target_mean_aa_delta':aa,'honghu_mean_oa_delta':hong,
            'per_class_mean_recall_delta':recalls,'gates':gates,'passes_all_gates':all(gates.values()),
            'role':'primary' if method==METHODS[1] else 'descriptive_control'}
    return {'by_target':by_target,'macro':macro,'contrasts':contrasts,
        'training_seeds':list(seeds),'support_seed':64,'support_repetition_claim':False,'significance_claim':False}


def freeze_for_query(training_directory,launch_sha256,acceptance_sha256,verification_source,out_dir,*,device):
    """The sole production label-release predecessor; never reads raw query GT."""
    out=external_directory(out_dir);out.mkdir(exist_ok=False)
    try:
        stage=verify_training(training_directory,launch_sha256,acceptance_sha256,verification_source)
        checks=[]
        for row in stage['rows']:
            checks.append(audit_native_case(row,device=device))
            print('G4_PREQUERY_NATIVE_PASSED',row['case']['name'],flush=True)
        if len(checks)!=27 or any(r['status']!='passed' for r in checks):raise ValueError('Every native prequery check required')
        check_files(stage['protected_files'])
        model_cases={r['case']['name']:r['case'] for r in stage['rows']}
        bound_models=[{**m,'case_sha256':digest(model_cases[m['name']]),
            'family_hash':model_cases[m['name']]['provenance']['family_hash'],
            'target':model_cases[m['name']]['target'],'seed':model_cases[m['name']]['seed'],
            'method':model_cases[m['name']]['method']} for m in stage['acceptance']['models']]
        freeze={'policy':POLICY,'status':'frozen_before_query','verification_code':verification_source,
            'training_code':stage['launch']['code'],'launch_sha256':launch_sha256,
            'training_acceptance_sha256':acceptance_sha256,'training_directory':str(Path(training_directory).resolve()),
            'protected_files':stage['protected_files'],'models':bound_models,
            'native_checks':checks,'query_labels_read':False,'optimizer_updates':0,
            'head_metric_cases':sum(r['head_metrics'] for r in checks),'class_recalls':sum(r['class_recalls'] for r in checks)}
        write_json(out/'freeze.json',freeze)
        return stage,freeze
    except Exception as error:
        write_json(out/'prequery-failure.json',{'status':'failed','error':repr(error),'query_labels_read':False,'optimizer_updates':0})
        raise


def require_frozen_boundary(freeze):
    """Artifact integrity contract, not a filesystem security sandbox."""
    if (freeze.get('policy')!=POLICY or freeze.get('status')!='frozen_before_query'
            or freeze.get('training_code',{}).get('commit')!=TRAINING_COMMIT
            or len(freeze.get('models',[]))!=27 or len({r['name'] for r in freeze['models']})!=27
            or {(r['target'],r['seed'],r['method']) for r in freeze['models']}!=
                {(t,s,m) for t in NAMES for s in (64,512,2048) for m in METHODS}
            or len(freeze.get('native_checks',[]))!=27
            or {r['name'] for r in freeze['models']}!={r['name'] for r in freeze['native_checks']}
            or any(r.get('status')!='passed' for r in freeze['native_checks'])
            or freeze.get('head_metric_cases')!=162 or freeze.get('class_recalls')!=2160
            or freeze.get('query_labels_read') is not False or freeze.get('optimizer_updates')!=0):
        raise ValueError('Full27 native-verified query boundary required')
    check_files(freeze['protected_files'])


def read_query_labels(freeze,case,data_root,prepared_manifest,prepared_sha256):
    """Only called after the global native barrier; authenticates actual GT/weight."""
    require_frozen_boundary(freeze)
    model=next(m for m in freeze['models'] if m['name']==case['name'])
    if digest(case)!=model['case_sha256'] or case['provenance']['family_hash']!=model['family_hash']:
        raise ValueError('Query case/family is outside the frozen model manifest')
    if prepared_sha256!=PREPARATION_HASH or file_hash(prepared_manifest)!=prepared_sha256:
        raise ValueError('Accepted cohort metadata required for final labels')
    manifest=read_json(prepared_manifest);target=case['target'];raw=manifest['datasets'][target]['raw']
    if manifest['protocol']!='acc1p_joint_prepared_v1' or raw['class_count']!=DATASETS[target][2]:
        raise ValueError('Final label cohort/target differs')
    family_path=case['inputs']['family']
    if file_hash(family_path)!=case['inputs']['family_sha256']:raise ValueError('Query label family changed')
    family=read_json(family_path)
    if family['family_hash']!=case['provenance']['family_hash']:raise ValueError('Frozen support/query family differs')
    weight_path=Path(case['out'])/'primary.pt'
    actual_weight_sha=file_hash(weight_path)
    if actual_weight_sha!=model['primary_sha256']:raise ValueError('Actual frozen query checkpoint changed')
    path=bounded(Path(data_root).resolve(strict=True),raw['label_path'])
    if file_hash(path)!=raw['label_sha256']:raise ValueError('Final raw label file changed')
    from scipy.io import loadmat
    key=DATASETS[target][1]+'_gt';labels=loadmat(path,variable_names=[key])[key]
    release=final_query(family,labels,expected_hash=family['family_hash'],checkpoint_sha256=actual_weight_sha)
    if file_hash(path)!=raw['label_sha256']:raise ValueError('Raw labels changed during final release')
    return release,{str(path):raw['label_sha256']}

"""Independent numerical oracle for saved G4 predictions; no fitting or selection."""
import json,math,statistics
from pathlib import Path
import numpy as np
import torch
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS,file_hash
from loadData.fixed_splits import label_hash
from model.research_config import digest
from source_data_context import bounded
from target_initialization import METHODS

HEADS=('fused','graph','cnn');SEEDS=(64,512,2048)


def oracle_metrics(labels,probability):
    truth=np.asarray(labels);p=np.asarray(probability,dtype=np.float64)
    if (truth.ndim!=1 or truth.dtype.kind not in 'iu' or p.ndim!=2 or p.shape[0]!=len(truth)
            or not len(truth) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or (truth<1).any() or (truth>p.shape[1]).any()
            or not np.allclose(p.sum(1),1,rtol=0,atol=2e-6)):
        raise ValueError('Malformed saved query predictions/truth')
    classes=p.shape[1];pred=p.argmax(1)+1;cm=np.zeros((classes,classes),np.int64)
    np.add.at(cm,(truth-1,pred-1),1)
    supports=[int(sum(cm[c])) for c in range(classes)]
    predicted=[int(sum(cm[:,c])) for c in range(classes)]
    if not all(supports):raise ValueError('Every query class needs support')
    count=len(truth);correct=sum(int(cm[c,c]) for c in range(classes));oa=correct/count
    expected=sum(supports[c]*predicted[c] for c in range(classes))/(count*count)
    per_class=[]
    for c in range(classes):
        tp=int(cm[c,c]);per_class.append({'class_id':c+1,'support':supports[c],'predicted_count':predicted[c],
            'precision':tp/predicted[c] if predicted[c] else None,'recall':tp/supports[c],
            'f1':2*tp/(supports[c]+predicted[c])})
    aa=statistics.fmean(r['recall'] for r in per_class);mf1=statistics.fmean(r['f1'] for r in per_class)
    nll=math.fsum(-math.log(max(1e-8,min(1.,float(p[i,int(c)-1])))) for i,c in enumerate(truth))/count
    ids=list(range(1,classes+1))
    return {'class_ids':ids,'confusion_matrix':cm.tolist(),'count':count,'oa':oa,'aa':aa,
        'kappa':(oa-expected)/(1-expected) if abs(1-expected)>1e-15 else None,'macro_f1':mf1,
        'macro_class_ids':ids,'missing_class_ids':[],'macro_policy':'ground_truth_supported_classes',
        'full_class_coverage':True,'aa_full':aa,'macro_f1_full':mf1,'per_class':per_class,'nll':nll}


def oracle_branch(labels,probabilities):
    labels=np.asarray(labels);pred={h:np.asarray(probabilities[h]).argmax(1)+1 for h in HEADS}
    def selected(mask):
        truth=labels[mask];c=pred['cnn'][mask]==truth;g=pred['graph'][mask]==truth;f=pred['fused'][mask]==truth
        def ratio(a,b):return int(a)/int(b) if b else None
        strata={}
        for name,ids in [('both_correct',c&g),('cnn_only_correct',c&~g),('graph_only_correct',~c&g),('both_wrong',~c&~g)]:
            n=int(ids.sum());yes=int((ids&f).sum());strata[name]={'count':n,'fused_correct':yes,'fused_accuracy':ratio(yes,n)}
        one=c^g;rescue=int((one&f).sum());both=c&g;harm=int((both&~f).sum());n=len(truth)
        return {'count':n,'strata':strata,'cnn_accuracy':ratio(c.sum(),n),'graph_accuracy':ratio(g.sum(),n),
            'fused_accuracy':ratio(f.sum(),n),'branch_oracle_accuracy':ratio((c|g).sum(),n),
            'prediction_disagreement_count':int((pred['cnn'][mask]!=pred['graph'][mask]).sum()),
            'correction':{'numerator':rescue,'denominator':int(one.sum()),'rate':ratio(rescue,one.sum()),'population':'exactly_one_branch_correct'},
            'harm':{'numerator':harm,'denominator':int(both.sum()),'rate':ratio(harm,both.sum()),'population':'both_branches_correct'}}
    result=selected(np.ones(len(labels),dtype=bool))
    result['per_class']={str(c):selected(labels==c) for c in range(1,next(iter(probabilities.values())).shape[1]+1)}
    return result


def oracle_summary(rows):
    lookup={(r['target'],r['seed'],r['method']):r for r in rows}
    if len(rows)!=27 or set(lookup)!={(t,s,m) for t in NAMES for s in SEEDS for m in METHODS}:
        raise ValueError('Independent paired coverage incomplete')
    def stats(values):
        values=list(map(float,values));return {'mean':statistics.fmean(values),'sample_sd':statistics.stdev(values),'values':values}
    targets={};macro={}
    for m in METHODS:
        targets[m]={t:{k:stats(lookup[t,s,m]['metrics']['fused'][k] for s in SEEDS) for k in ('oa','aa','kappa','nll')} for t in NAMES}
        macro[m]={k:stats(statistics.fmean(lookup[t,s,m]['metrics']['fused'][k] for t in NAMES) for s in SEEDS) for k in ('oa','aa','kappa','nll')}
    contrasts={}
    for m in METHODS[1:]:
        delta=[statistics.fmean(lookup[t,s,m]['metrics']['fused']['oa']-lookup[t,s,METHODS[0]]['metrics']['fused']['oa'] for t in NAMES) for s in SEEDS]
        aa={t:statistics.fmean(lookup[t,s,m]['metrics']['fused']['aa']-lookup[t,s,METHODS[0]]['metrics']['fused']['aa'] for s in SEEDS) for t in NAMES}
        hong=statistics.fmean(lookup['whu-honghu',s,m]['metrics']['fused']['oa']-lookup['whu-honghu',s,METHODS[0]]['metrics']['fused']['oa'] for s in SEEDS)
        recalls={t:[statistics.fmean(lookup[t,s,m]['metrics']['fused']['per_class'][c]['recall']-
            lookup[t,s,METHODS[0]]['metrics']['fused']['per_class'][c]['recall'] for s in SEEDS) for c in range(DATASETS[t][2])] for t in NAMES}
        checks={'positive_paired_macro_mean':statistics.fmean(delta)>0,'all3_macro_positive':all(v>0 for v in delta),
            'all_target_aa_protection':all(v+1e-12>=-.01 for v in aa.values()),'honghu_oa_protection':hong+1e-12>=-.01,
            'all_class_recall_protection':all(v+1e-12>=-.02 for values in recalls.values() for v in values)}
        contrasts[m]={'paired_macro_oa_delta':stats(delta),'target_mean_aa_delta':aa,'honghu_mean_oa_delta':hong,
            'per_class_mean_recall_delta':recalls,'gates':checks,'passes_all_gates':all(checks.values()),
            'role':'primary' if m==METHODS[1] else 'descriptive_control'}
    return {'by_target':targets,'macro':macro,'contrasts':contrasts,'training_seeds':list(SEEDS),'support_seed':64,
        'support_repetition_claim':False,'significance_claim':False}


def assert_tree(expected,actual,path='root'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):raise ValueError('Keys differ at '+path)
        for key in expected:assert_tree(expected[key],actual[key],path+'.'+str(key))
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):raise ValueError('Length differs at '+path)
        for i,(a,b) in enumerate(zip(expected,actual)):assert_tree(a,b,path+'['+str(i)+']')
    elif isinstance(expected,float):
        if not isinstance(actual,(float,int)) or not math.isfinite(actual) or not math.isclose(expected,float(actual),rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Number differs at '+path)
    elif expected!=actual:raise ValueError('Value differs at '+path)


def verify_saved(directory,result_sha256,current_code):
    """No model construction or optimizer; independent GT and saved predictions."""
    root=Path(directory).resolve(strict=True)
    if file_hash(root/'result.json')!=result_sha256:raise ValueError('Pinned query result changed')
    read=lambda p:json.loads(Path(p).read_text());result=read(root/'result.json');producer=result['identity']['code']
    if (result.get('status')!='completed' or result.get('policy')!='g4_all27_frozen_query_v1'
            or producer.get('dirty') or current_code.get('dirty') or not producer.get('files')
            or producer['source_hash']!=digest(producer['files'])
            or any(current_code['files'].get(k)!=v for k,v in producer['files'].items())):
        raise ValueError('Completed unchanged query implementation required')
    protected={str(root/n):h for n,h in result['artifacts'].items() if Path(n).name==n}
    if len(protected)!=len(result['artifacts']):raise ValueError('Unsafe query artifact name')
    protected[str(root/'result.json')]=result_sha256
    protected.update(result['protected_training_files']);protected.update(result['raw_label_pins'])
    def guard():
        if any(file_hash(p)!=h for p,h in protected.items()):raise ValueError('Independent audit protected file changed')
    guard();freeze=read(root/'freeze.json');identity=result['identity']
    if (freeze['verification_code']!=producer or freeze['training_code']['commit']!='85801647e4824ea4adadebbde3394a38f75cb62c'
            or len(freeze['models'])!=27 or len(freeze['native_checks'])!=27
            or any(v['status']!='passed' for v in freeze['native_checks'])
            or freeze['head_metric_cases']!=162 or freeze['class_recalls']!=2160
            or freeze['protected_files']!=result['protected_training_files']):raise ValueError('Global prequery proof differs')
    cohort=read(identity['prepared_manifest'])
    if file_hash(identity['prepared_manifest'])!=identity['prepared_manifest_sha256']:raise ValueError('Cohort metadata changed')
    launch=read(Path(identity['training_root'])/'launch.json');cases={c['name']:c for c in launch['cases']}
    models={m['name']:m for m in freeze['models']};truth={};families={};oracle_rows=[]
    head_count=recall_count=class_branch_count=0
    for row in result['rows']:
        name=row['name'];case=cases[name];model=models[name];target=case['target'];classes=DATASETS[target][2]
        assert_tree(row,read(root/(name+'-metrics.json')),name+'.metrics-file')
        if (digest(case)!=model['case_sha256'] or row['target']!=target or row['method']!=case['method'] or row['seed']!=case['seed']
                or row['checkpoint_sha256']!=model['primary_sha256']
                or file_hash(Path(case['out'])/'primary.pt')!=model['primary_sha256']):raise ValueError('Query row/model case differs')
        if target not in truth:
            from scipy.io import loadmat
            raw=cohort['datasets'][target]['raw'];path=bounded(Path(identity['data_root']).resolve(strict=True),raw['label_path'])
            if file_hash(path)!=raw['label_sha256']:raise ValueError('Independent raw GT pin differs')
            key=DATASETS[target][1]+'_gt';truth[target]=loadmat(path,variable_names=[key])[key]
            family=read(case['inputs']['family'])
            if (file_hash(case['inputs']['family'])!=case['inputs']['family_sha256'] or label_hash(truth[target])!=family['label_hash']
                    or family['family_hash']!=model['family_hash']):raise ValueError('Independent GT/family identity differs')
            families[target]=family
        family=families[target]
        if row['family_hash']!=family['family_hash'] or model['family_hash']!=family['family_hash']:raise ValueError('Paired query family differs')
        payload_path=root/row['query_payload']
        if Path(row['query_payload']).name!=row['query_payload'] or file_hash(payload_path)!=row['query_payload_sha256']:
            raise ValueError('Saved query payload differs')
        payload=torch.load(payload_path,map_location='cpu',weights_only=True)
        ids=payload['indices'].numpy()
        if (payload['target']!=target or payload['seed']!=case['seed'] or payload['method']!=case['method']
                or ids.tolist()!=family['query_indices']):raise ValueError('Saved query target/seed/indices differ')
        labels=truth[target].reshape(-1)[ids].astype(np.int64)
        if (payload['labels'].tolist()!=labels.tolist()
                or payload['checkpoint_sha256']!=model['primary_sha256'] or payload['family_hash']!=family['family_hash']
                or payload['freeze_sha256']!=file_hash(root/'freeze.json')):raise ValueError('Saved query truth/indices/weight binding differs')
        probabilities={h:payload['probabilities'][h].numpy() for h in HEADS};metrics={}
        for h in HEADS:
            if probabilities[h].shape!=(len(ids),classes):raise ValueError('Saved target class count differs')
            metrics[h]=oracle_metrics(labels,probabilities[h]);assert_tree(metrics[h],row['metrics'][h],name+'.'+h)
            head_count+=1;recall_count+=classes
        branch=oracle_branch(labels,probabilities);assert_tree(branch,row['branch'],name+'.branch');class_branch_count+=classes
        oracle_rows.append({'target':target,'seed':case['seed'],'method':case['method'],'metrics':metrics})
    summary=oracle_summary(oracle_rows);assert_tree(summary,result['summary'],'summary');assert_tree(summary,read(root/'summary.json'),'summary-file')
    guard()
    return {'status':'passed','verification_code':current_code,'query_producer':producer,'result_sha256':result_sha256,
        'head_metric_cases':head_count,'class_recalls':recall_count,'branch_records':len(oracle_rows),
        'class_branch_records':class_branch_count,'training_files_unchanged':len(result['protected_training_files']),
        'evaluation_artifacts_unchanged':len(result['artifacts'])+1,'raw_input_files_unchanged':len(result['raw_label_pins']),
        'analysis_optimizer_updates':0,'query_used_for_selection':False,'summary':summary}

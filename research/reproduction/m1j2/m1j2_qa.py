"""Independent raw-label metric, paired statistic and preregistered gate audit."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m1j2-issue127');BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from loadData.dataset_registry import inspect_dataset,file_hash
from research_runtime import write_json,code_snapshot

load=lambda p:json.loads(Path(p).read_text())
out=BASE/'runs/m1j2-issue127-analysis01';batch=BASE/'runs/m1j2-issue127-full01'
report=load(out/'results.json');launch=load(batch/'launch.json');frozen=load(out/'frozen.json');source=code_snapshot(ROOT)
assert source==report['source']==launch['code'] and not source['dirty'] and report['status']=='completed'
assert file_hash(batch/'launch.json')==frozen['launch_sha256']
seeds=(64,512,2048);names=('paviaU','xuzhou','whu-honghu');conditions=('full','random75','random50','stride2')
rows={(r['preset'],r['seed'],r['condition'],r['dataset']):r for r in report['rows']}
assert len(rows)==len(report['rows'])==72
truths={};counts={}
for name in names:
 packet=torch.load(out/(name+'-labels.pt'),weights_only=True)
 gt=inspect_dataset(launch['data_root'],name)[1].ravel()
 split=load(Path(launch['prepared'])/(name+'-split.json'))['indices']
 ids=np.sort(np.asarray(split['test']));truth=gt[ids].astype(int)-1
 assert np.array_equal(packet['ids'].numpy(),ids) and np.array_equal(packet['truth'].numpy(),truth)
 c=int(truth.max()+1);count=np.bincount(gt[np.asarray(split['train'])].astype(int),minlength=c+1)[1:]
 assert count.tolist()==packet['train_counts']==report['train_counts'][name]
 truths[name]=truth;counts[name]=count
head_cases=0;class_cases=0;complementarity=0
for key,row in rows.items():
 truth=truths[row['dataset']];c=len(counts[row['dataset']]);packet=torch.load(out/row['packet'],weights_only=True)
 predictions={}
 for head,p in packet.items():
  pred=p['pred'].long().numpy();prob=p['truth_probability'];assert len(pred)==len(truth)
  assert pred.min()>=0 and pred.max()<c and torch.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
  cm=np.zeros((c,c),dtype=np.int64);np.add.at(cm,(truth,pred),1)
  score=row['scores'][head];recall=np.diag(cm)/cm.sum(1)
  assert cm.tolist()==score['confusion']
  assert abs(float(np.mean(pred==truth))-score['oa'])<1e-12
  assert abs(float(recall.mean())-score['aa'])<1e-12
  np.testing.assert_allclose(recall,score['recall'],rtol=0,atol=1e-12)
  assert abs(float(-np.log(np.maximum(prob.numpy().astype(float),1e-8)).mean())-score['nll'])<2e-6
  head_cases+=1;class_cases+=c;predictions[head]=pred
 a=predictions['cnn']==truth;b=predictions['graph']==truth;f=predictions['fused']==truth;one=a^b;wrong=~a&~b
 expected={'both_wrong':float(wrong.mean()),'exactly_one_correct':float(one.mean()),
           'fusion_correction_when_one_correct':float(f[one].mean()) if one.any() else None,
           'both_wrong_by_class':[float(wrong[truth==i].mean()) for i in range(c)]}
 assert expected==row['complementarity'];complementarity+=1
 assert row['bands']==len(launch['evaluation_bank'][row['condition']][row['dataset']]['selected_positions'])

def score(preset,seed,condition,name,metric='oa',head='fused'):
 return rows[preset,seed,condition,name]['scores'][head][metric]
def stat(x):
 x=np.asarray(x,dtype=float)*100
 return {'values':x.tolist(),'mean':float(x.mean()),'std':float(x.std(ddof=1))}
tables={};paired={};guard_details=[];small_details=[]
for preset in ('full','alternate75'):
 tables[preset]={}
 for name in names:
  tables[preset][name]={}
  for head in ('fused','cnn','graph'):
   tables[preset][name][head]={
    'full_oa':stat([score(preset,s,'full',name,head=head) for s in seeds]),
    'full_aa':stat([score(preset,s,'full',name,'aa',head) for s in seeds]),
    'robust_oa':stat([np.mean([score(preset,s,c,name,head=head) for c in conditions[1:]]) for s in seeds]),
    'robust_aa':stat([np.mean([score(preset,s,c,name,'aa',head) for c in conditions[1:]]) for s in seeds])}
 tables[preset]['macro']={
  'full_oa':stat([np.mean([score(preset,s,'full',n) for n in names]) for s in seeds]),
  'robust_oa':stat([np.mean([score(preset,s,c,n) for n in names for c in conditions[1:]]) for s in seeds])}
for kind,conds in [('full',('full',)),('robust',conditions[1:])]:
 deltas=[np.mean([score('alternate75',s,c,n)-score('full',s,c,n) for n in names for c in conds]) for s in seeds]
 paired[kind]={**stat(deltas),'positive_seeds':int(sum(d>0 for d in deltas)),
               'passed':bool(np.mean(deltas)>0 and sum(d>0 for d in deltas)>=2)}
 for name in names:
  delta=float(np.mean([score('alternate75',s,c,name,'aa')-score('full',s,c,name,'aa') for s in seeds for c in conds]))
  guard_details.append({'kind':kind,'dataset':name,'aa_change_pp':delta*100,'passed':delta>=-.005-1e-12})
  for i,count in enumerate(counts[name]):
   delta=float(np.mean([score('alternate75',s,c,name,'recall')[i]-score('full',s,c,name,'recall')[i] for s in seeds for c in conds]))
   small_details.append({'kind':kind,'dataset':name,'class':i+1,'train_support':int(count),
                          'recall_change_pp':delta*100,'small_protected':bool(count<=25),'passed':bool(count>25 or delta>=-.02-1e-12)})
gates={'full_oa':bool(paired['full']['passed']),'robust_oa':bool(paired['robust']['passed']),
       'dataset_aa':all(g['passed'] for g in guard_details),'small_class':all(g['passed'] for g in small_details)}
gates['joint_benefit']=all(gates.values())
for p,h in frozen['training_artifacts'].items():assert file_hash(p)==h
for p,h in launch['inputs'].items():assert file_hash(p)==h
reviews={f'{p}-{s}':load(batch/'reviews'/f'{p}-{s}.json') for p in ('full','alternate75') for s in seeds}
receipt={'status':'passed','code':source,'training_commit':source['commit'],'qa_sha256':file_hash(__file__),'head_metrics_recomputed':head_cases,
 'class_recalls_recomputed':class_cases,'complementarity_recomputed':complementarity,
 'raw_test_labels_and_training_counts_checked':True,'tables':tables,'paired':paired,'aa_guards':guard_details,
 'class_changes':small_details,'gates':gates,'training_artifacts_unchanged':len(frozen['training_artifacts']),
 'inputs_unchanged':len(launch['inputs']),'new_training_updates':0,'batch_status':load(batch/'status.json'),
 'evaluation_seconds':report['elapsed_seconds'],
 'training_reviews':{k:{f:v for f,v in r.items() if f not in ('code','artifacts')} for k,r in reviews.items()}}
write_json(out/'independent-receipt.json',receipt)
write_json(out/'summary.json',{k:v for k,v in receipt.items() if k not in ('code','batch_status','training_reviews')})
print(json.dumps({'status':'passed','head_metrics':head_cases,'class_recalls':class_cases,'gates':gates,
                  'full_control':tables['full']['macro'],'augmented':tables['alternate75']['macro'],'paired':paired}))

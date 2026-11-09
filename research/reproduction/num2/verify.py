"""Independent two-precision raw-label metrics, fixed benefit gates and sensitivity."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m4s2-issue135');BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from loadData.dataset_registry import inspect_dataset,file_hash
from research_runtime import code_snapshot,write_json
load=lambda p:json.loads(Path(p).read_text())
batch=BASE/'runs/m4s2-issue135-full01';out=BASE/'runs/m4s2-issue135-analysis02'
report=load(out/'results.json');launch=load(batch/'launch.json');frozen=load(out/'frozen.json')
assert report['source']==launch['code']==frozen['source']==code_snapshot(ROOT)
assert report['consumer']==frozen['consumer']==code_snapshot('/home/lbx/hsi-pioneer271828/checkouts/num2-issue143')
assert report['status']=='completed' and report['original_probability_policy']=='not_stably_passed' and not report['probability_calibration_validated']
assert file_hash(batch/'launch.json')==frozen['launch_sha256']
names=('paviaU','xuzhou','whu-honghu');seeds=(64,512,2048);precisions=('fp32','fp64');truths={};counts={}
for name in names:
 gt=inspect_dataset(launch['data_root'],name)[1].ravel();split=load(Path(launch['prepared'])/(name+'-split.json'))['indices']
 ids=np.sort(split['test']);truths[name]=gt[ids].astype(int)-1
 packet=torch.load(out/(name+'-labels.pt'),weights_only=True)
 assert np.array_equal(packet['ids'].numpy(),ids) and np.array_equal(packet['truth'].numpy(),truths[name])
 counts[name]=np.bincount(gt[np.asarray(split['train'])].astype(int),minlength=int(gt.max())+1)[1:]
 assert counts[name].tolist()==packet['train_counts']==launch['train_class_counts'][name]
rows={};packets={};metrics=recalls=0
for row in report['rows']:
 key=(row['mode'],row['supervision'],row['seed'],row['dataset'],row['precision']);assert key not in rows;rows[key]=row
 truth=truths[row['dataset']];c=len(counts[row['dataset']]);packet=torch.load(out/row['packet'],weights_only=True);packets[key]=packet
 assert set(packet)==set(row['scores'])==({'fused'} if row['supervision']=='primary_only' else {'fused','cnn','graph'})
 for head,p in packet.items():
  pred=p['pred'].long().numpy();tp=p['truth_probability'].numpy().astype(float)
  assert len(pred)==len(truth) and pred.min()>=0 and pred.max()<c and np.isfinite(tp).all() and (tp>=0).all() and (tp<=1).all()
  cm=np.zeros((c,c),dtype=int);np.add.at(cm,(truth,pred),1);recall=np.diag(cm)/cm.sum(1);score=row['scores'][head]
  assert cm.tolist()==score['confusion'] and abs(float(np.mean(truth==pred))-score['oa'])<1e-12
  assert abs(float(recall.mean())-score['aa'])<1e-12
  np.testing.assert_allclose(recall,score['recall'],rtol=0,atol=1e-12)
  assert abs(float(-np.log(np.maximum(tp,1e-8)).mean())-score['nll_descriptive'])<2e-6
  metrics+=1;recalls+=c
assert len(rows)==72 and metrics==144 and recalls==1920
def score(cell,seed,name,precision,metric='oa',head='fused'):return rows[(*cell,seed,name,precision)]['scores'][head][metric]
def macro(cell,seed,precision,head='fused'):return np.mean([score(cell,seed,n,precision,head=head) for n in names])
def stat(values):
 x=np.asarray(values)*100;return {'values':x.tolist(),'mean':float(x.mean()),'std':float(x.std(ddof=1))}
cells=[(m,s) for m in ('none','x_residual') for s in ('legacy_three_heads','primary_only')]
tables={};paired={};gates={};class_deltas={};branch_tables={}
for precision in precisions:
 tables[precision]={f'{m}-{s}':{'macro_oa':stat([macro((m,s),z,precision) for z in seeds]),
  'datasets':{n:{metric:stat([score((m,s),z,n,precision,metric) for z in seeds]) for metric in ('oa','aa')} for n in names}} for m,s in cells}
 branch_tables[precision]={m:{head:stat([macro((m,'legacy_three_heads'),z,precision,head) for z in seeds]) for head in ('fused','cnn','graph')} for m in ('none','x_residual')}
 comparisons=[('residual_legacy',cells[2],cells[0]),('residual_primary',cells[3],cells[1]),('supervision_none',cells[1],cells[0]),('supervision_residual',cells[3],cells[2])]
 paired[precision]={label:stat([macro(a,z,precision)-macro(b,z,precision) for z in seeds]) for label,a,b in comparisons}
 paired[precision]['interaction']=stat([(macro(cells[3],z,precision)-macro(cells[1],z,precision))-(macro(cells[2],z,precision)-macro(cells[0],z,precision)) for z in seeds])
 gates[precision]=[];class_deltas[precision]=[];candidate=tuple(launch['study']['candidate']);gate=launch['study']['gate']
 for control in map(tuple,launch['study']['comparators']):
  delta=[macro(candidate,z,precision)-macro(control,z,precision) for z in seeds];aa=[];small=[]
  for name in names:
   d=float(np.mean([score(candidate,z,name,precision,'aa')-score(control,z,name,precision,'aa') for z in seeds]));aa.append({'dataset':name,'delta_pp':d*100,'passed':d>=gate['each_dataset_aa_floor']-1e-12})
   for k,count in enumerate(counts[name]):
    d=float(np.mean([score(candidate,z,name,precision,'recall')[k]-score(control,z,name,precision,'recall')[k] for z in seeds]))
    row={'control':list(control),'dataset':name,'class':k+1,'train_count':int(count),'delta_pp':d*100,'protected':bool(count<=25),'passed':bool(count>25 or d>=-.02-1e-12)}
    small.append(row);class_deltas[precision].append(row)
  honghu=float(np.mean([score(candidate,z,'whu-honghu',precision)-score(control,z,'whu-honghu',precision) for z in seeds]))
  tests={'macro_oa':bool(np.mean(delta)>0),'positive_seeds':bool(sum(d>0 for d in delta)>=2),'dataset_aa':all(r['passed'] for r in aa),'honghu_oa':honghu>=-.005-1e-12,'small_class':all(r['passed'] for r in small)}
  gates[precision].append({'control':list(control),'paired':stat(delta),'positive_seeds':int(sum(d>0 for d in delta)),'aa':aa,'honghu_delta_pp':honghu*100,'checks':tests,'passed':all(tests.values())})
sensitivity=[]
for mode,supervision in cells:
 for seed in seeds:
  for name in names:
   key=(mode,supervision,seed,name);a=packets[(*key,'fp32')];b=packets[(*key,'fp64')];truth=truths[name]
   for head in a:
    x=a[head]['pred'].numpy();y=b[head]['pred'].numpy();changed=x!=y
    sensitivity.append({'mode':mode,'supervision':supervision,'seed':seed,'dataset':name,'head':head,'changes':int(changed.sum()),'n':len(x),
     'by_class_changes':np.bincount(truth[changed],minlength=len(counts[name])).tolist(),'oa_delta_pp':float(((truth==y).mean()-(truth==x).mean())*100)})
agreement=all(a['checks']==b['checks'] and [g['passed'] for g in a['aa']]==[g['passed'] for g in b['aa']] for a,b in zip(gates['fp32'],gates['fp64']))
agreement=agreement and all(a['passed']==b['passed'] for a,b in zip(class_deltas['fp32'],class_deltas['fp64']))
for p,h in frozen['training_artifacts'].items():assert file_hash(p)==h
for p,h in launch['inputs'].items():assert file_hash(p)==h
reviews={c['name']:load(batch/'reviews'/(c['name']+'.json')) for c in launch['cases']}
receipt={'status':'passed','source':report['source'],'consumer':report['consumer'],'qa_sha256':file_hash(__file__),'policy':report['policy'],
 'original_probability_policy':'not_stably_passed','probability_calibration_validated':False,'metrics_recomputed':metrics,'recalls_recomputed':recalls,
 'tables':tables,'branch_tables':branch_tables,'paired':paired,'gates':gates,'class_changes':class_deltas,'precision_sensitivity':sensitivity,
 'precision_gate_agreement':bool(agreement),'stable_joint_candidate_benefit':bool(agreement and all(g['passed'] for p in precisions for g in gates[p])),
 'training_artifacts_unchanged':len(frozen['training_artifacts']),'inputs_unchanged':len(launch['inputs']),'optimizer_updates':0,
 'training_reviews':{k:{f:v for f,v in r.items() if f not in ('code','artifacts','selected_validation')} for k,r in reviews.items()},
 'batch_status':load(batch/'status.json'),'evaluation_seconds':report['elapsed_seconds']}
write_json(out/'independent-receipt.json',receipt)
write_json(out/'summary.json',{k:v for k,v in receipt.items() if k not in ('source','consumer','batch_status')})
print(json.dumps({'status':'passed','metrics':metrics,'recalls':recalls,'tables':tables,'paired':paired,'gates':gates,'precision_gate_agreement':bool(agreement),'prediction_changes':sum(r['changes'] for r in sensitivity)}))

"""Independently recompute raw-label metrics, paired contrasts and frozen protection gates."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m2j2-issue139');BASE=ROOT.parents[1];sys.path.insert(0,str(ROOT))
from loadData.dataset_registry import inspect_dataset,file_hash
from research_runtime import code_snapshot,write_json
load=lambda p:json.loads(Path(p).read_text())
batch=BASE/'runs/m2j2-issue139-full01';out=BASE/'runs/m2j2-issue139-analysis01'
launch=load(batch/'launch.json');report=load(out/'results.json');frozen=load(out/'frozen.json');source=code_snapshot(ROOT)
assert source==launch['code']==report['source']==frozen['source'] and report['status']=='completed'
assert file_hash(batch/'launch.json')==frozen['launch_sha256']
names=('paviaU','xuzhou','whu-honghu');seeds=(64,512,2048);truths={};counts={}
for name in names:
 gt=inspect_dataset(launch['data_root'],name)[1].ravel();split=load(Path(launch['prepared'])/(name+'-split.json'))['indices']
 ids=np.sort(split['test']);truths[name]=gt[ids].astype(int)-1
 packet=torch.load(out/(name+'-labels.pt'),weights_only=True)
 assert np.array_equal(packet['ids'].numpy(),ids) and np.array_equal(packet['truth'].numpy(),truths[name])
 counts[name]=np.bincount(gt[np.asarray(split['train'])].astype(int),minlength=int(gt.max())+1)[1:]
 assert counts[name].tolist()==packet['train_counts']==launch['train_class_counts'][name]
rows={};metrics=recalls=0
for row in report['rows']:
 key=(row['mode'],row['seed'],row['dataset']);assert key not in rows;rows[key]=row
 truth=truths[row['dataset']];c=len(counts[row['dataset']]);packet=torch.load(out/row['packet'],weights_only=True)
 assert set(packet)==set(row['scores'])=={'fused','cnn','graph'}
 for head,p in packet.items():
  pred=p['pred'].long().numpy();tp=p['truth_probability'].numpy().astype(float)
  assert len(pred)==len(truth) and pred.min()>=0 and pred.max()<c
  cm=np.zeros((c,c),dtype=int);np.add.at(cm,(truth,pred),1);recall=np.diag(cm)/cm.sum(1);score=row['scores'][head]
  assert cm.tolist()==score['confusion'] and abs(float(np.mean(truth==pred))-score['oa'])<1e-12
  assert abs(float(recall.mean())-score['aa'])<1e-12
  np.testing.assert_allclose(recall,score['recall'],rtol=0,atol=1e-12)
  assert abs(float(-np.log(np.maximum(tp,1e-8)).mean())-score['nll'])<2e-6
  metrics+=1;recalls+=c
assert len(rows)==27 and metrics==81 and recalls==1080
def score(cell,seed,name,metric='oa'):return rows[cell,seed,name]['scores']['fused'][metric]
def macro(cell,seed):return np.mean([score(cell,seed,n) for n in names])
def stat(values):
 x=np.asarray(values)*100;return {'values':x.tolist(),'mean':float(x.mean()),'std':float(x.std(ddof=1))}
cells=['wavelength','index','uniform']
tables={m:{'macro_oa':stat([macro(m,z) for z in seeds]),
 'datasets':{n:{metric:stat([score(m,z,n,metric) for z in seeds]) for metric in ('oa','aa')} for n in names}} for m in cells}
paired={f'wavelength_minus_{control}':stat([macro('wavelength',z)-macro(control,z) for z in seeds]) for control in ('index','uniform')}
gates=[];class_deltas=[]
candidate=launch['study']['candidate'];gate=launch['study']['gate']
for control in launch['study']['comparators']:
 delta=[macro(candidate,z)-macro(control,z) for z in seeds];aa=[];small=[]
 for name in names:
  d=float(np.mean([score(candidate,z,name,'aa')-score(control,z,name,'aa') for z in seeds]));aa.append({'dataset':name,'delta_pp':d*100,'passed':d>=gate['each_dataset_aa_floor']-1e-12})
  for k,count in enumerate(counts[name]):
   d=float(np.mean([score(candidate,z,name,'recall')[k]-score(control,z,name,'recall')[k] for z in seeds]))
   row={'control':control,'dataset':name,'class':k+1,'train_count':int(count),'delta_pp':d*100,'protected':bool(count<=25),'passed':bool(count>25 or d>=-.02-1e-12)}
   small.append(row);class_deltas.append(row)
 honghu=float(np.mean([score(candidate,z,'whu-honghu')-score(control,z,'whu-honghu') for z in seeds]))
 tests={'macro_oa':bool(np.mean(delta)>0),'positive_seeds':bool(sum(d>0 for d in delta)>=2),'dataset_aa':all(r['passed'] for r in aa),
        'honghu_oa':honghu>=-.005-1e-12,'small_class':all(r['passed'] for r in small)}
 gates.append({'control':control,'paired':stat(delta),'positive_seeds':int(sum(d>0 for d in delta)),'aa':aa,'honghu_delta_pp':honghu*100,'checks':tests,'passed':all(tests.values())})
for p,h in frozen['training_artifacts'].items():assert file_hash(p)==h
for p,h in launch['inputs'].items():assert file_hash(p)==h
reviews={c['name']:load(batch/'reviews'/(c['name']+'.json')) for c in launch['cases']}
receipt={'status':'passed','source':source,'script_sha256':file_hash(__file__),'metrics_recomputed':metrics,'class_recalls_recomputed':recalls,
 'tables':tables,'paired':paired,'gates':gates,'joint_candidate_passed':all(g['passed'] for g in gates),'class_changes':class_deltas,
 'training_artifacts_unchanged':len(frozen['training_artifacts']),'inputs_unchanged':len(launch['inputs']),
 'training_reviews':{k:{f:v for f,v in r.items() if f not in ('code','artifacts','selected_validation')} for k,r in reviews.items()},
 'batch_status':load(batch/'status.json'),'new_optimizer_updates':0}
write_json(out/'independent-receipt.json',receipt)
write_json(out/'summary.json',{k:v for k,v in receipt.items() if k not in ('source','batch_status')})
print(json.dumps({'status':'passed','metrics':metrics,'recalls':recalls,'tables':tables,'paired':paired,'gates':gates}))

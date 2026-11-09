"""Frozen M2-J2 test readout. Run only after nine per-training reviews pass."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m2j2-issue139');BASE=ROOT.parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
from joint_training import import_preparation,load_plan
from joint_adapter_modes import AdapterModeRuntime
from joint_adapter_study import load_study,select_case
from joint_baseline import forward
from loadData.dataset_registry import inspect_dataset,file_hash
from research_runtime import code_snapshot,require_frozen_code,write_json
from evaluation.joint_finalization import verify_probability_pair

load=lambda p:json.loads(Path(p).read_text())
batch=BASE/'runs/m2j2-issue139-full01';launch=load(batch/'launch.json');status=load(batch/'status.json')
source=code_snapshot();assert source==launch['code'] and not source['dirty']
assert status['status']=='training_completed' and len(status['jobs'])==9 and all(j['status']=='completed' for j in status['jobs'])
out=BASE/'runs/m2j2-issue139-analysis01';out.mkdir(exist_ok=False);started=time.monotonic()
_,items,_=import_preparation(launch['prepared'],launch['data_root'],load_plan('configs/experiments/base97_training.json'),source)
protected={}
for case in launch['cases']:
 assert select_case(launch,source,load_study(),case['mode'],case['seed'])==case
 folder=Path(case['out']);r=load(folder/'result.json');review=load(batch/'reviews'/(case['name']+'.json'))
 assert r['settings']==case['settings'] and r['optimizer_steps']==review['updates']==1503
 for n,h in r['artifacts'].items():assert file_hash(folder/n)==h
 protected.update({str(p):file_hash(p) for p in folder.iterdir() if p.is_file()})
write_json(out/'frozen.json',{'source':source,'launch_sha256':file_hash(batch/'launch.json'),'training_artifacts':protected})
preflight=[]
for case in launch['cases']:
 folder=Path(case['out']);r=load(folder/'result.json');rt=AdapterModeRuntime(case['mode'])
 model=rt.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{r["selected_step"]:06d}.pt',device='cuda')
 moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
 _,values=rt.validation_snapshot(model,moved);saved=torch.load(folder/'selected-probabilities.pt',weights_only=True,map_location='cpu')
 for n in items:
  for phase in ('train','val'):
   assert set(saved[n][phase])==set(values[n][phase])
   for head in saved[n][phase]:preflight.append({'case':case['name'],'dataset':n,'phase':phase,'head':head,**verify_probability_pair(saved[n][phase][head],values[n][phase][head])})
 del model,moved,values,saved;torch.cuda.empty_cache()
write_json(out/'all-case-preflight.json',{'checks':preflight,'test_scoring':False,'deterministic':torch.are_deterministic_algorithms_enabled()})
truths={}
for name,item in items.items():
 gt=inspect_dataset(launch['data_root'],name)[1].ravel();split=load(Path(launch['prepared'])/(name+'-split.json'))['indices']
 ids=np.sort(np.asarray(split['test']));truth=gt[ids].astype(np.int64)-1
 assert not np.intersect1d(ids,split['train']).size and not np.intersect1d(ids,split['val']).size
 counts=np.bincount(gt[np.asarray(split['train'])].astype(int),minlength=item['class_num']+1)[1:].tolist()
 assert counts==launch['train_class_counts'][name]
 truths[name]=(torch.tensor(ids),torch.tensor(truth))
 torch.save({'ids':torch.tensor(ids),'truth':torch.tensor(truth),'train_counts':counts},out/(name+'-labels.pt'))
rows=[];checks=[]
for case in launch['cases']:
 folder=Path(case['out']);r=load(folder/'result.json');runtime=AdapterModeRuntime(case['mode'])
 model=runtime.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{r["selected_step"]:06d}.pt',device='cuda')
 moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
 _,values=runtime.validation_snapshot(model,moved)
 saved=torch.load(folder/'selected-probabilities.pt',weights_only=True,map_location='cpu')
 for name in items:
  for phase in ('train','val'):
   assert set(saved[name][phase])==set(values[name][phase])
   for head in saved[name][phase]:checks.append({'case':case['name'],'dataset':name,'phase':phase,'head':head,**verify_probability_pair(saved[name][phase][head],values[name][phase][head])})
 del values,saved
 for name,item in moved.items():
  with torch.no_grad():outputs=forward(model,name,item)
  ids,truth=truths[name];target=truth.to('cuda');scores={};packet={}
  for head,value in zip(('fused','graph','cnn'),outputs[:3]):
   if head not in ('fused','graph','cnn'):continue
   prob=value.index_select(0,ids.to('cuda'));assert torch.isfinite(prob).all()
   assert (prob>=0).all() and (prob<=1).all() and torch.allclose(prob.sum(1),torch.ones(len(ids),device='cuda'),rtol=0,atol=2e-6)
   pred=prob.argmax(1).cpu();tp=prob[torch.arange(len(ids),device='cuda'),target].cpu();c=item['class_num']
   cm=np.bincount(truth.numpy()*c+pred.numpy(),minlength=c*c).reshape(c,c);recall=np.diag(cm)/cm.sum(1)
   scores[head]={'oa':float(cm.trace()/cm.sum()),'aa':float(recall.mean()),'recall':recall.tolist(),'confusion':cm.tolist(),'nll':float(-tp.clamp_min(1e-8).log().mean())}
   packet[head]={'pred':pred.to(torch.int16),'truth_probability':tp}
  path=case['name']+'-'+name+'.pt';torch.save(packet,out/path)
  rows.append({'case':case['name'],'mode':case['mode'],'seed':case['seed'],'dataset':name,'scores':scores,'packet':path})
 del model,moved,outputs;torch.cuda.empty_cache();print('EVALUATED',case['name'],flush=True)
assert all(file_hash(p)==h for p,h in protected.items()) and all(file_hash(p)==h for p,h in launch['inputs'].items())
require_frozen_code(source)
write_json(out/'results.json',{'status':'completed','source':source,'script_sha256':file_hash(__file__),'rows':rows,'probability_checks':checks,
 'verification_context':{'deterministic':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
 'training_artifacts_unchanged':len(protected),'inputs_unchanged':len(launch['inputs']),'optimizer_updates':0,'test_scoring':True,'elapsed_seconds':time.monotonic()-started})
print('EVALUATION_COMPLETE',len(rows),flush=True)

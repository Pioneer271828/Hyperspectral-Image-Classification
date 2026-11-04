"""Amended frozen decision evaluation, with both FP32/FP64 results published."""
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);args=p.parse_args()
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
CONSUMER=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CONSUMER))
from decision_evaluation import POLICY,verify_decisions,verify_state
from numerical_audit import double_reference_items,verify_exact_double_cast
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m4s2-issue135');BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from joint_training import import_preparation,load_plan
from joint_supervision import SupervisionRuntime
from joint_baseline import forward
from loadData.dataset_registry import inspect_dataset,file_hash
from research_runtime import code_snapshot,write_json

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
load=lambda p:json.loads(Path(p).read_text())
batch=BASE/'runs/m4s2-issue135-full01';launch=load(batch/'launch.json');source=code_snapshot(ROOT);consumer=code_snapshot(CONSUMER)
assert source==launch['code'] and not source['dirty'] and not consumer['dirty'] and consumer['commit']==args.expected_commit
assert all(consumer['files'].get(n)==h for n,h in source['files'].items())
status=load(batch/'status.json');assert status['status']=='training_completed' and len(status['jobs'])==12 and all(j['status']=='completed' for j in status['jobs'])
out=BASE/'runs/m4s2-issue135-analysis02';out.mkdir(exist_ok=False);started=time.monotonic()
audits=[load(BASE/f'runs/num1-issue141-audit02/slot-{i}.json') for i in (0,1)]
assert all(a['producer']==source and a['optimizer_updates']==0 and not a['test_scoring'] for a in audits)
assert sum(len(a['rows']) for a in audits)==576 and all(r['classification_changes']==0 for a in audits for r in a['rows'])
_,items,_=import_preparation(launch['prepared'],launch['data_root'],load_plan('configs/experiments/base97_training.json'),source)
protected={};checks=[]
for case in launch['cases']:
 folder=Path(case['out']);r=load(folder/'result.json');progress=load(folder/'progress.json')
 assert r['settings']==case['settings'] and r['optimizer_steps']==1503 and r['start_code']==source
 for n,h in r['artifacts'].items():assert file_hash(folder/n)==h
 assert file_hash(folder/progress['state_file'])==progress['state_sha256']
 state=torch.load(folder/progress['state_file'],weights_only=True,map_location='cpu')
 assert state['best_step']==r['selected_step'] and state['identity']==r['identity']
 rt=SupervisionRuntime(case['mode'],case['supervision'])
 model=rt.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{r["selected_step"]:06d}.pt',device='cuda')
 verify_state(state['best_model'],model.state_dict())
 moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
 _,values=rt.validation_snapshot(model,moved);saved=torch.load(folder/'selected-probabilities.pt',weights_only=True,map_location='cpu')
 for n in items:
  for phase in ('train','val'):
   assert set(saved[n][phase])==set(values[n][phase])
   for head in saved[n][phase]:checks.append({'case':case['name'],'dataset':n,'phase':phase,'head':head,**verify_decisions(saved[n][phase][head],values[n][phase][head])})
 verify_state(state['best_model'],model.state_dict())
 protected.update({str(p):file_hash(p) for p in folder.iterdir() if p.is_file()})
 del state,model,moved,values,saved;torch.cuda.empty_cache()
write_json(out/'frozen.json',{'source':source,'consumer':consumer,'launch_sha256':file_hash(batch/'launch.json'),'training_artifacts':protected,'policy':POLICY})
write_json(out/'all-case-preflight.json',{'checks':checks,'test_scoring':False,'policy':POLICY,'original_probability_policy':'not_stably_passed'})
truths={}
for n,item in items.items():
 gt=inspect_dataset(launch['data_root'],n)[1].ravel();split=load(Path(launch['prepared'])/(n+'-split.json'))['indices']
 ids=np.sort(np.asarray(split['test']));truth=gt[ids].astype(np.int64)-1
 assert truth.min()>=0 and truth.max()<item['class_num']
 assert not np.intersect1d(ids,split['train']).size and not np.intersect1d(ids,split['val']).size
 counts=np.bincount(gt[np.asarray(split['train'])].astype(int),minlength=item['class_num']+1)[1:].tolist();assert counts==launch['train_class_counts'][n]
 truths[n]=(torch.tensor(ids),torch.tensor(truth));torch.save({'ids':torch.tensor(ids),'truth':torch.tensor(truth),'train_counts':counts},out/(n+'-labels.pt'))
rows=[]
for case in launch['cases']:
 folder=Path(case['out']);r=load(folder/'result.json');rt=SupervisionRuntime(case['mode'],case['supervision'])
 model=rt.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{r["selected_step"]:06d}.pt',device='cuda')
 state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
 for precision in ('fp32','fp64'):
  if precision=='fp64':model.double();verify_exact_double_cast(state,model.state_dict());moved=double_reference_items(moved)
  for n,item in moved.items():
   with torch.no_grad():outputs=forward(model,n,item)
   ids,truth=truths[n];target=truth.to('cuda');scores={};packet={};c=item['class_num']
   for head,value in zip(('fused','graph','cnn'),outputs[:3]):
    if head not in case['settings']['supervision_policy']['reported_heads']:continue
    prob=value.index_select(0,ids.to('cuda'));assert torch.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
    assert torch.allclose(prob.sum(1),torch.ones_like(prob[:,0]),rtol=0,atol=2e-6)
    pred=prob.argmax(1).cpu();tp=prob[torch.arange(len(ids),device='cuda'),target].cpu()
    cm=np.bincount(truth.numpy()*c+pred.numpy(),minlength=c*c).reshape(c,c);recall=np.diag(cm)/cm.sum(1)
    scores[head]={'oa':float(cm.trace()/cm.sum()),'aa':float(recall.mean()),'recall':recall.tolist(),'confusion':cm.tolist(),'nll_descriptive':float(-tp.clamp_min(1e-8).log().mean())}
    packet[head]={'pred':pred.to(torch.int16),'truth_probability':tp}
   path=case['name']+'-'+precision+'-'+n+'.pt';torch.save(packet,out/path)
   rows.append({'case':case['name'],'mode':case['mode'],'supervision':case['supervision'],'seed':case['seed'],'dataset':n,'precision':precision,'scores':scores,'packet':path})
  if precision=='fp32':verify_state(state,model.state_dict())
  else:verify_exact_double_cast(state,model.state_dict())
 del model,moved,outputs,state;torch.cuda.empty_cache();print('EVALUATED',case['name'],flush=True)
assert all(file_hash(p)==h for p,h in protected.items()) and all(file_hash(p)==h for p,h in launch['inputs'].items())
assert code_snapshot(ROOT)==source and code_snapshot(CONSUMER)==consumer
write_json(out/'results.json',{'status':'completed','source':source,'consumer':consumer,'policy':POLICY,'rows':rows,'test_scoring':True,
 'original_probability_policy':'not_stably_passed','probability_calibration_validated':False,'optimizer_updates':0,
 'script_sha256':file_hash(__file__),'elapsed_seconds':time.monotonic()-started,'training_artifacts_unchanged':len(protected),'inputs_unchanged':len(launch['inputs'])})
print('DECISION_EVALUATION_COMPLETE',len(rows),flush=True)

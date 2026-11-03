"""Frozen six-run readout; write compact independent-audit packets, never train."""
import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m1j2-issue127');BASE=ROOT.parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from research_runtime import code_snapshot,require_frozen_code,write_json
from loadData.dataset_registry import file_hash,inspect_dataset
from loadData.data_read import read_band_metadata
from joint_training import import_preparation,load_plan,validation_snapshot
from joint_baseline import forward
from joint_observation import JointObservationRuntime
from joint_observation_study import load_study,evaluation_bank,condition_view
from joint_calibrated_selection import observations
from model.bn_calibration import calibrate_denoise,CONTEXT,POLICY,state_digest
from evaluation.joint_finalization import verify_probability_pair

torch.set_num_threads(4)
load=lambda p:json.loads(Path(p).read_text())
batch=BASE/'runs/m1j2-issue127-full01';launch=load(batch/'launch.json');status=load(batch/'status.json')
source=code_snapshot();assert source==launch['code'] and not source['dirty']
assert status['status']=='training_completed' and all(j['status']=='completed' for j in status['jobs'])
out=BASE/'runs/m1j2-issue127-analysis01';out.mkdir(exist_ok=False);started=time.monotonic()
_,items,provenance=import_preparation(launch['prepared'],launch['data_root'],load_plan('configs/experiments/base97_training.json'),source)
metadata={n:read_band_metadata(n,i['bands']) for n,i in items.items()}
assert launch['evaluation_bank']==evaluation_bank(load_study(),metadata,provenance['prepared_manifest_sha256'])
protected={}
for case in launch['cases']:
 folder=Path(case['out']);result=load(folder/'result.json')
 assert result['optimizer_steps']==1503 and result['settings']==case['settings'] and result['start_code']==source
 for name,h in result['artifacts'].items():assert file_hash(folder/name)==h
 protected.update({str(p):file_hash(p) for p in folder.iterdir() if p.is_file()})
write_json(out/'frozen.json',{'source':source,'launch_sha256':file_hash(batch/'launch.json'),
                            'training_artifacts':protected,'evaluation_bank':launch['evaluation_bank']})
truths={};train_counts={}
for name,item in items.items():
 gt=inspect_dataset(launch['data_root'],name)[1].ravel()
 split=load(Path(launch['prepared'])/(name+'-split.json'))['indices']
 ids=np.sort(np.asarray(split['test']));train_ids=np.asarray(split['train'])
 assert not np.intersect1d(ids,train_ids).size and not np.intersect1d(ids,np.asarray(split['val'])).size
 truth=gt[ids].astype(np.int64)-1;assert truth.min()>=0 and truth.max()<item['class_num']
 truths[name]=(torch.tensor(ids),torch.tensor(truth))
 train_counts[name]=np.bincount(gt[train_ids].astype(int),minlength=item['class_num']+1)[1:].tolist()
 torch.save({'ids':torch.tensor(ids),'truth':torch.tensor(truth),'train_counts':train_counts[name]},out/(name+'-labels.pt'))
rows=[];checks=[];calibrations=[]
for case in launch['cases']:
 folder=Path(case['out']);result=load(folder/'result.json');runtime=JointObservationRuntime(case['settings'])
 primary=runtime.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{result["selected_step"]:06d}.pt',device='cuda')
 primary_before=state_digest(primary)
 moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in item.items()} for n,item in items.items()}
 _,values=validation_snapshot(primary,moved)
 saved=torch.load(folder/'selected-probabilities.pt',weights_only=True,map_location='cpu')
 for name in items:
  for phase in ('train','val'):
   for head in saved[name][phase]:
    checks.append({'preset':case['preset'],'seed':case['seed'],'dataset':name,'phase':phase,'head':head,
                   **verify_probability_pair(saved[name][phase][head],values[name][phase][head])})
 del values,saved,moved
 for condition,bank in launch['evaluation_bank'].items():
  start=time.monotonic()
  views={n:condition_view(i,metadata[n],bank[n],provenance['prepared_manifest_sha256']) for n,i in items.items()}
  if condition=='full':model=primary;receipt=None
  else:
   model,receipt=calibrate_denoise(primary,observations(views),mode=POLICY,context=CONTEXT,
                                   source_checkpoint_sha256=file_hash(folder/'primary.pt'))
  calibration_seconds=time.monotonic()-start
  for name,view in views.items():
   local={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in view.items()}
   with torch.no_grad():probabilities=forward(model,name,local)
   ids,truth=truths[name];ids=ids.to('cuda');target=truth.to('cuda');c=view['class_num']
   predictions={};packet={};scores={}
   for head,value in zip(('fused','graph','cnn'),probabilities[:3]):
    prob=value.index_select(0,ids)
    assert torch.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
    assert torch.allclose(prob.sum(1),torch.ones(len(ids),device='cuda'),rtol=0,atol=2e-6)
    pred=prob.argmax(1).cpu();truep=prob[torch.arange(len(ids),device='cuda'),target].cpu()
    cm=np.bincount(truth.numpy()*c+pred.numpy(),minlength=c*c).reshape(c,c);recall=cm.diagonal()/cm.sum(1)
    scores[head]={'oa':float(cm.trace()/cm.sum()),'aa':float(recall.mean()),'recall':recall.tolist(),
                  'confusion':cm.tolist(),'nll':float(-truep.clamp_min(1e-8).log().mean())}
    predictions[head]=pred;packet[head]={'pred':pred.to(torch.int16),'truth_probability':truep}
   cnn=predictions['cnn'].eq(truth);graph=predictions['graph'].eq(truth);fused=predictions['fused'].eq(truth)
   one=cnn^graph;bothwrong=~cnn&~graph
   comp={'both_wrong':float(bothwrong.double().mean()),'exactly_one_correct':float(one.double().mean()),
         'fusion_correction_when_one_correct':float(fused[one].double().mean()) if one.any() else None,
         'both_wrong_by_class':[float(bothwrong[truth==i].double().mean()) for i in range(c)]}
   packet_name=f'{case["preset"]}-{case["seed"]}-{condition}-{name}.pt';torch.save(packet,out/packet_name)
   rows.append({'preset':case['preset'],'seed':case['seed'],'condition':condition,'dataset':name,
                'bands':view['bands'],'scores':scores,'complementarity':comp,'packet':packet_name})
   del local,probabilities,prob
  calibrations.append({'preset':case['preset'],'seed':case['seed'],'condition':condition,
                        'seconds':calibration_seconds,'receipt':receipt})
  assert state_digest(primary)==primary_before
  if model is not primary:del model
  del views
  print('EVALUATED',case['preset'],case['seed'],condition,flush=True)
 del primary
 torch.cuda.empty_cache()
assert all(file_hash(p)==h for p,h in protected.items())
assert all(file_hash(p)==h for p,h in launch['inputs'].items());require_frozen_code(source)
write_json(out/'results.json',{'status':'completed','source':source,'evaluator_sha256':file_hash(__file__),'rows':rows,'calibrations':calibrations,
 'full_probability_checks':checks,'train_counts':train_counts,'training_artifacts_unchanged':len(protected),
 'inputs_unchanged':len(launch['inputs']),'optimizer_updates':0,'test_scoring':True,'elapsed_seconds':time.monotonic()-started})
print('EVALUATION_COMPLETED',len(rows),flush=True)

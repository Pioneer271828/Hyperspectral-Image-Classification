import os,sys,json,math,time
from pathlib import Path
import torch
r=Path('/home/lbx/hsi-pioneer271828');repo=r/'checkouts/m4g4b-issue168';os.chdir(repo);sys.path.insert(0,str(repo))
from research_runtime import code_snapshot,write_json
from m4_source_study import M4SourceFullRuntime
from source_data_context import load_source_context
from m4_transfer import graph_binding
from joint_training import load_plan,_tree_fingerprint
from evaluation.g4_target import independent_trainval
from target_initialization import state_id
from model.research_config import ResearchConfig
from loadData.dataset_registry import file_hash
code=code_snapshot();assert code['commit']=='5b4be2aebc528c0442264a18426bb68adceeef5d' and not code['dirty']
d=r/'runs/m4g4b-issue168-full01';launch=json.loads((d/'launch.json').read_text());status=json.loads((d/'status.json').read_text());assert status['status']=='failed'
cases=launch['cases'][:2];assert [c['mode'] for c in cases]==['capacity_control','x_residual']
assert len(list((d/'cases').iterdir()))==2
protected={str(p):file_hash(p) for c in cases for p in Path(c['out']).iterdir() if p.is_file()}
protected.update({str(d/n):file_hash(d/n) for n in ['launch.json','status.json','first-two-updates.json']})
inputs=cases[0]['source_context']['protected_files'];assert all(file_hash(p)==h for p,h in inputs.items())
out=r/'runs/m4g4b-issue168-failure-audit01';out.mkdir(exist_ok=False);started=time.monotonic();rows=[];metrics=recalls=0
ctx=load_source_context(launch['prepared'],launch['data_root'],held_out='paviaU',consumer=code,manifest_sha256=load_plan('configs/experiments/base97_training.json')['prepared_manifest_sha256'])
for c in cases:
 folder=Path(c['out']);s=torch.load(folder/'state-001002.pt',map_location='cpu',weights_only=True)
 primary=torch.load(folder/'primary.pt',map_location='cpu',weights_only=True);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);saved=torch.load(folder/'selected-probabilities.pt',map_location='cpu',weights_only=True)
 assert s['cursor']==1002 and s['start_code']==code and len(s['trace'])==1002 and len(s['history'])==51
 assert s['best_step']==max(s['history'],key=lambda h:h['macro_oa'])['step']==1002
 for k in ('model','optimizer','scheduler'):assert _tree_fingerprint(s[k])==s['state_fingerprints'][k]
 assert _tree_fingerprint(primary)==_tree_fingerprint(s['best_model']) and _tree_fingerprint(last)==s['state_fingerprints']['model'] and _tree_fingerprint(saved)==_tree_fingerprint(s['best_probabilities'])
 meta=json.loads((folder/'primary.pt.json').read_text())['training_config'];assert meta['research_config']==c['settings']['config'] and meta['provenance']==c['provenance'] and meta['checkpoint_optimizer_step']==1002
 assert ctx.record==c['source_context'] and graph_binding(ctx.items)==c['graph_binding']
 selected=s['history'][-1]
 for name,item in ctx.items.items():
  v={'class_num':item['class_num']}
  for phase in ['train','val']:
   gt=item[phase+'_samples_gt'];v[phase+'_labels']=gt[gt>0].long()-1
  z=independent_trainval(saved[name],v,selected['metrics'][name]);metrics+=z['head_metrics'];recalls+=z['class_recalls']
 with torch.random.fork_rng(devices=[0]):
  torch.random.default_generator.manual_seed(c['seed']);initial=ResearchConfig.from_dict(c['settings']['config']).build_model(ctx.class_counts)
  assert state_id(initial)==c['initial_state_sha256'];changes={}
  for label,prefix in {'adapter':'backbone.adapter.','denoise':'backbone.CNN_denoise.','cnn':'backbone.CNN_Branch.','graph':'backbone.HGNN_Branch.','m4':'backbone.region_residual.'}.items():
   params={n:p.detach() for n,p in initial.named_parameters() if n.startswith(prefix)};assert params
   changes[label]=sum(int(torch.count_nonzero(last[n]-p)) for n,p in params.items());assert changes[label]>0
  del initial
  gpuitems={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in item.items()} for n,item in ctx.items.items()}
  runtime=M4SourceFullRuntime();model=runtime.load_selected(folder/'primary.pt',gpuitems,folder/'raw-validation-001002.pt',device='cuda');assert _tree_fingerprint(model.state_dict())==_tree_fingerprint(s['best_model'])
  original_model=state_id(model);checks=[]
  for repeat in range(3):
   _,values=runtime.validation_snapshot(model,gpuitems)
   for name in values:
    for phase in ['train','val']:
     for head in ['fused','cnn','graph']:
      old=saved[name][phase][head].cpu();new=values[name][phase][head].detach().cpu();delta=(new-old).abs();limit=1e-6+1e-5*old.abs()
      checks.append({'repeat':repeat,'task':name,'phase':phase,'head':head,'max_abs_error':float(delta.max()),'original_tolerance_violations':int((delta>limit).sum()),'argmax_changes':int((new.argmax(1)!=old.argmax(1)).sum())})
  assert state_id(model)==original_model;del model,gpuitems;torch.cuda.empty_cache()
 rows.append({'name':c['name'],'mode':c['mode'],'updates':1002,'history_events':51,'selected_step':1002,'native_completed_result_exists':(folder/'result.json').exists(),'source_val_macro_oa':selected['macro_oa'],'training_fused_oa':{n:selected['metrics'][n]['train']['fused']['oa'] for n in ctx.sources},'loss_first30':sum(x['loss']['total'] for x in s['trace'][:30])/30,'loss_last30':sum(x['loss']['total'] for x in s['trace'][-30:])/30,'last50epoch_validation_gain':s['history'][-1]['macro_oa']-s['history'][-6]['macro_oa'],'m4_positive_gradient_steps':sum(x['m4_parameter_gradient_norm']>0 for x in s['trace']),'changed_parameters':changes,'native_repeats':checks})
 del s,primary,last,saved
assert all(file_hash(p)==h for p,h in protected.items()) and all(file_hash(p)==h for p,h in inputs.items()) and code_snapshot()==code
summary={'status':'diagnostic_completed_original_gate_failure_retained','code':code,'actual_training_updates':2004,'planned_training_updates':18036,'planned_cases':18,'completed_cases':1,'failed_finalization_cases':1,'not_started_cases':16,'independent_head_metrics':metrics,'independent_class_recalls':recalls,'protected_training_files_unchanged':len(protected),'protected_inputs_unchanged':len(inputs),'new_optimizer_updates':0,'target_query_read':False,'original_gate_relaxed':False,'elapsed_seconds':time.monotonic()-started,'cases':rows}
write_json(out/'summary.json',summary);print('M4G4B_FAILURE_AUDIT_DONE',metrics,recalls,[(x['mode'],sum(z['argmax_changes'] for z in x['native_repeats']),sum(z['original_tolerance_violations'] for z in x['native_repeats'])) for x in rows],flush=True)

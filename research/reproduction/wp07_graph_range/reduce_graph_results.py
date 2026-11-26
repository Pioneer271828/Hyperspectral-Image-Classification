"""Independent pairing, gate-intervention invariants, parameter and timing reduction."""
import csv,hashlib,json,math,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['WP07_GRAPH_SOURCE'])
import torch
from transfer.wp07_graph_range import require,require_checks,initialize
from transfer.wp07_graph_training import BankProvider
from transfer.native_scale_campaign import read,packet
from transfer import registered_development as dev
from transfer.g3_training import state_hash


def write(path,rows):
 with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def union_seconds(intervals):
 intervals=sorted(intervals);total=0;left=right=None
 for a,b in intervals:
  if left is None:left,right=a,b
  elif a<=right:right=max(right,b)
  else:total+=right-left;left,right=a,b
 return (total+(right-left if left is not None else 0))/1e9

def main():
 require_checks();torch.set_num_threads(1);base=Path(sys.argv[1]);root=base.parent;plan=read(dev._pin(base/'pair/plan.json'),root);receipt=read(plan['preflight'],root)
 costs=[];parameters=[];gate_rows=[];init={};stages={};prediction_count=0
 for f in sorted(base.glob('lane-gpu*.log')):
  for line in f.read_text().splitlines():
   if line.startswith('{'):
    e=json.loads(line)
    if 'phase' in e:
     k=(e['case'],e['phase']);require(k not in stages,'Duplicate committed phase');stages[k]=e
 require(len(stages)==36,'Expected 36 committed phases')
 for case in plan['cases']:
  d=Path(case['destination']);spec=read(dev._pin(d/'registration.json'),root);data=read(dev._pin(d/'independent.json'),root);state=packet(read(dev._pin(d/'completed.json'),root)['checkpoint'],root)
  provider=BankProvider(root,receipt,case['seed']);initial=initialize(provider.baseline(case['seed']),spec['module_seed']);require(state_hash(initial)==spec['initialization']['state'],'Initial model fingerprint differs')
  init.setdefault(case['seed'],set()).add(spec['initialization']['state'])
  changed=[k for k,v in initial.state_dict().items() if not torch.equal(v,state['model'][k])]
  parameters.append({'case':case['id'],'parameters':spec['parameters'],'state_tensors_changed':len(changed),'range_gate_changed':sum(k.startswith('backbone.range_content.') for k in changed),'graph_feature_norm_changed':sum(k.startswith('backbone.learnable_graph.feature_norm.') for k in changed),'assignment_or_edge_scorer_changed':sum(k.startswith(tuple('backbone.learnable_graph.'+n+'.' for n in ['pixel_query','seed_key','graph_query','graph_key'])) for k in changed),'gate_update_steps':sum(v for v in data['fresh_updates'].values())})
  for endpoint in data['endpoints'].values():prediction_count+=len(endpoint['original_point_predictions'])
  stage_seconds=sum(stages[case['id'],p]['wall_seconds'] for p in ['prefix','resume','completed'])
  costs.append({'case':case['id'],'stage_wall_seconds_sum':stage_seconds,'first_audit_wall_seconds':read(dev._pin(d/'audit.json'),root)['audit_wall_seconds'],'independent_audit_wall_seconds':data['audit_wall_seconds']})
  groups=[('train',state['history']),('selected_devval',data['endpoints']['selected']['validation']['rows']),('last_devval',data['endpoints']['last']['validation']['rows'])]
  for scope,rows in groups:
   for r in rows:
    a,b=r['graph']['weights']['graph']['weights'];gate_rows.append({'case':case['id'],'scope':scope,'step':r.get('step',data['selected_step'] if scope=='selected_devval' else 60),'tile':r['tile'],'band':r['band'],'near_weight':a,'wide_weight':b,'preference':'near' if a>b else 'wide' if b>a else 'tie','note':'both paths execute; preference is not hard routing'})
 require(all(len(v)==1 for v in init.values()) and len({next(iter(v)) for v in init.values()})==3,'Paired complete initial states differ')
 require(prediction_count==6480,'Independent point prediction count differs')
 write(base/'results/case-cost.csv',costs);write(base/'results/parameter-participation.csv',parameters);write(base/'results/selection-weights.csv',gate_rows)
 cf=[];cfclasses=[];invariants=0
 for folder in sorted(base.glob('counterfactual-gpu*-output')):
  ok=read(dev._pin(folder/'receipt.json'),root);require(ok['status']=='passed' and ok['optimizer_updates']==0,'Counterfactual receipt differs')
  for item in ok['cases']:
   d=read(item,root);require(d['checkpoint_unchanged'] and d['optimizer_updates']==0,'Counterfactual changed checkpoint')
   seed=int(d['case'].split('/')[1])
   for endpoint,controls in d['comparisons'].items():
    reference=controls['content']['validation']['rows']
    for control,values in controls.items():
     if control!='content':
      for a,b in zip(reference,values['validation']['rows'],strict=True):
       require(a['heads']['cnn']==b['heads']['cnn'] and a['released']==b['released'] and a['observation']==b['observation'] and a['graph']['range_identity']==b['graph']['range_identity'],'Counterfactual changed CNN or input/frozen support');invariants+=1
     for head,bands in values['metrics'].items():
      for band,m in bands.items():
       require(m['count']==45,'Counterfactual denominator differs');common={'seed':seed,'checkpoint':d['case'],'endpoint':endpoint,'coefficient':control,'head':head,'band':band,'scope':'same_content_checkpoint_zero_update'}
       cf.append({**common,**{k:m[k] for k in ['oa','aa','nll']}})
       for i,c in enumerate(m['per_class']):cfclasses.append({**common,'class_id':c['class_id'],'correct':m['confusion_matrix'][i][i],'support':c['support'],'recall':c['recall']})
 require(invariants==1476 and len(cf)==144 and len(cfclasses)==1296,'Counterfactual coverage differs')
 write(base/'results/counterfactual-endpoints.csv',cf);write(base/'results/counterfactual-per-class.csv',cfclasses)
 ci={(r['seed'],r['endpoint'],r['coefficient'],r['head'],r['band']):r for r in cf};pc={(r['seed'],r['endpoint'],r['coefficient'],r['head'],r['band'],r['class_id']):r for r in cfclasses};pairs=[];classpairs=[]
 for seed in [64,512,2048]:
  for endpoint in ['selected','last']:
   for a,b in [('content','mean'),('wide','near'),('mean','near'),('content','near')]:
    for head in ['fused','graph','cnn']:
     for band in ['full','even_retained_positions']:
      x,y=(ci[seed,endpoint,z,head,band] for z in [a,b]);common={'seed':seed,'endpoint':endpoint,'candidate':a,'baseline':b,'head':head,'band':band};guards=[]
      for cls in range(1,10):
       c,d=(pc[seed,endpoint,z,head,band,cls] for z in [a,b]);delta=c['recall']-d['recall'];guards.append(delta>=-.02);classpairs.append({**common,'class_id':cls,'candidate_correct':c['correct'],'baseline_correct':d['correct'],'support':5,'recall_delta':delta,'protected':guards[-1]})
      pairs.append({**common,**{k+'_delta':x[k]-y[k] for k in ['oa','aa','nll']},'all_classes_protected':all(guards)})
 write(base/'results/counterfactual-pairs.csv',pairs);write(base/'results/counterfactual-class-pairs.csv',classpairs)
 jobs=[{'name':f.stem.replace('.timing',''),**json.loads(f.read_text())} for f in base.glob('*.timing.json')]
 require(all(j['exit_code']==0 for j in jobs),'A traced computation failed')
 intervals=[(j['start_monotonic_ns'],j['end_monotonic_ns']) for j in jobs]
 timing={'tracked_jobs':jobs,'parallel_job_union_seconds':union_seconds(intervals),'tracked_job_wall_seconds_sum_not_net':sum(j['elapsed_seconds'] for j in jobs),'child_cpu_core_seconds_sum':sum(j['child_user_cpu_seconds']+j['child_system_cpu_seconds'] for j in jobs),'resource_threshold_sleep_seconds':0,'other_pure_wait_seconds':None,'pure_gpu_kernel_seconds':None,'scope':'from final qualification onward; early geometry/tests, development/transport/reporting not fully instrumented; jobs may overlap, never sum their wall times as net duration'}
 (base/'results/timing.json').write_text(json.dumps(timing,indent=2)+'\n')
 result={'status':'passed','worker_commit':plan['producer'],'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'real_cases':12,'real_updates':720,'committed_phases':36,'independent_point_predictions':prediction_count,'complete_initial_state_equal_within_each_seed':True,'same_weights_CNN_input_and_frozen_support_invariant_pairs':invariants,'counterfactual_endpoints':len(cf),'counterfactual_class_rows':len(cfclasses),'stage_wall_seconds_sum_not_parallel_net':sum(c['stage_wall_seconds_sum'] for c in costs),'parallel_tracked_job_union_seconds':timing['parallel_job_union_seconds'],'query_accessed':False,'raw_gt_loaded':False}
 (base/'results/reduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

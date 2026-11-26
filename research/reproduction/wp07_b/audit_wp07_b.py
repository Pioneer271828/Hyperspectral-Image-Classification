"""Independent zero-update replay of B's pinned checkpoints and released points."""
import csv, json, math, os, sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
sys.path.insert(0, os.environ['WP07_B_FROZEN_WORKER'])
from research_runtime import code_snapshot, hash_file, require_frozen_code
from scripts import run_wp07_b_pair as b
from scripts import run_native_released_points as worker
from transfer import native_released_campaign as c, native_released_real as real
from transfer import registered_development as transport
from transfer.native_scale_campaign import packet,read
from transfer.native_scale_training import Run
from transfer.g3_training import state_hash
from transfer.native_scale_observation import head_hashes
from monitoring.native_region_analysis import stages
from evaluation.classification_metrics import metrics_from_confusion

@torch.no_grad()
def audit(root, plan_path, gpu, output):
 code=code_snapshot();require_frozen_code(code)
 assert code['commit']=='96921a409f1ce69ac500b35ceffa5b6aabd4ff20'
 plan=b.load_plan(root,transport._pin(plan_path),code)
 output.mkdir();rows=[]
 with b.observation_only_resources():
  for case in plan['cases']:
   if case['gpu']!=gpu:continue
   started=perf_counter();guard,dest=c.authorized_case(root,case,case['approval'],code,intake=None)
   spec=read(transport._pin(dest/'registration.json'),root)
   state=packet(read(transport._pin(dest/'completed.json'),root)['checkpoint'],root)
   previous=read(transport._pin(dest/'b-audit.json'),root)
   with worker.arithmetic(root,gpu) as execution:
    run=Run(spec,code['commit'],execution=execution,released=guard);run.restore(state,62)
    initial=state_hash(run.model);model_states={}
    for endpoint,weights,historical,field in [('selected',state['selected']['state'],min(state['validation'],key=lambda r:r['fused_nll']),'head_metrics'),('last',state['model'],state['validation'][-1],'last_head_metrics')]:
     run.model.load_state_dict(weights,strict=True);run.model.eval();before=state_hash(run.model)
     metrics={};support=[];points=[]
     for event in spec['validation']:
      actual,labels,targets=run._materialize('devval',event);captured=[]
      def first(invoke,values,a,dataset):
       graph=a.coalesce();mask=graph.values()>0
       captured.append((graph.indices()[:,mask].cpu().numpy(),graph.values()[mask].cpu().numpy()))
       return invoke(values,a,dataset)
      def second(invoke,values,a,dataset):return invoke(values,a,dataset)
      with stages(run.model.backbone,first,second):out,trace,_=run._forward(actual,'devval','missing')
      saved=next(r for r in historical['rows'] if r['tile']==event['tile'] and r['band']==event['band'])
      assert head_hashes(out,actual)==saved['heads'] and trace==saved['graph']
      n=trace['q_shape'][1];edges,values=captured[0]
      parts=[]
      for k in range(2 if trace['range_nnz'] and len(trace['range_nnz'])==2 else 1):
       mask=(edges[0]>=k*n)&(edges[0]<(k+1)*n)&(edges[1]>=k*n)&(edges[1]<(k+1)*n)
       parts.append({'edges':(edges[:,mask]-k*n).tolist(),'values':values[mask].tolist()})
      support.append({'tile':event['tile'],'band':event['band'],'graph':trace,'supports':parts,'identical_positive_support':len(parts)==2 and parts[0]['edges']==parts[1]['edges'],'identical_weighted_support':len(parts)==2 and parts[0]==parts[1]})
      y0,_,x0,_=actual.lattice.box; width=actual.lattice.scene_shape[1]; f=actual.lattice.factor; w=actual.lattice.shape[1]
      for head,tensor in zip(('fused','graph','cnn'),out[:3]):
       p=tensor.cpu().numpy(); key=(head,event['band']);bucket=metrics.setdefault(key,{'cm':np.zeros((9,9),dtype=np.int64),'nll':[]})
       for ordinal,(pixel,label) in enumerate(targets.points):
        y,x=divmod(pixel,width);cell=((y-y0)//f)*w+(x-x0)//f;prob=p[cell].astype(float)
        bucket['cm'][label-1,int(prob.argmax())]+=1;bucket['nll'].append(-math.log(float(np.clip(prob[label-1],1e-8,1))))
        points.append({'endpoint':endpoint,'head':head,'band':event['band'],'tile':event['tile'],'released_ordinal':ordinal,'class_id':label,'probabilities':prob.tolist()})
     result={}
     for (head,band),v in metrics.items():
      score=metrics_from_confusion(v['cm'],range(1,10));score['nll']=math.fsum(v['nll'])/len(v['nll'])
      expected=previous[field][head][band]
      assert score['confusion_matrix']==expected['confusion_matrix'] and len(v['nll'])==45
      assert abs(score['nll']-expected['nll'])<=1e-12
      result.setdefault(head,{})[band]=score
     assert state_hash(run.model)==before
     model_states[endpoint]={'metrics':result,'support':support,'points':points,'state':before}
    assert state_hash(run.model)==initial
   result={'case':case['id'],'worker_commit':code['commit'],'script_sha256':hash_file(__file__),'optimizer_updates':0,'query_accessed':False,'raw_gt_loaded':False,'endpoints':model_states,'wall_seconds':perf_counter()-started}
   pin=transport._json_new(output/(case['id'].replace('/','-')+'.json'),result);rows.append(pin)
   print(json.dumps({'case':case['id'],'independent':pin,'wall_seconds':result['wall_seconds']}),flush=True)
 transport._json_new(output/'receipt.json',{'status':'passed','cases':rows,'script_sha256':hash_file(__file__),'worker_commit':code['commit'],'optimizer_updates':0,'query_accessed':False,'raw_gt_loaded':False})

if __name__=='__main__':
 root=Path('/home/lbx/hsi-pioneer271828/runs');out=Path(sys.argv[1]);audit(root,root/'wp07-b-20261126/pair-v2/plan.json',os.environ['CUDA_VISIBLE_DEVICES'],out)

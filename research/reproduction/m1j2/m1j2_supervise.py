"""Frozen six-case training only; no test scoring until all cases are audited."""
import json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/home/lbx/hsi-pioneer271828/checkouts/m1j2-issue127');BASE=ROOT.parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import numpy as np
from research_runtime import code_snapshot,require_frozen_code,write_json
from loadData.dataset_registry import file_hash
from loadData.data_read import read_band_metadata
from joint_training import import_preparation,load_plan,schedule_for
from joint_observation_study import load_study,case_settings,evaluation_bank

SHA='9942c0f31949bed746f3038e26b53b51d0dddc26';source=code_snapshot();assert source['commit']==SHA and not source['dirty']
out=BASE/'runs/m1j2-issue127-full01';out.mkdir(exist_ok=False)
for n in ('logs','reviews','cases'):(out/n).mkdir()
load=lambda p:json.loads(Path(p).read_text())
started=time.monotonic();study=load_study();base_plan=load_plan('configs/experiments/base97_training.json')
prep=BASE/'runs/base97a-honghu-issue97-20261103-prepared01';data=BASE/'datasets/snapshots/joint-honghu-20261103-issue97/dataset'
baseline,items,provenance=import_preparation(prep,data,base_plan,source)
inputs=load(BASE/'runs/bn-align2-issue105-20261103-replay01/plan.json')['input_files']
assert all(file_hash(p)==h for p,h in inputs.items())
metas={n:read_band_metadata(n,i['bands']) for n,i in items.items()}
bank=evaluation_bank(study,metas,provenance['prepared_manifest_sha256'])
gpus=[7,6];cases=[]
for seed in study['training_seeds']:
 for preset in study['presets']:
  path=out/'cases'/preset/f'seed-{seed}';path.parent.mkdir(exist_ok=True)
  settings=case_settings(study,base_plan,baseline,items,provenance,preset,seed)
  cases.append({'preset':preset,'seed':seed,'out':str(path),'settings':settings})
launch={'code':source,'study':study,'prepared':str(prep),'data_root':str(data),'provenance':provenance,
 'inputs':inputs,'cases':cases,'evaluation_bank':bank,'supervisor_sha256':file_hash(__file__),
 'gpus':gpus,'test_scoring':False,'expected_updates':9018}
write_json(out/'launch.json',launch);launch_hash=file_hash(out/'launch.json');del items
jobs=[]
for i,case in enumerate(cases):
 cmd=[sys.executable,str(ROOT/'scripts/train_joint_observation_full.py'),'--expected-commit',SHA,
      '--launch-plan',str(out/'launch.json'),'--launch-sha256',launch_hash,'--preset',case['preset'],'--seed',str(case['seed'])]
 jobs.append({'preset':case['preset'],'seed':case['seed'],'out':case['out'],'gpu':gpus[i%2],
              'status':'queued','command':cmd})
lock=threading.RLock();stop=threading.Event()
def publish(final=None):
 with lock:write_json(out/'status.json',{'status':final or ('failed' if stop.is_set() else 'running'),
    'elapsed_seconds':time.monotonic()-started,'jobs':jobs,'test_scoring':False,'launch_sha256':launch_hash})
def update(job,**values):
 with lock:job.update(**values);publish()
def invoke(command,env,log):
 with log.open('x') as stream:
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  return proc.wait()
def review(job,env):
 folder=Path(job['out']);result=load(folder/'result.json');trace=load(folder/'trace.json');history=load(folder/'history.json')
 case=next(c for c in cases if (c['preset'],c['seed'])==(job['preset'],job['seed']))
 assert result['settings']==case['settings'] and result['start_code']==source
 assert result['status']=='completed' and result['optimizer_steps']==len(trace)==1503 and len(history)==51
 assert [r['dataset'] for r in trace]==schedule_for(case['settings'])
 assert result['selected_step']==max(history,key=lambda e:e['macro_oa'])['step']
 for i,t in enumerate(trace):
  event=case['settings']['joint_observation']['records'][i]
  assert t['step']==i+1 and t['observation']['record_hash']==event['observation']['record_hash']
  assert t['observation']['bands']==len(event['observation']['selected_positions'])
 assert all(np.isfinite(v) for t in trace for v in t['loss'].values())
 assert all(np.isfinite(v) for t in trace for v in t['gradient_norms'].values())
 assert sum(t['graph_changed_elements'] for t in trace)>0
 before={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
 log=out/'logs'/f'{job["preset"]}-{job["seed"]}-resume.log'
 assert invoke(job['command']+['--resume'],env,log)==0
 answer=json.loads(log.read_text().strip().splitlines()[-1]);assert answer['updates_this_call']==0
 assert before=={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
 receipt={'code':source,'preset':job['preset'],'seed':job['seed'],'updates':1503,'selected_step':result['selected_step'],
  'loss_first30':float(np.mean([t['loss']['total'] for t in trace[:30]])),
  'loss_last30':float(np.mean([t['loss']['total'] for t in trace[-30:]])),
  'last50_epoch_validation_gain':history[-1]['macro_oa']-history[-6]['macro_oa'],
  'selected_validation':result['selected_validation'],'artifacts':before,'test_scoring':False,
  'observed_band_counts':sorted({(t['dataset'],t['observation']['bands']) for t in trace}),
  'adapter_positive_gradient_steps':sum(t['gradient_norms']['adapter']>0 for t in trace)}
 write_json(out/'reviews'/f'{job["preset"]}-{job["seed"]}.json',receipt)
def worker(slot):
 for job in jobs[slot::2]:
  if stop.is_set():return
  try:
   require_frozen_code(source)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu']),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
   while True:
    free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(','))
    if free>=11264 and util<=60:break
    update(job,status='waiting_resource');time.sleep(15)
    if stop.is_set():return
   log=out/'logs'/f'{job["preset"]}-{job["seed"]}.log';update(job,status='running',log=str(log));t=time.monotonic()
   code=invoke(job['command'],env,log);update(job,exit_code=code,training_seconds=time.monotonic()-t)
   if code:
    folder=Path(job['out']);progress=load(folder/'progress.json')
    assert progress['cursor']==1503 and all((folder/n).is_file() for n in ('primary.pt','last.pt','selected-probabilities.pt'))
    update(job,status='zero_update_finalization',original_failure_preserved=True)
    recovery=out/'logs'/f'{job["preset"]}-{job["seed"]}-finalize.log'
    if invoke(job['command']+['--finalize-only'],env,recovery):raise RuntimeError('Additive finalization failed')
   update(job,status='reviewing');review(job,env);update(job,status='completed')
   print('CASE_COMPLETE',job['preset'],job['seed'],job['training_seconds'],flush=True)
  except Exception as error:
   stop.set();update(job,status='failed',error=repr(error));traceback.print_exc();return
publish()
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
require_frozen_code(source);assert all(file_hash(p)==h for p,h in inputs.items())
status='training_completed' if all(j['status']=='completed' for j in jobs) else 'failed'
publish(status);print('CAMPAIGN_FINISHED',status,flush=True)

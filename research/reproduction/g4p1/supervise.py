"""Freeze nine source cases, then run two isolated resource-gated GPU queues."""
import argparse,json,os,sys,time,threading,subprocess,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from source_data_context import load_source_context
from source_pretrain import load_study,full_settings,select_case
from joint_training import load_plan,schedule_for
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json
from loadData.dataset_registry import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpus',type=int,nargs=2);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-existing',action='store_true')
    a=p.parse_args();source=code_snapshot();study=load_study()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    if a.prepare_only and a.run_existing:raise ValueError('Preparation and existing launch are exclusive')
    if not a.prepare_only and (a.gpus is None or len(set(a.gpus))!=2 or min(a.gpus)<0):raise ValueError('Two distinct nonnegative GPU indices required')
    out=external_directory(a.out_dir);plan=load_plan('configs/experiments/base97_training.json')
    load=lambda p:json.loads(Path(p).read_text())
    if a.run_existing:
        launch=load(out/'launch.json')
        if (launch['code']!=source or launch['study']!=study or launch['prepared']!=a.prepared_dir
                or launch['data_root']!=a.data_root):raise ValueError('Existing source launch identity mismatch')
        for c in launch['cases']:select_case(launch,source,c['held_out'],c['seed'])
    else:
        out.mkdir(exist_ok=False)
        for n in ('cases','logs','reviews'):(out/n).mkdir()
        cases=[]
        for held in study['held_targets']:
            context=load_source_context(a.prepared_dir,a.data_root,held_out=held,consumer=source,
                manifest_sha256=plan['prepared_manifest_sha256'])
            for seed in study['training_seeds']:
                name=held+'-s'+str(seed);settings=full_settings(held,seed)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed);model=context.build_model()
                assert set(model.heads)==set(context.sources) and not any(held in n.split('.') for n in model.state_dict())
                cases.append({'name':name,'held_out':held,'seed':seed,'settings':settings,
                    'out':str(out/'cases'/name),'source_context':context.record,
                    'registered_parameters':sum(p.numel() for p in model.parameters())})
                del model
            del context
        launch={'code':source,'study':study,'cases':cases,'prepared':a.prepared_dir,'data_root':a.data_root,
            'test_scoring':False,'expected_updates':9018,'supervisor_sha256':file_hash(__file__)}
        for c in cases:select_case(launch,source,c['held_out'],c['seed'])
        require_frozen_code(source);write_json(out/'launch.json',launch)
    launch_hash=file_hash(out/'launch.json')
    if a.prepare_only:
        write_json(out/'preflight.json',{'status':'passed','source':source,'launch_sha256':launch_hash,
            'case_count':9,'optimizer_updates':0,'target_payload_in_source_case':False,'test_scoring':False})
        print('SOURCE_PREFLIGHT_PASSED',launch_hash,flush=True);return
    started=time.monotonic();jobs=[];lock=threading.RLock();stop=threading.Event()
    for index,case in enumerate(launch['cases']):
        command=[sys.executable,str(ROOT/'scripts/train_source_full.py'),'--expected-commit',a.expected_commit,
            '--launch-plan',str(out/'launch.json'),'--launch-sha256',launch_hash,'--held-out',case['held_out'],'--seed',str(case['seed'])]
        jobs.append({'name':case['name'],'held_out':case['held_out'],'seed':case['seed'],'out':case['out'],
            'command':command,'gpu':a.gpus[index%2],'status':'queued'})
    def publish(final=None):
        with lock:write_json(out/'status.json',{'status':final or ('failed' if stop.is_set() else 'running'),
            'source':source,'jobs':jobs,'elapsed_seconds':time.monotonic()-started,'launch_sha256':launch_hash,'test_scoring':False})
    def update(job,**values):
        with lock:job.update(**values);publish()
    def invoke(cmd,env,log):
        with log.open('x') as stream:return subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
    def review(job,env):
        folder=Path(job['out']);case=select_case(launch,source,job['held_out'],job['seed'])
        result=load(folder/'result.json');trace=load(folder/'trace.json');history=load(folder/'history.json')
        assert result['status']=='completed' and result['settings']==case['settings'] and result['start_code']==source
        assert result['optimizer_steps']==len(trace)==1002 and len(history)==51
        assert [x['dataset'] for x in trace]==schedule_for(case['settings'],task_names=tuple(case['settings']['source_tasks']))
        assert [x['step'] for x in trace]==list(range(1,1003)) and job['held_out'] not in result['per_dataset_updates']
        assert result['selected_step']==max(history,key=lambda x:x['macro_oa'])['step']
        assert set(result['selected_validation']['metrics'])==set(case['settings']['source_tasks'])
        assert all(np.isfinite(v) for x in trace for v in x['loss'].values())
        assert all(np.isfinite(v) for x in trace for v in x['gradient_norms'].values())
        assert sum(x['graph_changed_elements'] for x in trace)>0
        before={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
        log=out/'logs'/(job['name']+'-completed-resume.log')
        assert invoke(job['command']+['--resume'],env,log)==0
        assert load_last(log)['updates_this_call']==0 and before=={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
        assert all(file_hash(p)==h for p,h in case['source_context']['protected_files'].items())
        receipt={'source':source,'name':job['name'],'updates':1002,'selected_step':result['selected_step'],
            'loss_first30':float(np.mean([x['loss']['total'] for x in trace[:30]])),
            'loss_last30':float(np.mean([x['loss']['total'] for x in trace[-30:]])),
            'last50epoch_validation_gain':history[-1]['macro_oa']-history[-6]['macro_oa'],
            'selected_validation':result['selected_validation'],'artifacts':before,'test_scoring':False,
            'positive_gradient_steps':{k:sum(x['gradient_norms'][k]>0 for x in trace) for k in trace[0]['gradient_norms']}}
        write_json(out/'reviews'/(job['name']+'.json'),receipt)
    def worker(gpu):
        for job in [j for j in jobs if j['gpu']==gpu]:
            if stop.is_set():return
            try:
                require_frozen_code(source)
                while True:
                    free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    if free>=12288 and util<=60:break
                    update(job,status='waiting_resource');time.sleep(15)
                    if stop.is_set():return
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                command=job['command']+(['--resume'] if Path(job['out']).exists() else [])
                log=out/'logs'/(job['name']+'.log');update(job,status='running',log=str(log));t=time.monotonic()
                rc=invoke(command,env,log);update(job,exit_code=rc,training_seconds=time.monotonic()-t)
                if rc:raise RuntimeError('Native source training/finalization failed; retained without numerical relaxation')
                update(job,status='reviewing');review(job,env);update(job,status='completed');print('SOURCE_CASE_COMPLETE',job['name'],flush=True)
            except Exception as error:
                stop.set();update(job,status='failed',error=repr(error));traceback.print_exc();return
    def load_last(path):return json.loads(path.read_text().strip().splitlines()[-1])
    with (out/'.supervisor.lock').open('x') as stream:stream.write(str(os.getpid()))
    try:
        publish()
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,a.gpus))
        require_frozen_code(source);state='training_completed' if all(j['status']=='completed' for j in jobs) else 'failed'
        publish(state);print('SOURCE_CAMPAIGN_FINISHED',state,flush=True)
        if state=='failed':raise SystemExit(1)
    finally:(out/'.supervisor.lock').unlink()


if __name__=='__main__':main()

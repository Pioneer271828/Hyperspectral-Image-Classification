"""Freeze27 target cases; resource-gated independent queues, no query labels."""
import argparse,copy,json,os,shutil,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from joint_baseline import NAMES
from joint_training import schedule_for
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,external_directory,require_frozen_code,write_json
from target_data_context import load_target_context
from target_initialization import initialize_target,METHODS
from target_study import (load_study,full_settings,case_name,select_case,validate_source_case)


def prepare(source,out,observations,families,sources):
    out=external_directory(out);out.mkdir(exist_ok=False)
    for name in ('cases','logs','reviews','operation-costs'):(out/name).mkdir()
    if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('Need20GiB artifact headroom; no files removed')
    cases=[]
    for target in NAMES:
        graph=Path(observations)/target;family=Path(families)/(target+'-64')/'curator-family.json'
        inputs={'observation':str((graph/'observation.pt').resolve()),'record':str((graph/'record.json').resolve()),
            'record_sha256':file_hash(graph/'record.json'),'family':str(family.resolve()),'family_sha256':file_hash(family)}
        data,base_provenance,guard=load_target_context(target,inputs['observation'],inputs['record'],inputs['record_sha256'],
            inputs['family'],inputs['family_sha256'],per_class=10,support_seed=64)
        for seed in (64,512,2048):
            source_dir=Path(sources)/(target+'-s'+str(seed));result_path=source_dir/'result.json'
            result=json.loads(result_path.read_text());spec={'source_path':str((source_dir/'primary.pt').resolve()),
                'source_weight_sha256':result['artifacts']['primary.pt'],'source_sidecar_sha256':result['artifacts']['primary.pt.json']}
            binding={'path':str(result_path.resolve()),'sha256':file_hash(result_path)}
            for method in METHODS:
                name=case_name(target,seed,method);scratch=method==METHODS[0]
                provenance=copy.deepcopy(base_provenance);provenance['source_spec']={} if scratch else spec
                if not scratch:provenance['protected_files'].update({spec['source_path']:spec['source_weight_sha256'],
                    spec['source_path']+'.json':spec['source_sidecar_sha256'],binding['path']:binding['sha256']})
                case={'target':target,'seed':seed,'method':method,'name':name,'settings':full_settings(target,method,seed),
                    'out':str(out/'cases'/name),'inputs':inputs,'source_spec':{} if scratch else spec,
                    'source_result':None if scratch else binding,'provenance':provenance}
                validate_source_case(case)
                model,initial=initialize_target(target,method,model_seed=seed,head_seed=seed+1000,
                    device='cpu',**case['source_spec'])
                case['initial_state_sha256']=initial['initial_state_sha256'];del model
                cases.append(case)
        guard();del data
    launch={'code':source,'study':load_study(),'out_dir':str(out),'cases':cases,'query_labels_accessed':False,
        'input_roots':{'observations':str(Path(observations).resolve()),'label_families':str(Path(families).resolve()),
            'source_runs':str(Path(sources).resolve())},
        'test_scoring':False,'expected_updates':40500,'supervisor_sha256':file_hash(__file__)}
    for c in cases:select_case(launch,source,c['target'],c['seed'],c['method'])
    require_frozen_code(source);write_json(out/'launch.json',launch)
    write_json(out/'preflight.json',{'status':'passed','source':source,'launch_sha256':file_hash(out/'launch.json'),
        'case_count':27,'optimizer_updates':0,'query_labels_accessed':False,'test_scoring':False})
    return launch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','observations','label-families','source-runs','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpus',type=int,nargs=2);p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-existing',action='store_true')
    a=p.parse_args();source=code_snapshot();study=load_study()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    if a.prepare_only and a.run_existing:raise ValueError('Prepare and existing execution are exclusive')
    if not a.prepare_only and (a.gpus is None or len(set(a.gpus))!=2 or min(a.gpus)<0):raise ValueError('Two distinct GPU indices required')
    out=external_directory(a.out_dir)
    if a.run_existing:
        launch=json.loads((out/'launch.json').read_text())
        if launch['out_dir']!=str(out):raise ValueError('Existing output root differs')
        if launch['input_roots']!={'observations':str(Path(a.observations).resolve()),
                'label_families':str(Path(a.label_families).resolve()),'source_runs':str(Path(a.source_runs).resolve())}:
            raise ValueError('Existing input roots differ')
        for c in launch['cases']:select_case(launch,source,c['target'],c['seed'],c['method'])
        if launch['supervisor_sha256']!=file_hash(__file__):raise ValueError('Frozen controller changed')
    else:launch=prepare(source,out,a.observations,a.label_families,a.source_runs)
    launch_hash=file_hash(out/'launch.json')
    if a.prepare_only:print('TARGET_FULL_PREFLIGHT_PASSED',launch_hash,flush=True);return
    started=time.monotonic();lock=threading.RLock();stop=threading.Event();jobs=[]
    for index,c in enumerate(launch['cases']):
        command=[sys.executable,str(ROOT/'scripts/train_target_full.py'),'--expected-commit',a.expected_commit,
            '--launch-plan',str(out/'launch.json'),'--launch-sha256',launch_hash,'--target',c['target'],
            '--seed',str(c['seed']),'--method',c['method']]
        jobs.append({'name':c['name'],'command':command,'out':c['out'],'gpu':a.gpus[index%2],'status':'queued'})
    def publish(final=None):
        with lock:write_json(out/'status.json',{'status':final or ('failed' if stop.is_set() else 'running'),
            'source':source,'jobs':jobs,'launch_sha256':launch_hash,'elapsed_seconds':time.monotonic()-started,
            'query_labels_accessed':False,'test_scoring':False})
    def update(job,**values):
        with lock:job.update(**values);publish()
    def invoke(command,env,log):
        with log.open('x') as stream:return subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
    def review(job,case,env,attempt):
        directory=Path(case['out']);result=json.loads((directory/'result.json').read_text())
        trace=json.loads((directory/'trace.json').read_text());history=json.loads((directory/'history.json').read_text())
        if (result['status']!='completed' or result['settings']!=case['settings'] or result['start_code']!=source
                or result['optimizer_steps']!=1500 or len(trace)!=1500 or len(history)!=51
                or [r['step'] for r in trace]!=list(range(1,1501))
                or [r['dataset'] for r in trace]!=schedule_for(case['settings'],task_names=(case['target'],))
                or result['selected_step']!=min(history,key=lambda r:r['val_loss'])['step']):raise ValueError('Full target budget/selection differs')
        if any(not np.isfinite(v) for row in trace for v in (*row['loss'].values(),*row['gradient_norms'].values())):
            raise ValueError('Nonfinite target training record')
        if case['method']==METHODS[2]:
            if any(r['graph_changed_elements']!=0 or r['gradient_norms']['graph']!=0 or r['gradient_norms']['adapter']!=0 for r in trace):
                raise ValueError('Frozen target backbone changed')
        elif not any(r['graph_changed_elements']>0 for r in trace):raise ValueError('Trainable target graph never updated')
        before={p.name:file_hash(p) for p in directory.iterdir() if p.is_file()}
        log=out/'logs'/(job['name']+'-readonly-'+str(attempt)+'.log')
        started_readonly=time.monotonic()
        if invoke(job['command']+['--resume'],env,log):raise RuntimeError('Completed resume failed')
        write_json(out/'operation-costs'/(job['name']+'-readonly-'+str(attempt)+'.json'),
            {'operation':'verified_completed','updates_this_call':0,'elapsed_seconds':time.monotonic()-started_readonly})
        if json.loads(log.read_text().strip().splitlines()[-1])['updates_this_call']!=0:raise ValueError('Completed resume updated model')
        if before!={p.name:file_hash(p) for p in directory.iterdir() if p.is_file()}:raise ValueError('Completed artifacts changed')
        if any(file_hash(p)!=h for p,h in case['provenance']['protected_files'].items()):raise ValueError('Protected target/source input changed')
        tail=[h for h in history if h['step']>=1400]
        receipt={'status':'completed','name':case['name'],'source':source,'updates':1500,'selected_step':result['selected_step'],
            'loss_first30':float(np.mean([r['loss']['total'] for r in trace[:30]])),
            'loss_last30':float(np.mean([r['loss']['total'] for r in trace[-30:]])),
            'final_validation_window_steps':[h['step'] for h in tail],
            'final_window_nll_change':tail[-1]['val_loss']-tail[0]['val_loss'],
            'selected_validation':result['selected_validation'],'artifacts':before,
            'positive_gradient_steps':{k:sum(r['gradient_norms'][k]>0 for r in trace) for k in trace[0]['gradient_norms']},
            'completed_resume_updates':0,'query_labels_accessed':False,'test_scoring':False}
        path=out/'reviews'/(job['name']+'.json')
        if path.exists():
            if json.loads(path.read_text())!=receipt:raise ValueError('Existing scientific review changed')
        else:write_json(path,receipt)
    def worker(gpu):
        for job in [j for j in jobs if j['gpu']==gpu]:
            if stop.is_set():return
            try:
                require_frozen_code(source)
                case=next(c for c in launch['cases'] if c['name']==job['name'])
                validate_source_case(case)
                while not (Path(job['out'])/'result.json').exists():
                    free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    if free>=12288 and util<=60:break
                    update(job,status='waiting_resource');time.sleep(15)
                    if stop.is_set():return
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                attempt=1+len(list((out/'logs').glob(job['name']+'-run-*.log')))
                command=job['command']+(['--resume'] if Path(job['out']).exists() else [])
                log=out/'logs'/(job['name']+'-run-'+str(attempt)+'.log')
                update(job,status='running',log=str(log));t=time.monotonic();rc=invoke(command,env,log)
                seconds=time.monotonic()-t
                update(job,exit_code=rc,latest_operation_seconds=seconds)
                if rc:raise RuntimeError('Native target training/finalization failed; retained, no tolerance relaxation')
                operation=json.loads(log.read_text().strip().splitlines()[-1])
                write_json(out/'operation-costs'/(job['name']+'-run-'+str(attempt)+'.json'),
                    {'operation':operation['operation'],'updates_this_call':operation['updates_this_call'],'elapsed_seconds':seconds})
                update(job,status='reviewing');review(job,case,env,attempt);update(job,status='completed')
                print('TARGET_FULL_CASE_COMPLETE',job['name'],flush=True)
            except Exception as error:
                stop.set();update(job,status='failed',error=repr(error));traceback.print_exc();return
    with (out/'.supervisor.lock').open('x') as stream:stream.write(str(os.getpid()))
    try:
        publish()
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,a.gpus))
        require_frozen_code(source)
        state='training_completed' if all(j['status']=='completed' for j in jobs) else 'failed';publish(state)
        if state=='failed':raise SystemExit(1)
        models=[]
        for c in launch['cases']:
            d=Path(c['out']);r=json.loads((d/'result.json').read_text())
            models.append({'name':c['name'],'primary_sha256':file_hash(d/'primary.pt'),
                'sidecar_sha256':file_hash(d/'primary.pt.json'),'result_sha256':file_hash(d/'result.json'),
                'selected_step':r['selected_step'],'review_sha256':file_hash(out/'reviews'/(c['name']+'.json'))})
        accepted={'status':'passed','source':source,'launch_sha256':launch_hash,
            'models':models,'optimizer_updates':40500,'query_labels_accessed':False,'test_scoring':False}
        acceptance=out/'training-acceptance.json'
        if acceptance.exists():
            if json.loads(acceptance.read_text())!=accepted:raise ValueError('Previously frozen training acceptance changed')
        else:write_json(acceptance,accepted)
        print('TARGET_FULL_ALL_TRAINING_PASSED',len(models),flush=True)
    finally:(out/'.supervisor.lock').unlink()


if __name__=='__main__':main()

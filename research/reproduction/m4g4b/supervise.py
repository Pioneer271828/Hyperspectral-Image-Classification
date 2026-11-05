"""Freeze eighteen M4 source cases and run two isolated resource-gated queues."""
import argparse,copy,json,os,shutil,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from joint_training import load_plan,schedule_for
from m4_source_study import load_study,full_settings,select_case,case_name
from m4_transfer import graph_binding
from source_data_context import load_source_context
from model.research_config import ResearchConfig
from target_initialization import state_id
from research_runtime import code_snapshot,external_directory,require_frozen_code,write_json
from loadData.dataset_registry import file_hash


def prepare(code,out,prepared,data_root,*,ordered=False):
    if ordered:
        from m4_ordered_source import load_study as study_loader, full_settings as settings_for, select_case as selector
    else:
        study_loader,settings_for,selector=load_study,full_settings,select_case
    out.mkdir(exist_ok=False)
    if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('Need20GiB artifact headroom')
    for n in ('cases','logs','reviews','operation-costs'):(out/n).mkdir()
    study=study_loader();cases=[];manifest=load_plan('configs/experiments/base97_training.json')['prepared_manifest_sha256']
    for held in study['held_targets']:
        ctx=load_source_context(prepared,data_root,held_out=held,consumer=code,manifest_sha256=manifest)
        binding=graph_binding(ctx.items)
        for seed in study['training_seeds']:
            for mode in study['modes']:
                settings=settings_for(held,seed,mode);config=ResearchConfig.from_dict(settings['config'])
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    model=config.build_model({n:ctx.items[n]['class_num'] for n in ctx.sources})
                if tuple(model.heads)!=tuple(ctx.sources) or any(held in n.split('.') for n in model.state_dict()):
                    raise ValueError('Held target created in source model')
                name=case_name(held,seed,mode);prov=copy.deepcopy(ctx.record)
                prov.update(m4_mode=mode,m4_graph_binding=binding)
                cases.append({'name':name,'held_out':held,'seed':seed,'mode':mode,'settings':settings,
                    'out':str(out/'cases'/name),'source_context':ctx.record,'provenance':prov,'graph_binding':binding,
                    'initial_state_sha256':state_id(model),'registered_parameters':sum(p.numel() for p in model.parameters())})
                del model
        if any(file_hash(p)!=h for p,h in ctx.record['protected_files'].items()):raise ValueError('Source input changed')
        del ctx
    launch={'code':code,'study':study,'out_dir':str(out),'prepared':prepared,'data_root':data_root,'cases':cases,
        'test_scoring':False,'expected_updates':18036,'supervisor_sha256':file_hash(__file__)}
    for c in cases:selector(launch,code,c['held_out'],c['seed'],c['mode'])
    require_frozen_code(code);write_json(out/'launch.json',launch)
    write_json(out/'preflight.json',{'status':'passed','code':code,'case_count':18,'optimizer_updates':0,
        'launch_sha256':file_hash(out/'launch.json'),'target_payload_accessed':False,'test_scoring':False})
    return launch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--ordered-validation-v2',action='store_true')
    p.add_argument('--run-existing',action='store_true');a=p.parse_args();code=code_snapshot()
    if a.ordered_validation_v2:
        from m4_ordered_source import load_study as study_loader, select_case as selector
    else:
        study_loader,selector=load_study,select_case
    study_loader()
    if sys.platform!='linux' or not __debug__ or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Clean expected remote code and assertions required')
    if a.prepare_only and a.run_existing:raise ValueError('Prepare/existing exclusive')
    allowed_counts=(1,2) if a.ordered_validation_v2 else (2,)
    if not a.prepare_only and (not a.gpus or len(a.gpus) not in allowed_counts
                              or len(set(a.gpus))!=len(a.gpus) or min(a.gpus)<0):
        raise ValueError('Distinct GPUs required: v1 two, v2 one or two')
    out=external_directory(a.out_dir)
    if a.run_existing:
        launch=json.loads((out/'launch.json').read_text())
        if (launch['out_dir']!=str(out) or launch['prepared']!=a.prepared_dir or launch['data_root']!=a.data_root
                or launch['supervisor_sha256']!=file_hash(__file__)):raise ValueError('Existing launch/controller differs')
        for c in launch['cases']:selector(launch,code,c['held_out'],c['seed'],c['mode'])
    else:launch=prepare(code,out,a.prepared_dir,a.data_root,ordered=a.ordered_validation_v2)
    launch_hash=file_hash(out/'launch.json')
    if a.prepare_only:print('M4_SOURCE_FULL_PREFLIGHT_PASSED',launch_hash,flush=True);return
    started=time.monotonic();stop=threading.Event();lock=threading.RLock();jobs=[]
    for index,c in enumerate(launch['cases']):
        command=[sys.executable,str(ROOT/'scripts/train_m4_source_full.py'),'--expected-commit',a.expected_commit,
            '--launch-plan',str(out/'launch.json'),'--launch-sha256',launch_hash,'--held-out',c['held_out'],
            '--seed',str(c['seed']),'--mode',c['mode']]
        if a.ordered_validation_v2:command.append('--ordered-validation-v2')
        jobs.append({'name':c['name'],'out':c['out'],'command':command,'gpu':a.gpus[index%len(a.gpus)],'status':'queued'})
    def publish(final=None):
        with lock:write_json(out/'status.json',{'status':final or ('failed' if stop.is_set() else 'running'),
            'code':code,'jobs':jobs,'elapsed_seconds':time.monotonic()-started,'launch_sha256':launch_hash,'test_scoring':False})
    def update(job,**kw):
        with lock:job.update(**kw);publish()
    def invoke(cmd,env,log):
        with log.open('x') as f:return subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    def last(path):return json.loads(path.read_text().strip().splitlines()[-1])
    def review(job,c,env,attempt):
        d=Path(c['out']);r=json.loads((d/'result.json').read_text())
        trace=json.loads((d/'trace.json').read_text());hist=json.loads((d/'history.json').read_text())
        if (r['status']!='completed' or r['settings']!=c['settings'] or r['start_code']!=code
                or r['optimizer_steps']!=len(trace) or len(trace)!=1002 or len(hist)!=51
                or [x['step'] for x in trace]!=list(range(1,1003))
                or [x['dataset'] for x in trace]!=schedule_for(c['settings'],task_names=tuple(c['settings']['source_tasks']))
                or r['selected_step']!=max(hist,key=lambda x:x['macro_oa'])['step']
                or c['held_out'] in r['per_dataset_updates']):raise ValueError('Source completion/selection/budget differs')
        if (any(not np.isfinite(v) for x in trace for v in (*x['loss'].values(),*x['gradient_norms'].values(),x['m4_parameter_gradient_norm']))
                or not any(x['m4_parameter_gradient_norm']>0 for x in trace)
                or sum(x['m4_changed_elements'] for x in trace)==0):raise ValueError('M4 source did not update finitely')
        before={p.name:file_hash(p) for p in d.iterdir() if p.is_file()}
        log=out/'logs'/(job['name']+'-readonly-'+str(attempt)+'.log');t=time.monotonic()
        if invoke(job['command']+['--resume'],env,log) or last(log)['updates_this_call']!=0:raise ValueError('Completed resume failed')
        write_json(out/'operation-costs'/(job['name']+'-readonly-'+str(attempt)+'.json'),
            {'operation':'verified_completed','updates_this_call':0,'elapsed_seconds':time.monotonic()-t})
        if before!={p.name:file_hash(p) for p in d.iterdir() if p.is_file()}:raise ValueError('Completed files changed')
        if any(file_hash(p)!=h for p,h in c['source_context']['protected_files'].items()):raise ValueError('Source input changed')
        receipt={'status':'passed','name':c['name'],'code':code,'updates':1002,'selected_step':r['selected_step'],
            'loss_first30':float(np.mean([x['loss']['total'] for x in trace[:30]])),
            'loss_last30':float(np.mean([x['loss']['total'] for x in trace[-30:]])),
            'last50epoch_validation_gain':hist[-1]['macro_oa']-hist[-6]['macro_oa'],
            'selected_validation':r['selected_validation'],'artifacts':before,'positive_m4_gradient_steps':sum(x['m4_parameter_gradient_norm']>0 for x in trace),
            'm4_changed_elements':sum(x['m4_changed_elements'] for x in trace),'completed_resume_updates':0,'test_scoring':False}
        path=out/'reviews'/(job['name']+'.json')
        if path.exists():
            if json.loads(path.read_text())!=receipt:raise ValueError('Frozen scientific review differs')
        else:write_json(path,receipt)
    def worker(gpu):
        for job in [j for j in jobs if j['gpu']==gpu]:
            if stop.is_set():return
            try:
                require_frozen_code(code)
                c=next(c for c in launch['cases'] if c['name']==job['name'])
                selector(launch,code,c['held_out'],c['seed'],c['mode'])
                while not (Path(c['out'])/'result.json').exists():
                    free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    if free>=12288 and util<=60:break
                    update(job,status='waiting_resource');time.sleep(15)
                    if stop.is_set():return
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                attempt=len(list((out/'logs').glob(job['name']+'-run-*.log')))+1
                command=job['command']+(['--resume'] if Path(c['out']).exists() else [])
                log=out/'logs'/(job['name']+'-run-'+str(attempt)+'.log');update(job,status='running');t=time.monotonic()
                rc=invoke(command,env,log);seconds=time.monotonic()-t
                if rc:raise RuntimeError('Native source training/finalization failed; no relaxation')
                op=last(log);write_json(out/'operation-costs'/(job['name']+'-run-'+str(attempt)+'.json'),
                    {'operation':op['operation'],'updates_this_call':op['updates_this_call'],'elapsed_seconds':seconds})
                update(job,status='reviewing');review(job,c,env,attempt);update(job,status='completed')
                print('M4_SOURCE_FULL_CASE_COMPLETE',job['name'],flush=True)
            except Exception as e:
                stop.set();update(job,status='failed',error=repr(e));traceback.print_exc();return
    with (out/'.supervisor.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        publish()
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
        require_frozen_code(code)
        if not all(j['status']=='completed' for j in jobs):publish('failed');raise SystemExit(1)
        accepted={'status':'passed','code':code,'launch_sha256':launch_hash,'optimizer_updates':18036,
            'cases':[{'name':c['name'],'result_sha256':file_hash(Path(c['out'])/'result.json'),
                'review_sha256':file_hash(out/'reviews'/(c['name']+'.json'))} for c in launch['cases']],
            'target_payload_accessed':False,'test_scoring':False}
        path=out/'training-acceptance.json'
        if path.exists():
            if json.loads(path.read_text())!=accepted:raise ValueError('Frozen acceptance changed')
        else:write_json(path,accepted)
        publish('training_completed');print('M4_SOURCE_FULL_ALL_PASSED',18,flush=True)
    finally:(out/'.supervisor.lock').unlink()


if __name__=='__main__':main()

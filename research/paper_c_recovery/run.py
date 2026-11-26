"""Register/run an isolated, explicitly budgeted released-development comparison."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace

import torch

from transactions import execute,inspect,new_json,paired_plan,sha,digest
from tasks import NATIVE,AUTHOR,PILOT,OPTIMIZER,MODEL,LOSSES,NativeTask,DSNetTask,prior_module,materialize


def clean(root,expected):
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git('rev-parse','HEAD')!=expected or git('status','--porcelain'):raise ValueError('Pinned clean code identity differs')


def verify_files(protected):
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Protected input or code file changed')


def prepare(a,root):
    if sha(a.pilot)!=PILOT:raise ValueError('Prior supervision registration differs')
    pilot=json.loads(Path(a.pilot).read_text());bindings=json.loads(Path(a.bindings).read_text())
    for p,h in bindings['protected'].items():
        if sha(p)!=h:raise ValueError('Measured binding/profile changed')
    if a.sweeps not in (3,40):raise ValueError('Explicit engineering3 or proposed full40 budget required')
    updates=2*3*sum(len(r['train_roster'])*a.sweeps for r in pilot['scenes'].values())
    if a.real_update_budget!=updates:raise ValueError('Explicit maximum real update budget differs')
    planned_bytes=bindings['checkpoint_bytes_estimate'][str(a.sweeps)]
    if planned_bytes>a.max_checkpoint_bytes:raise ValueError('Declared storage budget insufficient')
    if planned_bytes>shutil.disk_usage(Path(a.out).parent).free*.25:raise ValueError('Full checkpoints exceed a quarter of current free space; no training admitted')
    protected={**pilot['protected_inputs'],**bindings['protected'],str(Path(a.pilot).resolve()):PILOT,
               str(Path(a.bindings).resolve()):sha(a.bindings)}
    # Code/license bytes are part of the registration, including sibling adapters.
    for folder in (root/'research/paper_c_recovery',root/'research/paper1_c'):
        for p in folder.iterdir():
            if p.is_file():protected[str(p.resolve())]=sha(p)
    for name in ('model.py','LICENSE'):
        p=Path(a.author)/name;protected[str(p.resolve())]=sha(p)
    cases=[]
    for scene in sorted(pilot['scenes']):
        for seed in (64,512,2048):
            for method in ('native','dsnet'):
                roster=pilot['scenes'][scene]['train_roster'];n=len(roster)
                events=list(range(0,n*a.sweeps+1,n if a.sweeps==3 else n*5))
                c={'roster':roster,'sweeps':a.sweeps,'seed':seed,'plan':paired_plan(roster,a.sweeps,seed),
                   'updates':n*a.sweeps,'selection_steps':events,'device':'cpu','query_reads':0,'confirmation':False,
                   'optimizer':OPTIMIZER,'protected':protected,'scene':scene,'method':method,
                   'producer':a.expected_commit,'dependency':NATIVE,'author':AUTHOR,'native_model':MODEL,'native_losses':LOSSES,
                   'scope':'engineering recovery/exposure validation; not convergence or confirmation' if a.sweeps==3 else 'matched development recipes; not independent confirmation'}
                cases.append({'id':f'{scene}-s{seed}-{method}','contract':c})
    verify_files(protected)
    value={'schema':'paper_c_registered_equal_exposure_execution_v1','producer':a.expected_commit,
           'container_hostname':socket.gethostname(),'device':'cpu','threads':1,'cases':cases,
           'pilot':str(Path(a.pilot).resolve()),'dependency_root':str(Path(a.dependency).resolve()),'author_root':str(Path(a.author).resolve()),
           'runs_root':str(Path(a.runs).resolve()),'real_optimizer_update_budget':updates,
           'maximum_label_exposures':2*3*sum(r['geometry']['coverage']['devtrain']['released_per_class'][0]*
               len(r['geometry']['coverage']['devtrain']['released_per_class'])*a.sweeps for r in pilot['scenes'].values()),
           'checkpoint_bytes_estimate':planned_bytes,'checkpoint_bytes_limit':a.max_checkpoint_bytes,
           'shape_based_compute_seconds_estimate':bindings['shape_based_compute_seconds_estimate'][str(a.sweeps)],
           'planning_elapsed_minutes':[10,60] if a.sweeps==3 else [30,180],
           'elapsed_limit_note':'Planning range with loading/hash/checkpoint/selection margin; backup is separate. Not measured real throughput or convergence evidence.',
           'authorization_basis':'User continued the four existing project execution chats after backup; main delegated registered development execution, no query/new labels.',
           'selection':'first minimum classification NLL; native uses fused probability NLL, DSNet logits CE; no query',
           'confounds':['same updates and exact per-sweep label exposure, different context/capacity/FLOPs and method-specific losses',
                        'native uses unchanged three-head plus graph regularizers; DSNet original unclamped CE+SAD',
                        'explicit common Adam1e-4 differs from legacy native two-rate and author training recipes'],
           'new_optimizer_updates_at_registration':0,'query_reads':0,'confirmation':False}
    new_json(a.out,value);print(json.dumps({'registration_sha256':sha(a.out),'real_update_budget':updates,'checkpoint_bytes_estimate':planned_bytes,'planning_elapsed_minutes':value['planning_elapsed_minutes']}))


def run(a,root):
    if not a.registration_sha256 or sha(a.registration)!=a.registration_sha256:raise ValueError('Explicit frozen registration SHA required')
    reg=json.loads(Path(a.registration).read_text())
    if (reg['producer']!=a.expected_commit or reg['container_hostname']!=socket.gethostname()
            or reg['dependency_root']!=str(Path(a.dependency).resolve()) or reg['author_root']!=str(Path(a.author).resolve())
            or reg['runs_root']!=str(Path(a.runs).resolve()) or reg['device']!='cpu'):
        raise ValueError('Producer, device, container or dependency mapping differs')
    if sum(r['contract']['updates'] for r in reg['cases'])!=reg['real_optimizer_update_budget']:
        raise ValueError('Aggregate update budget differs')
    out=Path(a.out)
    if out.exists():
        if not a.resume or sha(out/'registration.json')!=a.registration_sha256:raise ValueError('Explicit unchanged resume required')
    else:
        if a.resume:raise ValueError('Missing run cannot resume')
        out.mkdir(mode=0o700)
        with (out/'registration.json').open('xb') as f:f.write(Path(a.registration).read_bytes())
    sys.path.insert(0,str(Path(a.dependency).resolve()))
    cache={};results=[];new_updates=0
    def factory(c):
        scene=c['scene']
        if scene not in cache:
            previous=prior_module(root,'pilot');adapter=prior_module(root,'adapter')
            bank,receipt=previous.load_bank(SimpleNamespace(runs_root=a.runs),scene)
            old=json.loads(Path(reg['pilot']).read_text())['scenes'][scene]
            if receipt!=old['receipt'] or bank.pin!=old['pin'] or previous.geometry(bank)!=old['geometry']:
                raise ValueError('Bank role, normalizer, receipt or geometry differs')
            data,identities=materialize(bank,adapter)
            cache[scene]=(data,adapter,adapter.load_author_model(a.author),bank.release.classes,bank.release.shape[-1],identities)
        data,adapter,author,classes,bands,_=cache[scene]
        return NativeTask(data,scene,classes) if c['method']=='native' else DSNetTask(data,scene,classes,bands,adapter,author)
    for case in reg['cases']:
        c=case['contract'];folder=out/case['id']
        count=lambda: max(0,len([p for p in (folder/'stages').glob('*') if p.is_dir() and p.name.isdigit()])-1)
        before=count()
        try:
            result=execute(folder,c,lambda:factory(c),resume=folder.exists() and a.resume,stop_after=a.stop_after_case_updates)
            new_updates+=result['new_updates'];results.append({'case':case['id'],'status':'PASSED' if result.get('complete') or result.get('completed_reentry') else 'STOPPED','result':result})
            print(json.dumps(results[-1]),flush=True)
            if result.get('stopped_after_commit') is not None:break
        except Exception as error:
            committed_delta=count()-before;new_updates+=committed_delta
            results.append({'case':case['id'],'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
                            'new_committed_updates_before_failure':committed_delta,'automatic_retry':False,
                            'uncertain_update_intents':[p.name for p in (folder/'stages').glob('.pending-*')]})
            print(json.dumps({'case':case['id'],'status':'FAILED','error':str(error)}),flush=True)
    # Capture receipts per invocation, never overwrite earlier failures.
    complete=len(results)==len(reg['cases']) and all(r['status']=='PASSED' for r in results)
    receipt={'registration_sha256':a.registration_sha256,'producer':a.expected_commit,'results':results,
             'complete':complete,'new_committed_optimizer_updates':new_updates,
             'query_reads':0,'new_labels_released':0,'confirmation':False,
             'materialized_scenes':list(cache),'materialized_context_digest':{s:digest(v[-1]) for s,v in cache.items()}}
    # Entirely completed reentry leaves the existing run untouched.
    if (out/'summary.json').exists():
        if not complete or new_updates!=0 or cache:raise ValueError('Completed reentry unexpectedly executed work')
        print(json.dumps({'completed_reentry':True,'new_updates':0,'new_neural_calls':0}));return
    index=len(list(out.glob('attempt-*.json')))+1;new_json(out/f'attempt-{index:04d}.json',receipt)
    if complete:new_json(out/'summary.json',receipt)
    if any(r['status']=='FAILED' for r in results):raise RuntimeError('Failed cases retained; no automatic retry')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run'))
    for name in ('expected-commit','dependency','author','runs','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--pilot');p.add_argument('--bindings');p.add_argument('--sweeps',type=int)
    p.add_argument('--real-update-budget',type=int);p.add_argument('--max-checkpoint-bytes',type=int)
    p.add_argument('--registration');p.add_argument('--registration-sha256');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-case-updates',type=int);a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Authorized Linux CPU only')
    clean(root,a.expected_commit);clean(a.dependency,NATIVE)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(a.out).resolve();runs=Path(a.runs).resolve()
    if out==runs or not out.is_relative_to(runs):raise ValueError('External authorized run path required')
    if a.mode=='prepare':
        if any(getattr(a,k) is None for k in ('pilot','bindings','sweeps','real_update_budget','max_checkpoint_bytes')):raise ValueError('Explicit proposal, cost and budget fields required')
        prepare(a,root)
    else:
        if not a.registration:raise ValueError('Explicit registration required')
        run(a,root)
    clean(root,a.expected_commit);clean(a.dependency,NATIVE)


if __name__=='__main__':main()

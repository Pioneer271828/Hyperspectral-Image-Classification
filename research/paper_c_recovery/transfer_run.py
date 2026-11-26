"""New explicitly registered native3→target3 engineering transfer, not old G3."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback
from types import SimpleNamespace

import torch

from transactions import execute,new_json,sha,restore_rng
from tasks import NativeTask,prior_module,materialize,NATIVE
from run import clean,verify_files
from inventory_models import REMAINING


class TransferTask(NativeTask):
    def __init__(self,data,target,classes,source_checkpoint,target_initial_checkpoint):
        super().__init__(data,target,classes)
        src=torch.load(source_checkpoint,weights_only=True,map_location='cpu')
        base=torch.load(target_initial_checkpoint,weights_only=True,map_location='cpu')
        if base['step']!=0 or base['optimizer']['state']:raise ValueError('Matched target initialization is not fresh')
        state=self.model.state_dict();back={k:v for k,v in src['model'].items() if k.startswith('backbone.')}
        heads={k:v for k,v in base['model'].items() if not k.startswith('backbone.')}
        if set(state)!=set(back)|set(heads):raise ValueError('Source backbone/target head key sets differ')
        for k,v in {**back,**heads}.items():
            if v.shape!=state[k].shape or v.dtype!=state[k].dtype:raise ValueError('Shared backbone or target head tensor contract differs')
        self.model.load_state_dict({**back,**heads},strict=True)
        # No source head, source optimizer or source RNG is inherited. Pair the
        # target initialization/random stream with its already frozen scratch arm.
        restore_rng(base['rng'])


def prepare(a,root):
    oldrun=Path(a.baseline_run);oldreg=json.loads((oldrun/'registration.json').read_text())
    audited=json.loads(Path(a.baseline_audit).read_text());by={(r['scene'],r['seed']):r for r in audited['cases'] if r['method']=='native'}
    clean(a.baseline_producer,oldreg['producer']);clean(oldreg['dependency_root'],NATIVE)
    if audited['producer']!=oldreg['producer']:raise ValueError('Baseline audit producer differs')
    if len(by)!=9 or audited['committed_optimizer_updates']!=2286:raise ValueError('Complete original engineering pair required')
    if sha(root/'research/paper_c_recovery/tasks.py')!=sha(Path(a.baseline_producer)/'research/paper_c_recovery/tasks.py'):
        raise ValueError('Native model/data/loss adapter differs from matched control')
    oldcases={r['id']:r['contract'] for r in oldreg['cases']};cases=[];storage=0
    for source,target in REMAINING:
        for seed in (64,512,2048):
            src=by[source,seed];base=by[target,seed];srcfolder=oldrun/src['case'];basefolder=oldrun/base['case']
            srcstate=srcfolder/'stages'/f'{src["selected_step"]:06d}'/'state.pt'
            base0=basefolder/'stages'/'000000'/'state.pt'
            srcrow=srcstate.parent/'record.json'
            if sha(srcstate)!=json.loads(srcrow.read_text())['state_sha256'] or src['selected_step']<=0:
                raise ValueError('Missing trained single-source selected checkpoint')
            c=copy.deepcopy(oldcases[base['case']])
            if c['sweeps']!=3 or c['method']!='native':raise ValueError('Matched target control recipe differs')
            protected={**c['protected'],str(srcstate):sha(srcstate),str(base0):sha(base0),
                str(srcfolder/'contract.json'):sha(srcfolder/'contract.json'),str(srcrow):sha(srcrow),
                str(oldrun/'registration.json'):sha(oldrun/'registration.json'),str(Path(a.baseline_audit)):sha(a.baseline_audit)}
            for f in (root/'research/paper_c_recovery').iterdir():
                if f.is_file():protected[str(f.resolve())]=sha(f)
            c.update(producer=a.expected_commit,method='native_single_source_transfer_engineering3',protected=protected,
                scope='new source-native3/target-native3 engineering development; not old G3 or confirmation',
                transfer={'source':source,'target':target,'source_case':src['case'],'source_inherited_executed_updates':src['committed_optimizer_updates'],
                    'source_selected_step':src['selected_step'],'source_selected_state':str(srcstate),'source_selected_sha256':sha(srcstate),
                    'target_scratch_case':base['case'],'target_scratch_contract_sha256':base['contract_sha256'],
                    'target_initial_state':str(base0),'target_initial_sha256':sha(base0),
                    'initialization':'only exact source backbone; exact previously frozen fresh target heads and target RNG; new empty Adam'})
            cases.append({'id':f'{source}-to-{target}-s{seed}','contract':c})
            # Native profile max size plus a planning margin for state metadata.
            storage+=(c['updates']+1)*900000
    updates=sum(c['contract']['updates'] for c in cases)
    if updates!=2007 or a.real_update_budget!=updates:raise ValueError('Explicit fifteen-case remaining-direction budget must be2007')
    if storage>a.max_checkpoint_bytes or storage>shutil.disk_usage(Path(a.out).parent).free*.25:raise ValueError('Insufficient declared/current storage for complete states')
    value={'schema':'paper_c_remaining_five_native3_transfer_v1','producer':a.expected_commit,
        'container_hostname':socket.gethostname(),'device':'cpu','threads':1,'cases':cases,
        'dependency_root':oldreg['dependency_root'],'author_root':oldreg['author_root'],'runs_root':oldreg['runs_root'],'pilot':oldreg['pilot'],
        'baseline_producer':oldreg['producer'],'baseline_run':str(oldrun),'baseline_audit_sha256':sha(a.baseline_audit),
        'real_optimizer_update_budget':updates,'checkpoint_bytes_estimate_with_margin':storage,
        'planning_elapsed_minutes':[10,45],'new_source_updates':0,'query_reads':0,'confirmation':False,
        'authorization_basis':'User continued existing project execution; delegated remaining-direction development with preregistration. No new labels/query.',
        'selection':'same target first-minimum fused classification NLL at0/1/2/3 sweeps as frozen scratch control',
        'limits':['source and target each use original released K5/V5 banks; all historically exposed development roles',
                  'source models are this engineering3 package, not original G3 source10000/target1000 models',
                  'source selected weights carry extra source training cost; matching target budget is not equal total compute',
                  'no independent calibration/confirmation, no claim of convergence or paper efficacy',
                  'do not promote a newly created model to a verified lazy light/full pair until independent lineage, endpoint and graph-bypass audit']}
    verify_files(cases[0]['contract']['protected'])
    # Source-specific protected files must also be frozen before publication.
    for case in cases:verify_files(case['contract']['protected'])
    new_json(a.out,value);print(json.dumps({'registration_sha256':sha(a.out),'cases':len(cases),'new_target_update_budget':updates,'checkpoint_bytes_estimate':storage,'planning_elapsed_minutes':value['planning_elapsed_minutes']}))


def run(a,root):
    if sha(a.registration)!=a.registration_sha256:raise ValueError('Frozen transfer registration SHA differs')
    reg=json.loads(Path(a.registration).read_text())
    if reg['producer']!=a.expected_commit or reg['container_hostname']!=socket.gethostname() or reg['real_optimizer_update_budget']!=2007:
        raise ValueError('Explicit producer/container/budget differs')
    clean(reg['dependency_root'],NATIVE);sys.path.insert(0,reg['dependency_root'])
    out=Path(a.out)
    if out.exists():
        if not a.resume or sha(out/'registration.json')!=a.registration_sha256:raise ValueError('Explicit unchanged resume required')
    else:
        if a.resume:raise ValueError('Missing transfer run cannot resume')
        out.mkdir(mode=0o700)
        with (out/'registration.json').open('xb') as f:f.write(Path(a.registration).read_bytes())
    cache={};rows=[];new_updates=0
    def factory(c):
        target=c['scene']
        if target not in cache:
            old=prior_module(root,'pilot');adapter=prior_module(root,'adapter');bank,receipt=old.load_bank(SimpleNamespace(runs_root=reg['runs_root']),target)
            pilot=json.loads(Path(reg['pilot']).read_text())['scenes'][target]
            if receipt!=pilot['receipt'] or bank.pin!=pilot['pin'] or old.geometry(bank)!=pilot['geometry']:raise ValueError('Original released role binding differs')
            data,identities=materialize(bank,adapter);cache[target]=(data,bank.release.classes)
        data,classes=cache[target];t=c['transfer']
        return TransferTask(data,target,classes,t['source_selected_state'],t['target_initial_state'])
    for case in reg['cases']:
        folder=out/case['id'];count=lambda:max(0,len([f for f in (folder/'stages').glob('*') if f.is_dir() and f.name.isdigit()])-1);before=count()
        try:
            r=execute(folder,case['contract'],lambda:factory(case['contract']),resume=folder.exists() and a.resume)
            new_updates+=r['new_updates'];rows.append({'case':case['id'],'status':'PASSED','result':r})
        except Exception as error:
            delta=count()-before;new_updates+=delta;rows.append({'case':case['id'],'status':'FAILED','new_committed_before_failure':delta,
                'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False})
        print(json.dumps(rows[-1]),flush=True)
    complete=all(r['status']=='PASSED' for r in rows)
    value={'registration_sha256':a.registration_sha256,'producer':a.expected_commit,'complete':complete,'results':rows,
           'new_committed_updates':new_updates,'query_reads':0,'confirmation':False,'new_source_updates':0}
    if (out/'summary.json').exists():
        if not complete or new_updates or cache:raise ValueError('Completed transfer reentry unexpectedly executed work')
        print(json.dumps({'completed_reentry':True,'new_updates':0,'new_neural_calls':0}));return
    new_json(out/f'attempt-{len(list(out.glob("attempt-*.json")))+1:04d}.json',value)
    if complete:new_json(out/'summary.json',value)
    if not complete:raise RuntimeError('Transfer failures retained; no automatic retry or seed substitution')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run'))
    for n in ('expected-commit','out'):p.add_argument('--'+n,required=True)
    for n in ('baseline-run','baseline-audit','baseline-producer','registration','registration-sha256'):p.add_argument('--'+n)
    p.add_argument('--real-update-budget',type=int);p.add_argument('--max-checkpoint-bytes',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Authorized Linux CPU only')
    root=Path(__file__).resolve().parents[2];clean(root,a.expected_commit);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Explicit project run-root confinement also applies to registration output.
    scope=json.loads((Path(a.baseline_run)/'registration.json').read_text()) if a.mode=='prepare' else json.loads(Path(a.registration).read_text())
    output=Path(a.out).resolve();runs=Path(scope['runs_root']).resolve()
    if output==runs or not output.is_relative_to(runs):raise ValueError('External authorized run subdirectory required')
    prepare(a,root) if a.mode=='prepare' else run(a,root)
    clean(root,a.expected_commit)


if __name__=='__main__':main()

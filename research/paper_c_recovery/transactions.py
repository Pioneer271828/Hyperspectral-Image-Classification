"""Isolated CPU optimizer transactions; no project trainer or checkpoint changes."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def new_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch_cpu':torch.get_rng_state()}


def restore_rng(value):
    random.setstate(value['python']);n=value['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(value['torch_cpu'])


def model_hash(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():
        a=x.detach().cpu().contiguous().numpy()
        h.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def paired_plan(roster,sweeps,seed):
    if (type(sweeps) is not int or sweeps<1 or not roster or len(set(roster))!=len(roster)
            or any(type(x) is not int or x<0 for x in roster)):
        raise ValueError('Nonempty unique integer roster and positive sweep count required')
    rng=np.random.default_rng(seed)
    return [int(x) for _ in range(sweeps) for x in rng.permutation(roster)]


def validate_contract(c):
    plan=paired_plan(c['roster'],c['sweeps'],c['seed'])
    if c['plan']!=plan or c['updates']!=len(plan):raise ValueError('Equal-exposure plan differs')
    events=c['selection_steps']
    if (not events or events[0]!=0 or events[-1]!=len(plan)
            or events!=sorted(set(events)) or any(type(x) is not int or not 0<=x<=len(plan) for x in events)):
        raise ValueError('Complete ordered selection events required')
    if c.get('device')!='cpu' or c.get('query_reads')!=0 or c.get('confirmation') is not False:
        raise ValueError('CPU development-only contract required')
    if c['optimizer']!={'name':'Adam','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False}:
        raise ValueError('Registered Adam recipe differs')
    for path,h in c['protected'].items():
        if sha(path)!=h:raise ValueError('Original protected input changed')


def inspect(root,c):
    """Validate bytes without torch.load, task construction or any neural calls."""
    root=Path(root);validate_contract(c)
    if json.loads((root/'contract.json').read_text())!=c:raise ValueError('Exact contract differs')
    if set(p.name for p in root.iterdir())-{'contract.json','lock','stages','done.json'}:
        raise ValueError('Unexpected run artifact')
    stages=root/'stages';names=sorted(p.name for p in stages.iterdir())
    if any(not n.isdigit() or len(n)!=6 for n in names):raise ValueError('Pending or uncommitted artifact retained; no replay')
    if names!=[f'{i:06d}' for i in range(len(names))]:raise ValueError('Committed stage sequence has a gap')
    rows=[];previous=None
    for step,name in enumerate(names):
        folder=stages/name
        if set(p.name for p in folder.iterdir())!={'state.pt','record.json','intent.json'}:raise ValueError('Stage artifact set differs')
        row=json.loads((folder/'record.json').read_text())
        if (row['step']!=step or row['contract']!=digest(c) or row['previous']!=previous
                or row['state_sha256']!=sha(folder/'state.pt') or row['intent_sha256']!=sha(folder/'intent.json')
                or row['optimizer_updates']!=int(step>0)
                or row['block']!=(None if step==0 else c['plan'][step-1])):
            raise ValueError('Committed state, intent, plan or hash chain changed')
        previous=sha(folder/'record.json');rows.append(row)
    if len(rows)>c['updates']+1:raise ValueError('Update budget exceeded')
    done=root/'done.json'
    if done.exists():
        expected={'contract':digest(c),'updates':c['updates'],'last_record':previous,
                  'stage_count':len(rows),'selected_step':rows[-1]['selected_step']}
        if len(rows)!=c['updates']+1 or json.loads(done.read_text())!=expected:
            raise ValueError('Completion receipt differs')
    return rows,done.exists()


@contextmanager
def locked(root):
    with (root/'lock').open('a+b') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another writer owns this run') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def _snapshot(task,step,best,events):
    return {'model':task.model.state_dict(),'optimizer':task.optimizer.state_dict(),
            'rng':rng_state(),'step':step,'best':best,'selection_events':events,
            'module_modes':{n:m.training for n,m in task.model.named_modules()}}


def _load(task,state):
    task.model.load_state_dict(state['model'],strict=True)
    task.optimizer.load_state_dict(state['optimizer'])
    modes=state['module_modes']
    if set(modes)!=set(dict(task.model.named_modules())):raise ValueError('Module roster differs')
    for name,module in task.model.named_modules():module.training=modes[name]
    restore_rng(state['rng'])


def execute(root,c,factory,*,resume=False,stop_after=None,fault=lambda phase,step:None):
    """factory creates a task with model, optimizer, loss(block), evaluate()."""
    root=Path(root);validate_contract(c)
    if stop_after is not None and (type(stop_after) is not int or stop_after<1):raise ValueError('Positive explicit stop boundary required')
    exists=root.exists()
    if exists and not resume:raise ValueError('Explicit resume required; existing output preserved')
    if not exists:
        if resume:raise ValueError('Cannot resume missing output')
        root.mkdir(mode=0o700);new_json(root/'contract.json',c);(root/'stages').mkdir();sync_dir(root)
    with locked(root):
        rows,done=inspect(root,c)
        if done:return {'completed_reentry':True,'new_updates':0,'new_neural_calls':0,'selected_step':rows[-1]['selected_step']}
        # Construct only after validating every previously committed byte and pin.
        seed_all(c['seed']);task=factory()
        if not isinstance(task.optimizer,torch.optim.Adam):raise ValueError('Task optimizer must be Adam')
        config=task.optimizer.param_groups
        if len(config)!=1 or any(config[0][k]!=v for k,v in {'lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':0.,'amsgrad':False}.items()):
            raise ValueError('Task optimizer parameters differ')
        best=None;events=[]
        if rows:
            state=torch.load(root/'stages'/f'{len(rows)-1:06d}'/'state.pt',weights_only=True,map_location='cpu')
            if state['step']!=len(rows)-1:raise ValueError('State cursor differs')
            _load(task,state);best=state['best'];events=state['selection_events']
        made=0
        for step in range(len(rows),c['updates']+1):
            pending=root/'stages'/f'.pending-{step:06d}';pending.mkdir()
            intent={'contract':digest(c),'step':step,'block':None if step==0 else c['plan'][step-1],
                    'optimizer_call_upper_bound':int(step>0),'note':'An interrupted update is never automatically replayed'}
            new_json(pending/'intent.json',intent);sync_dir(pending);sync_dir(pending.parent)
            timings={};loss_value=None
            try:
                fault('after_intent',step)
                if step:
                    task.model.train();task.optimizer.zero_grad(set_to_none=True)
                    t=perf_counter();loss=task.loss(c['plan'][step-1]);timings['forward_seconds']=perf_counter()-t
                    if loss.ndim!=0 or not torch.isfinite(loss):raise FloatingPointError('Nonfinite or nonscalar original loss')
                    loss_value=float(loss.detach());t=perf_counter();loss.backward();timings['backward_seconds']=perf_counter()-t
                    grads=[p.grad for p in task.model.parameters() if p.requires_grad]
                    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):raise FloatingPointError('Missing or nonfinite gradient')
                    fault('after_backward',step)
                    t=perf_counter();task.optimizer.step();timings['optimizer_seconds']=perf_counter()-t
                    fault('after_optimizer',step)
                    if any(not torch.isfinite(x).all() for x in task.model.state_dict().values()):raise FloatingPointError('Nonfinite parameter or buffer')
                metrics=None
                if step in c['selection_steps']:
                    t=perf_counter();task.model.eval()
                    before=model_hash(task.model)
                    with torch.no_grad():metrics=task.evaluate()
                    if model_hash(task.model)!=before:raise ValueError('Selection mutated model parameters or buffers')
                    timings['selection_seconds']=perf_counter()-t
                    if not math.isfinite(metrics['nll']):raise FloatingPointError('Nonfinite selection NLL')
                    # Ensure serializable finite aggregate evidence before checkpointing.
                    digest(metrics);events=events+[{'step':step,'metrics':metrics}]
                    if best is None or metrics['nll']<best['nll']:best={'step':step,'nll':metrics['nll']}
                t=perf_counter()
                with (pending/'state.pt').open('xb') as f:
                    torch.save(_snapshot(task,step,best,events),f);f.flush();os.fsync(f.fileno())
                timings['checkpoint_seconds']=perf_counter()-t
                row={'step':step,'contract':digest(c),'previous':None if not rows else sha(root/'stages'/f'{step-1:06d}'/'record.json'),
                     'state_sha256':sha(pending/'state.pt'),'intent_sha256':sha(pending/'intent.json'),
                     'block':intent['block'],'optimizer_updates':int(step>0),'loss':loss_value,
                     'selection':metrics,'selected_step':best['step'],'timings':timings,
                     'checkpoint_bytes':(pending/'state.pt').stat().st_size}
                new_json(pending/'record.json',row);sync_dir(pending)
                target=root/'stages'/f'{step:06d}'
                if target.exists():raise ValueError('Committed destination already exists')
                os.rename(pending,target);sync_dir(target.parent);rows.append(row)
            except BaseException as error:
                if pending.exists() and not (pending/'failure.json').exists():
                    new_json(pending/'failure.json',{'step':step,'error':repr(error),'committed_updates_before':max(0,step-1),
                        'uncertain_optimizer_call_upper_bound':int(step>0),'automatic_retry':False})
                    sync_dir(pending)
                raise
            made+=int(step>0)
            fault('after_commit',step)
            if stop_after is not None and made>=stop_after and step<c['updates']:
                return {'stopped_after_commit':step,'new_updates':made,'selected_step':best['step']}
        validate_contract(c)
        new_json(root/'done.json',{'contract':digest(c),'updates':c['updates'],
            'last_record':sha(root/'stages'/f'{c["updates"]:06d}'/'record.json'),
            'stage_count':len(rows),'selected_step':best['step']});sync_dir(root)
        return {'complete':True,'new_updates':made,'selected_step':best['step']}

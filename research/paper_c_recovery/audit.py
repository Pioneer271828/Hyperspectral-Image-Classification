"""Independent saved-record/exposure/Adam-cursor audit; no producer imports."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit(root):
    root=Path(root);c=json.loads((root/'contract.json').read_text())
    rng=np.random.default_rng(c['seed'])
    expected=[int(i) for _ in range(c['sweeps']) for i in rng.permutation(c['roster'])]
    if expected!=c['plan'] or len(expected)!=c['updates']:raise ValueError('Independent permutation reconstruction differs')
    for p,h in c['protected'].items():
        if sha(p)!=h:raise ValueError('Original protected input changed')
    names=sorted(p.name for p in (root/'stages').iterdir())
    if names!=[f'{i:06d}' for i in range(c['updates']+1)]:raise ValueError('Incomplete or uncertain update set')
    previous=None;selections=[];best=None;updates=[];checkpoint_bytes=0;timings={}
    for step,name in enumerate(names):
        folder=root/'stages'/name;r=json.loads((folder/'record.json').read_text())
        if (r['step']!=step or r['previous']!=previous or r['contract']!=digest(c)
                or r['state_sha256']!=sha(folder/'state.pt') or r['intent_sha256']!=sha(folder/'intent.json')
                or r['optimizer_updates']!=int(step>0) or r['block']!=(None if not step else expected[step-1])):
            raise ValueError('Independent hash/counter/plan check failed')
        if r['checkpoint_bytes']!=(folder/'state.pt').stat().st_size:raise ValueError('Saved-state size differs')
        checkpoint_bytes+=r['checkpoint_bytes'];previous=sha(folder/'record.json')
        for key,value in r['timings'].items():
            if not math.isfinite(value) or value<0:raise ValueError('Invalid measured time')
            timings.setdefault(key,[]).append(value)
        if step:
            if not math.isfinite(r['loss']):raise ValueError('Nonfinite recorded training loss')
            updates.append(r['block'])
        if (r['selection'] is not None)!=(step in c['selection_steps']):raise ValueError('Selection schedule differs')
        if r['selection'] is not None:
            m=r['selection'];selections.append({'step':step,'metrics':m})
            if not math.isfinite(m['nll']):raise ValueError('Nonfinite selection value')
            if best is None or m['nll']<best['nll']:best={'step':step,'nll':m['nll']}
            if 'confusion' in m:
                cm=np.asarray(m['confusion']);support=cm.sum(1)
                if (cm.ndim!=2 or cm.shape[0]!=cm.shape[1] or np.any(cm<0) or np.any(support==0)
                        or not np.array_equal(cm,np.floor(cm)) or not np.array_equal(support,m['support'])
                        or not np.allclose(np.diag(cm)/support,m['recall'],atol=1e-15,rtol=0)
                        or not math.isclose(np.trace(cm)/cm.sum(),m['oa'],abs_tol=1e-15)
                        or not math.isclose(np.mean(np.diag(cm)/support),m['aa'],abs_tol=1e-15)):
                    raise ValueError('Independent confusion/recall check failed')
        if r['selected_step']!=best['step']:raise ValueError('First-minimum identity differs')
    done=json.loads((root/'done.json').read_text())
    if done!={'contract':digest(c),'updates':len(expected),'last_record':previous,'stage_count':len(names),'selected_step':best['step']}:
        raise ValueError('Completion receipt differs')
    last=torch.load(root/'stages'/names[-1]/'state.pt',weights_only=True,map_location='cpu')
    if last['step']!=len(expected) or last['best']!=best or last['selection_events']!=selections:raise ValueError('Saved cursor/selection differs')
    adam=last['optimizer']['state']
    if not adam or any(s['step'].numel()!=1 or float(s['step'])!=len(expected) for s in adam.values()):raise ValueError('Adam step counters differ')
    if any(not torch.isfinite(t).all() for t in last['model'].values()):raise ValueError('Nonfinite saved model')
    if set(last['rng'])!={'python','numpy','torch_cpu'} or last['rng']['torch_cpu'].dtype!=torch.uint8:raise ValueError('Saved RNG envelope differs')
    if Counter(updates)!={block:c['sweeps'] for block in c['roster']}:raise ValueError('Per-block exposure differs')
    return {'status':'PASSED','scene':c.get('scene'),'method':c.get('method'),'seed':c['seed'],
        'contract_sha256':sha(root/'contract.json'),'plan_sha256':digest(expected),
        'committed_optimizer_updates':len(updates),'sweeps':c['sweeps'],'blocks_per_sweep':len(c['roster']),
        'selected_step':best['step'],'selected_metrics':selections[c['selection_steps'].index(best['step'])]['metrics'],
        'checkpoint_bytes':checkpoint_bytes,'full_optimizer_RNG_buffer_cursor_saved':True,
        'measured_components':{k:{'count':len(v),'sum_seconds':sum(v),'median_seconds':float(np.median(v)),
                                  'p90_seconds':float(np.quantile(v,.9))} for k,v in timings.items()},
        'new_optimizer_updates_during_audit':0,'new_neural_forwards_during_audit':0,'query_reads':0,'confirmation':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(a.run)
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

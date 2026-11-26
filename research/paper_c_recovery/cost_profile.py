"""Synthetic exact-model forward/backward/selection/durable-checkpoint profile."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from transactions import execute,new_json,paired_plan,seed_all
from tasks import AUTHOR,NATIVE,MODEL,LOSSES,OPTIMIZER,NativeTask,DSNetTask,prior_module,synthetic_data


def clean(root,sha):
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git('rev-parse','HEAD')!=sha or git('status','--porcelain'):raise ValueError('Clean pinned code required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True)
    p.add_argument('--dependency',required=True);p.add_argument('--author',required=True)
    p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];clean(root,a.expected_commit);clean(a.dependency,NATIVE)
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Authorized Linux CPU only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(a.out).resolve();runs=Path(a.runs).resolve()
    if out==runs or not out.is_relative_to(runs):raise ValueError('External run subdirectory required')
    out.mkdir(mode=0o700)
    sys.path.insert(0,str(Path(a.dependency).resolve()))
    adapter=prior_module(root,'adapter');author=adapter.load_author_model(a.author)
    rows=[];start=perf_counter()
    for scene,bands,classes,nblocks in (('paviaU',103,9,30),('xuzhou',436,9,31),('whu-honghu',270,22,66)):
        data=synthetic_data(adapter,bands,classes)
        for method in ('native','dsnet'):
            c={'roster':[0],'sweeps':3,'seed':64,'plan':paired_plan([0],3,64),'updates':3,
               'selection_steps':[0,1,2,3],'device':'cpu','query_reads':0,'confirmation':False,
               'optimizer':OPTIMIZER,'protected':{},'scene':scene,'method':method,'scope':'synthetic-cost-and-integration-only'}
            factory=(lambda:NativeTask(data,scene,classes)) if method=='native' else (lambda:DSNetTask(data,scene,classes,bands,adapter,author))
            folder=out/f'{scene}-{method}'
            t=perf_counter();result=execute(folder,c,factory);elapsed=perf_counter()-t
            records=[json.loads(f.read_text()) for f in sorted((folder/'stages').glob('*/record.json'))]
            times={name:[r['timings'][name] for r in records if name in r['timings']]
                   for name in ('forward_seconds','backward_seconds','optimizer_seconds','selection_seconds','checkpoint_seconds')}
            # Actual real tile/validation counts must replace the synthetic count.
            checkpoints=[r['checkpoint_bytes'] for r in records if r['step']]
            row={'scene':scene,'method':method,'bands':bands,'classes':classes,'synthetic_shape':[28,28],
                 'synthetic_optimizer_updates':3,'new_real_data_optimizer_updates':0,'elapsed_seconds':elapsed,
                 'components_seconds':times,'component_median_seconds':{k:float(np.median(v)) for k,v in times.items()},
                 'checkpoint_median_bytes':float(np.median(checkpoints)),'checkpoint_max_bytes':max(checkpoints),
                 'proposed_full40_updates_all_three_seeds':nblocks*40*3,
                 'proposed_full40_checkpoint_bytes_estimate':max(checkpoints)*(nblocks*40+1)*3,
                 'estimate_scope':'Shape-specific synthetic measurement; not real-data throughput or convergence estimate. Validation tile counts and data loading still required.',
                 'result':result}
            rows.append(row);print(json.dumps({k:row[k] for k in ('scene','method','elapsed_seconds','checkpoint_max_bytes')}),flush=True)
    value={'code':a.expected_commit,'dependency':NATIVE,'author':AUTHOR,'model':MODEL,'native_losses':LOSSES,
           'synthetic_optimizer_updates':sum(r['synthetic_optimizer_updates'] for r in rows),
           'new_real_data_optimizer_updates':0,'query_reads':0,'confirmation':False,'rows':rows,
           'elapsed_seconds':perf_counter()-start,'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
           'full40_checkpoint_bytes_estimate':sum(r['proposed_full40_checkpoint_bytes_estimate'] for r in rows)}
    new_json(out/'profile.json',value);clean(root,a.expected_commit);clean(a.dependency,NATIVE)
    print(json.dumps({'synthetic_updates':value['synthetic_optimizer_updates'],'real_updates':0,
                      'full40_checkpoint_bytes_estimate':value['full40_checkpoint_bytes_estimate']}))


if __name__=='__main__':main()

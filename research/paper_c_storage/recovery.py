"""Random-boundary recovery against existing C13 synthetic golden states."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .benchmark import BASE,ROOT,clean
from .bridge import import_case,materialize
from .store import Store
from .codec import require
from research.paper_c_spatial_baseline.protocol import admit
from research.paper_c_spatial_baseline.tasks import Task,dependencies
from research.paper_c_spatial_baseline.run import execute


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();clean(a.expected_commit);torch.set_num_threads(1)
    out=Path(a.out).resolve();require(out.is_relative_to(BASE/'runs') and not out.exists(),'New authorized output required');out.mkdir(mode=0o700)
    prior=BASE/'runs/paper-c-spatial-baseline-20261127-01';source=prior/'execution05'
    # Reusing old producer contracts for a TEST is valid only with exact
    # matching engine/adapter bytes. This does not authorize real replay.
    old='01b287412a0737eaa20fc5eb2087ccd47a3a3b93'
    for name in ['research/paper_c_recovery/transactions.py','research/paper_c_spatial_baseline/tasks.py','research/paper_c_spatial_baseline/protocol.py']:
        require(subprocess.check_output(['git','show',old+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'Recovery dependency differs from old producer')
    reg=json.loads((prior/'registration05.json').read_text());require(reg['kind']=='synthetic','Real roles rejected')
    native=reg['native_root'];author=reg['author_root'];adapter,upstream=dependencies(ROOT,native,author)
    data=admit(reg['manifest'],synthetic=True)['data']['toy'];store=Store(out/'optional-v2',layout='zip-xor',max_bytes=64*1024**2)
    rng=np.random.default_rng(614);rows=[];updates=0;exact_files=0
    for method in ['none','x_residual','capacity_control','dsnet']:
        name=f'toy-s64-{method}';case=source/name;contract=json.loads((case/'contract.json').read_text())
        import_case(store,case,name)
        boundary=int(rng.integers(1,4));dest=materialize(store,name,out/(name+'-forensic-prefix'),through=boundary,forensic=True)
        for step in range(boundary+1):
            for file in ['state.pt','record.json','intent.json']:
                require((dest/'stages'/f'{step:06d}'/file).read_bytes()==(case/'stages'/f'{step:06d}'/file).read_bytes(),'Prefix original bytes differ');exact_files+=1
        # Deliberate independently located synthetic replay, never resuming an
        # old real run or deleting a pending intent. Updates are counted anew.
        result=execute(dest,contract,lambda:Task('toy',data,method,adapter,upstream),resume=True)
        updates+=result['new_updates'];states=[]
        for step in range(boundary+1,5):
            actual=torch.load(dest/'stages'/f'{step:06d}'/'state.pt',weights_only=True)
            golden=torch.load(case/'stages'/f'{step:06d}'/'state.pt',weights_only=True)
            require(equal(actual,golden),'Restored model/Adam/RNG/modes/history continuation differs');states.append(step)
        rows.append({'case':name,'random_committed_boundary':boundary,'new_synthetic_updates':result['new_updates'],
                     'exact_continuation_state_steps':states,'all_full_states_equal':True})
    store.checkpoint();store.close()
    value={'schema':'c14_random_synthetic_continuation_v1','producer':a.expected_commit,'cases':rows,
           'new_synthetic_optimizer_updates':updates,'real_updates':0,'query_scores':0,'original_prefix_files_identical':exact_files,
           'original_assets_modified':False,'latest_only_production_restore':True,'older_prefix_used_only_for_explicit_forensic_synthetic_test':True}
    (out/'receipt.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()

"""Reload each selected real checkpoint on the same released development role."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import torch
from tasks import NATIVE,NativeTask,DSNetTask,prior_module,materialize
from transactions import sha,model_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--producer-root',required=True);p.add_argument('--expected-commit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    git=lambda path,*args:subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
    if git(root,'rev-parse','HEAD')!=a.expected_commit or git(root,'status','--porcelain'):raise ValueError('Explicit clean verifier commit required')
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Authorized Linux CPU only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=json.loads((Path(a.run)/'registration.json').read_text());source=Path(a.producer_root)
    if git(source,'rev-parse','HEAD')!=reg['producer'] or git(source,'status','--porcelain'):raise ValueError('Frozen producer changed')
    for n in ('tasks.py','transactions.py'):
        if sha(root/'research/paper_c_recovery'/n)!=sha(source/'research/paper_c_recovery'/n):raise ValueError('Verifier task/state adapter differs from producer')
    if git(reg['dependency_root'],'rev-parse','HEAD')!=NATIVE or git(reg['dependency_root'],'status','--porcelain'):raise ValueError('Native dependency differs')
    first=reg['cases'][0]['contract']['protected']
    for path,h in first.items():
        if sha(path)!=h:raise ValueError('Original protected byte identity changed')
    if any(c['contract']['protected']!=first for c in reg['cases']):raise ValueError('Unexpected mixed input protection')
    expected={x['case']:x for x in json.loads(Path(a.audit).read_text())['cases']}
    sys.path.insert(0,reg['dependency_root']);previous=prior_module(source,'pilot');adapter=prior_module(source,'adapter')
    author=adapter.load_author_model(reg['author_root']);pilot=json.loads(Path(reg['pilot']).read_text());cache={};rows=[];start=perf_counter()
    for case in reg['cases']:
        c=case['contract'];scene=c['scene']
        if scene not in cache:
            bank,receipt=previous.load_bank(SimpleNamespace(runs_root=reg['runs_root']),scene)
            if receipt!=pilot['scenes'][scene]['receipt'] or bank.pin!=pilot['scenes'][scene]['pin']:raise ValueError('Bank admission differs')
            data,contexts=materialize(bank,adapter);cache[scene]=(data,bank.release.classes,bank.release.shape[-1])
        data,classes,bands=cache[scene]
        folder=Path(a.run)/case['id'];r=expected[case['id']];step=r['selected_step'];state_folder=folder/'stages'/f'{step:06d}'
        record=json.loads((state_folder/'record.json').read_text())
        if sha(state_folder/'state.pt')!=record['state_sha256']:raise ValueError('Selected state SHA differs')
        task=NativeTask(data,scene,classes) if c['method']=='native' else DSNetTask(data,scene,classes,bands,adapter,author)
        state=torch.load(state_folder/'state.pt',weights_only=True,map_location='cpu');task.model.load_state_dict(state['model'],strict=True)
        task.model.eval();before=model_hash(task.model);t=perf_counter()
        with torch.no_grad():actual=task.evaluate()
        elapsed=perf_counter()-t
        if actual!=r['selected_metrics'] or model_hash(task.model)!=before:raise ValueError('Selected reload metrics differ or evaluation mutated model')
        rows.append({'case':case['id'],'selected_step':step,'selected_state_sha256':record['state_sha256'],
            'metrics_exactly_equal':True,'new_development_neural_calls':len(data['devval']),
            'evaluation_seconds':elapsed,'optimizer_updates':0})
    result={'status':'PASSED','producer':reg['producer'],'verifier':a.expected_commit,'cases':rows,
        'neural_calls':sum(r['new_development_neural_calls'] for r in rows),'new_optimizer_updates':0,
        'development_point_exposures':1200,'distinct_reused_development_points':200,
        'query_reads':0,'new_labels_released':0,'confirmation':False,'elapsed_seconds':perf_counter()-start}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','neural_calls','new_optimizer_updates','elapsed_seconds')}))


if __name__=='__main__':main()

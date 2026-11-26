"""Audit completed paired engineering cases and aggregate descriptive results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from audit import audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.run);reg=json.loads((root/'registration.json').read_text());results=[];paired=[]
    for case in reg['cases']:
        result=audit(root/case['id']);result['case']=case['id'];results.append(result)
    for scene in sorted({r['scene'] for r in results}):
        seeds=[]
        for seed in (64,512,2048):
            by={r['method']:r for r in results if r['scene']==scene and r['seed']==seed}
            if set(by)!={'native','dsnet'} or by['native']['plan_sha256']!=by['dsnet']['plan_sha256']:
                raise ValueError('Paired methods do not share exact block exposure order')
            seeds.append({'seed':seed,'native_selected_step':by['native']['selected_step'],'dsnet_selected_step':by['dsnet']['selected_step'],
                'native_OA':by['native']['selected_metrics']['oa'],'dsnet_OA':by['dsnet']['selected_metrics']['oa'],
                'dsnet_minus_native_OA_pp':100*(by['dsnet']['selected_metrics']['oa']-by['native']['selected_metrics']['oa']),
                'native_NLL':by['native']['selected_metrics']['nll'],'dsnet_NLL':by['dsnet']['selected_metrics']['nll']})
        paired.append({'scene':scene,'seeds':seeds,'mean_dsnet_minus_native_OA_pp':float(np.mean([r['dsnet_minus_native_OA_pp'] for r in seeds])),
            'mean_native_OA':float(np.mean([r['native_OA'] for r in seeds])),'mean_dsnet_OA':float(np.mean([r['dsnet_OA'] for r in seeds]))})
    total=sum(r['committed_optimizer_updates'] for r in results)
    if total!=reg['real_optimizer_update_budget']:raise ValueError('Committed total differs from registered budget')
    value={'schema':'paper_c_engineering_pair_audit_v1','producer':reg['producer'],'cases':results,'paired':paired,
        'committed_optimizer_updates':total,'independent_audit_new_optimizer_updates':0,'query_reads':0,'confirmation':False,
        'scope':'Three-sweep recovery/exposure engineering on old development roles, not convergence/architecture efficacy or independent sensor generalization.',
        'checkpoint_bytes':sum(r['checkpoint_bytes'] for r in results),
        'measured_component_seconds':{key:sum(r['measured_components'].get(key,{}).get('sum_seconds',0) for r in results)
            for key in sorted({k for r in results for k in r['measured_components']})},
        'cost_limit':'Measured update/selection/checkpoint components only; complete cold loading, bank/source verification, audit, IPC and backup wall time are additional. No GPU or isolated deployment speed claim.',
        'matching':'Same per-scene label IDs, exact per-seed per-sweep block permutation, updates, optimizer and selected event schedule. Different original context/loss/capacity recipes remain disclosed.'}
    out=Path(a.out)
    with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    with out.with_suffix('.csv').open('x',newline='') as f:
        fields=('scene','seed','method','selected_step','OA','AA','NLL','updates');w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            m=r['selected_metrics'];w.writerow(dict(scene=r['scene'],seed=r['seed'],method=r['method'],selected_step=r['selected_step'],OA=m['oa'],AA=m['aa'],NLL=m['nll'],updates=r['committed_optimizer_updates']))
    print(json.dumps({'status':'PASSED','cases':len(results),'updates':total,'paired':paired}))


if __name__=='__main__':main()

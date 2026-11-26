"""Independently prove source-only backbone inheritance and matched fresh targets."""
import argparse
import json
from pathlib import Path

import torch
from audit import audit,sha


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--baseline-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.run);reg=json.loads((root/'registration.json').read_text())
    if sha(a.baseline_audit)!=reg['baseline_audit_sha256']:raise ValueError('Frozen matched-control audit changed')
    base={r['case']:r for r in json.loads(Path(a.baseline_audit).read_text())['cases']};rows=[]
    original_cases={r['id']:r['contract'] for r in json.loads((Path(reg['baseline_run'])/'registration.json').read_text())['cases']}
    for case in reg['cases']:
        c=case['contract'];t=c['transfer'];r=audit(root/case['id']);baseline=base[t['target_scratch_case']]
        initial=torch.load(root/case['id']/'stages'/'000000'/'state.pt',weights_only=True,map_location='cpu')
        source=torch.load(t['source_selected_state'],weights_only=True,map_location='cpu')
        target=torch.load(t['target_initial_state'],weights_only=True,map_location='cpu')
        if (sha(t['source_selected_state'])!=t['source_selected_sha256'] or sha(t['target_initial_state'])!=t['target_initial_sha256']
                or initial['step']!=0 or initial['optimizer']['state'] or target['optimizer']['state']):
            raise ValueError('Initial states or empty target Adam differ')
        for key,value in initial['model'].items():
            expected=source['model'][key] if key.startswith('backbone.') else target['model'][key]
            if not torch.equal(value,expected):raise ValueError('Source backbone or matched fresh target head differs')
        if not equal(initial['rng'],target['rng']) or not equal(initial['optimizer'],target['optimizer']):
            raise ValueError('Target RNG or initial optimizer differs from scratch control')
        old=original_cases[t['target_scratch_case']]
        for key in ('roster','sweeps','plan','updates','selection_steps','optimizer','scene','native_model','native_losses'):
            if c[key]!=old[key]:raise ValueError('Target budget/recipe/exposure differs from paired scratch')
        r.update(case=case['id'],source=t['source'],target=t['target'],source_selected_step=t['source_selected_step'],
            source_inherited_executed_updates=t['source_inherited_executed_updates'],
            source_backbone_exact=True,fresh_target_heads_exact=True,target_initial_rng_optimizer_exact=True,
            target_control_case=t['target_scratch_case'],target_control_selected_metrics=baseline['selected_metrics'],
            transfer_minus_scratch_OA_pp=100*(r['selected_metrics']['oa']-baseline['selected_metrics']['oa']))
        rows.append(r)
    total=sum(r['committed_optimizer_updates'] for r in rows)
    if total!=2007 or len(rows)!=15:raise ValueError('Registered five-direction target budget differs')
    value={'schema':'paper_c_new_five_direction_lineage_audit_v1','status':'PASSED','producer':reg['producer'],'cases':rows,
        'new_target_optimizer_updates':total,'new_source_optimizer_updates':0,'audit_neural_forwards':0,'audit_optimizer_updates':0,
        'directions':sorted({(r['source'],r['target']) for r in rows}),
        'scope':'New native-source3/target3 engineering only; differs from old G3 and needs separate lazy-endpoint/route audit before paired runtime admission.',
        'query_reads':0,'confirmation':False}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASSED','cases':len(rows),'new_target_updates':total,'directions':value['directions']}))


if __name__=='__main__':main()

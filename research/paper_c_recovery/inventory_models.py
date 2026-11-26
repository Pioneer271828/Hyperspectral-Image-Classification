"""Bounded frozen model-sidecar inventory; no tensor deserialization or label reads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket

SCENES=('paviaU','xuzhou','whu-honghu')
REMAINING=tuple((s,t) for s in SCENES for t in SCENES if s!=t and (s,t)!=('paviaU','xuzhou'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):
    if p.is_symlink() or p.stat().st_size>8*1024**2:raise ValueError('Sidecar symlink or size exceeds bounded inventory')
    return json.loads(p.read_text())


def qualify(row,source,target):
    reasons=[]
    if row['source_datasets']!=[source]:reasons.append('not_verified_single_source_ancestry')
    if row['target_dataset']!=target:reasons.append('not_verified_target_adaptation')
    if target not in row['class_counts']:reasons.append('no_matching_target_semantic_head')
    if row['class_counts'].get(target)!=dict(paviaU=9,xuzhou=9,**{'whu-honghu':22})[target]:reasons.append('target_class_count_differs_or_missing')
    if row['graph_mode']!='learned':reasons.append('not_local_learned_graph')
    if row['norm_type']!='group_norm':reasons.append('not_verified_group_norm_lazy_profile')
    if not row['weight_present']:reasons.append('weight_bytes_missing')
    if row['method'] not in ('source_pretrained_full_finetune','source_pretrained_head_only'):reasons.append('not_a_verified_original_target_method')
    return reasons


def inventory(project,roots):
    project=Path(project).resolve();files=set();scopes=[]
    for raw in roots:
        root=Path(raw).resolve()
        if not root.is_relative_to(project/'runs') and not root.is_relative_to(project/'references'):
            raise ValueError('Only explicitly named project runs/references roots may be inventoried')
        scopes.append({'relative':str(root.relative_to(project)),'exists':root.exists()})
        if not root.exists():continue
        for pattern in ('*.pt.json','frozen-task.json'):
            files.update(root.rglob(pattern))
    if len(files)>500:raise ValueError('Inventory exceeds explicit 500-sidecar bound; narrow roots')
    tasks={p:read(p) for p in files if p.name=='frozen-task.json'}
    rows=[]
    for p in sorted(files):
        if p.name=='frozen-task.json':continue
        d=read(p);cfg=d.get('training_config',{});back=d.get('backbone_kwargs',{})
        ancestors=[q for q in tasks if q.parent==p.parent or q.parent in p.parents]
        task=tasks[max(ancestors,key=lambda x:len(x.parts))] if ancestors else {}
        tcfg=task.get('config',{})
        sources=[tcfg['source_dataset']] if tcfg.get('source_dataset') else cfg.get('research_config',{}).get('data',{}).get('datasets',[])
        target=tcfg.get('target_dataset',cfg.get('target_dataset'))
        weight=p.with_suffix('')
        row={'sidecar':str(p.relative_to(project)),'sidecar_sha256':sha(p),
             'weight_present':weight.is_file() and not weight.is_symlink(),'source_datasets':sources,
             'target_dataset':target,'class_counts':d.get('class_counts',{}),'method':cfg.get('method'),
             'graph_mode':back.get('graph_mode'),'norm_type':back.get('norm_type'),
             'optimizer_step':cfg.get('optimizer_steps',cfg.get('checkpoint_optimizer_step')),
             'statistics_hash':cfg.get('statistics_hash'),'task_hash':cfg.get('task_hash',task.get('task_hash')),
             'source_updates':tcfg.get('source_updates'),'target_updates':tcfg.get('target_updates')}
        if row['weight_present']:row.update(weight_bytes=weight.stat().st_size,weight_sha256=sha(weight))
        rows.append(row)
    directions=[]
    for source,target in REMAINING:
        candidates=[r['sidecar'] for r in rows if not qualify(r,source,target)]
        directions.append({'source':source,'target':target,'candidate_sidecars':candidates,
            'gate':'REQUIRES_FROZEN_TASK_WEIGHT_NORMALIZER_ROLE_AUDIT' if candidates else 'NO_ADMITTED_PAIR_IN_EXPLICIT_ROOTS',
            'rejection_counts':dict(Counter(reason for row in rows for reason in qualify(row,source,target))),
            'development_execution_admitted':False,'independent_confirmation':False})
    return {'schema':'paper_c_remaining_five_model_inventory_v1','reader_hostname':socket.gethostname(),
            'scopes':scopes,'sidecars':rows,'frozen_tasks':[{'path':str(p.relative_to(project)),'sha256':sha(p),
                'source':d.get('config',{}).get('source_dataset'),'target':d.get('config',{}).get('target_dataset')} for p,d in sorted(tasks.items())],
            'directions':directions,'new_neural_forwards':0,'new_optimizer_updates':0,'new_GT_or_query_reads':0,
            'limit':'Bounded explicit roots only. Absence here is not proof of absence elsewhere. Equal head cardinality is not shared semantics; target-local or two-source weights are not single-source transfer.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True);p.add_argument('--root',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    value=inventory(a.project,a.root)
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'sidecars':len(value['sidecars']),'frozen_tasks':len(value['frozen_tasks']),
                      'remaining_directions':len(value['directions']),'candidate_directions':sum(bool(d['candidate_sidecars']) for d in value['directions'])}))

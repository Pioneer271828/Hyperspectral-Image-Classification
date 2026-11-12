"""Authenticated selected-state matched deletions on complete development validation.

No retraining/reselection/query. Grid seed coordinates are fixed pixel units,
not learned centroids or physical distances. Unavailable blocks invalidate a
whole control rather than silently reducing its validation population.
"""
from copy import deepcopy
import time
import numpy as np
import torch
from evaluation.graph_edge_intervention import _selected, _probe
from evaluation.matched_edge_controls import freeze, verify
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.development_training import development_counts, sparse_labels
from transfer.graph_route_context import _context, route_state_hash


def coordinates(context):
    if context['node_order']!='row_major_seed_grid_v1' or context['node_position']!='grid_seed_centers':
        raise ValueError('Fixed seed-grid node semantics required')
    gh,gw=context['grid_shape'];h,w=context['shape'][:2]
    ids=torch.arange(gh*gw,dtype=torch.float64)
    return torch.stack(((torch.floor(ids/gw)+.5)*h/gh-.5,(ids%gw+.5)*w/gw-.5),1)


class MatchedReviewPlan:
    def __init__(self,value,*,expected_hash):
        self._value=deepcopy(value)
        if (set(value)!={'version','contract_hash','selected_step','selected_state_hash','contexts','names','plan_hash'}
                or value['version']!=1 or value['plan_hash']!=expected_hash
                or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash):
            raise ValueError('Matched review identity differs')
        self.plan_hash=expected_hash

    def to_dict(self):return deepcopy(self._value)

    @classmethod
    def freeze(cls,run,*,per_row,distance_bins,seeds,time_limit=30.,frozen_controls=None):
        model=_selected(run);state=route_state_hash(model);rows=[]
        for tile in run.record['roles']:
            if tile['role']!='devval':continue
            context=_context(run.scene,run.record,tile['id'])
            _,(graph,_)=_probe(run,model,tile['id'])
            if frozen_controls is None:
                plan=freeze(graph,coordinates(context),per_row=per_row,distance_bins=distance_bins,seeds=seeds,time_limit=time_limit)
            else:
                saved=frozen_controls[tile['id']]
                plan=saved['control_plan']
                if saved['context']!=context or (plan['per_row'],plan['distance_bins'],plan['seeds'],plan['time_limit'])!=(
                        per_row,distance_bins,list(seeds),time_limit):
                    raise ValueError('Frozen validation control settings or context differ')
                verify(plan,graph,coordinates(context),expected_hash=plan['plan_hash'])
            rows.append({'context':context,'control_plan':plan})
        if not rows or state!=route_state_hash(model):raise ValueError('Empty validation or changed selected model')
        run._check()
        value={'version':1,'contract_hash':run.contract_hash,'selected_step':run.selected_step,'selected_state_hash':state,
               'contexts':rows,'names':['none','important','low_weight',*[f'random:{s}' for s in seeds]]}
        value['plan_hash']=digest(value);return cls(value,expected_hash=value['plan_hash'])


def review(run,plan,*,expected_plan_hash,observe=None,blocked_controls=None):
    started=time.perf_counter();model=_selected(run);state=route_state_hash(model)
    if type(plan) is not MatchedReviewPlan:raise ValueError('Authenticated matched review plan required')
    value=MatchedReviewPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict()
    if (value['contract_hash']!=run.contract_hash or value['selected_step']!=run.selected_step
            or value['selected_state_hash']!=state):raise ValueError('Selected training state differs')
    tiles=[t for t in run.record['roles'] if t['role']=='devval'];contexts=[_context(run.scene,run.record,t['id']) for t in tiles]
    if [row['context'] for row in value['contexts']]!=contexts:raise ValueError('Complete validation contexts required')
    # Validate every graph and mask before any intervention/metric calculation.
    unavailable=deepcopy({} if blocked_controls is None else blocked_controls);names=value['names']
    if not set(unavailable)<=set(names[2:]) or any(not reasons for reasons in unavailable.values()):
        raise ValueError('Unrecognized externally blocked control')
    for tile,row in zip(tiles,value['contexts']):
        p=row['control_plan'];expected=['none','important','low_weight',*[f'random:{s}' for s in p['seeds']]]
        if names!=expected or set(p['controls'])!=set(expected[2:]):raise ValueError('Control population differs')
        base,(graph,_)=_probe(run,model,tile['id'],expected_graph_hash=p['adjacency_sha256'])
        verify(p,graph,coordinates(row['context']),expected_hash=p['plan_hash'])
        for name,c in p['controls'].items():
            if c['status']!='optimal':unavailable.setdefault(name,[]).append({'tile_id':tile['id'],'status':c['status'],'reason':c['reason']})
    active=[name for name in names if name not in unavailable]
    matrices={name:np.zeros((3,run.scene.classes,run.scene.classes),dtype=np.int64) for name in active}
    nll={name:np.zeros(3) for name in active};traces={name:[] for name in active}
    truth=sparse_labels(run.record,run.scene.plan,'validation_support');total=development_counts(run.record)[1]
    for tile,row in zip(tiles,value['contexts']):
        base,(graph,_)=_probe(run,model,tile['id'],expected_graph_hash=row['control_plan']['adjacency_sha256'])
        p=row['control_plan'];labels=truth[tile_slice(tile)].ravel();valid=labels>0;y=labels[valid]-1
        for name in active:
            removed=[] if name=='none' else p['baseline']['removed'] if name=='important' else p['controls'][name]['removed']
            if name=='none':outputs,actual=base,graph
            else:outputs,(_,actual)=_probe(run,model,tile['id'],removed,p['adjacency_sha256'])
            if not torch.equal(outputs[2],base[2]):raise ValueError('Deletion changed CNN probabilities')
            for head,prob in enumerate(outputs[:3]):
                q=prob.detach().numpy()[valid];pred=q.argmax(1)
                matrices[name][head]+=np.bincount(y*run.scene.classes+pred,minlength=run.scene.classes**2).reshape(run.scene.classes,run.scene.classes)
                nll[name][head]-=float(np.log(np.maximum(q[np.arange(len(y)),y],1e-8)).sum())
            if observe is not None:
                observe(name,tile['id'],outputs[:3],graph,actual,removed,p)
            traces[name].append({'tile_id':tile['id'],'removed':removed,'original_nnz':graph._nnz(),'actual_nnz':actual._nnz(),
                                'probability_hashes':[tensor_hash(x) for x in outputs[:3]],'actual_graph_hash':tensor_hash(actual)})
    result={name:{'status':'unavailable','blocking_contexts':reasons,'metrics':None} for name,reasons in unavailable.items()}
    for name,m in matrices.items():
        support=m.sum(2)
        if not np.all(support==5) or not np.all(support.sum(1)==total):raise ValueError('Fixed class support differs')
        recall=np.stack([np.diag(c) for c in m])/support
        result[name]={'status':'evaluated','metrics':{'confusion':m.tolist(),'recall':recall.tolist(),'support':support.tolist(),
            'oa':[float(np.trace(c)/total) for c in m],'aa':recall.mean(1).tolist(),'nll':(nll[name]/total).tolist()},'traces':traces[name]}
    baseline=result['none']['metrics'];selected=run.history[run.selected_step-1]['validation']
    if baseline['confusion'][0]!=selected['confusion'] or baseline['nll'][0]!=selected['nll']:
        raise ValueError('Original selected validation changed')
    if state!=route_state_hash(model):raise ValueError('Selected weights changed')
    run._check()
    return {'plan_hash':expected_plan_hash,'selected_step':run.selected_step,'controls':result,'optimizer_updates':0,'query':0,
            'reselection':False,'elapsed_seconds':time.perf_counter()-started,
            'scope':'Frozen post-route sensitivity on reused development validation; unavailable means whole control omitted, not a reduced population'}

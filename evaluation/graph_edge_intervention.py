"""Frozen selected-state edge deletion controls; no optimization or query access."""
from copy import deepcopy
import time
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from monitoring.graph_support import _matrix
from research_runtime import tensor_hash
from transfer.development_training import sparse_labels
from transfer.graph_route_context import _context,route_state_hash
from transfer.graph_route_training import GraphRouteRun


def deletion_masks(adjacency, *, per_row, random_seeds, tile_id):
    indices,values,_=_matrix(adjacency,'A');n=adjacency.shape[0]
    if adjacency.shape[1]!=n or int((indices[0]==indices[1]).sum())!=n:
        raise ValueError('Square graph with mandatory self loops required')
    if type(per_row) is not int or per_row<0 or type(tile_id) is not int or tile_id<0:
        raise ValueError('Explicit nonnegative row deletion count and tile id required')
    if (not isinstance(random_seeds,(list,tuple)) or not random_seeds or len(set(random_seeds))!=len(random_seeds)
            or any(type(s) is not int or not 0<=s<2**32 for s in random_seeds)):
        raise ValueError('Unique explicit uint32 random-control seeds required')
    masks={'important':[]};masks.update({f'random:{s}':[] for s in random_seeds})
    rng={s:np.random.default_rng(np.random.SeedSequence([s,tile_id])) for s in random_seeds}
    counts=[];available=[];degenerate=[]
    for i in range(n):
        ids=np.flatnonzero((indices[0]==i)&(indices[1]!=i));k=min(per_row,len(ids));counts.append(k);available.append(len(ids))
        if k==0 or k==len(ids):degenerate.append(i)
        ranked=sorted(ids.tolist(),key=lambda t:(-float(values[t]),int(indices[1,t])))
        masks['important'].extend(int(i*n+indices[1,t]) for t in ranked[:k])
        for seed in random_seeds:
            chosen=rng[seed].permutation(ids)[:k]
            masks[f'random:{seed}'].extend(int(i*n+indices[1,t]) for t in chosen)
    return dict(removed={k:sorted(v) for k,v in masks.items()},counts_per_row=counts,available_per_row=available,
                degenerate_rows=degenerate,requested_per_row=per_row)


def delete_support(adjacency, removed):
    indices,_,_=_matrix(adjacency,'A');n=adjacency.shape[0]
    if adjacency.shape[1]!=n or int((indices[0]==indices[1]).sum())!=n:raise ValueError('Mandatory self loops required')
    if not isinstance(removed,list) or any(type(e) is not int for e in removed) or removed!=sorted(set(removed)):
        raise ValueError('Sorted unique directed edge ids required')
    edges=(indices[0]*n+indices[1]).tolist()
    if not set(removed)<=set(edges) or any(e//n==e%n for e in removed):raise ValueError('Delete only existing nonself edges')
    if not removed:return adjacency
    removed_set=set(removed)
    keep=torch.tensor([e not in removed_set for e in edges],dtype=torch.bool)
    ix=adjacency.indices()[:,keep];values=adjacency.values()[keep]
    maxima=values.new_zeros(n).scatter_reduce(0,ix[0],values.detach(),reduce='amax')
    scaled=values/maxima[ix[0]];mass=values.new_zeros(n).scatter_add(0,ix[0],scaled)
    result=torch.sparse_coo_tensor(ix,scaled/mass[ix[0]],adjacency.shape).coalesce();_matrix(result,'pruned A')
    return result


def _selected(run):
    if type(run) is not GraphRouteRun:raise ValueError('Accepted authenticated GraphRouteRun required')
    run.finish();model=deepcopy(run.model).eval();model.load_state_dict(run.selected_state)
    return model


def _probe(run,model,tile_id,removed=None,expected_graph_hash=None):
    router=model.backbone.graph_router
    if 'forward' in router.__dict__:raise ValueError('Router already intercepted')
    original=router.forward;graphs=[]
    def forward(*args,**kwargs):
        graph,info=original(*args,**kwargs)
        if expected_graph_hash is not None and tensor_hash(graph)!=expected_graph_hash:raise ValueError('Frozen final graph changed')
        actual=graph if removed is None else delete_support(graph,removed)
        graphs.append((graph.detach().clone(),actual.detach().clone()))
        if removed:
            info=deepcopy(info);degree=torch.bincount(actual.indices()[0],minlength=actual.shape[0])
            info.update(route_output_nnz=actual._nnz(),route_min_degree=int(degree.min()),route_max_degree=int(degree.max()),
                        route_indices=actual.indices().clone(),route_weights=actual.values().clone())
        return actual,info
    router.forward=forward
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():outputs=run._predict(model,run.scene.plan['tiles'][tile_id],0)
    finally:del router.forward
    if len(graphs)!=1:raise ValueError('Exactly one routed stage required')
    return outputs,graphs[0]


class EdgeDeletionPlan:
    def __init__(self,value,*,expected_hash):
        self._value=deepcopy(value)
        if (set(value)!={'version','contract_hash','selected_step','selected_state_hash','per_row','random_seeds','contexts','plan_hash'}
                or value['version']!=1 or value['plan_hash']!=expected_hash
                or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash):
            raise ValueError('Frozen deletion plan identity differs')
        self.plan_hash=expected_hash

    def to_dict(self):return deepcopy(self._value)

    @classmethod
    def freeze(cls,run,*,per_row,random_seeds):
        model=_selected(run);state=route_state_hash(model);rows=[]
        for tile in run.record['roles']:
            if tile['role']!='devval':continue
            context=_context(run.scene,run.record,tile['id']);_,(graph,_)=_probe(run,model,tile['id'])
            masks=deletion_masks(graph,per_row=per_row,random_seeds=random_seeds,tile_id=tile['id'])
            rows.append(dict(context=context,graph_hash=tensor_hash(graph),masks=masks))
        if state!=route_state_hash(model):raise ValueError('Plan construction changed selected model')
        run._check()
        value=dict(version=1,contract_hash=run.contract_hash,selected_step=run.selected_step,selected_state_hash=state,
                   per_row=per_row,random_seeds=list(random_seeds),contexts=rows)
        value['plan_hash']=digest(value);return cls(value,expected_hash=value['plan_hash'])


def review_deletions(run,plan,*,expected_plan_hash):
    started=time.perf_counter();model=_selected(run)
    if not isinstance(plan,EdgeDeletionPlan):raise ValueError('Frozen deletion plan required')
    value=EdgeDeletionPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict();state=route_state_hash(model)
    if value['contract_hash']!=run.contract_hash or value['selected_step']!=run.selected_step or value['selected_state_hash']!=state:
        raise ValueError('Plan belongs to another selected state')
    tiles=[t for t in run.record['roles'] if t['role']=='devval']
    if [r['context'] for r in value['contexts']]!=[_context(run.scene,run.record,t['id']) for t in tiles]:
        raise ValueError('Frozen validation contexts differ')
    names=['none','important',*[f'random:{s}' for s in value['random_seeds']]]
    matrices={name:np.zeros((3,run.scene.classes,run.scene.classes),dtype=np.int64) for name in names}
    nll={name:np.zeros(3) for name in names};traces={name:[] for name in names};labels=sparse_labels(run.record,run.scene.plan,'validation_support')
    for tile,row in zip(tiles,value['contexts']):
        tid=tile['id'];base,(graph,_)=_probe(run,model,tid,expected_graph_hash=row['graph_hash'])
        if row['masks']!=deletion_masks(graph,per_row=value['per_row'],random_seeds=value['random_seeds'],tile_id=tid):
            raise ValueError('Frozen per-row controls differ')
        truth=labels[tile_slice(tile)].ravel();valid=truth>0;y=truth[valid]-1
        for name in names:
            removed=[] if name=='none' else row['masks']['removed'][name]
            if name=='none':outputs,actual=base,graph
            else:
                outputs,(_,actual)=_probe(run,model,tid,removed,row['graph_hash'])
            if not torch.equal(outputs[2],base[2]):raise ValueError('Post-route deletion unexpectedly changed CNN head')
            for head,p in enumerate(outputs[:3]):
                prob=p.detach().numpy()[valid];pred=prob.argmax(1)
                matrices[name][head]+=np.bincount(y*run.scene.classes+pred,minlength=run.scene.classes**2).reshape(run.scene.classes,run.scene.classes)
                nll[name][head]-=float(np.log(np.maximum(prob[np.arange(len(y)),y],1e-8)).sum())
            traces[name].append(dict(tile_id=tid,original_graph_hash=row['graph_hash'],actual_graph_hash=tensor_hash(actual),
                original_nnz=graph._nnz(),actual_nnz=actual._nnz(),removed=removed,probability_hashes=[tensor_hash(x) for x in outputs[:3]]))
    result={}
    for name in names:
        m=matrices[name];support=m.sum(2)
        if not np.all(support==5) or not np.all(support.sum(1)==45):raise ValueError('Fixed validation support changed')
        recall=np.stack([np.diag(x) for x in m])/support
        result[name]=dict(confusion=m.tolist(),recall=recall.tolist(),oa=[float(np.trace(x)/45) for x in m],
                         aa=recall.mean(1).tolist(),nll=(nll[name]/45).tolist(),support=support.tolist(),traces=traces[name])
    selected=run.history[run.selected_step-1]['validation']
    if result['none']['confusion'][0]!=selected['confusion'] or result['none']['nll'][0]!=selected['nll']:
        raise ValueError('No-deletion baseline differs from original selected validation')
    if state!=route_state_hash(model):raise ValueError('Read-only intervention changed weights')
    run._check()
    return dict(plan_hash=expected_plan_hash,selected_step=run.selected_step,selected_state_hash=state,controls=result,
        identical_important_random={name:all(a['removed']==b['removed'] for a,b in zip(traces['important'],traces[name])) for name in names if name.startswith('random:')},
        query=0,optimizer_updates=0,reselection=False,elapsed_seconds=time.perf_counter()-started,
        interpretation='post-final-route deletion sensitivity on reused selection validation; not independent accuracy or deployed speed')

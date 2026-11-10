"""Fixed-reference normalized Dirichlet energy; descriptive, never a selector.

Independent implementation of arXiv:2006.13318v1 Definition 3.1. Directed
message graphs are explicitly symmetrized for this diagnostic only. Neither
that paper's convergence theorem nor accuracy benefits are asserted here.
"""
from copy import deepcopy
import math
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.graph_route_context import _context, route_state_hash
from transfer.graph_route_training import GraphRouteRun
from transfer.development_training import sparse_labels
from .classification_metrics import classification_metrics
from .representation_redundancy import matrix_summary
from .graph_edge_intervention import _selected


def _sparse(a):
    if (not torch.is_tensor(a) or a.device.type!='cpu' or a.layout!=torch.sparse_coo
            or a.ndim!=2 or a.shape[0]!=a.shape[1] or not 1<=a.shape[0]<=8192
            or not a.is_coalesced() or not a.is_floating_point() or a._nnz()>262144):
        raise ValueError('Bounded coalesced CPU floating square COO required')
    ix=a.indices();v=a.values().detach().double()
    if not torch.isfinite(v).all() or (v<0).any() or (ix<0).any() or (ix>=a.shape[0]).any():
        raise ValueError('Nonnegative finite valid graph entries required')
    return ix,v


class ReferenceGraph:
    """Immutable sparse diagnostic projection, bound to ordered node context."""
    def __init__(self,value,*,expected_hash):
        v=deepcopy(value)
        if (set(v)!={'version','context','node_ids','indices','weights','reference_hash'} or v['version']!=1
                or v['reference_hash']!=expected_hash or digest({k:x for k,x in v.items() if k!='reference_hash'})!=expected_hash
                or not isinstance(v['context'],dict) or not v['context'] or not isinstance(v['node_ids'],list)
                or any(type(i) is not int or i<0 for i in v['node_ids']) or len(set(v['node_ids']))!=len(v['node_ids'])):
            raise ValueError('Reference identity/context/node order differs')
        n=len(v['node_ids'])
        if (not isinstance(v['indices'],list) or len(v['indices'])!=2
                or any(not isinstance(row,list) or any(type(i) is not int for i in row) for row in v['indices'])):
            raise ValueError('Integer reference edge indices required')
        try:
            ix=torch.tensor(v['indices'],dtype=torch.int64);w=torch.tensor(v['weights'],dtype=torch.float64)
            if ix.ndim!=2 or ix.shape!=(2,len(w)):raise ValueError('Reference COO shape differs')
            if v['indices']!=ix.tolist():raise ValueError('Integer edge indices required')
            a=torch.sparse_coo_tensor(ix,w,(n,n)).coalesce();_sparse(a)
        except (TypeError,RuntimeError,IndexError) as exc:raise ValueError('Malformed reference graph') from exc
        if (a.indices().tolist()!=v['indices'] or a.values().tolist()!=v['weights'] or not (w>0).all()
                or not torch.equal(a.transpose(0,1).coalesce().indices(),a.indices())
                or not torch.equal(a.transpose(0,1).coalesce().values(),a.values())):
            raise ValueError('Canonical positive symmetric reference required')
        loops=ix[0]==ix[1]
        if int(loops.sum())!=n or not (w[loops]==1).all():raise ValueError('Exactly unit self loops required')
        degree=torch.zeros(n,dtype=torch.float64).scatter_add_(0,ix[0],w)
        if not torch.isfinite(degree).all():raise ValueError('Degree overflows FP64')
        self._value=v;self.reference_hash=expected_hash

    def to_dict(self):return deepcopy(self._value)

    @classmethod
    def freeze(cls,adjacency,*,context,node_ids):
        ix,w=_sparse(adjacency);n=adjacency.shape[0]
        if len(node_ids)!=n:raise ValueError('One ordered id per node required')
        keep=(ix[0]!=ix[1])&(w>0);ix=ix[:,keep];w=w[keep]/2
        diagonal=torch.arange(n)
        a=torch.sparse_coo_tensor(torch.cat((ix,ix.flip(0),torch.stack((diagonal,diagonal))),1),
            torch.cat((w,w,torch.ones(n,dtype=torch.float64))),(n,n)).coalesce();_sparse(a)
        v=dict(version=1,context=deepcopy(context),node_ids=list(node_ids),indices=a.indices().tolist(),weights=a.values().tolist())
        v['reference_hash']=digest(v);return cls(v,expected_hash=v['reference_hash'])


@torch.no_grad()
def energy_summary(features,reference,*,expected_hash,context,node_ids,norm_floor=1e-24):
    if not isinstance(reference,ReferenceGraph):raise ValueError('Frozen ReferenceGraph required')
    v=ReferenceGraph(reference.to_dict(),expected_hash=expected_hash).to_dict();n=len(v['node_ids'])
    if context!=v['context'] or list(node_ids)!=v['node_ids']:raise ValueError('Reference node semantics differ')
    if (not torch.is_tensor(features) or features.layout!=torch.strided or features.ndim!=2
            or features.shape[0]!=n or not 1<=features.shape[1]<=512 or not features.is_floating_point()
            or type(norm_floor) not in (int,float) or not math.isfinite(norm_floor) or norm_floor<0):
        raise ValueError('Finite bounded features and norm floor required')
    x=features.detach().to(device='cpu',dtype=torch.float64)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite features')
    ix=torch.tensor(v['indices']);w=torch.tensor(v['weights'],dtype=torch.float64)
    degree=torch.zeros(n,dtype=torch.float64).scatter_add_(0,ix[0],w);z=x/degree.sqrt()[:,None]
    energies=torch.zeros(x.shape[1],dtype=torch.float64)
    for start in range(0,len(w),1024):
        rows,cols=ix[:,start:start+1024];weights=w[start:start+1024]
        energies+=(weights[:,None]*(z[rows]-z[cols]).square()).sum(0)/2
    norms=x.square().sum(0);total=norms.sum();energy=energies.sum()
    if not torch.isfinite(total) or not torch.isfinite(energy):raise ValueError('Energy/norm overflows FP64')
    nonself=int((ix[0]!=ix[1]).sum())
    reason='no_cross_node_edges' if nonself==0 else 'feature_norm_below_floor' if total<=norm_floor else None
    normalized=None if reason else float(energy/total)
    if normalized is not None and not 0<=normalized<=2+1e-12:raise ValueError('Normalized energy outside spectral range')
    return dict(reference_hash=expected_hash,nodes=n,channels=x.shape[1],nonself_directed_entries=nonself,
        projection='W=sym(A0 without diagonal)+I; L=I-D^-1/2 W D^-1/2',
        energy=float(energy),feature_norm_squared=float(total),energy_by_channel=energies.tolist(),norm_squared_by_channel=norms.tolist(),
        normalized_energy=normalized,normalization='trace(X.T L X)/sum(X**2)',norm_floor=norm_floor,reason=reason,
        feature_hash=tensor_hash(features),representation=matrix_summary(x),
        interpretation='fixed reference descriptive energy and centered covariance participation ratio, not an oversmoothing decision')


def _capture(run,model,tile_id):
    """Observe private original methods, returning exactly original predictions."""
    b=model.backbone;g=b.learnable_graph
    targets=[(g,'build_graph'),(b,'graph_stage1'),(b,'graph_stage2')]
    if any(name in owner.__dict__ for owner,name in targets):raise ValueError('Already intercepted private model')
    build=g.build_graph;s1=b.graph_stage1;s2=b.graph_stage2;values={}
    def graph(*args,**kw):
        if 'Q' in values:raise ValueError('Repeated graph construction')
        result=build(*args,**kw);values.update(Q=result.Q.detach().clone(),A0=result.A0.detach().clone());return result
    def first(x,a,*args,**kw):
        if 'input' in values:raise ValueError('Repeated stage1')
        result=s1(x,a,*args,**kw);values.update(input=x.detach().clone(),stage1=result.detach().clone());return result
    def second(x,a,*args,**kw):
        if 'stage2' in values:raise ValueError('Repeated stage2')
        result=s2(x,a,*args,**kw);values.update(stage2=result.detach().clone(),final_A=a.detach().clone());return result
    before=route_state_hash(model)
    g.build_graph=graph;b.graph_stage1=first;b.graph_stage2=second
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():outputs=run._predict(model,run.scene.plan['tiles'][tile_id],0)
    finally:
        for owner,name in reversed(targets):delattr(owner,name)
    if set(values)!={'Q','A0','input','stage1','stage2','final_A'} or before!=route_state_hash(model):raise ValueError('Incomplete/state-changing capture')
    with torch.random.fork_rng(devices=[]),torch.no_grad():direct=run._predict(model,run.scene.plan['tiles'][tile_id],0)
    if any(not torch.equal(a,b) for a,b in zip(outputs[:3],direct[:3])):raise ValueError('Capture changed prediction')
    return values,outputs


def freeze_source_reference(run,tile_id):
    if type(run) is not GraphRouteRun:raise ValueError('Authenticated completed GraphRouteRun required')
    run.finish();context=_context(run.scene,run.record,tile_id);source=deepcopy(run.source).eval()
    captured,_=_capture(run,source,tile_id);n=math.prod(context['grid_shape'])
    if captured['A0'].shape!=(n,n):raise ValueError('Unexpected source grid')
    ref=ReferenceGraph.freeze(captured['A0'],context=context,node_ids=list(range(n)))
    value=dict(version=1,context=context,source_state=route_state_hash(run.source),source_Q_hash=tensor_hash(captured['Q']),
        source_A0_hash=tensor_hash(captured['A0']),reference=ref.to_dict(),norm_floor=1e-24,
        stages=['input','stage1','stage2'],selection='existing selected checkpoint, route step0',query=0)
    value['plan_hash']=digest(value);return value


def review_smoothness(run,plan,*,expected_plan_hash):
    if (not isinstance(plan,dict) or plan.get('plan_hash')!=expected_plan_hash
            or digest({k:v for k,v in plan.items() if k!='plan_hash'})!=expected_plan_hash):raise ValueError('Plan identity differs')
    if freeze_source_reference(run,plan['context']['tile_id'])!=plan:raise ValueError('Frozen common source/context changed')
    source_before=route_state_hash(run.source);before=route_state_hash(run.model);model=_selected(run)
    tid=plan['context']['tile_id'];values,outputs=_capture(run,model,tid)
    reference=ReferenceGraph(plan['reference'],expected_hash=plan['reference']['reference_hash']);n=len(plan['reference']['node_ids'])
    summaries={name:energy_summary(values[name],reference,expected_hash=reference.reference_hash,
        context=plan['context'],node_ids=list(range(n)),norm_floor=plan['norm_floor']) for name in plan['stages']}
    y0,y1,x0,x1=run.scene.plan['tiles'][tid]['box'];role=plan['context']['role']
    labels=sparse_labels(run.record,run.scene.plan,'train_support' if role=='devtrain' else 'validation_support')[y0:y1,x0:x1]
    heads={name:classification_metrics(labels,(p.argmax(1)+1).numpy().reshape(labels.shape),run.record['classes'],labels>0)
        for name,p in zip(('fused','graph','cnn'),outputs[:3])}
    if source_before!=route_state_hash(run.source) or before!=route_state_hash(run.model):raise ValueError('Read-only review changed model')
    run._check()
    return dict(plan_hash=expected_plan_hash,reference_hash=reference.reference_hash,contract_hash=run.contract_hash,
        selected_step=run.selected_step,selected_state_hash=route_state_hash(model),stages=summaries,heads=heads,
        current_Q_hash=tensor_hash(values['Q']),current_A0_hash=tensor_hash(values['A0']),current_final_A_hash=tensor_hash(values['final_A']),
        probability_hashes=[tensor_hash(p) for p in outputs[:3]],query=0,optimizer_updates=0,
        limitations=['same seed-grid node ids, but learned soft memberships may differ',
        'directed propagation is not the symmetric diagnostic graph','no automatic oversmoothing/accuracy conclusion'])

"""Read-only, directed support diagnostics on authenticated development states.

A row i -> j denotes dependency of destination i on source j (A @ X).
Soft Q column mass is not a labelled object size or a class-recall metric.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.graph_route_context import _context, route_state_hash
from transfer.graph_route_training import GraphRouteRun


def _matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.sparse_coo
            or value.device.type != 'cpu' or value.ndim != 2 or min(value.shape) <= 0
            or not value.is_coalesced() or not value.is_floating_point()):
        raise ValueError(name+' must be coalesced CPU floating COO')
    values = value.values().detach()
    if not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError(name+' must contain only finite positive stored entries')
    index = value.indices()
    if any(bool(((index[k] < 0) | (index[k] >= value.shape[k])).any()) for k in (0,1)):
        raise ValueError(name+' has invalid indices')
    sums = torch.zeros(value.shape[0], dtype=values.dtype).scatter_add_(0, index[0], values)
    error = float((sums-1).abs().max())
    if error > 2e-5: raise ValueError(name+' has nonstochastic/empty rows')
    return index.numpy(), values.numpy(), error


def _components(adj, reverse):
    n = len(adj); seen = set(); weak = []
    for start in range(n):
        if start in seen: continue
        group = []; todo = [start]; seen.add(start)
        while todo:
            i = todo.pop(); group.append(i)
            for j in (*adj[i], *reverse[i]):
                if j not in seen: seen.add(j); todo.append(j)
        weak.append(sorted(group))
    seen = set(); order = []
    for start in range(n):
        if start in seen: continue
        seen.add(start); stack = [(start, iter(adj[start]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None: order.append(node); stack.pop()
            elif child not in seen: seen.add(child); stack.append((child, iter(adj[child])))
    seen = set(); strong = []
    for start in reversed(order):
        if start in seen: continue
        seen.add(start); todo = [start]; group = []
        while todo:
            i = todo.pop(); group.append(i)
            for j in reverse[i]:
                if j not in seen: seen.add(j); todo.append(j)
        strong.append(sorted(group))
    return sorted(weak, key=lambda x:x[0]), sorted(strong, key=lambda x:x[0])


def support_summary(adjacency, assignment, *, small_mass_cutoff):
    """O(nodes + stored edges + assignments); no dense graph/Q allocation."""
    if type(small_mass_cutoff) not in (int,float) or not math.isfinite(small_mass_cutoff) or small_mass_cutoff < 0:
        raise ValueError('Explicit finite nonnegative soft-pixel mass cutoff required')
    ai,av,ae = _matrix(adjacency, 'A'); qi,qv,qe = _matrix(assignment, 'Q')
    n = adjacency.shape[0]
    if adjacency.shape[1] != n or assignment.shape[1] != n: raise ValueError('Graph/assignment node count differs')
    rows, cols = ai; edges = (rows*n+cols).tolist(); adj = [[] for _ in range(n)]; reverse = [[] for _ in range(n)]
    for i,j in zip(rows,cols): adj[int(i)].append(int(j)); reverse[int(j)].append(int(i))
    out = np.bincount(rows,minlength=n); inc = np.bincount(cols,minlength=n)
    loops = rows==cols; self_nodes = rows[loops]; nonself = ~loops
    if len(self_nodes) != n: raise ValueError('Mandatory self loop missing')
    out_nonself = np.bincount(rows[nonself],minlength=n); in_nonself = np.bincount(cols[nonself],minlength=n)
    weak,strong = _components(adj,reverse)
    mass = np.bincount(qi[1],weights=qv,minlength=n); small = np.flatnonzero(mass <= small_mass_cutoff)
    return dict(convention='row i depends on column j; messages flow j to i in A@X', nodes=n, pixels=assignment.shape[0],
        graph_hash=tensor_hash(adjacency), assignment_hash=tensor_hash(assignment), edges=edges, nnz=len(edges),
        row_dependency_degree=out.tolist(), column_dependency_degree=inc.tolist(),
        row_nonself_degree=out_nonself.tolist(), column_nonself_degree=in_nonself.tolist(),
        self_loop_count=int(loops.sum()), self_loop_fraction=float(loops.sum()/len(edges)),
        weak_components=weak, strong_components=strong, nonself_isolated_nodes=np.flatnonzero((out_nonself+in_nonself)==0).tolist(),
        soft_pixel_mass=mass.tolist(), empty_nodes=np.flatnonzero(mass==0).tolist(), small_mass_cutoff=small_mass_cutoff,
        small_nodes=small.tolist(), small_row_connections=int(out[small].sum()), small_nonself_row_connections=int(out_nonself[small].sum()),
        graph_row_error=ae, assignment_row_error=qe,
        size_definition='sum of soft Q weights, not labelled connected-component area')


def support_change(current, previous):
    """Compare same registered context at strictly increasing checkpoint steps."""
    for value in (current,previous):
        if value.get('snapshot_hash') != digest({k:v for k,v in value.items() if k!='snapshot_hash'}):
            raise ValueError('Snapshot identity changed')
    if (current['contract_hash'] != previous['contract_hash'] or current['context'] != previous['context']
            or current['state_kind'] != previous['state_kind'] or current['probe_route_step'] != previous['probe_route_step']
            or current['summary']['small_mass_cutoff'] != previous['summary']['small_mass_cutoff']
            or current['summary']['nodes'] != previous['summary']['nodes']
            or current['optimization_step'] <= previous['optimization_step']):
        raise ValueError('Support changes require same series/context and increasing checkpoint step')
    n = current['summary']['nodes']; results = {}
    for name,exclude_self in (('all',False),('nonself',True)):
        def selected(v): return {e for e in v['summary']['edges'] if not exclude_self or e//n!=e%n}
        before,after = selected(previous),selected(current); union = before|after; common = before&after
        results[name] = dict(before=len(before),after=len(after),added=len(after-before),removed=len(before-after),
            retained=len(common),jaccard_distance=len(before^after)/len(union) if union else 0.,
            retained_fraction=len(common)/len(before) if before else None)
    return dict(previous_snapshot_hash=previous['snapshot_hash'],current_snapshot_hash=current['snapshot_hash'],
                definition='symmetric difference / union of directed supports; fixed route step checkpoint probes',**results)


def capture_support(run, tile_id, *, state_kind, small_mass_cutoff):
    """Inspect a private model copy at route step0; never update or reselect.

    This measures checkpoint probes, not historical training-call edge traces.
    The caller must authenticate any checkpoint before loading GraphRouteRun.
    """
    if type(run) is not GraphRouteRun or state_kind not in ('last','selected'):
        raise ValueError('Accepted GraphRouteRun and explicit last/selected state required')
    run._check(); context = _context(run.scene,run.record,tile_id)
    if state_kind=='selected' and run.selected_state is None: raise ValueError('No selected state exists')
    source_before = route_state_hash(run.source); before = route_state_hash(run.model)
    model = deepcopy(run.model).eval()
    if state_kind=='selected': model.load_state_dict(run.selected_state)
    model_hash = route_state_hash(model); graph = model.backbone.learnable_graph; router = model.backbone.graph_router
    if 'build_graph' in graph.__dict__ or 'forward' in router.__dict__:
        raise ValueError('Model already intercepted')
    original_graph,original_route = graph.build_graph,router.forward; observed_q=[]; observed_a=[]
    def build(*args,**kwargs):
        result=original_graph(*args,**kwargs);observed_q.append(result.Q.detach().clone());return result
    def route(*args,**kwargs):
        result=original_route(*args,**kwargs);observed_a.append(result[0].detach().clone());return result
    graph.build_graph=build;router.forward=route
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            outputs=run._predict(model,run.scene.plan['tiles'][tile_id],0)
    finally:
        del graph.build_graph;del router.forward
    if len(observed_q)!=1 or len(observed_a)!=1: raise ValueError('Exactly one actual Q and final routed A required')
    summary=support_summary(observed_a[0],observed_q[0],small_mass_cutoff=small_mass_cutoff)
    if summary['nodes']!=math.prod(context['grid_shape']): raise ValueError('Unexpected node ordering/shape')
    if source_before!=route_state_hash(run.source) or before!=route_state_hash(run.model) or model_hash!=route_state_hash(model):
        raise ValueError('Read-only support probe changed model state')
    run._check()
    value=dict(version=1,contract_hash=run.contract_hash,context=context,state_kind=state_kind,
        optimization_step=run.completed if state_kind=='last' else run.selected_step,probe_route_step=0,
        model_state_hash=model_hash,summary=summary,probability_hashes=[tensor_hash(x) for x in outputs[:3]],
        observation='post-checkpoint no-grad eval copy; not historical training-call trace',query=0,optimizer_updates=0)
    value['snapshot_hash']=digest(value);return value

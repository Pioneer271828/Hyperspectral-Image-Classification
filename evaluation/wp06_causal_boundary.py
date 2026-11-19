"""Probe alternative routes using the same source swap on the frozen native network."""

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from evaluation.controlled_signal import SignalPlan, review_signals
from evaluation.frozen_training_stages import require
from transfer import wp_causal_graph_design as design, wp_region_graph as regional
from transfer.band_graph_control import row_coo
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.wp_causal_graph import source_swap


@contextmanager
def normalization_context(model, moments, *, freeze=False):
    """Hold each actual GroupNorm's observed baseline moments for a locality probe."""
    originals={}
    for name,module in model.named_modules():
        if not isinstance(module,nn.GroupNorm):
            continue
        require("forward" not in vars(module),"Normalization boundary already instrumented")
        originals[module]=module.forward

        def visit(x,module=module,name=name,original=module.forward):
            groups=x.reshape(x.shape[0],module.num_groups,-1)
            if not freeze:
                moments[name]=torch.var_mean(groups,dim=-1,unbiased=False,keepdim=True)
                return original(x)
            variance,mean=moments[name]
            unit=((groups-mean)*(variance+module.eps).rsqrt()).reshape_as(x)
            if module.affine:
                dims=(1,-1)+(1,)*(x.ndim-2)
                unit=unit*module.weight.view(dims)+module.bias.view(dims)
            return unit

        module.forward=visit
    try:
        yield
    finally:
        for module in originals:
            del module.forward


def _positive(graph):
    positive=graph.coalesce().values()>0
    return graph.indices()[:,positive].T.cpu().tolist()


def _native_reach(q,a,membership):
    groups={}
    for cell,region in q:
        groups.setdefault(cell,set()).add(region)
    edges=[tuple(edge) for edge in a]
    result={}
    for target,source in zip(design.TARGETS,design.SOURCES,strict=True):
        source_cells=np.flatnonzero(membership==source)
        targets=np.flatnonzero(membership==target)
        source_regions={j for cell in source_cells for j in groups[cell]}
        intersection={j for cell in targets for j in groups[cell]}&source_regions
        reachable=sum(any(design.reachable(edges,edges,j,s) for j in groups[cell] for s in source_regions)
                      for cell in targets)
        result[str(target)]=dict(target_pixels=len(targets),source_regions=len(source_regions),
            shared_assignments=len(intersection),reachable=reachable)
    return result


@torch.no_grad()
def native_boundary(model, actual, producer, execution, contract):
    tile=str(actual.receipt["source"]["tile"])
    require(contract["tiles"][tile]["regime"]=="new_path","Native test must use its predeclared separated source")
    members=design.memberships(contract["layout"]).reshape(-1)
    indices=torch.from_numpy(np.flatnonzero(np.isin(members,design.TARGETS)))
    positions=[i for i,b in enumerate(actual.view.bands.band_ids) if b in contract["cue_bands"]]
    target_xy=np.argwhere(np.isin(members.reshape(design.SIZE,design.SIZE),design.TARGETS))
    source_xy=np.argwhere(np.isin(members.reshape(design.SIZE,design.SIZE),design.SOURCES))
    separation=int(np.abs(target_xy[:,None]-source_xy[None,:]).max(2).min())
    def swap(x):
        member=torch.from_numpy(members.copy()).to(x.device)
        return source_swap(x,member,positions,design.SOURCES)
    def capture(changed):
        bindings={"input_transform":swap} if changed else {}
        with regional.bind(model,audit_native=True,**bindings) as frame:
            out,_=forward(model,actual,producer,"devval",execution=execution,graph_scale=regional.SCALE)
        return frame,out
    state=state_hash(model)
    moments={}
    with normalization_context(model,moments):
        first,left=capture(False)
    last,right=capture(True)
    with normalization_context(model,moments,freeze=True):
        _,frozen_left=capture(False)
        _,frozen_right=capture(True)
    raw=torch.from_numpy(actual.view.observation.spectral.copy())
    unchanged=float((raw-swap(raw)).flatten(0,1)[indices].abs().max())
    def difference(a,b):
        return float((a-b)[indices].abs().max())
    q,a=_positive(first["Q"]),_positive(first["A0"])
    reach=_native_reach(q,a,members)
    require(state_hash(model)==state and unchanged==0 and separation>2 and positions,
            "Native source perturbation lost the frozen state or crossed the target pixels")
    return dict(scene=actual.dataset,block=int(tile),assignment=reach,
        changed_q_indices=q!=_positive(last["Q"]),max_q_value_change=float(
            (first["Q"].to_dense()-last["Q"].to_dense()).abs().max()),
        changed_a0_indices=a!=_positive(last["A0"]),max_a0_value_change=float(
            (first["A0"].to_dense()-last["A0"].to_dense()).abs().max()),
        input_target_max_difference=unchanged,min_chebyshev_pixels=separation,
        stem_target_max_difference=difference(first["local_features"].flatten(0,1),last["local_features"].flatten(0,1)),
        cnn_target_max_difference=difference(left[2],right[2]),
        cnn_frozen_groupnorm_target_max_difference=difference(frozen_left[2],frozen_right[2]),
        graph_target_max_difference=difference(left[1],right[1]),
        status="native_non_identifiable" if any(row["reachable"] for row in reach.values())
            or difference(left[2],right[2])>1e-6 else "native_requires_additional_controls")


def pure_boundary(model, scene, layout):
    graph=design.support(layout,"new_path")
    branch=model.backbone.HGNN_Branch
    n=len(design.POSITIONS)
    source_input=torch.randn(n,branch.hgc1.weight.shape[0],generator=torch.Generator().manual_seed(846))
    request=dict(sources=[[4,0],[5,0]],targets=[[2,0,0],[2,1,0]],epsilon=.01,cut_nodes=[4,5])
    results={}
    for phase in ("original","matched"):
        pairs=graph[phase]
        adjacency=row_coo(pairs,[1.]*len(pairs),n)
        plan=SignalPlan.freeze(branch,source_input,adjacency,request=request,dataset=scene)
        rows=review_signals(branch,source_input,adjacency,plan,expected_plan_hash=plan.plan_hash)
        results[phase]=dict(rows=rows["rows"],graph_path_calls=rows["graph_path_calls"])
    return results

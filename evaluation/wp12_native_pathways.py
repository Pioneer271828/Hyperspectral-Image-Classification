"""Bounded signals to original three-head logits along the native two-stage graph."""

from contextlib import ExitStack
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from evaluation.controlled_signal import _distances
from evaluation.frozen_training_stages import HEADS, require
from model.region_pixel_alignment import broadcast_regions, pool_regions
from research_runtime import tensor_hash
from transfer.g3_training import _forward, state_hash

SITES = ("stage1_cnn","stage1_graph","stage2_cnn","stage2_graph","fusion")
COMPONENTS = (*HEADS,"fused_cnn","fused_graph")
GROUPS = ("center","same_region","soft_overlap","one_hop","two_hops","far_or_unreachable")


def graph_for(model):
    b = model.backbone
    require(not model.training and b.graph_mode=="learned" and all(item is None for item in (
        b.region_residual,b.interaction,b.edge_updater,b.graph_router)),
        "This fixed native route requires the original two-stage Q/A without a dynamic edge updater")
    return b


@torch.no_grad()
def capture(model,scene,tile,view):
    b = graph_for(model)
    original = b.forward_stages
    saved = {}

    def stages(cnn_input,region_input,q,a,*args,**kwargs):
        require(not saved,"Expected one native spatial dispatch")
        saved.update(clean=cnn_input.detach(),q=q.detach(),a=a.detach())
        return original(cnn_input,region_input,q,a,*args,**kwargs)

    with patch.object(b,"forward_stages",new=stages):
        result = _forward(model,scene,tile,observation_view=view)
    require(set(saved)=={"clean","q","a"},"Archived model did not expose its actual original Q/A")
    return saved["clean"],saved["q"].coalesce(),saved["a"].coalesce(),tuple(result[:3])


def spatial_forward(model,scene,clean,q,a,*,fixed=True,features=False,split=False):
    b = graph_for(model)
    if not fixed:
        state = b.learnable_graph.build_graph(clean,100,assignment_mode=b.graph_assignment_mode,
                                              adjacency_mode=b.graph_adjacency_mode)
        q,a,_ = state.to_legacy()
        q,a = q.coalesce(),a.coalesce()
    rows = {}
    slots = (("cnn_stage1","stage1_cnn",False),("graph_stage1","stage1_graph",True),
             ("cnn_stage2","stage2_cnn",False),("graph_stage2","stage2_graph",True))

    def observe(handler,name,regional):
        def read(*args,**kwargs):
            value = handler(*args,**kwargs)
            rows[name] = broadcast_regions(q,value) if regional else value
            return value
        return read

    regions = pool_regions(q,clean.reshape(-1,64))
    with ExitStack() as hooks:
        if features:
            for method,name,regional in slots:
                hooks.enter_context(patch.object(b,method,observe(getattr(b,method),name,regional)))
        cnn,graph,_ = b.forward_stages(clean,regions,q,a,scene)
    fused = b.fuse_features(cnn,graph)
    if features:
        rows["fusion"] = fused
        require(set(rows)==set(SITES),"Native model missed a registered spatial stage")
    logits = dict(zip(HEADS,model.predict_features(scene,fused,graph,cnn,return_logits=True),strict=True))
    if split:
        logits["fused_cnn"] = model.heads[scene](b.fuse_features(cnn,graph.detach()))
        logits["fused_graph"] = model.heads[scene](b.fuse_features(cnn.detach(),graph))
    return logits,rows,q,a


class Distances:
    """Minimum directed region distance between positive supports in the actual soft Q."""
    def __init__(self,q,a):
        require(q.shape[1]==a.shape[0]==a.shape[1] and a.shape[0]<=512,
                "Bounded native block matching the two-stage graph required")
        weights = q.detach().to_dense().cpu()
        require(torch.isfinite(weights).all() and (weights>=0).all() and (weights.sum(1)>0).all(),
                "Invalid native Q support")
        self.support = [set(t.tolist()) for t in (row.nonzero().flatten() for row in weights)]
        self.dominant = weights.argmax(1).tolist()
        adjacency = a.detach().to_dense().cpu()
        require(torch.isfinite(adjacency).all() and (adjacency>=0).all(),"Expected nonnegative native adjacency")
        self.hop = [_distances(adjacency,node) for node in range(a.shape[0])]

    def groups(self,cell):
        target = self.support[cell]
        result = {name:[] for name in GROUPS}
        for source,regions in enumerate(self.support):
            if source==cell:
                name = "center"
            elif self.dominant[source]==self.dominant[cell]:
                name = "same_region"
            elif target&regions:
                name = "soft_overlap"
            else:
                paths = [self.hop[i][j] for i in regions for j in target if self.hop[i][j] is not None]
                distance = min(paths,default=3)
                name = {1:"one_hop",2:"two_hops"}.get(distance,"far_or_unreachable")
            result[name].append(source)
        require(sum(map(len,result.values()))==len(self.support),"Pixel groups overlap or omit a source")
        return result


def direction(indices,like,channel):
    require(type(channel) is int and 0<=channel<like.shape[-1],"Bounded channel direction")
    result = torch.zeros_like(like)
    flat = result.reshape(-1,like.shape[-1])
    if indices:
        flat[indices,channel] = 1/np.sqrt(len(indices))
    return result


def topology(values):
    return {k:dict(indices=tensor_hash(x.indices()),values=tensor_hash(x.values()),nnz=x._nnz(),shape=list(x.shape))
            for k,x in values.items()}


def measure(model,scene,clean,q,a,targets,*,reference=None,finite=False,rebuild=False,epsilon=.01,channel=0):
    require(1e-4<=epsilon<=.05 and clean.ndim==3 and clean.shape[-1]==64 and clean.numel()<=512*512*64,
            "Use bounded original clean features and an explicit small perturbation")
    owner = state_hash(model)
    started = perf_counter()
    before = topology(dict(q=q,a=a))
    distances = Distances(q,a)
    leaf = clean.detach().clone().requires_grad_(True)
    rows = []
    with torch.enable_grad():
        logits,features,_,_ = spatial_forward(model,scene,leaf,q,a,features=True,split=True)
        probability_delta = 0.
        if reference is not None:
            probability_delta = max(float((logits[h].softmax(1)-p).detach().abs().max())
                                    for h,p in zip(HEADS,reference,strict=True))
            require(probability_delta<=3e-6,"Explicit original Q/A staging changed the archived native probabilities")
        for cell,y,r in targets:
            require(0<=cell<leaf.shape[0]*leaf.shape[1] and 1<=y<=logits["fused"].shape[1]
                    and 1<=r<=logits["fused"].shape[1] and y!=r,"Invalid released true/rival point")
            score = {h:logits[h][cell,y-1]-logits[h][cell,r-1] for h in COMPONENTS}
            grads = {h:torch.autograd.grad(t,leaf,retain_graph=True)[0].reshape(-1,64) for h,t in score.items()}
            require(all(torch.isfinite(g).all().item() for g in grads.values()) and torch.allclose(
                    grads["fused"],grads["fused_cnn"]+grads["fused_graph"],atol=2e-6,rtol=2e-5),
                    "Native fused-path gradient does not reconstruct")
            groups = {}
            for name,ids in distances.groups(cell).items():
                if not ids:
                    groups[name] = dict(n=0,heads=None)
                    continue
                d = direction(ids,leaf,channel)
                row = {}
                for h,g in grads.items():
                    values = g[ids]
                    row[h] = dict(l1=float(values.abs().sum()),l2=float(values.norm()),
                                  signed_direction=float((g*d.reshape(-1,64)).sum()),
                                  zero_sources=int((values==0).all(1).sum()))
                if finite:
                    for fixed in ((True,False) if rebuild else (True,)):
                        def margin(z):
                            with torch.no_grad():
                                out,_,oq,oa = spatial_forward(model,scene,z,q,a,fixed=fixed)
                            return {h:float(out[h][cell,y-1]-out[h][cell,r-1]) for h in HEADS},topology(dict(q=oq,a=oa))
                        plus,after_plus = margin(clean+epsilon*d)
                        minus,after_minus = margin(clean-epsilon*d)
                        anchor,matched = ({h:float(score[h].detach()) for h in HEADS},before) if fixed else margin(clean)
                        for h in HEADS:
                            row[h]["fixed_fd" if fixed else "rebuilt_fd"] = {
                                "central_difference":(plus[h]-minus[h])/(2*epsilon),
                                "positive":plus[h]-anchor[h],"negative":minus[h]-anchor[h],
                                "anchor":anchor[h],"same_graph_indices":all(
                                    match[k]["indices"]==matched[k]["indices"] for k in ("q","a")
                                    for match in (after_plus,after_minus)),
                                "same_graph_values":all(match[k]["values"]==matched[k]["values"] for k in ("q","a")
                                    for match in (after_plus,after_minus))}
                groups[name] = dict(n=len(ids),heads=row)
            rows.append(dict(y=y,rival=r,selected_cell=cell,heads={
                h:dict(margin=float(score[h].detach()),predicted=int(logits[h][cell].argmax())+1) for h in HEADS},
                fused_gradient_reconstruction=float((grads["fused"]-grads["fused_cnn"]-grads["fused_graph"]).abs().max()),
                groups=groups))
    require(owner==state_hash(model) and before==topology(dict(q=q,a=a)),"Path measurement modified frozen model or topology")
    return dict(rows=rows,features={k:v.detach() for k,v in features.items()},graph=before,
                probability_parity=probability_delta,seconds=perf_counter()-started,
                derivative_scope="fixed_Q_A_at_clean_64_channel_pixel_input",
                rebuilt_scope="finite_response_only" if finite and rebuild else None)

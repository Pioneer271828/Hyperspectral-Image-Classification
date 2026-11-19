"""Coordinate- and support-bound frozen feature intervention and directional response."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import math

import torch

from research_runtime import tensor_hash

HEADS = ("fused","graph","cnn")
STAGES = ("vca","cnn1","cnn2","graph_pre","graph1","graph2","fusion")
GRAPH_SITES = ("graph_pre","graph1","graph2")
EPSILON = .01


@dataclass(frozen=True)
class LayerIdentity:
    model: str
    scene: str
    coordinates: str
    stage: str
    role: str
    support: str | None = None

    def __post_init__(self):
        if self.role not in ("devtrain","devval","synthetic_confirmation"):
            raise ValueError("Frozen localization requires a released or synthetic role")
        if self.stage not in ("sensor",*STAGES,"residual") or any(
                not isinstance(x,str) or not x for x in (self.model,self.scene,self.coordinates)):
            raise ValueError("Explicit model, geometry and feature site required")
        if self.stage in GRAPH_SITES and not self.support:
            raise ValueError("Graph tokens require exact Q/A identity")


@dataclass(frozen=True)
class Feature:
    identity: LayerIdentity
    two_views: str
    tensor: torch.Tensor
    value_hash: str = field(init=False)

    def __post_init__(self):
        if (not isinstance(self.two_views,str) or not self.two_views or self.tensor.layout!=torch.strided
                or self.tensor.ndim<1 or self.tensor.numel()>2_000_000 or not self.tensor.is_floating_point()
                or not torch.isfinite(self.tensor).all()):
            raise ValueError("Bounded finite feature required")
        object.__setattr__(self,"value_hash",tensor_hash(self.tensor))

    def verify(self):
        if tensor_hash(self.tensor)!=self.value_hash:
            raise ValueError("Captured feature bytes changed after identity binding")


def require_exchange(base, donor):
    base.verify()
    donor.verify()
    x,y = base.tensor,donor.tensor
    if (base.identity!=donor.identity or x.shape!=y.shape or x.dtype!=y.dtype or x.device!=y.device):
        raise ValueError("Layer, coordinate, model, shape, type or graph identity differs")
    return x,y


def retained_coordinates(full,full_ids,target,target_ids):
    """Map by authorized original band IDs, preserving target order and dtype."""
    full.verify()
    target.verify()
    if (full.identity!=target.identity or full.identity.stage!="sensor"
            or full.tensor.shape[:-1]!=target.tensor.shape[:-1] or full.tensor.device!=target.tensor.device
            or full.tensor.dtype!=target.tensor.dtype or len(full_ids)!=full.tensor.shape[-1]
            or len(target_ids)!=target.tensor.shape[-1] or len(set(full_ids))!=len(full_ids)
            or len(set(target_ids))!=len(target_ids) or not set(target_ids)<=set(full_ids)):
        raise ValueError("No legal original-band projection between these sensor coordinates")
    projected = full.tensor[..., [full_ids.index(b) for b in target_ids]]
    return Feature(target.identity,full.two_views,projected)


def probability_state(outputs, pixels, truth, competitors):
    if len(outputs)<3 or not (len(pixels)==len(truth)==len(competitors) and 0<len(pixels)<=16):
        raise ValueError("One to sixteen aligned released class comparisons required")
    scores = {}
    for name,value in zip(HEADS,outputs[:3],strict=True):
        ids = torch.as_tensor(pixels,dtype=torch.long,device=value.device)
        targets = torch.as_tensor(truth,dtype=torch.long,device=value.device)-1
        other = torch.as_tensor(competitors,dtype=torch.long,device=value.device)-1
        if (value.ndim!=2 or any(not (0<=i<value.shape[0]) for i in pixels)
                or any(not 1<=i<=value.shape[1] for i in (*truth,*competitors))
                or (targets==other).any()):
            raise ValueError("Class or pixel identity escapes the frozen probabilities")
        out = value[ids]
        if not torch.isfinite(out).all() or (out<0).any() or not torch.allclose(
                out.sum(dim=1),torch.ones_like(out[:,0]),atol=2e-6,rtol=0):
            raise ValueError("Nonfinite frozen probabilities or wrong head output")
        row = torch.arange(len(ids),device=value.device)
        margin = out[row,targets]-out[row,other]
        rank = 1+(out>out[row,targets].unsqueeze(1)).sum(dim=1)
        scores[name] = dict(predicted=(out.argmax(dim=1)+1).tolist(),rank=rank.tolist(),
                            margin=margin,probabilities=out)
    return scores


def serial(values):
    return {k:{key:(field.detach().cpu().tolist() if torch.is_tensor(field) else field)
               for key,field in row.items()} for k,row in values.items()}


def local_response(predict, base, donor, *, pixels, truth, competitors, epsilon=EPSILON):
    """A gradient and centered difference on one frozen graph and a single site.

    Supply a registered predictor that holds graph, routing and other inputs fixed.
    The endpoint at the donor is a separate finite intervention.
    """
    if not isinstance(epsilon,(float,int)) or not math.isfinite(epsilon) or not 1e-5<=epsilon<=.1:
        raise ValueError("Finite prespecified directional step required")
    x,y = require_exchange(base,donor)
    direction = (y-x).detach()
    z = x.detach().requires_grad_(True)
    with torch.enable_grad():
        zero = probability_state(predict(z),pixels,truth,competitors)
        derivative = {}
        all_gradients = [(name,val) for name,row in zero.items() for val in row["margin"]]
        for name,val in all_gradients:
            g = torch.autograd.grad(val,z,allow_unused=True,retain_graph=True)[0] if val.requires_grad else None
            derivative.setdefault(name,[]).append(None if g is None else float((g*direction).sum()))
    with torch.no_grad():
        plus = probability_state(predict(x+epsilon*direction),pixels,truth,competitors)
        minus = probability_state(predict(x-epsilon*direction),pixels,truth,competitors)
        moved = probability_state(predict(y),pixels,truth,competitors)
    reported = {}
    for name in HEADS:
        centered = ((plus[name]["margin"]-minus[name]["margin"])/(2*epsilon)).tolist()
        reported[name] = dict(directional=derivative[name],finite_difference=centered,
                              absolute_error=[None if a is None else abs(a-b)
                                              for a,b in zip(derivative[name],centered,strict=True)])
    return dict(base=serial(zero),plus=serial(plus),minus=serial(minus),donor=serial(moved),slope=reported,
                direction_sha256=tensor_hash(direction),direction_l2=float(direction.double().norm()),
                donor_source=donor.two_views,target_source=base.two_views,epsilon=epsilon,actual_forwards=4)


@contextmanager
def _method(module, name, wrapper):
    if name in module.__dict__:
        raise ValueError("Cannot nest mutable localization hooks")
    original = getattr(module,name)
    setattr(module,name,wrapper(original))
    try:
        yield
    finally:
        delattr(module,name)


@contextmanager
def taps(model,dataset,*,replace_at=None,replacement=None):
    """Expose a fixed set of original model tensors and remove hooks on failure."""
    if model.training or (replace_at is None)!=(replacement is None) or (
            replace_at is not None and replace_at not in STAGES):
        raise ValueError("Layer tracing applies only to explicitly frozen model evaluation")
    owner = model.backbone
    values = {}
    def put(name,tensor):
        if name in values or not torch.is_tensor(tensor):
            raise ValueError("Expected one dense tensor at the declared model site")
        values[name]=tensor
        return replacement if name==replace_at else tensor
    def adapter(_module,_args,output):
        if isinstance(output,tuple):
            return put("vca",output[0]),*output[1:]
        return put("vca",output)
    def readout(_module,args):
        return put("fusion",args[0]),*args[1:]
    def cnn(stage):
        def wrap(invoke):
            def inner(*args,**kwargs):
                return put(stage,invoke(*args,**kwargs))
            return inner
        return wrap
    def graph_first(invoke):
        def inner(value,*args,**kwargs):
            return put("graph1",invoke(put("graph_pre",value),*args,**kwargs))
        return inner
    with ExitStack() as stack:
        stack.callback(owner.adapter.register_forward_hook(adapter).remove)
        stack.callback(model.heads[dataset].register_forward_pre_hook(readout).remove)
        for name,func in (("cnn_stage1",cnn("cnn1")),("cnn_stage2",cnn("cnn2")),
                          ("graph_stage1",graph_first),("graph_stage2",cnn("graph2"))):
            stack.enter_context(_method(owner,name,func))
        yield values
        if set(values)!=set(STAGES):
            raise ValueError("The native forward did not traverse all registered layer sites")

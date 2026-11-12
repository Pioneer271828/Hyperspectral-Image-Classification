"""Bounded program-generated two-hop signal recovery through the original SharedHGCN."""

from collections import Counter
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from torch import nn

from evaluation.controlled_signal import _distances
from model.VCASAHGCN import SharedHGCN
from model.regional_support_evolution import Schedule, SupportEvolution, development_context
from model.research_config import digest
from research_runtime import tensor_hash


SEEDS = (64, 512, 2048)
TOPOLOGIES = ("chain", "bridge", "bottleneck")
SIGNAL = ("local", "random", "reachable", "cut")
EVOLUTION = ("local", "fixed_budget", "tighten", "regenerate", "random_regenerate", "fixed_final")
STREAMS = {"training": 24601, "validation": 38411, "confirmation": 82793}
VALIDATION_STEPS = {"signal": tuple(range(10, 201, 10)), "evolution": tuple(range(10, 121, 10))}
BATCH, EVALUATION = 4, 64
WIDTH = 12
LEARNING_RATE = .0125


def topology(name):
    if name not in TOPOLOGIES:
        raise ValueError("Unknown controlled topology")
    n, source, target = 12, 0, 11
    if name == "chain":
        links = [(i+1,i) for i in range(n-1)]
        cut = (6,5)
    elif name == "bridge":
        links = [(i,j) for lo in (0,6) for i in range(lo,lo+6) for j in range(lo,lo+6) if i!=j]
        links.extend(((6,5),(5,6)))
        cut = (6,5)
    else:
        links = [(i,0) for i in (1,2,3)] + [(4,i) for i in (1,2,3)] + [(5,4)]
        links += [(i,5) for i in (6,7,8,9,10)] + [(11,i) for i in (6,7,8,9,10)]
        cut = (5,4)
    local = sorted(set(links)|{(i,i) for i in range(n)})
    # Match the remote control in added edge count, receiver and local directed hop distance when feasible.
    # In these constructions the source is the only sender at the positive edge's distance for some topologies.
    return dict(name=name,nodes=n,source=source,target=target,cut=list(cut),local=[list(v) for v in local])


def adjacency(pairs, n):
    ix=torch.tensor(pairs,dtype=torch.long).t().contiguous()
    weights=1./torch.bincount(ix[0],minlength=n).to(torch.float32)[ix[0]]
    return torch.sparse_coo_tensor(ix,weights,(n,n)).coalesce()


def support(top, mode, seed):
    base=set(map(tuple,top["local"]))
    src,tgt=top["source"],top["target"]
    if mode=="cut":
        base.remove(tuple(top["cut"]))
    if mode=="reachable":
        base.add((tgt,src))
    if mode=="random":
        candidates=[j for j in range(top["nodes"]) if (tgt,j) not in base]
        choice=int(np.random.default_rng(seed).choice(candidates))
        base.add((tgt,choice))
    return [list(v) for v in sorted(base)]


def geometry(top,pairs):
    n,src,tgt=top["nodes"],top["source"],top["target"]
    original=adjacency(top["local"],n).to_dense()
    dense=adjacency(pairs,n).to_dense()
    added=set(map(tuple,pairs))-set(map(tuple,top["local"]))
    return dict(distance=_distances(dense,src)[tgt],nnz=len(pairs),
                receive=[int((dense[i]>0).sum()) for i in range(n)],
                send=[int((dense[:,i]>0).sum()) for i in range(n)],
                added=[dict(edge=[i,j],original_hops=_distances(original,j)[i]) for i,j in sorted(added)])


def streams(seed):
    # A counter-indexed PCG64 stream separates split identity from optimizer order.
    rng=lambda role:np.random.default_rng(np.random.SeedSequence([seed, STREAMS[role]]))
    return {role:rng(role).uniform(-1,1,size=(200,BATCH) if role=="training" else EVALUATION).astype(np.float32)
            for role in ("training","validation")}


def confirmation(seed):
    return np.random.default_rng(np.random.SeedSequence([seed,STREAMS["confirmation"]])).uniform(-1,1,size=EVALUATION).astype(np.float32)


def model(seed, score=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        graph=SharedHGCN(3,WIDTH,1,dropout=0.).eval()
        # Running mean/variance stay in eval even while affine weights learn.
        scoring=nn.Linear(4,4,bias=False) if score else None
    return graph,scoring


def descriptors(n, seed):
    p=np.arange(n,dtype=np.float32)
    spectral=torch.tensor(np.stack([np.cos(p*.4),np.sin(p*.4),p/n,(p%3)/3],axis=1))
    rng=np.random.default_rng(seed+70221)
    graph=torch.from_numpy(rng.normal(size=(n,4)).astype(np.float32))
    return spectral,graph


def _inputs(top, signals):
    x=torch.zeros((len(signals)*top["nodes"],3))
    positions=torch.arange(len(signals))*top["nodes"]
    x[positions+top["source"],0]=torch.as_tensor(signals)
    x[positions+top["source"],1]=1
    x[positions+top["target"],2]=1
    return x,positions+top["target"]


def predict(graph, a, top, signals, *, permutation=None):
    x, target=_inputs(top,signals)
    n=top["nodes"]
    if permutation is not None:
        idx=torch.as_tensor(permutation,dtype=torch.long)
        x=x.reshape(-1,n,3)[:,idx].reshape(-1,3)
        target=torch.arange(len(signals))*n+int((idx==top["target"]).nonzero().item())
        a=a.to_dense()[idx][:,idx].to_sparse().coalesce()
    offsets=torch.arange(len(signals))*n
    ij=(a.indices().unsqueeze(0)+offsets[:,None,None]).permute(1,0,2).reshape(2,-1)
    block=torch.sparse_coo_tensor(ij,a.values().repeat(len(signals)),(len(x),len(x)),is_coalesced=True)
    first=graph.forward_stage1(x,block)
    return graph.forward_stage2(first,block)[target,0]


def score_readout(values, predictions):
    a=np.asarray(values,dtype=np.float64)
    b=np.asarray(predictions,dtype=np.float64)
    error=b-a
    return dict(mse=float(np.mean(error**2)),mae=float(np.mean(np.abs(error))),
                sign_accuracy=float(np.mean((b>=0)==(a>=0))),constant_mse=float(np.mean(a*a)))


def evolution_settings(mode):
    if mode not in EVOLUTION:
        raise ValueError("Unknown engineering comparison")
    if mode=="local":
        return Schedule(8,8,119,1,0,120),"local"
    if mode=="fixed_budget":
        return Schedule(4,4,7,16,0,120),"joint"
    if mode=="fixed_final":
        return Schedule(8,3,7,16,16,120),"joint"
    return Schedule(8,3,7,16,0 if mode=="tighten" else 16,120),("random" if mode=="random_regenerate" else "joint")


def evolver(top,mode,seed):
    n=top["nodes"]
    spectral,features=descriptors(n,seed)
    policy,search=evolution_settings(mode)
    # A self-loop and the lightest-index original nonself neighbor per receiver are protected.
    mandatory=[]
    for i in range(n):
        neighbors=sorted(j for r,j in top["local"] if r==i and i!=j)
        if neighbors:
            mandatory.append([i,neighbors[0]])
    return SupportEvolution(adjacency(top["local"],n),spectral,features,
            context=development_context(role="synthetic",block=top["name"],node_ids=list(range(n)),partitions=[0]*n),
            mandatory=mandatory,schedule=policy,mode=search,seed=seed)


def _hash(model):
    return digest({k:tensor_hash(v) for k,v in model.state_dict().items()})


class Trajectory:
    def __init__(self, case, *, fixed=None):
        self.case=deepcopy(case)
        self.seed,self.mode=case["seed"],case["mode"]
        self.kind=case["kind"]
        self.top=topology(case["topology"])
        self.budget=200 if self.kind=="signal" else 120
        self.graph,self.scorer=model(self.seed,self.kind=="evolution")
        self.parameters=dict(self.graph.named_parameters())
        if self.scorer is not None:
            self.parameters.update({"score."+k:v for k,v in self.scorer.named_parameters()})
        self.optimizer=torch.optim.Adam(list(self.parameters.values()),lr=LEARNING_RATE)
        self.streams=streams(self.seed)
        self.initial=_hash(self.graph)
        self.fixed=None if fixed is None else deepcopy(fixed)
        self.router=None
        if self.kind=="signal":
            self.static=support(self.top,self.mode,self.seed)
            self.matrix=adjacency(self.static,self.top["nodes"])
        else:
            self.router=evolver(self.top,self.mode,self.seed)
            if (self.fixed is None)!=(self.mode!="fixed_final"):
                raise ValueError("Fixed-final requires the matching sealed evolution support")
            self.static=self.fixed if self.fixed is not None else None
        self.history=[]
        self.graph_events=[]
        self.selected=None
        self.selected_state=None
        self.timing=dict(render=0.,aggregation=0.,backward=0.,optimizer=0.,validation=0.)

    def graph_state(self):
        return dict(graph=deepcopy(self.graph.state_dict()),score=None if self.scorer is None else deepcopy(self.scorer.state_dict()))

    def restore_weights(self,state):
        self.graph.load_state_dict(state["graph"])
        if self.scorer is not None:
            self.scorer.load_state_dict(state["score"])

    def matrix_at(self,step, *, advance=False, pairs=None):
        if self.router is None:
            return self.matrix
        if advance and self.fixed is None:
            self.router.advance(step)
        descriptors=self.scorer(self.router.features)
        return self.router.render(scores=descriptors,support=self.fixed if pairs is None else pairs)

    def support_at(self):
        if self.static is not None:
            return deepcopy(self.static)
        return [list(p) for p in self.router.selected]

    def step(self):
        number=len(self.history)+1
        if number>self.budget:
            raise ValueError("Complete trajectory has zero further updates")
        before={k:v.detach().clone() for k,v in self.parameters.items()}
        self.optimizer.zero_grad(set_to_none=True)
        t=perf_counter()
        a=self.matrix_at(number,advance=True)
        self.timing["render"]+=perf_counter()-t
        signal=self.streams["training"][number-1]
        t=perf_counter()
        prediction=predict(self.graph,a,self.top,signal)
        loss=(prediction-torch.from_numpy(signal)).square().mean()
        self.timing["aggregation"]+=perf_counter()-t
        t=perf_counter()
        loss.backward()
        self.timing["backward"]+=perf_counter()-t
        grad={k:float(p.grad.abs().sum()) if p.grad is not None else 0. for k,p in self.parameters.items()}
        if not torch.isfinite(loss) or any(not np.isfinite(v) for v in grad.values()):
            raise ValueError("Nonfinite controlled update")
        t=perf_counter()
        self.optimizer.step()
        self.timing["optimizer"]+=perf_counter()-t
        changes={k:int((v.detach()!=before[k]).sum()) for k,v in self.parameters.items()}
        pairs=self.support_at()
        support_hash=digest(pairs)
        router_receipt=None if self.router is None or self.fixed is not None else self.router.receipt()
        refresh=None if router_receipt is None else router_receipt["refresh_step"]
        row=dict(step=number,loss=float(loss.detach()),gradient=grad,changes=changes,nnz=a._nnz(),
                 support_sha256=support_hash,refresh=refresh,sample_sha256=tensor_hash(signal),
                 budget=None if self.router is None else (max(Counter(r for r,_ in pairs).values()) if self.fixed is not None
                         else self.router.schedule.budget(number)),validation=None)
        old=self.graph_events[-1] if self.graph_events else None
        if old is None or old["support_sha256"]!=support_hash or old["refresh"]!=refresh or old["budget"]!=row["budget"]:
            self.graph_events.append(dict(step=number,support=pairs,support_sha256=support_hash,refresh=refresh,budget=row["budget"],
                                          pool=None if router_receipt is None else router_receipt["pool"]))
        if number in VALIDATION_STEPS[self.kind]:
            t=perf_counter()
            with torch.no_grad():
                val=predict(self.graph,self.matrix_at(number),self.top,self.streams["validation"]).tolist()
            row["validation"]=score_readout(self.streams["validation"],val)
            self.timing["validation"]+=perf_counter()-t
            if self.selected is None or row["validation"]["mse"]<self.selected["quality"]["mse"]:
                self.selected=dict(step=number,quality=row["validation"],support=pairs,predictions=val)
                self.selected_state=self.graph_state()
        self.history.append(row)

    def packet(self):
        return dict(case=self.case,initial=self.initial,history=self.history,graph_events=self.graph_events,
                    weights=self.graph_state(),optimizer=self.optimizer.state_dict(),torch_rng=torch.get_rng_state().clone(),
                    selected=self.selected,selected_state=self.selected_state,fixed=self.fixed,
                    route=None if self.router is None or self.fixed is not None else self.router.receipt())

    def restore(self,value):
        if self.history or value["case"]!=self.case or value["initial"]!=self.initial or value["fixed"]!=self.fixed:
            raise ValueError("Cross-case or changed initial/support checkpoint")
        rows=value["history"]
        if len(rows)>self.budget or any(v["step"]!=i or v["sample_sha256"]!=tensor_hash(self.streams["training"][i-1])
                                       for i,v in enumerate(rows,1)):
            raise ValueError("Invalid checkpoint update cursor or sample flow")
        if self.router is not None and self.fixed is None:
            self.router.restore(value["route"])
            if self.router.step!=len(rows):
                raise ValueError("Restored graph cursor and optimizer differ")
        self.restore_weights(value["weights"])
        self.optimizer.load_state_dict(value["optimizer"])
        torch.set_rng_state(value["torch_rng"])
        self.history=deepcopy(rows)
        self.graph_events=deepcopy(value["graph_events"])
        self.selected=deepcopy(value["selected"])
        self.selected_state=deepcopy(value["selected_state"])

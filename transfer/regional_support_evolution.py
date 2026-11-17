"""Explicit bounded directed support evolution for regional CPU tensors."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from model.budgeted_graph_router import BudgetedGraphRouter
from model.research_config import digest
from research_runtime import tensor_hash


MODES = ("local", "random", "spectral", "graph", "joint")
PHASES = ("devtrain", "devval", "synthetic")


@dataclass(frozen=True)
class Schedule:
    start_budget: int
    end_budget: int
    warmup: int
    tighten_every: int
    refresh_every: int
    updates: int
    pool: int = 8

    def __post_init__(self):
        if (any(type(v) is not int for v in asdict(self).values()) or not 1 <= self.end_budget <= self.start_budget <= 16
                or not 0 <= self.warmup < self.updates <= 200 or not 1 <= self.tighten_every <= self.updates
                or not 0 <= self.refresh_every <= self.updates or not 1 <= self.pool <= 32):
            raise ValueError("Invalid bounded support schedule")

    def budget(self, step):
        return max(self.end_budget, self.start_budget-max(0, step-self.warmup)//self.tighten_every)

    def refresh(self, step):
        return (step == self.warmup+1 or (step > self.warmup and self.refresh_every > 0
                and (step-self.warmup-1)%self.refresh_every == 0))


def _features(value, n):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.ndim != 2 or value.shape[0] != n
            or not 1 <= value.shape[1] <= 32 or not torch.isfinite(value).all()):
        raise ValueError("Finite bounded CPU FP32 feature tensor required")
    return value


def development_context(*, role, block, node_ids, partitions):
    if (role not in PHASES or not isinstance(block, str) or not block.strip() or len(block) > 128
            or len(node_ids) != len(partitions) or not 1 <= len(node_ids) <= 256
            or any(type(i) is not int or i < 0 for i in (*node_ids,*partitions)) or len(set(node_ids)) != len(node_ids)):
        raise ValueError("Bounded synthetic or released development block context required")
    return dict(role=role,block=block,node_ids=list(node_ids),partitions=list(partitions))


def _keys(graph):
    return list(map(tuple, graph.indices().t().tolist()))


def _tensor_pairs(pairs,n):
    if (not isinstance(pairs,list) or any(not isinstance(x,(tuple,list)) or len(x)!=2
            or any(type(i) is not int or not 0<=i<n for i in x) for x in pairs)
            or [tuple(x) for x in pairs]!=sorted(set(tuple(x) for x in pairs))):
        raise ValueError("Unique ordered support within node range required")
    return [tuple(x) for x in pairs]


def _cosine(features, rows, cols):
    values=F.normalize(features,dim=-1)
    return (values[rows]*values[cols]).sum(-1)


def _graph(adjacency, partitions):
    if not torch.is_tensor(adjacency) or adjacency.device.type!="cpu" or adjacency.dtype!=torch.float32:
        raise ValueError("CPU FP32 adjacency required")
    n=len(partitions)
    if adjacency.shape!=(n,n):
        raise ValueError("Adjacency and bounded node context differ")
    if adjacency.is_sparse and adjacency._nnz()>n*n:
        raise ValueError("Unbounded input COO")
    graph,_,_=BudgetedGraphRouter._graph(adjacency)
    rows,cols=graph.indices()
    if any(partitions[i]!=partitions[j] for i,j in zip(rows.tolist(),cols.tolist(),strict=True)):
        raise ValueError("Cross-partition original adjacency")
    if adjacency.is_sparse:
        raw=[tuple(v) for v in adjacency._indices().t().tolist()]
        if len(raw)!=len(set(raw)):
            raise ValueError("Duplicate input COO support")
    return graph


class SupportEvolution:
    """Discrete selection uses frozen descriptors; selected values remain differentiable.

    The caller advances an integer update cursor. A separate render reads its
    support, including during validation. RNG state is derived from the frozen
    seed and exact deterministic refresh history and checked on restoration.
    """
    VERSION=1

    def __init__(self, adjacency, spectra, graph_features, *, context, mandatory, schedule, mode, seed):
        if (mode not in MODES or type(seed) is not int or not 0<=seed<2**63
                or not isinstance(schedule,Schedule) or not isinstance(context,dict)
                or set(context)!={"role","block","node_ids","partitions"}):
            raise ValueError("Explicit bounded support contract required")
        self.context=development_context(**context)
        self.n=len(context["node_ids"])
        self.base=_graph(adjacency,context["partitions"])
        self.local=set(_keys(self.base))
        self.spectra=_features(spectra,self.n).detach().clone()
        self.features=_features(graph_features,self.n).detach().clone()
        self.mandatory=set(_tensor_pairs(mandatory,self.n)) | {(i,i) for i in range(self.n)}
        if (not self.mandatory<=self.local or max(sum(i==r for i,_ in self.mandatory) for r in range(self.n))>schedule.end_budget):
            raise ValueError("Mandatory local support conflicts with final budget")
        self.schedule,self.mode,self.seed=schedule,mode,seed
        self.contract=digest(dict(version=self.VERSION,context=self.context,graph=tensor_hash(self.base),
            spectra=tensor_hash(self.spectra),features=tensor_hash(self.features),mandatory=sorted(self.mandatory),
            schedule=asdict(schedule),mode=mode,seed=seed))
        self._rng=np.random.default_rng(seed)
        self.step=0
        self.pool=[]
        self.selected=sorted(self.local)
        self.refresh_step=0
        self.cost=dict(retrieval_seconds=0.,selection_seconds=0.,scoring_seconds=0.,sparse_build_seconds=0.)
        self.last_stats=dict(candidate_pairs=0,score_pairs=0)

    def _retrieve(self):
        ids=self.context["node_ids"]
        groups=self.context["partitions"]
        spectra=F.normalize(self.spectra,dim=-1)
        features=F.normalize(self.features,dim=-1)
        # The temporary block is bounded to 16 rows. All-pairs work stays quadratic.
        pairs=[]
        tested=0
        if self.mode!="local":
            order=sorted(range(self.n),key=ids.__getitem__)
            for start in range(0,self.n,16):
                stop=min(start+16,self.n)
                block=order[start:stop]
                cos=(spectra[block]@spectra.T) if self.mode in ("spectral","joint") else None
                graph=(features[block]@features.T) if self.mode in ("graph","joint") else None
                for offset,i in enumerate(block):
                    eligible=sorted((j for j in range(self.n) if (i,j) not in self.local and groups[i]==groups[j]),key=ids.__getitem__)
                    tested+=len(eligible)
                    jitter={j:float(self._rng.random()) for j in eligible}
                    def score(j):
                        a=0. if cos is None else float(cos[offset,j])
                        b=0. if graph is None else float(graph[offset,j])
                        return jitter[j] if self.mode=="random" else a+b+.15*jitter[j]
                    best=sorted(eligible,key=lambda j:(-score(j),ids[j]))[:self.schedule.pool]
                    pairs.extend((i,j) for j in best)
        self.last_stats["candidate_pairs"]=tested
        return sorted(pairs)

    def _select(self, budget, pool):
        ids=self.context["node_ids"]
        # A refresh reserves one optional slot for a remote candidate when possible.
        # Local rows fill remaining capacity in stable original-ID order.
        prior={p:i for i,p in enumerate(_keys(self.base))}
        weights=self.base.values().detach().tolist()
        selected=set(self.mandatory)
        for i in sorted(range(self.n),key=ids.__getitem__):
            local=sorted((p for p in self.local-self.mandatory if p[0]==i),
                         key=lambda p:(-weights[prior[p]],ids[p[1]]))
            remote=[p for p in pool if p[0]==i]
            room=budget-sum(p[0]==i for p in self.mandatory)
            if room and remote:
                remote.sort(key=lambda p:(ids[p[1]]))
                r=torch.tensor([p[0] for p in remote])
                c=torch.tensor([p[1] for p in remote])
                if self.mode=="random":
                    remote=[remote[j] for j in self._rng.permutation(len(remote)).tolist()]
                else:
                    scores=torch.zeros(len(remote))
                    if self.mode in ("spectral","joint"):
                        scores+=_cosine(self.spectra,r,c)
                    if self.mode in ("graph","joint"):
                        scores+=_cosine(self.features,r,c)
                    # Refresh can alter the pool even with static descriptors.
                    remote=[remote[j] for j in sorted(range(len(remote)),key=lambda j:(-float(scores[j]),ids[remote[j][1]]))]
                selected.add(remote[0])
                room-=1
            selected.update((local+remote[1:])[:room])
        return sorted(selected)

    def advance(self, step):
        if type(step) is not int or not self.step<=step<=min(self.step+1,self.schedule.updates):
            raise ValueError("Support update cursor must advance by at most one")
        if step==self.step:
            return
        if self.context["role"]=="devval":
            raise ValueError("Development validation cannot advance graph state")
        previous=self.schedule.budget(self.step)
        budget=self.schedule.budget(step)
        refresh=self.schedule.refresh(step)
        if refresh:
            start=perf_counter()
            self.pool=self._retrieve()
            self.cost["retrieval_seconds"]+=perf_counter()-start
            self.refresh_step=step
        if step==1 or refresh or previous!=budget:
            start=perf_counter()
            self.selected=self._select(budget,self.pool)
            self.cost["selection_seconds"]+=perf_counter()-start
        self.step=step

    def receipt(self):
        state=dict(version=self.VERSION,contract=self.contract,step=self.step,budget=self.schedule.budget(self.step),
            phase="warmup" if self.step<=self.schedule.warmup else "evolution",
            refresh_step=self.refresh_step,pool=[list(p) for p in self.pool],
            selected=[list(p) for p in self.selected],rng=deepcopy(self._rng.bit_generator.state))
        return dict(**state,state_sha256=digest(state))

    def restore(self, receipt):
        if (not isinstance(receipt,dict) or receipt.get("contract")!=self.contract or self.step!=0
                or type(receipt.get("step")) is not int or not 0<=receipt["step"]<=self.schedule.updates):
            raise ValueError("Support resume contract, features or cursor differs")
        # Validate a detached replay before mutating the live instance.
        shadow=SupportEvolution(self.base,self.spectra,self.features,context=self.context,
             mandatory=[list(p) for p in sorted(self.mandatory)],schedule=self.schedule,mode=self.mode,seed=self.seed)
        for step in range(1,receipt["step"]+1):
            shadow.advance(step)
        if shadow.receipt()!=receipt:
            raise ValueError("Support, budget, RNG or refresh history differs")
        self.step,self.pool,self.selected=shadow.step,shadow.pool,shadow.selected
        self.refresh_step,self._rng=shadow.refresh_step,shadow._rng

    def render(self, *, scores=None, support=None):
        pairs=self.selected if support is None else _tensor_pairs(support,self.n)
        groups=self.context["partitions"]
        if (not self.mandatory<=set(pairs) or any(groups[i]!=groups[j] for i,j in pairs)
                or max(sum(r==i for r,_ in pairs) for i in range(self.n))>max(self.schedule.start_budget,
                                                                             self.schedule.end_budget)):
            raise ValueError("Rendered fixed support violates local or partition contract")
        start=perf_counter()
        x=self.features if scores is None else _features(scores,self.n)
        ij=torch.tensor(pairs,dtype=torch.int64).t().contiguous()
        row,col=ij
        vals=(_cosine(x,row,col)*.5).exp()
        # Full mix at selected coordinates only. Other entries have no value/gradient.
        prior=self.base.to_dense()[row,col]
        vals=0.5*prior+0.5*vals
        self.cost["scoring_seconds"]+=perf_counter()-start
        start=perf_counter()
        totals=vals.new_zeros(self.n).scatter_add(0,row,vals)
        out=torch.sparse_coo_tensor(ij,vals/totals[row],(self.n,self.n)).coalesce()
        if out._nnz()!=len(pairs) or (out.values()<=0).any():
            raise ValueError("Support lost a selected positive edge")
        self.cost["sparse_build_seconds"]+=perf_counter()-start
        self.last_stats["score_pairs"]=len(pairs)
        return out

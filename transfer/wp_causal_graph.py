"""Synthetic source isolation on the existing two-stage regional classifier."""

from time import perf_counter

import numpy as np
import torch
from torch import nn

from evaluation.frozen_training_stages import require
from model.graph_state import GraphState
from model.research_config import digest
from research_runtime import tensor_hash
from transfer import wp_causal_graph_design as design, wp_region_graph as regional
from transfer.band_graph_control import FixedNativeGraph, observation_id, row_coo
from transfer.g3_training import state_hash
from transfer.regional_support_evolution import development_context

INTERVENTIONS = ("source_swap","unrelated_swap","cut")


def make_state(layout, first):
    membership = torch.from_numpy(design.memberships(layout).reshape(-1).copy())
    n = len(design.POSITIONS)
    index = torch.stack((torch.arange(len(membership)),membership))
    q = torch.sparse_coo_tensor(index,torch.ones(len(membership)),(len(membership),n)).coalesce()
    a0 = row_coo(first,[1.]*len(first),n)
    xy = torch.tensor(design.lattice(layout),dtype=torch.float32)/(design.SIZE-1)
    return GraphState(q,a0,xy,(design.SIZE,design.SIZE),(1,n)).validate()


def source_swap(x, membership, positions, nodes):
    if not positions or not nodes:
        return x
    copy = x.clone()
    flat = copy.flatten(0,1)
    rows = [torch.where(membership==node)[0] for node in nodes]
    cols = torch.as_tensor(positions,device=x.device,dtype=torch.long)
    flat[rows[0][:,None],cols] = x.flatten(0,1)[rows[1][:,None],cols]
    flat[rows[1][:,None],cols] = x.flatten(0,1)[rows[0][:,None],cols]
    return copy


class SealedInput:
    def __init__(self, layout, actual, cue, intervention, regime):
        self.membership = torch.from_numpy(design.memberships(layout).reshape(-1).copy())
        self.positions = [i for i,b in enumerate(actual.view.bands.band_ids) if b in cue]
        self.intervention,self.regime = intervention,regime
        self.signal = None

    def sanitize(self, x):
        members = self.membership.to(x.device)
        nodes = (design.emitters(self.regime) if self.intervention=="source_swap" else
                 design.UNRELATED if self.intervention=="unrelated_swap" else ())
        changed = source_swap(x,members,self.positions,nodes)
        self.signal = x.new_zeros(len(design.POSITIONS))
        if not self.positions:
            return changed
        result = changed.clone()
        result[:,:,self.positions] = .5
        values = changed.flatten(0,1)[:,self.positions]
        for node in (*design.TARGETS,*design.SOURCES,*design.UNRELATED):
            self.signal[node] = 12*(values[members==node].mean()-.5)
        return result

    def inject(self, features, q):
        require(self.signal is not None, "Protected spectral input was not separated before local normalization")
        shape = features.shape
        require(shape[0]==len(design.POSITIONS) and isinstance(q,torch.Tensor),
                "Injected signal escaped its fixed region assignment")
        direction = features.new_zeros(shape[1])
        direction[0] = 1
        return features + self.signal[:,None]*direction


class CausalProgram:
    """Same learned value rule on sealed hard-Q witnesses, without a data-driven support search."""

    def __init__(self, anchor, policy, seed, budget, proof):
        require(policy in design.POLICIES and type(seed) is int and 8<=budget<=200
                and isinstance(anchor.backbone.HGNN_Branch.batch_normalzation2.norm,nn.LayerNorm),
                "Causal route requires row-local graph normalization and registered policy")
        self.anchor,self.policy,self.seed,self.budget,self.proof = anchor,policy,seed,budget,proof
        self.source_state = state_hash(anchor)
        self.views = {}

    def choose(self, actual, producer, role, step, *, intervention=None):
        rec=actual.verify(producer,role)
        tile=rec["source"]["tile"]
        key=str(tile)
        roster=self.proof["tiles"]
        require(rec["source"]["bank"]==self.proof["bank"] and actual.dataset==self.proof["scene"]
                and key in roster and roster[key]["role"]==role
                and actual.lattice.shape==(design.SIZE,design.SIZE) and rec["factor"]==1
                and (intervention is None or role=="devval" and intervention in INTERVENTIONS),
                "Causal fixed graph escapes its signed observation, role or allowed diagnostic")
        regime=roster[key]["regime"]
        start=perf_counter()
        observation=rec["observation"]
        cold=observation not in self.views
        if cold:
            self.views[observation] = self._prepare(actual, regime)
        entry,graphs,state,selector = self.views[observation]
        constructed=perf_counter()-start if cold else 0.
        pick=perf_counter()
        phase=design.phase(self.policy,role,step,self.budget)
        if intervention=="cut":
            phase="original"
        pairs=graphs[phase]
        first=graphs["first"]
        points=list(map(list,design.lattice(self.proof["layout"])))
        a,b=(design.invariants(edges,points) for edges in (first,pairs))
        proof=dict(schema=regional.VERSION,policy=self.policy,orientation="receiver_sender",
            snapshot=digest([self.proof["witness"],phase]),cursor=step if role=="devtrain" else 0,
            refresh_step=0,candidate=digest([first,pairs]),candidate_count=len(pairs),
            points=points,distance_denominator_squared=1,original=[list(p) for p in first],
            selected=[list(p) for p in pairs],original_counts=a,counts=b,
            newly_added=len(set(pairs)-set(first)),same_sender_degree=a["send"]==b["send"],
            same_distance_counts=a["squared_distances"]==b["squared_distances"],strict_match=None,
            causal=dict(layout=self.proof["layout"],regime=regime,phase=phase,intervention=intervention,
                        witness=digest(self.proof["witness"])))
        times=dict(anchor_graph_seconds=constructed,selection_seconds=perf_counter()-pick,
                   strict_search_seconds=0.,retrieval_seconds=0.)
        bound=regional.BoundSupport(entry,pairs,selector,proof,times)
        bound.override=FixedNativeGraph(observation_id(actual),state)
        sealed=SealedInput(self.proof["layout"],actual,self.proof["cue_bands"],intervention,regime)
        bound.bindings=dict(input_transform=sealed.sanitize,region_transform=sealed.inject)
        return bound

    def _prepare(self, actual, regime):
        graphs=design.support(self.proof["layout"],regime)
        state=make_state(self.proof["layout"],graphs["first"])
        tile=actual.receipt["source"]["tile"]
        identity=dict(scene=actual.dataset,grid=list(state.grid_shape),bands=actual.view.bands.fingerprint,
                      source=self.source_state,assignment=tensor_hash(state.Q),
                      block_input=actual.receipt["observation"],admission=self.proof["bank"])
        n=state.region_count
        context=development_context(role="synthetic",block=str(tile),
            node_ids=[65536*tile+i for i in range(n)],partitions=[tile]*n)
        # Discrete topology is a checked witness. These fixed placeholders are
        # never ranked. The shared scorer uses live region and CNN projections.
        descriptor=torch.zeros(n,1,dtype=torch.float32)
        entry=dict(context=context,identity=identity,adjacency=state.A0,spectra=descriptor,
                   graph_features=descriptor,mandatory=[])
        limit=max(design.invariants(graphs["first"],design.lattice(self.proof["layout"]))["receive"])
        settings=dict(start_budget=limit,end_budget=limit,warmup=1,tighten_every=self.budget,
                      refresh_every=0,updates=self.budget,pool=1)
        selector=regional.make_selector(entry,dict(mode="local",schedule=settings),self.seed)
        return entry,graphs,state,selector


def verify_trace(proof, contract, tile, role, budget):
    causal=proof["causal"]
    declared=contract["tiles"][str(tile)]
    phase=design.phase(proof["policy"],role,proof["cursor"],budget)
    if causal["intervention"]=="cut":
        phase="original"
    layout=contract["layout"]
    graph=design.support(layout,declared["regime"])
    fixed_q=np.c_[np.arange(design.SIZE**2),design.memberships(layout).reshape(-1)].tolist()
    require(causal==dict(layout=layout,regime=declared["regime"],phase=phase,intervention=causal["intervention"],
                        witness=digest(contract["witness"]))
            and causal["intervention"] in (None,*INTERVENTIONS)
            and proof["assignment_support"]==fixed_q and proof["original"]==[list(e) for e in graph["first"]]
            and proof["selected"]==[list(e) for e in graph[phase]]
            and proof["same_sender_degree"] and proof["same_distance_counts"],
            "Actual staged propagation differs from the preregistered exact graph witness")
    return causal

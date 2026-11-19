"""Bind the existing native stage-two regional support to released-point observations."""

from collections import Counter
from contextlib import contextmanager
from time import perf_counter

import torch

from evaluation.frozen_training_stages import require
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.controlled_regional_support import geometry, invariants, match_support
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.native_support import NativeSupportModel, _active, make_selector
from transfer.regional_support_evolution import development_context

BACKEND = "region_support"
POLICIES = ("original", "random_fixed", "spectral_fixed", "spectral_bounded", "strict_matched")
SCALE = 16
VERSION = "released_native_stage_two_support_v1"


class RegionSupportModel(NativeSupportModel):
    """Use the archived tensors and stage functions with the device-bound WP05 forward."""

    def forward(self, dataset_name, x, *args, **kwargs):
        frame = _active.get()
        require(frame is not None and frame["backbone"] is self.backbone and x.dtype == torch.float32
                and x.device == next(self.parameters()).device and not args
                and all(kwargs.get(key) is None for key in ("Q", "A", "graph_state")),
                "Native stage-two support needs one bound FP32 observation")
        if frame["capture"]:
            spectral, _ = self.backbone._split_spectral_extra(x, kwargs.get("wavelengths"))
            frame["spectral"] = spectral.flatten(0, 1)
        return MultiDatasetVCASAHGCN.forward(self, dataset_name, x, **kwargs)


@contextmanager
def bind(model, selector=None):
    require(_active.get() is None, "Regional support frame already active")
    frame = dict(backbone=model.backbone, selector=selector, capture=selector is None, detail=True, timing={})
    token = _active.set(frame)
    try:
        yield frame
    finally:
        _active.reset(token)


def _sum_cost(value):
    return {key: sum(times) for key,times in value.items()}


def _cpu_capture(anchor, actual, producer, role):
    height,width = actual.lattice.shape
    grid = anchor.backbone.learnable_graph._grid_shape(height,width,SCALE)
    require(height*width<=4096 and 4<=grid[0]*grid[1]<=256,
            "Regional fixture exceeds the registered pixel or region bound")
    start = perf_counter()
    with torch.no_grad(), bind(anchor) as frame:
        _, trace = forward(anchor,actual,producer,role,graph_scale=SCALE)
    rec = actual.receipt
    gh,gw = trace["grid"]
    tile = rec["source"]["tile"]
    ids = [tile*65536+y*256+x for y in range(gh) for x in range(gw)]
    require(len(ids) <= 256 and len(ids) >= 4, "Regional support requires a bounded nontrivial native lattice")
    # This version is admitted only for engineering releases. Advancing a
    # selector replays precommitted unlabeled candidates, never calibration labels.
    context = development_context(role="synthetic",block=str(tile),node_ids=ids,partitions=[tile]*len(ids))
    identity = dict(scene=actual.dataset,grid=trace["grid"],bands=actual.view.bands.fingerprint,
                    source=state_hash(anchor),assignment=tensor_hash(frame["Q"]),
                    block_input=rec["observation"],admission=rec["source"]["bank"])
    entry = dict(context=context,identity=identity,adjacency=frame["A0"],
                 spectra=frame["spectra"].contiguous(),graph_features=frame["descriptor"],mandatory=[])
    return entry,perf_counter()-start


def _edges(value):
    return list(map(tuple,value.coalesce().indices().t().tolist()))


def _replace_one_per_row(base, proposed):
    """Consume the existing selector's remote entries within the fixed original row counts."""
    original = _edges(base)
    weights = dict(zip(original,base.coalesce().values().tolist(),strict=True))
    rows = {}
    for row,col in original:
        rows.setdefault(row,[]).append(col)
    remote = {}
    for row,col in proposed:
        if (row,col) not in weights:
            remote.setdefault(row,[]).append(col)
    result = set(original)
    for row,cols in remote.items():
        replaceable = [col for col in rows[row] if col!=row]
        if replaceable:
            remove = min(replaceable,key=lambda col:(weights[row,col],col))
            result.remove((row,remove))
            result.add((row,cols[0]))
    return sorted(result)


class GraphProgram:
    """Unlabeled frozen controller inputs and deterministic per-view sparse supports."""

    def __init__(self, anchor, policy, seed, budget):
        require(policy in POLICIES and type(seed) is int and 0<=seed<2**32 and 8<=budget<=200,
                "Explicit bounded regional support schedule required")
        self.anchor,self.policy,self.seed,self.budget = anchor,policy,seed,budget
        self.views = {}

    def choose(self, actual, producer, role, step):
        observation = actual.receipt["observation"]
        new = observation not in self.views
        if new:
            self.views[observation] = (*_cpu_capture(self.anchor,actual,producer,role), {})
        entry,build,cache = self.views[observation]
        # Evaluation uses the same declared terminal context for fixed and
        # bounded arms. Only registered optimization events vary the cursor.
        tick = step if self.policy=="spectral_bounded" and role=="devtrain" else self.budget
        boundary = (0 if tick <= 1 else 2 + (tick-2)//4*4)
        if self.policy in ("original", "strict_matched"):
            boundary = self.budget
        if boundary not in cache:
            cache[boundary] = self._select(entry,actual.receipt["box"],boundary)
            search = cache[boundary][-1]
        else:
            search = {k:0. for k in cache[boundary][-1]}
        pairs,selector,proof,_ = cache[boundary]
        timings = dict(anchor_graph_seconds=build if new else 0.,**search)
        return BoundSupport(entry,pairs,selector,proof,timings)

    def matching_probe(self, actual, producer, role):
        require(self.policy=="strict_matched","Only the strict policy can preflight exact feasibility")
        entry,_ = _cpu_capture(self.anchor,actual,producer,role)
        _,_,proof,timings = self._select(entry,actual.receipt["box"],self.budget,require_match=False)
        return dict(view=actual.receipt["observation"],match=proof["strict_match"],timing=timings,
                    nnz=proof["original_counts"]["nnz"])

    def _select(self, entry, box, tick, *, require_match=True):
        started = perf_counter()
        original = _edges(entry["adjacency"])
        degree = Counter(row for row,_ in original)
        cap = max(degree.values())
        mode = "random" if self.policy=="random_fixed" else "spectral"
        schedule = dict(start_budget=cap,end_budget=cap,warmup=1,tighten_every=self.budget,
                        refresh_every=4,updates=self.budget,pool=5)
        selector = make_selector(entry,dict(mode=mode,schedule=schedule),self.seed)
        if self.policy!="original":
            selector.advance(tick)
        pairs = original if self.policy=="original" else _replace_one_per_row(entry["adjacency"],selector.selected)
        points, denominator = geometry(entry, box)
        match = None
        match_seconds = 0.
        if self.policy=="strict_matched":
            allowed = sorted(set(original) | set(selector.pool) | set(pairs))
            match = match_support(pairs,allowed,entry["mandatory"],points,entry["context"]["partitions"],
                                  entry["context"]["node_ids"],seed=self.seed,limit=4096)
            match_seconds = match.pop("seconds")
            if match["status"]=="MATCHED":
                pairs = list(map(tuple,match["pairs"]))
            else:
                require(not require_match,"Nontrivial exact degree/distance support unavailable")
        fixed = invariants(original,points)
        chosen = invariants(pairs,points)
        require(fixed["nnz"]==chosen["nnz"] and fixed["receive"]==chosen["receive"],
                "Stage-two edge count or receiver degree differs from the registered original")
        snap = selector.snapshot()
        proof = dict(schema=VERSION,policy=self.policy,orientation="receiver_sender",
            snapshot=snap["state_sha256"],cursor=snap["step"],refresh_step=snap["refresh_step"],
            candidate=digest(snap["pool"]),candidate_count=len(snap["pool"]),points=points,
            distance_denominator_squared=denominator**2,original=[list(x) for x in original],
            selected=[list(x) for x in pairs],original_counts=fixed,counts=chosen,
            newly_added=len(set(pairs)-set(original)),same_sender_degree=fixed["send"]==chosen["send"],
            same_distance_counts=fixed["squared_distances"]==chosen["squared_distances"],strict_match=match)
        times = dict(selection_seconds=perf_counter()-started,strict_search_seconds=match_seconds,
                     retrieval_seconds=selector.cost["retrieval_seconds"])
        return pairs,selector,proof,times


class BoundSupport:
    """Render the original learned mixture on the admitted sparse support."""

    def __init__(self, entry, pairs, selector, proof, times):
        self.n,self.entry,self.selector = selector.n,entry,selector
        self.pairs,self.proof,self.times = pairs,proof,times

    def snapshot(self):
        # NativeSupportBackbone only reports this value. The computed support
        # and verified original degree constraints are carried separately.
        return self.proof["snapshot"]

    def render_live(self, adjacency, features):
        start = perf_counter()
        require(_edges(adjacency) == _edges(self.entry["adjacency"]),
                "Learned first-stage graph no longer has the frozen original ordered support")
        old = dict(self.selector.cost)
        # The archived scorer is CPU FP32. Transfers are differentiable and
        # their full cost is included in stage-two render timing.
        result = self.selector._render(adjacency.cpu(),features.cpu(),self.pairs,max(self.proof["counts"]["receive"]))
        result = result.to(features.device)
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        self.times.update(render_seconds=perf_counter()-start,
            scoring_seconds=self.selector.cost["scoring_seconds"]-old["scoring_seconds"],
            sparse_build_seconds=self.selector.cost["sparse_build_seconds"]-old["sparse_build_seconds"])
        return result


def forward_region(model, actual, producer, role, execution, program, *, step=0):
    selected = program.choose(actual,producer,role,step)
    timing = {}
    with bind(model,selected) as frame:
        out, trace = forward(model,actual,producer,role,execution=execution,timed=timing,graph_scale=SCALE)
    actual_support = frame["stage2"].coalesce()
    require(_edges(actual_support)==selected.pairs and bool((actual_support.values()>0).all()),
            "Native propagation changed the selected support or lost positive edges")
    receipt = {**selected.proof, "view":actual.receipt["observation"],
               "first_indices":trace["a0_indices"],"stage2_indices":tensor_hash(actual_support.indices()),
               "positive_nnz":actual_support._nnz(),
               "stage2_values":tensor_hash(actual_support)}
    receipt["assignment_support"] = frame["Q"].indices().T.cpu().tolist()
    receipt["assignment_indices"] = trace["q_indices"]
    # Wall timings never enter the exact recovery journal or graph match.
    timing["regional"] = {**selected.times,**_sum_cost(frame["timing"])}
    trace["regional_support"] = receipt
    return out, trace, timing


def verify_trace(proof, observation, *, require_policy=None):
    require(proof["schema"]==VERSION and proof["orientation"]=="receiver_sender" and proof["view"]==observation
            and (require_policy is None or require_policy==proof["policy"]),
            "Frozen stage-two evidence changes the retained observation or policy")
    original,selected = ([tuple(e) for e in proof[k]] for k in ("original","selected"))
    points = proof["points"]
    n = len(points)
    require(all(s==sorted(set(s)) and all(0<=i<n and 0<=j<n for i,j in s)
                and all((i,i) in s for i in range(n)) for s in (original,selected)),
            "Stage-two evidence changes the bounded directed graph")
    assignments = [tuple(p) for p in proof["assignment_support"]]
    require(assignments==sorted(set(assignments)) and all(i>=0 and 0<=j<n for i,j in assignments)
            and proof["assignment_indices"]==tensor_hash(torch.tensor(assignments,dtype=torch.long).T.contiguous()),
            "Actual Q support differs from the held regional coordinates")
    a,b = (invariants(s,points) for s in (original,selected))
    require((a,b)==(proof["original_counts"],proof["counts"]) and a["receive"]==b["receive"]
            and proof["first_indices"]==tensor_hash(torch.tensor(original,dtype=torch.long).T.contiguous())
            and b["nnz"]==proof["positive_nnz"] and a["nnz"]==b["nnz"]
            and proof["stage2_indices"]==tensor_hash(torch.tensor(selected,dtype=torch.long).T.contiguous())
            and proof["newly_added"]==len(set(selected)-set(original))
            and proof["same_sender_degree"]==(a["send"]==b["send"])
            and proof["same_distance_counts"]==(a["squared_distances"]==b["squared_distances"]),
            "Recounted actual stage-two graph support differs")
    if proof["policy"]=="strict_matched":
        match = proof["strict_match"]
        require(match["status"]=="MATCHED" and match["constraints"]==b
                and match["support_sha256"]==digest(selected) and match["changed_edges"]>0,
                "Strict alternate no longer has exact sender, receiver and distance matching")
    return proof


def signal_visibility(signal, graph, band_ids, target_cells, truth):
    """Recount planted source reachability through the actual A0 then A2 path."""
    if signal is None:
        return None
    region_count = len(graph["points"])
    assignment = {}
    for cell,region in graph["assignment_support"]:
        require(0<=region<region_count,"Pixel assignment leaves the registered region graph")
        assignment.setdefault(cell,set()).add(region)
    first,second = ({i:set() for i in range(region_count)} for _ in range(2))
    for a,b in graph["original"]:
        first[a].add(b)
    for a,b in graph["selected"]:
        second[a].add(b)
    visible = bool(set(signal["cue_bands"]) & set(band_ids))
    classes = {}
    for label,cells in signal["sources"].items():
        sources = {r for cell in cells for r in assignment[cell]} if visible else set()
        targets = [assignment[cell] for cell,y in zip(target_cells,truth,strict=True) if y==int(label)]
        reached = sum(any(first[intermediate]&sources for destination in regions
                          for intermediate in second[destination]) for regions in targets)
        classes[label] = dict(points=len(targets),reachable=reached,visible_source_regions=len(sources))
    return dict(case=signal["case"],cue_present=visible and any(signal["sources"].values()),classes=classes)

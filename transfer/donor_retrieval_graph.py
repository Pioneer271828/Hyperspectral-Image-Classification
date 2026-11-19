"""Observable synthetic donor graphs and independent exact deletion controls."""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from time import perf_counter

import numpy as np
import torch

from evaluation.graph_edge_intervention import delete_support
from evaluation.matched_edge_controls import freeze, verify
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.band_graph_control import row_coo
from transfer.regional_support_evolution import Schedule, SupportEvolution, development_context

SEEDS = (64, 512, 2048)
LAYOUTS = ("horizontal", "offset")
DIRECTIONS = ("directed", "reciprocal")
ROLES = {"training": (811, 8), "validation": (947, 6), "confirmation": (1193, 8)}
ORDERS = tuple(permutations(range(4)))
LOOPS = tuple((i, i) for i in range(8))
BINS = [4, 12, 36, 96]
RANDOM_SEED = 7719
UPDATES = 120
PREFIX = 7
BATCH = 4
ARMS = ("local", "oracle", "random", "spectral", "learned", "fixed_final", "fixed_cut")


@dataclass(frozen=True)
class Truth:
    """Sealed evaluator view. Ordinary routing receives no instance of this class."""
    donor_order: tuple
    positive: int
    null_positive: int


def links(targets, donors, order, reciprocal):
    pairs = {(targets[i], donors[j]) for i, j in enumerate(order)}
    if reciprocal:
        pairs |= {(j, i) for i, j in pairs}
    return pairs


class DonorGraph:
    def __init__(self, *, targets, donors, xy, keys, context, reciprocal, seed):
        self.targets, self.donors = targets, donors
        self.xy, self.keys, self.context, self.reciprocal = xy, keys, context, reciprocal
        self.features = (keys[targets, None] * keys[donors][None, :]).reshape(4, 4, 2, 4).sum(-1)
        started = perf_counter()
        base = row_coo(LOOPS, [1.] * len(LOOPS), 8)
        eligible = sorted((int(i), int(j)) for i in targets for j in donors)
        self.support = SupportEvolution(
            base, keys, keys[:, :4], context=context, mandatory=[], eligible=eligible,
            schedule=Schedule(2, 2, 0, 1, 0, UPDATES, 4), mode="spectral", seed=seed)
        self.support.advance(1)
        if set(self.support.pool) != set(eligible) or len(self.support.pool) != 16:
            raise ValueError("Frozen candidate search omitted a registered endpoint")
        self.order_tensor = torch.tensor(ORDERS)
        self.option_features = self.features[torch.arange(4), self.order_tensor].sum(dim=1)
        norm = torch.nn.functional.normalize(keys, dim=1)
        cosine = norm[targets] @ norm[donors].T
        self.spectral_choice = int(cosine[torch.arange(4), self.order_tensor].sum(dim=1).argmax())
        self.local_pairs = tuple((int(i), int(j)) for group in (targets, donors)
                                 for i, j in zip(group, group[[1, 0, 3, 2]], strict=True))
        self.id = digest(dict(geometry=tensor_hash(xy), support=self.support.receipt()["state_sha256"],
                              targets=targets.tolist(), donors=donors.tolist(), reciprocal=reciprocal))
        self.cost = {"graph_preparation_seconds": perf_counter()-started,
                     "candidate_search_seconds": self.support.cost["retrieval_seconds"],
                     "candidate_selection_seconds": self.support.cost["selection_seconds"]}
        self.cache = {}
        self.message_scores = torch.ones(8,1)

    def pairs(self, choice):
        if choice == "local":
            pairs = self.local_pairs if self.reciprocal else tuple(p for p in self.local_pairs if p[0] in self.targets)
        elif choice == "cut":
            pairs = ()
        elif isinstance(choice, int) and 0 <= choice < len(ORDERS):
            pairs = links(self.targets.tolist(), self.donors.tolist(), ORDERS[choice], self.reciprocal)
        else:
            raise ValueError("Unknown bounded support choice")
        return sorted(set(LOOPS) | set(pairs))

    def adjacency(self, choice):
        if choice not in self.cache:
            start = perf_counter()
            sparse = self.support.render(scores=self.message_scores,support=self.pairs(choice))
            self.cost["sparse_render_seconds"] = self.cost.get("sparse_render_seconds", 0.) + perf_counter()-start
            self.cache[choice] = sparse
        return self.cache[choice]


def sample_graph(seed, layout, direction, role, index, draw=0):
    """Build an independent graph and draw payload from distinct per-role streams."""
    split, size = ROLES[role]
    if not 0 <= index < size:
        raise ValueError("Graph outside registered split")
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, index]))
    node_order = rng.permutation(8)
    target, donor = node_order[:4], node_order[4:]
    true_order = tuple(int(i) for i in rng.permutation(4))
    distractor = rng.permutation(4)
    latent = torch.eye(4)[rng.permutation(4)]
    nuisance = torch.eye(4)[rng.permutation(4)]
    keys = torch.zeros(8, 8)
    keys[target, :4] = latent
    keys[donor[list(true_order)], :4] = latent
    keys[target, 4:] = 2*nuisance
    keys[donor[distractor], 4:] = 2*nuisance
    xy = torch.zeros(8, 2)
    center = np.asarray([[0., i*2.] for i in range(4)] + [[48., i*2.] for i in range(4)])
    if layout == "offset":
        center[4:, 1] += 18.
        center = center[:, ::-1].copy()
    if rng.integers(2):
        center[:, 0] *= -1
    center += rng.uniform(-.2, .2, size=(8, 2))
    xy[node_order] = torch.tensor(center, dtype=torch.float32)
    stable_ids = rng.choice(10_000, size=8, replace=False).tolist()
    context = development_context(role="synthetic", block=f"{layout}/{direction}/{role}/{index}",
                                  node_ids=stable_ids, partitions=[0]*8)
    graph = DonorGraph(targets=target, donors=donor, xy=xy, keys=keys, context=context,
                       reciprocal=direction=="reciprocal", seed=seed)
    return graph, true_order


def observe(graph, true_order, *, seed, role, index, draw):
    """Release observed donor payload and target labels; retain pairing for evaluator."""
    split, _ = ROLES[role]
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, index, draw, 2731]))
    positive, null_positive = (int(i) for i in rng.integers(4, size=2))
    x = torch.zeros(8, 3)
    x[graph.targets, 1] = 1.
    x[:, 2] = 1.
    x[graph.donors, 0] = -1.
    x[graph.donors[true_order[positive]], 0] = 1.
    labels = torch.zeros(4, dtype=torch.long)
    labels[positive] = 1
    return x, labels, Truth(true_order, positive, null_positive)


def _component_counts(adjacency, targets, donors, known):
    rows, columns = adjacency.indices().tolist()
    edges = set(zip(rows, columns, strict=True))
    reached = []
    for i, j in zip(targets, known, strict=True):
        connected = {i}
        frontier = {i}
        while frontier:
            more = {b for a, b in edges if a in frontier} - connected
            connected.update(more)
            frontier = more
        reached.append(donors[j] in connected)
    return dict(nnz=adjacency._nnz(), true_direct=sum((i, donors[j]) in edges for i, j in zip(targets, known, strict=True)),
                truth_reachable=sum(reached), incoming=torch.bincount(adjacency.indices()[0], minlength=8).tolist(),
                outgoing=torch.bincount(adjacency.indices()[1], minlength=8).tolist())


def exact_deletions(graph, true_order):
    """Independently enumerate the whole bounded matching family before the MILP."""
    targets, donors = graph.targets.tolist(), graph.donors.tolist()
    permitted = {(i, j) for i in targets for j in donors}
    if graph.reciprocal:
        permitted |= {(j, i) for i, j in permitted}
    known = links(targets, donors, true_order, graph.reciprocal)
    # Exactly one weak complete pairing. All other weights stay in the middle.
    weak_order = (*true_order[1:], true_order[0])
    weak = links(targets, donors, weak_order, graph.reciprocal)
    edges = sorted(permitted | set(LOOPS))
    raw = [5. if p in known else .15 if p in weak else 1. for p in edges]
    adjacency = row_coo(edges, raw, 8)
    def edge_keys(pairs):
        return tuple(sorted(8*i+j for i,j in pairs))
    def signature(pairs):
        return Counter(int(np.searchsorted(BINS, torch.linalg.vector_norm(graph.xy[i]-graph.xy[j]).item(), side="right"))
                       for i,j in pairs)
    frequencies = signature(known)
    feasible = {edge_keys(links(targets, donors, order, graph.reciprocal))
                for order in ORDERS if signature(links(targets, donors, order, graph.reciprocal)) == frequencies}
    if len(feasible)!=24:
        raise ValueError("Independent enumeration must exhibit 24 exact masks")
    plan = freeze(adjacency, graph.xy, per_row=1, distance_bins=BINS, seeds=[RANDOM_SEED],
                  symmetric=graph.reciprocal, separate_weak=True)
    verify(plan, adjacency, graph.xy, expected_hash=plan["plan_hash"])
    expected = edge_keys(known)
    if tuple(plan["baseline"]["removed"]) != expected:
        raise ValueError("Important deletion did not remove the planted donor pairing")
    names = ("low_weight", f"random:{RANDOM_SEED}", f"conditional_random:{RANDOM_SEED}")
    items = [plan["controls"][name] for name in names]
    masks = [tuple(item["removed"]) for item in items if item["status"]=="optimal"]
    if len(masks)!=3 or not set(masks) <= feasible-{expected} or masks[0]!=edge_keys(weak) or masks[0]==masks[-1]:
        raise ValueError("Matched masks conflict with independent enumeration")
    arms = {"none": [], "important": list(expected), **{name: list(mask) for name,mask in zip(names,masks,strict=True)}}
    graphs = {name: delete_support(adjacency, mask) for name,mask in arms.items()}
    audit = {name: _component_counts(value, targets, donors, true_order) for name,value in graphs.items()}
    return graphs, dict(feasible=len(feasible), independent_alternatives=len(feasible)-2,
                        natural_equals_weak=masks[0]==masks[1], weak_differs_conditioned=masks[0]!=masks[-1],
                        masks=arms, matched_support=audit, plan=plan)

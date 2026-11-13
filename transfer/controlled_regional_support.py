"""Bounded, directed interventions over a frozen regional candidate pool."""

from collections import Counter
from itertools import combinations
from math import comb
from random import Random
from time import perf_counter

from model.research_config import digest
from transfer.regional_support_evolution import _tensor_pairs


MAX_TRIALS = 16384
EXACT_SUBSETS = 4096
MAX_SWAPS = 16
ARMS = ("reference", "matched", "old_random")


def _distance(points, edge):
    a, b = (points[index] for index in edge)
    return sum((x - y)**2 for x, y in zip(a, b, strict=True))


def invariants(pairs, points):
    n = len(points)
    receive, send = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    distances = Counter(str(_distance(points, edge)) for edge in pairs)
    return dict(nnz=len(pairs), receive=[receive[i] for i in range(n)], send=[send[i] for i in range(n)],
                squared_distances=dict(distances))


def geometry(entry, box):
    """Exact integer embedding of native regular-grid centers in pixel coordinates."""
    gh, gw = entry["identity"]["grid"]
    ids = entry["context"]["node_ids"]
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    block = int(entry["context"]["block"])
    indices = [divmod(node - 65536 * block, 256) for node in ids]
    if (sorted(indices) != [(r, c) for r in range(gh) for c in range(gw)] or min(h, w) <= 0):
        raise ValueError("Original region IDs do not match admitted grid")
    denominator = 2 * gh * gw
    points = [[y0 * denominator + (2 * r + 1) * h * gw,
               x0 * denominator + (2 * c + 1) * w * gh] for r, c in indices]
    return points, denominator


def _domain(reference, allowed, mandatory, points, partitions, node_ids):
    n = len(points)
    if (not 1 <= n <= 256 or len(partitions) != n or len(node_ids) != n or len(set(node_ids)) != n
            or any(type(i) is not int or i < 0 for i in node_ids)
            or any(type(i) is not int or i < 0 for i in partitions)
            or any(len(point) != 2 or any(type(x) is not int for x in point) for point in points)):
        raise ValueError("Bounded stable node IDs, partitions and integer coordinates required")
    original = set(_tensor_pairs(reference, n))
    pool = set(_tensor_pairs(allowed, n))
    required = set(_tensor_pairs(mandatory, n)) | {(i, i) for i in range(n)}
    if not required <= original <= pool or any(partitions[i] != partitions[j] for i, j in pool):
        raise ValueError("Support must retain protected edges and stay inside the same-partition candidate pool")
    return original, pool, required


def _ordered(edges, ids):
    return sorted(edges, key=lambda pair: (ids[pair[0]], ids[pair[1]]))


def match_support(reference, allowed, mandatory, points, partitions, node_ids, *, seed, limit=MAX_TRIALS):
    """Search exact switches or exhaustively enumerate a small remaining domain."""
    started = perf_counter()
    if type(seed) is not int or not 0 <= seed < 2**32 or type(limit) is not int or not 1 <= limit <= MAX_TRIALS:
        raise ValueError("Bounded uint32-seeded search required")
    original, pool, required = _domain(reference, allowed, mandatory, points, partitions, node_ids)
    target = invariants(original, points)
    original_set = original.copy()
    free = _ordered(pool - required, node_ids)
    rng = Random(seed)
    count = len(original - required)
    space = comb(len(free), count)
    tested, accepted, switches = 0, 0, []
    domains_checked = 0
    exact = space <= min(EXACT_SUBSETS, limit)
    method = "exhaustive_optional_edge_subsets" if exact else "bounded_directed_two_edge_switch"
    distance = {edge: _distance(points, edge) for edge in pool}
    if exact:
        for subset in combinations(free, count):
            tested += 1
            alternative = required | set(subset)
            if alternative != original_set and invariants(alternative, points) == target:
                domains_checked += 1
                if rng.randrange(domains_checked) == 0:
                    original = alternative
        accepted = domains_checked
    else:
        for _ in range(limit):
            free_active = _ordered(original - required, node_ids)
            if len(free_active) < 2:
                break
            left, right = rng.sample(free_active, 2)
            tested += 1
            swapped = ((left[0], right[1]), (right[0], left[1]))
            if (swapped[0] == swapped[1] or any(pair in original or pair not in pool for pair in swapped)
                    or sorted(map(distance.__getitem__, swapped)) != sorted((distance[left], distance[right]))):
                continue
            alternative = (original - {left, right}) | set(swapped)
            # Retain each accepted change. The search produces a seeded
            # perturbation, without a claim of uniform sampling.
            if len(original_set - alternative) <= len(original_set - original):
                continue
            original = alternative
            switches.append(dict(remove=[list(left), list(right)], add=[list(pair) for pair in swapped]))
            accepted += 1
            if accepted == MAX_SWAPS:
                break
    changed = len(original_set - original)
    if invariants(original, points) != target:
        raise RuntimeError("Search changed a protected degree or distance invariant")
    status = "MATCHED" if changed else "INFEASIBLE" if exact else "NOT_FOUND_WITHIN_BUDGET"
    return dict(status=status, method=method, seed=seed, limit=limit, possible_subsets=str(space),
                exhaustive=exact, proposals=tested, accepted=accepted, swaps=switches,
                changed_edges=changed, changed_fraction=changed / len(original_set),
                reference_sha256=digest(sorted(original_set)), support_sha256=digest(sorted(original)),
                candidate_sha256=digest(sorted(pool)), constraints=target,
                pairs=[list(pair) for pair in sorted(original)], seconds=perf_counter() - started)

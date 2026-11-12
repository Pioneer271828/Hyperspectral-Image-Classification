"""NumPy support oracle independent of the bounded switch constructor."""

from itertools import combinations
from math import comb

import numpy as np


def population(pairs, coordinates):
    n = len(coordinates)
    edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    points = np.asarray(coordinates, dtype=np.int64)
    delta = points[edges[:, 0]] - points[edges[:, 1]]
    lengths, counts = np.unique(np.einsum("ij,ij->i", delta, delta), return_counts=True)
    return dict(nnz=len(edges), receive=np.bincount(edges[:, 0], minlength=n).tolist(),
                send=np.bincount(edges[:, 1], minlength=n).tolist(),
                squared_distances={str(k): int(v) for k, v in zip(lengths, counts, strict=True)})


def audit(receipt):
    result = receipt["match"]
    original = set(map(tuple, receipt["reference"]["selected"]))
    allowed = set(map(tuple, receipt["allowed"]))
    proposed = set(map(tuple, result["pairs"]))
    points = receipt["points"]
    if len(proposed) != len(result["pairs"]) or not proposed <= allowed or population(proposed, points) != population(original, points):
        raise ValueError("Independent degree, distance or pool check failed")
    if result["constraints"] != population(original, points):
        raise ValueError("Recorded invariants differ")
    must = {tuple(pair) for pair in receipt["mandatory"]} if "mandatory" in receipt else {(i, i) for i in range(len(points))}
    partitions = receipt.get("partitions", [0] * len(points))
    if any(partitions[a] != partitions[b] for a, b in allowed):
        raise ValueError("Independent candidate partition check failed")
    if not must <= proposed:
        raise ValueError("Protected edges missing")
    if len(original - proposed) != result["changed_edges"] or (proposed != original) != (result["status"] == "MATCHED"):
        raise ValueError("Claimed support perturbation differs")
    if result["exhaustive"]:
        free = sorted(allowed - must)
        count = len(original - must)
        expected = comb(len(free), count)
        feasible = sum(candidate != original and population(candidate, points) == result["constraints"]
                       for subset in combinations(free, count) for candidate in (must | set(subset),))
        if result["proposals"] != expected or result["accepted"] != feasible or bool(feasible) != (proposed != original):
            raise ValueError("Exhaustive support claim differs")
        return expected
    active = original.copy()
    for row in result["swaps"]:
        removed, added = set(map(tuple, row["remove"])), set(map(tuple, row["add"]))
        if (len(removed) != 2 or len(added) != 2 or not removed <= active - must
                or not added <= allowed - active or population(removed, points) != population(added, points)):
            raise ValueError("Invalid directed switch witness")
        active = (active - removed) | added
    if active != proposed or len(result["swaps"]) != result["accepted"] or result["proposals"] > result["limit"]:
        raise ValueError("Incomplete bounded switch witness")
    return 0

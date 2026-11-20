"""NumPy reference for the registered candidate search and sparse aggregation."""

from collections import Counter
from copy import deepcopy

import numpy as np


def _normalize(array):
    value = np.asarray(array, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), np.float32(1e-12))


def support_trace(adjacency, spectra, features, *, node_ids, partitions, mandatory, schedule, mode, seed, steps,
                  version=2):
    """Read raw arrays and reconstruct PCG64 draws, candidate sets, and supports."""
    matrix = np.asarray(adjacency)
    count = len(node_ids)
    order = sorted(range(count), key=node_ids.__getitem__)
    if len(set(node_ids)) != count or matrix.shape != (count, count) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Invalid independent graph input")
    local = {(i, j) for i in order for j in order if matrix[i, j] > 0 or i == j}
    if any(partitions[i] != partitions[j] for i, j in local):
        raise ValueError("Independent graph crosses partitions")
    required = {tuple(pair) for pair in mandatory} | {(i, i) for i in order}
    if (not required <= local or max(Counter(i for i, _ in required).values()) > schedule["end_budget"]):
        raise ValueError("Independent mandatory budget conflict")
    rng = np.random.Generator(np.random.PCG64(seed))
    spectrum = _normalize(spectra)
    graph = _normalize(features)
    pool = []
    refresh_at = 0
    previous = None
    selected = sorted(local)
    result = []
    for step in range(steps + 1):
        budget = max(schedule["end_budget"], schedule["start_budget"] -
                     max(0, step - schedule["warmup"]) // schedule["tighten_every"])
        since = step - schedule["warmup"] - 1
        every = schedule["refresh_every"]
        refreshed = since == 0 or since > 0 and every > 0 and since % every == 0
        if refreshed:
            refresh_at = step
            pool = []
            if mode != "local":
                for i in order:
                    eligible = [j for j in order if (i, j) not in local and partitions[i] == partitions[j]]
                    draws = {j: rng.random() for j in eligible}
                    scores = {}
                    for j in eligible:
                        if mode == "random":
                            scores[j] = draws[j]
                        else:
                            spectral = float(np.dot(spectrum[i], spectrum[j])) if mode in ("spectral", "joint") else 0.
                            structure = float(np.dot(graph[i], graph[j])) if mode in ("graph", "joint") else 0.
                            scores[j] = spectral + structure + .15 * draws[j]
                    pool.extend((i, j) for j in sorted(eligible, key=lambda j: (-scores[j], node_ids[j]))[:schedule["pool"]])
            pool.sort()
        if (step == 0 and version == 2 or step > 0 and (step == 1 or refreshed or previous != budget)):
            selected = set(required)
            for i in order:
                near = [j for r, j in local - required if r == i]
                near.sort(key=lambda j: (-(matrix[i, j] if matrix[i, j] > 0 else 1.), node_ids[j]))
                distant = [j for r, j in pool if r == i]
                distant.sort(key=node_ids.__getitem__)
                capacity = budget - sum(r == i for r, _ in required)
                if distant and capacity:
                    if mode == "random":
                        distant = [distant[k] for k in rng.permutation(len(distant))]
                    else:
                        scores = {}
                        for j in distant:
                            a = (spectrum[i] * spectrum[j]).sum(dtype=np.float32) if mode in ("spectral", "joint") else np.float32(0.)
                            b = (graph[i] * graph[j]).sum(dtype=np.float32) if mode in ("graph", "joint") else np.float32(0.)
                            scores[j] = a + b
                        distant.sort(key=lambda j: (-scores[j], node_ids[j]))
                    selected.add((i, distant.pop(0)))
                    capacity -= 1
                selected.update((i, j) for j in (near + distant)[:capacity])
            selected = sorted(selected)
        result.append(dict(step=step, budget=budget, phase="warmup" if step <= schedule["warmup"] else "evolution",
                           refresh_step=refresh_at, pool=[list(x) for x in pool],
                           selected=[list(x) for x in selected], rng=deepcopy(rng.bit_generator.state)))
        previous = budget
    return result


def sparse_forward(a0, features, pairs, stage1, weight, bias, assignment):
    """Recompute the normalized COO and stage-two propagation and Q projection."""
    norms = _normalize(features)
    row, col = np.asarray(pairs, dtype=np.int64).T
    values = .5 * (np.asarray(a0)[row, col] + np.exp(.5 * (norms[row] * norms[col]).sum(axis=1)))
    totals = np.zeros(a0.shape[0], dtype=np.float32)
    np.add.at(totals, row, values)
    values /= totals[row]
    linear = np.asarray(stage1) @ weight + bias
    regions = np.zeros_like(linear)
    np.add.at(regions, row, values[:, None] * linear[col])
    return values, regions, np.asarray(assignment) @ regions

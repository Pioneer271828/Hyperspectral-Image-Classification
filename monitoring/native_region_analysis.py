"""Registered interventions on actual graph ranges of a private selected model."""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from research_runtime import tensor_hash
from transfer.g3_training import state_hash
from transfer.multiscale_region_graph import MultiRangeBackbone
from transfer.native_released_points import HEADS, scores


SCHEMA = "wp06_native_trained_region_analysis_v1"
EPSILON = .002


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class Plan:
    target: int
    donor: int | None
    source: int | None
    channel: int = 0


def region_options(q, lattice):
    """Select coverage extrema from geometry alone and find the closest distinct rows."""
    pixels, regions = q.indices()
    shape = q.shape
    best = q.values().new_full((shape[0],), -1.)
    best.scatter_reduce_(0, pixels, q.values(), reduce="amax")
    selected = q.values() == best[pixels]
    indices = torch.full((shape[0],), shape[1], device=regions.device, dtype=regions.dtype)
    indices.scatter_reduce_(0, pixels[selected], regions[selected], reduce="amin")
    require(bool((indices < shape[1]).all()), "Every native pixel needs an actual assignment")
    dominant = indices.cpu().numpy()
    counts = np.bincount(dominant, minlength=shape[1])
    partial = lattice.areas() < lattice.factor**2
    if partial.any():
        edge = int(dominant[np.flatnonzero(partial)[0]])
    else:
        edge = int(dominant[-1])
    core = int(np.argmax(counts))
    least = int(np.argmin(counts))
    # Apply the same selector without observing labels, errors or scores.
    targets = dict(core=core, edge=edge, least=least)
    plans = {}
    centers = lattice.centers()
    center = np.stack([(centers[dominant == n].mean(0) if counts[n] else centers.mean(0)) for n in range(shape[1])])
    for label, row in targets.items():
        others = sorted((float(np.square(center[row]-center[key]).sum()), key)
                        for key in range(shape[1]) if key != row and counts[key])
        donor = others[0][1] if others else None
        source = others[-1][1] if others else None
        plans[label] = Plan(row, donor, source)
    return plans, dominant


@contextmanager
def stages(backbone, stage1, stage2):
    saved = {}
    names = {"graph_stage1": stage1, "graph_stage2": stage2}
    try:
        for method, transform in names.items():
            require(method not in vars(backbone), "Registered stage already has a competing observer")
            original = getattr(backbone, method)

            def apply(values, adjacency, dataset=None, *, fn=transform, original=original):
                return fn(original, values, adjacency, dataset)

            setattr(backbone, method, apply)
            saved[method] = original
        yield
    finally:
        for method in saved:
            delattr(backbone, method)


def branch_update(x, offset, plan, operation, variable):
    target = offset+plan.target
    if operation == "self":
        row = x[target]
    elif operation == "replace":
        require(plan.donor is not None and plan.donor != plan.target, "Replacement requires a distinct same-observation donor")
        row = x[offset+plan.donor]
    elif operation == "zero":
        row = torch.zeros_like(x[target])
    elif operation == "source_swap":
        require(plan.source is not None and plan.source != plan.target, "Source permutation requires two distinct rows")
        source = offset+plan.source
        ids = torch.tensor([target, source], device=x.device)
        return x.index_copy(0, ids, x.index_select(0, ids.flip(0)))
    elif operation == "delta":
        require(plan.source is not None, "No distinct source for a directional probe")
        index = offset+plan.source
        row = x[index]+torch.nn.functional.one_hot(torch.tensor(plan.channel, device=x.device), x.shape[1])*variable
        return x.index_copy(0, torch.tensor([index], device=x.device), row[None])
    else:
        raise ValueError("Unregistered operation")
    return x.index_copy(0, torch.tensor([target], device=x.device), row[None])


def evaluate(model, actual, targets, invoke, execution, historical):
    require(actual.receipt["role"] == "devval" and len(targets.points) > 0,
            "Analysis only accepts a released development endpoint")
    model.eval()
    start_model = state_hash(model)
    multirange = isinstance(model.backbone, MultiRangeBackbone)
    ranges = {"narrow": 0, "wide": 1} if multirange else {"single": 0}
    rows = np.array(targets.cells, copy=True)
    truth = np.asarray([target for _, target in targets.points])
    prior_capture = execution.capture_values, execution.graphs
    execution.capture_values = True
    support = []

    def call(*, branch=None, plan=None, operation=None, variable=None):
        support.clear()
        def first(original, x, a, dataset):
            support.append(tensor_hash(a))
            return original(x, a, dataset)

        def second(original, x, a, dataset):
            support.append(tensor_hash(a))
            offset = (len(x)//len(ranges))*ranges.get(branch, 0)
            if operation in ("source_swap", "delta"):
                x = branch_update(x, offset, plan, operation, variable)
            result = original(x, a, dataset)
            if operation in ("self", "replace", "zero"):
                result = branch_update(result, offset, plan, operation, variable)
            return result

        with stages(model.backbone, first, second):
            output, graph = invoke()
        require(len(support) == 2, "Expected both graph stages to run")
        return output, graph, tuple(support)

    try:
        with torch.no_grad():
            base, graph, used = call()
            old = scores(base, targets, detailed=False)
            require(graph == historical["graph"] and old == historical["scores"],
                    "Selected probabilities or baseline graph changed")
            coordinates, weights = execution.graphs[0]
            q = torch.sparse_coo_tensor(coordinates, weights, graph["q_shape"]).coalesce()
            plans, argmax = region_options(q, actual.lattice)
        points = [{"pixel": pixel, "label": label, "cell": int(cell), "dominant_region": int(argmax[cell])}
                  for (pixel, label), cell in zip(targets.points, rows, strict=True)]

        def verified(**kwargs):
            output, check, check_support = call(**kwargs)
            require(check == graph and check_support == used and torch.equal(output[2], base[2]),
                    "Analysis changed admissible Q, graph propagation support or CNN probabilities")
            return output

        baseline = {head: base[i][rows].detach().cpu().numpy() for i, head in enumerate(HEADS)}
        cases = {}
        for label, plan in plans.items():
            membership = np.flatnonzero(argmax == plan.target)
            hard = argmax[rows] == plan.target
            masks = {"all_released": np.ones(len(rows), dtype=bool), "in_dominant_region": hard,
                     "outside_dominant_region": ~hard}
            pix, region = q.indices().numpy()
            right = (region == plan.target) & (q.values().numpy() > 0)
            covered = np.isin(rows, pix[right])
            record = {"target": plan.target, "donor": plan.donor, "source": plan.source,
                      "native_cells": membership.tolist(), "dominant_released": int(hard.sum()),
                      "soft_released": int(covered.sum()),
                      "has_partial_cell": bool(np.any(actual.lattice.areas()[membership] < actual.lattice.factor**2)),
                      "branch_results": {}}
            for branch in ranges:
                tested = {}
                for operation in ("self", "zero", "replace", "source_swap"):
                    if (operation == "replace" and plan.donor is None) or (operation == "source_swap" and plan.source is None):
                        tested[operation] = {"status": "infeasible_distinct_geometry"}
                        continue
                    with torch.no_grad():
                        output = verified(branch=branch, plan=plan, operation=operation)
                    measured = {}
                    for i, head in enumerate(HEADS):
                        observed = output[i][rows].cpu().numpy()
                        before = baseline[head]
                        require(operation != "self" or np.array_equal(observed, before),
                                "Self identity control changed the endpoint")
                        pred = observed.argmax(1)+1
                        prior = before.argmax(1)+1
                        m = {key: classification_metrics(truth, pred, range(1, actual.classes+1), mask)
                             for key, mask in masks.items()}
                        measured[head] = {"max_probability_change": float(np.max(np.abs(observed-before))),
                            "probability_l1_mean": float(np.abs(observed-before).sum(1).mean()),
                            "argmax_changes": int((pred != prior).sum()),
                            "correct_to_wrong": int(((prior == truth) & (pred != truth)).sum()),
                            "wrong_to_correct": int(((prior != truth) & (pred == truth)).sum()),
                            "predictions": pred.tolist(), "probabilities": observed.tolist(), "metrics": m}
                    tested[operation] = {"status": "measured", "heads": measured}
                if plan.source is not None:
                    if membership.size:
                        cell = int(membership[0])
                    else:
                        candidates = np.flatnonzero(right)
                        require(bool(candidates.size), "Graph row does not represent any native cell")
                        cell = int(pix[candidates[np.argmax(q.values().numpy()[candidates])]])
                    target_class = int(base[1][cell].argmax().item())

                    def target_value(delta):
                        output = verified(branch=branch, plan=plan, operation="delta", variable=delta)
                        return output[1][cell, target_class]

                    zero = torch.tensor(0., requires_grad=True)
                    grad, = torch.autograd.grad(target_value(zero), zero)
                    with torch.no_grad():
                        up, down = float(target_value(torch.tensor(EPSILON))), float(target_value(torch.tensor(-EPSILON)))
                        finite = (up-down)/(2*EPSILON)
                    require(abs(float(grad)-finite) <= .001 + .04*abs(finite), "Regional Jacobian failed finite difference")
                    tested["source_direction"] = {"status": "measured", "source": plan.source, "channel": plan.channel,
                        "target_cell": cell, "class_column": target_class, "epsilon": EPSILON, "autograd": float(grad),
                        "finite_difference": finite, "branch": branch}
                else:
                    tested["source_direction"] = {"status": "infeasible_distinct_geometry"}
                record["branch_results"][branch] = tested
            cases[label] = record
        with torch.no_grad():
            rest = verified()
        require(all(torch.equal(a, b) for a, b in zip(base[:3], rest[:3], strict=True))
                and state_hash(model) == start_model, "Analysis altered original outputs or state")
        return {"observation": actual.receipt["observation"], "role": "devval", "graph": graph,
                "stage_support": list(used), "model_state": start_model, "roster": targets.summary(),
                "point_rows": points, "baseline_probabilities": {key: value.tolist() for key, value in baseline.items()},
                "ranges": list(ranges), "plans": cases, "state_and_outputs_restored": True}
    finally:
        execution.capture_values, execution.graphs = prior_capture

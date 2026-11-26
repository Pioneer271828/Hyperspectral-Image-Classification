"""Development-only six-arm planning over the existing WP11 lazy runtime.

No fitting, labels, model changes, or confirmation certification occur here.
Coverage is a delivered-column fraction, not a calibrated SRF overlap.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

ARMS = ("light", "full", "content", "content_coverage", "permuted_coverage", "matched_random")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def admit(table, source, target, *, scope, role, ancestry):
    """Validate metadata before any image/weight deserialization."""
    if role != "devval" or scope not in ("cross_device_development", "target_local_development_control"):
        raise ValueError("Only existing development readout is admitted; V5 is not confirmation")
    if scope == "target_local_development_control":
        if source != target or ancestry != {"source": source, "target": target, "kind": "target_scratch"}:
            raise ValueError("Target-local control cannot impersonate a source-target transfer")
        return {"scope": scope, "source": source, "target": target, "confirmation": False}
    rows = list(csv.DictReader(Path(table).open(encoding="utf-8-sig")))
    matches = [r for r in rows if (r["source"], r["target"]) == (source, target)]
    if len(matches) != 1 or matches[0]["status"] != "READY_DEVELOPMENT_ONLY":
        raise ValueError("Direction has no released development protocol")
    if ancestry != {"source": source, "target": target, "kind": "verified_source_pretrain_target_adaptation"}:
        raise ValueError("Missing exact single-source transfer ancestry")
    return {"scope": scope, "source": source, "target": target, "confirmation": False}


def plan(requests, margins, arm, *, full_fraction=.5, seed=64, reference=None):
    """Batch quota policy; pays light passes before selecting any heavy calls.

    The fixed half-call quota is selected without labels. It is not a calibrated
    online threshold. Random matches reference counts per actual workload cohort.
    """
    if arm not in ARMS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Unknown arm or RNG seed")
    if type(full_fraction) not in (float, int) or not 0 <= full_fraction <= 1:
        raise ValueError("Invalid explicit full-call quota")
    ids = [r.key for r in requests]
    if not ids or len(set(ids)) != len(ids) or set(margins) != set(ids):
        raise ValueError("Complete unique requests and cheap scores required")
    if any(not math.isfinite(margins[k]) or not 0 <= margins[k] <= 1 for k in ids):
        raise ValueError("Invalid confidence score")
    if any(not r.known or not math.isfinite(r.coverage) or not 0 < r.coverage <= 1 for r in requests):
        raise ValueError("This development policy requires admitted delivered-band metadata")
    if arm == "light":
        return [], {}
    if arm == "full":
        return sorted(ids), {}
    coverage = {r.key: r.coverage for r in requests}
    if arm == "permuted_coverage":
        # Within parent observation, rotate full/half/quarter descriptor values.
        # Spectra and wavelength inputs remain unmodified.
        groups = {}
        for r in requests:
            groups.setdefault(r.parent, []).append(r.key)
        for keys in groups.values():
            ordered = sorted(keys, key=lambda k: (digest([seed, k]), k))
            values = [coverage[k] for k in ordered]
            for k, v in zip(ordered, values[1:]+values[:1]):
                coverage[k] = v
    if arm == "matched_random":
        if reference is None or not set(reference) <= set(ids):
            raise ValueError("Frozen complete reference plan required")
        groups = {}
        for r in requests:
            # Preserve actual band/view and spatial-shape call counts.
            groups.setdefault((r.band, tuple(r.x.shape[:2])), []).append(r.key)
        chosen = []
        for cohort, keys in sorted(groups.items()):
            n = len(set(keys) & set(reference))
            chosen += sorted(keys, key=lambda k: (digest([seed, cohort, k]), k))[:n]
        return sorted(chosen), coverage
    score = {k: 1-margins[k] + (0 if arm == "content" else 1-coverage[k]) for k in ids}
    n = math.ceil(len(ids)*full_fraction)
    return sorted(sorted(ids, key=lambda k: (-score[k], k))[:n]), coverage


def execute(runtime, requests, arm, *, full_fraction=.5, seed=64, reference=None):
    """Reuse WP11 Runtime; full/Q/A are only invoked for selected blocks."""
    started = perf_counter()
    from inference.tile_cascade import gate_score
    if arm not in ARMS or type(full_fraction) not in (int, float) or not 0 <= full_fraction <= 1:
        raise ValueError("Unknown arm or invalid quota before model execution")
    if arm == "full":
        full = runtime.run(requests, "full")
        return {**full, "arm": arm, "pipeline_ms": (perf_counter()-started)*1000,
                "planning_ms": 0., "full_plan": sorted(r.key for r in requests)}
    cheap = runtime.run(requests, "light")
    begin = perf_counter()
    margins = {r.key: gate_score(p[0], "margin") for r, p in zip(requests, cheap["predictions"])}
    selected, descriptors = plan(requests, margins, arm, full_fraction=full_fraction, seed=seed, reference=reference)
    if arm == "matched_random":
        from inference.native_deployment import matched_random
        ref={"routes":[{"key":r.key,"route":"full" if r.key in reference else "light"} for r in requests]}
        cheap_ids=set(matched_random(requests,ref,seed,0.))
        selected=sorted(r.key for r in requests if r.key not in cheap_ids)
    planning_ms = (perf_counter()-begin)*1000
    subset = [r for r in requests if r.key in selected]
    heavy = runtime.run(subset, "full") if subset else None
    heavy_by_id = {} if heavy is None else {r.key: p for r, p in zip(subset, heavy["predictions"])}
    output = [heavy_by_id.get(r.key, p) for r, p in zip(requests, cheap["predictions"])]
    counts = {k: cheap["counts"][k]+(0 if heavy is None else heavy["counts"][k]) for k in cheap["counts"]}
    if counts["full_calls"] != len(selected) or counts["graph_builds"] != len(selected) or counts["graph_reuses"]:
        raise ValueError("Actual heavy/graph calls differ from policy")
    return {"arm": arm, "predictions": output, "full_plan": selected,
            "routes": [{"key": r.key, "route": "full" if r.key in selected else "light"} for r in requests],
            "counts": counts, "pipeline_ms": (perf_counter()-started)*1000, "planning_ms": planning_ms,
            "graph_ms": 0. if heavy is None else heavy["graph_ms"], "descriptor_values": descriptors,
            "margin": margins, "confirmation": False, "optimizer_updates": 0}


def metrics(labels, probabilities, classes):
    """Only explicit released point labels (1-based); never infer unlabelled 0."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities)
    if y.ndim != 1 or p.shape != (len(y), classes) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Finite released-point probabilities required")
    if np.any(y < 1) or np.any(y > classes) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Class order/label or probabilities differ")
    c = np.bincount((y-1)*classes+p.argmax(1), minlength=classes**2).reshape(classes, classes)
    support = c.sum(1)
    recall = [float(c[i,i]/n) if n else None for i,n in enumerate(support)]
    return {"confusion": c.tolist(), "support": support.tolist(), "recall": recall,
            "oa": float(c.trace()/c.sum()), "aa": float(np.mean(recall)) if all(n for n in support) else None}

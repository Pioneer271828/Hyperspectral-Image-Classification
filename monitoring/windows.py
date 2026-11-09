"""Sustained, train-only MON candidate evidence with fixed condition protection."""
import argparse
import json
import math
from pathlib import Path

from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import external_directory, write_json

from .readonly import REQUIRED_CONTEXT, threshold


METRICS = ("response_rate", "weight_rms_before", "gradient_rms", "relative_update")
IDENTITY = ("run_id", "commit", "data_hash", "split_hash", "dataset", "condition")


def _vector(values, channels, *, rate=False):
    if not isinstance(values, list) or len(values) != channels:
        raise ValueError("Every channel needs a measurement")
    checked = [threshold(v) for v in values]
    if rate and any(v > 1 for v in checked):
        raise ValueError("Response rates must be within [0, 1]")
    return checked


def _context(context, partition):
    if (not REQUIRED_CONTEXT <= context.keys() or context["partition"] not in partition
            or type(context["step"]) is not int or context["step"] < 0
            or any(not isinstance(context[k], str) or not context[k] for k in REQUIRED_CONTEXT - {"step"})):
        raise ValueError("Explicit permitted partition and complete monitoring identity required")
    return context


def sustained_candidates(plan, samples, probe):
    """Pair one activation and one actual update per preregistered sampled step."""
    windows = plan["windows"]
    steps = [step for window in windows for step in window]
    if (len(windows) < 2 or any(not window for window in windows)
            or any(type(step) is not int or step < 0 for step in steps)
            or any(a >= b for a, b in zip(steps, steps[1:])) or len(samples) != len(steps)):
        raise ValueError("At least two nonempty ordered windows with exact declared samples required")
    if (set(plan["limits"]) != set(METRICS) or plan["channel_role"] != "internal_feature"
            or not isinstance(plan["threshold_source"], str) or not plan["threshold_source"]
            or type(plan["axis"]) is not int or plan["axis"] < 0
            or type(plan["call_index"]) is not int or plan["call_index"] < 0):
        raise ValueError("Frozen thresholds and explicit internal feature mapping required")
    limits = {k: threshold(v) for k, v in plan["limits"].items()}
    protection_limit = threshold(plan["protection_response"])
    if limits["response_rate"] > 1 or protection_limit > 1 or protection_limit <= limits["response_rate"]:
        raise ValueError("Distinct bounded response and protection thresholds required")
    activation_epsilon = threshold(plan["activation_epsilon"])
    relative_epsilon = threshold(plan["relative_epsilon"])
    if relative_epsilon <= 0:
        raise ValueError("Positive relative-update epsilon required")

    context = _context(samples[0]["activation"]["context"], ("train",))
    identity = {k: context[k] for k in IDENTITY}
    channels = samples[0]["activation"]["statistics"]["channels"]
    if type(channels) is not int or channels <= 0:
        raise ValueError("Positive internal channel count required")
    layout = None
    weight_identity = None
    window_results, evidence = [], []
    offset = 0
    for window in windows:
        counts, observations = [], []
        for step, sample in zip(window, samples[offset:offset + len(window)]):
            a, u = sample["activation"], sample["update"]
            context = _context(a["context"], ("train",))
            if (a["schema_version"] != 1 or u["schema_version"] != 1 or a["kind"] != "activation"
                    or u["kind"] != "optimizer_update" or u["contract"] != "fp32_single_backward"
                    or u["optimizer_steps"] != 1 or context != u["context"] or context["step"] != step
                    or any(context[k] != v for k, v in identity.items())
                    or (a["layer"], a["call_index"], a["unit"]) != (plan["layer"], plan["call_index"], plan["unit"])
                    or u["epsilon"] != relative_epsilon):
                raise ValueError("Activation and actual update must share the frozen train identity and call")
            signature = (a["layout"], a["position"], a["dtype"], a["model_mode"])
            if layout is not None and signature != layout:
                raise ValueError("Activation representation changed across samples")
            layout = signature
            matching = [p for p in u["parameters"] if p["parameter"] == plan["parameter"]]
            if len(matching) != 1:
                raise ValueError("Exactly one declared parameter required")
            p = matching[0]
            shape = p["shape"]
            if (p["axis"] != plan["axis"] or p["unit"] != plan["unit"] or len(shape) < 2
                    or p["axis"] >= len(shape) or shape[p["axis"]] != channels
                    or p["requires_grad"] is not True or p["optimizer_group"] is None
                    or p["gradient_state"] not in ("zero", "present") or p["gradient_present"] is not True):
                raise ValueError("Frozen, missing or mismatched parameter evidence is ineligible")
            parameter_signature = (shape, p["dtype"])
            if weight_identity is not None and parameter_signature != weight_identity:
                raise ValueError("Parameter representation changed across samples")
            weight_identity = parameter_signature
            stats = a["statistics"]
            count = stats["count_per_channel"]
            if stats["channels"] != channels or stats["epsilon"] != activation_epsilon or type(count) is not int or count <= 0:
                raise ValueError("Every sample needs positive valid activation support and frozen epsilon")
            counts.append(count)
            observations.append({"response_rate": _vector(stats["response_rate"], channels, rate=True),
                                 **{k: _vector(p[k], channels) for k in METRICS[1:]}})
            evidence.append({"step": step, "checkpoint_hash": context["checkpoint_hash"],
                             "sample_id": context["sample_id"], "snapshot_bytes": u["snapshot_bytes"]})
        offset += len(window)
        total = sum(counts)
        response = [math.fsum(o["response_rate"][i] * (n / total) for o, n in zip(observations, counts))
                    for i in range(channels)]
        values = {"response_rate": response,
                  **{k: [max(o[k][i] for o in observations) for i in range(channels)] for k in METRICS[1:]}}
        window_results.append({"steps": window, "valid_elements_per_channel": total, "metrics": values})

    pctx = _context(probe["context"], ("train", "val"))
    if (probe["version"] != 1 or probe["probe_identity"] != plan["probe_identity"]
            or probe["threshold_source"] != plan["threshold_source"] or pctx["step"] != steps[-1]
            or pctx["checkpoint_hash"] != context["checkpoint_hash"]
            or any(pctx[k] != identity[k] for k in IDENTITY if k != "condition")):
        raise ValueError("Exact frozen condition probe on the final sampled checkpoint required")
    expected = plan["protection_buckets"]
    matching = [r for r in probe["records"]
                if (r["layer"], r["call_index"], r["unit"]) == (plan["layer"], plan["call_index"], plan["unit"])]
    minimum = plan["min_probe_elements"]
    if (not expected or any(not isinstance(b, str) or not b for b in expected) or len(set(expected)) != len(expected)
            or sorted(r["bucket"] for r in matching) != sorted(expected)
            or type(minimum) is not int or minimum <= 0):
        raise ValueError("All declared protection buckets are required exactly once")
    protected = {}
    for row in matching:
        stats = row["statistics"]
        count = stats["count_per_channel"]
        if ((row["layout"], row["position"]) != layout[:2] or stats["channels"] != channels
                or stats["epsilon"] != activation_epsilon or type(count) is not int or count < minimum):
            raise ValueError("Protection bucket layout, threshold or support is insufficient")
        protected[row["bucket"]] = {"count_per_channel": count,
                                    "response_rate": _vector(stats["response_rate"], channels, rate=True)}

    candidates = [i for i in range(channels)
                  if all(w["metrics"][k][i] <= v for w in window_results for k, v in limits.items())
                  and all(p["response_rate"][i] < protection_limit for p in protected.values())]
    return {"schema_version": 1, "policy": "mon_sustained_train_candidates_v1", "plan_sha256": digest(plan),
            "identity": identity, "layer": plan["layer"], "parameter": plan["parameter"], "channels": channels,
            "windows": window_results, "samples": evidence, "condition_protection": protected,
            "candidate_channel_indices": candidates, "automatic_pruning": False, "needs_intervention": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON containing plan, samples and probe")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh evidence directory outside checkout")
    args = parser.parse_args(argv)
    if file_hash(args.input) != args.sha256:
        raise ValueError("Exact frozen MON evidence required")
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    receipt = sustained_candidates(bundle["plan"], bundle["samples"], bundle["probe"])
    if file_hash(args.input) != args.sha256:
        raise ValueError("MON input changed while reading")
    out = external_directory(args.out)
    out.mkdir(exist_ok=False)
    write_json(out / "candidates.json", {**receipt, "input_sha256": args.sha256})


if __name__ == "__main__":
    main()

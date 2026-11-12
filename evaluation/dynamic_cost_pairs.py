"""Compare fixed, preregistered adaptive/control pairs from one development curve."""

from evaluation.dynamic_curve import number, pin, require, timing_summary
from model.research_config import digest


POLICY = "dy-fixed-cost-pairs-v1"


def quality_delta(adaptive, fixed):
    require(adaptive["support"] == fixed["support"], "Matched class support required")

    def difference(a, b):
        return None if a is None or b is None else a - b

    require(len(adaptive["recall"]) == len(fixed["recall"]) == len(fixed["support"]),
            "Aligned per-class recall required")
    return dict(support=fixed["support"][:], oa=difference(adaptive["oa"], fixed["oa"]),
                aa=difference(adaptive["aa"], fixed["aa"]),
                recall=[difference(a, b) for a, b in zip(adaptive["recall"], fixed["recall"])],
                nll=difference(adaptive["nll"], fixed["nll"]), ece=difference(adaptive["ece"], fixed["ece"]))


def matched_cost_pairs(curve, specification, *, expected_spec_hash, expected_curve_hash):
    """Use existing measured work points. Never search or interpolate pairs."""
    require(pin(expected_curve_hash) and digest(curve) == expected_curve_hash, "Pinned curve report required")
    require(pin(expected_spec_hash) and digest(specification) == expected_spec_hash,
            "Pinned pair registration required")
    require(isinstance(specification, dict)
            and specification.keys() == {"policy", "curve_protocol_hash", "relative_tolerances", "pairs"}
            and specification["policy"] == POLICY
            and curve["format"] == "dy-development-curve-report-v1"
            and pin(curve["protocol_hash"]) and curve["protocol_hash"] == specification["curve_protocol_hash"],
            "Registered development curve required")
    limits = specification["relative_tolerances"]
    require(isinstance(limits, dict) and limits.keys() == {"p50_ms", "p95_ms"}
            and all(number(v) and 0 <= v <= 1 for v in limits.values()), "Explicit p50/p95 tolerances required")
    points = {p["id"]: p for p in curve["points"]}
    require(len(points) == len(curve["points"]), "Duplicate curve points")
    pairs = specification["pairs"]
    require(isinstance(pairs, list) and pairs and all(isinstance(p, dict)
            and p.keys() == {"adaptive", "fixed"}
            and all(isinstance(k, str) and k in points for k in p.values()) for p in pairs),
            "Registered adaptive/fixed point IDs required")
    adaptive_ids = {p["id"] for p in points.values() if p["method"] == "adaptive"}
    require(len(pairs) == len(adaptive_ids) and {p["adaptive"] for p in pairs} == adaptive_ids,
            "Exactly one registered fixed comparator for every adaptive point required")

    comparisons = []
    for pair in pairs:
        adaptive, fixed = points[pair["adaptive"]], points[pair["fixed"]]
        require(fixed["method"] == "fixed", "Comparator must be a registered fixed path")
        require(adaptive["total_tiles"] == fixed["total_tiles"], "Same confirmation workload required")
        repeats = len(fixed["cost"]["wall_ms"])
        cost = {}
        a = timing_summary(adaptive["cost"]["wall_ms"], repeats=repeats, blocks=adaptive["total_tiles"])
        f = timing_summary(fixed["cost"]["wall_ms"], repeats=repeats, blocks=fixed["total_tiles"])
        for metric, tolerance in limits.items():
            require(a[metric] == adaptive["cost"][metric] and f[metric] == fixed["cost"][metric],
                    "Timing summaries must match raw samples")
            relative = (a[metric] - f[metric]) / f[metric]
            cost[metric] = dict(adaptive=a[metric], fixed=f[metric], relative_delta=relative,
                                within_tolerance=abs(relative) <= tolerance)
        groups = [{(g["scene"], g["condition"]): g for g in p["groups"]} for p in (adaptive, fixed)]
        require(groups[0].keys() == groups[1].keys()
                and all(len(g) == len(p["groups"]) for g, p in zip(groups, (adaptive, fixed))),
                "Same unique scene and condition groups required")
        deltas = [dict(scene=key[0], condition=key[1],
                       delta=quality_delta(groups[0][key]["quality"], groups[1][key]["quality"]))
                  for key in sorted(groups[0])]
        comparisons.append(dict(pair, cost=cost, cost_matched=all(v["within_tolerance"] for v in cost.values()),
                                adaptive_all_groups_protected=adaptive["all_groups_protected"],
                                fixed_all_groups_protected=fixed["all_groups_protected"],
                                delta=quality_delta(adaptive["quality"], fixed["quality"]), groups=deltas))
    return dict(policy=POLICY, registration_sha256=expected_spec_hash, curve_sha256=expected_curve_hash,
                data_sha256=curve["data_hash"], relative_tolerances=dict(limits), comparisons=comparisons,
                selection_performed=False, statistical_noninferiority=False, real_speed_claim=False)

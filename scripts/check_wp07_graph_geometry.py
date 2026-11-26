"""Independent scalar oracle for published graph supports, weights and distances."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checks_enabled():
    if sys.flags.optimize:
        raise RuntimeError("Graph geometry oracle refuses optimized Python (-O/-OO)")


def validate_view(view):
    gh, gw = view["grid"]
    height, width = view["identity"]["grid_shape"]
    factor = view["factor"]
    require(view["q"]["shape"] == [height*width, gh*gw], "Q and lattice dimensions differ")
    actual = []
    for radius, name in ((1, "near"), (2, "wide")):
        data = view[name]
        links = list(zip(*data["positive_edges"]))
        weights = data["positive_weights"]
        require(len(links) == len(weights) == len(set(links)), "Duplicate or missing weighted graph edges")
        expected = {(a, b) for a in range(gh*gw) for b in range(gh*gw)
                    if max(abs(a//gw-b//gw), abs(a%gw-b%gw)) <= radius}
        require(set(links) == expected, f"Radius {radius} positive support differs from independent grid enumeration")
        require(all(math.isfinite(v) and v > 0 for v in weights), "Nonpositive or nonfinite graph weight")
        require(all(abs(math.fsum(v for (a, _), v in zip(links, weights) if a == row)-1) <= 1e-5
                    for row in range(gh*gw)), "Row-normalized weights differ")
        distances = [math.hypot((a//gw-b//gw)*height*factor/gh,
                               (a%gw-b%gw)*width*factor/gw) for a, b in links]
        for field, value in (("distance_min", min(distances)), ("distance_max", max(distances)),
                             ("distance_mean", math.fsum(distances)/len(distances))):
            require(math.isclose(data[field], value, rel_tol=1e-12, abs_tol=1e-10), "Distance distribution summary differs")
        hops = Counter(str(max(abs(a//gw-b//gw), abs(a%gw-b%gw))) for a, b in links)
        require(dict(hops) == data["chebyshev_hop_counts"], "Hop distance distribution differs")
        actual.append(dict(zip(links, weights)))
    near, wide = actual
    common = set(near) & set(wide)
    return {"identifiable": set(near) < set(wide), "support_equivalent": set(near) == set(wide),
            "weighted_equivalent": near == wide,
            "max_common_weight_delta": max(abs(near[e]-wide[e]) for e in common),
            "added_distance2_edges": len(set(wide)-set(near))}


def read_pin(pin, root):
    p = Path(pin["path"])
    require(p == p.resolve(strict=True) and root in p.parents, "Geometry evidence escapes the authorized root")
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin["sha256"], "Geometry evidence digest differs")
    return json.loads(data)


def main():
    checks_enabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    data = args.geometry.resolve(strict=True).read_bytes()
    source = json.loads(data)
    require(source["labels_scored"] == source["classifier_heads_evaluated"] == source["optimizer_updates"] == 0,
            "Expected pre-score zero-update geometry")
    started = perf_counter()
    summaries, all_distinct, eligible = {}, [], []
    count = 0
    for name in source["protocol_order"]:
        protocol = read_pin(source["protocols"][name], root)
        results = [validate_view(v) for v in protocol["views"]]
        count += len(results)
        roles = {role: Counter(str(r["identifiable"]) for v, r in zip(protocol["views"], results) if v["role"] == role)
                 for role in ("devtrain", "devval")}
        require(roles == protocol["role_counts"], "Published role admission differs from scalar oracle")
        require(all(r["identifiable"] == v["comparison"]["identifiable"] for v, r in zip(protocol["views"], results)),
                "Geometry range eligibility differs")
        summaries[name] = {"views": len(results), "role_counts": roles,
                          "identifiable": sum(r["identifiable"] for r in results),
                          "equivalent_supports": sum(r["support_equivalent"] for r in results),
                          "exact_weighted_equivalent": sum(r["weighted_equivalent"] for r in results),
                          "max_common_weight_delta": max(r["max_common_weight_delta"] for r in results)}
        if all(r["identifiable"] for r in results):
            all_distinct.append(name)
        if all(v.get("True", 0) > 0 for v in roles.values()):
            eligible.append(name)
    selected = (all_distinct or eligible or [None])[0]
    require(selected == source["selected_protocol"], "Pre-score protocol choice differs")
    result = {"status": "passed", "geometry_sha256": hashlib.sha256(data).hexdigest(),
              "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "views": count,
              "selected_protocol": selected, "protocols": summaries, "optimizer_updates": 0,
              "labels_scored": 0, "classifier_heads_evaluated": 0, "elapsed_seconds": perf_counter()-started}
    with args.out.resolve().open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Independently audit released PaviaU edge deletion and export deidentified tables."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import median

SEEDS = (64, 512, 2048)
HEADS = ("fused", "graph", "cnn")
ARMS = ("none", "important", "low_weight", "random:64", "random:512", "random:2048")
APPROVED = "67e0a881412da6a86e10ca80617f023ac709ff2dc06fad969dbf154a17d34927"


def read(path, digest):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Published file digest differs: {path}")
    return json.loads(payload)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(edges, size, grid, shape):
    gh, gw = grid
    h, w = shape
    receiving, sending, lengths = Counter(), Counter(), Counter()
    for edge in edges:
        a, b = divmod(edge, size)
        ra, ca = divmod(a, gw)
        rb, cb = divmod(b, gw)
        distance = (Fraction((ra - rb) * h, gh) ** 2 + Fraction((ca - cb) * w, gw) ** 2)
        lengths[str(int(distance * size ** 2))] += 1
        receiving[a] += 1
        sending[b] += 1
    return [receiving[i] for i in range(size)], [sending[i] for i in range(size)], lengths


def l1(one, two):
    return sum(abs(one[k] - two[k]) for k in one.keys() | two.keys())


def audit_graph(record, blocks, points, graph_rows, scores):
    seed, unit, n = [record[k] for k in ("seed", "unit", "n")]
    base_edges, base_weights = [record[k] for k in ("support", "weights")]
    weights = dict(zip(base_edges, base_weights, strict=True))
    check(base_edges == sorted(set(base_edges)) and len(base_edges) <= n * n
          and n == math.prod(record["grid"]) and record["shape"] == blocks[unit]["shape"]
          and all(0 <= e < n * n and math.isfinite(weights[e]) and weights[e] > 0 for e in base_edges)
          and all((n + 1) * i in weights for i in range(n)), f"Malformed original sparse graph: {seed}, {unit}")
    buckets = defaultdict(list)
    for edge in base_edges:
        row, col = divmod(edge, n)
        if row != col:
            buckets[row].append(edge)
    important = sorted(min(es, key=lambda e: (-weights[e], e)) for es in buckets.values())
    check(important == record["arms"]["important"]["removed"], "Important-edge ranking differs")
    target_r, target_s, target_d = geometry(important, n, record["grid"], record["shape"])
    reference = record["arms"]["none"]
    locked = points[unit]
    seen_masks = defaultdict(list)
    for arm in ARMS:
        data = record["arms"][arm]
        removed = data["removed"]
        check(removed == sorted(set(removed)) and set(removed) <= weights.keys()
              and all(e // n != e % n for e in removed), "Invalid mask or mandatory self loop removed")
        seen_masks[tuple(removed)].append(arm)
        remaining = sorted(weights.keys() - set(removed))
        check(data["graph_after"] == remaining and data["actual_nnz"] == len(remaining),
              "Effective edge support differs")
        observed_r, observed_s, observed_d = geometry(removed, n, record["grid"], record["shape"])
        receiver_l1 = sum(abs(a - b) for a, b in zip(observed_r, target_r, strict=True))
        sender_l1 = sum(abs(a - b) for a, b in zip(observed_s, target_s, strict=True))
        distance_l1 = l1(observed_d, target_d)
        count_delta = len(removed) - len(important)
        check(data["difference"] == dict(count_delta=count_delta, receiver_l1=receiver_l1,
                                          sender_l1=sender_l1, distance_exact_l1=distance_l1)
              and observed_r == data["geometry"]["receiver_degree"] and observed_s == data["geometry"]["sender_degree"]
              and observed_d == data["geometry"]["distance_histogram"], "Stored geometry certificate differs")
        if arm == "none":
            check(not removed and data["status"] == "reference", "No-deletion baseline differs")
        else:
            check(receiver_l1 == count_delta == 0, "Cardinality or receive degree differs")
            if arm != "important":
                if data["status"] == "strict":
                    check(sender_l1 == distance_l1 == 0 and removed != important and data["solver"] == "optimal",
                          "Invalid strict control")
                else:
                    check(data["status"] in ("unmatched_same_mask", "unmatched_diagnostic")
                          and data["solver"] in ("no_distinct_control", "unresolved")
                          and bool(removed == important) == (data["status"] == "unmatched_same_mask")
                          and (data["solver"] == "unresolved" or removed == important or sender_l1 or distance_l1),
                          "Unmatched fallback disposition differs")
        remaining_weights = dict(zip(remaining, data["actual_weights"], strict=True))
        mass = Counter()
        for edge in remaining:
            mass[edge // n] += weights[edge]
        for edge in remaining:
            expected = weights[edge] / mass[edge // n] if removed else weights[edge]
            check(math.isclose(remaining_weights[edge], expected, rel_tol=1e-6, abs_tol=5e-8),
                  "Effective sparse weights differ from independent row normalization")
        mm = data["sparse_mm"]
        check(mm and all(row["nnz"] == len(remaining) and row["shape"] == [n, n]
                        and row["lhs_sha256"] == data["a_after_sha256"] and row["rhs_width"] > 0
                        for row in mm), "Sparse multiply lacks effective support witness")
        check(data["q_sha256"] == reference["q_sha256"]
              and data["a_before_sha256"] == reference["a_before_sha256"], "Original Q/A differs")
        costs = [data[k] for k in ("forward_seconds", "native_build_seconds", "deletion_build_seconds")]
        kernel_seconds = math.fsum(row["seconds"] for row in mm)
        check(all(math.isfinite(v) and v >= 0 for v in (*costs, kernel_seconds))
              and costs[1] + costs[2] + kernel_seconds <= costs[0], "Invalid nested timings")
        graph_rows.append(dict(seed=seed, unit=unit, arm=arm, status=data["status"],
                               solver=data["solver"], reason=data.get("reason"), nodes=n,
                               original_nnz=len(base_edges), effective_nnz=len(remaining), removed=len(removed),
                               count_delta=count_delta, receiver_l1=receiver_l1, sender_l1=sender_l1,
                               distance_exact_l1=distance_l1, deleted_weight=data["geometry"]["deleted_weight"],
                               baseline_overlap=len(set(removed) & set(important)),
                               native_build_ms=1000 * costs[1], deletion_build_ms=1000 * costs[2],
                               sparse_calls=len(mm), sparse_mm_ms=1000 * kernel_seconds, forward_ms=1000 * costs[0],
                               matching_ms=1000 * record["matching_seconds"] if arm == "none" else None,
                               solver_ms=1000 * record["solver_seconds"] if arm == "none" else None))
        ref_points = {p["point"]: p for p in reference["points"]}
        check(set(locked) == {p["point"] for p in data["points"]} and len(locked) == len(data["points"]),
              "Released point population differs")
        for point in data["points"]:
            alias = point["point"]
            label = locked[alias]["label"]
            check(point["label"] == label and point["probabilities"][2] == ref_points[alias]["probabilities"][2],
                  "Point label or unchanged CNN control differs")
            for i, (head, probs) in enumerate(zip(HEADS, point["probabilities"], strict=True)):
                check(len(probs) == 9 and all(v >= 0 and math.isfinite(v) for v in probs)
                      and math.isclose(math.fsum(probs), 1.0, abs_tol=1e-6), "Invalid probability vector")
                prediction = max(range(9), key=probs.__getitem__) + 1
                if arm == "none":
                    check(prediction == locked[alias]["prior"][str(seed)][
                        ("left", "native_graph_pred", "native_cnn_pred")[i]], "Baseline prediction differs")
                scores[seed, arm, head].append((label, prediction, -math.log(max(probs[label - 1], 1e-8))))
    return [members for members in seen_masks.values() if len(members) > 1]


def write_table(root, name, rows):
    path = root / name
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(raw, registration, root):
    check(raw["role"] == registration["role"] == "devval" and raw["scene"] == "paviaU"
          and raw["optimizer_updates"] == 0 and raw["query_accessed"] is False
          and tuple(raw["arms"]) == ARMS and tuple(raw["heads"]) == HEADS
          and raw["registration_sha256"] == APPROVED, "Wrong released intervention scope")
    blocks = {r["unit"]: r for r in registration["blocks"]}
    points = defaultdict(dict)
    for point in registration["points"]:
        points[point["unit"]][point["point"]] = point
    graph_rows, scores, aliases = [], defaultdict(list), []
    check(len(raw["records"]) == 51 and {(r["seed"], r["unit"]) for r in raw["records"]} ==
          {(seed, unit) for seed in SEEDS for unit in blocks}, "Incomplete context roster")
    for record in raw["records"]:
        seed = record["seed"]
        check(registration["checkpoints"][str(seed)] ==
              dict(step=record["step"], sha256=record["checkpoint_sha256"]), "Checkpoint differs")
        for arms in audit_graph(record, blocks, points, graph_rows, scores):
            aliases.append(dict(seed=seed, unit=record["unit"], arms=";".join(arms)))
    summary, class_rows, cost_summary = [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            graphs = [row for row in graph_rows if (row["seed"], row["arm"]) == (seed, arm)]
            status = Counter(row["status"] for row in graphs)
            interpretation = ("reference" if arm in ("none", "important") else
                              "strict" if status["strict"] == 17 else "contains_unmatched_diagnostics")
            cost_summary.append(dict(seed=seed, arm=arm, blocks=len(graphs), interpretation=interpretation,
                                     strict_blocks=status["strict"], unmatched_blocks=status["unmatched_diagnostic"],
                                     same_mask_blocks=status["unmatched_same_mask"],
                                     original_nnz=sum(row["original_nnz"] for row in graphs),
                                     effective_nnz=sum(row["effective_nnz"] for row in graphs),
                                     **{f"p50_{field}": median(r[field] for r in graphs)
                                        for field in ("native_build_ms", "deletion_build_ms", "sparse_mm_ms", "forward_ms")}))
            for head in HEADS:
                identity = dict(seed=seed, arm=arm, head=head, interpretation=interpretation)
                rows = scores[seed, arm, head]
                support = Counter(label for label, _, _ in rows)
                check(support == {c: 5 for c in range(1, 10)}, "Expected 45 released points, five per class")
                tally = Counter(label for label, pred, _ in rows if label == pred)
                baseline = scores[seed, "none", head]
                base_count = sum(a == b for a, b, _ in baseline)
                correct = sum(tally.values())
                summary.append(dict(**identity, support=45, correct=correct, oa=correct / 45,
                                    aa=math.fsum(tally[c] / support[c] for c in support) / 9,
                                    nll=math.fsum(loss for _, _, loss in rows) / 45,
                                    delta_oa_pp=(correct - base_count) * 100 / 45))
                for c in range(1, 10):
                    class_rows.append(dict(**identity, class_id=c, support=support[c], correct=tally[c],
                                           recall=tally[c] / support[c]))
    root.mkdir()
    tables = {"graph-and-cost.csv": graph_rows, "three-heads.csv": summary, "per-class.csv": class_rows,
              "cost-summary.csv": cost_summary}
    if aliases:
        tables["duplicate-masks.csv"] = aliases
    receipt = {"status": "passed", "registration_sha256": APPROVED, "commit": raw["commit"],
               "scene": "paviaU", "role": "devval", "contexts": len(raw["records"]),
               "updates": 0, "query_accessed": False, "arm_status": dict(Counter(row["status"] for row in graph_rows)),
               "tables": {name: {"sha256": write_table(root, name, rows), "rows": len(rows)} for name, rows in tables.items()}}
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = export(read(args.source, args.source_sha256), read(args.registration, APPROVED), args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

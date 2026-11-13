"""Recompute WP10 INT8 classification protection and paired costs from receipt rows."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

from research_runtime import external_directory, write_json
from transfer import matched_route_admission as admission


ARMS = ("qat-first", "qat-second", "qat-both")
HEADS = ("fused", "graph", "cnn")


def recalc(row, head):
    target, prediction = row["targets"], row["predictions"][head]
    if Counter(target) != dict.fromkeys(range(1, 10), 5):
        raise ValueError("Original ordered K5/V5 labels required")
    counts = Counter(zip(target, prediction, strict=True))
    confusion = [[counts[i, j] for j in range(1, 10)] for i in range(1, 10)]
    hits = [confusion[i][i] for i in range(9)]
    prevalence = [sum(row[i] for row in confusion) for i in range(9)]
    oa = Fraction(sum(hits), 45)
    f1 = sum(Fraction(2*k, 5+n) for k, n in zip(hits, prevalence, strict=True))/9
    return dict(confusion_matrix=confusion, oa=float(oa), aa=float(oa), kappa=float((oa-Fraction(1, 9))/Fraction(8, 9)),
                macro_f1=float(f1)), hits


def protect(reference, candidate, head):
    if reference["targets"] != candidate["targets"]:
        raise ValueError("Unpaired labels")
    _, old = recalc(reference, head)
    _, new = recalc(candidate, head)
    drops = {str(i+1): Fraction(a-b, 5) for i, (a, b) in enumerate(zip(old, new, strict=True))}
    missed = Counter(label for label, a, b in zip(reference["targets"], reference["predictions"][head],
                                                  candidate["predictions"][head], strict=True) if label == a and b != label)
    overall = sum(drops.values())/9
    failed = [int(k) for k, v in drops.items() if v > Fraction(1, 200)]
    return dict(oa_drop=float(overall), aa_drop=float(overall),
                per_class_recall_drop={k: float(v) for k, v in drops.items()}, failed_recall_classes=failed,
                new_small_class_errors={str(k): missed[k] for k in range(1, 10)},
                passes_metric_guards=overall <= Fraction(1, 200) and not failed and not missed)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--quality", nargs=2, required=True)
    parser.add_argument("--cost", nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    quality, cost = (admission.read(dict(zip(("path", "sha256"), items, strict=True)))
                     for items in (args.quality, args.cost))
    if any(quality[key] for key in ("query", "gt", "optimizer_updates", "calibration_new_calls")):
        raise ValueError("Only frozen inference after the original calibration is admitted")
    rows = [quality["original"], *quality["native"].values(), *quality["candidate"].values()]
    for row in rows:
        for head in HEADS:
            recomputed, hits = recalc(row, head)
            for key, expected in recomputed.items():
                actual = row["metrics"][head][key]
                if isinstance(expected, list):
                    assert actual == expected
                else:
                    assert abs(actual-expected) <= 1e-14
            for entry, hit in zip(row["metrics"][head]["per_class"], hits, strict=True):
                assert entry["recall"] == hit/5
    for arm in ARMS:
        candidate = quality["candidate"][arm]
        basis = quality["native"][arm]
        for head in HEADS:
            for baseline, label in ((quality["native"]["fp32"], "versus_same_budget_fp32_guards"),
                                    (basis, "versus_native_guards"), (quality["original"], "versus_original_guard")):
                independent = protect(baseline, candidate, head)
                stored = candidate[label][head]
                for k, v in independent.items():
                    if isinstance(v, float):
                        assert abs(v-stored[k]) <= 1e-14
                    elif k == "per_class_recall_drop":
                        assert all(abs(v[i]-stored[k][i]) <= 1e-14 for i in v)
                    else:
                        assert v == stored[k]
        assert candidate["probability_hashes"] == basis["probability_hashes"]
        assert sum(candidate["dispatch"].values()) == (10 if arm == "qat-both" else 5)
    failed = protect(quality["native"]["fp32"], quality["candidate"]["qat-both"], "cnn")
    assert failed["failed_recall_classes"] == [4] and failed["new_small_class_errors"]["4"] == 1
    for family in ("resident", "versus_same_budget_fp32"):
        for arm in ARMS:
            values = cost[family][arm]
            order_counts = Counter(tuple(s["order"]) for s in values["samples"])
            assert len(order_counts) == 2 and set(order_counts.values()) == {6}
            for name, summary in values["summary"].items():
                elapsed = sorted(s["timing"][name]["wall_ms"] for s in values["samples"])
                assert summary["p50_ms"] == median(elapsed)
                # Inclusive percentile uses linear interpolation between the adjacent sorted ranks.
                pos = .95*(len(elapsed)-1)
                i = int(pos)
                interpolated = elapsed[i] + (pos-i)*(elapsed[i+1]-elapsed[i])
                assert abs(summary["p95_ms"] - interpolated) <= 1e-12
    workers = [x["pid"] for v in cost["isolated_startup"].values() for r in v for x in r["samples"].values()]
    assert len(workers) == len(set(workers)) == 12
    out = external_directory(args.out)
    if out.exists():
        raise FileExistsError(out)
    write_json(out, dict(status="passed", models=len(rows), confusion_matrices=len(rows)*3,
                         guard_rows=27, resident_ab_pairs=72, startup_processes=len(workers),
                         old_cnn_failed_class=4, old_cnn_new_errors=1))


if __name__ == "__main__":
    main()

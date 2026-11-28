"""Compare two SLIC capture receipts without opening spectra, models, or labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import median
import subprocess

ROSTER = {"synthetic": 6, "xuzhou": 5, "paviaU": 41, "whu-honghu": 10}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, digest):
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == digest, f"Capture SHA256 differs: {path.name}")
    return json.loads(raw)


def validated(row):
    labels = row["labels"]
    h, w, bands = row["shape"]
    check(h * w == len(labels) == row["q_nnz"] and h > 0 and w > 0 and bands > 0,
          "Incomplete membership")
    seen = list(dict.fromkeys(labels))
    check(seen == list(range(row["q_columns"])) and all(type(i) is int for i in labels),
          "Invalid raster-order region identities")
    role = "synthetic" if row["scene"] == "synthetic" else "devval"
    check(row["role"] == role and (role == "synthetic" or
          (row["input"]["fit_partition"] == "train" and row["input"]["fit_role"] in ("devtrain", None))),
          "Validation block lacks its original training normalization authority")
    check(len(row["elapsed_ms"]) == 3 and all(isfinite(t) and t > 0 for t in row["elapsed_ms"]),
          "Expected three timing samples")
    return labels


def neighbors(i, h, w):
    y, x = divmod(i, w)
    if x:
        yield i - 1
    if x + 1 < w:
        yield i + 1
    if y:
        yield i - w
    if y + 1 < h:
        yield i + w


def geometry(labels, h, w):
    visited = set()
    components = Counter()
    for i, label in enumerate(labels):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        components[label] += 1
        while stack:
            for j in neighbors(stack.pop(), h, w):
                if j not in visited and labels[j] == label:
                    visited.add(j)
                    stack.append(j)
    borders = {(i, j) for i, label in enumerate(labels) for j in neighbors(i, h, w)
               if i < j and labels[j] != label}
    return sum(value > 1 for value in components.values()), borders


def pair_count(histogram):
    return sum(n * (n - 1) // 2 for n in histogram.values())


def compare(one, two):
    check(all(one[field] == two[field] for field in
              ("name", "scene", "role", "input_sha256", "input", "shape", "scale")),
          f"Pair inputs differ: {one['name']}")
    a, b = validated(one), validated(two)
    h, w, bands = one["shape"]
    ad, ae = geometry(a, h, w)
    bd, be = geometry(b, h, w)
    joint = pair_count(Counter(zip(a, b)))
    split_or_join = pair_count(Counter(a)) + pair_count(Counter(b)) - 2 * joint
    exact = a == b
    check(exact == (split_or_join == 0), "Partition comparison and coassignment counts disagree")
    return dict(case=one["name"], scene=one["scene"], role=one["role"], bands=bands,
                pixels=h*w, scale=one["scale"], equal_partition=exact,
                reference_regions=one["q_columns"], candidate_regions=two["q_columns"],
                q_nnz=one["q_nnz"], changed_coassignment_pairs=split_or_join,
                changed_four_neighbor_borders=len(ae ^ be),
                reference_disconnected_regions=ad, candidate_disconnected_regions=bd,
                reference_median_ms=median(one["elapsed_ms"]), candidate_median_ms=median(two["elapsed_ms"]))


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args):
    captures = [read(path, digest) for path, digest in
                ((args.reference, args.reference_sha256), (args.candidate, args.candidate_sha256))]
    records = []
    for actual, expected in zip(captures, ("0.24.0", "0.25.2"), strict=True):
        check(actual["schema"] == "slic_portability_v1" and actual["commit"] == args.expected_commit
              and actual["source"]["version"] == expected
              and not any(actual[key] for key in ("query_accessed", "raw_gt_loaded", "statistics_refit", "real_updates")),
              "Unqualified capture or scope")
        cohort = {row["name"]: row for row in actual["cases"]}
        check(len(cohort) == len(actual["cases"]) and Counter(row["scene"] for row in cohort.values()) == ROSTER,
              "Invalid released or fixture roster")
        records.append(cohort)
    a, b = records
    check(a.keys() == b.keys(), "Case identities differ")
    pairs = [compare(a[name], b[name]) for name in sorted(a)]
    summary = []
    for scene in ROSTER:
        rows = [row for row in pairs if row["scene"] == scene]
        summary.append(dict(scene=scene, role=rows[0]["role"], cases=len(rows),
                            exact_partitions=sum(row["equal_partition"] for row in rows),
                            pixels=sum(row["pixels"] for row in rows),
                            changed_pairs=sum(row["changed_coassignment_pairs"] for row in rows),
                            changed_borders=sum(row["changed_four_neighbor_borders"] for row in rows),
                            reference_median_ms=median(r["reference_median_ms"] for r in rows),
                            candidate_median_ms=median(r["candidate_median_ms"] for r in rows)))
    args.out.mkdir()
    hashes = {name: write_rows(args.out / name, rows) for name, rows in
              (("paired-blocks.csv", pairs), ("scene-summary.csv", summary))}
    result = dict(status="passed", audited_cases=len(pairs), exact_partitions=sum(p["equal_partition"] for p in pairs),
                  sources=[v["source"] for v in captures], captures=[args.reference_sha256, args.candidate_sha256],
                  checkout=args.expected_commit, approvals={"production_replacement": False, "query": False},
                  real_updates=0, table_sha256=hashes, summary=summary)
    (args.out / "receipt.json").write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("reference-sha256", "candidate-sha256", "expected-commit"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    check(subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() == args.expected_commit
          and not subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True)
          and not args.out.resolve().is_relative_to(root), "Exact clean consumer checkout and external output required")
    result = export(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

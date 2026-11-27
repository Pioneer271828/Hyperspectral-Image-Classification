"""Publish the approved frozen PaviaU matched-deletion evidence through WP12."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from scripts import audit_pavia_matched_deletion as audit


SOURCE = "ad07800d285a60b1fce73bbbad3fa11f45de0b057037f7170b70c095d55ae07a"
PRODUCER = "cd263c9e64ff7538d6fc7b831157df025a038b58"
ORIGINAL_AUDIT = "e4aa84077ff66a3173bb047aa3c6452c9bc123b1"
RECALL_DROP_PP = 2


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _protection(rows, *, certified):
    strata = defaultdict(list)
    for row in rows:
        strata[row["seed"], row["arm"], row["head"], row["interpretation"]].append(row)
    result = []
    for (seed, arm, head, interpretation), items in strata.items():
        measured = {row["class_id"]: float(row["delta_recall_pp"]) for row in items if int(row["support"])}
        unsupported = [row["class_id"] for row in items if not int(row["support"])]
        drops = [c for c, value in measured.items() if value < -RECALL_DROP_PP]
        eligible = certified and arm not in ("none", "important")
        if eligible:
            status = "exceeds_limit" if drops else "partial_support" if unsupported else "within_limit"
        else:
            status = "reference" if arm in ("none", "important") else "diagnostic_only"
        result.append(dict(seed=int(seed), arm=arm, head=head, population=interpretation,
                           reference="important" if certified else "none",
                           points=sum(int(row["support"]) for row in items),
                           supported_classes=len(measured), unsupported_classes=",".join(unsupported),
                           worst_recall_delta_pp=min(measured.values(), default=None),
                           classes_beyond_drop_limit=",".join(drops), status=status))
    return result


def _scope(root):
    graphs = _rows(root / "graph-and-cost.csv")
    audit.check(len(graphs) == 306 and all(
        int(row["sparse_calls"]) == 1 and int(row["nodes"]) == 16 and int(row["original_nnz"]) == 100
        and int(row["effective_nnz"]) == (100 if row["arm"] == "none" else 84)
        for row in graphs), "Expected single observed native sparse multiply per block and arm")
    subsets = _rows(root / "strict-subset-heads.csv")
    selected = [row for row in subsets if row["head"] == "fused" and row["arm"] == "important"]
    denominators = [dict(seed=int(row["seed"]), total_contexts=17, strict_contexts=int(row["contexts"]),
                         diagnostic_contexts=17 - int(row["contexts"]),
                         all_development_points=45, strict_intersection_points=int(row["support"]))
                    for row in selected]
    audit.check([(r["seed"], r["strict_contexts"], r["strict_intersection_points"]) for r in denominators]
                == [(64, 6, 21), (512, 7, 21), (2048, 12, 32)], "Strict enrolled population changed")
    return dict(contexts=51, strict_contexts=25, diagnostic_contexts=26, by_seed=denominators,
                native_stage_two_sparse_calls=len(graphs), original_nnz_per_graph=100,
                deleted_nnz_per_graph=84)


def publish(args):
    code = code_snapshot()
    require_frozen_code(code)
    audit.check(sys.platform == "linux" and code["commit"] == args.expected_commit,
                "Clean expected Linux evidence checkout required")
    audit.check(args.source_sha256 == SOURCE, "Use the approved archived capture digest")
    raw = audit.read(args.source, SOURCE)
    audit.check(raw["commit"] == PRODUCER and raw["policy"] == "wp06_wp08_pavia_stage2_matched_deletion_v1",
                "Original PaviaU capture producer or policy differs")
    registration = audit.read(args.registration, audit.APPROVED)
    out = external_directory(args.out)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
        stage = Path(temporary) / "evidence"
        legacy = audit.export(raw, registration, stage, code["commit"], SOURCE)
        scope = _scope(stage)
        protection = _protection(_rows(stage / "strict-subset-classes.csv"), certified=True)
        protection += _protection(_rows(stage / "per-class.csv"), certified=False)
        table = "recall-protection.csv"
        digest = audit.write_table(stage, table, protection)
        matching = math.fsum(row["matching_seconds"] for row in raw["records"])
        solver = math.fsum(row["solver_seconds"] for row in raw["records"])
        audit.check(0 <= solver <= matching <= raw["wall_seconds"], "Archived matching timings differ")
        result = dict(policy="wp12_pavia_matched_deletion_v1", consumer_commit=code["commit"],
                      producer_commit=PRODUCER, original_audit_commit=ORIGINAL_AUDIT, source_sha256=SOURCE,
                      registration_sha256=audit.APPROVED, scene="paviaU", role="devval",
                      scope=scope, recall_drop_limit_pp=RECALL_DROP_PP, independent_confirmation=False,
                      original_cost_scope="RM Vision CPU archived single-thread frozen forwards",
                      archived_cost=dict(wall_seconds=raw["wall_seconds"], peak_rss_bytes=raw["peak_rss_bytes"],
                                         matching_seconds=matching, solver_seconds=solver),
                      new_forward_calls=0, optimizer_updates=0, query_accessed=False,
                      tables={**legacy["tables"], table: dict(sha256=digest, rows=len(protection)),
                              "receipt.json": dict(sha256=hash_file(stage / "receipt.json"))})
        (stage / "wp12-evidence.json").write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        require_frozen_code(code)
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return result


def main(argv):
    if argv and argv[0] == "capture":
        # Only the original standalone entry may import the pinned native checkout first.
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_pavia_matched_deletion.py"
        subprocess.run([sys.executable, "-B", str(script), *argv[1:]], check=True)
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Run capture --help for the pinned read-only producer.")
    parser.add_argument("action", choices=("publish",))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    for name in ("source", "registration", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = publish(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True))

"""Full54 independent-audit freeze or final matched query; never used for training decisions."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from evaluation.g4_target import predict_query, query_metrics
from g4_matched_target_audit import check_files, read
from g4_matched_target_evaluation import freeze_for_query, new_output, paired_summary, read_query_labels, require_frozen_boundary
from g4_matched_target_execution import case_data, read_registration, unchanged_producer
from g4_matched_target_study import runtime
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.research_config import digest
from research.reproduction.m4g4b.none_source_campaign import wait_gpu
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_initialization import state_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit", required=True)
    sub = p.add_subparsers(dest="operation", required=True)
    freeze = sub.add_parser("freeze", help="No query labels: authenticate completed full training and independent audit")
    for name in ("run", "launch-sha256", "acceptance-sha256", "audit", "audit-sha256", "out"):
        freeze.add_argument("--" + name, required=True)
    query = sub.add_parser("query", help="Final all-model-frozen query only")
    for name in ("freeze", "freeze-sha256", "data-root", "prepared-manifest", "prepared-sha256", "out"):
        query.add_argument("--" + name, required=True)
    query.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote evaluation commit required")
    if args.operation == "freeze":
        freeze_for_query(args.run, args.launch_sha256, args.acceptance_sha256, args.audit, args.audit_sha256, args.out)
        return
    if file_hash(args.freeze) != args.freeze_sha256:
        raise ValueError("Pinned all54 final query boundary required")
    boundary = read(args.freeze)
    require_frozen_boundary(boundary)
    unchanged_producer(boundary["code"], code)
    root = Path(boundary["training_root"])
    launch = read(root / "launch.json")
    registration = read_registration(launch["registration_path"], launch["registration_sha256"], launch["code"])
    if file_hash(registration["analysis_plan"]["path"]) != registration["analysis_plan"]["sha256"]:
        raise ValueError("Preregistered query analysis plan changed")
    out = external_directory(args.out)
    freeze_root = Path(args.freeze).resolve().parent
    new_output(out, launch, (freeze_root, Path(args.prepared_manifest).resolve().parent))
    if any(out == old or old in out.parents or out in old.parents for old in (root, freeze_root)):
        raise ValueError("New independent query output required")
    out.mkdir(exist_ok=False)
    rows, started = [], time.monotonic()
    for case in launch["cases"]:
        wait_gpu(args.gpu, out)
        require_frozen_boundary(boundary, full_integrity=False)
        items, _, guard = case_data(case)
        moved = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in value.items()} for n, value in items.items()}
        adapter, folder = runtime(case, items), Path(case["out"])
        frozen = next(m for m in boundary["models"] if m["name"] == case["name"])
        model = adapter.load_selected(folder / "primary.pt", moved,
            folder / f'raw-validation-{frozen["selected_step"]:06d}.pt', device="cuda")
        before = state_id(model)
        release, label_pins = read_query_labels(boundary, case, args.data_root, args.prepared_manifest, args.prepared_sha256)
        with OrderedValidation():
            probabilities = predict_query(model, moved[case["target"]], release["indices"])
        metrics, branches = query_metrics(release["labels"], probabilities)
        guard()
        check_files(label_pins)
        if state_id(model) != before:
            raise ValueError("Final query inference mutated frozen model")
        row = {**{k: case[k] for k in ("name", "target", "seed", "method", "mode")}, "metrics": metrics,
               "branches": branches, "release_sha256": digest(release), "checkpoint_sha256": frozen["primary_sha256"]}
        rows.append(row)
        write_json(out / (case["name"] + ".json"), row)
        del model, moved, items
        torch.cuda.empty_cache()
    require_frozen_code(code)
    require_frozen_boundary(boundary)
    if file_hash(args.freeze) != args.freeze_sha256:
        raise ValueError("Final query boundary file changed")
    write_json(out / "receipt.json", {"status": "passed", "policy": boundary["policy"], "code": code,
        "freeze_sha256": args.freeze_sha256, "models": rows, "paired_summary": paired_summary(rows),
        "query_labels_accessed": True, "optimizer_updates": 0, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

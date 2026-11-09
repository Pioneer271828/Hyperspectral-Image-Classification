"""Independent full54 selected-model audit; updates0/query0, fresh sibling output."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from g4_matched_target_audit import AUDIT_POLICY, audit_case, check_files, files, full_acceptance
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from research.reproduction.m4g4b.none_source_campaign import wait_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "run", "launch-sha256", "acceptance-sha256", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected independent remote commit required")
    root, out = external_directory(args.run), external_directory(args.out)
    if out.parent != root.parent or out == root:
        raise ValueError("New sibling audit output required")
    launch, accepted, pins = full_acceptance(root, args.launch_sha256, args.acceptance_sha256, code)
    pins.update(files(root))
    out.mkdir(exist_ok=False)
    rows, started = [], time.monotonic()
    for case in launch["cases"]:
        wait_gpu(args.gpu, out)
        row = audit_case(case, launch["code"], args.launch_sha256, launch["registration_sha256"])
        original = next(r for r in accepted["models"] if r["name"] == row["name"])
        if any(row[k] != original[k] for k in row):
            raise ValueError("Independent full target evidence differs from training acceptance")
        rows.append(row)
        write_json(out / "progress.json", rows)
        torch.cuda.empty_cache()
    require_frozen_code(code)
    check_files(pins)
    write_json(out / "receipt.json", {"policy": AUDIT_POLICY, "status": "passed", "code": code,
        "producer": launch["code"], "launch_sha256": args.launch_sha256, "acceptance_sha256": args.acceptance_sha256,
        "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False, "cases": rows,
        "gpu": args.gpu,
        "protected_files_and_inputs": len(pins), "protected_manifest_sha256": file_hash(root / "protected-files.json"),
        "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

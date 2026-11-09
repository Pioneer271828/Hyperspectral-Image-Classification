"""Freeze every role, graph and shared initial state before any optimizer update."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_none_source_launch import prepare_launch
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "reference-launch", "reference-launch-sha256", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--profile", choices=("short", "full"), required=True)
    a = p.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != a.expected_commit:
        raise RuntimeError("Clean remote expected commit required")
    launch = prepare_launch(a.reference_launch, a.reference_launch_sha256, a.out_dir, code, full=a.profile == "full")
    print(json.dumps({"launch_sha256": file_hash(Path(launch["out_dir"]) / "launch.json"),
                      "expected_updates": launch["expected_updates"], "optimizer_updates": 0,
                      "query_labels_accessed": False}), flush=True)

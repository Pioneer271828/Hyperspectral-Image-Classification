"""Register, execute and review the Issue 184 frozen probes across three scenes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from evaluation import allscene_probe_campaign as campaign, allscene_probe_registry as registry
from evaluation.frozen_probe_registry import ANALYSIS_SHA, FREEZE_SHA, pinned_json
from research_runtime import code_snapshot, hash_file, require_frozen_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    run = actions.add_parser("run")
    accept = actions.add_parser("accept")
    for action in (prepare, run, accept):
        action.add_argument("--expected-commit", required=True)
        action.add_argument("--out", type=Path, required=True)
    for name in ("freeze", "analysis", "capture-index", "capture-index-sha256",
                 "pavia-scope", "pavia-execution", "pavia-review"):
        prepare.add_argument("--"+name, required=True)
    prepare.add_argument("--missing-capture-scene", choices=("xuzhou", "whu-honghu"), action="append", default=[])
    for action in (run, accept):
        action.add_argument("--scope", type=Path, required=True)
        action.add_argument("--scope-sha256", required=True)
    accept.add_argument("--execution", type=Path, required=True)
    accept.add_argument("--execution-sha256", required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux consumer required")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.action == "prepare":
        paths = (args.pavia_scope, args.pavia_execution, args.pavia_review)
        path = registry.prepare(args.freeze, FREEZE_SHA, args.analysis, ANALYSIS_SHA,
            args.capture_index, args.capture_index_sha256, paths, args.out, code, missing=args.missing_capture_scene)
        status = "registered"
    elif args.action == "run":
        path = campaign.run(args.scope, args.scope_sha256, args.out, code)
        status = pinned_json(path, hash_file(path))["status"]
    else:
        path = campaign.review(args.scope, args.scope_sha256, args.execution, args.execution_sha256, args.out, code)
        status = pinned_json(path, hash_file(path))["status"]
    print(json.dumps({"path": str(path), "sha256": hash_file(path), "status": status}))
    return int(status == "incomplete_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())

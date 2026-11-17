"""Register, run or independently review the accepted PaviaU frozen probe matrix."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from evaluation import frozen_probe_campaign as campaign, frozen_probe_registry as registry
from research_runtime import code_snapshot, hash_file, require_frozen_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    run = modes.add_parser("run")
    accept = modes.add_parser("accept")
    for mode in (prepare, run, accept):
        mode.add_argument("--expected-commit", required=True)
        mode.add_argument("--out", type=Path, required=True)
    for name in ("freeze", "analysis", "capture-index", "capture-index-sha256"):
        prepare.add_argument("--"+name, required=True)
    for mode in (run, accept):
        mode.add_argument("--scope", type=Path, required=True)
        mode.add_argument("--scope-sha256", required=True)
    accept.add_argument("--execution", type=Path, required=True)
    accept.add_argument("--execution-sha256", required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux consumer required")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.mode == "prepare":
        path = registry.prepare(args.freeze, registry.FREEZE_SHA, args.analysis, registry.ANALYSIS_SHA,
                                args.capture_index, args.capture_index_sha256, args.out, code)
        status = "registered"
    elif args.mode == "run":
        path = campaign.run(args.scope, args.scope_sha256, args.out, code)
        status = registry.pinned_json(path, hash_file(path))["status"]
    else:
        path = campaign.review(args.scope, args.scope_sha256, args.execution, args.execution_sha256, args.out, code)
        status = registry.pinned_json(path, hash_file(path))["status"]
    print(json.dumps({"path": str(path), "sha256": hash_file(path), "status": status}))
    return int(status in ("incomplete_or_failed", "execution_failed"))


if __name__ == "__main__":
    raise SystemExit(main())

"""Register, execute and independently review the fixed initial/terminal pairs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from evaluation import initial_final_probe_campaign as campaign,initial_final_probe_registry as registry
from evaluation.frozen_probe_registry import pinned_json
from research_runtime import code_snapshot,hash_file,require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action",required=True)
    prep,run,accept = [actions.add_parser(name) for name in ("prepare","run","accept")]
    for action in (prep,run,accept):
        action.add_argument("--expected-commit",required=True)
        action.add_argument("--out",type=Path,required=True)
    for name in ("capture","capture-sha256","terminal-scope","terminal-execution","terminal-review"):
        prep.add_argument("--"+name,required=True)
    for action in (run,accept):
        action.add_argument("--scope",type=Path,required=True)
        action.add_argument("--scope-sha256",required=True)
    accept.add_argument("--execution",type=Path,required=True)
    accept.add_argument("--execution-sha256",required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or code["commit"] != args.expected_commit or not __debug__:
        raise ValueError("Exact Linux consumer required")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.action == "prepare":
        path = registry.prepare(args.capture,args.capture_sha256,
            (args.terminal_scope,args.terminal_execution,args.terminal_review),args.out,code)
        status = "registered"
    elif args.action == "run":
        path = campaign.run(args.scope,args.scope_sha256,args.out,code)
        status = pinned_json(path,hash_file(path))["status"]
    else:
        path = campaign.review(args.scope,args.scope_sha256,args.execution,args.execution_sha256,args.out,code)
        status = pinned_json(path,hash_file(path))["status"]
    print(json.dumps({"path":str(path),"sha256":hash_file(path),"status":status}))
    return int(status == "incomplete_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())

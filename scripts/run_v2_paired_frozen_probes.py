"""Register, execute and independently review both new full54 v2 probe stages."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from evaluation import v2_paired_probe_campaign as campaign, v2_paired_probe_registry as registry
from research_runtime import code_snapshot, hash_file, require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action",required=True)
    prep,run,review = [actions.add_parser(name) for name in ("prepare","run","review")]
    for sub in (prep,run,review):
        sub.add_argument("--expected-commit",required=True)
        sub.add_argument("--out",type=Path,required=True)
    names = ("v1-scope","v1-execution","v1-review","diagnosis","draft")
    for name in names:
        prep.add_argument("--"+name,type=Path,required=True)
    for sub in (run,review):
        sub.add_argument("--scope",type=Path,required=True)
        sub.add_argument("--scope-sha256",required=True)
    review.add_argument("--execution",type=Path,required=True)
    review.add_argument("--execution-sha256",required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform!="linux" or code["commit"]!=args.expected_commit or not __debug__:
        raise ValueError("Exact Linux consumer required")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.action=="prepare":
        result = registry.prepare(tuple(getattr(args,key.replace("-","_")) for key in names),args.out,code)
        status = "registered"
    elif args.action=="run":
        result = campaign.run(args.scope,args.scope_sha256,args.out,code)
        status = json.loads(result.read_text())["status"]
    else:
        result = campaign.review(args.scope,args.scope_sha256,args.execution,args.execution_sha256,args.out,code)
        status = json.loads(result.read_text())["status"]
    print(json.dumps({"path":str(result),"sha256":hash_file(result),"status":status}))
    return int(status=="incomplete_or_failed")


if __name__=="__main__":
    raise SystemExit(main())

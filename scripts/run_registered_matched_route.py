"""Admit registered dynamic donors, review frozen controls and audit synthetic recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer import matched_route_admission as admission, registered_matched_route as pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action",required=True)
    inv,prep,review,run,audit = [actions.add_parser(name) for name in
                                ("inventory","prepare","review","run","audit-recovery")]
    for command in (inv,prep,review,run,audit):
        command.add_argument("--expected-commit",required=True)
        command.add_argument("--out",type=Path,required=True)
    for command in (inv,prep):
        command.add_argument("--catalog",type=Path,required=True)
        command.add_argument("--catalog-sha256",required=True)
        command.add_argument("--allow-synthetic",action="store_true")
    prep.add_argument("--candidate",required=True)
    prep.add_argument("--per-row",type=int,default=1)
    prep.add_argument("--distance-bins",type=float,nargs="+",default=[2.,4.,100.])
    prep.add_argument("--time-limit",type=float,default=30.)
    for command in (review,run,audit):
        command.add_argument("--scope",type=Path,required=True)
        command.add_argument("--scope-sha256",required=True)
    run.add_argument("--arm",choices=pipeline.ARMS,required=True)
    run.add_argument("--segment",choices=pipeline.SEGMENTS,required=True)
    run.add_argument("--resume-attempt",type=Path)
    run.add_argument("--resume-attempt-sha256")
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"]!=args.expected_commit:
        parser.error("Clean exact consumer commit required")
    require_frozen_code(code)
    if args.action in ("inventory","prepare"):
        catalog = {"path":str(args.catalog.resolve(strict=True)),"sha256":args.catalog_sha256}
        if args.action=="inventory":
            result = pipeline.catalog_inventory(catalog,args.out,code,allow_synthetic=args.allow_synthetic)
        else:
            settings = {**pipeline.SETTINGS,"per_row":args.per_row,"distance_bins":args.distance_bins,"time_limit":args.time_limit}
            result = pipeline.prepare(catalog,args.candidate,args.out,code,allow_synthetic=args.allow_synthetic,settings=settings)
    else:
        scope = {"path":str(args.scope.resolve(strict=True)),"sha256":args.scope_sha256}
        if args.action=="review":
            result = pipeline.frozen_review(scope,args.out,code)
        elif args.action=="audit-recovery":
            result = pipeline.audit_recovery(scope,args.out,code)
        else:
            if (args.resume_attempt is None)!=(args.resume_attempt_sha256 is None):
                parser.error("Resume attempt and SHA256 must be paired")
            resume = None if args.resume_attempt is None else {
                "path":str(args.resume_attempt.resolve(strict=True)),"sha256":args.resume_attempt_sha256}
            result = pipeline.execute(scope,args.arm,args.segment,args.out,code,resume=resume)
    status = admission.read(result).get("status","inventoried")
    print(json.dumps({**result,"status":status}))
    return int(status in ("failed","incomplete_or_failed"))


if __name__=="__main__":
    raise SystemExit(main())

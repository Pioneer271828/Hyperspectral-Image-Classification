"""Prepare, review, and prove isolated synthetic light-adaptation queues."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g4_light_engineering_registration as eng
from g4_light_engineering_review import evidence_proofs, readiness, verify_queue
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code, write_json


def pin_arg(values):
    return {"path":str(external_directory(values[0])),"sha256":values[1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    commands=parser.add_subparsers(dest="step",required=True)
    create=commands.add_parser("fixture")
    register=commands.add_parser("register")
    review=commands.add_parser("review")
    proofs=commands.add_parser("proofs")
    ready=commands.add_parser("readiness")
    for command in (create,register,review,proofs,ready):
        command.add_argument("--out",required=True)
    register.add_argument("--fixture",nargs=2,required=True)
    register.add_argument("--campaign-out",required=True)
    register.add_argument("--engineering-issue",type=int,required=True)
    register.add_argument("--profile",choices=("representative","all"),default="representative")
    hardware=register.add_mutually_exclusive_group(required=True)
    hardware.add_argument("--cpu-slots",type=int,choices=(2,3))
    hardware.add_argument("--gpu-slot",nargs=2,action="append",metavar=("INDEX","UUID"))
    review.add_argument("--registration",nargs=2,required=True)
    proofs.add_argument("--review",nargs=2,action="append",required=True)
    for field in ("worker","campaign","inventory"):
        ready.add_argument("--"+field,nargs=2,required=True)
    args=parser.parse_args()
    code=code_snapshot()
    if sys.platform!="linux" or not __debug__ or code["dirty"] or code["commit"]!=args.expected_commit:
        raise ValueError("Clean expected Linux engineering checkout required")
    out=external_directory(args.out)
    if out.exists():
        raise ValueError("New engineering output required")
    if args.step=="fixture":
        sys.path.insert(0,str(ROOT/"tests"))
        result=eng.fixture(out,code)
    elif args.step=="register":
        devices=([{"slot":f"cpu-{n}"} for n in range(args.cpu_slots)] if args.cpu_slots else
                 [{"slot":int(n),"uuid":uuid} for n,uuid in args.gpu_slot])
        result=eng.register(pin_arg(args.fixture),code,external_directory(args.campaign_out),out,
                            args.engineering_issue,devices,profile=args.profile)
    elif args.step=="review":
        value=verify_queue(pin_arg(args.registration),code)
        batch=Path(value["queue_status"]["path"]).parent
        if out==batch or batch in out.parents or out in batch.parents:
            raise ValueError("Review output must be independent of the campaign")
        write_json(out,value)
        result=eng.pin(out)
    elif args.step=="proofs":
        evidence_proofs([pin_arg(value) for value in args.review],code,out)
        result=eng.pin(out/"campaign-proof.json")
    else:
        value=readiness(pin_arg(args.worker),pin_arg(args.campaign),pin_arg(args.inventory),code)
        write_json(out,value)
        result=eng.pin(out)
    require_frozen_code(code)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()

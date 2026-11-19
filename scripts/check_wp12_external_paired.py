"""Exercise publication, independent capture and recovery of externally released paired heads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import wp12_external_paired as evidence
from evaluation.frozen_training_stages import file_pin, json_pin, require
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_native_released_points import arithmetic
from transfer import native_external_paired as paired
from transfer.xuzhou_multiscale_audit import _exact


def recovery(root,program,output,code):
    plan,_ = paired.load(root,program,code,test=True)
    proofs = {}
    for key,entry in plan["releases"].items():
        if not key.endswith("OneA"):
            continue
        request = entry["request"]
        seed = request["model_seeds"][0]
        arm = paired.ARMS[request["kind"]][0]
        _,spec,found,body = evidence.selected_record(root,program,code,key,seed,arm,test=True)
        _,_,reference,ref_body = evidence.selected_record(root,program,code,key,seed,arm,test=True,reference=True)
        require(body["history"]==[0,3,7,12] and ref_body["history"]==[0,12]
                and _exact(found,reference),"Two boundary replay differs from uninterrupted trajectory")
        proofs[key] = dict(history=body["history"],reference=ref_body["history"],numeric=spec["numeric"],
                           trajectory_identical=True,final=body["checkpoint"],control=ref_body["checkpoint"])
    require(len(proofs)==2,"Both scenes must prove the same-device continuation")
    return evidence.add_json(root,output,dict(schema="wp05_wp06_external_paired_recovery_v1",proofs=proofs))


def negative(root,program,output,code,fixture):
    plan,_ = paired.load(root,program,code,test=True)
    registered = {rec["request"]["kind"]:rec["request"] for rec in plan["releases"].values()}
    fixture_doc = json_pin(root,file_pin(fixture))
    outcomes = {}
    for kind, examples in fixture_doc["negative"].items():
        base = registered[kind]
        for name,pins in examples.items():
            try:
                paired.open_request_bank(root,program,{**base,**pins,"k":5,"support_seed":512,"spatial_repeat":"a"},test=True)
            except ValueError as error:
                require("coverage" in str(error),"Fixture rejected for the wrong admission check")
                outcomes[f"{kind}/{name}"] = str(error)
            else:
                raise ValueError("Accepted insufficient external support or spatial buffer")
        key = next(key for key,rec in plan["releases"].items() if rec["request"]["kind"]==kind)
        seed = plan["releases"][key]["request"]["model_seeds"][0]
        p = paired.ExternalPairedProvider(root,program,code,key,seed,paired.ARMS[kind][0],test=True)
        for name,callback in (
                ("fit_evaluation",lambda: p.materialize("devtrain",dict(tile=p.roles["evaluation"][0],factor=1,
                                                                      band="full"),p.producer)),
                ("select_evaluation",lambda: p.materialize("devval",dict(tile=p.roles["evaluation"][0],factor=1,
                                                                         band="full"),p.producer)),
                ("evaluate_support",lambda: p.consume(p.roles["support"][0],"full",use="independent_accuracy")),
                ("evaluate_calibration",lambda: p.consume(p.roles["calibration"][0],"full",use="independent_accuracy"))):
            try:
                callback()
            except ValueError as error:
                outcomes[f"{kind}/{name}"] = str(error)
            else:
                raise ValueError("Unauthorized role was consumed")
        require(p.bank.reads==0,"Role rejection opened a spectrum")
    return evidence.add_json(root,output,dict(schema="wp05_wp06_external_paired_negative_v1",outcomes=outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("fixture","publish","capture","review","recovery","negative"))
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--program",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--release")
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--capture",nargs=2)
    parser.add_argument("--fail-after",type=int)
    parser.add_argument("--test-publication",action="store_true")
    args = parser.parse_args()
    root = args.runs.resolve(strict=True)
    program = external_directory(args.program)
    out = external_directory(args.out) if args.out else None
    require(root in program.parents and (out is None or root in out.parents),"Evidence escapes the authorized runs")
    require((args.action=="publish")== (out is None),"An explicit evidence destination is required")
    require((args.fail_after is None or args.action=="publish") and (args.gpu is None or args.action in ("fixture","capture"))
            and (args.release is None or args.action=="capture"),"Irrelevant fixture, publication or device option")
    code = code_snapshot()
    require(code["commit"]==args.expected_commit,"Expected frozen source commit")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.action=="fixture":
        from transfer.native_external_paired_fixture import make

        require(args.test_publication,"Fixture generation requires the separate test-only scope")
        result = make(out,program,code,gpu=args.gpu)
    elif args.action=="publish":
        result = evidence.publish(root,program,code,test=args.test_publication,fail_after=args.fail_after)
    elif args.action=="capture":
        with arithmetic(root,args.gpu) as execution:
            result = evidence.capture(root,program,out,code,execution,test=args.test_publication,release=args.release)
    elif args.action=="review":
        require(args.capture,"Review requires an exact raw capture pin")
        result = evidence.review(root,dict(zip(("path","sha256"),args.capture,strict=True)),out,code)
    elif args.action=="recovery":
        require(args.test_publication,"Uninterrupted duplicates are fixture-only")
        result = recovery(root,program,out,code)
    else:
        require(args.test_publication and args.fixture,"Negative fixtures require their own published receipt")
        result = negative(root,program,out,code,args.fixture)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()

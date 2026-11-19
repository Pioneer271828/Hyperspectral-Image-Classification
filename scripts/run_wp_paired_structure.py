"""Run, resume, publish and confirm a preregistered native cross-model pair roster."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from evaluation import wp12_cross_backend as comparison
from evaluation.frozen_training_stages import file_pin, json_pin, require
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_native_long_coverage import invoke
from scripts.run_native_released_points import arithmetic
from transfer import wp_paired_structure as structure, wp_paired_structure_run as run
from transfer.p4_runtime_device import UUID


def pin(value):
    require(value is not None,"Provide the document path and SHA-256")
    return dict(zip(("path","sha256"),value,strict=True))


def queue(root,out,code,*,test,workers):
    plan,_ = structure.open_plan(root,out,code,test=test)
    jobs = defaultdict(list)
    for name,group in plan["groups"].items():
        jobs[group["device"] if group["device"]!="cpu" else "cpu/"+name].append(name)
    logs = out/"commands"
    logs.mkdir(exist_ok=True)

    def train(task):
        device, groups = task
        gpu = None if device.startswith("cpu/") else device
        receipts = {}
        for name in groups:
            result = invoke(ROOT,sys.executable,"scripts/run_wp_paired_structure.py",
                ["group","--runs",root,"--campaign",out,"--expected-commit",code["commit"],
                 "--group",name,*(["--synthetic"] if test else [])],logs/name,gpu)
            require(result["status"]=="complete","A paired group is not ready for publication; continue its durable path")
            receipts[name] = result["arms"]
        return receipts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(train,jobs.items()))
    return dict(status="complete",groups={k:v for part in parts for k,v in part.items()})


def marker(root,out,code,*,test):
    link = file_pin(out/"publication.json")
    body = json_pin(root,link)
    plan,registered = structure.open_plan(root,out,code,test=test)
    require(body["schema"]==run.PUBLICATION and body["producer"]==code["commit"] and body["campaign"]==registered
            and body["scope"]==plan["scope"] and set(body["releases"])==set(plan["releases"])
            and set(body["cases"])==set(plan["groups"]) and body["optimizer_updates"]==plan["optimizer_updates"]
            and body["query_accessed"] is body["raw_gt_loaded"] is False,"Foreign all-arm frozen publication")
    return body,link


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fixture","register","arm","group","queue","freeze","capture","review","draft"))
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--request",nargs=2)
    p.add_argument("--approval",nargs=2)
    p.add_argument("--synthetic",action="store_true")
    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--qualification",action="store_true")
    suite.add_argument("--scientific",action="store_true")
    p.add_argument("--devices",nargs="+",default=[])
    p.add_argument("--gpu")
    p.add_argument("--group")
    p.add_argument("--arm")
    p.add_argument("--until",type=int)
    p.add_argument("--reference",action="store_true")
    p.add_argument("--release")
    p.add_argument("--capture",nargs=3,action="append",metavar=("RELEASE","PATH","SHA256"))
    p.add_argument("--evidence",nargs=2)
    p.add_argument("--workers",type=int,default=3)
    a = p.parse_args(argv)
    root,campaign = a.runs.resolve(strict=True),external_directory(a.campaign)
    out = None if a.out is None else external_directory(a.out)
    code = code_snapshot()
    require(code["commit"]==a.expected_commit and root in campaign.parents
            and (out is None or root in out.parents),"Use the exact clean checkout and authorized output root")
    require_frozen_code(code)
    require(1<=a.workers<=8 and (a.action=="fixture" or not (a.qualification or a.scientific or a.devices))
            and (a.action in ("arm","group","capture") or a.gpu is None),"Unexpected execution or device flags")
    torch.set_num_threads(1)
    if a.action=="fixture":
        from transfer.wp_paired_structure_fixture import generate, generate_scientific

        require(a.synthetic and out is not None and not out.exists() and out!=campaign
                and not a.request and not a.approval
                and 0<=len(a.devices)<=3 and len(set(a.devices))==len(a.devices)
                and all(UUID.fullmatch(v) for v in a.devices),"Explicit engineering fixture and admitted device identities required")
        value = (generate_scientific(out,campaign,code,a.devices) if a.scientific
                 else generate(out,campaign,code,a.devices,qualification=a.qualification))
    elif a.action=="register":
        value = structure.register(root,campaign,code,pin(a.request),pin(a.approval),test=a.synthetic)
    elif a.action in ("arm","group"):
        require(a.group and (a.action=="arm")==bool(a.arm) and (a.action=="arm" or not (a.until or a.reference)),
                "Register a group or exactly one arm and optional drain cursor")
        with arithmetic(root,a.gpu) as execution:
            if a.action=="arm":
                value = run.execute(root,campaign,code,a.group,a.arm,execution,test=a.synthetic,until=a.until,reference=a.reference)
            else:
                plan,_ = structure.open_plan(root,campaign,code,test=a.synthetic)
                require(a.group in plan["groups"], "Unknown registered physical group")
                arms = {}
                for arm in plan["groups"][a.group]["arms"]:
                    arms[arm] = run.execute(root,campaign,code,a.group,arm,execution,test=a.synthetic)
                    if arms[arm]["status"]!="complete":
                        break
                value = dict(status="complete" if all(r["status"]=="complete" for r in arms.values()) else "drained",arms=arms)
    elif a.action=="queue":
        value = queue(root,campaign,code,test=a.synthetic,workers=a.workers)
    elif a.action=="freeze":
        value = run.publication(root,campaign,code,test=a.synthetic)
    elif a.action=="capture":
        publication,_ = marker(root,campaign,code,test=a.synthetic)
        require(a.release in publication["releases"] and out is not None and not out.exists(),
                "Capture requires a fully frozen publication and one agreed release")
        with arithmetic(root,a.gpu) as execution:
            value = comparison.run_case(root,publication["releases"][a.release],out,code,execution,
                                        test_history=[] if a.synthetic else None)
    elif a.action=="draft":
        from evaluation.wp12_paired_structure_draft import draft

        require(out is not None and not out.exists() and a.evidence,
                "Provide a fresh draft path and the engineering acceptance receipt")
        value = draft(root,out,code,pin(a.evidence))
    else:
        from evaluation.wp12_paired_structure import review

        captures = a.capture or []
        require(out is not None and not out.exists() and len({key for key,_,_ in captures})==len(captures),
                "Independent consumer needs one capture per release and a new output directory")
        with arithmetic(root,None) as execution:
            value = review(root,campaign,out,code,{key:pin([path,sha]) for key,path,sha in captures},
                           execution,test=a.synthetic)
    require_frozen_code(code)
    print(json.dumps(value),flush=True)


if __name__=="__main__":
    main()

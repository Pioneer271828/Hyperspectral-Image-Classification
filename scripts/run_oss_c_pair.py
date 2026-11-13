"""Register or run a named, frozen short VCA cell in the authorized LXC."""

import argparse
from contextlib import contextmanager
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import _item_fingerprint, load_config
from joint_calibrated_selection import first_maximum
from joint_training import load_plan, options, run_training, schedule_for, validation_epochs
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts.audit_oss_a_history import PREPARATION, SCENES, digest as file_digest, read
from scripts.audit_oss_b_spectral import MANIFEST
from scripts.train_joint_m4 import graph_binding
from transfer.container_runtime import training_container_kind
from transfer.oss_c_vca import ARMS, GlobalReferenceRuntime, settings


SEEDS = (64,512,2048)
GPU = "GPU-b8dd997d-95e3-69ed-1952-e420e98bd236"
PLAN = Path("configs/experiments/base97_training.json")
BASE = Path("configs/experiments/base97_joint.json")
GATES = dict(primary="mean_macro_oa_delta_gt_0_and_at_least_2_positive_seeds",
    per_scene_aa_drop_max_pp=.5,honghu_oa_drop_max_pp=.5,small_class_training_support_at_most=25,
    all_small_class_mean_recall_drop_max_pp=2.,all_classes_reported=True,
    source="research/experiments/M2J2_VCA三模式完整配对预登记_20261104.md#事前评价",
    checkpoint_probability_max_abs=1e-5,checkpoint_rtol=1e-5,checkpoint_atol=1e-6)


def release(runs, *, materialize):
    path = runs/PREPARATION/"manifest.json"
    if path.is_symlink() or file_digest(path)!=MANIFEST:
        raise ValueError("Unexpected frozen train/validation manifest")
    manifest = read(path)
    if manifest["test_scoring"] is not False or tuple(manifest["datasets"])!=SCENES:
        raise ValueError("Frozen train/validation scene roles differ")
    items, sources = {}, {}
    for scene in SCENES:
        entry = manifest["datasets"][scene]
        payload = path.parent/(scene+"-prepared.pt")
        if payload.is_symlink() or entry["prepared_file"]!=payload.name or file_digest(payload)!=entry["prepared_sha256"]:
            raise ValueError("Frozen preparation payload changed")
        sources[scene] = dict(name=payload.name,sha256=entry["prepared_sha256"],
                              producer_fingerprint=digest(entry["prepared_fields"]))
        if materialize:
            item = torch.load(payload,map_location="cpu",weights_only=True,mmap=True)
            if (any(key.startswith(("test_","query_")) for key in item)
                    or _item_fingerprint(item)!=entry["prepared_fields"]):
                raise ValueError("Unexpected prepared identity or held-out fields")
            train,val = (item[role+"_samples_gt"]>0 for role in ("train","val"))
            if not train.any() or not val.any() or (train&val).any():
                raise ValueError("Train/validation roles must be nonempty and disjoint")
            items[scene] = item
    return items,sources


def campaign(runs,source):
    _,sources = release(runs,materialize=False)
    base,plan = load_config(BASE),load_plan(PLAN)
    cases = {}
    for arm,seed in product(ARMS,SEEDS):
        opts = settings(options(plan,"engineering",seed,base),arm)
        cases[f"{arm}-{seed}"] = dict(arm=arm,seed=seed,settings=opts,settings_sha256=digest(opts),
            schedule_sha256=digest(schedule_for(opts)),updates=12,
            validation_steps=[(epoch+1)*3 for epoch in validation_epochs(opts)])
    return dict(schema="oss-c-fixed-coordinate-real-v1",issue=463,
        source_commit=source["commit"],source_hash=source["source_hash"],config_sha256=file_digest(BASE),
        plan_sha256=file_digest(PLAN),data_manifest=MANIFEST,data_roles=["train","val"],
        split_root=64,data=sources,cases=cases,updates=108,gates=GATES,gpu_uuid=GPU,
        production_authorized=False,real_launch_authorized=False,heldout_reads=0)


def run_case(registered,key,items,source,directory,*,resume=False,device="cpu",stop_after=None):
    spec = registered["cases"][key]
    opts = spec["settings"]
    if spec["settings_sha256"]!=digest(opts) or spec["schedule_sha256"]!=digest(schedule_for(opts)):
        raise ValueError("Case settings or optimizer task schedule changed")
    runtime = GlobalReferenceRuntime(spec["arm"])
    prov = dict(campaign_sha256=digest(registered),case=key,
                m4_graph_binding=graph_binding(items),prepared=registered["data_manifest"])
    result = run_training(opts,items,directory,prov,source,resume=resume,device=device,
                          runtime_adapter=runtime,stop_after=stop_after,
                          finish_guard=lambda:require_frozen_code(source))
    if result["operation"]=="completed":
        receipt = result["result"]
        events = read(Path(directory)/"history.json")
        if ([event["step"] for event in events]!=spec["validation_steps"]
                or first_maximum(events)["step"]!=receipt["selected_step"]
                or receipt["selected_probability_max_abs_error"]>GATES["checkpoint_probability_max_abs"]):
            raise ValueError("Completed cell violates original selection or numerical contract")
    return result


def authorized_launch(path,pin,registered,key):
    if path is None or pin is None or path.is_symlink() or file_digest(path)!=pin:
        raise ValueError("An independently issued frozen launch receipt is required")
    value = read(path)
    if value!=dict(schema="oss-c-explicit-launch-v1",campaign_sha256=digest(registered),
                   source_commit=registered["source_commit"],case=key,updates=12,
                   gpu_uuid=registered["gpu_uuid"],scope="train_val_only",budget_notice_sent=True):
        raise ValueError("Launch receipt does not authorize the requested frozen cell")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=registered["gpu_uuid"] or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Bound GPU UUID and strict deterministic context required")


@contextmanager
def admitted_gpu(runs,uuid):
    path = runs/".p4_gpu_leases"/f"{uuid}.lock"
    with path.open("a") as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        output = subprocess.check_output(
            ["nvidia-smi","-i",uuid,"--query-gpu=uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],text=True).strip()
        actual,memory,utilization = (value.strip() for value in output.split(","))
        if actual!=uuid or int(memory)<12288 or int(utilization)>60:
            raise ValueError("Registered GPU does not satisfy the shared resource gate")
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    cmd = parser.add_subparsers(dest="mode",required=True)
    reg = cmd.add_parser("register")
    reg.add_argument("--out",type=Path,required=True)
    worker = cmd.add_parser("run")
    worker.add_argument("--registration",type=Path,required=True)
    worker.add_argument("--registration-sha256",required=True)
    worker.add_argument("--launch-receipt",type=Path,required=True)
    worker.add_argument("--launch-sha256",required=True)
    worker.add_argument("--case",required=True)
    worker.add_argument("--out",type=Path,required=True)
    worker.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    source = code_snapshot()
    if (not training_container_kind() or source["commit"]!=args.expected_commit):
        parser.error("Authorized LXC and expected source required")
    require_frozen_code(source)
    runs = args.runs.resolve(strict=True)
    out = args.out.resolve()
    if runs.name!="runs" or out==runs or not out.is_relative_to(runs):
        parser.error("Output must be inside the authorized runs tree")
    registered = campaign(runs,source)
    if args.mode=="register":
        with out.open("x",encoding="utf-8") as stream:
            json.dump(registered,stream,indent=2)
        print(json.dumps(dict(registration_sha256=file_digest(out),updates=108,real_updates=0)))
        return
    if (args.registration.is_symlink() or file_digest(args.registration)!=args.registration_sha256
            or read(args.registration)!=registered or args.case not in registered["cases"]):
        parser.error("Registered cell identity differs")
    if out != args.registration.resolve().parent/"cases"/args.case:
        parser.error("Use the one registered path for this arm and seed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    authorized_launch(args.launch_receipt,args.launch_sha256,registered,args.case)
    with admitted_gpu(runs,registered["gpu_uuid"]):
        items,_ = release(runs,materialize=True)
        result = run_case(registered,args.case,items,source,out,resume=args.resume,device="cuda:0")
    print(json.dumps({key:result[key] for key in ("operation","updates_this_call")}))


if __name__=="__main__":
    main()

"""Register or run a named, frozen short VCA cell in the authorized LXC."""

import argparse
from contextlib import contextmanager
import fcntl
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import _item_fingerprint, load_config
from joint_calibrated_selection import first_maximum
from joint_training import _tree_fingerprint, load_plan, options, run_training, schedule_for, validation_epochs
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, require_frozen_code
from scripts.audit_oss_a_history import PREPARATION, SCENES, digest as file_digest, read
from scripts.audit_oss_b_spectral import MANIFEST
from scripts.train_joint_m4 import graph_binding
from transfer.container_runtime import training_container_kind
from transfer.oss_c_vca import ARMS, GlobalReferenceRuntime, reference_masks, settings


SEEDS = (64,512,2048)
GPU = "GPU-b8dd997d-95e3-69ed-1952-e420e98bd236"
CLASSES = dict(zip(SCENES,(9,9,22),strict=True))
PLAN = Path("configs/experiments/base97_training.json")
BASE = Path("configs/experiments/base97_joint.json")
GATES = dict(primary="mean_macro_oa_delta_gt_0_and_at_least_2_positive_seeds",
    per_scene_aa_drop_max_pp=.5,honghu_oa_drop_max_pp=.5,small_class_training_support_at_most=25,
    all_small_class_mean_recall_drop_max_pp=2.,all_classes_reported=True,
    source="research/experiments/M2J2_VCA三模式完整配对预登记_20261104.md#事前评价",
    checkpoint_probability_max_abs=1e-5,checkpoint_rtol=1e-5,checkpoint_atol=1e-6)


def release(runs):
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
        item = torch.load(payload,map_location="cpu",weights_only=True,mmap=True)
        if (item["class_num"]!=CLASSES[scene] or any(key.startswith(("test_","query_")) for key in item)
                or _item_fingerprint(item)!=entry["prepared_fields"]):
            raise ValueError("Unexpected prepared identity or held-out fields")
        train,val = (item[role+"_samples_gt"]>0 for role in ("train","val"))
        if not train.any() or not val.any() or (train&val).any():
            raise ValueError("Train/validation roles must be nonempty and disjoint")
        items[scene] = item
    return items,sources


def campaign(runs,source, *, loaded=None):
    items,sources = release(runs) if loaded is None else loaded
    metadata = {}
    for scene,item in items.items():
        wave = torch.as_tensor(item["wavelengths"])
        valid = torch.ones_like(wave,dtype=torch.bool)
        candidate,control = reference_masks(wave,valid,400.,2500.)
        metadata[scene] = dict(class_count=item["class_num"],all_bands=len(wave),
            removed_candidate=(~candidate).nonzero().flatten().tolist(),
            removed_control=(~control).nonzero().flatten().tolist(),
            support={role:torch.bincount(item[role+"_samples_gt"].long().flatten(),
                                        minlength=item["class_num"]+1)[1:].tolist() for role in ("train","val")})
    del items
    base,plan = load_config(BASE),load_plan(PLAN)
    cases = {}
    initializations = {}
    for arm,seed in product(ARMS,SEEDS):
        opts = settings(options(plan,"engineering",seed,base),arm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = GlobalReferenceRuntime(arm).build_model(ResearchConfig.from_dict(opts["config"]),CLASSES)
        initial = _tree_fingerprint(model.state_dict())
        if initializations.setdefault(seed,initial)!=initial:
            raise ValueError("Shared paired initialization differs")
        cases[f"{arm}-{seed}"] = dict(arm=arm,seed=seed,settings=opts,settings_sha256=digest(opts),
            initial_state_sha256=initial, optimizer="joint Adam over the original backbone and three task heads",
            scheduler="ReduceLROnPlateau on calibrated fused validation NLL",
            optimizer_sha256=digest({"implementation":"joint_training.run_training","settings":opts["config"]["training"]}),
            scheduler_sha256=digest({"implementation":"ReduceLROnPlateau","settings":opts["scheduler"],
                                     "policy":opts["validation_policy"]}),
            schedule_sha256=digest(schedule_for(opts)),updates=12,
            validation_steps=[(epoch+1)*3 for epoch in validation_epochs(opts)])
    return dict(schema="oss-c-fixed-coordinate-real-v1",issue=463,
        source_commit=source["commit"],source_hash=source["source_hash"],config_sha256=file_digest(BASE),
        plan_sha256=file_digest(PLAN),data_manifest=MANIFEST,data_roles=["train","val"],
        split_root=64,data=sources,metadata=metadata,cases=cases,updates=108,gates=GATES,gpu_uuid=GPU,
        gpu_policy=dict(min_free_mib=12288,max_utilization=60,device_assignment="one registered UUID"),
        production_authorized=False,real_launch_authorized=False,heldout_reads=0)


def run_case(registered,key,items,source,directory,*,resume=False,device="cpu",stop_after=None):
    spec = registered["cases"][key]
    opts = spec["settings"]
    if spec["settings_sha256"]!=digest(opts) or spec["schedule_sha256"]!=digest(schedule_for(opts)):
        raise ValueError("Case settings or optimizer task schedule changed")
    runtime = GlobalReferenceRuntime(spec["arm"])
    if "initial_state_sha256" in spec:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec["seed"])
            model = runtime.build_model(ResearchConfig.from_dict(opts["config"]),
                                        {name:item["class_num"] for name,item in items.items()})
        if _tree_fingerprint(model.state_dict())!=spec["initial_state_sha256"]:
            raise ValueError("Bound initial model state differs")
    prov = dict(campaign_sha256=digest(registered),case=key,
                m4_graph_binding=graph_binding(items),prepared=registered["data_manifest"])
    result = run_training(opts,items,directory,prov,source,resume=resume,device=device,
                          runtime_adapter=runtime,stop_after=stop_after,
                          finish_guard=lambda:require_frozen_code(source))
    if result["operation"] in ("completed","verified_completed"):
        receipt = result["result"]
        events = read(Path(directory)/"history.json")
        if ([event["step"] for event in events]!=spec["validation_steps"]
                or first_maximum(events)["step"]!=receipt["selected_step"]
                or receipt["selected_probability_max_abs_error"]>GATES["checkpoint_probability_max_abs"]):
            raise ValueError("Completed cell violates original selection or numerical contract")
    return result


def score_pairs(registered,case_root):
    """Apply the M2-J2 efficacy rule to selected validation metrics only."""
    values = {}
    for key,spec in registered["cases"].items():
        case = case_root/key
        result = read(case/"result.json")
        error = result["selected_probability_max_abs_error"]
        if (result["status"]!="completed" or result["optimizer_steps"]!=12
                or result["settings"]!=spec["settings"] or result["test_scoring"] is not False
                or result["identity"]["prepared"]["campaign_sha256"]!=digest(registered)
                or result["start_code"]["commit"]!=registered["source_commit"]
                or result["identity"]["prepared"]["case"]!=key
                or result["schedule_sha256"]!=spec["schedule_sha256"]
                or not math.isfinite(error) or not 0<=error<=GATES["checkpoint_probability_max_abs"]):
            raise ValueError("Comparison requires the original completed frozen cell")
        history = case/"history.json"
        if file_digest(history)!=result["artifacts"][history.name]:
            raise ValueError("Frozen cell history changed")
        events = read(history)
        if [row["step"] for row in events]!=spec["validation_steps"] or first_maximum(events)["step"]!=result["selected_step"]:
            raise ValueError("Frozen cell selection differs")
        metrics = result["selected_validation"]
        for scene in SCENES:
            for phase in ("train","val"):
                heads = metrics["metrics"][scene][phase]
                if set(heads)!={"fused","graph","cnn"}:
                    raise ValueError("Missing or unexpected comparison head")
                for row in heads.values():
                    size = CLASSES[scene]
                    if (len(row["recall"])!=size or len(row["confusion"])!=size or any(
                            len(line)!=size for line in row["confusion"]) or not all(
                            math.isfinite(x) and 0<=x<=1 for x in [row["oa"],row["aa"],*row["recall"]])):
                        raise ValueError("Incomplete registered class roster")
        values[spec["arm"],spec["seed"]] = metrics
    results = {}
    for reference in ("native","matched_boundary_dropout"):
        macro, honghu, aa, recalls = [],[],{scene:[] for scene in SCENES},{scene:[] for scene in SCENES}
        for seed in SEEDS:
            left,right = values["exclude_out_of_reference",seed],values[reference,seed]
            macro.append(100*(left["macro_oa"]-right["macro_oa"]))
            for scene in SCENES:
                a,b = (item["metrics"][scene]["val"]["fused"] for item in (left,right))
                aa[scene].append(100*(a["aa"]-b["aa"]))
                if scene=="whu-honghu":
                    honghu.append(100*(a["oa"]-b["oa"]))
                train = left["metrics"][scene]["train"]["fused"]["confusion"]
                other = right["metrics"][scene]["train"]["fused"]["confusion"]
                supports = [sum(row) for row in train]
                if supports!=[sum(row) for row in other] or supports!=registered["metadata"][scene]["support"]["train"]:
                    raise ValueError("Paired training class support differs")
                recalls[scene].append((supports,[100*(x-y) for x,y in zip(a["recall"],b["recall"],strict=True)]))
        mean = lambda series:sum(series)/len(series)
        by_scene = {}
        for scene,records in recalls.items():
            support = records[0][0]
            if any(row[0]!=support for row in records):
                raise ValueError("Training support changes across model seeds")
            by_scene[scene] = [dict(label=i+1,train_support=n,mean_delta_pp=mean([row[1][i] for row in records]))
                              for i,n in enumerate(support)]
        passed = (mean(macro)>0 and sum(delta>0 for delta in macro)>=2
                  and all(mean(diff)>=-.5 for diff in aa.values()) and mean(honghu)>=-.5
                  and all(row["mean_delta_pp"]>=-2. for rows in by_scene.values()
                          for row in rows if row["train_support"]<=25))
        results[reference] = dict(paired_macro_delta_pp=macro,mean_macro_delta_pp=mean(macro),
            per_scene_aa_mean_delta_pp={scene:mean(diff) for scene,diff in aa.items()},
            honghu_oa_mean_delta_pp=mean(honghu),all_class_recall_delta_pp=by_scene,development_gate_passed=passed)
    all_arms = {f"{arm}-{seed}":values[arm,seed] for arm in ARMS for seed in SEEDS}
    return dict(comparisons=results,all_arms=all_arms,heldout_reads=0,production_authorized=False)


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
    scoring = cmd.add_parser("score")
    scoring.add_argument("--registration",type=Path,required=True)
    scoring.add_argument("--registration-sha256",required=True)
    scoring.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    source = code_snapshot()
    if (not training_container_kind() or source["commit"]!=args.expected_commit):
        parser.error("Authorized LXC and expected source required")
    require_frozen_code(source)
    runs = args.runs.resolve(strict=True)
    out = args.out.resolve()
    if runs.name!="runs" or out==runs or not out.is_relative_to(runs):
        parser.error("Output must be inside the authorized runs tree")
    if args.mode=="register":
        registered = campaign(runs,source)
        with out.open("x",encoding="utf-8") as stream:
            json.dump(registered,stream,indent=2)
        print(json.dumps(dict(registration_sha256=file_digest(out),updates=108,real_updates=0)))
        return
    if (args.registration.is_symlink() or not args.registration.resolve().is_relative_to(runs)
            or file_digest(args.registration)!=args.registration_sha256):
        parser.error("Registered cell identity differs")
    registered = read(args.registration)
    if args.mode=="score":
        if registered!=campaign(runs,source):
            parser.error("Registered scoring identity differs")
        with out.open("x",encoding="utf-8") as stream:
            json.dump(score_pairs(registered,args.registration.resolve().parent/"cases"),stream,indent=2)
        return
    if args.case not in registered["cases"]:
        parser.error("Registered cell does not exist")
    if registered.get("gpu_uuid")!=GPU or registered.get("source_commit")!=source["commit"]:
        parser.error("Registered source or GPU does not match")
    if out != args.registration.resolve().parent/"cases"/args.case:
        parser.error("Use the one registered path for this arm and seed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    authorized_launch(args.launch_receipt,args.launch_sha256,registered,args.case)
    with admitted_gpu(runs,registered["gpu_uuid"]):
        loaded = release(runs)
        if registered!=campaign(runs,source,loaded=loaded):
            parser.error("Registered configuration, source or observed inputs differ")
        items,_ = loaded
        result = run_case(registered,args.case,items,source,out,resume=args.resume,device="cuda:0")
    print(json.dumps({key:result[key] for key in ("operation","updates_this_call")}))


if __name__=="__main__":
    main()

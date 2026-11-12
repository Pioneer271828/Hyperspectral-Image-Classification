"""Promote the pinned full54 stable-probe proposal to a once-only execution scope."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch

from evaluation import allscene_probe_registry as scenes, authenticated_probe_pack as terminal_pack
from evaluation import frozen_probe_registry as reg, initial_probe_pack as initial_pack
from evaluation import linear_probe, probe_numerical_diagnostics as history
from g4_matched_target_study import METHODS, MODES, case_name
from g4_support_context import check_files
from model.research_config import digest
from research_runtime import require_frozen_code, write_json


POLICY = "wp06_full54_paired_new_v2_probes_v1"
ROLES = ("initial","terminal")
DIAGNOSIS_SHA = "9d362b81ba24b348c08dcc7a621a6d079e159c367a9a631f9089d8d9fb2548f2"
DRAFT_SHA = "8b87b103719c1e4c531aca04b6fa82d653973df997894aa12ff42ffd26a7e151"
DIAGNOSIS_COMMIT = "7e008e43a0e6219b1def7c17b31a91b16c707f1d"
DRAFT_POLICY = "wp06_stable_standardization_matched_initial_terminal_plan_v2"
ROLE_KEYS = ("name","target","seed","method","mode","case_sha256","family_hash","phase_hashes",
             "released_rows_sha256","initial_step","terminal_step","initial_state_sha256",
             "terminal_model_sha256","observation_Q_sha256","regional_effective_dimensions")


@dataclass
class Evidence:
    sources: dict
    cases: list
    plans: dict
    paths: dict
    columns: dict
    pins: dict
    legacy: dict
    synthetic: bool = False


def source_pin(path, sha, pins):
    path = Path(path).resolve(strict=True)
    reg.require(pins.get(str(path))==sha, f"Source is outside the certified original files: {path}")
    return {"path":str(path),"sha256":sha}


def stage_source(stage, root, executed_sha, pins, proposal, role, old_proof, old_scope):
    folder = root/role["name"]
    execution = source_pin(folder/"execution.json",executed_sha,pins)
    archived = reg.pinned_json(Path(execution["path"]),execution["sha256"])
    reg.require(archived["case"]==role["name"] and archived["completed_resume_updates"]==0,
                "Original execution role differs")
    pack_file = folder/"pack.pt"
    pack = source_pin(pack_file,archived["pack_sha256"],pins)
    probability_path = folder/"selected-logp.pt"
    probabilities = source_pin(probability_path,archived["files"][str(probability_path)],pins)
    receipt_path = folder/"pack-receipt.json"
    receipt = source_pin(receipt_path,archived["files"][str(receipt_path)],pins)
    pack_receipt = reg.pinned_json(Path(receipt["path"]),receipt["sha256"])
    certified = old_proof["role"]
    capture = source_pin(certified["capture_receipt"],certified["capture_receipt_sha256"],pins)
    captured = reg.pinned_json(Path(capture["path"]),capture["sha256"])
    state = captured["evidence"]["state_sha256"]
    expected = {"case":role["name"], "capture_receipt_sha256":capture["sha256"],
                "freeze_sha256":old_scope["freeze_sha256"],"analysis_sha256":old_scope["analysis_sha256"],
                "family_hash":role["family_hash"],"phase_hashes":role["phase_hashes"],
                "query_labels_accessed":False,"label_authorization":"accepted_case_family_K10_V5_phases"}
    if stage=="initial":
        reg.require(captured["stage_role"]=="pre_adaptation" and captured["selected_step"]==role["initial_step"]==0
                    and state==captured["initial_state_sha256"]==role["initial_state_sha256"],
                    "Original pre-adaptation state or step differs")
        expected.update(initial_pack.contract(role))
    else:
        reg.require((certified["primary_sha256"],certified["selected_step"])
                    ==(role["terminal_model_sha256"],role["terminal_step"]), "Original selected checkpoint differs")
        expected.update(policy=terminal_pack.POLICY,selected_model_sha256=role["terminal_model_sha256"],
                        selected_step=role["terminal_step"])
    reg.require(all(certified[key]==role[key] for key in
                    ("target","seed","method","mode","case_sha256","family_hash","phase_hashes","released_rows_sha256"))
                and captured["case"]==role["name"] and captured["status"]=="passed"
                and captured["evidence"]["observation"]["Q"]==role["observation_Q_sha256"]
                and captured["evidence"]["registered_state_flags_rng_inputs_unchanged"] is True
                and pack_receipt["scope_sha256"]==archived["scope_sha256"]
                and pack_receipt["capture_receipt_sha256"]==capture["sha256"]
                and pack_receipt["pack_sha256"]==pack["sha256"]==archived["files"][pack["path"]]
                and pack_receipt["case"]==role["name"] and pack_receipt["policy"]==expected["policy"]
                and pack_receipt["query_labels_accessed"] is False and pack_receipt["backbone_updates"]==0
                and proposal=={"pack_path":pack["path"],"pack_sha256":pack["sha256"],
                    "feature_pack_fingerprint":pack_receipt["pack_identity"]}, "Original role/pack/Q authorization differs")
    return {"pack":pack,"capture":capture,"execution":execution,"legacy_probabilities":probabilities,
            "model_state_sha256":state,"pack_identity":expected,"fingerprint":proposal["feature_pack_fingerprint"]}


def authenticate(source_paths):
    scope_path,execution_path,review_path,diagnosis_path,draft_path = source_paths
    paths,old,execution,_,pins,legacy = history.load_v1(scope_path,execution_path,review_path)
    diagnosis_path,draft_path = (Path(path).resolve(strict=True) for path in (diagnosis_path,draft_path))
    diagnosis = reg.pinned_json(diagnosis_path,DIAGNOSIS_SHA)
    draft = reg.pinned_json(draft_path,DRAFT_SHA)
    counts = diagnosis["endpoint_counters"]
    reg.require(diagnosis["status"]=="zero_update_checks_passed" and diagnosis["policy"]==history.POLICY
                and diagnosis["commit"]==DIAGNOSIS_COMMIT and diagnosis["source_pins"]==history.V1_SHA
                and diagnosis["outputs"][str(draft_path)]==DRAFT_SHA and diagnosis["protected_files_unchanged"] is True
                and len(counts)==6 and all(row["endpoints"]==144 and row["v2_standardization_failures"]
                    ==row["v2_fixed_head_failures"]==0 for row in counts.values())
                and diagnosis["v2_real_fitting_performed"] is False
                and diagnosis["new_head_updates"]==diagnosis["new_model_forwards"]==diagnosis["backbone_updates"]
                    ==diagnosis["bn_updates"]==0
                and diagnosis["query_labels_accessed"] is diagnosis["raw_gt_deserialized"] is False,
                "Pinned real zero-update v2 evidence differs")
    sources = {key:{"path":str(path),"sha256":history.V1_SHA[key]} for key,path in paths.items()}
    sources.update(diagnosis={"path":str(diagnosis_path),"sha256":DIAGNOSIS_SHA},
                   draft={"path":str(draft_path),"sha256":DRAFT_SHA})
    pins = scenes.merge_pins(pins,diagnosis["outputs"],{str(diagnosis_path):DIAGNOSIS_SHA})
    plans = {scene:{**reg.PLAN,"policy":linear_probe.POLICY_V2,"classes":classes}
             for scene,classes in scenes.CLASSES.items()}
    columns = {stage:reg.probe_columns(64,64,width) for stage,width in (("stage1",128),("stage2",64))}
    reg.require(draft["policy"]==DRAFT_POLICY and draft["status"]=="planning_only"
        and draft["source_scope_sha256"]==history.V1_SHA["scope"] and draft["source_v1_review_sha256"]==history.V1_SHA["review"]
        and draft["paths"]==old["paths"]==reg.PATHS and draft["plans"]==plans and draft["columns"]==old["columns"]==columns
        and draft["layers"]==old["layers"]==list(reg.LAYERS) and draft["selection"]=="every_step_first_minimum_val_nll"
        and draft["planned_heads_per_backbone_stage"]==216 and draft["planned_total_new_probe_updates"]==432000
        and draft["planned_updates_per_backbone_stage"]=={"full":108000,"split":108000,"completed":0}
        and draft["planned_historical_head_reuse"]==draft["actual_new_probe_updates"]==0,
        "Planning-only draft controls or fixed v2 budget differ")
    terminal = history.terminal_index(old,pins)
    cases = []
    for proposal,case,executed in zip(draft["cases"],old["cases"],execution["cases"],strict=True):
        name = case["name"]
        reg.require((proposal["case"],proposal["scene"],executed["case"],executed["scene"])
                    ==(name,case["target"],name,case["target"]) and executed["status"]=="executed"
                    and all(proposal[key]==case[key] for key in ("seed","method","mode","initial_step","terminal_step"))
                    and legacy[name]["role"]==case, "Original ordered proposal/role differs")
        case = {key:case[key] for key in ROLE_KEYS}
        case["authorized_roles_fingerprints"] = proposal["authorized_roles_fingerprints"]
        case["stages"] = {}
        for stage,(root,sha) in zip(ROLES,((paths["execution"].parent,executed["execution_sha256"]),terminal[name]),strict=True):
            case["stages"][stage] = stage_source(stage,root,sha,pins,proposal[stage],case,legacy[name][stage],old)
        cases.append(case)
    return Evidence(sources,cases,plans,reg.PATHS,columns,pins,legacy)


def payload(evidence, code):
    expected = {(scene,seed,method,mode) for scene,seed,method,mode in
                product(scenes.CLASSES,reg.SEEDS,METHODS[:2],MODES)}
    steps = evidence.paths["full"][0]
    split = evidence.paths["split"]
    columns = {stage:reg.probe_columns(64,64,width) for stage,width in (("stage1",128),("stage2",64))}
    reg.require(len(evidence.cases)==54 and set(evidence.plans)==set(scenes.CLASSES)
        and {(r["target"],r["seed"],r["method"],r["mode"]) for r in evidence.cases}==expected
        and all(r["name"]==case_name(r["target"],r["seed"],r["method"],r["mode"])
                and set(r["stages"])==set(ROLES) and r["initial_step"]==0 and r["terminal_step"]>0 for r in evidence.cases)
        and len({r["stages"][stage]["pack"]["path"] for r in evidence.cases for stage in ROLES})==108
        and all(evidence.plans[scene]=={**reg.PLAN,"classes":classes,"steps":steps,"policy":linear_probe.POLICY_V2}
                for scene,classes in scenes.CLASSES.items())
        and evidence.columns==columns and set(evidence.paths)=={"full","split","completed"}
        and evidence.paths["full"]==evidence.paths["completed"]==[steps]
        and len(split)==3 and 0<split[0]<split[1]<split[2]==steps
        and (evidence.synthetic or (evidence.paths==reg.PATHS and steps==500)),
        "Exact v2 two-role matrix, plan and distinct archived packs required")
    updates = 54*len(reg.LAYERS)*steps
    return {"policy":POLICY,"commit":code["commit"],"source_hash":code["source_hash"],"synthetic":evidence.synthetic,
        "sources":evidence.sources,"cases":evidence.cases,"plans":evidence.plans,"paths":evidence.paths,
        "columns":evidence.columns,"layers":list(reg.LAYERS),"scene_classes":scenes.CLASSES,
        "protected_file_count":len(evidence.pins),"protected_files_sha256":digest(evidence.pins),
        "selection":"every_step_first_minimum_val_nll","planned_heads_per_stage":216,
        "planned_updates_per_stage":{"full":updates,"split":updates,"completed":0},
        "planned_total_updates":4*updates,"historical_head_reuse":0,
        "backbone_updates":0,"bn_updates":0,"query_labels_accessed":False}


def load_pack(case,stage,classes):
    source = case["stages"][stage]
    pin = source["pack"]
    path = Path(pin["path"])
    check_files({str(path):pin["sha256"]})
    pack = torch.load(path,map_location="cpu",weights_only=True)
    reg.require(set(pack)=={"features","labels","row_ids","identity"} and
        all(pack["identity"].get(key)==value for key,value in source["pack_identity"].items())
        and linear_probe.fingerprint(pack)==source["fingerprint"]
        and set(pack["features"])==set(reg.LAYERS)
        and set(pack["labels"])==set(pack["row_ids"])=={"train","val"},
        "Authenticated feature pack identity, stage or layer set differs")
    overlay = {}
    for role,count in (("train",10),("val",5)):
        labels,ids = pack["labels"][role],pack["row_ids"][role]
        reg.require(labels.dtype==ids.dtype==torch.int64 and labels.shape==ids.shape==(count*classes,)
            and bool(((labels>=0)&(labels<classes)).all()) and torch.bincount(labels,minlength=classes).tolist()==[count]*classes
            and bool((ids>=0).all()) and ids.unique().numel()==ids.numel(), "Original K10/V5 class or row support differs")
        overlay[role+"_labels"],overlay[role+"_indices"] = labels,ids
        for layer,features in pack["features"].items():
            width = 128 if layer.startswith("stage1") else 64
            reg.require(set(features)=={"train","val"} and features[role].dtype==torch.float32
                and features[role].shape==(count*classes,width) and bool(torch.isfinite(features[role]).all()),
                "Original FP32 four-layer feature shape or validity differs")
    reg.require(linear_probe.fingerprint(overlay)==case["released_rows_sha256"]
        and {key:linear_probe.fingerprint(pack[key]) for key in ("labels","row_ids")}
            ==case["authorized_roles_fingerprints"]
        and torch.cat(tuple(pack["row_ids"].values())).unique().numel()==15*classes,
        "Original ordered labels or row IDs differ")
    return pack


def preflight(scope):
    for case in scope["cases"]:
        for stage in ROLES:
            load_pack(case,stage,scope["scene_classes"][case["target"]])


def prepare(source_paths,out,code):
    require_frozen_code(code)
    evidence = authenticate(source_paths)
    scope = payload(evidence,code)
    check_files(evidence.pins)
    preflight(scope)
    require_frozen_code(code)
    out = history.fresh_output(out,{key:Path(item["path"]) for key,item in evidence.sources.items()},evidence.pins)
    path = out/"scope.json"
    write_json(path,scope)
    return path


def load(path,sha,code):
    require_frozen_code(code)
    scope = reg.pinned_json(path,sha)
    reg.require(scope["policy"]==POLICY and scope["commit"]==code["commit"] and scope["source_hash"]==code["source_hash"],
                "Frozen v2 consumer or protocol differs")
    order = ("scope","execution","review","diagnosis","draft")
    evidence = authenticate(tuple(scope["sources"][key]["path"] for key in order))
    reg.require(scope==payload(evidence,code),"Registered v2 roles, input hashes or budget changed")
    return scope,evidence

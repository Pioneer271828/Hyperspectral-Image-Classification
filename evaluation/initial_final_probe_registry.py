"""Freeze one initial/terminal pair for each accepted Issue 184 training role."""
from pathlib import Path
import time

from evaluation import allscene_probe_registry as scenes, frozen_probe_registry as reg
from evaluation import initial_probe_pack as initial, terminal_probe_history as terminal
from g4_support_context import check_files
from model.research_config import digest
from research_runtime import require_frozen_code,write_json


POLICY = "wp06_full54_pre_adaptation_terminal_probe_pairs_v1"
case_scope = scenes.case_scope


def authenticate(capture, capture_sha, terminal_paths, *, synthetic):
    started = time.monotonic()
    old_scope, launch, proofs, bounds, history, previous = terminal.authenticate(*terminal_paths,synthetic=synthetic)
    initial_launch,cases,new,source = initial.authenticate(capture,capture_sha,old_scope,bounds,synthetic=synthetic)
    reg.require(initial_launch == launch, "Initial and terminal launch differs")
    return old_scope,launch,cases,scenes.merge_pins(previous,new),proofs,history,source,time.monotonic()-started


def data(old_scope,cases,pins,history,source,code):
    cohort = [case for case,_,_ in cases]
    step = old_scope["paths"]["full"][0]
    planned = sum(c["disposition"]=="new" for c in cohort)*2*len(reg.LAYERS)*step
    return {**{k:old_scope[k] for k in ("freeze","freeze_sha256","analysis","analysis_sha256","paths",
            "probe_plans","layers","scene_classes","columns","validation","optimizer","synthetic")},
        "policy":POLICY,"commit":code["commit"],"source_hash":code["source_hash"],"initial_capture":source,
        "history":history,"cases":cohort,"reused_probe_updates":history["probe_updates"],
        "planned_new_probe_updates":planned,"expected_probe_updates":2*history["probe_updates"],
        "protected_file_count":len(pins),"protected_files_sha256":digest(pins),
        "backbone_updates":0,"bn_updates":0,"raw_gt_deserialized":False,"query_labels_accessed":False}


def output(out,launch,scope,*others):
    return scenes.new_output(out,launch,(Path(scope["freeze"]).parent,Path(scope["analysis"]).parent,
        Path(scope["initial_capture"]["path"]).parent,
        *(Path(file["path"]).parent for file in scope["history"]["sources"].values()),*others))


def prepare(capture,capture_sha,terminal_paths,out,code,*,synthetic=False):
    require_frozen_code(code)
    old,launch,cases,pins,_,history,source,elapsed = authenticate(capture,capture_sha,terminal_paths,synthetic=synthetic)
    scope = data(old,cases,pins,history,source,code)
    out = output(out,launch,scope)
    check_files(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    path = out/"scope.json"
    write_json(path,{**scope,"input_authentication_seconds":elapsed})
    return path


def load(scope_path,scope_sha,code):
    require_frozen_code(code)
    scope = reg.pinned_json(scope_path,scope_sha)
    reg.require((scope["policy"],scope["commit"],scope["source_hash"]) == (POLICY,code["commit"],code["source_hash"]),
                "Frozen initial/final consumer differs")
    capture = scope["initial_capture"]
    files = [scope["history"]["sources"][key]["path"] for key in ("scope","execution","review")]
    old,launch,cases,pins,reused,history,source,elapsed = authenticate(capture["path"],capture["sha256"],files,
                                                                     synthetic=scope["synthetic"])
    reg.require({key:value for key,value in scope.items() if key != "input_authentication_seconds"}
                == data(old,cases,pins,history,source,code), "Registered initial/final identities or budget changed")
    pins = scenes.merge_pins(pins,{str(Path(scope_path).resolve(strict=True)):scope_sha})
    return scope,launch,cases,pins,reused,elapsed

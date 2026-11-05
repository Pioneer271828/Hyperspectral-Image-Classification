"""Frozen source-only control identities and workers; no held-target payload reader."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

import torch
from g4_none_ordered_source import EXPECTED, NoneOrderedSourceRuntime, load_study, options
from joint_training import load_plan, run_training
from loadData.dataset_registry import file_hash
from m4_ordered_source import full_settings as active_settings
from model.research_config import ResearchConfig, digest
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code, write_json
from scripts.train_joint_m4 import graph_binding
from source_data_context import load_source_context
from target_initialization import state_id

STUDY_PATH = ROOT / "research/experiments/g4_none_ordered_source_full_v2.json"
REFERENCE_PRODUCER = "01c29c82e92ce0a529749c53962202e9bf0dc5fe"
REFERENCE_LAUNCH = "0ab7b356b0dd9c63aee67fb2e6b7aca147c63ec4fbb242e6d9c0089bbc63e260"


def initial_binding(settings, counts):
    """Fresh CPU construction also proves every none tensor matches both active initials."""
    seed = settings["model_seed"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ResearchConfig.from_dict(settings["config"]).build_model(counts)
    original = state_id(model)
    common = model.state_dict()
    peers = {}
    for mode in ("capacity_control", "x_residual"):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            peer = ResearchConfig.from_dict(active_settings(settings["held_out"], seed, mode)["config"]).build_model(counts)
        state = peer.state_dict()
        if any(key not in state or not torch.equal(value, state[key]) for key, value in common.items()):
            raise ValueError("None/active shared initialization differs")
        peers[mode] = original
        del peer
    return {"initial_state_sha256": original, "common_initial_sha256": peers,
            "common_state_tensors": len(common), "shared_initialization_equal": True}


def case_binding(context, settings):
    left, right = copy.deepcopy(settings["config"]), context.config.to_dict()
    for value in (left, right):
        value.pop("name")
        value.pop("training")
    if left != right or list(context.sources) != settings["source_tasks"]:
        raise ValueError("Selective source configuration differs")
    return {"source_context": context.record, "graph_binding": graph_binding(context.items),
            **initial_binding(settings, context.class_counts)}


def grid(full):
    if type(full) is not bool:
        raise ValueError("Explicit short/full profile required")
    return [(held, seed) for held in EXPECTED["held_targets"]
            for seed in (EXPECTED["training_seeds"] if full else (64,))]


def select_bound_case(launch, producer, held_out, seed, *, full):
    root = external_directory(launch["out_dir"])
    pairs = grid(full)
    study_hash = file_hash(STUDY_PATH)
    manifest = load_plan(ROOT / "configs/experiments/base97_training.json")["prepared_manifest_sha256"]
    if (launch["code"] != producer or producer["dirty"] or launch["study"] != load_study(STUDY_PATH)
            or launch["study_file_sha256"] != study_hash or launch["profile"] != ("full" if full else "short")
            or launch["expected_updates"] != (9018 if full else 24)
            or launch["prepared_manifest_sha256"] != manifest or launch["execution_device"] != "cuda"
            or launch["reference_launch_sha256"] != REFERENCE_LAUNCH
            or launch["test_scoring"] is not False or launch["query_labels_accessed"] is not False
            or type(seed) is not int or (held_out, seed) not in pairs):
        raise ValueError("Frozen none source launch/profile/producer differs")
    cases = launch["cases"]
    if len(cases) != len(pairs) or {(c["held_out"], c["seed"]) for c in cases} != set(pairs):
        raise ValueError("Complete distinct frozen source cohort required")
    for case in cases:
        held, s = case["held_out"], case["seed"]
        name = held + "-s" + str(s)
        sources = options(held, s, full=full)["source_tasks"]
        context = case["source_context"]
        if (type(s) is not int or case["name"] != name or case["settings"] != options(held, s, full=full)
                or Path(case["out"]).resolve() != root / "cases" / name
                or context["held_out"] != held or context["sources"] != sources
                or context["target_payload_accessed"] is not False or set(case["graph_binding"]) != set(sources)
                or case["shared_initialization_equal"] is not True or case["common_state_tensors"] <= 0
                or not re.fullmatch("[0-9a-f]{64}", case["initial_state_sha256"])
                or case["common_initial_sha256"] != dict.fromkeys(("capacity_control", "x_residual"), case["initial_state_sha256"])):
            raise ValueError("Frozen source output/settings/context/initial binding differs")
    return next(c for c in cases if (c["held_out"], c["seed"]) == (held_out, seed))


def prepare_launch(reference_launch, reference_sha256, out_dir, producer, *, full):
    require_frozen_code(producer)
    path = Path(reference_launch).resolve(strict=True)
    if reference_sha256 != REFERENCE_LAUNCH or file_hash(path) != REFERENCE_LAUNCH:
        raise ValueError("Accepted source launch pin required")
    reference = json.loads(path.read_text())
    if reference["code"]["commit"] != REFERENCE_PRODUCER or reference["code"]["dirty"]:
        raise ValueError("Accepted source producer required")
    root = external_directory(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    manifest = load_plan(ROOT / "configs/experiments/base97_training.json")["prepared_manifest_sha256"]
    cases = []
    for held, seed in grid(full):
        context = load_source_context(reference["prepared"], reference["data_root"], held_out=held,
                                      consumer=producer, manifest_sha256=manifest)
        settings = options(held, seed, full=full)
        name = held + "-s" + str(seed)
        cases.append({"name": name, "held_out": held, "seed": seed, "settings": settings,
                      "out": str(root / "cases" / name), **case_binding(context, settings)})
        del context
    launch = {"schema_version": 1, "protocol": "g4_none_ordered_source_launch_v2", "code": producer,
              "profile": "full" if full else "short", "study": load_study(STUDY_PATH),
              "study_file_sha256": file_hash(STUDY_PATH), "out_dir": str(root), "cases": cases,
              "prepared": reference["prepared"], "data_root": reference["data_root"],
              "prepared_manifest_sha256": manifest, "reference_launch": str(path),
              "reference_launch_sha256": REFERENCE_LAUNCH, "execution_device": "cuda",
              "expected_updates": 9018 if full else 24, "test_scoring": False, "query_labels_accessed": False}
    for held, seed in grid(full):
        select_bound_case(launch, producer, held, seed, full=full)
    require_frozen_code(producer)
    write_json(root / "launch.json", launch)
    return launch


class BoundNoneSourceRuntime(NoneOrderedSourceRuntime):
    def __init__(self, binding):
        self.binding = copy.deepcopy(binding)

    def build_model(self, config, counts):
        model = super().build_model(config, counts)
        if state_id(model) != self.binding["initial_state_sha256"]:
            raise ValueError("Actual initial source state differs before first update")
        return model

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), "none_source_launch": copy.deepcopy(self.binding)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        if model.training_config.get("none_source_launch") != self.binding:
            raise ValueError("Selected source launch identity differs")
        return model


def launch_binding(launch, case, launch_sha256):
    return {"launch_sha256": launch_sha256, "study_file_sha256": launch["study_file_sha256"],
            "profile": launch["profile"], "held_out": case["held_out"], "seed": case["seed"],
            "initial_state_sha256": case["initial_state_sha256"], "graph_binding": case["graph_binding"],
            "source_context_sha256": digest(case["source_context"])}


def worker_main(*, full):
    parser = argparse.ArgumentParser(description="Frozen ordered none source " + ("full 1002" if full else "short 8") + " update case")
    for name in ("expected-commit", "launch-plan", "launch-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--held-out", choices=EXPECTED["held_targets"], required=True)
    parser.add_argument("--seed", type=int, choices=EXPECTED["training_seeds"] if full else (64,), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    producer = code_snapshot()
    if sys.platform != "linux" or not __debug__ or producer["dirty"] or producer["commit"] != args.expected_commit:
        raise RuntimeError("Clean remote expected commit required")
    if file_hash(args.launch_plan) != args.launch_sha256:
        raise ValueError("Frozen source launch changed")
    launch = json.loads(Path(args.launch_plan).read_text())
    case = select_bound_case(launch, producer, args.held_out, args.seed, full=full)
    context = load_source_context(launch["prepared"], launch["data_root"], held_out=args.held_out,
        consumer=producer, manifest_sha256=launch["prepared_manifest_sha256"])
    if case_binding(context, case["settings"]) != {k: case[k] for k in case_binding_keys()}:
        raise ValueError("Actual source context/graph/initial state changed")
    binding = launch_binding(launch, case, args.launch_sha256)
    provenance = {**context.record, "none_source_launch": binding}

    def guard():
        require_frozen_code(producer)
        if (file_hash(args.launch_plan) != args.launch_sha256 or file_hash(STUDY_PATH) != launch["study_file_sha256"]
                or file_hash(launch["reference_launch"]) != REFERENCE_LAUNCH
                or any(file_hash(p) != sha for p, sha in context.record["protected_files"].items())):
            raise ValueError("Frozen launch/study/source inputs changed")

    guard()
    result = run_training(case["settings"], context.items, case["out"], provenance, producer,
        device=launch["execution_device"], resume=args.resume, stop_after=args.stop_after,
        runtime_adapter=BoundNoneSourceRuntime(binding), finish_guard=guard)
    guard()
    print(json.dumps({"operation": result["operation"], "updates_this_call": result["updates_this_call"],
                      "held_out": args.held_out, "total_budget": case["settings"]["optimizer_steps"],
                      "test_scoring": False}), flush=True)


def case_binding_keys():
    return ("source_context", "graph_binding", "initial_state_sha256", "common_initial_sha256",
            "common_state_tensors", "shared_initialization_equal")

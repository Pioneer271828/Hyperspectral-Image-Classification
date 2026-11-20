"""Synthetic-only fixture and launch registration for the existing full light worker."""
from collections import Counter
import json
from pathlib import Path
import re

from g4_light_adapter_memory import RECOMPUTE
from g4_light_target import ADAPTER_HEADS
from g4_matched_target_study import METHODS, select_case
from g4_support_context import check_files, read
from g4_support_target import source_pins
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import external_directory, write_json

FIXTURE = "g4_light_synthetic_fixture_v1"
REGISTRATION = "g4_light_synthetic_full54_registration_v1"
SCOPE = "synthetic_engineering"
MEMORY = "adapter_backward_policy"
PHASE = "engineering_zero_update"
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
REPRESENTATIVES = dict(zip(NAMES, ((64,"none"), (512,"capacity_control"), (2048,"x_residual")), strict=True))


def pin(path):
    return {"path": str(Path(path).resolve(strict=True)), "sha256": file_hash(path)}


def pinned(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Pinned light artifact required")
    path = external_directory(value["path"])
    if str(path) != value["path"] or file_hash(path) != value["sha256"]:
        raise ValueError("Pinned light artifact changed or is not in its original location")
    return read(path)


def code_identity(code):
    return {key:code[key] for key in ("commit", "source_hash")}


def selected(matrix):
    return [key for key in matrix if (key[1],key[3]) == REPRESENTATIVES[key[0]]]


def fixture(out, code):
    # Use the repository's deterministic engineering fixtures. Their simulated
    # source receipts exist only to exercise the frozen initialization loader.
    import torch
    from g4_matched_target_study import initialize
    from scripts.train_joint_m4 import graph_binding
    from test_g4_matched_target_study import launch_fixture
    from test_m4_full_transfer import fixture as active_source
    from test_none_full_transfer import fixture as none_source
    from test_target_training import observation_fixture

    out.mkdir()
    reference = out.parent / (out.name + "-reference-disabled")
    launch = launch_fixture(reference)
    launch.update(code=code, short_acceptances={}, fixture_scope=SCOPE)
    observations, pins = {}, {}
    for scene in NAMES:
        folder = out / "observations" / scene
        folder.mkdir(parents=True)
        (items, context, _), (observation, record, family) = observation_fixture(folder, scene)
        inputs = {"observation": str(observation), "record": str(record), "family": str(family),
                  "record_sha256": file_hash(record), "family_sha256": file_hash(family)}
        observations[scene] = (inputs, context, graph_binding(items))
        pins.update(context["protected_files"])
    for case in launch["cases"]:
        scene, seed, method, mode = (case[field] for field in ("target","seed","method","mode"))
        inputs, context, graph = observations[scene]
        case.update(inputs=inputs, target_context=context, graph_binding=graph)
        if method == METHODS[1]:
            folder = out / "sources" / f"{scene}-{seed}-{mode}"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if mode == "none":
                    _, spec = none_source(folder, fixture_target=scene, fixture_seed=seed)
                else:
                    _, spec = active_source(folder, mode, fixture_target=scene, fixture_seed=seed)
            _, initial = initialize(scene, seed, method, mode, source_spec=spec)
            case.update(initialization=initial, initial_state_sha256=initial["initial_state_sha256"], source_spec=spec)
            pins.update(source_pins(spec))
    launch_path = out / "launch.json"
    write_json(launch_path, launch)
    path = out / "fixture.json"
    write_json(path, {"policy": FIXTURE, "scope": SCOPE, "code":code_identity(code), "launch":pin(launch_path),
                      "protected_files":pins, "real_training_authorized":False, "query_access":False})
    return pin(path)


def read_fixture(value, code):
    descriptor = pinned(value)
    root = Path(value["path"]).parent
    if (Path(value["path"]).name != "fixture.json" or descriptor["policy"] != FIXTURE or descriptor["scope"] != SCOPE
            or descriptor["code"] != code_identity(code) or descriptor["real_training_authorized"] is not False
            or descriptor["query_access"] is not False or Path(descriptor["launch"]["path"]) != root / "launch.json"):
        raise ValueError("Expected a same-code synthetic-only light fixture")
    launch = pinned(descriptor["launch"])
    if launch["fixture_scope"] != SCOPE or launch["code"] != code:
        raise ValueError("Fixture producer changed")
    select_case(launch, code, NAMES[0], 64, METHODS[1], "none", phase=PHASE)
    pins = {}
    for case in launch["cases"]:
        scene = case["target"]
        inputs = case["inputs"]
        folder = root / "observations" / scene
        if any(Path(inputs[name]) != folder / filename for name,filename in
               (("observation","observation.pt"),("record","record.json"),("family","family.json"))):
            raise ValueError("Synthetic observations escaped their fixture")
        record = read(folder / "record.json")
        fields = record["prepared_fields"]
        size = 18 if DATASETS[scene][2] == 9 else 20
        if (record["target"], fields["height"], fields["width"], fields["bands"]) != (scene,size,size,7):
            raise ValueError("Synthetic observation has a foreign scene shape")
        pins.update(case["target_context"]["protected_files"])
        if case["method"] == METHODS[1]:
            source = source_pins(case["source_spec"])
            if (not source or any(not Path(path).is_relative_to(root / "sources") for path in source)
                    or read(case["source_spec"]["source_launch"])["code"].get("fixture") is not True):
                raise ValueError("Non-fixture source cannot enter a synthetic run")
            pins.update(source)
    if pins != descriptor["protected_files"] or any(not Path(path).is_relative_to(root) for path in pins):
        raise ValueError("Synthetic input pins escaped or differ from the fixture")
    check_files(pins)
    return launch, {**pins, value["path"]: value["sha256"], descriptor["launch"]["path"]: descriptor["launch"]["sha256"]}


def role_ledger(launch, devices, *, profile):
    from g4_light_full_target import UPDATES, roles

    rows, matrix = [], roles()
    active = set(matrix if profile == "all" else selected(matrix))
    for index,key in enumerate(matrix):
        target,seed,method,mode = key
        case = next(c for c in launch["cases"] if (c["target"],c["seed"],c["method"],c["mode"])
                    == (target,seed,METHODS[1],mode))
        rows.append({"role": list(key), "slot":devices[index % len(devices)]["slot"],
                     "registered_updates":UPDATES, "scheduled":key in active,
                     "strategy":RECOMPUTE if method == ADAPTER_HEADS else None,
                     "reference_sha256":digest(case), "initial_sha256":case["initial_state_sha256"],
                     "input_sha256":digest(case["target_context"]["protected_files"])})
    return rows


def verify_devices(devices):
    if not isinstance(devices, list) or len(devices) not in (2,3):
        raise ValueError("Two or three explicit synthetic slots required")
    slots = [row["slot"] for row in devices]
    if len(set(slots)) != len(slots):
        raise ValueError("Duplicate synthetic dispatch slot")
    if all(set(d) == {"slot"} for d in devices):
        if slots != [f"cpu-{i}" for i in range(len(slots))]:
            raise ValueError("Explicit ordered CPU slots required")
    elif not all(set(row)=={"slot","uuid"} and type(row["slot"]) is int and row["slot"] >= 0
                 and GPU_UUID.fullmatch(row["uuid"]) for row in devices) or len({row["uuid"] for row in devices})!=len(devices):
        raise ValueError("Distinct physical GPU indices and UUIDs required")
    return slots


def register(value, code, out, output_path, issue, devices, *, profile):
    if type(issue) is not int or issue <= 249 or profile not in ("representative","all"):
        raise ValueError("Use a fresh engineering issue and a known coverage profile")
    slots = verify_devices(devices)
    launch, protected = read_fixture(value, code)
    reject_overlap(out, value, protected)
    if output_path.exists() or output_path == out or out in output_path.parents or out.exists():
        raise ValueError("Engineering registration and cases need fresh distinct paths")
    rows = role_ledger(launch, devices, profile=profile)
    result = {"policy": REGISTRATION, "scope":SCOPE, "fixture":value, "producer":code_identity(code),
              "experiment_issue":issue,"devices":devices,"gpu_ids":slots,"out_dir":str(out),"profile":profile,
              "roles":rows,"planned_optimizer_updates":sum(r["registered_updates"] for r in rows),
              "scheduled_optimizer_updates":sum(r["registered_updates"] for r in rows if r["scheduled"]),
              MEMORY:RECOMPUTE,"real_training_authorized":False,"query_labels_during_training":False,
              "test_scoring_during_training":False}
    write_json(output_path,result)
    return pin(output_path)


def reject_overlap(out, value, protected):
    fixture_root = Path(value["path"]).parent
    if (out == fixture_root or fixture_root in out.parents or out in fixture_root.parents
            or any(out == Path(path) or out in Path(path).parents for path in protected)):
        raise ValueError("Engineering output overlaps frozen fixture inputs")


def read_registration(value, path, sha256, code):
    expected = {"policy","scope","fixture","producer","experiment_issue","devices","gpu_ids","out_dir","profile",
                "roles","planned_optimizer_updates","scheduled_optimizer_updates", MEMORY,"real_training_authorized",
                "query_labels_during_training","test_scoring_during_training"}
    if (set(value) != expected or value["scope"] != SCOPE or value["producer"] != code_identity(code)
            or type(value["experiment_issue"]) is not int or value["experiment_issue"] <= 249
            or value["profile"] not in ("representative","all") or value[MEMORY] != RECOMPUTE
            or value["real_training_authorized"] is not False or value["query_labels_during_training"] is not False
            or value["test_scoring_during_training"] is not False):
        raise ValueError("Synthetic registration identity, scope or permission differs")
    if value["gpu_ids"] != verify_devices(value["devices"]):
        raise ValueError("Registered slot list differs")
    launch,pins = read_fixture(value["fixture"],code)
    rows = role_ledger(launch,value["devices"],profile=value["profile"])
    if (value["roles"] != rows or value["planned_optimizer_updates"] != sum(r["registered_updates"] for r in rows)
            or value["scheduled_optimizer_updates"] != sum(r["registered_updates"] for r in rows if r["scheduled"])):
        raise ValueError("Synthetic roster or update budget changed")
    out = external_directory(value["out_dir"])
    pins[str(path)] = sha256
    reject_overlap(out,value["fixture"],pins)
    return value,pins

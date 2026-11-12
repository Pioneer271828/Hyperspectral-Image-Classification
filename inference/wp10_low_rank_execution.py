"""Expose audited low-rank packages through the shared frozen CPU entry."""

import io
from pathlib import Path
import subprocess
from time import perf_counter_ns

import torch

from compression.low_rank_recovery import factor_calls, factor_counts
from compression.low_rank_recovery_audit import HEADS, load_package
from compression.precision_cost import (ResidentPrecisionTile, _tile_identity, benchmark_precision_cost,
                                        prepare_resident_precision_tiles)
from compression.precision_startup import _memory_kib
from compression.registered_low_rank_recovery import POLICY as RECOVERY_POLICY, _design
from compression.registered_precision_recovery import runtime
from research_runtime import ROOT, external_directory, require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash


KIND = "wp10-low-rank"
POLICY = "wp10-low-rank-frozen-execution-v1"
WORKER_SCOPE = "authenticated_package_load_and_first_resident_full_sweep"
RUNTIME_PATHS = ("model", "loadData", "configs", "transfer/curriculum_training.py", "transfer/g3_training.py",
                 "compression/low_rank.py", "compression/low_rank_checkpoint.py", "compression/low_rank_recovery.py",
                 "compression/low_rank_recovery_audit.py", "compression/registered_low_rank_recovery.py")


def _elapsed(start):
    return (perf_counter_ns() - start) / 1e6


def _case_package(registered, audited, case):
    name = case["case"]
    report = audited["report"][name]
    package = report["selected"]["package"]
    path = Path(registered["root"]) / "deployment" / name / "selected.pt"
    if (report["case"] != case or report["steps"] != registered["budget"]
            or package["weights"]["path"] != str(path) or package["sidecar"]["path"] != str(path) + ".json"):
        raise ValueError("Selected package differs from the independently registered case")
    return package


def _campaign(request_pin):
    request = admission.read(request_pin)
    if set(request) != {"kind", "registration", "audit"} or request["kind"] != KIND:
        raise ValueError("Exact low-rank deployment request required")
    registration, audited = (admission.read(request[k]) for k in ("registration", "audit"))
    origin = Path(registration["root"])
    if (registration["policy"] != RECOVERY_POLICY or request["registration"]["path"] != str(origin / "registration.json")
            or request["audit"]["path"] != str(origin / "audit.json") or registration["runtime"] != runtime()
            or audited["status"] != "passed_low_rank_recovery_independent"
            or audited["registration"] != request["registration"] or audited["actual_updates"] != registration["maximum_updates"]
            or any(audited[key] != 0 for key in ("query", "gt", "independent_updates"))):
        raise ValueError("Frozen low-rank registration and successful independent audit required")
    return request, registration, audited


def _selected_cases(registration, audited):
    cases = {}
    for case in registration["cases"]:
        name = case["case"]
        cases[name] = dict(case=case, package=_case_package(registration, audited, case),
                           initial=registration["initial"][name]["evidence"],
                           selected_step=audited["report"][name]["selected"]["step"])
    if not cases:
        raise ValueError("No registered case covered all support")
    return cases


def freeze(request_pin, code, out):
    require_frozen_code(code)
    request, registration, audited = _campaign(request_pin)
    producer = registration["commit"]
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", producer, code["commit"]], check=True)
    changed = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", producer, code["commit"],
                                       "--", *RUNTIME_PATHS], text=True)
    if changed.strip():
        raise ValueError("Producer and deployment code differ in the frozen model or recovery paths")
    spec, _, scene, _, _ = _design(registration["request"])
    if any(registration[k] != value for k, value in spec.items()):
        raise ValueError("Frozen scope, rank, observation or exposure design differs")
    cases = _selected_cases(registration, audited)
    pins = [request_pin, request["registration"], request["audit"], *registration["protected"].values()]
    for entry in cases.values():
        package_pin = entry["package"]
        model = package(entry)
        if state_hash(model) != package_pin["state"]:
            raise ValueError("Low-rank package differs from the independent endpoint")
        pins.extend(package_pin[k] for k in ("weights", "sidecar"))
    out = external_directory(out)
    if out.exists() or any(out == Path(pin["path"]).resolve() or out in Path(pin["path"]).resolve().parents
                           for pin in pins):
        raise ValueError("New deployment output outside protected inputs required")
    tiles = prepare_resident_precision_tiles(scene, registration["validation"]["tiles"])
    tile_identity = _tile_identity(tiles)
    if registration["scope"] == "wp08_xuzhou_off" and (len(tiles), sum(row["pixels"] for row in tile_identity)) != (5, 3920):
        raise ValueError("Original 5-block, 3920-pixel workload required")
    out.mkdir()
    path = out / "resident-inputs.pt"
    image_pin = dict(path=str(path), sha256=CurriculumRun._write(
        [dict(tile_id=t.tile_id, image=t.image, wavelengths=t.wavelengths) for t in tiles], path))
    admission.check_pins(pins)
    require_frozen_code(code)
    return dev._json_new(out / "manifest.json", dict(kind=KIND, policy=POLICY, commit=code["commit"],
        source_hash=code["source_hash"], producer_commit=producer, request=request_pin,
        registration=request["registration"], audit=request["audit"], runtime=runtime(),
        dataset=scene.dataset, class_ids=registration["validation"]["class_ids"],
        data_protocol="wp10_xuzhou_k5_v5" if registration["scope"] == "wp08_xuzhou_off" else "synthetic",
        resident_inputs=image_pin, tiles=tile_identity, cases=cases, case_order=[r["case"] for r in registration["cases"]],
        default="original_full_fp32", selection=registration["selection"], budget=registration["budget"],
        observation_role="devval_image_only", query=0, gt=0, optimizer_updates=0))


def load_manifest(pin, code):
    require_frozen_code(code)
    reg = admission.read(pin)
    if (reg["kind"] != KIND or reg["policy"] != POLICY or reg["commit"] != code["commit"]
            or reg["source_hash"] != code["source_hash"] or reg["runtime"] != runtime()
            or reg["observation_role"] != "devval_image_only"
            or reg["resident_inputs"]["path"] != str(Path(pin["path"]).parent / "resident-inputs.pt")
            or any(reg[k] != 0 for k in ("query", "gt", "optimizer_updates"))):
        raise ValueError("Low-rank manifest code, timing, path or source identity differs")
    request, registration, audited = _campaign(reg["request"])
    if (any(reg[k] != request[k] for k in ("registration", "audit"))
            or reg["producer_commit"] != registration["commit"]
            or reg["case_order"] != [c["case"] for c in registration["cases"]]
            or reg["cases"] != _selected_cases(registration, audited)
            or reg["class_ids"] != registration["validation"]["class_ids"]
            or [row["tile_id"] for row in reg["tiles"]] != [v["id"] for v in registration["validation"]["tiles"]]
            or reg["default"] != "original_full_fp32" or reg["selection"] != registration["selection"]
            or reg["budget"] != registration["budget"]):
        raise ValueError("Frozen manifest differs from its audited recovery case or source")
    rows = torch.load(io.BytesIO(dev._read(reg["resident_inputs"])), map_location="cpu", weights_only=True)
    if any(set(row) != {"tile_id", "image", "wavelengths"} for row in rows):
        raise ValueError("Registered image-only input required")
    tiles = [ResidentPrecisionTile(**row) for row in rows]
    if not tiles or _tile_identity(tiles) != reg["tiles"]:
        raise ValueError("Resident observation bytes or identities differ")
    return reg, tiles


def _scan(model, reg, tiles, *, evidence=False):
    scale = 50 if reg["dataset"] == "paviaU" else 100
    with torch.inference_mode():
        if evidence:
            with factor_calls(model) as counts:
                values = [model(reg["dataset"], t.image, wavelengths=t.wavelengths, superpixel_scale=scale,
                                return_graph_info=True)[:3] for t in tiles]
            if counts != factor_counts(model, len(tiles)):
                raise ValueError("Both original factors did not execute")
        else:
            values = [model(reg["dataset"], t.image, wavelengths=t.wavelengths, superpixel_scale=scale,
                            return_graph_info=True)[:3] for t in tiles]
            counts = {}
    return values, counts


def _outputs(values, reg):
    result = []
    for tile, row in zip(reg["tiles"], values, strict=True):
        if len(row) != len(HEADS) or any(p.shape != (tile["pixels"], len(reg["class_ids"])) or not torch.isfinite(p).all() for p in row):
            raise ValueError("Expected actual finite three-head full-block outputs")
        result.append({head: tensor_hash(p) for head, p in zip(HEADS, row, strict=True)})
    return result


def package(entry):
    model = load_package(entry["package"], initial=entry["initial"], case=entry["case"])
    # The native factor checkpoint restores tensors, while cache policy belongs to deployment.
    model.backbone.adapter.cache_attention = False
    return model


def startup(pin, code, name):
    import os
    before = _memory_kib()
    reg, tiles = load_manifest(pin, code)
    entry = reg["cases"][name]
    ready = _memory_kib()
    started = perf_counter_ns()
    model = package(entry)
    load_ms = _elapsed(started)
    loaded = _memory_kib()
    started = perf_counter_ns()
    values, _ = _scan(model, reg, tiles)
    first_ms = _elapsed(started)
    first = _memory_kib()
    checked, counts = _scan(model, reg, tiles, evidence=True)
    hashes = _outputs(values, reg)
    if hashes != _outputs(checked, reg) or _tile_identity(tiles) != reg["tiles"]:
        raise ValueError("Startup score identity changed after first sweep")
    require_frozen_code(code)
    return dict(case=name, manifest=pin, scope=WORKER_SCOPE, pid=os.getpid(), load_ms=load_ms,
                first_sweep_ms=first_ms, before_input=before, before_model=ready, after_model=loaded,
                after_first_sweep=first, package_bytes=entry["package"]["bytes"],
                parameters=entry["package"]["parameters"], state=entry["package"]["state"],
                factor_calls=counts, output_hashes=hashes, tiles=reg["tiles"], query=0, gt=0, optimizer_updates=0)


def execute(pin, code, name, out, *, export=False):
    reg, tiles = load_manifest(pin, code)
    if name not in reg["cases"]:
        raise ValueError("Only registered low-rank static cases can execute")
    out = external_directory(out)
    if out.exists() or out in Path(pin["path"]).resolve().parents:
        raise ValueError("Fresh frozen execution output required")
    model = package(reg["cases"][name])
    values, counts = _scan(model, reg, tiles, evidence=True)
    hashes = _outputs(values, reg)
    require_frozen_code(code)
    out.mkdir()
    result = dict(status="passed", mode=name, manifest=pin, output_hashes=hashes, factor_calls=counts,
                  case=reg["cases"][name]["case"], tiles=reg["tiles"], query=0, gt=0, optimizer_updates=0)
    if export:
        path = out / "predictions.pt"
        checksum = CurriculumRun._write([dict(tile_id=t.tile_id, probabilities=dict(zip(HEADS, v, strict=True)))
                                         for t, v in zip(tiles, values, strict=True)], path)
        result["predictions"] = dict(path=str(path), sha256=checksum)
    return dev._json_new(out / "execution.json", result)


def resident(pin, code):
    reg, tiles = load_manifest(pin, code)
    groups = {}
    for seed in dict.fromkeys(reg["cases"][name]["case"]["seed"] for name in reg["case_order"]):
        members = [name for name in reg["case_order"] if reg["cases"][name]["case"]["seed"] == seed]
        dense = next(name for name in members if reg["cases"][name]["case"]["arm"] == "dense")
        control = next(name for name in members if reg["cases"][name]["case"]["arm"] == "full-rank")
        reduction = [name for name in members if name not in (dense, control)]
        divisions = [(f"s{seed}-reduction-{i//2+1}", [dense, *reduction[i:i+2]]) for i in range(0, len(reduction), 2)]
        divisions.append((f"s{seed}-full-rank-control", [dense, control]))
        for group, names in divisions:
            entries = {reg["cases"][name]["case"]["arm"]: reg["cases"][name] for name in names}
            models = {arm: package(entry) for arm, entry in entries.items()}
            evidence = {arm: dict(state=entry["package"]["state"], plan=entry["initial"]["plan"],
                                  parameters=entry["package"]["parameters"]) for arm, entry in entries.items()}
            counts = {}
            for arm, model in models.items():
                _, counts[arm] = _scan(model, reg, tiles, evidence=True)
            row = benchmark_precision_cost(models, tiles, reg["dataset"], low_rank=evidence)
            groups[group] = dict(seed=seed, cases={entry["case"]["arm"]: entry["case"]["case"] for entry in entries.values()},
                                 kind="expansion_control" if control in names else "reduction_and_random",
                                 factors=counts, timing=row)
    return reg, groups

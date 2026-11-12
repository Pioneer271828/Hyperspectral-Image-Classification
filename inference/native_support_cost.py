"""Paired CPU endpoint latency with separately timed fresh-process startup."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from inference.native_support_checkpoint import load_endpoint, predict_registered_tile
from research_runtime import ROOT, tensor_hash
from transfer import native_support_audit as audit, native_support_campaign as campaign, registered_development as dev
from transfer.native_support import NativeSupportModel, _measure, make_selector, native_forward


ARMS = ("off", "local", "joint", "fixed")
WARM_REPEATS = 8
COLD_REPEATS = 3


def _summary(values):
    p = np.quantile(values, [.1, .5, .9])
    return dict(samples=values, p10=float(p[0]), median=float(p[1]), p90=float(p[2]))


@torch.no_grad()
def _repeat(model, state, scene, record, tile, bank, timings, *, selector=None):
    if isinstance(model, NativeSupportModel):
        settings = state["trial"]
        if selector is None:
            selector = make_selector(bank[str(tile["id"])], settings, settings["support_seed"])
        before = selector.cost.copy()
        output, frame = native_forward(model, scene, scene.plan["tiles"][tile["id"]], selector=selector,
                                       detail=True, timing=timings)
        timings.update({name: [seconds - before[name]] for name, seconds in selector.cost.items()})
    else:
        backbone = model.backbone
        with ExitStack() as stack:
            for module, method, name in (
                (backbone.adapter, "forward", "vca"), (backbone.learnable_graph, "build_graph", "q_a0_build"),
                (backbone, "cnn_stage1", "cnn_stage1"), (backbone, "cnn_stage2", "cnn_stage2"),
                (backbone, "graph_stage1", "graph_stage1"), (backbone, "graph_stage2", "graph_stage2"),
                (model, "predict_features", "three_heads"),
            ):
                stack.enter_context(_measure(module, method, timings, name))
            output, frame = predict_registered_tile(model, state, scene, record, tile["id"], bank)
    info = output[3]
    details = dict(
        probability_hashes=[tensor_hash(x) for x in output[:3]], shapes=[list(x.shape) for x in output[:3]],
        assignment_nnz=info["assignment_nnz"], stage1_nnz=info["graph_nnz"],
        stage2_nnz=frame["trace"]["stage2_nnz"] if frame else info["graph_nnz"], nodes=info["token_count"],
        block_shape=[tile["box"][1] - tile["box"][0], tile["box"][3] - tile["box"][2], len(scene.bands.band_ids)])
    return details


def _cold(registration, case, package):
    started = perf_counter()
    data = campaign._case_data(case, bank=case["work_package"] == "WP08")
    loaded = perf_counter()
    model, state = load_endpoint(package, registration, case, data)
    restored = perf_counter()
    _, scene, record, _, bank = data
    tile = next(row for row in record["roles"] if row["role"] == "devval")
    timing = {}
    details = _repeat(model, state, scene, record, tile, bank, timing)
    return dict(case_materialize_seconds=loaded - started, package_restore_seconds=restored - loaded,
                forward_and_diagnostics_seconds=perf_counter() - restored, internal_seconds=perf_counter() - started,
                details=details, stages=timing)


def _train_tile_replay(model, state, scene, record, tile, bank):
    timing = {}
    selector = None
    started = perf_counter()
    if isinstance(model, NativeSupportModel):
        settings = state["trial"]
        key = str(tile["id"])
        selector = make_selector(bank[key], settings, settings["support_seed"])
        if state["donor"] is not None:
            selector = selector.authenticate(state["donor"]["snapshots"][key])
        else:
            selector.advance(state["step"])
    prepared = perf_counter()
    # The original endpoint API admits validation only. The cost probe obtains
    # its diagnostic training tile from the already authenticated release.
    if selector is None:
        from transfer.g3_training import _forward
        with torch.no_grad():
            outputs = _forward(model, scene, scene.plan["tiles"][tile["id"]])
        details = dict(stage2_nnz=outputs[3]["graph_nnz"], nodes=outputs[3]["token_count"],
                       probability_hashes=[tensor_hash(head) for head in outputs[:3]])
    else:
        details = _repeat(model, state, scene, record, tile, bank, timing, selector=selector)
    finished = perf_counter()
    return dict(prepare_seconds=prepared - started, forward_seconds=finished - prepared,
                end_to_end_seconds=finished - started, details=details, stages=timing,
                replay=None if selector is None else selector.snapshot(),
                retrieval_seconds=None if selector is None else selector.cost["retrieval_seconds"],
                selection_seconds=None if selector is None else selector.cost["selection_seconds"])


def measure(registration, frozen, code, producer_commit):
    producer = campaign.pinned_producer(registration, producer_commit)
    root, reg, publication = audit.load_frozen(registration, frozen, producer)
    key = "wp08-nine8-64"
    case = reg["cases"][key]
    started = perf_counter()
    data = campaign._case_data(case)
    read_seconds = perf_counter() - started
    _, scene, record, _, bank = data
    tile = next(row for row in record["roles"] if row["role"] == "devval")
    training_tile = next(row for row in record["roles"] if row["role"] == "devtrain"
                         and str(row["id"]) in bank)
    rows = {}
    for arm in ARMS:
        package = publication["rows"][f"{key}/{arm}"]["endpoints"]["selected"]
        load_start = perf_counter()
        model, state = load_endpoint(package, registration, case, data)
        rows[arm] = dict(package=package, model=model, state=state, warm=[], cold=[],
                         restore_seconds=perf_counter() - load_start)
    for index in range(WARM_REPEATS + 1):
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            row = rows[arm]
            timing = {}
            begin = perf_counter()
            details = _repeat(row["model"], row["state"], scene, record, tile, bank, timing)
            end = perf_counter()
            if index:
                row["warm"].append(dict(seconds=end - begin, details=details, stages=timing, pair=index))
    command = [sys.executable, "-m", "inference.native_support_cost", registration["path"], registration["sha256"],
               frozen["path"], frozen["sha256"], code["commit"], producer_commit, key]
    for index in range(COLD_REPEATS):
        order = ARMS[index:] + ARMS[:index]
        for arm in order:
            begin = perf_counter()
            proc = subprocess.run(command + [arm], cwd=ROOT, capture_output=True, text=True, check=True)
            elapsed = perf_counter() - begin
            rows[arm]["cold"].append(dict(json.loads(proc.stdout), process_seconds=elapsed, pair=index + 1))
    for index in range(WARM_REPEATS):
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            row = rows[arm]
            sample = _train_tile_replay(row["model"], row["state"], scene, record, training_tile, bank)
            row.setdefault("paired_train_role_replay", []).append(dict(sample, pair=index + 1))
    for arm, row in rows.items():
        state = row.pop("state")
        model = row.pop("model")
        row["parameter_count"] = sum(p.numel() for p in model.parameters())
        row["package_bytes"] = Path(row["package"]["path"]).stat().st_size
        row["warm_summary"] = _summary([x["seconds"] for x in row["warm"]])
        row["cold_summary"] = _summary([x["process_seconds"] for x in row["cold"]])
        row["paired_warm_delta_vs_off"] = _summary([a["seconds"] - b["seconds"]
                                                   for a, b in zip(row["warm"], rows["off"]["warm"], strict=True)])
        if any(x["details"] != row["warm"][0]["details"] for x in row["warm"] + row["cold"]):
            raise ValueError("Paired endpoint inference changed across cached and cold loads")
        if any(x["details"] != row["paired_train_role_replay"][0]["details"]
               for x in row["paired_train_role_replay"]):
            raise ValueError("Training role cost probe changed its predictions")
        row["train_role_total_summary"] = _summary([x["end_to_end_seconds"] for x in row["paired_train_role_replay"]])
        row["scorer_seed"] =  case["scorer_seed"] if state["trial"] is not None else None
        row["one_time_bank_seconds"] = case["bank_build_seconds"] if state["trial"] is not None else 0.
        row["one_time_donor_seconds"] = (json.loads(dev._read(dev._pin(root / "matrix" / key / "donor.json")))
            ["construction_seconds"] if arm == "fixed" else 0.)
        row["amortized_once_seconds_per_forward"] = {
            str(n): (row["one_time_bank_seconds"] + row["one_time_donor_seconds"]) / n for n in (1, 10, 100)}
    return dev._json_new(root / "cost.json", dict(version=2, scope="cpu_fp32_development_only", case=key,
        registration=registration, freeze=frozen, producer=producer, consumer=code["commit"], roles=["devval"], query=0, optimization=0, repeats=rows,
        source_data_bank_read_seconds=read_seconds, bank_capture_timing=case["bank_capture_timing"],
        training_diagnostic_role="devtrain", training_diagnostic_tile=training_tile["id"],
        amortized_once_over=[1, 10, 100],
        notes=["Stage measurements are nested and cannot be summed to obtain wall time.",
               "Cold starts include Python, imports, process launch and exit. OS cache was not flushed.",
               "Warm uses a resident endpoint and source bank. Q/A0 and supported values are rebuilt.",
               "Validation uses the fixed step-zero support in every active arm, including the donor arm."]))


if __name__ == "__main__":
    from research_runtime import code_snapshot, require_frozen_code
    registration_path, registration_sha, freeze_path, freeze_sha, expected, producer_commit, key, arm = sys.argv[1:]
    torch.set_num_threads(1)
    code = code_snapshot()
    require_frozen_code(code)
    if code["commit"] != expected:
        raise ValueError("Cold worker checkout changed")
    registration = dict(path=registration_path, sha256=registration_sha)
    frozen = dict(path=freeze_path, sha256=freeze_sha)
    _, reg, pub = audit.load_frozen(registration, frozen, campaign.pinned_producer(registration, producer_commit))
    print(json.dumps(_cold(registration, reg["cases"][key], pub["rows"][f"{key}/{arm}"]["endpoints"]["selected"])))

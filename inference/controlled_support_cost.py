"""Paired end-to-end native intervention timing, including process cold start."""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from inference.native_support_checkpoint import load_endpoint
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code, tensor_hash
from transfer import controlled_support_campaign as campaign, native_support_campaign as native
from transfer import registered_development as dev
from transfer.controlled_regional_support import ARMS
from transfer.native_support_intervention import forward_intervention
from transfer.controlled_support_evidence import _bound


def _inputs(pin, code, producer, evidence_pin):
    _, reg, old, frozen, _, _ = campaign.load(pin, code, producer_commit=producer)
    evidence = campaign.read(evidence_pin)
    if evidence["registration"] != pin or evidence["updates"] or evidence["query"]:
        raise ValueError("Wrong frozen intervention evidence")
    identity = "wp08-nine8-64/fixed"
    key = identity.split("/")[0]
    package = frozen["rows"][identity]["endpoints"]["last"]
    data = native._case_data(old["cases"][key])
    _, scene, record, _, bank = data
    row = next(row for row in evidence["synthetic"][identity + "/last"]["rows"] if row["block"] == "2")
    receipt = row["receipt"]
    if receipt["match"]["status"] != "MATCHED" or receipt["owner"]["endpoint"] != package:
        raise ValueError("Predeclared cost tile unavailable")
    return reg, old["cases"][key], package, data, row


@torch.no_grad()
def _sample(model, data, case, row, arm):
    _, scene, record, _, bank = data
    receipt = row["receipt"]
    block = receipt["block"]
    tile = scene.plan["tiles"][int(block)]
    started = perf_counter()
    bound = _bound(bank[block], case, receipt, arm, receipt["owner"])
    prepared = perf_counter()
    timings = {}
    output, frame = forward_intervention(model, scene, record, tile, bound, detail=True, timing=timings)
    finished = perf_counter()
    hashes = [tensor_hash(x) for x in output[:3]]
    if hashes != row["traces"][arm]["probabilities"]:
        raise ValueError("CPU cost workload differs from fixed endpoint intervention")
    return dict(prepare=prepared - started, forward=finished - prepared, seconds=finished - started,
                stages=timings, score=bound.cost["scoring_seconds"], sparse=bound.cost["sparse_build_seconds"],
                nnz=frame["trace"]["stage2_nnz"], shape=list(frame["stage2"].shape), probability_hashes=hashes)


def _stats(rows, name):
    values = [row[name] for row in rows]
    lo, mid, hi = np.quantile(values, [.1, .5, .9])
    return dict(p10=float(lo), median=float(mid), p90=float(hi))


def cost(pin, code, output, evidence, *, producer_commit=None):
    started = perf_counter()
    reg, case, package, data, row = _inputs(pin, code, producer_commit, evidence)
    data_cost = perf_counter() - started
    started = perf_counter()
    model, payload = load_endpoint(package, reg["original"]["registration"], case, data)
    load_cost = perf_counter() - started
    warm, cold = ({arm: [] for arm in ARMS} for _ in range(2))
    for arm in ARMS:
        _sample(model, data, case, row, arm)
    for index in range(8):
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            warm[arm].append(dict(pair=index, **_sample(model, data, case, row, arm)))
    command = [sys.executable, "-m", "inference.controlled_support_cost", code["commit"],
               reg["code"]["commit"], pin["path"], pin["sha256"], evidence["path"], evidence["sha256"]]
    for index in range(3):
        for arm in ARMS[index:] + ARMS[:index]:
            start = perf_counter()
            proc = subprocess.run(command + [arm], cwd=ROOT, check=True, capture_output=True, text=True)
            cold[arm].append(dict(pair=index, wall=perf_counter() - start, **json.loads(proc.stdout)))
    summary = {arm: dict(warm=_stats(warm[arm], "seconds"), cold=_stats(cold[arm], "wall"),
                         paired_vs_reference=_stats([{"delta": a["seconds"] - b["seconds"]}
                             for a, b in zip(warm[arm], warm["reference"], strict=True)], "delta")) for arm in ARMS}
    return dev._json_new(external_directory(output), dict(registration=pin, intervention=evidence, code=code["commit"],
        scope="synthetic_devtrain_cpu_fp32", endpoint=package, block=row["block"], parameters=sum(p.numel() for p in model.parameters()),
        package_bytes=Path(package["path"]).stat().st_size, model_state=payload["state_hash"],
        search_cost=row["search_cost"], input_seconds=data_cost, model_load_seconds=load_cost, warm=warm, cold=cold,
        summary=summary, nested_timing=True, filesystem_cache_flushed=False, updates=0, query=0))


if __name__ == "__main__":
    expected, producer, path, checksum, intervention_path, intervention_sha, arm = sys.argv[1:]
    torch.set_num_threads(1)
    code = code_snapshot()
    require_frozen_code(code)
    if code["commit"] != expected:
        raise ValueError("Wrong cold worker SHA")
    started = perf_counter()
    reg, case, package, data, row = _inputs(dict(path=path, sha256=checksum), code, producer,
                                            dict(path=intervention_path, sha256=intervention_sha))
    prepared = perf_counter()
    model, _ = load_endpoint(package, reg["original"]["registration"], case, data)
    loaded = perf_counter()
    result = _sample(model, data, case, row, arm)
    print(json.dumps(dict(data_seconds=prepared - started, model_load_seconds=loaded - prepared, sample=result)))

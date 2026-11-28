"""Registered CPU diagnostics for the existing WP11 Xuzhou devval release."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from inference.native_deployment import matched_random
from inference.wp11_frozen_deployment import open_runtime, read_inputs, sha


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("light", "full", "geometry", "matched_random", "nonfinite_light")
REPEATS = 6
SEED = 64


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(requests, input_sha, commit):
    cohorts = defaultdict(list)
    for r in requests:
        cohorts[r.cohort].append(r.key)
    fast = sorted(key for keys in cohorts.values() for key in sorted(keys)[:(len(keys)+1)//2])
    reference = {"routes": [{"key":r.key, "route":"light" if r.key in fast else "full"}
                             for r in requests]}
    random_fast = matched_random(requests, reference, SEED, 0.)
    workloads = {r.key:list(r.x.shape) + [r.waves.numel(), r.scale] for r in requests}
    if (fast == random_fast or not 0 < len(fast) < len(requests)
            or Counter(tuple(workloads[k]) for k in fast)
            != Counter(tuple(workloads[k]) for k in random_fast)):
        raise ValueError("Distinct random control with identical workload counts required")
    return {"commit":commit, "inputs_sha256":input_sha, "repeats":REPEATS, "random_seed":SEED,
            "geometry_fast":fast, "matched_random_fast":random_fast, "fault":fast[0],
            "requests":[r.key for r in requests], "workload":workloads,
            "rule":"First half rounded up by stable tile key within each native workload cohort",
            "scope":"Existing Xuzhou devval only", "optimizer_updates":0, "query_reads":0}


def worker(base, plan, arm, out):
    start = perf_counter()
    inputs, requests = read_inputs(base, plan["inputs_sha256"])
    input_ms = (perf_counter()-start)*1000
    loaded = []
    start = perf_counter()
    runtime = open_runtime(base, inputs, arm, full_loads=loaded)
    setup_ms = (perf_counter()-start)*1000
    start = perf_counter()
    fast = plan["matched_random_fast"] if arm == "matched_random" else plan["geometry_fast"]
    mode = arm if arm in ("light", "full") else "random"
    selection_ms = (perf_counter()-start)*1000
    context = nullcontext()
    if arm == "nonfinite_light":
        head = runtime.light.heads["xuzhou"]
        actual = head.forward
        target = plan["requests"].index(plan["fault"])
        calls = iter(range(len(requests)))

        def fault(x):
            result = actual(x)
            return torch.full_like(result, float("nan")) if next(calls) == target else result

        context = patch.object(head, "forward", fault)
    with context:
        result = runtime.run(requests, mode, random_fast=fast)
    outputs = {r.key:tuple(p.detach().cpu() for p in values)
               for r, values in zip(requests, result.pop("predictions"), strict=True)}
    if ((arm == "light" and loaded) or (arm != "light" and len(loaded)!=1)
            or result["counts"]["full_calls"] != result["counts"]["graph_builds"]
            or result["counts"]["graph_reuses"]):
        raise ValueError("Observed weight/graph executions contradict the frozen protocol")
    with out.with_suffix(".pt").open("xb") as stream:
        torch.save(outputs, stream)
    counts = result["counts"]
    if counts["light_failures"] != (arm == "nonfinite_light"):
        raise ValueError("Unexpected numerical failure count")
    if arm == "nonfinite_light" and not any(
        row["key"] == plan["fault"] and row["route"] == "full" and row["reason"] == "light_failure"
        for row in result["routes"]
    ):
        raise ValueError("Injected numerical error did not reach the real fallback")
    result.update(input_ms=input_ms, setup_ms=setup_ms, selection_ms=selection_ms,
                  full_checkpoint_loads=len(loaded), full_checkpoint_ms=sum(loaded),
                  rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  outputs_sha256=sha(out.with_suffix(".pt")))
    write(out, result)


def summarize(out, base, plan, inputs):
    label_file = base/"labels.json"
    if sha(label_file) != inputs["files"]["labels.json"]:
        raise ValueError("Frozen V5 support differs")
    labels = json.loads(label_file.read_text())
    if sum(map(len, labels.values())) != 45 or set(labels) != set(plan["requests"]):
        raise ValueError("Only the original 45 V5 labels are admitted")
    rows = []
    for arm in ARMS:
        samples = [json.loads((out/f"{i}-{arm}.json").read_text()) for i in range(REPEATS)]
        observed = samples[0]["counts"]
        if any(s["counts"] != observed or s["routes"] != samples[0]["routes"] for s in samples):
            raise ValueError("Frozen routing drifted across cold processes")
        row = dict(arm=arm, repeats=REPEATS, **observed, full_checkpoint_loads=samples[0]["full_checkpoint_loads"])
        for field in ("process_ms", "input_ms", "setup_ms", "selection_ms", "elapsed_ms", "graph_ms",
                      "full_checkpoint_ms"):
            row[field+"_p50"], row[field+"_p95"] = map(float, np.percentile([s[field] for s in samples], [50,95]))
        row["peak_rss_mib"] = max(s["rss_kib"] for s in samples)/1024
        rows.append(row)
    quality = {}
    for i in range(REPEATS):
        def predictions(arm):
            path = out/f"{i}-{arm}.pt"
            sample = json.loads(path.with_suffix(".json").read_text())
            if sha(path) != sample["outputs_sha256"]:
                raise ValueError("Frozen output changed")
            return sample, torch.load(path, map_location="cpu", weights_only=True)

        _, light = predictions("light")
        _, full = predictions("full")
        for arm in ("geometry", "matched_random", "nonfinite_light"):
            sample, values = predictions(arm)
            split = {row["key"]:row["route"] for row in sample["routes"]}
            counts = {c:Counter(support=0, accepted=0, wrong_exit=0, full_correct_wrong_exit=0)
                      for c in range(1,10)}
            for key, path in split.items():
                expected = light[key] if path == "light" else full[key]
                for actual, ref in zip(values[key], expected, strict=True):
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                predicted = values[key][0].argmax(1)+1
                baseline = full[key][0].argmax(1)+1
                for index, cls in labels[key]:
                    cls_count = counts[cls]
                    cls_count["support"] += 1
                    if path == "light":
                        cls_count["accepted"] += 1
                        cls_count["wrong_exit"] += int(predicted[index] != cls)
                        cls_count["full_correct_wrong_exit"] += int(predicted[index] != cls and baseline[index] == cls)
            if i and counts != quality[arm]:
                raise ValueError("Repeated V5 readout differs")
            quality[arm] = counts
    for name, records in (("costs", rows), ("devval", [
        dict(arm=arm, class_id=cls, **row) for arm, classes in quality.items() for cls, row in classes.items()
    ])):
        with (out/f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, records[0])
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    commit = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"], text=True).strip()
    if (sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or commit != args.expected_commit
            or subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"])):
        raise ValueError("Frozen clean Linux checkout and explicitly hidden GPUs required")
    torch.set_num_threads(1)
    if args.out.resolve().is_relative_to(ROOT):
        raise ValueError("Artifacts belong outside the checkout")
    plan_path = args.out/"plan.json"
    if args.worker:
        plan = json.loads(plan_path.read_text())
        if plan["commit"] != commit or plan["inputs_sha256"] != args.input_sha:
            raise ValueError("Worker plan differs")
        worker(args.inputs, plan, args.worker, args.out/f"{args.repeat}-{args.worker}.json")
        return
    inputs, requests = read_inputs(args.inputs, args.input_sha)
    plan = freeze(requests, args.input_sha, commit)
    args.out.mkdir()
    write(plan_path, plan)
    for i in range(REPEATS):
        for arm in ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)]:
            command = [sys.executable, "-B", "-m", "scripts.run_wp11_deferred_diagnostic", *sys.argv[1:],
                       "--worker", arm, "--repeat", str(i)]
            start = perf_counter()
            subprocess.run(command, cwd=ROOT, check=True)
            latency = (perf_counter()-start)*1000
            sample = args.out/f"{i}-{arm}.json"
            row = json.loads(sample.read_text())
            row.update(process_ms=latency)
            sample.write_text(json.dumps(row)+"\n")
    summarize(args.out, args.inputs, plan, inputs)


if __name__ == "__main__":
    main()

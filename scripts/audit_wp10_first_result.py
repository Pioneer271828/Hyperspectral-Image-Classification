"""Verify WP10 first-result receipts and place them in the existing WP12 cost fields."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

SCOPE = "parent_launch_to_first_verified_full_three_head_result"
BASE_WP12 = "1b7cd18eb81f19a07ad9ad5104b1c1721264a2f045c48fb5de78dcaa0c9c53d8"


def pin(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return {"path": str(path), "sha256": sha.hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def checked(ref):
    path = Path(ref["path"])
    if pin(path) != ref:
        raise ValueError(f"Wrong evidence digest: {path}")
    return read(path)


def expect(ok, message):
    if not ok:
        raise ValueError(message)


def ordered_samples(work, schedule):
    expect(len(schedule) == 54 and set(work.glob("*.sample.json")) == {
        work / f"{index:02d}-{entry['case']}.sample.json" for index, entry in enumerate(schedule)
    }, "Expected exactly the 54 scheduled sample files")
    for index, entry in enumerate(schedule):
        file = work / f"{index:02d}-{entry['case']}.sample.json"
        sample = read(file)
        expect(type(sample["index"]) is int and sample["index"] == index
               and type(sample["repeat"]) is int and sample["repeat"] == entry["repeat"]
               and sample["case"] == entry["case"], "Sample index, repeat or case differs from the frozen schedule")
        yield file, sample


def verify_timing(data, clock):
    sequence = [data["start_ns"], *(clock[k] for k in ("python", "initialized", "inputs", "model", "forward", "verified")),
                data["accepted_ns"], data["exit_ns"]]
    expect(all(type(ns) is int for ns in sequence) and sequence == sorted(sequence),
           "First result clocks are out of order or have the wrong type")
    names = ("launch_to_python", "initialize", "authenticate_and_prepare_input", "load_authenticated_package",
             "first_full_graph_and_forward", "validate_and_hash_outputs", "serialize_deliver_and_verify", "exit_after_result")
    stages = {name: (sequence[i + 1] - sequence[i]) / 1e6 for i, name in enumerate(names)}
    expect(data["stages_ms"] == stages, "Measured stage duration differs from the raw clocks")
    ms = (data["accepted_ns"] - data["start_ns"]) / 1e6
    expect(0 < ms == data["first_result_ms"]
           and data["subprocess_exit_ms"] == (data["exit_ns"] - data["start_ns"]) / 1e6,
           "First result timing includes exit or differs from clock evidence")
    return ms


def apply_cost(report, work, historical, manifest, schedule):
    expect(report["status"] == "production_development_evidence", "WP12 production report required")
    timings = {}
    seen = set()
    for file, sample in ordered_samples(work, schedule):
        row = sample["result"]
        name = sample["case"]
        expect(row == read(file.with_name(file.name.replace(".sample.json", ".stdout.log"))),
               "First result differs from the received subprocess line")
        start = historical["startup"][name][0]
        package = manifest["cases"][name]["package"]
        expect(row["case"] == name and row["pid"] not in seen and row["cpu"] == 0 and row["threads"] == 1
               and row["manifest"] == historical["manifest"] and row["tiles"] == manifest["tiles"]
               and row["state"] == package["state"] and row["package_bytes"] == package["bytes"]
               and row["parameters"] == package["parameters"] and row["factor_calls"] == start["factor_calls"]
               and row["output_hashes"] == start["output_hashes"]
               and all(row[k] == 0 for k in ("query", "gt", "optimizer_updates", "calibration")),
               "Subprocess result differs from the original authorized work")
        ms = verify_timing(sample["timing"], row["clocks_ns"])
        seen.add(row["pid"])
        timings.setdefault(name, []).append((sample["repeat"], ms))
    updated = set()
    for row in report["rows"]:
        name = row.get("case_id")
        if row["module"] != "WP10-LR" or name not in timings:
            continue
        cost = row["cost"]
        pkg = manifest["cases"][name]["package"]
        expect(row["identity"]["data_protocol"] == "wp10_xuzhou_k5_v5"
               and row["state"] == pkg["state"] and row["package"] == pkg["weights"]["sha256"]
               and cost["blocks"] == 5 and cost["pixels"] == 3920 and cost["device"] == "cpu"
               and cost["end_to_end_ms"]["status"] == "NOT_RUN", "Cost target is outside the original scope")
        repeats = sorted(timings[name])
        expect([n for n, _ in repeats] == [0, 1, 2], "Each original package requires one sample per round")
        values = [v for _, v in repeats]
        cost["end_to_end_ms"] = dict(status="MEASURED", p50=median(values), p95=None, n=3, samples=values,
                                      range_ms=[min(values), max(values)], scope=SCOPE, cpu=0)
        updated.add(name)
    expect(updated == set(timings) == set(manifest["cases"]), "Expected all and only the original 18 cost rows")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("measurement", "baseline-table", "adapter", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    work = args.measurement.resolve()
    receipt_pin = dict(path=str(work / "receipt.json"), sha256=args.receipt_sha256)
    receipt = checked(receipt_pin)
    plan = checked(receipt["registration"])
    expect(receipt["scope"] == plan["scope"] == SCOPE and receipt["status"] == "passed"
           and receipt["first_results"] == 54 and plan["cpu"] == 0 and plan["blocks"] == 5 and plan["pixels"] == 3920
           and plan["input_role"] == "devval_image_only" and plan["repeats"] == 3 and plan["warmups"] == 0,
           "Measured receipt scope differs from the fixed CPU workload")
    historical = checked(next(ref for ref in plan["source_pins"] if ref["path"].endswith("wp10-low-rank-cpu-cost01/receipt.json")))
    manifest = checked(plan["manifest"])
    base = dict(path=str(args.baseline_table), sha256=BASE_WP12)
    report = checked(base)
    updated = apply_cost(report, work, historical, manifest, plan["schedules"])
    reference = receipt["summary"]
    for row in report["rows"]:
        if row.get("case_id") in updated and row["module"] == "WP10-LR":
            cost, observed = row["cost"]["end_to_end_ms"], reference[row["case_id"]]
            expect(cost["samples"] == observed["first_result_ms"] and cost["p50"] == observed["median_ms"],
                   "Independent raw sample arithmetic differs from the measured receipt")
    sys.path.insert(0, str(args.adapter.resolve()))
    from evaluation.wp12_production_evidence import write_table

    args.out.mkdir()
    write_table(args.out, dict(report, first_result_evidence=receipt_pin, previous_production_table=base))
    print(json.dumps(dict(status="passed", cases=len(updated), processes=54, three_head_block_hashes=54*5*3)))


if __name__ == "__main__":
    main()

"""Measure real factor packages in matched CPU groups and isolated first-sweep workers."""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from evaluation.wp12_cost_primitives import timing_summary
from inference.wp10_low_rank_execution import resident
from research_runtime import ROOT, external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev


def cost(pin, code, out):
    out = external_directory(out)
    if out.exists() or out in Path(pin["path"]).resolve().parents:
        raise ValueError("Fresh external cost directory required")
    out.mkdir()
    start = perf_counter()
    reg, groups = resident(pin, code)
    resident_seconds = perf_counter() - start
    resident_pin = dev._json_new(out / "resident.json", groups)
    known = {}
    for row in groups.values():
        for arm, name in row["cases"].items():
            result = row["timing"]["output_hashes"][arm]
            if known.setdefault(name, result) != result:
                raise ValueError("Matched dense group output changed")
    commands, workers = [], {key: [] for key in reg["case_order"]}
    program = str(ROOT / "scripts" / "run_frozen_cpu_deployment.py")
    names = list(workers)
    started = perf_counter()
    for repeat in range(3):
        for name in names[repeat:] + names[:repeat]:
            command = [sys.executable, program, "--expected-commit", code["commit"], "startup",
                       "--input", pin["path"], pin["sha256"], "--mode", name]
            log, err = (out / f"startup-{repeat}-{name}.{ext}.log" for ext in ("stdout", "stderr"))
            with log.open("x") as stdout, err.open("x") as stderr:
                subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
            row = json.loads(log.read_text())
            entry = reg["cases"][name]["package"]
            if (row["case"] != name or row["manifest"] != pin or row["tiles"] != reg["tiles"]
                    or row["output_hashes"] != known[name] or row["state"] != entry["state"]
                    or row["package_bytes"] != entry["bytes"] or row["parameters"] != entry["parameters"]
                    or any(row[k] != 0 for k in ("query", "gt", "optimizer_updates"))):
                raise ValueError("Independent startup disagrees with the resident package")
            commands.append(command)
            workers[name].append(row)
    startup_seconds = perf_counter() - started
    if len({row["pid"] for rows in workers.values() for row in rows}) != 3*len(names):
        raise ValueError("Expected a fresh process for every package sweep")
    summary = {}
    for name, rows in workers.items():
        values = dict(load_ms=[v["load_ms"] for v in rows], first_sweep_ms=[v["first_sweep_ms"] for v in rows],
                      peak_rss_kib=[v["after_first_sweep"]["peak_rss_kib"] for v in rows])
        summary[name] = {key: timing_summary(v, repeats=3, blocks=len(reg["tiles"])) for key, v in values.items()}
    admission.check_pins([pin, reg["resident_inputs"], reg["registration"], reg["audit"],
                          *[r["package"][key] for r in reg["cases"].values() for key in ("weights", "sidecar")]])
    require_frozen_code(code)
    return dev._json_new(out / "receipt.json", dict(status="passed_low_rank_cpu_cost", manifest=pin,
        registration=reg["registration"], audit=reg["audit"], commit=code["commit"],
        resident=resident_pin, startup=workers, startup_summary=summary, commands=commands,
        resident_seconds=resident_seconds, startup_seconds=startup_seconds, filesystem_cache="uncontrolled",
        reduction_and_expansion_measured_separately=True, optimizer_updates=0, gt=0, query=0))

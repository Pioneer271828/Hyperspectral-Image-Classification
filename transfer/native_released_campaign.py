"""Prepare, execute and independently audit the frozen released-point matrix."""
from collections import Counter
import csv
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.classification_metrics import metrics_from_confusion
from model.research_config import digest
from research_runtime import require_frozen_code
from transfer import native_released_preflight as preflight, native_released_real as real
from transfer import native_released_training as released, p4_runtime_device as device
from transfer import registered_development as dev
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_training import Run, SEEDS
from transfer.xuzhou_multiscale_audit import _exact


MATRIX = "wp07_native_released_comparison_matrix_v1"
RELEASE = "wp07_native_released_comparison_release_v1"
AUDIT = "wp07_native_released_comparison_audit_v1"
REAL_PRE_FLIGHT_SHA = "22afb23982d1a8559ee71f5cc197934675485ac58cf1273245ab3267600ada10"
ROUTES = {"full": ("full", "completed"), "split": ("prefix", "resume", "completed")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_case_names(scene):
    return [f"{scene}/{seed}/{factor}/{arm}" for seed, (factor, arm) in product(SEEDS, released.COMPARISONS)]


def prepare(root, out, execution, preflight_pin, code, *, engineering=False):
    require_frozen_code(code)
    request = read(preflight_pin, root)
    expected_schema = real.ENGINEERING if engineering else preflight.SCHEMA
    require(request["status"] == "preflight_only" and request["schema"] == expected_schema
            and request["optimization_updates"] == 0
            and request["raw_gt_loaded"] is request["query_accessed"] is False
            and set(request["scenes"]) == set(preflight.SCENES)
            and (engineering or preflight_pin["sha256"] == REAL_PRE_FLIGHT_SHA), "Unrelated frozen comparison preflight")
    require(out != execution and all(root in p.parents and p == p.resolve() and not p.exists()
                                      for p in (out, execution)), "Expected empty isolated matrix and execution paths")
    cases = []
    for name in preflight.SCENES:
        pin = request["scenes"][name]
        receipt = read(pin, root)
        schedule_rows = {row["id"]: row for row in receipt["draft"]["cases"]}
        order = expected_case_names(name)
        require(receipt["scene"] == name and receipt["schema"] == expected_schema
                and set(schedule_rows) == set(order) and len(receipt["draft"]["cases"]) == 15
                and receipt["draft"]["authorized_for_training"] is False, "Frozen complete 15-case scene draft required")
        steps = len(schedule_rows[order[0]]["schedule"]) if engineering else 2 * preflight.EXPECTED[name][-1]
        for key in order:
            row = schedule_rows[key]
            seed, factor, arm = key.split("/")[1:]
            require((row["model_seed"], row["factor"], row["arm"]) == (int(seed), int(factor), arm)
                    and row["planned_updates"] == steps and row["events"] == [0, steps//2, steps]
                    and len(row["schedule"]) == steps
                    and (engineering or row["train_exposures_per_class"] == [10]*preflight.EXPECTED[name][0]),
                    "Fitting role, budget, exposure or calendar differs")
            cases.append({"id": key, "preflight": pin, "sha256": digest(row), "updates": steps,
                          "destination": str(execution / key.replace("/", "-"))})
        for seed in SEEDS:
            group = [schedule_rows[key] for key in order if key.split("/")[1] == str(seed)]
            require(all(row["schedule"] == group[0]["schedule"] for row in group), "Paired exposure schedule differs")
    budget = sum(row["updates"] for row in cases)
    require(engineering or budget == 2910, "Original real comparison budget is 2910 updates")
    out.mkdir()
    value = {"schema": MATRIX, "controller": {"commit": code["commit"], "source_hash": code["source_hash"]},
             "synthetic": engineering, "preflight": preflight_pin, "cases": cases, "optimizer_updates": budget,
             "execution": str(execution), "authorization": "separate_release_required",
             "optimization_updates": 0, "query_authorized": False}
    return dev._json_new(out / "matrix.json", value)


def load(root, item, code, release, *, engineering):
    require_frozen_code(code)
    spec = read(item, root)
    require(spec["schema"] == MATRIX and spec["synthetic"] is engineering
            and spec["controller"] == {"commit": code["commit"], "source_hash": code["source_hash"]}
            and spec["query_authorized"] is False and spec["optimization_updates"] == 0 and len(spec["cases"]) == 30,
            "Wrong matrix source, scope or matrix cardinality")
    require(release is not None, "Only prepare is permitted without a separate matrix release")
    approved = read(release, root)
    keys = {"schema", "decision", "matrix", "execution", "synthetic", "optimizer_updates", "cases", "query_authorized"}
    require(set(approved) == keys and approved["schema"] == RELEASE and approved["decision"] == "approved"
            and approved["matrix"] == item and approved["synthetic"] is engineering
            and approved["execution"] == spec["execution"] and approved["optimizer_updates"] == spec["optimizer_updates"]
            and approved["query_authorized"] is False
            and set(approved["cases"]) == {case["id"] for case in spec["cases"]}, "Matrix release differs")
    return spec, approved["cases"]


def receipt(root, out, phase, spec_pin):
    item = dev._pin(out / f"{phase}.json")
    row = read(item, root)
    require(row["phase"] == phase and row["registration"] == spec_pin
            and row["checkpoint"]["path"] == str(out / f"{phase}.pt"),
            "Completed sparse stage receipt does not belong to the case")
    previous = packet(row["checkpoint"], root)
    return item, row, previous


def authorized_case(root, case, proof, code, *, intake):
    out = Path(case["destination"])
    raw = read(case["preflight"], root)
    chosen = [row for row in raw["draft"]["cases"] if row["id"] == case["id"]]
    require(len(chosen) == 1 and case["sha256"] == digest(chosen[0]), "Prepared case differs from its preflight")
    guard = real.RealReleasedRun(root, case["preflight"], case["id"], proof, out, code, intake=intake)
    return guard, out


def run(root, item, release, code, *, intake=None, cli):
    spec, approved = load(root, item, code, release, engineering=intake is not None)
    out = Path(spec["execution"])
    require(out == out.resolve() and root in out.parents
            and all(Path(case["destination"]).parent == out for case in spec["cases"]),
            "Matrix execution paths escape the authorized root")
    out.mkdir(exist_ok=True)
    started = perf_counter()
    counts = Counter()
    with device.lease(out / ".matrix.lock"):
        owner = {"matrix": item, "release": release}
        path = out / "owner.json"
        if path.is_file():
            require(read(dev._pin(path), root) == owner, "Matrix execution owner differs")
        else:
            require(set(p.name for p in out.iterdir()) == {".matrix.lock"}, "Unowned execution directory")
            dev._json_new(path, owner)
        for case in spec["cases"]:
            proof = approved[case["id"]]
            guard, dest = authorized_case(root, case, proof, code, intake=intake)
            for phase in ROUTES[guard.route]:
                if (dest / f"{phase}.json").is_file():
                    pin = dev._pin(dest / "registration.json")
                    _, row, previous = receipt(root, dest, phase, pin)
                    frozen = read(pin, root)
                    chosen = guard.case
                    expected = guard.contract(guard.classes, chosen["model_seed"], chosen["arm"], chosen["factor"],
                                              code["commit"], numeric=frozen["numeric"])
                    require(frozen == expected and previous["registration"] == digest(frozen)
                            and previous["cursor"] == {"full": frozen["steps"], "prefix": frozen["prefix"],
                                                       "resume": frozen["steps"], "completed": frozen["steps"]}[phase],
                            "Committed phase cannot be reused under a different science contract")
                    continue
                require(not (dest / f"{phase}.intent.json").exists(), "Interrupted sparse stage retains its execution intent")
                try:
                    pin = cli.real_stage(root, dest, case["preflight"], proof, case["id"], phase,
                                         code, guard.physical_uuid, intake=intake)
                except device.ResourceUnavailable as error:
                    return {"status": "waiting_for_resources", "case": case["id"], "reason": str(error),
                            "new_updates": counts["updates"], "elapsed_seconds": perf_counter()-started}
                row = read(pin, root)
                counts["updates"] += row["updates"]
                counts["stages"] += 1
        return {"status": "completed", "new_updates": counts["updates"], "new_stages": counts["stages"],
                "elapsed_seconds": perf_counter()-started}


def scoring(endpoint, historical, classes):
    require([row["tile"] for row in endpoint["rows"]] == [row["tile"] for row in historical["rows"]],
            "Independent score replay changed the validation order")
    head_results = {}
    for head in ("fused", "graph", "cnn"):
        buckets = {}
        for observed, saved in zip(endpoint["rows"], historical["rows"], strict=True):
            got, old = observed["scores"][head], saved["scores"][head]
            require(all(got[key] == old[key] for key in ("nll_sum", "count"))
                    and observed["heads"][head] == saved["heads"][head]
                    and got["point_confusion"] == got["coarse_known_count_confusion"],
                    "Independent released-point validation differs from the selected checkpoint")
            key = observed["band"]
            bag = buckets.setdefault(key, {"matrix": np.zeros((classes, classes), dtype=np.int64), "nll": 0., "count": 0})
            bag["matrix"] += got["point_confusion"]
            bag["nll"] += got["nll_sum"]
            bag["count"] += got["count"]
        head_results[head] = {band: {**metrics_from_confusion(row["matrix"], range(1, classes+1)),
                                    "nll": row["nll"]/row["count"]} for band, row in buckets.items()}
    return head_results


def audit_case(root, case, proof, code, *, intake, cli):
    guard, out = authorized_case(root, case, proof, code, intake=intake)
    spec_pin = dev._pin(out / "registration.json")
    spec = read(spec_pin, root)
    chosen = guard.case
    records = [receipt(root, out, phase, spec_pin) for phase in ROUTES[guard.route]]
    pins, rows, packets = zip(*records, strict=True)
    cursor = 0
    for row, packet_value in zip(rows, packets, strict=True):
        cursor += row["updates"]
        require(cursor == packet_value["cursor"] and packet_value["registration"] == digest(spec),
                "Sparse scientific checkpoint, budget or cursor differs")
    state = packets[-2]
    require(_exact(state, packets[-1]) and rows[-1]["updates"] == 0 and cursor == case["updates"],
            "Completed sparse recovery changed the state or budget")
    with cli.arithmetic(root, guard.physical_uuid) as execution:
        run = Run(spec, code["commit"], execution=execution, released=guard)
        run.restore(state, cursor)
        selected = min(state["validation"], key=lambda row: row["fused_nll"])
        run.model.load_state_dict(state["selected"]["state"], strict=True)
        evaluated = run.evaluate(selected["step"], detailed=True)
    require(evaluated["fused_nll"] == selected["fused_nll"], "First-min checkpoint or validation prediction differs")
    exposed = np.sum(np.asarray([row["released"]["per_class"] for row in state["history"]]), axis=0).tolist()
    require(exposed == chosen["train_exposures_per_class"] and spec["events"] == [v["step"] for v in state["validation"]],
            "Released per-class exposure or validation calendar differs")
    return {"case": case["id"], "optimizer_updates": cursor, "completed_resume_updates": 0,
            "selected_step": selected["step"], "fused_nll": selected["fused_nll"],
            "head_metrics": scoring(evaluated, selected, guard.classes), "released_train_exposures": exposed,
            "validation": evaluated, "training_exposure": [row["released"] for row in state["history"]],
            "first_loss": state["history"][0]["loss"], "last_loss": state["history"][-1]["loss"],
            "source_state": spec["source_state"], "initialization": spec["initialization"],
            "parameters": spec["parameters"], "stages": list(pins), "timing": [row["timing"] for row in rows]}


def audit(root, item, release, out, code, *, intake=None, cli):
    spec, approvals = load(root, item, code, release, engineering=intake is not None)
    execution = Path(spec["execution"])
    started = perf_counter()
    with device.lease(execution / ".matrix.lock"):
        owner = read(dev._pin(execution / "owner.json"), root)
        require(owner == {"release": release, "matrix": item}, "Sparse audit owner differs")
        rows = [audit_case(root, case, approvals[case["id"]], code, intake=intake, cli=cli) for case in spec["cases"]]
    require(sum(row["optimizer_updates"] for row in rows) == spec["optimizer_updates"], "Incomplete matrix")
    paired = {}
    for row in rows:
        scene, seed, _, _ = row["case"].split("/")
        key = (scene, seed)
        require(paired.setdefault(key, row["source_state"]) == row["source_state"],
                "Paired cases do not share the same seeded pre-training weights")
    require(all(len({row["source_state"] for row in rows if row["case"].startswith(scene + "/")}) == 3
                for scene in preflight.SCENES), "The matrix lacks three distinct seeded initializations")
    out.mkdir()
    export_csv(out / "selected-devval.csv", rows, spec["synthetic"])
    result = {"schema": AUDIT, "status": "passed", "synthetic": spec["synthetic"], "matrix": item,
              "release": release, "producer": code["commit"], "cases": rows, "optimizer_updates": spec["optimizer_updates"],
              "audit_optimizer_updates": 0, "elapsed_seconds": perf_counter()-started, "query_authorized": False}
    result["selected_devval_table"] = dev._pin(out / "selected-devval.csv")
    return dev._json_new(out / "comparison-audit.json", result)


def export_csv(path, rows, synthetic):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scope", "case", "head", "band", "selected_step", "oa", "aa", "kappa", "macro_f1",
                         "nll", "class_id", "released_support", "recall"))
        for row in rows:
            for head, by_band in row["head_metrics"].items():
                for band, observed in by_band.items():
                    for per_class in observed["per_class"]:
                        writer.writerow(("synthetic" if synthetic else "released_devval", row["case"], head, band,
                            row["selected_step"], *(observed[name] for name in ("oa", "aa", "kappa", "macro_f1", "nll")),
                            *(per_class[name] for name in ("class_id", "support", "recall"))))

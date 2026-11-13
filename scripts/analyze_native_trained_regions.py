"""Register and execute post-training interventions on authenticated released banks."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.research_config import digest
from monitoring import native_region_analysis as analysis
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_native_released_points import arithmetic
from tests.native_released_bank_fixture import release_fixture
from transfer import development_bank as bank, development_bank_training as original
from transfer import native_released_preflight as preflight, native_released_real as real
from transfer import native_released_training as released, registered_development as transport
from transfer.development_block_probe import VIEWS
from transfer.g3_training import state_hash
from transfer.native_released_campaign import REAL_PRE_FLIGHT_SHA
from transfer.native_released_points import bind
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_observation import from_bank, forward
from transfer.native_scale_training import Run


DUTY_SHA = "f6c1a4d706167a002d9b639378b2f33acf7ed403f07fa5f1dd0559a40c9f0547"
SPECTRAL_SHA = "f0c7242aacb8b22edb15d4a5901e1b3b4632b7836d7dc25c0dfbfcfdb51a00"
DUTY_PRODUCER = "570dc46aa9f721feafd253aaa2ccc496dcf6e1b3"
SPECTRAL_PRODUCER = "19c5ff2e3916f7bbe91146ba779e6298a057fcce"
GEOMETRY_RULE = "dominant sparse Q with lowest row tie; most/least assigned native cells and first partial edge; closest donor and farthest source by native centers"
ARM_RULE = "stage2 output self/zero/replacement, stage2 input source swap and derivative; only registered graph range"


def roster(root, inputs, code):
    duty, spectral, blocked = [read(inputs[key], root) for key in ("duty", "spectral", "real")]
    analysis.require([inputs[key]["sha256"] for key in ("duty", "spectral", "real")] ==
                     [DUTY_SHA, SPECTRAL_SHA, REAL_PRE_FLIGHT_SHA]
                     and duty["producer"] == DUTY_PRODUCER and duty["synthetic"] is True and duty["status"] == "passed"
                     and duty["optimizer_updates"] == 1368 and len(duty["cases"]) == 114
                     and spectral["producer"] == SPECTRAL_PRODUCER and spectral["status"] == "passed"
                     and spectral["scope"] == "synthetic_engineering" and spectral["optimizer_updates"] == 360
                     and blocked["status"] == "preflight_only" and blocked["optimization_updates"] == 0,
                     "Sources must be the audited duties, spectral controls and real development banks")
    # A compact roster exercises small real-shaped duties with no donor and
    # existing multi-region selected checkpoints with both graph ranges.
    cases = []
    for row in duty["cases"]:
        kind = row["case"].split("/")
        if kind[1] in ("64", "512") and "/".join(kind[2:]) in ("1/off", "2/joint", "4/graph_content"):
            seal = row["stages"][-1]
            receipt = read(seal, root)
            cases.append({"id": row["case"], "source": "bank", "registration": receipt["registration"],
                          "stage": seal, "checkpoint": receipt["checkpoint"], "selected_step": row["selected_step"]})
    for key, item in sorted(spectral["cases"].items()):
        if not key.endswith("f2-joint") or "-512-" not in key:
            continue
        seal = item["stages"][-1]
        analysis.require(item["exact_full_split_complete"] is True, "Spectral selected state is not accepted")
        cases.append({"id": key, "source": "spectral_fixture", "registration": seal["registration"],
                      "stage": seal, "checkpoint": seal["checkpoint"], "selected_step": seal["selected"]})
    analysis.require(len(cases) == 14, "Registered route and scope cardinality changed")
    return {"schema": analysis.SCHEMA, "scope": "frozen_synthetic_with_real_prepare_only",
            "producer": {k: code[k] for k in ("commit", "source_hash")},
            "inputs": inputs, "geometry_rule": GEOMETRY_RULE, "operation_rule": ARM_RULE,
            "cases": cases, "real_training_updates": 0, "synthetic_training_updates": 0,
            "query_accessed": False, "raw_gt_loaded": False}


def build_bank(spec, artifact, root, execution):
    home = Path(artifact["checkpoint"]["path"]).parents[2]
    classes = spec["classes"]
    fixture, _ = release_fixture(classes)
    old = read(transport._pin(home / f"source-{classes}.json"), root)
    admitted = bank.Bank(old["bank"], home, fixture)
    analysis.require(admitted.pin == spec["bank"] and spec["producer"] == DUTY_PRODUCER
                     and spec["registered_graph_scale"] == 100 and spec["synthetic"] is True
                     and spec["numeric"] == execution.numeric, "Bank and model registration differ")
    source = original._baseline(fixture, 81271)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        for heads in (source.heads, source.hgnn_heads, source.cnn_heads):
            heads[fixture.scene].reset_parameters()
    analysis.require(state_hash(source) == spec["source_state"], "Synthetic bank initial state changed")
    model = real.RealReleasedRun.model_factory(source, spec["arm"], spec["module_seed"])

    def materialize(event):
        actual, row = from_bank(admitted, "devval", event["tile"], VIEWS[event["band"]],
                                event["factor"], spec["producer"], engineering=True)
        targets = bind(actual, row, spec["producer"], "devval")

        def call():
            return forward(model, actual, spec["producer"], "devval", execution=execution,
                           graph_scale=spec["registered_graph_scale"])

        return actual, targets, call

    return model, materialize


def build_fixture(spec, execution):
    analysis.require(spec["producer"] == SPECTRAL_PRODUCER and spec["numeric"] == execution.numeric,
                     "Foreign selected spectral fixture")
    run = Run(spec, spec["producer"], execution=execution, released=released)

    def materialize(event):
        actual, _, targets = run._materialize(run._val, event)

        def call():
            return run._forward(actual, "devval", event["condition"])[:2]

        return actual, targets, call

    return run.model, materialize


def execute(root, spec, execution):
    output = []
    for case in spec["cases"]:
        stamp = (read(case["stage"], root) if case["source"] == "bank" else case["stage"])
        path = stamp["checkpoint"]
        key = case["registration"]
        config = read(key, root)
        cp = packet(path, root)
        analysis.require(stamp["phase"] == "completed" and stamp["updates"] == 0 and path == case["checkpoint"]
                         and stamp["registration"] == key and cp["registration"] == digest(config)
                         and cp["cursor"] == config["steps"] and cp["selected"]["step"] == case["selected_step"],
                         "Source is not the audited frozen selected state")
        model, materialize = (build_bank(config, case, root, execution) if case["source"] == "bank"
                              else build_fixture(config, execution))
        model.load_state_dict(cp["selected"]["state"], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        chosen = next(r for r in cp["validation"] if r["step"] == cp["selected"]["step"])
        rows = []
        for event, historical in zip(config["validation"], chosen["rows"], strict=True):
            actual, targets, call = materialize(event)
            analysis.require(targets.summary() == historical["released"]
                             and actual.receipt["observation"] == historical["observation"],
                             "Observation or released point geometry changed")
            rows.append({"event": event, **analysis.evaluate(model, actual, targets, call, execution, historical)})
        output.append({**case, "source_identity": config["initialization"], "views": rows})
    return output


def actual_real_admission(root, item):
    value = read(item, root)
    out = {}
    for scene, pin in value["scenes"].items():
        admitted, prior, _ = preflight.open_bank(root, scene)
        expected = read(pin, root)
        analysis.require(admitted.pin == expected["bank"] and prior == expected["prior_production"] and not admitted.reads,
                         "Real source differs from admitted bank")
        out[scene] = {"bank": admitted.pin, "released_observation_coverage": value["scenes"][scene],
                      "checkpoint_status": "compatible_real_selected_checkpoint_unavailable",
                      "optimization_updates": 0, "spectral_array_reads": 0, "metrics": None}
    return {**out, "paviaU": {"checkpoint_status": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    cmds = p.add_subparsers(dest="action", required=True)
    prepare = cmds.add_parser("register")
    for key in ("duty", "spectral", "real"):
        prepare.add_argument("--"+key, nargs=2, required=True)
    run = cmds.add_parser("analyze")
    run.add_argument("--registration", nargs=2, required=True)
    args = p.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    analysis.require(code["commit"] == args.expected_commit and root in out.parents, "Analysis source or path differs")
    torch.set_num_threads(1)
    pin = lambda row: dict(zip(("path", "sha256"), row, strict=True))
    tick = perf_counter()
    if args.action == "register":
        spec = roster(root, {key: pin(getattr(args, key)) for key in ("duty", "spectral", "real")}, code)
        out.mkdir()
        result = transport._json_new(out / "registration.json", spec)
    else:
        passed = pin(args.registration)
        spec = read(passed, root)
        analysis.require(Path(passed["path"]) == out / "registration.json" and roster(root, spec["inputs"], code) == spec,
                         "Registration differs from the authenticated selected roster")
        transport._json_new(out / "analysis.claim.json", {"registration": passed})
        with arithmetic(root, None) as execution:
            measured = execute(root, spec, execution)
        ready = actual_real_admission(root, spec["inputs"]["real"])
        require_frozen_code(code)
        result = transport._json_new(out / "analysis.json", {"schema": analysis.SCHEMA, "status": "measured",
            "scope": spec["scope"], "producer": spec["producer"], "registration": passed, "cases": measured,
            "real": ready, "real_optimizer_updates": 0, "synthetic_optimizer_updates": 0,
            "real_gain": None, "raw_gt_loaded": False, "query_accessed": False, "seconds": perf_counter()-tick})
    print(result, flush=True)


if __name__ == "__main__":
    main()

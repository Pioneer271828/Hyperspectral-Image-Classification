"""Measure fixed graph-representation interventions on registered released points."""
import argparse
from collections import Counter
from contextlib import nullcontext
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from model.research_config import digest
from monitoring.released_region import RegionAction, point_geometry, recovery_contract, region_action
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_native_released_points import arithmetic
from transfer import development_bank as bank, development_bank_training as original
from transfer import native_released_preflight as preflight, native_released_training as released
from transfer import native_scale_training as training, registered_development as transport
from transfer.development_block_probe import VIEWS
from transfer.g3_training import state_hash
from transfer.native_released_points import HEADS, bind, scores
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_observation import from_bank, forward
from transfer.xuzhou_multiscale_audit import _exact


SCHEMA = "wp06_released_graph_representation_evidence_v1"


def require(test, message):
    if not test:
        raise ValueError(message)


@torch.no_grad()
def evaluate(model, actual, targets, before_call, historical=None):
    all_counts = range(1, actual.classes+1)
    if not targets.points:
        raise ValueError("Registered released development view has no valid points")
    model.eval()
    start = perf_counter()
    frozen = state_hash(model)
    base, q = before_call(None)
    geometry = point_geometry(actual, targets, include_points=True)
    truth = np.asarray([label for _, label in targets.points])
    row = {"role": "devval", "observation": actual.receipt["observation"], "graph": q,
           "release": targets.summary(), "geometry": geometry, "arms": {}}
    if historical is not None:
        scored = scores(base, targets, detailed=False)
        require(all(scored[name] == historical["scores"][name] for name in HEADS)
                and q == historical["graph"], "Reconstructed frozen baseline differs from selected devval")
    for name in ("baseline", "zero", "donor"):
        if name == "donor" and q["q_shape"][1] < 2:
            row["arms"][name] = {"status": "no_same_tile_donor"}
            continue
        action = {"baseline": None, "zero": RegionAction("zero"), "donor": RegionAction("replace", donor=1)}[name]
        result, graph = (base, q) if action is None else before_call(action)
        require(graph == q and torch.equal(base[2], result[2]), "Intervention altered Q/A or the CNN head")
        heads = {}
        for i, head in enumerate(HEADS):
            prev = base[i].argmax(dim=1).cpu().numpy()[targets.cells]+1
            pred = result[i].argmax(dim=1).cpu().numpy()[targets.cells]+1
            lose, gain = (prev == truth) & (pred != truth), (prev != truth) & (pred == truth)
            heads[head] = {"prediction": pred.tolist(), "correct_to_wrong": int(lose.sum()),
                           "wrong_to_correct": int(gain.sum()), "prediction_changes": int((prev != pred).sum()),
                           "metrics": classification_metrics(truth, pred, all_counts),
                           "strata": {str(radius): classification_metrics(
                               truth, pred, all_counts, np.asarray([radius in values for values in geometry["point_bands"]]))
                               for radius in (1, 2, 3)}}
        row["arms"][name] = {"status": "measured", "target": None if action is None else action.target,
                             "donor": None if action is None else action.donor, "heads": heads}
    rerun, final_q = before_call(None)
    require(all(torch.equal(a, b) for a, b in zip(base[:3], rerun[:3], strict=True))
            and graph_identity(q) == graph_identity(final_q) and state_hash(model) == frozen,
            "Disabled intervention did not restore exact outputs and frozen state")
    return {**row, "source_state": frozen, "frozen_state_unchanged": True,
            "elapsed_seconds": perf_counter()-start,
            "points": [{"pixel": pixel, "label": label, "coarse_cell": int(cell)}
                       for (pixel, label), cell in zip(targets.points, targets.cells, strict=True)]}


def graph_identity(q):
    return {key: q[key] for key in ("q", "a0", "q_shape")}


def trained_bank(root, audit_pin, execution, ops):
    from tests.native_released_bank_fixture import release_fixture
    from transfer import native_released_real as real

    frozen = read(audit_pin, root)
    require(frozen["schema"] == "wp07_native_released_comparison_audit_v1"
            and frozen["status"] == "passed" and frozen["synthetic"] is True and len(frozen["cases"]) == 114
            and frozen["optimizer_updates"] == 1368 and frozen["audit_optimizer_updates"] == 0,
            "Expected the full independent synthetic duties audit")
    result = []
    for record in frozen["cases"]:
        scene, seed, factor, arm = record["case"].split("/")
        if arm != "joint" or factor not in ("2", "4"):
            continue
        classes = {"xuzhou": 9, "whu-honghu": 22}[scene]
        fixture, _ = release_fixture(classes)
        home = execution.parent
        authority = read(transport._pin(home / f"source-{classes}.json"), root)
        admitted = bank.Bank(authority["bank"], home, fixture)
        case_dir = execution / record["case"].replace("/", "-")
        done = read(record["stages"][-1], root)
        spec = read(done["registration"], root)
        cp = packet(done["checkpoint"], root)
        require(done["phase"] == "completed" and done["updates"] == 0
                and Path(done["checkpoint"]["path"]).parent == case_dir and spec["bank"] == admitted.pin
                and spec["registered_graph_scale"] == 100 and spec["source_state"] == record["source_state"]
                and spec["producer"] == frozen["producer"] and spec["synthetic"] is True
                and cp["registration"] == digest(spec) and cp["cursor"] == record["optimizer_updates"]
                and cp["selected"]["step"] == record["selected_step"],
                "Frozen bank endpoint source, registration or selected state differs")
        source = original._baseline(fixture, 81271)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for heads in (source.heads, source.hgnn_heads, source.cnn_heads):
                heads[scene].reset_parameters()
        require(state_hash(source) == spec["source_state"], "Synthetic bank source state differs")
        model = real.RealReleasedRun.model_factory(source, arm, spec["module_seed"])
        model.load_state_dict(cp["selected"]["state"], strict=True)
        chosen = next(row for row in cp["validation"] if row["step"] == cp["selected"]["step"])
        views = []
        for event, old in zip(spec["validation"], chosen["rows"], strict=True):
            actual, row = from_bank(admitted, "devval", event["tile"], VIEWS[event["band"]], int(factor),
                                    spec["producer"], engineering=True)
            targets = bind(actual, row, spec["producer"], "devval")

            def run(action):
                with nullcontext() if action is None else region_action(model.backbone, action):
                    return forward(model, actual, spec["producer"], "devval", execution=ops,
                                   graph_scale=spec["registered_graph_scale"])

            views.append({"event": event, **evaluate(model, actual, targets, run, old)})
        result.append({"case": record["case"], "checkpoint": done["checkpoint"], "advance_updates": 0,
                       "selected_step": record["selected_step"], "initialization": spec["initialization"],
                       "training_exposures": record["released_train_exposures"], "source": spec["source_state"],
                       "selected_after_update": bool(record["selected_step"]), "views": views})
    require(len(result) == 12, "Expected both factors, both class counts and all three seeds")
    return result


def paired_recovery(root, out, producer, execution):
    from transfer.native_released_training import validate_history

    recovery = []
    for classes in (9, 22):
        arms = {}
        for operation in ("baseline", "zero"):
            folder = out / f"recovery-{classes}-{operation}"
            folder.mkdir()
            spec = recovery_contract(classes, 64, "joint", 2, producer, operation, execution.numeric)
            pin = transport._json_new(folder / "registration.json", spec)
            stages, checkpoints = [], {}
            for phase in ("full", "prefix", "resume", "completed"):
                predecessor = {"resume": "prefix", "completed": "resume"}.get(phase)
                path = folder / f"{phase}.pt"
                prior = checkpoints.get(predecessor)
                outcome = training.execute(spec, phase, path, producer, prior, execution=execution, released=released)
                pinned = transport._pin(path)
                checkpoints[phase] = packet(pinned, root)
                stages.append({**outcome, "checkpoint": pinned})
            full = checkpoints["full"]
            require(_exact(full, checkpoints["resume"]) and _exact(full, checkpoints["completed"])
                    and [row["updates"] for row in stages] == [6, 2, 4, 0],
                    "Fixed intervention recovery changed selected state, optimizer, RNG or history")
            validate_history(spec, full)
            run = training.Run(spec, producer, execution=execution, released=released)
            run.model.load_state_dict(full["selected"]["state"], strict=True)
            views = []
            for event in spec["validation"]:
                actual, _, targets = run._materialize(run._val, event)

                def forward_selected(action):
                    # Study compare three representation operations on the same selected state.
                    from transfer.native_scale_fixture import condition_for

                    condition = condition_for(actual, event["condition"])
                    with nullcontext() if action is None else region_action(run.model.backbone, action):
                        return forward(run.model, actual, producer, "devval", condition=condition, execution=execution)

                views.append({"event": event, **evaluate(run.model, actual, targets, forward_selected)})
            arms[operation] = {"registration": pin, "stages": stages, "selected_step": full["selected"]["step"],
                               "history": full["history"], "calendar": spec["events"], "source": spec["initialization"],
                               "training": spec["training"], "views": views, "exact_full_split_complete": True}
        left, right = arms["baseline"], arms["zero"]
        require(all(left[key] == right[key] for key in ("source", "training", "calendar"))
                and [row["released"] for row in left["history"]] == [row["released"] for row in right["history"]]
                and any(a["heads"] != b["heads"] for a, b in zip(left["history"], right["history"], strict=True)),
                "Fixed intervention recovery did not preserve matching or did not act")
        recovery.append({"classes": classes, "arms": arms})
    return recovery


def real_coverage(root, frozen_manifest):
    catalog = read(frozen_manifest, root)
    from transfer.native_released_campaign import REAL_PRE_FLIGHT_SHA

    require(frozen_manifest["sha256"] == REAL_PRE_FLIGHT_SHA, "Expected the original frozen observed scene release")
    result = {}
    for scene, pin in catalog["scenes"].items():
        admitted, prior, _ = preflight.open_bank(root, scene)
        frozen = read(pin, root)
        require(frozen["bank"] == admitted.pin and frozen["prior_production"] == prior, "Real freeze differs")
        roles = {}
        for role in ("devtrain", "devval"):
            rows = []
            references = {(row["tile"], row["factor"]): row for row in frozen["views"]
                          if row["role"] == role and row["band"] == "full"}
            for row, factor in product(admitted.role_rows(role), (1, 2, 4)):
                if not row["samples"]:
                    continue
                actual, authority = from_bank(admitted, role, row["id"], VIEWS["full"], factor, frozen["producer"])
                targets = bind(actual, authority, frozen["producer"], role)
                ref = references[row["id"], factor]
                require(targets.summary() == ref["released"] and actual.receipt["observation"] == ref["observation"],
                        "Real release changed since zero-update preparation")
                rows.append({"tile": row["id"], "factor": factor, "released": targets.summary(),
                             "geometry": point_geometry(actual, targets)})
            totals = {str(f): dict(sum((Counter(r["geometry"]["support"]) for r in rows if r["factor"] == f), Counter()))
                      for f in (1, 2, 4)}
            roles[role] = {"totals": totals, "rows": rows}
        result[scene] = {"roles": roles, "bank": admitted.pin, "optimization_updates": 0,
                         "frozen_endpoint_status": "compatible_real_trained_endpoint_unavailable"}
    result["paviaU"] = {"status": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "out", "execution"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("audit", "real-preflight"):
        parser.add_argument("--"+name, nargs=2, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    require(args.expected_commit == code["commit"] and root in out.parents and args.execution.is_relative_to(root),
            "Wrong frozen WP06 source or runs folder")
    out.mkdir()
    torch.set_num_threads(1)
    pinned = lambda entry: dict(zip(("path", "sha256"), entry, strict=True))
    started = perf_counter()
    with arithmetic(root, None) as execution:
        trained = trained_bank(root, pinned(args.audit), args.execution, execution)
        recovery = paired_recovery(root, out, code["commit"], execution)
    coverage = real_coverage(root, pinned(args.real_preflight))
    require_frozen_code(code)
    result = {"schema": SCHEMA, "status": "passed", "producer": code["commit"], "source_hash": code["source_hash"],
              "synth_trained": trained, "synth_recovery": recovery, "real_coverage": coverage,
              "synth_optimizer_updates": 48, "real_optimizer_updates": 0, "completed_resume_updates": 0,
              "real_effect_measured": False, "raw_gt_loaded": False, "query_accessed": False,
              "cost_seconds": perf_counter()-started, "real_scientific_cost": None,
              "donor_rule": "same tile and range; fixed graph row 1 to row 0 if it exists",
              "geometry_rule": "four-neighbor released support; Manhattan distance strictly less than band width",
              "frozen_worker_audit": pinned(args.audit), "real_preflight": pinned(args.real_preflight)}
    print(transport._json_new(out / "evidence.json", result), flush=True)


if __name__ == "__main__":
    main()

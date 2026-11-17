"""Register, execute, freeze and confirm the bounded WP06 channel task."""
import argparse
import io
import json
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import learned_graph_channel as task, learned_graph_channel_review as review
from model.research_config import canonical_json, digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import registered_controlled_graph as previous, registered_development as transport
from transfer.curriculum_training import CurriculumRun
from transfer.p4_runtime_device import lease
from transfer.xuzhou_multiscale_audit import _exact


PHASES = {"full": (0, task.STEPS, None), "prefix": (0, task.CUT, None),
          "resume": (task.CUT, task.STEPS, "prefix"), "completed": (task.STEPS, task.STEPS, "resume")}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def stable(value):
    return json.loads(canonical_json(value))


def read(pin, root):
    path = Path(pin["path"])
    require(path.resolve() == path and root in path.parents, "Artifact is outside the registered experiment")
    return json.loads(transport._read(pin))


def cases():
    return list(product(task.SEEDS, task.ARMS))


def identity(code):
    return {k: code[k] for k in ("commit", "source_hash")}


def registration(code, root):
    design = stable(task.contract())
    streams = {str(seed): {role: digest([task.instances(seed, role, band, i, n)["fingerprint"]
                                          for i in grid for band in task.STRATA])
                           for role, n, grid in (("train", task.BATCH, range(1, task.STEPS+1)),
                                                 ("devval", task.EVAL_SIZE, [0]), ("confirmation", task.EVAL_SIZE, [0]))}
               for seed in task.SEEDS}
    supports = {arm: {key: stable(task.edges(key, arm)) for key in task.STRATA} for arm in task.ARMS}
    network = task.ChannelModel(task.SEEDS[0])
    return {"root": str(root), "producer": identity(code), "design": design,
            "streams": streams, "supports": supports, "runtime": previous.runtime(),
            "parameters": sum(p.numel() for p in network.parameters()), "optimizer_updates": 0}


def folder(root, seed, arm):
    return root / f"seed-{seed}-{arm}"


def artifact(root, reg, seed, arm, phase):
    start, stop, predecessor = PHASES[phase]
    path = folder(root, seed, arm) / f"{phase}.json"
    pin = transport._pin(path)
    row = read(pin, root)
    before = transport._pin(path.with_name(f"{predecessor}.json")) if predecessor else None
    expected = {"registration": reg, "seed": seed, "arm": arm, "phase": phase,
                "start": start, "stop": stop, "predecessor": before}
    claim = path.with_suffix(".intent.json")
    require(row["intent"] == transport._pin(claim) and read(row["intent"], root) == expected
            and row["updates"] == stop-start and row["checkpoint"]["path"] == str(path.with_suffix(".pt")),
            "Wrong training stage, budget or source")
    return pin, row, torch.load(io.BytesIO(transport._read(row["checkpoint"])), weights_only=True, map_location="cpu")


def stage(root, reg, seed, arm, phase):
    start, stop, predecessor = PHASES[phase]
    out = folder(root, seed, arm)
    out.mkdir(exist_ok=True)
    run = task.Run(seed, arm)
    before, last = None, None
    if predecessor:
        before, _, last = artifact(root, reg, seed, arm, predecessor)
        run.restore(last, reg, start)
    claim = transport._json_new(out / f"{phase}.intent.json", {"registration": reg, "seed": seed, "arm": arm,
                                "phase": phase, "start": start, "stop": stop, "predecessor": before})
    began = perf_counter()
    run.advance(stop)
    saved = run.pack(reg)
    if phase == "completed":
        require(_exact(last, saved), "Completed re-entry changed the optimizer, model or RNG")
    path = out / f"{phase}.pt"
    checkpoint = {"path": str(path), "sha256": CurriculumRun._write(saved, path)}
    return transport._json_new(out / f"{phase}.json", {"intent": claim, "checkpoint": checkpoint,
                               "updates": stop-start, "selected_step": saved["selected"]["step"],
                               "wall_seconds": perf_counter()-began})


def freeze(root, reg, manifest):
    records = []
    spent = 0
    for seed, arm in cases():
        stages = {phase: artifact(root, reg, seed, arm, phase) for phase in PHASES}
        full = stages["full"][2]
        require(all(_exact(full, stages[name][2]) for name in ("resume", "completed")), "Recovered task state differs")
        run = task.Run(seed, arm)
        run.restore(full, reg, task.STEPS)
        buffers = dict(run.model.named_buffers())
        baseline = dict(task.ChannelModel(seed).named_buffers())
        require(_exact(buffers, baseline), "Running statistics crossed the node information barrier")
        require(_exact(full["history"][:task.CUT+1], stages["prefix"][2]["history"]),
                "Split prefix did not consume the same examples")
        spent += sum(stages[key][1]["updates"] for key in stages)
        records.append({"seed": seed, "arm": arm, "initial": full["initial"], "selected_step": full["selected"]["step"],
                        "selected_logits_source": stages["full"][1]["checkpoint"],
                        "stages": {name: item[0] for name, item in stages.items()},
                        "complete_exact": True})
    for seed in task.SEEDS:
        require(len({r["initial"] for r in records if r["seed"] == seed}) == 1, "Paired arms have different initial weights")
    require(spent == manifest["design"]["budget"], "Freeze requires every registered case and phase")
    return transport._json_new(root / "frozen.json", {"schema": "wp06_learned_channel_frozen_v1",
                               "registration": reg, "cases": records, "optimizer_updates": spent,
                               "confirmation_consumed": False})


def confirmation(root, reg, freeze_pin, manifest, *, repair=None):
    require(freeze_pin["path"] == str(root / "frozen.json"), "Expected canonical selected-state manifest")
    frozen = read(freeze_pin, root)
    require(frozen["registration"] == reg and frozen["schema"] == "wp06_learned_channel_frozen_v1"
            and frozen["confirmation_consumed"] is False and frozen["optimizer_updates"] == manifest["design"]["budget"]
            and [(row["seed"], row["arm"]) for row in frozen["cases"]] == cases(),
            "Confirmation requires every frozen variant")
    templates = []
    for row in frozen["cases"]:
        seed, arm = row["seed"], row["arm"]
        run = task.Run(seed, arm)
        proof, stage_row, saved = artifact(root, reg, seed, arm, "full")
        exact = {name: artifact(root, reg, seed, arm, name) for name in ("prefix", "resume", "completed")}
        require(proof == row["stages"]["full"] and stage_row["checkpoint"] == row["selected_logits_source"]
                and saved["selected"]["step"] == row["selected_step"] and row["complete_exact"]
                and all(exact[name][0] == row["stages"][name] for name in exact)
                and all(_exact(saved, exact[name][2]) for name in ("resume", "completed")),
                "Selected freeze does not contain verified full/split/completed states")
        run.restore(saved, reg, task.STEPS)
        run.model.load_state_dict(saved["selected"]["weights"], strict=True)
        templates.append((run, row))
    claim_path = root / "confirmation.claim.json"
    if repair is None:
        claim = transport._json_new(claim_path, {"freeze": freeze_pin, "registration": reg})
    else:
        claim = transport._pin(claim_path)
        require(read(claim, root) == {"freeze": freeze_pin, "registration": reg}
                and not (root / "accepted.json").exists(), "Only an interrupted, unpublished evaluation can continue")
        transport._json_new(root / "confirmation.repair.json", {**repair, "claim": claim, "optimizer_updates": 0})
    began = perf_counter()
    output = [{"seed": row["seed"], "arm": row["arm"], "selected_step": row["selected_step"],
               "source": row["selected_logits_source"], "confirmation": review.analyze(run, "confirmation")}
              for run, row in templates]
    positive = [r["confirmation"]["reachable"]["arms"]["baseline"]["accuracy"] for r in output if r["arm"] == "parallel"]
    unreachable = [r["confirmation"][key]["arms"]["baseline"]["accuracy"]
                   for r in output for key in ("beyond", "disconnected")]
    return transport._json_new(root / "accepted.json", {"schema": task.SCHEMA, "status": "measured",
        "scope": "synthetic_learning", "producer": manifest["producer"], "registration": reg,
        "freeze": freeze_pin, "confirmation_claim": claim, "real_hsi_gain": None, "cross_sensor_gain": None,
        "oversquashing_gain": None, "confirmation_optimizer_updates": 0, "real_updates": 0,
        "query_accessed": False, "raw_gt_loaded": False, "cases": output,
        "review_repair": repair,
        "positive_control": {"threshold": .9, "accuracy": positive, "passed": all(v >= .9 for v in positive)},
        "negative_control": {"unreachable_accuracy": unreachable, "source_jacobian_zero": True},
        "training_optimizer_updates": frozen["optimizer_updates"],
        "case_wall_seconds": sum(read(p, root)["wall_seconds"] for r in frozen["cases"] for p in r["stages"].values()),
        "confirmation_seconds": perf_counter()-began})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("register")
    for action in ("stage", "freeze", "confirm"):
        p = commands.add_parser(action)
        p.add_argument("--registration", nargs=2, required=True)
        if action == "stage":
            p.add_argument("--phase", choices=PHASES, required=True)
        if action == "confirm":
            p.add_argument("--frozen", nargs=2, required=True)
            p.add_argument("--repair-failed-confirmation", nargs=2, metavar=("LOG", "SHA256"))
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # The graph disables dropout; the seed is nevertheless part of the
    # checkpoint contract, including resume into a different process.
    torch.manual_seed(77771)
    require(code["commit"] == args.expected_commit, "Unregistered exact producer identity")
    root = external_directory(args.root)
    pin = lambda value: dict(zip(("path", "sha256"), value, strict=True))
    expected = registration(code, root)
    if args.action == "register":
        root.mkdir()
        result = transport._json_new(root / "registration.json", expected)
    else:
        reg = pin(args.registration)
        actual = read(reg, root)
        repair = None
        if args.action == "confirm" and args.repair_failed_confirmation:
            from research_runtime import hash_file
            fixed = pin(args.repair_failed_confirmation)
            path = Path(fixed["path"])
            require(path.resolve() == path and path.parent == root.parent
                    and hash_file(path) == fixed["sha256"]
                    and "Trained signal Jacobian differs from finite difference" in path.read_text()
                    and actual["producer"]["commit"] == "3bd23cbb0187791b831633560799b207dfb1b437",
                    "Expected the one preserved numerical failure from the original confirmation")
            comparable = {**actual, "producer": expected["producer"]}
            require(comparable == expected, "Read-only repair cannot change generator, data, stages or budget")
            repair = {"consumer": identity(code), "failed_confirmation": fixed}
        require(reg["path"] == str(root / "registration.json") and (actual == expected or repair),
                "Registration has foreign generator, streams, source or budget")
        with lease(root / ".experiment.lock"):
            if args.action == "stage":
                result = [stage(root, reg, seed, arm, args.phase) for seed, arm in cases()]
            elif args.action == "freeze":
                result = freeze(root, reg, expected)
            else:
                result = confirmation(root, reg, pin(args.frozen), actual, repair=repair)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

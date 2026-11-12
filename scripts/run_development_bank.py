"""Publish and validate an existing-authority development bank in the authorized LXC."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from transfer import development_bank as bank, development_bank_consumers as consumers
from transfer import development_bank_custody as custody, development_bank_training as training
from transfer import development_block_custody as original, registered_development as transport
from transfer.container_runtime import training_container_kind
from transfer.development_blocks import ROLES, SCENES, scene_name, read_pin
from transfer.xuzhou_multiscale_audit import _exact


LEGACY = "7f0c985eaccbc38d430b4320f2c8691cd23b101249467b54802266031e910e2e"


def _read(pin, root):
    return json.loads(read_pin(pin, root))


def _pin(path):
    return transport._pin(path)


def _output(root, path):
    result = external_directory(path)
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Evidence output must stay within the authorized runs")
    return result


def inspect(root, output, code):
    result = {}
    for name, provider in custody.PROVIDERS.items():
        intake = provider(root)
        identity = intake.release.identity()
        coverage = intake.release.coverage()
        result[name] = dict(identity=identity, coverage=coverage,
            complete_role_arrays=all(c["authorized_blocks"] == c["publishable_blocks"] for c in coverage.values()),
            all_classes_with_readable_released_support=all(min(c["publishable_per_class"]) > 0
                                                         for c in coverage.values()),
            complete_training_claim=False, real_training_authorized=False)
    return transport._json_new(_output(root, output), dict(code=code, scenes=result, decoded_arrays=0,
                                                           raw_gt=0, query=0, new_labels=0, updates=0))


def _legacy(root):
    return _read({"path": str(root / "wp02-development-block-probe07/admission.json"), "sha256": LEGACY}, root)


def scan(root, bank_pin, output, name, code):
    started = perf_counter()
    intake = custody.PROVIDERS[name](root)
    opened = bank.Bank(bank_pin, root, intake.release)
    old = _legacy(root)["scenes"][name]
    prior_tiles = tuple(sorted({row["context"]["tile"] for row in old.get("rows", ())}))
    result = consumers.scan(opened, intake.baseline, prior_tiles=prior_tiles)
    old_comparisons = 0
    for row in old.get("rows", []):
        hit = next(item for item in result["rows"] if item["view"] == row["condition"] and
                   item["context"]["tile"] == row["context"]["tile"])
        expected = dict(row["probabilities"])
        expected["native"] = expected.pop("native_step_zero")
        if hit["probabilities"] != expected or hit["context"]["raw"] != row["context"]["raw"]:
            raise ValueError("Historical #451 input or probability hashes differ")
        old_comparisons += 21
    return transport._json_new(_output(root, output), dict(scene=name, code=code, bank=opened.pin,
        **result, original_files=custody.protect(intake), previous_probability_hashes=old_comparisons,
        elapsed=perf_counter()-started, raw_gt=0, query=0, new_labels=0, real_updates=0))


def _synthetic(root, banks, classes):
    release, arrays = training.fixture(classes)
    pin = _read(banks[str(classes)], root)
    return bank.Bank(pin, root, release)


def register_fixture(root, output, code):
    output.mkdir()
    registrations = {}
    banks = {}
    for classes in (9, 22):
        release, arrays = training.fixture(classes)
        pin = bank.publish(root / f"{output.name}-{classes}", root, release, lambda row: arrays[row["id"]])
        banks[str(classes)] = transport._json_new(output / f"{classes}-bank.json", pin)
        opened = bank.Bank(pin, root, release)
        for seed in training.SEEDS:
            for arm in consumers.ARMS:
                name = f"{classes}-{seed}-{arm}"
                value = training.contract(opened, seed, arm)
                registrations[name] = transport._json_new(output / f"{name}.json", value)
    return transport._json_new(output / "matrix.json", dict(code=code, cases=registrations, banks=banks,
        engineering_maximum_updates=len(registrations)*2*training.STEPS, real_updates=0))


def stage(root, matrix_pin, case, name, code):
    reg = _read(matrix_pin, root)
    matrix = Path(matrix_pin['path'])
    if reg["code"] != code or case not in reg["cases"]:
        raise ValueError("Stage is outside the frozen matrix")
    value = _read(reg["cases"][case], root)
    directory = matrix.parent
    opened = _synthetic(root, reg["banks"], int(case.split("-")[0]))
    output = directory / case
    output.mkdir(exist_ok=True)
    prior = {"resume": "prefix", "completed": "resume"}.get(name)
    packet = None
    if prior is not None:
        parent = _read(_pin(output / f"{prior}.json"), root)
        if (parent["stage"] != prior or parent["case"] != case or parent["registration"] != reg["cases"][case]
                or parent["checkpoint"]["path"] != str(output / f"{prior}.pt")):
            raise ValueError("Stage predecessor identity differs")
        packet = torch.load(io.BytesIO(read_pin(parent["checkpoint"], output)), map_location="cpu", weights_only=True)
    row = training.execute(opened, value, name, output / f"{name}.pt", packet)
    row.update(case=case, registration=reg["cases"][case], checkpoint=_pin(row["checkpoint"]))
    return transport._json_new(output / f"{name}.json", row)


def audit_fixture(root, matrix_pin, output, code):
    reg = _read(matrix_pin, root)
    matrix = Path(matrix_pin['path'])
    if reg["code"] != code:
        raise ValueError("Synthetic audit code differs")
    results = {}
    for case in reg["cases"]:
        packets, summaries = [], []
        for name in ("full", "prefix", "resume", "completed"):
            row = _read(_pin(matrix.parent / case / f"{name}.json"), root)
            if (row["case"] != case or row["stage"] != name or row["registration"] != reg["cases"][case]):
                raise ValueError("Synthetic stage identity differs")
            packets.append(torch.load(io.BytesIO(read_pin(row["checkpoint"], root)),
                                      map_location="cpu", weights_only=True))
            summaries.append(row)
        full, prefix, resumed, complete = packets
        if (not _exact(full, resumed) or not _exact(full, complete)
                or not _exact(full["history"][:training.PREFIX], prefix["history"])
                or not _exact(full["validation"][:2], prefix["validation"])
                or [row["updates"] for row in summaries] != [24, 7, 17, 0]):
            raise ValueError(f"State, probabilities, clocks or schedule mismatch for {case}")
        classes = int(case.split("-")[0])
        opened = _synthetic(root, reg["banks"], classes)
        value = _read(reg["cases"][case], root)
        first = min(full["validation"], key=lambda item: item["fused_nll"])
        if (full["selected"]["step"], full["selected"]["fused_nll"]) != (first["step"], first["fused_nll"]):
            raise ValueError("Synthetic first-minimum record differs")
        run = training.Run(opened, value)
        checks = {}
        for name, state, selection in (("selected", full["selected"]["state"], first),
                                       ("last", full["model"], full["validation"][-1])):
            run.model.load_state_dict(state, strict=True)
            run.selected = None
            run.validation = []
            run._evaluate(selection["step"])
            if not _exact(run.validation[0], selection):
                raise ValueError("Synthetic validation endpoint failed independent replay")
            checks[name] = selection
        results[case] = dict(updates=sum(row["updates"] for row in summaries), stage_rows=summaries,
                            native_clocks=full["native_clocks"], history=full["history"],
                            validation=full["validation"], endpoint_replays=checks, exact=True)
    return transport._json_new(output, dict(code=code, matrix=matrix_pin, cases=results,
        synthetic_updates=sum(row["updates"] for row in results.values()), real_updates=0))


def supplement(root, inventory, scans, matrix, output, code):
    old = _read(original.historical(root, "wp12"), root)
    tables = {**old["preserved"], **old["tables"]}
    original._protect(root, tables.values())
    if len(tables) != 10 or (old["old_real_models"], old["old_real_heads"]) != (65, 193):
        raise ValueError("Historical WP12 table roster differs")
    scope = _read(_pin(inventory), root)
    result = {_read(_pin(path), root)["scene"]: _read(_pin(path), root) for path in scans}
    synthetic = _read(matrix, root)
    if (set(scope["scenes"]) != set(SCENES) or set(result) != set(SCENES)
            or any(row["real_updates"] for row in result.values()) or synthetic["real_updates"]):
        raise ValueError("Full scene and zero-update evidence required")
    output.mkdir()
    table = output / "protocol-scope.csv"
    with table.open("x", newline="", encoding="utf-8") as stream:
        csv_out = csv.writer(stream)
        csv_out.writerow(("scene", "role", "authorized", "published", "all_released_classes_readable", "real_updates"))
        for name, row in scope["scenes"].items():
            for role in ROLES:
                c = row["coverage"][role]
                csv_out.writerow((name, role, c["authorized_blocks"], c["publishable_blocks"],
                                  min(c["publishable_per_class"]) > 0, 0))
    requests = {}
    for name, row in scope["scenes"].items():
        authority = row["identity"]["custody"]
        missing = [c for c, role in row["coverage"].items() if min(role["publishable_per_class"]) == 0]
        requests[name] = dict(bank=result[name]["bank"], role_coverage=row["coverage"],
            conditions=["full", "even_retained_positions"], arms=list(consumers.ARMS),
            proposed_max_updates_per_arm=24, proposed_validation_steps=list(training.EVENTS),
            release_seed=authority["release_seed"], validation_seed=authority["validation_seed"],
            source=row["identity"]["source"], required_before_execution=[
                "separate_authorization", "freeze_exposure_and_assert_class_visits", "confirm_validation_and_protocol_scope",
            ], missing_roles=missing, source_has_fresh_head=authority["source_has_fresh_head"],
            complete_training_claim=False, training_authorized=False, optimizer_budget=0)
    request = transport._json_new(output / "future-request-draft.json", dict(code=code, scenes=requests))
    return transport._json_new(output / "supplement.json", dict(code=code, old_tables=tables,
        old_real_models=65, old_real_heads=193, new_real_models=0, real_updates=0,
        synthetic_engineering_updates=synthetic["synthetic_updates"], scope=_pin(table), request=request))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("inspect", "publish", "scan", "register-fixture", "stage", "audit-fixture", "supplement"):
        child = commands.add_parser(action)
        if action != "stage":
            child.add_argument("--out", type=Path, required=True)
        if action in ("publish", "scan"):
            child.add_argument("--scene", required=True)
        if action == "publish":
            child.add_argument("--resume", action="store_true")
            child.add_argument("--stop", type=int)
        if action == "scan":
            child.add_argument("--bank-pin", type=Path, required=True)
        if action in ("stage", "audit-fixture", "supplement"):
            child.add_argument("--matrix", nargs=2, required=True, metavar=("PATH", "SHA256"))
        if action == "stage":
            child.add_argument("--case", required=True)
            child.add_argument("--phase", required=True, choices=("full", "prefix", "resume", "completed"))
        if action == "supplement":
            child.add_argument("--inventory", type=Path, required=True)
            child.add_argument("--scans", nargs=3, type=Path, required=True)
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container with assertions required")
    torch.set_num_threads(1)
    snapshot = code_snapshot()
    if snapshot["commit"] != args.expected_commit:
        parser.error("Exact checked-out SHA required")
    require_frozen_code(snapshot)
    root = external_directory(args.runs.resolve(strict=True))
    if root.name != "runs":
        parser.error("Expected the authorized runs root")
    code, command = snapshot["commit"], args.command
    matrix = None if not hasattr(args, "matrix") else dict(zip(("path", "sha256"), args.matrix, strict=True))
    out = None if command == "stage" else _output(root, args.out)
    if command == "inspect":
        result = inspect(root, out, code)
    elif command == "publish":
        name = scene_name(args.scene)
        intake = custody.PROVIDERS[name](root)
        result = bank.publish(out, root, intake.release, intake.read, resume=args.resume, stop=args.stop)
        if result["status"] == "COMPLETE":
            result = transport._json_new(out.parent / (out.name + ".json"), result)
    elif command == "scan":
        result = scan(root, _read(_pin(args.bank_pin), root), out, scene_name(args.scene), code)
    elif command == "register-fixture":
        result = register_fixture(root, out, code)
    elif command == "stage":
        result = stage(root, matrix, args.case, args.phase, code)
    elif command == "audit-fixture":
        result = audit_fixture(root, matrix, out, code)
    else:
        result = supplement(root, args.inventory, args.scans, matrix, out, code)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Read-only evidence inventory for a stopped, failed light campaign; no acceptance."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


POLICY = "g4_light_failed_evidence_inventory_v1"
TARGETS = ("paviaU", "xuzhou", "whu-honghu")
CONTROLS = ("source_pretrained_head_only", "source_pretrained_adapter_heads")
MODES = ("none", "capacity_control", "x_residual")
ROLES = tuple(itertools.product(TARGETS, (64, 512, 2048), CONTROLS, MODES))
EVENTS = list(range(1, 1500, 30)) + [1500]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(path):
    path = Path(path).absolute()
    require(path.resolve() == path, "Symlink or noncanonical evidence path")
    return path


def sha256(path):
    with Path(path).open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else hashlib.sha256()
        if not hasattr(hashlib, "file_digest"):
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def name_for(role):
    target, seed, method, mode = role
    return f"{target}-s{seed}-{mode}-{method}"


class Evidence:
    def __init__(self):
        self.pins = {}

    def pin(self, path, expected=None):
        path = canonical(path)
        key = str(path)
        if expected is not None:
            require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid SHA256")
        if key not in self.pins:
            self.pins[key] = sha256(path)
        require(expected is None or self.pins[key] == expected, f"Evidence hash differs: {path}")
        return path

    def read(self, path, expected=None):
        return json.loads(self.pin(path, expected).read_text())

    def unchanged(self):
        for path, expected in self.pins.items():
            require(sha256(canonical(path)) == expected, f"Evidence changed during inventory: {path}")


def validate_case(evidence, root, row, role, registration, regpath, regsha):
    name = name_for(role)
    folder = canonical(root / "cases" / name)
    gpu = registration["gpu_ids"][ROLES.index(role) % len(registration["gpu_ids"])]
    require(row["status"] == "passed" and row["name"] == name and row["role"] == list(role)
            and row["physical_gpu"] == gpu and row["updates"] == 1500
            and row["validation_events"] == 51 and row["head_metrics"] == 6
            and row["class_recalls"] == 6 * (22 if role[0] == "whu-honghu" else 9)
            and row["exact_reload_repeats"] == 3 and row["new_optimizer_updates"] == 0
            and row["query_labels_accessed"] is False, "Historical case receipt contract differs")
    inputs, artifacts = row["protected_inputs"], row["artifacts"]
    require(inputs.get(str(regpath)) == regsha, "Receipt is not bound to the registration")
    for key in ("reference_launch", "engineering_acceptance", "analysis_plan"):
        ref = registration[key]
        require(inputs.get(ref["path"]) == ref["sha256"], "Receipt lacks a registered input")
    for path, expected in inputs.items():
        require(not canonical(path).is_relative_to(root), "Protected input overlaps campaign")
        evidence.pin(path, expected)
    actual = {str(canonical(p)) for p in folder.rglob("*") if p.is_file()}
    require(set(artifacts) == actual, "Receipt artifact set differs from the case directory")
    for path, expected in artifacts.items():
        require(canonical(path).parent == folder, "Artifact escapes its case")
        evidence.pin(path, expected)
    for filename, key in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"),
                          ("primary.pt.json", "sidecar_sha256")):
        evidence.pin(folder / filename, row[key])
    result, progress, identity, history, trace = [evidence.read(folder / f"{n}.json") for n in
                                                 ("result", "progress", "identity", "history", "trace")]
    settings = result["settings"]
    require(result["identity"] == identity and identity["settings"] == settings
            and settings["registration_sha256"] == regsha
            and (settings["target"], settings["model_seed"], settings["method"], settings["m4_mode"]) == role
            and settings["physical_gpu"] == gpu and settings["optimizer_steps"] == 1500
            and settings["test_scoring"] is False and result["test_scoring"] is False
            and result["start_code"]["commit"] == registration["producer_commit"]
            and result["start_code"]["dirty"] is False, "Case identity, producer, or query boundary differs")
    require(result["status"] == progress["status"] == "completed"
            and result["optimizer_steps"] == progress["cursor"] == 1500
            and result["per_dataset_updates"] == {role[0]: 1500}
            and progress["state_file"] == "state-001500.pt", "Case completion budget differs")
    evidence.pin(folder / progress["state_file"], progress["state_sha256"])
    require([h["step"] for h in history] == EVENTS
            and [t["step"] for t in trace] == list(range(1, 1501)), "Missing or reordered training events")
    require(all(math.isfinite(h["val_loss"]) and h["selection_score"] == -h["val_loss"] for h in history),
            "Invalid validation selection scores")
    selected = min(history, key=lambda h: h["val_loss"])
    require(row["selected_step"] == result["selected_step"] == progress["best_step"] == selected["step"]
            and progress["best_score"] == -selected["val_loss"]
            and result["selected_validation"] == {k: v for k, v in selected.items()
                                                 if k not in ("epoch", "step", "selection_score")},
            "First-minimum selection differs")
    require(all(t["dataset"] == role[0] and t["frozen_parameters_unchanged"] is True
                and t["frozen_buffers_unchanged"] is True and math.isfinite(t["loss"]["total"])
                and all(math.isfinite(v) for v in t["gradient_norms"].values()) for t in trace),
            "Training trace contradicts historical acceptance")
    changes = sum(t["adapter_changed_elements"] > 0 for t in trace)
    require(changes == row["adapter_changed_steps"] and (changes == 0) == (role[2] == CONTROLS[0])
            and any(t["changed_trainable_elements"] > 0 for t in trace), "Trainable mechanism evidence differs")
    return {"name": name, "role": list(role), "updates": 1500, "selected_step": selected["step"],
            "validation_events": 51, "historical_head_metrics": row["head_metrics"],
            "historical_class_recalls": row["class_recalls"], "historical_exact_reloads": 3}


def inventory(registration_path, registration_sha256, producer_commit, extra_receipts=()):
    started = time.monotonic()
    evidence = Evidence()
    regpath = canonical(registration_path)
    reg = evidence.read(regpath, registration_sha256)
    root = canonical(reg["out_dir"])
    require(reg["policy"] == "g4_light_matched_full_target_v1"
            and reg["producer_commit"] == producer_commit and reg["planned_optimizer_updates"] == 81000
            and len(reg["gpu_ids"]) in (2, 3) and len(set(reg["gpu_ids"])) == len(reg["gpu_ids"])
            and all(type(g) is int and g >= 0 for g in reg["gpu_ids"])
            and reg["query_labels_during_training"] is False and reg["test_scoring_during_training"] is False,
            "Registration budget, producer, or boundary differs")
    require(not regpath.is_relative_to(root), "Registration must be external")
    for key in ("reference_launch", "engineering_acceptance", "analysis_plan"):
        evidence.pin(reg[key]["path"], reg[key]["sha256"])
    before_paths = {str(canonical(p)) for p in root.rglob("*") if p.is_file()}
    for path in before_paths:
        evidence.pin(path)
    status = evidence.read(root / "status.json")
    owner = evidence.read(root / "registration-owner.json")
    require(owner == {"policy": reg["policy"], "registration_sha256": registration_sha256}, "Owner differs")
    require(status["status"] == "failed" and status["registration_sha256"] == registration_sha256
            and status["gpu_ids"] == reg["gpu_ids"] and status["planned_cases"] == 54
            and status["planned_optimizer_updates"] == 81000 and bool(status["errors"]),
            "Only an identified failed campaign can be inventoried")
    require(not (root / "training-acceptance.json").exists(), "Failed campaign unexpectedly has global acceptance")
    roles = {name_for(r): r for r in ROLES}
    require(all(p.name in roles for p in (root / "cases").iterdir()), "Unknown campaign case")
    receipts = [(p, None, "coordinator_receipt_revalidated") for p in sorted((root / "audits").glob("*.json"))]
    require(len(receipts) == status["completed_cases"], "Coordinator receipt count differs from status")
    receipts += [(Path(p), sha, "external_receipt_revalidated") for p, sha in extra_receipts]
    rows = {}
    for path, expected, level in receipts:
        row = evidence.read(path, expected)
        require(row["name"] in roles and row["name"] not in rows, "Unknown or duplicate receipt role")
        result = validate_case(evidence, root, row, roles[row["name"]], reg, regpath, registration_sha256)
        result.update(level=level, receipt=str(canonical(path)), receipt_sha256=evidence.pins[str(canonical(path))])
        rows[row["name"]] = result
    failed = {s["case"] for s in status["slots"].values() if s["status"] == "failed"}
    require(failed <= roles.keys() and not failed.intersection(rows), "Failure slot contradicts acceptance")
    for name, role in roles.items():
        if name in rows:
            continue
        folder = root / "cases" / name
        progress = evidence.read(folder / "progress.json") if (folder / "progress.json").is_file() else None
        level = "failed_case" if name in failed else (
            "completed_without_receipt" if (folder / "result.json").is_file() else
            "partial_without_receipt" if folder.exists() else "not_started")
        rows[name] = {"name": name, "role": list(role), "level": level,
                      "cursor_evidence_present": progress is not None,
                      "cursor_unverified": progress.get("cursor") if progress else None,
                      "files": sorted(p.name for p in folder.iterdir()) if folder.exists() else []}
    evidence.unchanged()
    require(before_paths == {str(canonical(p)) for p in root.rglob("*") if p.is_file()}, "Campaign file set changed")
    accepted = [r for r in rows.values() if r["level"].endswith("receipt_revalidated")]
    return {"policy": POLICY, "status": "inventory_passed", "producer_commit": producer_commit,
            "registration_sha256": registration_sha256, "campaign_status": "failed",
            "planned_cases": 54, "planned_optimizer_updates": 81000,
            "levels": dict(Counter(r["level"] for r in rows.values())),
            "historically_accepted_updates_revalidated": sum(r["updates"] for r in accepted),
            "unaccepted_budget": 81000 - sum(r["updates"] for r in accepted),
            "cases": [rows[n] for n in roles], "checked_files": evidence.pins,
            "elapsed_seconds": time.monotonic() - started, "new_optimizer_updates": 0,
            "query_labels_accessed": False, "new_model_reloads": 0, "global_acceptance": False,
            "resume_authorized": False,
            "scope": "Rehash historical receipts, inputs, artifacts and JSON trace/selection only; "
                     "no new probability, optimizer-state, scheduler, or GPU reload audit."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("registration", "registration-sha256", "producer-commit", "expected-commit", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--extra-receipt", nargs=2, action="append", default=[], metavar=("PATH", "SHA256"))
    args = parser.parse_args(argv)
    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
    require(sys.platform == "linux" and __debug__ and not dirty and commit == args.expected_commit,
            "Clean expected Linux checkout required")
    out = canonical(args.out)
    require(not out.exists() and out.suffix == ".json" and not out.is_relative_to(checkout), "Fresh external JSON required")
    value = inventory(args.registration, args.registration_sha256, args.producer_commit, args.extra_receipt)
    root = canonical(json.loads(Path(args.registration).read_text())["out_dir"])
    require(not out.is_relative_to(root) and not any(Path(p).is_relative_to(out) for p in value["checked_files"]),
            "Output overlaps evidence")
    value["inventory_commit"] = commit
    with out.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in value.items() if k not in ("cases", "checked_files")}))


if __name__ == "__main__":
    main()

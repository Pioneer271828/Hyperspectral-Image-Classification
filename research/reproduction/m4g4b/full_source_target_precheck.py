"""Freeze and verify 12 short target cases from the fully accepted source batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import independent_trainval
from joint_baseline import NAMES
from joint_training import _tree_fingerprint
from loadData.dataset_registry import file_hash
from m4_full_transfer import initialize_full_target
from m4_full_target_runtime import METHODS_ACTIVE, FullSourceTargetRuntime, settings
from m4_ordered_source import MODES
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_data_context import load_target_context
from target_initialization import state_id


def source_spec(root, audit, target, mode):
    folder = root / "cases" / (target + "-s64-" + mode)
    spec = {"source_path": str(folder / "primary.pt"),
            "source_weight_sha256": file_hash(folder / "primary.pt"),
            "source_sidecar_sha256": file_hash(folder / "primary.pt.json")}
    for field, path in {"source_result": folder / "result.json", "source_launch": root / "launch.json",
                        "source_acceptance": root / "training-acceptance.json",
                        "source_independent_audit": audit}.items():
        spec[field], spec[field + "_sha256"] = str(path), file_hash(path)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "source-root", "source-audit", "source-audit-sha256",
                 "observation-root", "family-root", "out-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Expected clean remote commit required")
    audit = Path(args.source_audit).resolve(strict=True)
    if file_hash(audit) != args.source_audit_sha256:
        raise ValueError("Pinned independent source audit required")
    source = Path(args.source_root).resolve(strict=True)
    protected = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    protected[str(audit)] = args.source_audit_sha256
    out = external_directory(args.out_dir)
    out.mkdir(exist_ok=False)
    (out / "logs").mkdir()
    (out / "source-specs").mkdir()
    cases = []
    for target in NAMES:
        observation = Path(args.observation_root) / target
        family = Path(args.family_root) / (target + "-64") / "curator-family.json"
        inputs = {"observation": str(observation / "observation.pt"), "record": str(observation / "record.json"),
                  "family": str(family), "record_sha256": file_hash(observation / "record.json"),
                  "family_sha256": file_hash(family)}
        _, provenance, _ = load_target_context(target, inputs["observation"], inputs["record"],
            inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
        protected.update(provenance["protected_files"])
        for mode in MODES:
            spec = source_spec(source, audit, target, mode)
            spec_path = out / "source-specs" / (target + "-" + mode + ".json")
            write_json(spec_path, spec)
            for method in METHODS_ACTIVE:
                used = {} if method == METHODS_ACTIVE[0] else spec
                model, receipt = initialize_full_target(target, method, mode, model_seed=64,
                    head_seed=1064, source_spec=used, device="cpu")
                name = target + "-" + mode + "-" + method
                command = [sys.executable, "scripts/train_m4_full_source_target_short.py",
                    "--expected-commit", args.expected_commit, "--target", target, "--method", method,
                    "--mode", mode, "--out-dir", str(out / "cases" / name),
                    "--observation", inputs["observation"], "--observation-record", inputs["record"],
                    "--observation-record-sha256", inputs["record_sha256"], "--label-family", inputs["family"],
                    "--label-family-sha256", inputs["family_sha256"],
                    "--initial-state-sha256", receipt["initial_state_sha256"]]
                if used:
                    command += ["--source-spec", str(spec_path), "--source-spec-sha256", file_hash(spec_path)]
                cases.append({"name": name, "target": target, "mode": mode, "method": method,
                              "command": command, "source_spec": used, "inputs": inputs,
                              "initial_state_sha256": receipt["initial_state_sha256"],
                              "settings": settings(target, method, mode, device="cuda")})
                del model
    write_json(out / "launch.json", {"code": code, "cases": cases, "updates": 144,
        "source_audit_sha256": args.source_audit_sha256, "query_labels_accessed": False})
    started = time.monotonic()
    rows = []
    for case in cases:
        folder = out / "cases" / case["name"]
        for suffix, extra in (("first2", ["--stop-after", "2"]), ("continue", ["--resume"])):
            with (out / "logs" / (case["name"] + "-" + suffix + ".log")).open("x") as log:
                subprocess.run(case["command"] + extra, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((folder / "result.json").read_text())
        if result["optimizer_steps"] != 12 or result["start_code"] != code or result["settings"] != case["settings"]:
            raise ValueError("Short target execution differs")
        before = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        with (out / "logs" / (case["name"] + "-readonly.log")).open("x") as log:
            subprocess.run(case["command"] + ["--resume"], check=True, stdout=log, stderr=subprocess.STDOUT)
        if "verified_completed 0" not in (out / "logs" / (case["name"] + "-readonly.log")).read_text():
            raise ValueError("Completed target recovery differs")
        inputs = case["inputs"]
        data, _, guard = load_target_context(case["target"], inputs["observation"], inputs["record"],
            inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
        gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()} for n, item in data.items()}
        saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
        independent = independent_trainval(saved[case["target"]], data[case["target"]],
                                           result["selected_validation"]["metrics"][case["target"]])
        runtime = FullSourceTargetRuntime(case["settings"], gpu, folder,
            source_spec=case["source_spec"], initial_state_sha256=case["initial_state_sha256"])
        comparisons = []
        for _ in range(3):
            model = runtime.load_selected(folder / "primary.pt", gpu,
                folder / f'raw-validation-{result["selected_step"]:06d}.pt', device="cuda")
            before_model = state_id(model)
            _, current = runtime.validation_snapshot(model, gpu)
            for phase in ("train", "val"):
                for head in saved[case["target"]][phase]:
                    old, new = saved[case["target"]][phase][head], current[case["target"]][phase][head]
                    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                    if not torch.equal(old.argmax(1), new.argmax(1)):
                        raise ValueError("Target decisions changed")
            comparisons.append(_tree_fingerprint(current))
            if state_id(model) != before_model:
                raise ValueError("Frozen target changed")
            del model
        guard()
        if before != {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}:
            raise ValueError("Frozen completed target files changed")
        trace = json.loads((folder / "trace.json").read_text())
        row = {"name": case["name"], "updates": 12, "selected_step": result["selected_step"],
               "independent": independent, "strict_reload_repeats": 3,
               "distinct_repeat_fingerprints": len(set(comparisons)),
               "positive_m4_gradient_steps": sum(r["m4_parameter_gradient_norm"] > 0 for r in trace),
               "actual_m4_changes": sum(r["m4_changed_elements"] for r in trace)}
        if row["positive_m4_gradient_steps"] != 12 or row["actual_m4_changes"] <= 0:
            raise ValueError("M4 target updates missing")
        rows.append(row)
        write_json(out / "progress.json", rows)
        print("FULL_SOURCE_TARGET_SHORT_CASE", row, flush=True)
        del data, gpu
    require_frozen_code(code)
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError("Protected source/target input changed")
    write_json(out / "receipt.json", {"status": "passed", "code": code, "updates": 144, "cases": rows,
        "protected_files": len(protected), "query_labels_accessed": False, "test_scoring": False,
        "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()


"""Zero-update native/CSR/ordered audit of the frozen failed target short case."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def compare(old, new):
    delta = (new - old).abs()
    return {"max_abs": float(delta.max()), "unequal": int(torch.count_nonzero(delta)),
            "over_tolerance": int((~torch.isclose(new, old, rtol=1e-5, atol=1e-6)).sum()),
            "decision_changes": int((old.argmax(1) != new.argmax(1)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from joint_training import _tree_fingerprint
    from loadData.dataset_registry import file_hash
    from m4_full_target_runtime import FullSourceTargetRuntime
    from m4_ordered_source import OrderedValidation
    from research_runtime import code_snapshot, external_directory, write_json
    from target_data_context import load_target_context
    from target_initialization import state_id

    code = code_snapshot()
    root = args.run.resolve(strict=True)
    launch = json.loads((root / "launch.json").read_text())
    if code != launch["code"] or code["dirty"] or code["commit"] != "fb3a8b11e887423d50614d8668719759fd257034":
        raise ValueError("Frozen target producer differs")
    out = external_directory(args.out)
    if out.parent != root.parent or out == root:
        raise ValueError("Fresh sibling diagnostic output required")
    name = "paviaU-x_residual-source_pretrained_full_finetune"
    case = next(c for c in launch["cases"] if c["name"] == name)
    folder = root / "cases" / name
    pins = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    source = Path(case["source_spec"]["source_launch"]).parent
    source_launch = json.loads((source / "launch.json").read_text())
    for c in source_launch["cases"]:
        d = Path(c["out"])
        result = json.loads((d / "result.json").read_text())
        for key, value in result["artifacts"].items():
            if file_hash(d / key) != value:
                raise ValueError("Original source artifact differs")
            pins[str(d / key)] = value
        pins.update(c["source_context"]["protected_files"])
    inputs = case["inputs"]
    data, prov, guard = load_target_context("paviaU", inputs["observation"], inputs["record"],
        inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
    pins.update(prov["protected_files"])
    gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()} for n, item in data.items()}
    result = json.loads((folder / "result.json").read_text())
    state = torch.load(folder / "state-000012.pt", map_location="cpu", weights_only=True)
    saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
    if result["start_code"] != code or state["cursor"] != 12:
        raise ValueError("Frozen target state differs")
    native_mm = torch.sparse.mm
    def csr_mm(matrix, features):
        return native_mm(matrix.coalesce().to_sparse_csr(), features)
    rows = []
    out.mkdir(exist_ok=False)
    for backend in ("native", "csr", "ordered"):
        first = None
        for repeat in range(3):
            runtime = FullSourceTargetRuntime(case["settings"], gpu, folder,
                source_spec=case["source_spec"], initial_state_sha256=case["initial_state_sha256"])
            model = runtime.load_selected(folder / "primary.pt", gpu,
                folder / f'raw-validation-{result["selected_step"]:06d}.pt', device="cuda")
            if _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"]):
                raise ValueError("Loaded target weights differ")
            before_model = state_id(model)
            context = {"native": nullcontext, "csr": lambda: patch.object(torch.sparse, "mm", csr_mm),
                       "ordered": OrderedValidation}[backend]
            with context():
                _, current = runtime.validation_snapshot(model, gpu)
            if first is None:
                first = current
            records = []
            for phase in ("train", "val"):
                for head in saved["paviaU"][phase]:
                    new = current["paviaU"][phase][head]
                    records.append({"phase": phase, "head": head,
                        "vs_saved": compare(saved["paviaU"][phase][head], new),
                        "vs_first": compare(first["paviaU"][phase][head], new)})
            rows.append({"backend": backend, "repeat": repeat, "comparisons": records})
            if state_id(model) != before_model:
                raise ValueError("Frozen target model changed")
            del model
    guard()
    if any(file_hash(p) != h for p, h in pins.items()) or code_snapshot() != code:
        raise ValueError("Protected source/target artifacts changed")
    write_json(out / "receipt.json", {"status": "diagnostic_completed_original_failure_retained",
        "producer": code, "driver_sha256": file_hash(__file__), "case": name,
        "new_optimizer_updates": 0, "target_query_accessed": False, "protected_files_and_inputs": len(pins),
        "trained_cases": 4, "trained_updates": 48, "accepted_cases": 3, "cases_not_started": 8,
        "original_tolerance_relaxed": False, "rows": rows})


if __name__ == "__main__":
    main()


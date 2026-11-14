"""Freeze the controller, worker, scientific inputs and destination independently."""
import json
import os
from pathlib import Path
import subprocess
import sys

from model.research_config import digest
from research_runtime import ROOT, code_snapshot, hash_file, require_frozen_code, write_json
from transfer import p4_coverage_batch as batch, p4_runtime_device as device
from transfer import p4_three_arm_contract as fit
from transfer.g2_contract import read_pinned


POLICY = "wp05_p4_independent_slot_scope_v1"
REAL = "honghu_real_prepared"
CPU = "cpu_engineering"
CUDA = "cuda_recovery_engineering"
RECOVERY = "synthetic_same_device_recovery"
PAIRED = {"full": [60], "split": [2, 30, 60]}
GPU1, GPU6 = fit.REFERENCE_QUALIFICATION_SHA
THRESHOLDS = {"free_mib_min": 12288, "utilization_max": 60}
RELEASE_SCHEMA = "wp05_p4_independent_execution_release_v1"
SCIENCE_KEYS = {
    "policy", "purpose", "commit", "source_hash", "scope_hash", "synthetic", "registration",
    "parent_registration", "prior_receipt", "protected_files", "protocol", "gpus", "source",
    "budget", "cases", "exposure", "observation_diagnostics", "preflight", "parameter_audit",
    "paths", "qualifications", "operator_geometry", "recovery_receipt",
}
SCOPE_KEYS = {"schema", "mode", "classes", "science", "science_fields", "controller", "worker", "execution",
              "slots", "thresholds", "physical_recoveries"}
WORKER_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": device.WORKSPACE}


def require(condition, reason):
    device.require(condition, reason)


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hash_file(path)}


def read(item):
    path = Path(item["path"])
    require(set(item) == {"path", "sha256"} and path.is_absolute() and path == path.resolve(strict=True),
            "Expected an absolute pinned artifact")
    return read_pinned(path, item["sha256"])


def identity(code):
    return {k: code[k] for k in ("commit", "source_hash")}


def environment(root, gpu=None):
    env = {**os.environ, **WORKER_ENV, "CUDA_VISIBLE_DEVICES": gpu or "",
           "PYTHONPATH": str(root)}
    env.pop("PYTHONSTARTUP", None)
    return env


def worker_identity(root):
    root = Path(root).resolve(strict=True)
    top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True)
    code = code_snapshot(root)
    require(root != ROOT and Path(top.strip()) == root and not code["dirty"]
            and identity(code) == {"commit": fit.ADMISSION_COMMIT, "source_hash": fit.ADMISSION_SOURCE},
            "A clean, exact frozen worker checkout is required")
    return {"root": str(root), **identity(code)}


def authenticate_worker(registration, row, recoveries):
    """Run the frozen loader and recovery verifier with the worker's own imports."""
    root = registration["worker"]["root"]
    script = '''
import json
from pathlib import Path
import sys
from research_runtime import code_snapshot
from transfer import p4_coverage_batch as batch
from transfer import p4_full_budget_registration as reg
from tests.test_p4_stable_cuda import synthetic_schema
from contextlib import nullcontext

scope_path, checksum, classes, recoveries = sys.argv[1:]
code = code_snapshot()
classes = int(classes)
with synthetic_schema(classes) if classes == 9 else nullcontext():
    scope, science = reg.load(Path(scope_path), checksum, code)
for uuid, item in json.loads(recoveries).items():
    proof = json.loads(Path(item["path"]).read_text())
    if proof["physical_uuid"] != uuid:
        raise ValueError("Recovery UUID differs")
    pins = reg.bind_recovery(Path(item["path"]), reg.ADMISSION, scope["protocol"], False, code)
    if pins[item["path"]] != item["sha256"]:
        raise ValueError("Recovery pin differs")
    for case in proof["cases"]:
        source = json.loads(Path(case["scope"]).read_text())
        if source["qualifications"] != {uuid: scope["qualifications"][uuid]}:
            raise ValueError("Recovery qualification differs")
    batch.check_pins(pins)
print(json.dumps({"registered_cases": len(scope["cases"]), "protected_files": len(scope["protected_files"])}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, registration["science"]["path"], registration["science"]["sha256"],
         str(registration["classes"]), json.dumps(recoveries)],
        cwd=root, env=environment(root), text=True, capture_output=True, check=False,
    )
    require(result.returncode == 0, "Frozen worker registration rejected: " + result.stderr)
    require(json.loads(result.stdout) == {"registered_cases": len(row["cases"]),
                                           "protected_files": len(row["protected_files"])},
            "Frozen worker registration summary differs")


def projection(science_pin, row, *, real):
    keys = set(row)
    if real:
        require(science_pin["sha256"] == fit.ADMISSION_SHA and keys == SCIENCE_KEYS,
                "Expected the exact original real science scope")
    elif row["purpose"] == RECOVERY:
        require(keys == (SCIENCE_KEYS - {"recovery_receipt"}) | {"recovery_classes"},
                "Unknown synthetic recovery scope fields")
    else:
        require(keys == SCIENCE_KEYS - {"recovery_receipt"}, "Unknown synthetic admission scope fields")
    require(identity(row) == {"commit": fit.ADMISSION_COMMIT, "source_hash": fit.ADMISSION_SOURCE}
            and row["paths"] == PAIRED and row["budget"]["steps"] == 60
            and row["budget"]["validation_steps"] == [30, 60]
            and row["budget"]["planned_optimizer_updates"] == 120 * len(row["cases"])
            and row["preflight"]["optimizer_updates"] == row["preflight"]["forward_calls"] == 0
            and row["preflight"]["raw_gt_loaded"] is row["preflight"]["query_accessed"] is False,
            "Unexpected paired science, source or budget")
    return {name: digest(row[name]) for name in sorted(keys)}


def slots(row, mode, classes):
    cases = row["cases"]
    if mode == REAL:
        require(classes == 22 and row["synthetic"] is False and row["purpose"] == "real_admission"
                and row["gpus"] == [GPU1, GPU6] and row["protocol"] == device.CUDA_FP64_WEIGHT
                and len(cases) == 20
                and {gpu: row["protected_files"][p] for gpu, p in row["qualifications"].items()}
                == fit.REFERENCE_QUALIFICATION_SHA, "Unexpected HongHu physical or scientific assignment")
        assignments = [(gpu, gpu, [c for c in cases if c["physical_uuid"] == gpu]) for gpu in row["gpus"]]
    elif mode == CUDA:
        require(classes in (9, 22) and row["synthetic"] is True and row["purpose"] == RECOVERY
                and row["protocol"] == device.CUDA_FP64_WEIGHT and row["gpus"] == [GPU6]
                and row["recovery_classes"] == classes and len(cases) == 1
                and row["protected_files"][row["qualifications"][GPU6]] == fit.REFERENCE_QUALIFICATION_SHA[GPU6],
                "CUDA engineering requires the registered GPU6 recovery scope")
        assignments = [(GPU6, GPU6, cases)]
    else:
        require(mode == CPU and classes in (9, 22) and row["synthetic"] is True
                and row["protocol"] == device.CPU and row["gpus"] == [] and not row["qualifications"],
                "CPU engineering requires an audited CPU fixture")
        if classes == 9:
            require(row["purpose"] == RECOVERY and row["recovery_classes"] == 9 and len(cases) == 1,
                    "Expected the bounded nine-class recovery fixture")
            assignments = [("cpu-a", None, cases)]
        else:
            require(row["purpose"] == "real_admission" and len(cases) == 20,
                    "Expected the twenty-case CPU fixture")
            assignments = [
                (name, None, [c for c in cases if ((c["k"] == 5) + (c["method"] == "shared_transfer")) % 2 == i])
                for i, name in enumerate(("cpu-a", "cpu-b"))
            ]
    return {name: {"physical_uuid": uuid, "cases": [c["id"] for c in members],
                  "updates": 120 * len(members)} for name, uuid, members in assignments}


def validate(resource, *, code=None, live_worker=False):
    require(set(resource) == SCOPE_KEYS and resource["schema"] == POLICY
            and resource["thresholds"] == THRESHOLDS, "Unknown resource scope schema or limits")
    controller, worker = resource["controller"], resource["worker"]
    require(set(controller) == {"commit", "source_hash"} and set(worker) == {"root", "commit", "source_hash"}
            and identity(worker) == {"commit": fit.ADMISSION_COMMIT, "source_hash": fit.ADMISSION_SOURCE},
            "Unexpected controller or worker identity fields")
    if code is not None:
        require_frozen_code(code)
        require(controller == identity(code), "Controller identity differs")
    science = read(resource["science"])
    mode, classes = resource["mode"], resource["classes"]
    real = mode == REAL
    execution = Path(resource["execution"])
    require(execution == batch.sibling(execution, resource["science"]["path"])
            and execution != Path(resource["science"]["path"]).parent, "Execution must be an isolated sibling")
    require(resource["science_fields"] == projection(resource["science"], science, real=real)
            and resource["slots"] == slots(science, mode, classes), "Science projection or slot mapping differs")
    recoveries = resource["physical_recoveries"]
    require(set(recoveries) == (set(science["gpus"]) if real else set()),
            "Expected both separately pinned real same-device recovery proofs")
    if real:
        require(recoveries[GPU1]["path"] == science["recovery_receipt"]
                and recoveries[GPU1]["sha256"] == science["protected_files"][science["recovery_receipt"]],
                "Original GPU1 recovery differs")
    if live_worker:
        require(worker_identity(worker["root"]) == worker, "Worker checkout differs")
        authenticate_worker(resource, science, recoveries)
    return science


def validate_release(item, resource, release):
    """Bind a future main-reviewed release to exactly one prepared real scope."""
    if resource["mode"] != REAL:
        require(release is None, "Synthetic engineering cannot consume a real execution release")
        return
    require(release is not None, "Real independent scope is preparation-only until separately released")
    row = read(release)
    expected = {
        "schema": RELEASE_SCHEMA, "decision": "approved", "research_issue": 470,
        "resource_scope": item, "science_scope": resource["science"], "controller": resource["controller"],
        "worker": identity(resource["worker"]), "execution": resource["execution"],
        "physical_recoveries": resource["physical_recoveries"], "slots": resource["slots"],
        "optimizer_updates": 2400, "raw_gt_authorized": False, "query_authorized": False,
    }
    require(row == expected, "Real independent execution release does not match the prepared scope")


def prepare(science_pin, target, out, worker_root, code, *, mode, classes, recoveries):
    require_frozen_code(code)
    row = read(science_pin)
    execution = batch.sibling(target, science_pin["path"])
    out = batch.sibling(out, science_pin["path"])
    require(out != execution and not out.exists() and not execution.exists(),
            "Resources and execution must use new sibling directories")
    resource = {"schema": POLICY, "mode": mode, "classes": classes, "science": science_pin,
                "science_fields": projection(science_pin, row, real=mode == REAL),
                "controller": identity(code), "worker": worker_identity(worker_root),
                "execution": str(execution), "slots": slots(row, mode, classes),
                "thresholds": THRESHOLDS, "physical_recoveries": recoveries}
    validate(resource, code=code, live_worker=True)
    out.mkdir()
    path = out / "resource-scope.json"
    write_json(path, resource)
    return pin(path)

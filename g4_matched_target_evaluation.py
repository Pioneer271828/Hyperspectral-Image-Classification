"""New all54 frozen query boundary; never accepts the historical native27 manifest."""
from pathlib import Path
import statistics
import math

from g4_matched_target_audit import AUDIT_POLICY, check_files, full_acceptance, read
from g4_matched_target_execution import case_data, unchanged_producer
from g4_matched_target_study import EXPECTED, MODES, SEEDS, case_name, grid
from joint_training import PREPARATION_HASH
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from source_data_context import bounded

QUERY_POLICY = "g4_matched_ordered_all54_frozen_query_v2"


def require_frozen_boundary(freeze, *, full_integrity=True):
    models, checks = freeze.get("models", []), freeze.get("independent_checks", [])
    expected = set(grid())
    if (freeze.get("policy") != QUERY_POLICY or freeze.get("status") != "all54_frozen_before_query"
            or freeze.get("study") != EXPECTED or len(models) != 54 or len(checks) != 54
            or len({m["name"] for m in models}) != 54
            or {(m["target"], m["seed"], m["method"], m["mode"]) for m in models} != expected
            or {m["name"] for m in models} != {c["name"] for c in checks}
            or any(c["status"] != "passed" or c["updates"] != 1500 or c["validation_events"] != 51 for c in checks)
            or sum(c["head_metrics"] for c in checks) != 324 or sum(c["class_recalls"] for c in checks) != 4320
            or sum(c["exact_reload_repeats"] for c in checks) != 162
            or freeze.get("training_optimizer_updates") != 81000 or freeze.get("new_optimizer_updates") != 0
            or freeze.get("query_labels_accessed") is not False):
        raise ValueError("All54 full training and independent ordered audit required before any query release")
    for m in models:
        row = next(c for c in checks if c["name"] == m["name"])
        if (m["name"] != case_name(m["target"], m["seed"], m["method"], m["mode"])
                or row["head_metrics"] != 6 or row["class_recalls"] != 6 * DATASETS[m["target"]][2]
                or row["exact_reload_repeats"] != 3 or row["result_sha256"] != m["result_sha256"]
                or row["primary_sha256"] != m["primary_sha256"] or row["sidecar_sha256"] != m["sidecar_sha256"]):
            raise ValueError("Full query model/audit pair differs")
        folder = Path(freeze["training_root"]) / "cases" / m["name"]
        for name, field in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            if freeze["boundary_files"].get(str(folder / name)) != m[field]:
                raise ValueError("Every frozen query model pair must remain protected")
    if any(freeze["protected_files"].get(p) != h for p, h in freeze["boundary_files"].items()):
        raise ValueError("Live boundary pins differ from full integrity manifest")
    check_files(freeze["protected_files"] if full_integrity else freeze["boundary_files"])
    return True


def freeze_for_query(root, launch_sha256, acceptance_sha256, audit_path, audit_sha256, out):
    """Only authenticated completed training + independent audit can create this boundary."""
    root, out = external_directory(root), external_directory(out)
    audit_path = Path(audit_path).resolve(strict=True)
    if out == root or root in out.parents or out in root.parents or out == audit_path.parent or audit_path.parent in out.parents:
        raise ValueError("New independent freeze output required")
    code = code_snapshot()
    launch, accepted, pins = full_acceptance(root, launch_sha256, acceptance_sha256, code)
    if file_hash(audit_path) != audit_sha256:
        raise ValueError("Pinned completed independent full54 audit required")
    audit = read(audit_path)
    unchanged_producer(launch["code"], audit["code"])
    unchanged_producer(audit["code"], code)
    if (audit["policy"] != AUDIT_POLICY or audit["status"] != "passed" or audit["producer"] != launch["code"]
            or audit["launch_sha256"] != launch_sha256 or audit["acceptance_sha256"] != acceptance_sha256
            or audit["protected_manifest_sha256"] != accepted["protected_manifest_sha256"]
            or audit["new_optimizer_updates"] != 0 or audit["query_labels_accessed"] is not False
            or audit["test_scoring"] is not False):
        raise ValueError("Independent full target audit does not match training")
    models = []
    live_pins = {str(root / "launch.json"): launch_sha256, str(root / "training-acceptance.json"): acceptance_sha256,
                 str(audit_path): audit_sha256}
    for case in launch["cases"]:
        row = next(r for r in accepted["models"] if r["name"] == case["name"])
        checked = next(r for r in audit["cases"] if r["name"] == case["name"])
        if any(row[k] != checked[k] for k in checked):
            raise ValueError("Independent full audit evidence differs")
        _, _, guard = case_data(case)
        guard()
        models.append({**{k: case[k] for k in ("name", "target", "seed", "method", "mode")},
            **{k: row[k] for k in ("result_sha256", "primary_sha256", "sidecar_sha256", "selected_step")},
            "case_sha256": digest(case), "family_hash": case["target_context"]["family_hash"]})
        for name, field in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            live_pins[str(Path(case["out"]) / name)] = row[field]
        live_pins.update({case["inputs"][key]: case["inputs"][key + "_sha256"] for key in ("record", "family")})
    pins[str(audit_path)] = audit_sha256
    freeze = {"policy": QUERY_POLICY, "status": "all54_frozen_before_query", "study": EXPECTED,
        "code": code, "producer": launch["code"], "training_root": str(root), "launch_sha256": launch_sha256,
        "acceptance_sha256": acceptance_sha256, "independent_audit_sha256": audit_sha256,
        "models": models, "independent_checks": audit["cases"], "training_optimizer_updates": 81000,
        "new_optimizer_updates": 0, "query_labels_accessed": False, "protected_files": pins, "boundary_files": live_pins}
    require_frozen_boundary(freeze)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    write_json(out / "freeze.json", freeze)
    return freeze


def read_query_labels(freeze, case, data_root, prepared_manifest, prepared_sha256):
    require_frozen_boundary(freeze, full_integrity=False)
    frozen = next(m for m in freeze["models"] if m["name"] == case["name"])
    if digest(case) != frozen["case_sha256"] or case["target_context"]["family_hash"] != frozen["family_hash"]:
        raise ValueError("Query case/label family is outside frozen54 manifest")
    if prepared_sha256 != PREPARATION_HASH or file_hash(prepared_manifest) != prepared_sha256:
        raise ValueError("Accepted prepared cohort metadata required")
    manifest, target = read(prepared_manifest), case["target"]
    raw = manifest["datasets"][target]["raw"]
    if manifest["protocol"] != "acc1p_joint_prepared_v1" or raw["class_count"] != DATASETS[target][2]:
        raise ValueError("Query cohort/target differs")
    inputs = case["inputs"]
    if file_hash(inputs["family"]) != inputs["family_sha256"]:
        raise ValueError("Frozen target label family changed")
    family = read(inputs["family"])
    weight = Path(case["out"]) / "primary.pt"
    if family["family_hash"] != frozen["family_hash"] or file_hash(weight) != frozen["primary_sha256"]:
        raise ValueError("Frozen query family/checkpoint differs")
    path = bounded(Path(data_root).resolve(strict=True), raw["label_path"])
    if file_hash(path) != raw["label_sha256"]:
        raise ValueError("Final raw target labels changed")
    from scipy.io import loadmat
    from loadData.target_label_protocol import final_query
    key = DATASETS[target][1] + "_gt"
    labels = loadmat(path, variable_names=[key])[key]
    release = final_query(family, labels, expected_hash=family["family_hash"], checkpoint_sha256=frozen["primary_sha256"])
    if file_hash(path) != raw["label_sha256"]:
        raise ValueError("Raw labels changed during final release")
    return release, {str(path): raw["label_sha256"]}


def paired_summary(rows):
    """Report planned pairs, without selecting a mode or changing a hypothesis."""
    lookup = {(r["target"], r["seed"], r["method"], r["mode"]): r for r in rows}
    if len(rows) != 54 or set(lookup) != set(grid()):
        raise ValueError("All54 paired query rows required")
    for row in rows:
        metric = row["metrics"]["fused"]
        classes = list(range(1, DATASETS[row["target"]][2] + 1))
        if (metric["class_ids"] != classes or [c["class_id"] for c in metric["per_class"]] != classes
                or any(c["recall"] is None or not math.isfinite(c["recall"]) for c in metric["per_class"])
                or any(not math.isfinite(metric[k]) for k in ("oa", "aa", "nll"))):
            raise ValueError("Complete finite class-supported query metrics required")
    grouped = {}
    for mode in MODES:
        for method in EXPECTED["methods"]:
            for target in EXPECTED["targets"]:
                values = [lookup[(target, s, method, mode)]["metrics"]["fused"] for s in SEEDS]
                grouped["/".join((mode, method, target))] = {k: {"mean": statistics.mean(v[k] for v in values),
                    "sample_sd": statistics.stdev(v[k] for v in values)} for k in ("oa", "aa")}
    pairs = {}
    contrasts = [(mode, EXPECTED["methods"][1], mode, EXPECTED["methods"][0]) for mode in MODES]
    contrasts += [(a, method, b, method) for a, b in (("x_residual", "capacity_control"),
        ("capacity_control", "none"), ("x_residual", "none")) for method in EXPECTED["methods"]]
    for a, am, b, bm in contrasts:
        per_seed = []
        for seed in SEEDS:
            delta = {k: statistics.mean(lookup[(t, seed, am, a)]["metrics"]["fused"][k]
                - lookup[(t, seed, bm, b)]["metrics"]["fused"][k] for t in EXPECTED["targets"]) for k in ("oa", "aa")}
            per_seed.append(delta)
        entry = {k: {"mean": statistics.mean(v[k] for v in per_seed),
            "sample_sd": statistics.stdev(v[k] for v in per_seed), "seed_deltas": [v[k] for v in per_seed]} for k in ("oa", "aa")}
        entry["per_class_mean_recall_delta"] = {t: [statistics.mean(
            lookup[(t, s, am, a)]["metrics"]["fused"]["per_class"][i]["recall"]
            - lookup[(t, s, bm, b)]["metrics"]["fused"]["per_class"][i]["recall"] for s in SEEDS)
            for i in range(DATASETS[t][2])] for t in EXPECTED["targets"]}
        pairs[a + "/" + am + " - " + b + "/" + bm] = entry
    return {"grouped_by_target": grouped, "paired_macro_scene_deltas_by_model_seed": pairs,
            "support_seed_repetitions": 1, "accuracy_hypothesis_decided_here": False}

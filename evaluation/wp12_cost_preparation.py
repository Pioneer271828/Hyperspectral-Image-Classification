"""Prepare label-free hardware workloads without admitting a changed runtime."""

from pathlib import Path

import torch

from evaluation.wp12_evidence import accuracy_records
from g4_matched_target_audit import read, check_files
from g4_matched_target_evaluation import new_output
from joint_baseline import _item_fingerprint
from joint_training import _tree_fingerprint
from loadData.dataset_registry import DATASETS
from model.research_config import digest
from research_runtime import hash_file, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS, observation_id


POLICY = "wp12-frozen54-cost-preparation-v1"


def runtime_admission(original, current):
    """Conservative original-file preservation gate; additions are permitted."""
    if (original.get("dirty") is not False or current.get("dirty") is not False
            or not original.get("files") or not current.get("files")
            or digest(original["files"]) != original["source_hash"]
            or digest(current["files"]) != current["source_hash"]):
        raise ValueError("Authentic clean source manifests required")
    differences = [dict(path=p, original_sha256=h, candidate_sha256=current["files"].get(p))
                   for p, h in sorted(original["files"].items()) if current["files"].get(p) != h]
    return dict(status="original_files_preserved" if not differences else "compatibility_pending",
                original_commit=original["commit"], candidate_commit=current["commit"],
                differences=differences, checked_files=len(original["files"]),
                note="File differences do not prove numerical incompatibility; no forward equivalence claimed")


def prepare(analysis, receipt, freeze, launch, code, out, *, analysis_sha256, freeze_sha256, launch_sha256,
            finish=lambda: None):
    accuracy = accuracy_records(analysis, receipt, freeze, analysis_sha=analysis_sha256, freeze_sha=freeze_sha256)
    if launch_sha256 != freeze["launch_sha256"] or launch["code"] != freeze["producer"]:
        raise ValueError("Frozen launch/source identity differs")
    admission = runtime_admission(freeze["producer"], code)
    frozen = {r["name"]: r for r in freeze["models"]}
    cases = launch["cases"]
    if len(cases) != 54 or {c["name"] for c in cases} != frozen.keys():
        raise ValueError("Exactly the frozen54 launch cases required")
    sources, pins, rows = {}, {}, []
    lookup = {(r["scene"], r["seed"], r["arm"]): r for r in accuracy["cases"]}
    for case in cases:
        model = frozen[case["name"]]
        if (digest(case) != model["case_sha256"] or case["target_context"]["family_hash"] != model["family_hash"]
                or Path(case["out"]).resolve() != Path(freeze["training_root"]) / "cases" / case["name"]):
            raise ValueError("Case differs from frozen launch")
        inputs, scene = case["inputs"], case["target"]
        record_path, obs_path = (str(Path(inputs[k]).resolve(strict=True)) for k in ("record", "observation"))
        if (freeze["boundary_files"].get(record_path) != inputs["record_sha256"]
                or hash_file(record_path) != inputs["record_sha256"]):
            raise ValueError("Frozen observation record differs")
        record = read(record_path)
        if (record.get("policy") != GRAPH_POLICY or record.get("target") != scene or record.get("graph_fit_labels") != 0
                or freeze["protected_files"].get(obs_path) != record.get("observation_sha256")):
            raise ValueError("Accepted label-free observation pin required")
        source = dict(record_path=record_path, observation_path=obs_path, record=record)
        if sources.setdefault(scene, source) != source:
            raise ValueError("Paired models must share one scene observation")
        pins.update({record_path: inputs["record_sha256"], obs_path: record["observation_sha256"]})
        row = lookup[scene, case["seed"], case["method"] + "/" + case["mode"]]
        folder = Path(case["out"])
        rows.append({**{k: row[k] for k in ("scene", "seed", "arm", "evaluation_sha256", "checkpoint_sha256")},
                     "checkpoint": {"path": str(folder / "primary.pt"), "sha256": model["primary_sha256"]},
                     "sidecar": {"path": str(folder / "primary.pt.json"), "sha256": model["sidecar_sha256"]}})
    check_files(pins)
    out = new_output(Path(out), launch)
    if out.exists():
        raise ValueError("New external preparation directory required")
    # Do not deserialize the family or any ground-truth file. The original
    # observation format is explicitly label-free, including its metadata.
    out.mkdir()
    workloads = {}
    for scene, source in sources.items():
        item = torch.load(source["observation_path"], map_location="cpu", weights_only=True)
        if (set(item) != OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
                or item["name"] != scene or item["class_num"] != DATASETS[scene][2]
                or _item_fingerprint(item) != source["record"]["prepared_fields"]):
            raise ValueError("Observation fields/metadata differ or include labels")
        payload = {k: item[k] for k in sorted(OBS_KEYS)}
        observation_id(payload)
        if (tuple(payload["net_input"].shape) != (item["height"], item["width"], item["bands"] + 2)):
            raise ValueError("Observation shape and metadata differ")
        path = out / (scene + "-observation.pt")
        torch.save(payload, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        identity = _tree_fingerprint(payload)
        if _tree_fingerprint(restored) != identity:
            raise ValueError("Prepared observation changed during serialization")
        workloads[scene] = dict(file={"path": str(path), "sha256": hash_file(path)},
                                tensor_sha256=identity, work_items=item["height"] * item["width"],
                                original_observation_sha256=source["record"]["observation_sha256"])
        del item, payload, restored
    check_files(pins)
    check_files(freeze["boundary_files"])
    finish()
    result = dict(policy=POLICY, status="prepared", accuracy_protocol_sha256=accuracy["protocol_sha256"],
                  freeze_sha256=freeze_sha256, launch_sha256=launch_sha256, analysis_sha256=analysis_sha256,
                  cases=rows, workloads=workloads, input_pins=pins, runtime_admission=admission,
                  executable_cost_jobs_created=False, hardware_plan_registered=False,
                  optimizer_updates=0, query_labels_deserialized=False, model_inference_executed=False,
                  remaining=["resolve runtime admission", "register hardware/environment/budget", "measure all54"])
    write_json(out / "preparation.json", result)
    return result

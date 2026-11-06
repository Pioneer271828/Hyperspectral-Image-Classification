"""Engineering-only bindings for independent support seeds/K; no training or query."""
import itertools
from pathlib import Path
import time

import torch
from g4_matched_target_execution import unchanged_producer
from joint_baseline import NAMES, _item_fingerprint
from loadData.dataset_registry import DATASETS, file_hash
from loadData.target_label_protocol import BUDGETS, phase_view
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS, observation_id

POLICY = "g4_independent_support_context_engineering_v1"
INPUT_POLICY = "g4_support_context_inputs_v1"
LABEL_PRODUCER = "2f9e661bfcb777ed6b8dd2e8b6211a998eac4fdc"
SEEDS = (64, 512, 2048)
LABEL_KEYS = {"train_indices", "train_labels", "val_indices", "val_labels"}


def read(path):
    import json
    return json.loads(Path(path).read_text())


def check_files(pins):
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Pinned support/observation input changed")


def grid():
    return list(itertools.product(NAMES, SEEDS, BUDGETS))


def context_name(target, support_seed, per_class):
    if (type(support_seed) is not int or type(per_class) is not int
            or (target, support_seed, per_class) not in grid()):
        raise ValueError("Explicit registered target/support seed/K required")
    return f"{target}-support{support_seed}-k{per_class}"


def load_inputs(path, sha256):
    path = Path(path).resolve(strict=True)
    if file_hash(path) != sha256:
        raise ValueError("Pinned explicit support context inputs required")
    spec = read(path)
    if (set(spec) != {"policy", "label_receipt", "independent_receipt", "observations", "families"}
            or spec["policy"] != INPUT_POLICY or set(spec["observations"]) != set(NAMES)
            or len(spec["families"]) != 9
            or {(r["target"], r["support_seed"]) for r in spec["families"]} != set(itertools.product(NAMES, SEEDS))):
        raise ValueError("Exactly three observations and nine accepted support families required")
    pins = {str(path): sha256}
    for key in ("label_receipt", "independent_receipt"):
        value = spec[key]
        if set(value) != {"path", "sha256"}:
            raise ValueError("Explicit accepted label receipt pin required")
        pins[value["path"]] = value["sha256"]
    for value in spec["families"]:
        if set(value) != {"target", "support_seed", "path", "sha256"} or type(value["support_seed"]) is not int:
            raise ValueError("Explicit support-family role and file pin required")
        pins[value["path"]] = value["sha256"]
    for value in spec["observations"].values():
        if set(value) != {"observation", "observation_sha256", "record", "record_sha256"}:
            raise ValueError("Explicit unlabeled observation/record pair required")
        pins[value["observation"]] = value["observation_sha256"]
        pins[value["record"]] = value["record_sha256"]
    if len(pins) != 18 or any(not Path(p).is_absolute() for p in pins):
        raise ValueError("Eighteen distinct absolute accepted input file pins required")
    check_files(pins)
    receipt, audit = [read(spec[k]["path"]) for k in ("label_receipt", "independent_receipt")]
    for value in (receipt, audit):
        if (value["status"] != "passed" or value["source"]["commit"] != LABEL_PRODUCER
                or value["source"]["dirty"] or value["optimizer_updates"] != 0
                or value["model_test_scoring"] is not False or len(value["rows"]) != 27):
            raise ValueError("Accepted zero-update curation and independent verification required")
    if (receipt["run_views_have_query_labels"] is not False or receipt["target_graph_built"] is not False
            or audit["release_checks"] != 54
            or {(r["target"], r["support_seed"], r["support_per_class"]) for r in receipt["rows"]} != set(grid())
            or {(r["target"], r["support_seed"], r["k"]) for r in audit["rows"]} != set(grid())):
        raise ValueError("All27 accepted support/K releases and54 checks required")
    return spec, receipt, audit, pins


def validate_overlay(overlay, target, per_class, shape):
    count, pixels = DATASETS[target][2], shape[0] * shape[1]
    if not isinstance(overlay, dict) or set(overlay) != LABEL_KEYS:
        raise ValueError("Only sanitized training/validation label tensors allowed")
    for phase, k in (("train", per_class), ("val", 5)):
        ids, labels = overlay[phase + "_indices"], overlay[phase + "_labels"]
        if (not torch.is_tensor(ids) or not torch.is_tensor(labels) or ids.dtype != torch.long
                or labels.dtype != torch.long or ids.device.type != "cpu" or labels.device.type != "cpu"
                or ids.shape != (count * k,) or labels.shape != ids.shape
                or (ids < 0).any() or (ids >= pixels).any() or (labels < 0).any() or (labels >= count).any()
                or not torch.equal(ids, ids.sort().values) or len(ids.unique()) != len(ids)
                or not torch.equal(torch.bincount(labels, minlength=count), torch.full((count,), k))):
            raise ValueError("Support/K indices, class counts or validation budget differ")
    if set(overlay["train_indices"].tolist()) & set(overlay["val_indices"].tolist()):
        raise ValueError("Support and validation roles overlap")


def overlay_identity(overlay):
    return digest({k: tensor_hash(overlay[k]) for k in sorted(LABEL_KEYS)})


def prepare_contexts(spec_path, spec_sha256, out, code):
    started = time.monotonic()
    spec, receipt, audit, pins = load_inputs(spec_path, spec_sha256)
    out = external_directory(out)
    owners = {Path(p).resolve().parent for p in pins}
    if out.exists() or any(out == p or p in out.parents or out in p.parents for p in owners):
        raise ValueError("Fresh output outside every accepted input tree required")
    records = {}
    for target, value in spec["observations"].items():
        record = read(value["record"])
        if (record["policy"] != GRAPH_POLICY or record["target"] != target or record["graph_fit_labels"] != 0
                or record["observation_sha256"] != value["observation_sha256"]):
            raise ValueError("Accepted label-free target observation required")
        item = torch.load(value["observation"], map_location="cpu", weights_only=True)
        if (set(item) != OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
                or item["name"] != target or item["class_num"] != DATASETS[target][2]
                or _item_fingerprint(item) != record["prepared_fields"]):
            raise ValueError("Unlabeled target observation identity differs")
        observation_id({k: item[k] for k in OBS_KEYS})
        records[target] = {"input": value, "shape": [item["height"], item["width"]],
            "graph_identity": digest(record["prepared_fields"])}
        del item
    rows, overlays = [], []
    for target, support_seed, per_class in grid():
        family_pin = next(v for v in spec["families"] if (v["target"], v["support_seed"]) == (target, support_seed))
        family = read(family_pin["path"])
        if (family["target"] != target or family["seed"] != support_seed
                or family["shape"] != records[target]["shape"]
                or family["classes"] != list(range(1, DATASETS[target][2] + 1))):
            raise ValueError("Support family target/seed/geometry differs")
        releases = {phase: phase_view(family, per_class, phase, expected_hash=family["family_hash"])
                    for phase in ("adapt", "validation")}
        accepted = next(v for v in receipt["rows"] if (v["target"], v["support_seed"], v["support_per_class"]) == (target, support_seed, per_class))
        checked = next(v for v in audit["rows"] if (v["target"], v["support_seed"], v["k"]) == (target, support_seed, per_class))
        count = DATASETS[target][2]
        if (accepted["family_hash"] != family["family_hash"]
                or accepted["query_indices_hash"] != releases["adapt"]["query_indices_hash"]
                or accepted["training_labels"] != count * per_class or accepted["validation_labels"] != count * 5
                or accepted["total_annotation_labels"] != count * (per_class + 5)
                or any(checked[k] != accepted[k] for k in ("training_labels", "validation_labels", "query_pixels"))):
            raise ValueError("Support family differs from accepted independent label releases")
        overlay = {}
        for source, phase in (("adapt", "train"), ("validation", "val")):
            overlay[phase + "_indices"] = torch.tensor(releases[source]["indices"], dtype=torch.long)
            overlay[phase + "_labels"] = torch.tensor(releases[source]["labels"], dtype=torch.long) - 1
        validate_overlay(overlay, target, per_class, family["shape"])
        name = context_name(target, support_seed, per_class)
        rows.append({"name": name, "target": target, "support_seed": support_seed, "support_per_class": per_class,
            "validation_per_class": 5, "total_annotation_labels": count * (per_class + 5),
            "shape": family["shape"], "family_hash": family["family_hash"],
            "query_indices_hash": releases["adapt"]["query_indices_hash"],
            "phase_hashes": {p: digest(v) for p, v in releases.items()},
            "graph_identity": records[target]["graph_identity"], "overlay_identity": overlay_identity(overlay),
            "file": name + ".pt"})
        overlays.append(overlay)
    check_files(pins)
    out.mkdir(exist_ok=False)
    for row, overlay in zip(rows, overlays):
        path = out / row["file"]
        torch.save(overlay, path)
        row["sha256"] = file_hash(path)
    manifest = {"policy": POLICY, "code": code, "phase": "engineering_only", "execution_enabled": False,
        "optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False,
        "full_target_scene_transductive": True, "spatial_isolation": False,
        "input_spec_sha256": spec_sha256, "out_dir": str(out), "observations": records, "contexts": rows,
        "protected_files": pins}
    validate_manifest(manifest, out)
    require_frozen_code(code)
    check_files(pins)
    write_json(out / "manifest.json", manifest)
    write_json(out / "receipt.json", {"status": "passed", "policy": POLICY, "code": code, "contexts": 27,
        "manifest_sha256": file_hash(out / "manifest.json"), "input_spec_sha256": spec_sha256,
        "source_files_verified_unchanged": len(pins), "optimizer_updates": 0,
        "query_labels_accessed": False, "test_scoring": False, "elapsed_seconds": time.monotonic() - started,
        "training_executable": False, "new_observation_or_graph": False})
    return manifest


def validate_manifest(value, root):
    rows = value["contexts"]
    if (value["policy"] != POLICY or value["phase"] != "engineering_only"
            or value["execution_enabled"] is not False or value["optimizer_updates"] != 0
            or value["query_labels_accessed"] is not False or value["test_scoring"] is not False
            or value["full_target_scene_transductive"] is not True or value["spatial_isolation"] is not False
            or Path(value["out_dir"]).resolve() != root or len(rows) != 27
            or set(value["observations"]) != set(NAMES)
            or {(r["target"], r["support_seed"], r["support_per_class"]) for r in rows} != set(grid())):
        raise ValueError("Complete engineering-only27 support contexts required")
    overlays = {}
    for target, observation in value["observations"].items():
        for key in ("observation", "record"):
            source = observation["input"]
            if value["protected_files"].get(source[key]) != source[key + "_sha256"]:
                raise ValueError("Shared observation pair is outside authenticated source pins")
    for row in rows:
        name = context_name(row["target"], row["support_seed"], row["support_per_class"])
        if (row["name"] != name or row["file"] != name + ".pt" or row["validation_per_class"] != 5
                or row["total_annotation_labels"] != DATASETS[row["target"]][2] * (row["support_per_class"] + 5)
                or row["graph_identity"] != value["observations"][row["target"]]["graph_identity"]):
            raise ValueError("Context role, budget, observation or overlay path differs")
        peers = [r for r in rows if (r["target"], r["support_seed"]) == (row["target"], row["support_seed"])]
        if any(any(r[k] != row[k] for k in ("family_hash", "query_indices_hash", "shape"))
               or r["phase_hashes"]["validation"] == "" for r in peers):
            raise ValueError("Fixed query family/geometry across K required")
    check_files(value["protected_files"])
    check_files({str(root / r["file"]): r["sha256"] for r in rows})
    for row in rows:
        overlay = torch.load(root / row["file"], map_location="cpu", weights_only=True)
        validate_overlay(overlay, row["target"], row["support_per_class"], row["shape"])
        if overlay_identity(overlay) != row["overlay_identity"]:
            raise ValueError("Sanitized overlay identity differs")
        overlays[row["name"]] = overlay
    for target, seed in itertools.product(NAMES, SEEDS):
        group = [overlays[context_name(target, seed, k)] for k in BUDGETS]
        for left, right in zip(group, group[1:]):
            if (any(not torch.equal(left[k], right[k]) for k in ("val_indices", "val_labels"))
                    or not set(zip(left["train_indices"].tolist(), left["train_labels"].tolist()))
                    <= set(zip(right["train_indices"].tolist(), right["train_labels"].tolist()))):
                raise ValueError("Nested support sets and identical V5 across K required")


def load_context(manifest_path, manifest_sha256, *, target, support_seed, per_class, model_seed):
    name = context_name(target, support_seed, per_class)
    if type(model_seed) is not int or model_seed not in SEEDS:
        raise ValueError("Model seed must be explicitly separate from support seed")
    manifest_path = Path(manifest_path).resolve(strict=True)
    if file_hash(manifest_path) != manifest_sha256:
        raise ValueError("Pinned support context manifest required")
    value = read(manifest_path)
    validate_manifest(value, manifest_path.parent)
    unchanged_producer(value["code"], code_snapshot())
    row = next(r for r in value["contexts"] if r["name"] == name)
    observation = value["observations"][target]
    pins = {**value["protected_files"], str(manifest_path): manifest_sha256,
            **{str(manifest_path.parent / r["file"]): r["sha256"] for r in value["contexts"]}}
    item = torch.load(observation["input"]["observation"], map_location="cpu", weights_only=True)
    record = read(observation["input"]["record"])
    if (set(item) != OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
            or item["name"] != target or item["class_num"] != DATASETS[target][2]
            or [item["height"], item["width"]] != row["shape"]
            or _item_fingerprint(item) != record["prepared_fields"]
            or digest(record["prepared_fields"]) != row["graph_identity"]):
        raise ValueError("Frozen shared unlabeled observation changed")
    observation_id({k: item[k] for k in OBS_KEYS})
    overlay = torch.load(manifest_path.parent / row["file"], map_location="cpu", weights_only=True)
    validate_overlay(overlay, target, per_class, row["shape"])
    if overlay_identity(overlay) != row["overlay_identity"]:
        raise ValueError("Sanitized overlay identity differs")
    item.update(overlay)
    provenance = {"policy": POLICY, "context": name, "manifest_sha256": manifest_sha256,
        "support_seed": support_seed, "support_per_class": per_class, "validation_per_class": 5,
        "model_seed": model_seed, "head_seed": model_seed + 1000,
        "total_annotation_labels": row["total_annotation_labels"], "family_hash": row["family_hash"],
        "phase_hashes": row["phase_hashes"], "graph_identity": row["graph_identity"],
        "full_target_scene_transductive": True, "query_labels_accessed": False,
        "execution_enabled": False, "optimizer_updates": 0}
    check_files(pins)
    return {target: item}, provenance, lambda: check_files(pins)

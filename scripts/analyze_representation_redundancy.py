"""Analyze already frozen train/validation features; never extract a model or read labels."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from evaluation.representation_redundancy import aligned_summary
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, require_frozen_code, write_json


def _hash(value, size=64):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(size) + "}", value):
        raise ValueError("Explicit lowercase hash identity required")
    return value


def analyze(manifest, expected_sha256, out):
    path = Path(manifest).resolve(strict=True)
    if file_hash(path) != _hash(expected_sha256):
        raise ValueError("Feature manifest hash differs")
    entry = json.loads(path.read_text())
    keys = {"schema_version", "split", "stage", "feature_file", "checkpoint", "observation_sha256", "model_commit", "selection"}
    if (set(entry) != keys or type(entry["schema_version"]) is not int or entry["schema_version"] != 1
            or entry["split"] not in ("train", "validation") or entry["stage"] not in ("stage1", "stage2")
            or set(entry["selection"]) != {"maximum_regions", "seed", "variance_floor"}):
        raise ValueError("Exact frozen train/validation feature manifest required; query refused")
    _hash(entry["observation_sha256"]); _hash(entry["model_commit"], 40)
    files = {str(path): expected_sha256}; resolved = {}
    for key in ("feature_file", "checkpoint"):
        value = entry[key]
        if set(value) != {"path", "sha256"} or not Path(value["path"]).is_absolute():
            raise ValueError("Explicit absolute frozen input/hash pair required")
        p = Path(value["path"]).resolve(strict=True); expected = _hash(value["sha256"])
        if str(p) in files or file_hash(p) != expected:
            raise ValueError("Frozen file collision or hash differs")
        files[str(p)] = expected
        resolved[key] = str(p)
    target = Path(out).resolve()
    if target.exists() or target.is_relative_to(ROOT):
        raise ValueError("New output outside checkout required")
    # Checkpoint is hashed only. No checkpoint deserialization or model forward.
    features = torch.load(resolved["feature_file"], map_location="cpu", weights_only=True)
    if not isinstance(features, dict) or set(features) != {"cnn", "graph", "Q"}:
        raise ValueError("Exact label-free CNN/Graph/Q tensor mapping required")
    width = 128 if entry["stage"] == "stage1" else 64
    if any(not torch.is_tensor(features[k]) or features[k].ndim != 2 or features[k].shape[1] != width for k in ("cnn", "graph")):
        raise ValueError("Declared original stage channel contract differs")
    summary = aligned_summary(features["cnn"], features["graph"], features["Q"], **entry["selection"])
    if any(file_hash(p) != sha for p, sha in files.items()):
        raise ValueError("Frozen diagnostic input changed")
    result = {"schema_version": 1, "protocol": "e0_region_redundancy_v1", "status": "computed",
        "manifest": entry, "resolved_input_paths": resolved, "protected_files": files, "files_verified_unchanged": len(files),
        "summary": summary, "checkpoint_deserialized": False, "model_forwards": 0, "optimizer_updates": 0,
        "query_labels_accessed": False, "extraction_provenance_verified": False,
        "limit": "Hash identities do not prove extraction or label-access provenance; accepted freeze/context binding remains separate"}
    target.mkdir(parents=True)
    write_json(target / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "manifest", "manifest-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux diagnostics checkout required")
    require_frozen_code(code)
    result = analyze(args.manifest, args.manifest_sha256, args.out)
    require_frozen_code(code)
    print(json.dumps({"out": args.out, "sampled_regions": result["summary"]["sampled_regions"]}, allow_nan=False))


if __name__ == "__main__":
    main()

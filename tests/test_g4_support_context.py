"""Independent support/K binding, existing-loader equivalence and access barriers."""
import copy
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import g4_support_context as module
from g4_support_context import (INPUT_POLICY, LABEL_KEYS, LABEL_PRODUCER, SEEDS, context_name, grid,
    load_context, overlay_identity, prepare_contexts, read, validate_manifest)
from joint_baseline import NAMES, _item_fingerprint
from loadData.dataset_registry import DATASETS, file_hash
from loadData.target_label_protocol import BUDGETS, make_target_family, phase_view
from research_runtime import code_snapshot, tensor_hash, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS
from target_data_context import load_target_context

torch.set_num_threads(1)


def fixture(root):
    root.mkdir()
    observations, families, rows, audited = {}, [], [], []
    for target in NAMES:
        folder = root / target
        folder.mkdir()
        count, side, bands, regions = DATASETS[target][2], 20, 7, 6
        item = {"name": target, "height": side, "width": side, "bands": bands, "class_num": count,
            "superpixel_scale": 50, "Q": torch.nn.functional.one_hot(torch.arange(side * side) % regions, regions).float().to_sparse().coalesce(),
            "A": torch.eye(regions), "net_input": torch.rand(side, side, bands + 2, generator=torch.Generator().manual_seed(64)),
            "wavelengths": torch.linspace(450, 950, bands), "fwhm": None}
        obs, record = folder / "observation.pt", folder / "record.json"
        torch.save(item, obs)
        write_json(record, {"policy": GRAPH_POLICY, "target": target, "graph_fit_labels": 0,
            "observation_sha256": file_hash(obs), "prepared_fields": _item_fingerprint(item)})
        observations[target] = {"observation": str(obs), "observation_sha256": file_hash(obs),
            "record": str(record), "record_sha256": file_hash(record)}
        labels = (np.arange(side * side).reshape(side, side) % count + 1).astype(np.int64)
        for seed in SEEDS:
            family = make_target_family(labels, target, seed)
            path = folder / ("family-" + str(seed) + ".json")
            write_json(path, family)
            families.append({"target": target, "support_seed": seed, "path": str(path), "sha256": file_hash(path)})
            for k in BUDGETS:
                release = phase_view(family, k, "adapt", expected_hash=family["family_hash"])
                rows.append({"target": target, "support_seed": seed, "support_per_class": k,
                    "family_hash": family["family_hash"], "query_indices_hash": release["query_indices_hash"],
                    "training_labels": count * k, "validation_labels": count * 5,
                    "total_annotation_labels": count * (k + 5), "query_pixels": len(family["query_indices"])})
                audited.append({"target": target, "support_seed": seed, "k": k,
                    "training_labels": count * k, "validation_labels": count * 5, "query_pixels": len(family["query_indices"])})
    common = {"status": "passed", "source": {"commit": LABEL_PRODUCER, "dirty": False},
        "optimizer_updates": 0, "model_test_scoring": False}
    receipt, audit = root / "accepted.json", root / "audited.json"
    write_json(receipt, {**common, "rows": rows, "run_views_have_query_labels": False, "target_graph_built": False})
    write_json(audit, {**common, "rows": audited, "release_checks": 54})
    spec = {"policy": INPUT_POLICY, "label_receipt": {"path": str(receipt), "sha256": file_hash(receipt)},
        "independent_receipt": {"path": str(audit), "sha256": file_hash(audit)},
        "observations": observations, "families": families}
    path = root / "inputs.json"
    write_json(path, spec)
    return path, spec


def prepared(base):
    path, spec = fixture(base / "inputs")
    out = base / "contexts"
    value = prepare_contexts(path, file_hash(path), out, code_snapshot())
    return path, spec, out, value


class SupportContextTests(unittest.TestCase):
    def test_all27_match_existing_contexts_and_separate_model_support_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec, root, value = prepared(Path(temp))
            manifest = root / "manifest.json"
            self.assertEqual(len(value["contexts"]), 27)
            self.assertEqual(len(value["protected_files"]), 18)
            self.assertEqual(len(list(root.glob("*.pt"))), 27)
            for target, support_seed, k in grid():
                family = next(v for v in spec["families"] if (v["target"], v["support_seed"]) == (target, support_seed))
                source = spec["observations"][target]
                old, provenance, guard = load_target_context(target, source["observation"], source["record"], source["record_sha256"],
                    family["path"], family["sha256"], per_class=k, support_seed=support_seed)
                new, bound, finish = load_context(manifest, file_hash(manifest), target=target,
                    support_seed=support_seed, per_class=k, model_seed=2048)
                self.assertEqual(bound["support_seed"], support_seed)
                self.assertEqual((bound["model_seed"], bound["head_seed"]), (2048, 3048))
                self.assertEqual(bound["phase_hashes"], provenance["phase_hashes"])
                for key in LABEL_KEYS | OBS_KEYS:
                    self.assertEqual(tensor_hash(new[target][key]) if torch.is_tensor(new[target][key]) else new[target][key],
                        tensor_hash(old[target][key]) if torch.is_tensor(old[target][key]) else old[target][key])
                self.assertFalse(any("query" in key or "test" in key for key in new[target]))
                guard(); finish()

    def test_consumer_never_deserializes_curator_family_and_overlays_remain_tiny(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec, root, value = prepared(Path(temp))
            families = {v["path"] for v in spec["families"]}
            real_read = module.read
            def no_family(path):
                self.assertNotIn(str(path), families)
                return real_read(path)
            with patch("g4_support_context.read", side_effect=no_family):
                items, provenance, guard = load_context(root / "manifest.json", file_hash(root / "manifest.json"),
                    target="xuzhou", support_seed=512, per_class=1, model_seed=64)
                guard()
            self.assertEqual((len(items["xuzhou"]["train_labels"]), len(items["xuzhou"]["val_labels"])), (9, 45))
            self.assertEqual(provenance["total_annotation_labels"], 54)
            self.assertLess(sum(p.stat().st_size for p in root.glob("*.pt")), 250000)

    def test_manifest_rejects_partial_duplicate_execution_budget_paths_and_seed_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, root, value = prepared(Path(temp))
            mutations = (lambda v: v["contexts"].pop(), lambda v: v["contexts"].__setitem__(-1, v["contexts"][0]),
                lambda v: v.__setitem__("execution_enabled", True), lambda v: v.__setitem__("optimizer_updates", 1),
                lambda v: v.__setitem__("query_labels_accessed", True), lambda v: v["contexts"][0].__setitem__("file", "../escape.pt"),
                lambda v: v["contexts"][0].__setitem__("total_annotation_labels", 1),
                lambda v: v["contexts"][0].__setitem__("family_hash", "changed"))
            for mutate in mutations:
                bad = copy.deepcopy(value); mutate(bad)
                with self.assertRaises(ValueError): validate_manifest(bad, root)
            for args in (("xuzhou", True, 1), ("xuzhou", 1024, 1), ("xuzhou", 64, 2)):
                with self.assertRaises(ValueError): context_name(*args)
            with self.assertRaises(ValueError): load_context(root / "manifest.json", file_hash(root / "manifest.json"),
                target="xuzhou", support_seed=64, per_class=1, model_seed=1024)

    def test_changed_overlay_forbidden_tensor_and_validation_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, root, value = prepared(Path(temp))
            row = next(r for r in value["contexts"] if r["name"] == context_name("xuzhou", 64, 1))
            path = root / row["file"]
            overlay = torch.load(path, weights_only=True)
            changed = {**overlay, "query_labels": torch.tensor([1])}
            torch.save(changed, path)
            with self.assertRaises(ValueError): validate_manifest(value, root)
            row["sha256"] = file_hash(path)
            with self.assertRaisesRegex(ValueError, "Only sanitized"): validate_manifest(value, root)
            changed = copy.deepcopy(overlay)
            # Preserve balanced V5 counts while breaking the shared validation identity.
            changed["val_labels"] = (changed["val_labels"] + 1) % 9
            torch.save(changed, path)
            row.update(sha256=file_hash(path), overlay_identity=overlay_identity(changed))
            with self.assertRaisesRegex(ValueError, "identical V5"): validate_manifest(value, root)

    def test_wrong_accepted_family_or_graph_and_unsafe_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path, spec = fixture(base / "inputs")
            for out in (base / "inputs" / "nested", base, base / "inputs"):
                with self.assertRaises(ValueError): prepare_contexts(path, file_hash(path), out, code_snapshot())
            value = read(spec["label_receipt"]["path"])
            value["rows"][0]["family_hash"] = "changed"
            write_json(spec["label_receipt"]["path"], value)
            spec["label_receipt"]["sha256"] = file_hash(spec["label_receipt"]["path"])
            write_json(path, spec)
            with self.assertRaisesRegex(ValueError, "accepted independent"): prepare_contexts(path, file_hash(path), base / "bad", code_snapshot())
            self.assertFalse((base / "bad").exists())

    def test_guard_and_producer_end_check_preserve_inputs_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path, spec, root, value = prepared(base)
            with self.assertRaises(ValueError): prepare_contexts(path, file_hash(path), root, code_snapshot())
            _, _, guard = load_context(root / "manifest.json", file_hash(root / "manifest.json"),
                target="xuzhou", support_seed=2048, per_class=5, model_seed=512)
            family = Path(spec["families"][0]["path"])
            family.write_bytes(family.read_bytes() + b" ")
            with self.assertRaises(ValueError): guard()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); path, spec = fixture(base / "inputs")
            def mutate(code):
                target = Path(spec["families"][0]["path"])
                target.write_bytes(target.read_bytes() + b" ")
            with patch("g4_support_context.require_frozen_code", side_effect=mutate):
                with self.assertRaises(ValueError): prepare_contexts(path, file_hash(path), base / "bad", code_snapshot())
            self.assertFalse((base / "bad" / "manifest.json").exists())

    def test_cli_wrong_sha_rejected_before_any_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = str(Path(temp) / "not-created")
            args = [sys.executable, "scripts/prepare_g4_support_contexts.py", "--expected-commit", "0" * 40,
                "--inputs", missing, "--inputs-sha256", "x", "--out", missing]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Clean expected remote support-context producer required", result.stderr)
            self.assertFalse(Path(missing).exists())


if __name__ == "__main__":
    unittest.main()

"""Read-only cost measurement of both recovery-selected G4 geometries."""
import json
from pathlib import Path
import tempfile
import unittest

from compression.g4_cpu_cost import benchmark_recovery_pair
from compression.g4_recovery_short import run_recovery_pair
from joint_training import _tree_fingerprint
from loadData.dataset_registry import file_hash
from target_initialization import METHODS, initialize_target
from test_target_training import fixture


class G4CPUCostTests(unittest.TestCase):
    def test_paired_measurement_and_tampered_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (items, provenance, guard), _ = fixture(root)
            model, _ = initialize_target("xuzhou", METHODS[0], model_seed=64, head_seed=1064)
            source_sha = _tree_fingerprint(model.state_dict())
            recovery = root / "recovery"
            run_recovery_pair(model, items, provenance, recovery, removed=[2],
                              expected_reference_state=source_sha, guard=guard)
            sha = file_hash(recovery / "receipt.json")
            cost = root / "cost"
            result = benchmark_recovery_pair(model, items, recovery, cost, expected_receipt_sha256=sha,
                                             guard=guard, warmups=1, pairs=4)
            plan = json.loads((cost / "plan.json").read_text())
            self.assertEqual(plan["order"], [["dense", "compact"], ["compact", "dense"]] * 2)
            self.assertLess(plan["models"]["compact"]["state_tensor_bytes"], plan["models"]["dense"]["state_tensor_bytes"])
            self.assertEqual(len(result["samples"]), 4)
            self.assertTrue(all(row[role]["wall_ms"] > 0 for row in result["samples"] for role in ("dense", "compact")))
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertFalse(result["query_labels_accessed"])
            self.assertEqual(_tree_fingerprint(model.state_dict()), source_sha)
            path = recovery / "compact" / "selected-state.pt"
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Exact recovery-selected"):
                benchmark_recovery_pair(model, items, recovery, root / "rejected", expected_receipt_sha256=sha,
                                        guard=guard, warmups=1, pairs=4)
            self.assertFalse((root / "rejected").exists())

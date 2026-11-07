"""The paired short recovery runs the stock and narrowed G4 through the same target policy."""
import tempfile
import unittest
from pathlib import Path

from compression.g4_recovery_short import run_recovery_pair
from joint_training import _tree_fingerprint
from target_initialization import METHODS, initialize_target
from test_target_training import fixture


class G4RecoveryShortTests(unittest.TestCase):
    def test_paired_updates_validation_and_reference_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (items, provenance, guard), _ = fixture(root)
            model, _ = initialize_target("xuzhou", METHODS[0], model_seed=64, head_seed=1064)
            source_sha = _tree_fingerprint(model.state_dict())
            receipt = run_recovery_pair(model, items, provenance, root / "pair", removed=[2, 95, 126],
                                        expected_reference_state=source_sha, guard=guard)
            dense, compact = (receipt["arms"][role] for role in ("dense", "compact"))
            self.assertLess(compact["parameters"], dense["parameters"])
            for arm in (dense, compact):
                self.assertEqual(arm["optimizer_step_ids"], list(range(1, 13)))
                self.assertEqual(arm["cnn_adam_updates"], 12)
                self.assertEqual(arm["validation_steps"], [1, 4, 7, 10, 12])
                losses = arm["validation_fused_nll"]
                self.assertEqual(arm["selected_step"], arm["validation_steps"][losses.index(min(losses))])
                self.assertTrue(arm["exact_selected_reload"])
            self.assertEqual(dense["learning_rates"], compact["learning_rates"])
            self.assertFalse(receipt["query_labels_accessed"])
            self.assertEqual(_tree_fingerprint(model.state_dict()), source_sha)
            with self.assertRaisesRegex(ValueError, "Pinned dense"):
                run_recovery_pair(model, items, provenance, root / "bad", removed=[2],
                                  expected_reference_state="changed", guard=guard)
            self.assertFalse((root / "bad").exists())

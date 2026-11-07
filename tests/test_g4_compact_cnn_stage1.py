"""Exercise structural shortening through the real G4 forward entry."""
import unittest

import torch

from compression.g4_cnn_stage1 import ANCHOR, compact_from_passed_groups, compare_geometry
from model.research_config import digest
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from monitoring import LayerSpec
from monitoring.dependency_groups import HEADS, evaluate_groups
from monitoring.interventions import _snapshot
from monitoring.probes import FrozenProbe
from research_runtime import tensor_hash
import copy


def fixture(norm="domain_bn", **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        model = MultiDatasetVCASAHGCN({"scene": 2, "other": 3}, norm_type=norm, **kwargs).eval()
        inputs = {"x": torch.rand(6, 6, 8), "q": torch.nn.functional.one_hot(torch.arange(36) % 4).float().to_sparse(),
                  "a": torch.eye(4).to_sparse()}
    context = dict(run_id="geometry", commit="commit", checkpoint_hash="weights", data_hash="data",
                   split_hash="split", dataset="scene", partition="val", step=9, sample_id="val", condition="fixed")
    probe = FrozenProbe(context, inputs, {"cnn1": (LayerSpec(ANCHOR, "NCHW", .01, "stage1", "pixel"), 0)},
                        {"cnn1": {"all": torch.ones(1, 6, 6, dtype=torch.bool)}}, threshold_source="fixed",
                        allowed_datasets=["scene"])
    return model, inputs, probe


def forward(model, values):
    result = model("scene", values["x"], values["q"], values["a"])
    return {head: row[:2] for head, row in zip(HEADS, result)}


class CompactG4Tests(unittest.TestCase):
    def test_domain_and_global_bn_full_g4_mask_vs_shrink(self):
        for norm in ("domain_bn", "batch_norm"):
            with self.subTest(norm=norm):
                model, inputs, probe = fixture(norm)
                before = _snapshot(model)
                evidence = compare_geometry(model, forward, inputs, probe, [2, 95, 126])
                self.assertEqual(len(evidence["retained"]), 125)
                self.assertEqual(evidence["decision_changes_by_head"], dict.fromkeys(HEADS, 0))
                self.assertEqual(_snapshot(model), before)
                self.assertLess(evidence["after_parameters"], evidence["before_parameters"])
                expected_axes = 12 if norm == "domain_bn" else 8
                self.assertEqual(len(evidence["affected_axes"]), expected_axes)

    def test_real_g4_shape_from_a_passing_synthetic_trial(self):
        model, inputs, probe = fixture()
        labels = torch.tensor([1, 2])
        selection = {"layer": ANCHOR, "call_index": 0, "unit": "pixel"}
        candidate = {"policy": "mon_sustained_train_candidates_v1", "automatic_pruning": False,
                     "plan_sha256": digest(selection), "identity": probe.manifest["context"],
                     "samples": [{"step": 9, "checkpoint_hash": "weights"}], "channels": 128,
                     "candidate_channel_indices": [2]}
        plan = {"candidate_sha256": digest(candidate), "probe_identity": probe.identity,
                "reference_state": _snapshot(model)["state"], "labels_sha256": tensor_hash(labels),
                "groups": [{"name": "feature2", "candidate_index": 2, "sites": [{"observation": "cnn1", "channel": 2}]}],
                "combinations": [], "class_ids": [1, 2], "train_support": {"1": 2, "2": 2},
                "small_class_max_train_support": 0, "max_drop": {"oa": 1., "aa": 1., "recall": 1.}}
        receipt = evaluate_groups(model, lambda: copy.deepcopy(model), forward, inputs, probe, labels, candidate,
                                  selection, plan, expected_plan_sha256=digest(plan))
        compact, evidence = compact_from_passed_groups(model, probe, plan, receipt, ["feature2"],
                                                        expected_receipt_sha256=digest(receipt))
        self.assertEqual(compact.backbone.CNN_Branch[0].depth_conv.groups, 127)
        self.assertEqual(compact.backbone.CNN_Branch[1].point_conv.in_channels, 127)
        self.assertEqual(evidence["axes"][0]["old_to_new"][2], None)
        with self.assertRaisesRegex(ValueError, "own checkpoint schema"):
            compact.export_config()

    def test_coupled_and_group_normalized_topologies_are_rejected(self):
        model, inputs, probe = fixture("group_norm")
        with self.assertRaisesRegex(ValueError, "channel-independent"):
            compare_geometry(model, forward, inputs, probe, [2])
        model, inputs, probe = fixture(interaction_mode="add")
        with self.assertRaisesRegex(ValueError, "no intermediate interaction"):
            compare_geometry(model, forward, inputs, probe, [2])


if __name__ == "__main__":
    unittest.main()

"""Exercise explicit fan-out mappings and validation safeguards with real forwards."""
import copy
import unittest

import torch

from monitoring import LayerSpec
from monitoring.dependency_groups import evaluate_groups
from monitoring.interventions import _snapshot
from monitoring.probes import FrozenProbe
from model.research_config import digest
from research_runtime import tensor_hash


class TwoPaths(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Identity()
        self.bypass = torch.nn.Identity()

    def forward(self, x):
        cnn, graph = self.features(x), self.bypass(x)
        scores = {}
        for head, value in (("fused", cnn + graph), ("graph", graph), ("cnn", cnn)):
            margin = value.sum(1) - .5
            scores[head] = torch.stack((margin, -margin), dim=1)
        return scores


def fixture(partition="val"):
    model = TwoPaths()
    inputs = {"x": torch.tensor([[1., 1.], [-1., -1.]])}
    labels = torch.tensor([1, 2])
    common = dict(run_id="fixture", commit="commit", data_hash="data", split_hash="split", dataset="scene")
    context = {**common, "condition": "held", "partition": partition, "checkpoint_hash": "checkpoint",
               "step": 9, "sample_id": "fixed"}
    calls = {k: (LayerSpec(k, "NC", .1, "post", "pixel"), 0) for k in ("features", "bypass")}
    buckets = {k: {"all": torch.ones(2, dtype=torch.bool)} for k in calls}
    probe = FrozenProbe(context, inputs, calls, buckets, threshold_source="fixed", allowed_datasets=["scene"])
    selection = {"layer": "features", "unit": "pixel", "call_index": 0}
    candidate = {"policy": "mon_sustained_train_candidates_v1", "automatic_pruning": False,
                 "plan_sha256": digest(selection), "identity": {**common, "condition": "train"},
                 "samples": [{"step": 9, "checkpoint_hash": "checkpoint"}], "channels": 2,
                 "candidate_channel_indices": [0, 1]}
    groups = [{"name": f"feature{i}", "candidate_index": i,
               "sites": [{"observation": k, "channel": i} for k in calls]} for i in (0, 1)]
    plan = {"candidate_sha256": digest(candidate), "probe_identity": probe.identity,
            "reference_state": _snapshot(model)["state"], "labels_sha256": tensor_hash(labels), "groups": groups,
            "combinations": [["feature0", "feature1"]], "class_ids": [1, 2],
            "train_support": {"1": 2, "2": 100}, "small_class_max_train_support": 10,
            "max_drop": {"oa": 0., "aa": 0., "recall": 0.}}
    return model, inputs, labels, probe, selection, candidate, plan


def evaluate(f, factory=None, expected=None):
    model, inputs, labels, probe, selection, candidate, plan = f
    return evaluate_groups(model, factory or (lambda: copy.deepcopy(model)), lambda m, x: m(x["x"]),
                           inputs, probe, labels, candidate, selection, plan,
                           expected_plan_sha256=digest(plan) if expected is None else expected)


class DependencyGroupsTests(unittest.TestCase):
    def test_two_single_groups_pass_while_combination_is_rejected(self):
        f = fixture()
        initial = _snapshot(f[0])
        result = evaluate(f)
        self.assertEqual([row["passed"] for row in result["trials"]], [True, True, False])
        combined = result["trials"][-1]
        self.assertEqual(combined["protection"]["failed_recall_classes"], [1])
        self.assertEqual(combined["protection"]["new_small_class_errors"], {"1": 1})
        self.assertEqual(set(combined["metrics"]), {"fused", "graph", "cnn"})
        self.assertEqual(combined["execution"]["interventions"], [
            {"observation": "features", "channels": [0, 1]}, {"observation": "bypass", "channels": [0, 1]}])
        self.assertEqual(_snapshot(f[0]), initial)
        self.assertEqual(result["optimizer_updates"], 0)

    def test_immutable_plan_and_candidate_mapping_checked_before_forward(self):
        f = fixture()
        def forbidden():
            self.fail("Invalid mapping reached model creation")
        with self.assertRaisesRegex(ValueError, "Exact fixed"):
            evaluate(f, forbidden, expected="changed")
        f[-1]["groups"][0]["sites"].pop(0)
        with self.assertRaisesRegex(ValueError, "exact screened anchor"):
            evaluate(f, forbidden)
        f = fixture()
        f[-1]["groups"][0]["candidate_index"] = 2
        with self.assertRaisesRegex(ValueError, "screened internal channel"):
            evaluate(f, forbidden)

    def test_changed_checkpoint_or_label_identity_is_rejected(self):
        f = fixture()
        f[-2]["samples"][-1]["checkpoint_hash"] = "another"
        f[-1]["candidate_sha256"] = digest(f[-2])
        with self.assertRaisesRegex(ValueError, "screened checkpoint"):
            evaluate(f)
        f = fixture()
        f[2][0] = 2
        with self.assertRaisesRegex(ValueError, "Pinned one-dimensional"):
            evaluate(f)
        with self.assertRaisesRegex(ValueError, "validation probe anchor"):
            evaluate(fixture("train"))


if __name__ == "__main__":
    unittest.main()

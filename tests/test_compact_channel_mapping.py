"""Physical tensor shrink matches reversible activation zeroing on a bounded topology."""
import copy
import unittest

import torch

from compression.channel_mapping import compact_state
from model.research_config import digest
from monitoring import LayerSpec
from monitoring.dependency_groups import HEADS, evaluate_groups
from monitoring.interventions import ChannelIntervention, _snapshot, isolated_probe
from monitoring.probes import FrozenProbe
from research_runtime import tensor_hash
from test_monitoring_dependency_groups import evaluate, fixture


class FanOut(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stem = torch.nn.Linear(2, width)
        self.norm = torch.nn.BatchNorm1d(width)
        self.features = torch.nn.Identity()
        self.heads = torch.nn.ModuleDict({head: torch.nn.Linear(width, 2) for head in HEADS})

    def forward(self, x):
        features = self.features(self.norm(self.stem(x)))
        return {k: layer(features) for k, layer in self.heads.items()}


def accepted_fixture():
    model = FanOut(3)
    with torch.no_grad():
        model.stem.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]))
        model.stem.bias.zero_()
        for layer in model.heads.values():
            layer.weight.copy_(torch.tensor([[1., .25, .5], [-1., -.25, -.5]]))
            layer.bias.zero_()
    target = FanOut(2)
    inputs, labels = {"x": torch.tensor([[1., 1.], [-1., 1.]])}, torch.tensor([1, 2])
    context = {"run_id": "shape", "commit": "commit", "data_hash": "data", "split_hash": "split", "dataset": "fixture",
               "condition": "fixed", "partition": "val", "step": 9, "checkpoint_hash": "weights", "sample_id": "val"}
    spec = LayerSpec("features", "NC", .1, "after_norm", "pixel")
    probe = FrozenProbe(context, inputs, {"anchor": (spec, 0)}, {"anchor": {"all": torch.ones(2, dtype=torch.bool)}},
                        threshold_source="fixed", allowed_datasets=["fixture"])
    selection = {"layer": "features", "unit": "pixel", "call_index": 0}
    candidate = {"policy": "mon_sustained_train_candidates_v1", "automatic_pruning": False,
                 "plan_sha256": digest(selection), "identity": context,
                 "samples": [{"step": 9, "checkpoint_hash": "weights"}], "channels": 3,
                 "candidate_channel_indices": [1]}
    site = {"observation": "anchor", "channel": 1}
    plan = {"candidate_sha256": digest(candidate), "probe_identity": probe.identity,
            "reference_state": _snapshot(model)["state"], "labels_sha256": tensor_hash(labels),
            "groups": [{"name": "internal1", "candidate_index": 1, "sites": [site]}], "combinations": [],
            "class_ids": [1, 2], "train_support": {"1": 1, "2": 20}, "small_class_max_train_support": 10,
            "max_drop": {"oa": 0., "aa": 0., "recall": 0.}}
    run = lambda m, x: m(x["x"])
    receipt = evaluate_groups(model, lambda: copy.deepcopy(model), run, inputs, probe, labels, candidate, selection,
                              plan, expected_plan_sha256=digest(plan))
    affected = [(k, 0) for k in ("stem.weight", "stem.bias", "norm.weight", "norm.bias",
                                  "norm.running_mean", "norm.running_var")]
    affected += [(f"heads.{h}.weight", 1) for h in HEADS]
    declaration = {"receipt_sha256": digest(receipt), "groups": ["internal1"],
                   "axes": [{"tensor": name, "axis": axis, "remove": [{**site, "group": "internal1", "index": 1}]}
                            for name, axis in affected]}
    return model, target, plan, receipt, declaration, inputs, probe, run


class CompactChannelMappingTests(unittest.TestCase):
    def test_true_smaller_fanout_matches_reversible_control(self):
        model, target, plan, receipt, declaration, inputs, probe, run = accepted_fixture()
        original, fresh = _snapshot(model), _snapshot(target)
        state, evidence = compact_state(model, target, plan, receipt, declaration, expected_sha256=digest(declaration))
        self.assertEqual(original, _snapshot(model))
        self.assertEqual(fresh, _snapshot(target))
        self.assertEqual(evidence["axes"][0]["old_to_new"], [0, None, 1])
        target.load_state_dict(state, strict=True)
        target.eval()
        ref = isolated_probe(model, lambda: copy.deepcopy(model), run, inputs, probe,
                             interventions=(ChannelIntervention("anchor", (1,)),))
        for head, outputs in target(inputs["x"]).items():
            torch.testing.assert_close(outputs, ref["outputs"][head], rtol=1e-6, atol=1e-7)
        self.assertLess(sum(p.numel() for p in target.parameters()), sum(p.numel() for p in model.parameters()))
        self.assertFalse(evidence["optimizer_state_transferred"])

    def test_missing_downstream_mapping_and_tampered_identity_rejected(self):
        model, target, plan, receipt, declaration, *_ = accepted_fixture()
        declaration["axes"].pop()
        with self.assertRaisesRegex(ValueError, "complete mapped state schema"):
            compact_state(model, target, plan, receipt, declaration, expected_sha256=digest(declaration))
        with self.assertRaisesRegex(ValueError, "Exact frozen"):
            compact_state(model, target, plan, receipt, declaration, expected_sha256="other")

    def test_individually_accepted_groups_do_not_authorize_failed_combination(self):
        f = fixture()
        receipt = evaluate(f)
        declaration = {"receipt_sha256": digest(receipt), "groups": ["feature0", "feature1"]}
        with self.assertRaisesRegex(ValueError, "exact removal combination"):
            compact_state(f[0], None, f[-1], receipt, declaration, expected_sha256=digest(declaration))


if __name__ == "__main__":
    unittest.main()

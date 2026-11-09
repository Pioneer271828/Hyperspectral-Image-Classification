"""Use actual MON events and a fixed protection probe to screen internal channels."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from monitoring import LayerSpec, ReadOnlyMonitor
from monitoring.probes import FrozenProbe
from monitoring.updates import ParameterSpec, UpdateMonitor
from monitoring.windows import main, sustained_candidates
from loadData.dataset_registry import file_hash
from research_runtime import write_json
from target_initialization import state_id


def collect():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0., 1.], [0., 0.], [.2, 0.]]))
    baseline = copy.deepcopy(model)
    spec = LayerSpec("0", "NC", .1, "post_projection", "latent_position")
    opts = [torch.optim.SGD(m.parameters(), lr=.01) for m in (model, baseline)]
    activations, updates = [], []
    activation = ReadOnlyMonitor(model, (spec,), activations.append)
    update = UpdateMonitor(model, opts[0], (ParameterSpec("0.weight", 0, spec.unit),), updates.append)
    data = torch.tensor([[0., 1.], [0., 2.]])
    probe = None
    for step in (1, 3, 5, 7):
        context = {"run_id": "mon-test", "commit": "synthetic", "checkpoint_hash": state_id(model),
                   "data_hash": "data", "split_hash": "split", "dataset": "fixture", "partition": "train",
                   "step": step, "sample_id": str(step), "condition": "train-cohort"}
        if step == 7:
            points = torch.tensor([[2., 0.], [0., 1.]])
            validation = {**context, "partition": "val", "sample_id": "fixed", "condition": "rare"}
            frozen = FrozenProbe(validation, {"x": points}, {"output": (spec, 0)},
                                 {"output": {"rare": torch.tensor([True, False]),
                                             "common": torch.tensor([False, True])}},
                                 threshold_source="fixed", allowed_datasets=["fixture"])
            probe = frozen.observe({"x": points}, {"output": model(points)}, context=validation)
        opts[0].zero_grad()
        with activation.sample(context):
            output = model(data)
        with update.sample(context, contract="fp32_single_backward"):
            output[:, 0].sum().backward()
            opts[0].step()
        opts[1].zero_grad()
        baseline(data)[:, 0].sum().backward()
        opts[1].step()
    if state_id(model) != state_id(baseline):
        raise AssertionError("Sampling changed the original optimizer trajectory")
    plan = {"windows": [[1, 3], [5, 7]], "layer": "0", "call_index": 0, "unit": spec.unit,
            "channel_role": "internal_feature", "parameter": "0.weight", "axis": 0,
            "activation_epsilon": .1, "relative_epsilon": 1e-12,
            "limits": {"response_rate": .25, "weight_rms_before": .2, "gradient_rms": .1, "relative_update": .01},
            "threshold_source": "fixed", "probe_identity": probe["probe_identity"],
            "protection_buckets": ["common", "rare"], "min_probe_elements": 1, "protection_response": .5}
    return plan, [{"activation": a, "update": u} for a, u in zip(activations, updates)], probe


class SustainedCandidateTests(unittest.TestCase):
    def test_real_events_preserve_rare_condition_channel(self):
        plan, samples, probe = collect()
        result = sustained_candidates(plan, samples, probe)
        self.assertEqual(result["candidate_channel_indices"], [1])
        self.assertFalse(result["automatic_pruning"])
        self.assertEqual(result["condition_protection"]["rare"]["response_rate"][2], 1)
        self.assertEqual(result["windows"][0]["valid_elements_per_channel"], 4)

    def test_unequal_support_is_weighted_and_late_activity_excludes_candidate(self):
        plan, samples, probe = collect()
        samples[0]["activation"]["statistics"]["count_per_channel"] = 8
        samples[1]["activation"]["statistics"]["response_rate"][1] = 1
        result = sustained_candidates(plan, samples, probe)
        self.assertAlmostEqual(result["windows"][0]["metrics"]["response_rate"][1], .2)
        self.assertEqual(result["candidate_channel_indices"], [1])
        samples[3]["update"]["parameters"][0]["relative_update"][1] = .5
        self.assertEqual(sustained_candidates(plan, samples, probe)["candidate_channel_indices"], [])

    def test_mixed_cohort_frozen_gradient_and_empty_condition_fail(self):
        plan, samples, probe = collect()
        with self.assertRaisesRegex(ValueError, "exact declared samples"):
            sustained_candidates(plan, samples[:-1], probe)
        wrong = copy.deepcopy(samples)
        wrong[1]["activation"]["context"]["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "permitted partition"):
            sustained_candidates(plan, wrong, probe)
        wrong = copy.deepcopy(samples)
        wrong[1]["update"]["parameters"][0]["gradient_state"] = "frozen"
        with self.assertRaisesRegex(ValueError, "ineligible"):
            sustained_candidates(plan, wrong, probe)
        probe["records"][0]["statistics"]["count_per_channel"] = 0
        with self.assertRaisesRegex(ValueError, "support is insufficient"):
            sustained_candidates(plan, samples, probe)

    def test_cli_pins_input_and_refuses_output_overwrite(self):
        plan, samples, probe = collect()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "input.json", root / "out"
            write_json(source, {"plan": plan, "samples": samples, "probe": probe})
            args = ["--input", str(source), "--sha256", file_hash(source), "--out", str(out)]
            with self.assertRaisesRegex(ValueError, "Exact frozen"):
                main([*args[:3], "wrong", *args[4:]])
            self.assertFalse(out.exists())
            main(args)
            self.assertTrue((out / "candidates.json").is_file())
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()

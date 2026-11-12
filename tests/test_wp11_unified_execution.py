"""Integration checks against the frozen WP11 fixtures and production release."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from inference import wp11_execution as workflow
from inference.wp11_execution import MODES, validate_result
from research_runtime import code_snapshot, hash_file, write_json


def pin(path):
    return dict(path=str(path), sha256=hash_file(path))


class UnifiedWP11Tests(unittest.TestCase):
    def test_stale_result_and_unadmitted_routes_fail_before_data(self):
        torch.set_num_threads(1)
        code = code_snapshot()
        with TemporaryDirectory() as name:
            root = Path(name)
            old = root / "old.json"
            value = dict(policy=workflow.POLICY, commit=code["commit"], source_hash=code["source_hash"],
                         routing=None, timing=workflow.environment(), static_input="a"*64)
            write_json(old, value)
            with patch.object(workflow, "release_metadata", side_effect=AssertionError("premature model read")):
                for mode in ("adaptive", "random"):
                    with self.assertRaisesRegex(ValueError, "independent confirmation"):
                        workflow.execute(pin(old), code, mode, root / "unused")
                broken = root / "changed-timing.json"
                write_json(broken, dict(value, timing=dict(value["timing"], fs_cache="warm")))
                with self.assertRaisesRegex(ValueError, "Exact execution code, timing"):
                    workflow.execute(pin(broken), code, "light", root / "unused")
            receipt = dict(status="passed", commit=code["commit"], manifest=pin(old), mode="light",
                           input_hash=value["static_input"], environment=value["timing"])
            self.assertEqual(validate_result(value, pin(old), receipt, "light"), receipt)
            for field, replacement in (("input_hash", "b"*64), ("mode", "full_source"),
                                       ("environment", dict(value["timing"], scope="end_to_end"))):
                with self.assertRaisesRegex(ValueError, "different code, input, mode or timing"):
                    validate_result(value, pin(old), dict(receipt, **{field: replacement}), "light")

    @unittest.skipUnless(os.environ.get("WP11_A_SYNTHETIC"), "requires the already verified WP11 fixture outputs")
    def test_existing_9_and_22_class_chain(self):
        torch.set_num_threads(1)
        code = code_snapshot()
        fixtures = Path(os.environ["WP11_A_SYNTHETIC"])
        output = Path(os.environ["WP11_B_SYNTHETIC"])
        output.mkdir()
        affinity = os.sched_getaffinity(0)
        try:
            for classes in (9, 22):
                with self.subTest(classes=classes):
                    source = fixtures / str(classes)
                    expected = json.loads((source / "confirmed/receipt.json").read_text())
                    os.sched_setaffinity(0, set(expected["timing"]["affinity"]))
                    target = output / str(classes)
                    target.mkdir()
                    request = target / "request.json"
                    write_json(request, dict(release=pin(source / "frozen-models/release.json"), static_cost=None,
                               confirmed_routes=dict(registration=pin(source / "ready/confirmation-registration.json"),
                                                     receipt=pin(source / "confirmed/receipt.json"))))
                    manifest = workflow.build(pin(request), code, target / "registered")
                    identity = pin(target / "registered/manifest.json")
                    label_path = source / "confirmation/labels.json"
                    read_bytes = Path.read_bytes

                    def no_labels(path):
                        if path == label_path:
                            raise AssertionError("Unified execution must not reopen confirmation labels")
                        return read_bytes(path)

                    with patch.object(Path, "read_bytes", new=no_labels):
                        results = {mode: workflow.execute(identity, code, mode, target / mode,
                                                          export=mode == "full_matched") for mode in MODES}
                    output = torch.load(results["full_matched"]["prediction_package"]["path"], weights_only=True)
                    self.assertEqual(output["input_hash"], manifest["static_input"])
                    self.assertTrue(all(set(row["probabilities"]) == {"fused", "graph", "cnn"} for row in output["rows"]))
                    self.assertEqual(len(manifest["classes"]), classes)
                    self.assertEqual(results["light"]["instrumentation"]["graph_calls"], 0)
                    self.assertEqual(results["all_fallback"]["output_hashes"]["probabilities"],
                                     results["full_matched"]["output_hashes"]["fused"])
                    self.assertEqual(results["all_fallback"]["counts"]["light_calls"], 0)
                    for mode in ("adaptive", "random"):
                        self.assertEqual(results[mode]["output_hashes"]["routed"], expected["timing"]["output_hashes"][mode])
                        self.assertEqual(results[mode]["counts"], expected["timing"]["summary"][mode]["calls"])
                        self.assertGreater(results[mode]["fallback_light_already_run"], 0)
                        self.assertTrue(validate_result(manifest, identity, results[mode], mode))
                    self.assertEqual(results["adaptive"]["instrumentation"]["gate_calls"], 3)
                    self.assertEqual(results["random"]["instrumentation"]["gate_calls"], 0)
                    self.assertEqual(results["adaptive"]["counts"], results["random"]["counts"])
                    invalid = target / "changed-model.json"
                    write_json(invalid, dict(manifest, models=dict(manifest["models"], light=dict(
                                             manifest["models"]["light"], state="0"*64))))
                    with self.assertRaisesRegex(ValueError, "Static model state differs"):
                        workflow.execute(pin(invalid), code, "light", target / "denied")
        finally:
            os.sched_setaffinity(0, affinity)

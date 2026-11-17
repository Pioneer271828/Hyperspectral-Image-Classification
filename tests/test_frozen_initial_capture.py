"""Original initializer, authorization, and frozen observation regression."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation import frozen_initial_capture as bridge, frozen_representation_capture as old
from g4_matched_target_study import METHODS, common_fresh_initials, initialize
from g4_support_context import read
from research_runtime import code_snapshot
from test_target_training import fixture


class FrozenInitialCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        _, files = fixture(Path(temp.name))
        cls.obs, cls.record = files[0], read(files[1])
        cls.proof = common_fresh_initials("xuzhou", 64)
        _, initialization = initialize("xuzhou", 64, METHODS[0], "none", source_spec={})
        cls.case = {"target": "xuzhou", "seed": 64, "method": METHODS[0], "mode": "none",
                    "initialization": initialization, "initial_state_sha256": initialization["initial_state_sha256"],
                    "fresh_state_sha256": cls.proof["fresh_state_sha256"], "source_spec": {},
                    "target_context": {"query_labels_accessed": False}}

    def test_unchanged_source_and_original_network_capture(self):
        bridge.verify_source(code_snapshot())
        seed = torch.get_rng_state().clone()
        stages, proof = bridge.capture_case(self.case, self.proof, self.record, self.obs)
        self.assertEqual(proof["state_sha256"], self.case["initial_state_sha256"])
        self.assertEqual(proof["stage_calls"], dict.fromkeys(old.STAGES, 1))
        self.assertEqual(stages["stage1"]["cnn"].shape[1], 128)
        self.assertEqual(stages["stage2"]["graph"].shape[1], 64)
        self.assertTrue(torch.equal(seed, torch.get_rng_state()))

    def test_identity_rejects_head_seed_labels_and_terminal_model(self):
        with patch.object(bridge, "initialize", side_effect=AssertionError("Identity gate must run first")):
            for key in ("head_seed", "query", "state"):
                altered = deepcopy(self.case)
                if key == "head_seed":
                    altered["initialization"]["head_seed"] += 1
                elif key == "query":
                    altered["target_context"]["query_labels_accessed"] = True
                else:
                    altered["initial_state_sha256"] = "unregistered"
                with self.assertRaises(ValueError):
                    bridge.rebuild(altered, self.proof)
        modified, evidence = initialize("xuzhou", 64, METHODS[0], "none", source_spec={})
        with torch.no_grad():
            buffer = next(value for name, value in modified.named_buffers() if name.endswith("running_mean"))
            buffer.add_(1)
        with patch.object(bridge, "initialize", return_value=(modified, evidence)):
            with self.assertRaisesRegex(ValueError, "exact initial state"):
                bridge.rebuild(self.case, self.proof)

    def test_observation_and_rng_mutations_are_rejected(self):
        record = deepcopy(self.record)
        record["prepared_fields"]["Q"] = "foreign-graph"
        with patch.object(old, "capture", side_effect=AssertionError("Observation gate must run first")):
            with self.assertRaisesRegex(ValueError, "label-free observation"):
                bridge.capture_case(self.case, self.proof, record, self.obs)
        original = old.capture

        def change(*args):
            result = original(*args)
            torch.rand(1)
            return result

        before = torch.get_rng_state().clone()
        with patch.object(old, "capture", side_effect=change):
            with self.assertRaisesRegex(ValueError, "changed model or RNG"):
                bridge.capture_case(self.case, self.proof, self.record, self.obs)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_wrong_full_source_role_and_source_hash_rejected(self):
        case = deepcopy(self.case)
        case["method"] = METHODS[1]
        spec = {"source_path":"/weights", "source_weight_sha256":"weight", "source_sidecar_sha256":"sidecar"}
        for key in ("source_result", "source_launch", "source_acceptance", "source_independent_audit",
                    "source_protected_manifest"):
            spec[key], spec[key+"_sha256"] = "/"+key, key
        case["source_spec"] = spec
        case["initialization"]["source"] = {"target":"xuzhou", "seed":64, "mode":"none"}
        pins = {spec[k]: spec[k+"_sha256"] for k in spec if k not in ("source_path",) and not k.endswith("_sha256")}
        pins.update({"/weights":"weight", "/weights.json":"sidecar"})
        self.assertEqual(bridge.source_pins(case,pins), pins)
        for altered in ({"seed":512}, {"mode":"capacity_control"}, {"target":"paviaU"}):
            broken = deepcopy(case)
            broken["initialization"]["source"].update(altered)
            with self.assertRaises(ValueError):
                bridge.source_pins(broken,pins)
        with self.assertRaises(ValueError):
            bridge.source_pins(case, {**pins, "/weights":"tamper"})
        with self.assertRaisesRegex(ValueError, "original54"):
            bridge.common_proofs({"cases":[], "common_initials":[]})

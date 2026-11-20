"""Exact saved-tensor offload parity on a synthetic G4 optimizer phase."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from g4_light_full_target import prepare_case
from g4_light_memory_probe import backward_snapshot, main
from g4_light_target import ADAPTER_HEADS
from model.research_config import ResearchConfig
from test_g4_light_full_target import registration_fixture
from loadData.dataset_registry import file_hash


class LightMemoryProbeTests(unittest.TestCase):
    def test_cli_requires_explicit_isolation_strategy(self):
        args = ["--expected-commit", "reviewed", "--producer-checkout", "frozen-tree",
                "--registration", "frozen-registration", "--out", "new-output", "--gpu", "5"]
        code = {"commit": "reviewed", "source_hash": "reviewed-files"}
        with (patch("g4_light_memory_probe.checked_probe", return_value=code),
              patch("g4_light_memory_probe.profile") as legacy,
              patch("g4_light_memory_probe.launch_recompute") as isolated,
              patch("builtins.print")):
            main(args)
            self.assertEqual(legacy.call_args.args[0].strategy, "offload")
            isolated.assert_not_called()
            legacy.reset_mock()
            main([*args, "--strategy", "recompute", "--verify-source-only"])
            selected, source = isolated.call_args.args
            self.assertEqual(selected.strategy, "recompute")
            self.assertTrue(selected.verify_source_only)
            self.assertIs(source, code)
            legacy.assert_not_called()

    def test_original_and_offloaded_backward_are_exact(self):
        device = os.environ.get("LIGHT_MEMORY_TEST_DEVICE", "cpu")
        with TemporaryDirectory() as directory:
            registration, spec, code = registration_fixture(Path(directory))
            with patch("g4_light_full_target.REFERENCE_LAUNCH_SHA256", spec["reference_launch"]["sha256"]):
                items, _, runtime, guard = prepare_case(registration, file_hash(registration), code,
                                                       ("xuzhou", 64, ADAPTER_HEADS, "none"), device=device)
            runtime.directory.mkdir(parents=True)
            config = ResearchConfig.from_dict(runtime.settings["config"])
            model = runtime.build_model(config, {"xuzhou": items["xuzhou"]["class_num"]})
            item = {k: v.to(device) if torch.is_tensor(v) else v for k, v in items["xuzhou"].items()}
            rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device != "cpu" else None
            native = backward_snapshot(runtime, model, item, config, offload=False)
            torch.random.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            offloaded = backward_snapshot(runtime, model, item, config, offload=True)
            for field in ("losses", "output_sha256", "gradient_sha256", "state_sha256"):
                self.assertEqual(native[field], offloaded[field], field)
            self.assertTrue(offloaded["gradient_sha256"])
            guard()


if __name__ == "__main__":
    unittest.main()

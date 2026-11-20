"""Regression tests for the diagnostic's dispatch, CUDA identity and review."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import Mock, patch

import torch

import g4_light_probe_pair as probe
from g4_light_engineering_registration import pin
from research_runtime import write_json
from transfer.shared_gpu_lease import observed_cuda_device

GPU0 = "GPU-0c958696-3721-4009-628f-befb0ef9b867"
GPU3 = "GPU-2cbff259-d7db-32cf-b267-8f296e34b1d5"
DEVICE = {"slot": 0, "uuid": GPU0}


class ProbeDeviceBindingTests(unittest.TestCase):
    def test_dispatch_overrides_reversed_parent_mask_and_keeps_parent_unchanged(self):
        reversed_mask = f"{GPU3},{GPU0}"
        lock = Mock()
        lock.fileno.return_value = 9

        @contextmanager
        def fake_lease(uuid):
            self.assertEqual(uuid, GPU0)
            yield lock

        with (TemporaryDirectory() as directory,
              patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": reversed_mask, "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}),
              patch.object(probe, "dispatch_lease", fake_lease),
              patch.object(probe, "hardware_gate", return_value={}),
              patch.object(probe.subprocess, "run") as run):
            root = Path(directory)
            probe.dispatch_samples(DEVICE, 12288, root / "samples", root,
                                   lambda label, descriptor: [label, str(descriptor)])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], reversed_mask)
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "FASTEST_FIRST")
            self.assertEqual(run.call_count, len(probe.SAMPLES))
            for label, call in zip(probe.SAMPLES, run.call_args_list, strict=True):
                self.assertEqual(call.args, ([label, "9"],))
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU0)
                self.assertEqual(call.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertEqual(call.kwargs["pass_fds"], (9,))

    def test_cuda_ordinal_mismatch_rejected_even_with_expected_text_mask(self):
        props = SimpleNamespace(uuid=GPU3[4:], name="NVIDIA GeForce RTX 4090", total_memory=1)
        with (patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU0}),
              patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "device_count", return_value=1),
              patch.object(torch.cuda, "current_device", return_value=0),
              patch.object(torch.cuda, "get_device_properties", return_value=props),
              patch.object(torch.cuda, "mem_get_info") as memory,
              patch.object(probe, "hardware_gate", return_value={"name": props.name})):
            with self.assertRaisesRegex(ValueError, "Actual CUDA UUID .* does not match leased UUID"):
                probe.sample_device(DEVICE, 12288)
            memory.assert_not_called()

    def test_multi_device_and_missing_runtime_uuid_fail_closed(self):
        props = SimpleNamespace(name="NVIDIA GeForce RTX 4090", total_memory=1)
        with (patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU0}),
              patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "device_count", return_value=2) as count,
              patch.object(torch.cuda, "current_device", return_value=0),
              patch.object(torch.cuda, "get_device_properties", return_value=props) as properties):
            with self.assertRaisesRegex(ValueError, "exactly one visible"):
                observed_cuda_device(GPU0)
            properties.assert_not_called()
            count.return_value = 1
            with self.assertRaisesRegex(ValueError, "Explicit full physical GPU UUID required"):
                observed_cuda_device(GPU0)

    def test_review_rejects_legacy_or_forged_actual_uuid_before_tensor_read(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "receipt.json"
            report = {"sample": "offload", "admission": {"physical_uuid": GPU0,
                       "name": "NVIDIA GeForce RTX 4090"},
                      "raw": {"path": str(folder / "tensors.pt"), "sha256": "a" * 64}, **probe.LIMITS}
            with patch.object(torch, "load") as load:
                for execution in (None, {"physical_uuid": GPU3, "device": "cuda:0", "visible_devices": GPU0,
                                          "device_count": 1, "gpu_name": report["admission"]["name"],
                                          "total_memory_bytes": 1, "cuda_runtime": "12.8",
                                          "arithmetic": probe.ARITHMETIC}):
                    report["execution"] = execution
                    write_json(path, report)
                    with self.assertRaisesRegex(ValueError, "Receipt actual CUDA UUID"):
                        probe.load_sample(pin(path), "offload", DEVICE)
                load.assert_not_called()

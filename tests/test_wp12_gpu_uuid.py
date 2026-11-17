from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.wp12_resident_cost import admit_gpu, canonical_gpu_uuid, gpu_measurement_lease


class CostGPUIdentityTests(unittest.TestCase):
    def test_cuda_bare_uuid_matches_nvml_prefixed_uuid(self):
        raw = "ab4a0000-0000-0000-0000-00000000000a"
        physical = "GPU-" + raw
        self.assertEqual(canonical_gpu_uuid(raw.upper()), physical)
        self.assertEqual(canonical_gpu_uuid(physical), physical)
        with patch("evaluation.wp12_resident_cost.subprocess.check_output",
                   return_value=f"{physical}, 12288, 60\n"):
            self.assertEqual(admit_gpu({"gpu": {"uuid": raw}}),
                             {"free_mib": 12288, "utilization_percent": 60})
            with self.assertRaisesRegex(ValueError, "cannot be matched"):
                admit_gpu({"gpu": {"uuid": raw[:-1] + "b"}})

    def test_short_ids_and_mig_devices_are_rejected(self):
        for value in ("GPU-ab4a0000", "0", "MIG-ab4a0000-0000-0000-0000-00000000000a"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "full physical"):
                canonical_gpu_uuid(value)

    def test_shared_gpu_lease_blocks_cost_worker_and_releases_after_gate_failure(self):
        physical = "GPU-ab4a0000-0000-0000-0000-00000000000a"
        with TemporaryDirectory() as root, patch.dict("os.environ", {"HSI_GPU_LEASE_ROOT": root}):
            Path(root, physical + ".lock").touch()
            with patch("evaluation.wp12_resident_cost.admit_gpu", side_effect=ValueError("busy resource")):
                with self.assertRaisesRegex(ValueError, "busy resource"), gpu_measurement_lease(physical):
                    self.fail("Failed gate admitted a worker")
            with patch("evaluation.wp12_resident_cost.admit_gpu") as admit:
                with gpu_measurement_lease(physical) as lease:
                    self.assertFalse(lease.closed)
                    with self.assertRaisesRegex(ValueError, "already leased"), gpu_measurement_lease(physical):
                        self.fail("Two workers acquired the same physical lease")
                with gpu_measurement_lease(physical):
                    pass
                self.assertEqual(admit.call_count, 2)

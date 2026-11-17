from pathlib import Path
import subprocess
import sys
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

    def test_inherited_lease_outlives_parent_descriptor_until_worker_exits(self):
        physical = "GPU-ab4a0000-0000-0000-0000-00000000000a"
        child_code = (
            "import os, sys; "
            "fd=int(sys.argv[1]); "
            "sys.stdin.read(1); "
            "print(os.fstat(fd).st_ino); "
            "sys.exit(int(sys.argv[2]))"
        )
        with TemporaryDirectory() as root, patch.dict("os.environ", {"HSI_GPU_LEASE_ROOT": root}):
            path = Path(root, physical + ".lock")
            path.touch()
            with patch("evaluation.wp12_resident_cost.admit_gpu"):
                for exit_code in (0, 7):
                    with self.subTest(exit_code=exit_code):
                        with gpu_measurement_lease(physical) as lease:
                            fd = lease.fileno()
                            child = subprocess.Popen([sys.executable, "-c", child_code, str(fd), str(exit_code)],
                                                     pass_fds=(fd,), stdin=subprocess.PIPE,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        try:
                            self.assertTrue(lease.closed)
                            self.assertIsNone(child.poll())
                            with self.assertRaisesRegex(ValueError, "already leased"), gpu_measurement_lease(physical):
                                self.fail("Running child did not retain the inherited physical lease")
                            out, err = child.communicate("x", timeout=15)
                            self.assertEqual((int(out.strip()), err, child.returncode),
                                             (path.stat().st_ino, "", exit_code))
                            with gpu_measurement_lease(physical):
                                pass
                        finally:
                            if child.poll() is None:
                                child.kill()
                                child.communicate(timeout=10)

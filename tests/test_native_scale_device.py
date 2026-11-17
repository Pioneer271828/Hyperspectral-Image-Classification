"""Native numerical contracts leave the old CPU API and arithmetic intact."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts import run_native_scale_device as cli
from scripts import run_native_released_points as released_cli
from transfer import native_scale_device_campaign as campaign, native_scale_training as training
from transfer import p4_runtime_device as p4
from transfer.native_scale_device import arithmetic, settings
from transfer.native_scale_fixture import Fixture


class NativeScaleDeviceTests(TestCase):
    def test_native_cuda_entry_points_reject_shared_lease_before_entering_handlers(self):
        gpu = "GPU-11111111-1111-1111-1111-111111111111"
        sha = "a" * 40
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": gpu}):
            root = Path(folder) / "runs"
            other_root = root / "different" / "runs"
            other_root.mkdir(parents=True)
            leases = Path(folder) / "shared"
            leases.mkdir(parents=True)
            common = ["--expected-commit", sha, "--runs", str(root),
                      "--protocol", p4.CUDA_FP64_WEIGHT, "--gpu", gpu]
            out = ["--out", str(root / "out")]
            mark = ["/unused/registration.json", "0" * 64]
            commands = {
                "qualify": [*out, "--geometry", *mark],
                "register": [*out, "--qualification", *mark],
                "stage": ["--matrix", *mark, "--case", next(iter(campaign.cases())), "--phase", campaign.PHASES[0]],
                "audit": [*out, "--matrix", *mark],
                "cost": [*out, "--geometry", *mark],
            }
            with patch.object(cli, "code_snapshot", return_value={"commit": sha}), \
                    patch.object(cli, "require_frozen_code"), patch.object(cli, "training_container_kind", return_value="lxc"), \
                    patch.dict("os.environ", {"HSI_GPU_LEASE_ROOT": str(leases)}), \
                    patch.object(cli, "dispatch") as dispatch:
                with p4.lease(leases / f"{gpu}.lock") as parent:
                    for mode, args in commands.items():
                        with self.subTest(mode=mode), self.assertRaisesRegex(p4.ResourceUnavailable, "Active lease"):
                            cli.main([*common, mode, *args])
                    other = [*common]
                    other[other.index(str(root))] = str(other_root)
                    with self.assertRaisesRegex(p4.ResourceUnavailable, "Active lease"):
                        cli.main([*other, "qualify", "--out", str(other_root / "out"), "--geometry", *mark])
                    with self.assertRaisesRegex(p4.ResourceUnavailable, "Active lease"):
                        with released_cli.arithmetic(other_root, gpu):
                            self.fail("Released points must share the exact native-scale lease")
                    dispatch.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "inherited worker operation"):
                        dispatch.side_effect = RuntimeError("inherited worker operation")
                        cli.main([*common, "--gpu-lease-fd", str(parent.fileno()), "qualify", *commands["qualify"]])
                    with self.assertRaisesRegex(p4.ResourceUnavailable, "Active lease"):
                        cli.main([*common, "qualify", *commands["qualify"]])
                with self.assertRaisesRegex(RuntimeError, "direct worker operation"):
                    dispatch.side_effect = RuntimeError("direct worker operation")
                    cli.main([*common, "qualify", *commands["qualify"]])
                with p4.lease(leases / f"{gpu}.lock"):
                    pass

                other = leases / "other.lock"
                with p4.lease(other) as wrong, self.assertRaisesRegex(ValueError, "does not match"):
                    cli.main([*common, "--gpu-lease-fd", str(wrong.fileno()), "qualify", *commands["qualify"]])
                self.assertEqual(dispatch.call_count, 2)

    def test_bad_lease_directory_or_file_never_reaches_cuda(self):
        gpu = "GPU-11111111-1111-1111-1111-111111111111"
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": gpu}):
            root = Path(folder)
            shared = root / "shared"
            shared.mkdir()
            alias = root / "alias"
            alias.symlink_to(shared, target_is_directory=True)
            for wrong in ("relative", str(root / "missing"), str(alias)):
                with self.subTest(wrong=wrong), patch.dict("os.environ", {"HSI_GPU_LEASE_ROOT": wrong}), \
                        self.assertRaisesRegex(ValueError, "GPU lease directory"):
                    with cli.physical_lease(gpu):
                        self.fail("An invalid lease directory must be rejected")
            original = shared / "different.lock"
            original.touch()
            (shared / f"{gpu}.lock").symlink_to(original)
            with patch.dict("os.environ", {"HSI_GPU_LEASE_ROOT": str(shared)}), \
                    self.assertRaisesRegex(ValueError, "lease file cannot be a symlink"):
                with cli.physical_lease(gpu):
                    self.fail("A symlink cannot redirect the shared physical lease")

    def test_context_restore_on_exception_and_protocol_reject(self):
        before = settings()
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with arithmetic(p4.CPU) as execution:
                self.assertEqual(execution.numeric["device"],"cpu")
                self.assertFalse(execution.numeric["deterministic_warn_only"])
                raise RuntimeError("injected")
        self.assertEqual(settings(),before)
        with self.assertRaises(ValueError),arithmetic(p4.CUDA_AUDITED):
            pass

    def test_legacy_contract_and_new_full_factor_exposure(self):
        old = training.contract(9,64,"off","sample")
        with arithmetic(p4.CPU) as first:
            raw = training.contract(9,64,"off","sample",numeric=first.numeric)
            roster = campaign._frozen_inputs([raw],"sample")
        with arithmetic(p4.CPU,inputs=roster) as execution:
            spec = training.contract(9,64,"off","sample",numeric=execution.numeric)
            self.assertEqual((old["steps"],old["prefix"],old["format"]),(96,7,training.FORMAT))
            self.assertNotIn("numeric",old)
            self.assertEqual((spec["steps"],spec["prefix"],spec["events"]),(12,5,[0,6,12]))
            combinations = {(x["factor"],x["band"]) for x in spec["training"]}
            self.assertEqual(len(combinations),6)
            self.assertEqual(spec["initialization"],old["initialization"])
            with patch.object(Fixture,"materialize",side_effect=AssertionError("bad contract reached fixture")):
                with self.assertRaises(ValueError):
                    training.Run(spec,"other",execution=execution)
                wrong = deepcopy(spec)
                wrong["numeric"]["host"] = "other"
                with self.assertRaises(ValueError):
                    training.Run(wrong,"sample",execution=execution)
            run = training.Run(spec,"sample",execution=execution)
            run.start()
            run.update(spec["training"][0])
            packed = run.pack()
            self.assertIsNone(packed["rng"]["cuda"])
            wrong = deepcopy(packed)
            wrong["history"][0]["observation"] = "other"
            with patch.object(Fixture,"materialize",side_effect=AssertionError("bad restore reached fixture")):
                with self.assertRaisesRegex(ValueError,"observation identity"):
                    run.restore(wrong,1)
                with self.assertRaises(ValueError):
                    training.Run(old,"sample",execution=execution)
                with self.assertRaises(ValueError):
                    training.Run(spec,"sample")
            run.restore(packed,1)

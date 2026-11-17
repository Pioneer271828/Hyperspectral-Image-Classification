"""Native numerical contracts leave the old CPU API and arithmetic intact."""

from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from transfer import native_scale_device_campaign as campaign, native_scale_training as training
from transfer import p4_runtime_device as p4
from transfer.native_scale_device import arithmetic, settings
from transfer.native_scale_fixture import Fixture


class NativeScaleDeviceTests(TestCase):
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

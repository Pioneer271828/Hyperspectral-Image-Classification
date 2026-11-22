"""A decoded-cache campaign needs a distinct, exact warm authorization."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_wp12_paired_observation as paired
from scripts.wp12_frozen54_deployment_contract import WARM, digest_text


class DecodedCacheWarmAuthorityTest(unittest.TestCase):
    def test_signed_authority_binds_new_scope_and_budget_before_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            control = root / "controller"
            control.mkdir()
            path = control / "paired-plan.json"
            permit_path = root / "approval.json"
            runtime = root / "worker"
            plan = dict(schema=paired.CACHE_PLAN, producer="new-code", engineering=False,
                        execution=str(root / "fresh-execution"), timing={WARM:{}},
                        policy={**paired.POLICY, "measured_requests":72, "warmup_requests":12,
                                "cold_requests":0, "optimizer_updates":0, "labels_read":0})

            def write(file, value):
                raw = json.dumps(value).encode()
                file.write_bytes(raw)
                return {"path":str(file), "sha256":digest_text(raw)}

            plan_ref = write(path, plan)
            with patch.object(paired, "draft", return_value=plan), \
                 patch("scripts.benchmark_wp12_observation_io.require_clean") as check:
                self.assertRaisesRegex(ValueError, "separately supplied", paired.admitted,
                                       root, path, plan_ref["sha256"], "new-code", runtime, None)
                wrong = paired.authority(plan, plan_ref, "approved", "independent review")
                wrong["budget"]["labels_read"] = 1
                self.assertRaisesRegex(ValueError, "does not bind", paired.admitted,
                                       root, path, plan_ref["sha256"], "new-code", runtime, write(permit_path, wrong))
                check.assert_not_called()
                real = write(permit_path, paired.authority(plan, plan_ref, "approved", "independent review"))
                self.assertEqual(paired.admitted(root, path, plan_ref["sha256"], "new-code", runtime, real),
                                 (plan_ref, plan))
            self.assertFalse(Path(plan["execution"]).exists())


if __name__ == "__main__":
    unittest.main()

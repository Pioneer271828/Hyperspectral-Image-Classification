"""Synthetic queue fault paths and independent audit of actual full-budget fits."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import g4_light_full_campaign as campaign
from g4_light_full_audit import audit_case
from g4_light_full_target import UPDATES, prepare_case
from g4_light_target import ADAPTER_HEADS, HEAD_ONLY
from g4_matched_target_audit import files
from joint_training import run_training
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, write_json
from test_g4_light_full_target import registration_fixture


KEYS = [("xuzhou", 64, method, "none") for method in (HEAD_ONLY, ADAPTER_HEADS)]


def cursor(folder, step, status):
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / f"state-{step:06d}.pt"
    state.write_text("Synthetic cursor")
    write_json(folder / "progress.json", {"cursor": step, "status": status,
        "state_file": state.name, "state_sha256": file_hash(state)})


class CampaignTests(unittest.TestCase):
    def test_recovery_and_two_gpu_completion_restart_from_partial_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            code = code_snapshot()
            campaign.claim_output(root, "r")
            write_json(root / "status.json", {"policy": campaign.POLICY, "registration_sha256": "r",
                "campaign_engineering_sha256": "e", "gpu_ids": [1, 3], "status": "running"})
            first = campaign.case_directory(root, KEYS[0])
            cursor(first, 31, "running")
            calls = []

            def invoke(_, folder, gpu, command, stage, lease):
                calls.append((folder.name, gpu, stage))
                if stage == "first2":
                    cursor(folder, 2, "interrupted")
                    return {"operation": "interrupted", "updates_this_call": 2}
                if stage == "resume":
                    previous = campaign.read(folder / "progress.json")["cursor"]
                    cursor(folder, UPDATES, "completed")
                    write_json(folder / "result.json", {"fixture": True})
                    return {"operation": "completed", "updates_this_call": UPDATES - previous}
                if stage == "readonly":
                    return {"operation": "verified_completed", "updates_this_call": 0}
                return {"status": "passed", "name": folder.name, "role": list(KEYS[0 if gpu == 1 else 1]),
                        "physical_gpu": gpu, "updates": UPDATES, "validation_events": 51,
                        "exact_reload_repeats": 3, "new_optimizer_updates": 0, "query_labels_accessed": False,
                        "protected_inputs": {}, "artifacts": files(folder)}

            with patch.object(campaign, "roles", return_value=KEYS), patch.object(campaign, "eligible", return_value=True), patch.object(campaign, "invoke", side_effect=invoke):
                campaign.run_queues(root, [1, 3], [], code, "r", {}, "e", resume=True)
                self.assertEqual(campaign.read(root / "training-acceptance.json")["optimizer_updates"], 3000)
                self.assertEqual([stage for name, gpu, stage in calls if gpu == 1], ["resume", "readonly", "audit"])
                self.assertEqual([stage for name, gpu, stage in calls if gpu == 3], ["first2", "resume", "readonly", "audit"])
                before = files(root)
                campaign.run_queues(root, [1, 3], [], code, "r", {}, "e", resume=True)
                self.assertEqual(files(root), before)
                status = campaign.read(root / "status.json")
                write_json(root / "status.json", {**status, "status": "running"})
                campaign.run_queues(root, [1, 3], [], code, "r", {}, "e", resume=True)
                self.assertEqual(files(root), before)

    def test_failure_preserves_logs_no_acceptance_and_no_automatic_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            code = code_snapshot()

            def fail(*args):
                (root / "logs" / "worker-failure.log").write_text("synthetic failure")
                raise RuntimeError("worker failure")

            with patch.object(campaign, "roles", return_value=KEYS), patch.object(campaign, "run_role", side_effect=fail):
                with self.assertRaises(RuntimeError):
                    campaign.run_queues(root, [1, 3], [], code, "r", {}, "e", resume=False)
            self.assertTrue((root / "logs" / "worker-failure.log").is_file())
            self.assertFalse((root / "training-acceptance.json").exists())
            self.assertEqual(campaign.read(root / "status.json")["status"], "failed")
            with self.assertRaises(ValueError):
                campaign.run_queues(root, [1, 3], [], code, "r", {}, "e", resume=True)

    def test_three_gpu_assignment_resource_wait_and_inherited_dispatch_descriptor(self):
        class Stop:
            stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, seconds):
                self.stopped = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            calls = []
            with (root / "dispatch").open("a") as lease:
                cmd = [sys.executable, "-c",
                       f"import os; os.fstat({lease.fileno()}); print('{{\"descriptor\": true}}')"]
                self.assertEqual(campaign.invoke(root, root / "case", 5, cmd, "test", lease), {"descriptor": True})
                with patch.object(campaign, "eligible", return_value=False), patch.object(campaign, "invoke") as child:
                    self.assertIsNone(campaign.run_role(root, KEYS[0], 5, [], Stop(), lambda *a: calls.append(a), lease))
                    child.assert_not_called()
            self.assertEqual(calls[0][0:2], (5, "waiting_for_resources"))
            grid = campaign.roles()
            gpus = [0, 1, 5]
            self.assertEqual([len(grid[i::len(gpus)]) for i in range(3)], [18, 18, 18])
            self.assertEqual(len(set(grid)), 54)

    def test_independent_reload_full_case_and_tampered_raw_event(self):
        device = os.environ.get("LIGHT_CAMPAIGN_TEST_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registration, spec, code = registration_fixture(root)
            sha = file_hash(registration)
            with patch("g4_light_full_target.REFERENCE_LAUNCH_SHA256", spec["reference_launch"]["sha256"]):
                for key in KEYS if device == "cpu" else KEYS[:1]:
                    items, provenance, runtime, guard = prepare_case(registration, sha, code, key, device=device)
                    runtime.directory.parent.mkdir(parents=True, exist_ok=True)
                    result = run_training(runtime.settings, items, runtime.directory, provenance, code,
                        runtime_adapter=runtime, finish_guard=guard, device=device)
                    before = files(runtime.directory)
                    row = audit_case(registration, sha, code, key, device=device)
                    self.assertEqual(row["updates"], UPDATES)
                    self.assertEqual(row["validation_events"], 51)
                    self.assertEqual(row["selected_step"], result["result"]["selected_step"])
                    self.assertEqual(row["head_metrics"], 6)
                    self.assertEqual(row["exact_reload_repeats"], 3)
                    self.assertEqual(files(runtime.directory), before)
                    (runtime.directory / "raw-validation-000001.pt").write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "artifact"):
                        audit_case(registration, sha, code, key, device=device)


if __name__ == "__main__":
    unittest.main()

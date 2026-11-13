"""Reject fabricated identities and keep planning states outside ranked evidence."""
import fcntl
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import wp12_p4_released_evidence as evidence
from evaluation import wp12_p4_progress as progress
from research_runtime import write_json
from transfer import registered_development as dev


class EvidenceTests(unittest.TestCase):
    def test_planned_running_waiting_failed_and_partial_counts(self):
        resource = {"slots": {"a": {"cases": ["a"], "updates": 120},
                              "b": {"cases": ["b"], "updates": 120}}}
        self.assertEqual({r["state"] for r in evidence.queue_states(resource).values()}, {"planned"})
        empty = {"planned_cases": 1, "accepted_cases": 0, "planned_updates": 120, "optimizer_updates": 0}
        committed = {**empty, "accepted_cases": 1, "optimizer_updates": 120}
        status = {"status": "running", "slots": {"a": empty, "b": empty}}
        self.assertEqual({r["state"] for r in evidence.queue_states(resource, status).values()}, {"running"})
        partial = {**status, "status": "waiting_for_resources",
                   "slots": {"a": committed, "b": empty}, "waiting": {"b": {"reason": "busy"}}}
        self.assertEqual([r["state"] for r in evidence.queue_states(resource, partial).values()],
                         ["audited_pending_batch_acceptance", "waiting"])
        failed = {**status, "status": "failed", "errors": {"a": {"reason": "exit"}}}
        self.assertEqual([r["state"] for r in evidence.queue_states(resource, failed).values()],
                         ["failed", "drained_after_peer_failed"])
        acceptance = {"status": "passed", "cases": 2, "optimizer_updates": 240,
                      "slots": {slot: {"cases": 1, "optimizer_updates": 120} for slot in ("a", "b")}}
        with self.assertRaisesRegex(ValueError, "full ledger"):
            evidence.queue_states(resource, partial, acceptance)
        terminal = {**status, "status": "completed", "slots": {"a": committed, "b": committed}}
        self.assertEqual({r["state"] for r in evidence.queue_states(resource, terminal, acceptance).values()},
                         {"accepted"})

    def test_read_only_running_status_uses_only_committed_acknowledgments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            out.mkdir()
            name = "s2048-k1-scratch-m512-none"
            resource = dev._json_new(root / "scope.json", {"execution": str(out),
                "schema": "wp05_p4_independent_slot_scope_v1", "science": {"sha256": "a"*64},
                "slots": {"cpu-a": {"updates": 120, "cases": [name]}}})
            write_json(out / "resource-owner.json", {"resource": resource})
            with (out / ".coordinator.lock").open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                call = out / "cases" / name / "full/calls/000000-000060.json"
                call.parent.mkdir(parents=True)
                write_json(call, {"case": name, "path": "full", "scope_sha256": "a"*64,
                                  "start": 0, "until": 60, "new_optimizer_updates": 60})
                stages = out / "stages" / name
                stages.mkdir(parents=True)
                live = lambda: progress.status(resource, root, approved_resource_sha=resource["sha256"])
                self.assertEqual(live()["slots"]["cpu-a"]["optimizer_updates"], 0)
                write_json(stages / "00.intent.json", {"resource_sha256": resource["sha256"], "slot": "cpu-a",
                    "case": name, "operation": "advance", "path": "full", "start": 0, "until": 60})
                write_json(stages / "00.receipt.json", {"evidence": dev._pin(call),
                                                       "attempt": dev._pin(stages / "00.intent.json")})
                self.assertEqual(live()["slots"]["cpu-a"]["optimizer_updates"], 60)
                good = call.read_bytes()
                write_json(call, {"changed": True})
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    live()
                call.write_bytes(good)
                write_json(out / "status.json", {"status": "running", "resource_sha256": resource["sha256"],
                                                  "slots": live()["slots"]})
            self.assertEqual(live()["status"], "interrupted_or_unknown")
            with (out / ".coordinator.lock").open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                write_json(out / "failure.json", {"cpu-a": {"case": name}})
                self.assertEqual(live()["status"], "draining_after_failure")
            self.assertEqual(live()["status"], "failed")

    def test_foreign_source_even_when_well_formed_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake = dev._json_new(root / "source.json", {"schema": evidence.SCHEMA, "status": "preflight_only",
                "producer": evidence.SPARSE_PRODUCER, "optimization_updates": 0})
            with self.assertRaisesRegex(ValueError, "authenticated real sparse"):
                evidence.sparse_prepared(fake, root)
            with patch.object(evidence, "read") as read:
                with self.assertRaisesRegex(ValueError, "final P4 CPU"):
                    evidence.p4_engineering(fake, root)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Reject fabricated identities and keep planning states outside ranked evidence."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import wp12_p4_released_evidence as evidence
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

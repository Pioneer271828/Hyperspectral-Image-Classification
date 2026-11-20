"""Regression checks for independent WP03 preparation, resource and receipt boundaries."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from g4_light_engineering_registration import pin
import g4_light_probe_pair as probe
import g4_light_real_campaign as admission
from g4_light_duration import schedule
from g4_light_full_target import assigned_role, roles
from research_runtime import write_json


class RealAdmissionTests(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.code = {"commit": "a" * 40, "source_hash": "b" * 64}
        self.device = {"slot": 0, "uuid": "GPU-0c958696-3721-4009-628f-befb0ef9b867"}

    def json(self, name, value):
        path = self.root / name
        write_json(path, value)
        return pin(path)

    def test_alternate_diagnostic_never_inherits_original_authority(self):
        old = {"experiment_issue": 249, "producer_commit": probe.PRODUCER_COMMIT}
        original = {"path": "/original", "sha256": probe.REGISTRATION_SHA256}
        with patch.object(probe, "pinned", return_value=old):
            native = probe.probe_intent(original, probe.ORIGINAL, self.code)
            self.assertEqual(native["minimum_free_mib"], 20 * 1024)
            with self.assertRaisesRegex(ValueError, "separate approved"):
                probe.probe_intent(original, self.device, self.code)

    def test_plan_rejects_missing_evidence_and_draft_authority(self):
        plan = {"producer": self.code, "scope": admission.REAL, "request": {}, "output": "/new",
                "devices": [self.device], "missing": ["real_diagnostic"]}
        plan_pin = self.json("plan.json", plan)
        with patch.object(admission, "prepare", return_value=plan):
            with self.assertRaisesRegex(ValueError, "Missing admission evidence"):
                admission.activate(plan_pin, None, self.code)
        plan["missing"] = []
        plan_pin = self.json("ready.json", plan)
        draft = self.json("draft.json", admission.release_for(plan_pin, plan, decision="review_required"))
        with patch.object(admission, "prepare", return_value=plan):
            with self.assertRaisesRegex(ValueError, "Separate approved"):
                admission.activate(plan_pin, draft, self.code)

    def test_tampered_plan_or_historical_acceptance_cannot_be_activated(self):
        original = {"scope": admission.REAL, "request": {}, "missing": [],
                    "historical": {"counted_in_new_acceptance": 0}}
        altered = deepcopy(original)
        altered["historical"]["counted_in_new_acceptance"] = 36
        altered_pin = self.json("altered.json", altered)
        with patch.object(admission, "prepare", return_value=original):
            with self.assertRaisesRegex(ValueError, "no longer reproduces"):
                admission.activate(altered_pin, None, self.code)

    def test_all54_2_and_3_slot_assignments_are_deterministic_and_conserve_counts(self):
        grid = roles()
        rows = [{"role": list(key), "update_seconds": [float(i + 1)] * 2,
                 "phase_seconds": [float(i + 2)] * 2} for i, key in enumerate(grid)]
        for n in (2, 3):
            for balanced in (False, True):
                result = schedule(rows, n, balanced=balanced)
                self.assertEqual(len(result["queues"]), n)
                flattened = [tuple(r) for queue in result["queues"] for r in queue["cases"]]
                self.assertCountEqual(flattened, grid)
                if not balanced:
                    ledger = [{"slot": i % n} for i in range(54)]
                    reg = {"roles": ledger}
                    for i,key in enumerate(grid):
                        slot, role = assigned_role(reg, key)
                        self.assertEqual(slot, i % n)
                        self.assertIs(role, ledger[i])

    def test_numeric_gate_keeps_negative_and_dtype_boundary(self):
        stable = torch.tensor([1e-3, 2e-2])
        self.assertTrue(probe.diff(stable, stable)["exact"])
        within = probe.diff(stable, stable + 1e-7)
        self.assertFalse(within["exact"])
        self.assertEqual(within["outside"], 0)
        self.assertEqual(probe.diff(stable, stable + 1e-4)["outside"], 2)
        with self.assertRaisesRegex(ValueError, "type or finiteness"):
            probe.diff(stable, stable.double())

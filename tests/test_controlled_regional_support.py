"""Independent invariants and native forward boundary for registered interventions."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from evaluation.controlled_support_metrics import released_samples
from evaluation.controlled_support_oracle import audit
from model.research_config import digest
from transfer.controlled_regional_support import match_support
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import state_hash
from transfer.native_support import NativeSupportModel, make_bank
from transfer.native_support_campaign import _fixture, _trial
from transfer.native_support_intervention import BoundIntervention, forward_intervention, prepare_block


class MatchedSupportTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.ids = [100, 5, 13, 31]
        self.partitions = [0] * 4
        self.diagonal = [[i, i] for i in range(4)]
        self.reference = sorted(self.diagonal + [[0, 1], [3, 2]])
        self.allowed = sorted(self.reference + [[0, 2], [3, 1]])

    def search(self, **overrides):
        data = dict(reference=self.reference, allowed=self.allowed, mandatory=[], points=self.points,
                    partitions=self.partitions, node_ids=self.ids, seed=64)
        return match_support(**{**data, **overrides})

    def test_exact_directed_exchange_and_independent_enumeration(self):
        row = self.search()
        self.assertEqual(row["status"], "MATCHED")
        self.assertEqual(row["pairs"], sorted(self.diagonal + [[0, 2], [3, 1]]))
        self.assertEqual(row["constraints"]["receive"], [2, 1, 1, 2])
        self.assertEqual(row["constraints"]["send"], [1, 2, 2, 1])
        self.assertEqual(row["constraints"]["squared_distances"], {"0": 4, "1": 2})
        self.assertEqual(audit(dict(match=row, reference={"selected": self.reference},
                                    allowed=self.allowed, points=self.points)), 6)

    def test_protected_and_distance_mismatch_exhaustion(self):
        row = self.search(mandatory=[[0, 1]])
        self.assertEqual(row["status"], "INFEASIBLE")
        self.assertEqual(audit(dict(match=row, reference={"selected": self.reference},
                                   allowed=self.allowed, points=self.points,
                                   mandatory=self.diagonal + [[0, 1]])), 3)
        changed = deepcopy(self.points)
        changed[2] = [2, 0]
        self.assertEqual(self.search(points=changed)["status"], "INFEASIBLE")
        self.assertEqual(self.search(allowed=self.reference)["status"], "INFEASIBLE")

    def test_invalid_edges_and_bounded_failure(self):
        with self.assertRaisesRegex(ValueError, "Unique"):
            self.search(allowed=sorted(self.allowed + [self.allowed[0]]))
        with self.assertRaisesRegex(ValueError, "same-partition"):
            self.search(partitions=[0, 1, 0, 1])
        changed = deepcopy(self.points)
        changed[2] = [2, 0]
        row = self.search(points=changed, limit=1)
        self.assertEqual(row["status"], "NOT_FOUND_WITHIN_BUDGET")
        self.assertLessEqual(row["proposals"], 1)

    def test_native_intervention_receipt_and_roles(self):
        with TemporaryDirectory() as name:
            pins, base, scene, record = _fixture(Path(name) / "fixture", 9, 8, 64, trusted=False)
            plan = ExposurePlan.freeze(scene, record, updates=48, seed=64,
                minimum_class_blocks={str(c): 2 for c in record["classes"]}, minimum_support_visits=1)
            bank = make_bank(base, scene, record, plan, pins["development"]["sha256"], 7321)
            model = NativeSupportModel.initialize(base, 7321)
            before = state_hash(model), state_hash(base)
            role = next(row for row in record["roles"] if row["role"] == "devval")
            entry = bank[str(role["id"])]
            joint, old = _trial("joint", 19100), _trial("random", 19100)
            owner = {"fixture": "nine"}
            receipt, _ = prepare_block(entry, role["box"], joint, old, 0, 64, owner)
            bound = BoundIntervention(entry, joint, old, receipt, "reference", owner, expected_sha256=receipt["sha256"])
            with torch.no_grad():
                outputs, frame = forward_intervention(model, scene, record, scene.plan["tiles"][role["id"]], bound)
            self.assertEqual(outputs[0].shape[-1], 9)
            self.assertEqual(frame["trace"]["origin_snapshot"], receipt["reference"])
            self.assertNotIn("snapshot", frame["trace"])
            self.assertEqual((state_hash(model), state_hash(base)), before)
            with self.assertRaisesRegex(ValueError, "development block"):
                forward_intervention(model, scene, record, scene.plan["tiles"][next(r["id"] for r in record["roles"]
                                     if r["role"] == "devtrain")], bound)
            forged = deepcopy(receipt)
            forged["match"]["pairs"].pop()
            forged["sha256"] = digest({key: value for key, value in forged.items() if key != "sha256"})
            with self.assertRaisesRegex(ValueError, "Invalid constrained"):
                BoundIntervention(entry, joint, old, forged, "reference", owner, expected_sha256=forged["sha256"])
            with self.assertRaisesRegex(ValueError, "owner"):
                BoundIntervention(entry, joint, old, receipt, "reference", {}, expected_sha256=receipt["sha256"])
            self.assertEqual(len(released_samples(record, role)), 45)
            with self.assertRaisesRegex(ValueError, "released"):
                released_samples(record, dict(role, role="query"))

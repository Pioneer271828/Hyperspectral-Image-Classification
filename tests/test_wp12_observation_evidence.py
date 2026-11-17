"""Verify the observation table's independent quality and accounting joins."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.classification_metrics import classification_metrics
from evaluation.wp12_observation_evidence import STAGES, _quality, _training, synthetic
from research_runtime import hash_file


class ObservationEvidenceTests(unittest.TestCase):
    def test_ordered_predictions_and_frozen_condition(self):
        classes = list(range(1, 10))
        targets = [c for c in classes for _ in range(5)]
        predicted = list(targets)
        predicted[0] = 2
        head = classification_metrics(targets, predicted, classes)
        head.update(nll=0.25, zero_recall=[])
        value = dict(condition_record="frozen", targets=targets, labels=45, forwards=5, seconds=1.25,
                     scores=[["a"*64, "b"*64, "c"*64]] * 5,
                     metrics={h: deepcopy(head) for h in ("fused", "graph", "cnn")},
                     predictions={h: list(predicted) for h in ("fused", "graph", "cnn")})
        result = _quality(value, classes, targets, "frozen", 5)
        self.assertEqual(result["quality"]["cnn"]["per_class"][0]["recall"], 0.8)
        with self.assertRaisesRegex(ValueError, "condition"):
            _quality(value, classes, targets, "another", 5)
        value["predictions"]["graph"][0] = 1
        with self.assertRaisesRegex(ValueError, "ordered predictions"):
            _quality(value, classes, targets, "frozen", 5)

    def test_full_v5_and_resume_verification_have_separate_costs(self):
        entries = {}
        processes = {}
        for stage, calls in STAGES.items():
            extra = 2 if stage in ("resume", "completed") else 0
            entries[stage] = dict(new_updates=calls, timing=dict(training_forward=dict(calls=calls),
                                  full_v5_event=dict(calls=calls + extra)), restore_or_prepare_seconds=4.,
                                  update_loop_seconds=5. if calls else 0., checkpoint_finish_seconds=3.)
            processes[f"case-{stage}"] = {"wall_seconds": 16.}
        item = dict(case={"case": "case"}, stage_costs=entries, v5_stage_events=360, updates=356,
                    exposure={"train_label_visits": 300})
        value = _training(item, {"stages": processes}, 5, 45)
        self.assertEqual((value["training_v5_forwards"], value["restore_v5_forwards"]), (1780, 20))
        self.assertEqual((value["training_v5_labels"], value["restore_v5_labels"]), (16020, 180))
        self.assertEqual((value["label_accesses"], value["subprocess_wall_seconds"]), (600, 64.))
        entries["completed"]["timing"]["training_forward"]["calls"] = 1
        with self.assertRaisesRegex(ValueError, "Stage timing"):
            _training(item, {"stages": processes}, 5, 45)

    @unittest.skipUnless(os.getenv("WP01_FACTORIAL_FIXTURES"), "existing independent synthetic audit receipts required")
    def test_factorial_receipts_do_not_add_production_rows(self):
        def pin(path):
            return dict(path=str(path), sha256=hash_file(path))

        root = Path(os.environ["WP01_FACTORIAL_FIXTURES"])
        spec = {str(c): pin(root/str(c)/"batch"/"audit.json") for c in (3, 9, 22)}
        receipts = synthetic(spec)
        self.assertEqual(receipts["real_training"], "NOT_RUN")
        self.assertEqual(receipts["production_rows_added"], 0)
        self.assertEqual(sum(v["synthetic_updates"] for v in receipts["fixtures"].values()), 448)
        original = json.loads(Path(spec["9"]["path"]).read_text())
        with TemporaryDirectory(dir=root.parent) as folder:
            path = Path(folder)/"audit.json"
            forged = deepcopy(original)
            forged["effects"]["graph"]["oa"]["interaction"] += 1
            path.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "effect arithmetic"):
                synthetic(dict(spec, **{"9": pin(path)}))
            original["optimizer_updates"] = 1
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "scope or execution budget"):
                synthetic(dict(spec, **{"9": pin(path)}))

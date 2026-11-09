"""Budget/access and immutable-ledger negative checks, without model imports."""
import copy
import tempfile
from pathlib import Path
import unittest
import g4_resource_final_audit as a


class ResourceFinalAuditTests(unittest.TestCase):
    def fixture(self):
        jobs = {name: "first2" for name in ("none", "capacity", "residual")}
        events = []
        for name in jobs:
            for stage in a.STAGES:
                if name == "capacity" and stage == "audit":
                    events.append(dict(case=name, stage=stage, gpu=7, exit_code=0, result=dict(a.WAIT)))
                result = dict(a.RESULTS[stage]) if stage != "audit" else {"audit": name + ".json", "sha256": "a"*64}
                events.append(dict(case=name, stage=stage, gpu=3, exit_code=0, result=result))
        return jobs, events

    def test_exact_budget_and_legal_wait(self):
        jobs, events = self.fixture()
        self.assertEqual(a.validate_events(jobs, [1, 3, 7], events),
                         dict(successful_stages=12, resource_wait_events=1, recorded_updates=4500))

    def test_positive_stage_cannot_repeat_or_skip_readonly(self):
        jobs, original = self.fixture()
        for events in (original[:2] + [original[1]] + original[2:],
                       [e for e in original if not (e["case"] == "none" and e["stage"] == "readonly")]):
            with self.assertRaises(ValueError):
                a.validate_events(jobs, [1, 3, 7], events)

    def test_budget_and_nonzero_wait_rejected(self):
        jobs, original = self.fixture()
        for index, result in ((1, {"operation": "completed", "updates_this_call": 1497}),
                              (7, {"operation": "resource_wait", "updates_this_call": 1})):
            events = copy.deepcopy(original)
            events[index]["result"] = result
            with self.assertRaises(ValueError):
                a.validate_events(jobs, [1, 3, 7], events)

    def test_unknown_gpu_job_and_nonzero_exit_rejected(self):
        jobs, original = self.fixture()
        for key, value in (("gpu", 0), ("case", "unexpected"), ("exit_code", 1)):
            events = copy.deepcopy(original)
            events[0][key] = value
            with self.assertRaises(ValueError):
                a.validate_events(jobs, [1, 3, 7], events)

    def rows(self):
        rows = [dict(name=str(i), status="passed", updates=1500, validation_events=51,
                     head_metrics=6, class_recalls=54 if i < 36 else 132,
                     exact_reload_repeats=3, completed_resume_updates=0,
                     query_labels_accessed=False) for i in range(54)]
        return ({"cases": [{"name": row["name"]} for row in rows]},
                {"models": copy.deepcopy(rows)}, rows, {"cases": copy.deepcopy(rows)},
                {"completed_rows": copy.deepcopy(rows[:51]), "jobs": {str(i): "first2" for i in range(51, 54)}})

    def test_complete54_and_preserved51(self):
        args = self.rows()
        self.assertEqual(a.validate_rows(*args)["updates"], 81000)
        args[-1]["completed_rows"][0]["selected_step"] = 100
        with self.assertRaises(ValueError):
            a.validate_rows(*args)

    def test_acceptance_budget_query_and_duplicate_mismatch(self):
        for key, value in (("updates", 1499), ("query_labels_accessed", True), ("completed_resume_updates", 1)):
            args = self.rows()
            args[1]["models"][-1][key] = args[2][-1][key] = args[3]["cases"][-1][key] = value
            with self.assertRaises(ValueError):
                a.validate_rows(*args)
        args = self.rows()
        args[2][-1]["name"] = args[2][-2]["name"]
        with self.assertRaises(ValueError):
            a.validate_rows(*args)

    def test_protected_byte_and_json_hash_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            path.write_text('{"status":"passed"}')
            pins = {str(path): a.sha(path)}
            a.verify_files(pins)
            self.assertEqual(a.checked_json(path, pins[str(path)])["status"], "passed")
            path.write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                a.verify_files(pins)
            with self.assertRaises(ValueError):
                a.checked_json(path, pins[str(path)])

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):
                a.read(path)

    def test_conflicting_pins_cannot_overwrite_original_guard(self):
        pins = {"protected": "original"}
        a.merge_pins(pins, {"protected": "original"})
        with self.assertRaises(ValueError):
            a.merge_pins(pins, {"protected": "changed"})
        self.assertEqual(pins, {"protected": "original"})


if __name__ == "__main__":
    unittest.main()


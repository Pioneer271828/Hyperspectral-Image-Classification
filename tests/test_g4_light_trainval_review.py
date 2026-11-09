"""Check review evidence and the score-side review gate."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from g4_light_full_target import roles
from g4_light_ordered_query import REVIEW_POLICY, case_directory, review_inputs
import g4_light_trainval_review as review


class LightReviewTests(unittest.TestCase):
    def test_complete_matching_review_gate_and_missing_or_changed_export(self):
        with TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            code = {"dirty": False, "files": {}, "source_hash": review.digest({})}
            models = [{"name": case_directory(out, k).name, "role": list(k), "selected_step": 1500} for k in roles()]
            freeze = {"acceptance_sha256": "acceptance", "models": models}
            receipt = {"policy": REVIEW_POLICY, "status": "passed", "freeze_sha256": "freeze",
                       "acceptance_sha256": "acceptance", "optimizer_updates_reviewed": 81000,
                       "new_optimizer_updates": 0, "query_labels_accessed": False, "cases": models, "code": code,
                       "exported_files": {}}
            for name in (m["name"] + suffix for m in models for suffix in (".png", "-curve.csv")):
                file = out / name
                file.write_text("fixture")
                receipt["exported_files"][name] = review.file_hash(file)
            path = out / "summary.json"
            review.write_json(path, receipt)
            review_inputs(path, "freeze", freeze, code)
            for changed in ({"cases": models[:-1]}, {"acceptance_sha256": "other"}, {"query_labels_accessed": True}):
                review.write_json(path, {**receipt, **changed})
                with self.assertRaisesRegex(ValueError, "Complete matching"):
                    review_inputs(path, "freeze", freeze, code)
            review.write_json(path, receipt)
            file.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Pinned"):
                review_inputs(path, "freeze", freeze, code)

    def test_export_rejects_running_campaign_before_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, out = root / "freeze.json", root / "review"
            review.write_json(path, {"policy": "running"})
            with patch.object(review, "light_inputs", side_effect=ValueError("Incomplete campaign")) as verify:
                with self.assertRaisesRegex(ValueError, "Incomplete campaign"):
                    review.export_review(path, review.file_hash(path), out, {})
                verify.assert_called_once()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

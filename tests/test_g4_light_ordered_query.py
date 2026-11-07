"""Exercise Issue249 scoring gates without releasing real labels."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import g4_light_ordered_query as query
import test_g4_light_frozen_boundary as fixture


class LightQueryInputsTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.LightBoundaryTest("test_complete_metadata_freezes_every_role")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for row in self.f.receipts.values():
            folder = query.case_directory(self.f.root, tuple(row["role"]))
            raw = self.f.write(folder / "raw-validation-001470.pt", {"synthetic": row["name"]})
            row["artifacts"].update(self.f.pins(raw))
            self.f.protected.update(self.f.pins(raw))
        self.freeze = self.f.freeze()
        for name, value in (("REGISTRATION_SHA256", self.f.registration_sha),
                            ("PLAN_SHA256", query.read(self.f.registration)["analysis_plan"]["sha256"])):
            mock = patch.object(query, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch("g4_light_full_target.REFERENCE_LAUNCH_SHA256", query.file_hash(self.f.launch))
        mock.start()
        self.addCleanup(mock.stop)

    def test_complete_authentication_has_no_label_side_effect(self):
        with patch.object(query, "read_query_labels") as label_reader:
            root, _, cases = query.light_inputs(self.freeze, self.f.code)
        self.assertEqual(root, self.f.root)
        self.assertEqual(len(cases), 27)
        label_reader.assert_not_called()

    def test_incomplete_or_changed_selected_artifact_is_rejected(self):
        changed = {**self.freeze, "models": self.freeze["models"][:-1]}
        with self.assertRaisesRegex(ValueError, "Exact Issue249"):
            query.light_inputs(changed, self.f.code)
        self.freeze["models"][0]["selected_step"] = 2
        with self.assertRaisesRegex(ValueError, "selected raw state"):
            query.light_inputs(self.freeze, self.f.code)

    def test_live_status_cannot_be_replaced_even_with_recomputed_pin(self):
        self.f.write(self.f.status_path, {**self.f.status, "status": "running"})
        self.freeze["boundary_files"].update(self.f.pins(self.f.status_path))
        with self.assertRaisesRegex(ValueError, "must be completed"):
            query.light_inputs(self.freeze, self.f.code)


class LightQueryReleaseTests(unittest.TestCase):
    def test_each_reference_release_must_bind_same_ordered_pixels_and_labels(self):
        releases = {target: {"parent_hash": target, "checkpoint_sha256": "ref", "indices": [2, 0],
                            "labels": [1, 2], "phase": "final_query", "model_evaluated": False}
                    for target in query.NAMES}
        references = {target: {"target": target} for target in query.NAMES}
        rows = [{"target": target, "checkpoint_sha256": weight,
                 "release_sha256": query.digest({**release, "checkpoint_sha256": weight})}
                for target, release in releases.items() for weight in ("a", "b")]
        with patch.object(query, "read_query_labels", side_effect=lambda _, case, *args: (releases[case["target"]], {})) as loader:
            accepted, _ = query.matched_releases({}, rows, references, "data", "prepared", "sha")
            self.assertEqual(accepted, releases)
            self.assertEqual(loader.call_count, len(query.NAMES))
            releases[query.NAMES[0]]["indices"].reverse()
            with self.assertRaisesRegex(ValueError, "pixel or label identity"):
                query.matched_releases({}, rows, references, "data", "prepared", "sha")

    def test_entry_point_checks_all_roles_before_releasing_labels_or_writing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for arg in ("freeze", "reference_freeze", "reference_query", "prepared_manifest"):
                paths[arg] = root / arg
                query.write_json(paths[arg], {"status": "running"} if arg == "freeze" else {})
            out = root / "out"
            args = Namespace(**paths, out=out, freeze_sha256=query.file_hash(paths["freeze"]),
                             prepared_sha256=query.file_hash(paths["prepared_manifest"]))
            with (patch.object(query, "REFERENCE_FREEZE_SHA256", query.file_hash(paths["reference_freeze"])),
                  patch.object(query, "REFERENCE_QUERY_SHA256", query.file_hash(paths["reference_query"])),
                  patch.object(query, "matched_releases") as release,
                  patch.object(query, "require_gpu") as gpu):
                with self.assertRaises((ValueError, KeyError)):
                    query.evaluate(args, {})
                release.assert_not_called()
                gpu.assert_not_called()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

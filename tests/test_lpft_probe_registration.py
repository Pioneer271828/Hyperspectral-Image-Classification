"""Read-only plan rejection and authenticated reference selection; no training."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lpft_probe import registration as reg
from loadData.dataset_registry import file_hash
from research_runtime import write_json
from test_g4_light_full_target import registration_fixture


def manifest(commit="a" * 40):
    return {"policy": reg.POLICY, "producer_commit": commit,
            "reference_launch": {"path": "/inputs/launch.json", "sha256": reg.REFERENCE_LAUNCH_SHA256},
            "out_dir": "/runs/new-short", "cases": [{"target": "xuzhou", "seed": 64,
                                                       "mode": "none", "gpu": 5}],
            "plan": {"total_steps": 12, "probe_steps": 4, "head_lr": .0005,
                     "backbone_lr": .000005, "weight_decay": 0.0},
            "saved_tensors": "device", "planned_total_updates": 12,
            "estimated_seconds": 300, "query_labels_accessed": False,
            "selection": "none_endpoint_only"}


class RegistrationTests(unittest.TestCase):
    def test_budget_label_memory_and_role_rejections_before_io(self):
        good = manifest()
        actual, plan = reg.validate(good, "a" * 40)
        self.assertEqual(plan.total_steps, 12)
        actual["cases"][0]["gpu"] = 0
        self.assertEqual(good["cases"][0]["gpu"], 5)
        corruptions = [
            lambda v: v.update(planned_total_updates=36),
            lambda v: v.update(planned_updates=12),
            lambda v: v["plan"].update(total_steps=1500),
            lambda v: v["plan"].update(probe_steps=12),
            lambda v: v["cases"].append(copy.deepcopy(v["cases"][0])),
            lambda v: v.update(cases=[]),
            lambda v: v["cases"][0].update(seed=True),
            lambda v: v["cases"][0].update(gpu=True),
            lambda v: v["cases"][0].update(target="unknown"),
            lambda v: v.update(saved_tensors="auto"),
            lambda v: v.update(query_labels_accessed=True),
            lambda v: v.update(selection="best_query"),
            lambda v: v.update(estimated_seconds=float("nan")),
            lambda v: v.update(producer_commit="b" * 40),
            lambda v: v.update(out_dir="relative"),
        ]
        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index):
                value = copy.deepcopy(good)
                corrupt(value)
                with self.assertRaises(ValueError):
                    reg.validate(value, "a" * 40)

    def test_authenticated_json_rejects_duplicate_fields_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text('{"budget":12,"budget":36}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                reg._read(path, file_hash(path))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                reg._read(path, "0" * 64)

    def test_real_reference_cli_no_mutation_and_output_input_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, existing, code = registration_fixture(root)
            value = manifest(code["commit"])
            value["reference_launch"] = existing["reference_launch"]
            value["out_dir"] = str(root / "lpft-new")
            owner = root / "lpft-registration"
            owner.mkdir()
            path = owner / "plan.json"
            write_json(path, value)
            pins = {str(p): file_hash(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(reg, "REFERENCE_LAUNCH_SHA256", existing["reference_launch"]["sha256"]):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    reg.main(["--registration", str(path), "--registration-sha256", file_hash(path),
                              "--expected-commit", code["commit"]])
                result = json.loads(stream.getvalue())
                self.assertEqual(result["planned_total_updates"], 12)
                self.assertEqual(result["optimizer_updates"], 0)
                self.assertFalse(result["training_started"])
                self.assertFalse(Path(value["out_dir"]).exists())
                self.assertEqual(pins, {str(p): file_hash(p) for p in root.rglob('*') if p.is_file()})
                Path(value["out_dir"]).mkdir()
                with self.assertRaisesRegex(ValueError, "unused"):
                    reg.preflight(path, file_hash(path), expected_commit=code["commit"])
                value["out_dir"] = str(root / "reference")
                write_json(path, value)
                with self.assertRaises(ValueError):
                    reg.preflight(path, file_hash(path), expected_commit=code["commit"])
                value["out_dir"] = str(root / "lpft-another")
                write_json(path, value)
                with Path(value["reference_launch"]["path"]).open('ab') as handle:
                    handle.write(b'changed')
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    reg.preflight(path, file_hash(path), expected_commit=code["commit"])


if __name__ == "__main__":
    unittest.main()

"""Exercise the all54 gate with metadata and stand-in artifacts, without query."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import g4_light_frozen_boundary as boundary
from g4_light_full_target import POLICY as REGISTRATION_POLICY


class LightBoundaryTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.parent = Path(directory.name).resolve()
        self.root = self.parent / "light"
        self.out = self.parent / "freeze"
        self.code = {"commit": "synthetic", "dirty": False, "files": {}, "source_hash": boundary.digest({})}
        self.keys = boundary.roles()
        self.gpus = [1, 5]
        self.launch = self.parent / "reference/launch.json"
        self.inputs = self.parent / "inputs"
        self.record = self.write(self.inputs / "record.json", {"support": "synthetic"})
        self.family = self.write(self.inputs / "family.json", {"family": "synthetic"})
        self.references = {}
        for target, seed, _, mode in self.keys:
            self.references[target, seed, mode] = {
                "target": target, "seed": seed, "mode": mode, "method": boundary.METHODS[1],
                "name": f"full-{target}-{seed}-{mode}", "initial_state_sha256": "initial",
                "inputs": {"record": str(self.record), "record_sha256": boundary.file_hash(self.record),
                           "family": str(self.family), "family_sha256": boundary.file_hash(self.family),
                           "observation": str(self.inputs / "observation.pt")},
                "source_spec": {}}
        self.write(self.launch, {"code": self.code, "out_dir": str(self.launch.parent),
                                "short_acceptances": {}, "cases": list(self.references.values())})
        plan = self.write(self.parent / "analysis.json", {"design": "synthetic"})
        worker = self.write(self.parent / "worker.json", {
            "issue": 245, "status": "engineering_passed", "reference_launch_sha256": boundary.file_hash(self.launch),
            "cpu_passed": True, "gpu_passed": True, "full_budget_recovery_passed": True,
            "real_training_updates": 0, "query_labels_accessed": False, "code": self.code})
        engineering = self.write(self.parent / "engineering.json", {"proof": "synthetic"})
        self.registration = self.write(self.parent / "registration.json", {
            "policy": REGISTRATION_POLICY, "experiment_issue": 249, "producer_commit": "synthetic",
            "planned_optimizer_updates": 81000, "estimated_seconds": 43200, "gpu_ids": self.gpus,
            "out_dir": str(self.root), "reference_launch": self.reference(self.launch),
            "engineering_acceptance": self.reference(worker), "analysis_plan": self.reference(plan),
            "query_labels_during_training": False, "test_scoring_during_training": False})
        self.registration_sha = boundary.file_hash(self.registration)
        header = {"policy": boundary.ACCEPTANCE_POLICY, "registration_sha256": self.registration_sha,
                  "campaign_engineering_sha256": boundary.file_hash(engineering), "gpu_ids": self.gpus}
        self.status = {"status": "completed", "completed_cases": 54, "planned_cases": 54,
                       "planned_optimizer_updates": 81000, "errors": {}, **header}
        self.status_path = self.write(self.root / "status.json", self.status)
        inputs = self.pins(self.registration, self.launch, plan, worker, self.record, self.family)
        self.protected = {**inputs, **self.pins(engineering)}
        self.receipts = {}
        self.names = []
        for index, key in enumerate(self.keys):
            target, seed, method, mode = key
            folder = boundary.case_directory(self.root, key)
            gpu = self.gpus[index % len(self.gpus)]
            reference = self.references[target, seed, mode]
            primary = self.write(folder / "primary.pt", {"stand_in_weights": list(key)})
            sidecar = self.write(folder / "primary.pt.json", {"stand_in_sidecar": list(key)})
            result = self.write(folder / "result.json", {
                "status": "completed", "start_code": self.code, "test_scoring": False, "optimizer_steps": 1500,
                "per_dataset_updates": {target: 1500}, "selected_step": 1470,
                "settings": {"target": target, "method": method, "physical_gpu": gpu,
                             "registration_sha256": self.registration_sha,
                             "reference_case_sha256": boundary.digest(reference),
                             "expected_initial_state_sha256": reference["initial_state_sha256"]},
                "identity": {"prepared": {"reference_case": reference["name"], "protected_files": inputs}},
                "artifacts": {p.name: boundary.file_hash(p) for p in (primary, sidecar)}})
            artifacts = self.pins(result, primary, sidecar)
            receipt = {"status": "passed", "name": folder.name, "role": list(key), "physical_gpu": gpu,
                       "updates": 1500, "validation_events": 51, "selected_step": 1470,
                       "head_metrics": 6, "class_recalls": 6 * boundary.DATASETS[target][2],
                       "exact_reload_repeats": 3, "new_optimizer_updates": 0, "query_labels_accessed": False,
                       "result_sha256": boundary.file_hash(result), "primary_sha256": boundary.file_hash(primary),
                       "sidecar_sha256": boundary.file_hash(sidecar), "protected_inputs": inputs, "artifacts": artifacts}
            self.receipts[self.root / "audits" / (folder.name + ".json")] = receipt
            self.protected.update(artifacts)
            self.names.append(folder.name)
        self.accepted = {"status": "passed", "code": self.code, "cases": self.names, "optimizer_updates": 81000,
                         "query_labels_accessed": False, "protected_files": self.protected, **header}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        boundary.write_json(path, value)
        return path

    def pins(self, *paths):
        return {str(p): boundary.file_hash(p) for p in paths}

    def reference(self, path):
        return {"path": str(path), "sha256": boundary.file_hash(path)}

    def freeze(self):
        audit_pins = self.pins(*(self.write(path, value) for path, value in self.receipts.items()))
        accepted = self.write(self.root / "training-acceptance.json", {**self.accepted, "audit_receipts": audit_pins})
        with (patch("g4_light_full_target.REFERENCE_LAUNCH_SHA256", boundary.file_hash(self.launch)),
              patch.object(boundary, "select_case") as select,
              patch.object(boundary, "require_frozen_code") as frozen):
            result = boundary.freeze(self.registration, self.registration_sha, boundary.file_hash(accepted), self.out, self.code)
            select.assert_called_once()
            frozen.assert_called_once_with(self.code)
            return result

    def test_complete_metadata_freezes_every_role(self):
        before = self.protected.copy()
        result = self.freeze()
        self.assertEqual(result, json.loads((self.out / "freeze.json").read_text()))
        self.assertEqual([m["role"] for m in result["models"]], [list(k) for k in self.keys])
        self.assertEqual(result["training_optimizer_updates"], 81000)
        self.assertIs(result["query_labels_accessed"], False)
        boundary.check_files(before)

    def test_running_status_fails_before_acceptance_access(self):
        self.write(self.status_path, {**self.status, "status": "running"})
        with self.assertRaisesRegex(ValueError, "Completed training status"):
            boundary.freeze(self.registration, self.registration_sha, "unpublished", self.out, self.code)
        self.assertFalse(self.out.exists())

    def test_missing_audit_rejected(self):
        self.receipts.pop(next(iter(self.receipts)))
        with self.assertRaisesRegex(ValueError, "all registered roles"):
            self.freeze()
        self.assertFalse(self.out.exists())

    def test_audit_role_rejected_even_when_repinned(self):
        receipt = next(iter(self.receipts.values()))
        receipt["physical_gpu"] = self.gpus[1]
        with self.assertRaisesRegex(ValueError, "Independent acceptance"):
            self.freeze()
        self.assertFalse(self.out.exists())

    def test_selected_weights_changed(self):
        folder = boundary.case_directory(self.root, self.keys[0])
        (folder / "primary.pt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Pinned support/observation input changed"):
            self.freeze()
        self.assertFalse(self.out.exists())

    def test_refuses_training_output(self):
        self.out = self.root
        with self.assertRaisesRegex(ValueError, "sibling output"):
            self.freeze()
        self.assertFalse((self.root / "freeze.json").exists())


if __name__ == "__main__":
    unittest.main()

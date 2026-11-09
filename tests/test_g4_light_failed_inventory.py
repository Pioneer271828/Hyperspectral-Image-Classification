"""Failure evidence must not silently become a resumed or accepted campaign."""
import json
from pathlib import Path
import tempfile
import unittest

from g4_light_failed_inventory import EVENTS, ROLES, inventory, name_for, sha256


class FailedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "campaign"
        (self.root / "cases").mkdir(parents=True)
        (self.root / "audits").mkdir()
        self.producer = "a" * 40
        self.regpath = self.base / "registration.json"
        self.reg = {"policy": "g4_light_matched_full_target_v1", "producer_commit": self.producer,
                    "planned_optimizer_updates": 81000, "gpu_ids": [1, 5], "out_dir": str(self.root),
                    "query_labels_during_training": False, "test_scoring_during_training": False}
        for key in ("reference_launch", "engineering_acceptance", "analysis_plan"):
            p = self.base / (key + ".json")
            self.write(p, {"fixture": key})
            self.reg[key] = {"path": str(p), "sha256": sha256(p)}
        self.write(self.regpath, self.reg)
        self.regsha = sha256(self.regpath)
        self.write(self.root / "registration-owner.json", {"policy": self.reg["policy"], "registration_sha256": self.regsha})
        self.status = {"policy": "g4_light_full_multigpu_acceptance_v1", "status": "failed",
                       "registration_sha256": self.regsha, "gpu_ids": [1, 5], "planned_cases": 54,
                       "planned_optimizer_updates": 81000, "completed_cases": 1, "errors": {"5": "fixture failure"},
                       "slots": {"5": {"status": "failed", "case": name_for(ROLES[36])}}}
        self.write(self.root / "status.json", self.status)
        self.receipt = self.make_case(ROLES[0])
        self.rpath = self.root / "audits" / (self.receipt["name"] + ".json")
        self.write(self.rpath, self.receipt)
        failed = self.root / "cases" / name_for(ROLES[36])
        failed.mkdir()
        self.write(failed / "identity.json", {"partial": True})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def make_case(self, role):
        folder = self.root / "cases" / name_for(role)
        folder.mkdir()
        gpu = self.reg["gpu_ids"][ROLES.index(role) % 2]
        settings = {"registration_sha256": self.regsha, "target": role[0], "model_seed": role[1],
                    "method": role[2], "config": {"model": {"region_residual_mode": role[3]}}, "physical_gpu": gpu,
                    "optimizer_steps": 1500, "test_scoring": False}
        if role[3] != "none":
            settings["m4_mode"] = role[3]
        identity = {"settings": settings}
        history = [{"step": step, "epoch": step - 1, "val_loss": 1., "selection_score": -1.} for step in EVENTS]
        trace = [{"step": i, "dataset": role[0], "loss": {"total": 1.}, "gradient_norms": {"head": 1.},
                  "frozen_parameters_unchanged": True, "frozen_buffers_unchanged": True,
                  "adapter_changed_elements": 0 if role[2].endswith("head_only") else 1,
                  "changed_trainable_elements": 1} for i in range(1, 1501)]
        result = {"settings": settings, "identity": identity, "start_code": {"commit": self.producer, "dirty": False},
                  "status": "completed", "optimizer_steps": 1500, "per_dataset_updates": {role[0]: 1500},
                  "test_scoring": False, "selected_step": 1, "selected_validation": {"val_loss": 1.}}
        for filename in ("primary.pt", "primary.pt.json", "state-001500.pt"):
            (folder / filename).write_bytes(b"fixture, never deserialized")
        progress = {"status": "completed", "cursor": 1500, "state_file": "state-001500.pt",
                    "state_sha256": sha256(folder / "state-001500.pt"), "best_step": 1, "best_score": -1.}
        for filename, value in (("identity", identity), ("result", result), ("history", history),
                                ("trace", trace), ("progress", progress)):
            self.write(folder / (filename + ".json"), value)
        inputs = {str(self.regpath): self.regsha}
        inputs.update({self.reg[k]["path"]: self.reg[k]["sha256"] for k in
                       ("reference_launch", "engineering_acceptance", "analysis_plan")})
        return {"status": "passed", "name": name_for(role), "role": list(role), "physical_gpu": gpu,
                "updates": 1500, "validation_events": 51, "head_metrics": 6,
                "class_recalls": 132 if role[0] == "whu-honghu" else 54, "exact_reload_repeats": 3,
                "new_optimizer_updates": 0, "query_labels_accessed": False,
                "adapter_changed_steps": 0 if role[2].endswith("head_only") else 1500, "selected_step": 1,
                "protected_inputs": inputs, "artifacts": {str(p): sha256(p) for p in folder.iterdir()},
                "result_sha256": sha256(folder / "result.json"), "primary_sha256": sha256(folder / "primary.pt"),
                "sidecar_sha256": sha256(folder / "primary.pt.json")}

    def run_inventory(self, extra=()):
        return inventory(self.regpath, self.regsha, self.producer, extra)

    def repin(self):
        folder = self.root / "cases" / self.receipt["name"]
        self.receipt["artifacts"] = {str(p): sha256(p) for p in folder.iterdir()}
        self.receipt["result_sha256"] = sha256(folder / "result.json")
        self.write(self.rpath, self.receipt)

    def test_partial_inventory_is_not_global_acceptance_or_resume(self):
        before = {str(p): sha256(p) for p in self.base.rglob("*") if p.is_file()}
        out = self.run_inventory()
        self.assertEqual(out["levels"], {"coordinator_receipt_revalidated": 1, "failed_case": 1, "not_started": 52})
        self.assertEqual(out["historically_accepted_updates_revalidated"], 1500)
        self.assertEqual(out["unaccepted_budget"], 79500)
        self.assertFalse(out["resume_authorized"] or out["global_acceptance"] or out["query_labels_accessed"])
        self.assertEqual(before, {str(p): sha256(p) for p in self.base.rglob("*") if p.is_file()})

    def test_external_drain_receipt_remains_distinct(self):
        receipt = self.make_case(ROLES[3])
        p = self.base / "drain.json"
        self.write(p, receipt)
        out = self.run_inventory([(p, sha256(p))])
        self.assertEqual(out["levels"]["external_receipt_revalidated"], 1)
        self.assertEqual(out["historically_accepted_updates_revalidated"], 3000)

    def test_external_receipt_requires_explicit_pin(self):
        with self.assertRaisesRegex(ValueError, "Pinned external"):
            self.run_inventory([(self.rpath, None)])

    def test_conflicting_architecture_alias_rejected(self):
        folder = self.root / "cases" / self.receipt["name"]
        result = json.loads((folder / "result.json").read_text())
        result["settings"]["m4_mode"] = "capacity_control"
        result["identity"]["settings"] = result["settings"]
        self.write(folder / "result.json", result)
        self.write(folder / "identity.json", result["identity"])
        self.repin()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_inventory()

    def test_artifact_tampering_rejected(self):
        (self.root / "cases" / self.receipt["name"] / "primary.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.run_inventory()

    def test_mixed_registration_and_duplicate_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_inventory([(self.rpath, sha256(self.rpath))])
        self.receipt["protected_inputs"][str(self.regpath)] = "0" * 64
        self.write(self.rpath, self.receipt)
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.run_inventory()

    def test_extra_file_and_missing_state_rejected(self):
        folder = self.root / "cases" / self.receipt["name"]
        (folder / "new.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "artifact set"):
            self.run_inventory()
        (folder / "new.txt").unlink()
        (folder / "state-001500.pt").unlink()
        self.repin()
        with self.assertRaises(FileNotFoundError):
            self.run_inventory()

    def test_first_tie_and_missing_step_rejected_even_if_repinned(self):
        folder = self.root / "cases" / self.receipt["name"]
        p = folder / "progress.json"
        v = json.loads(p.read_text()); v["best_step"] = 31; self.write(p, v); self.repin()
        with self.assertRaisesRegex(ValueError, "First-minimum"):
            self.run_inventory()
        v["best_step"] = 1; self.write(p, v)
        p = folder / "trace.json"; self.write(p, json.loads(p.read_text())[:-1]); self.repin()
        with self.assertRaisesRegex(ValueError, "Missing or reordered"):
            self.run_inventory()

    def test_forged_completion_query_and_gpu_rejected(self):
        for key, val in (("updates", 1499), ("query_labels_accessed", True), ("physical_gpu", 5)):
            old = self.receipt[key]; self.receipt[key] = val; self.write(self.rpath, self.receipt)
            with self.assertRaisesRegex(ValueError, "receipt contract"):
                self.run_inventory()
            self.receipt[key] = old

    def test_running_and_global_acceptance_rejected(self):
        self.status["status"] = "running"; self.write(self.root / "status.json", self.status)
        with self.assertRaisesRegex(ValueError, "failed campaign"):
            self.run_inventory()
        self.status["status"] = "failed"; self.write(self.root / "status.json", self.status)
        self.write(self.root / "training-acceptance.json", {})
        with self.assertRaisesRegex(ValueError, "global acceptance"):
            self.run_inventory()

    def test_unaccepted_completed_case_is_not_counted(self):
        self.make_case(ROLES[1])
        out = self.run_inventory()
        self.assertEqual(out["levels"]["completed_without_receipt"], 1)
        self.assertEqual(out["historically_accepted_updates_revalidated"], 1500)

    def test_protected_input_and_external_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.run_inventory([(self.rpath, "0" * 64)])
        Path(self.reg["analysis_plan"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.run_inventory()

    def test_symlink_rejected(self):
        alias = self.base / "alias.json"
        alias.symlink_to(self.regpath)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            inventory(alias, self.regsha, self.producer)


if __name__ == "__main__":
    unittest.main()

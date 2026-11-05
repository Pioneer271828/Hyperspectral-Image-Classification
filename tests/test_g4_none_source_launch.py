"""Cohort/profile rejection and bound source initialization, resume and evidence."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from g4_none_source_launch import (STUDY_PATH, REFERENCE_LAUNCH, BoundNoneSourceRuntime, grid,
    initial_binding, launch_binding, select_bound_case)
from g4_none_ordered_source import load_study, options
from joint_training import load_plan, run_training
from loadData.dataset_registry import file_hash
from research.reproduction.m4g4b.none_source_evidence import audit_case
from test_calibrated_selection import items
from test_joint_training import PROVENANCE, SOURCE, hashes

DEVICE = os.environ.get("G4_NONE_DEVICE", "cpu")


class NoneSourceLaunchTests(unittest.TestCase):
    def test_cohort_cannot_mix_short_full_or_outputs_and_context(self):
        with tempfile.TemporaryDirectory() as temp:
            for full in (False, True):
                root = Path(temp).resolve()
                producer = {"dirty": False, "commit": "fixture"}
                launch = {"out_dir": str(root), "code": producer, "study": load_study(STUDY_PATH),
                    "study_file_sha256": file_hash(STUDY_PATH), "profile": "full" if full else "short",
                    "expected_updates": 9018 if full else 24, "execution_device": "cuda",
                    "reference_launch_sha256": REFERENCE_LAUNCH, "query_labels_accessed": False, "test_scoring": False,
                    "prepared_manifest_sha256": load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"], "cases": []}
                for held, seed in grid(full):
                    settings = options(held, seed, full=full)
                    name = held + "-s" + str(seed)
                    launch["cases"].append({"held_out": held, "seed": seed, "name": name, "settings": settings,
                        "out": str(root / "cases" / name), "source_context": {"held_out": held,
                            "sources": settings["source_tasks"], "target_payload_accessed": False},
                        "graph_binding": dict.fromkeys(settings["source_tasks"], {"Q": "fixture", "A": "fixture"}),
                        "shared_initialization_equal": True, "common_state_tensors": 1, "initial_state_sha256": "a" * 64,
                        "common_initial_sha256": dict.fromkeys(("capacity_control", "x_residual"), "a" * 64)})
                self.assertEqual(select_bound_case(launch, producer, "xuzhou", 64, full=full)["seed"], 64)
                for mutation in ("missing", "profile", "out", "context", "initial", "study"):
                    bad = copy.deepcopy(launch)
                    if mutation == "missing": bad["cases"].pop()
                    if mutation == "profile": bad["cases"][0]["settings"] = options("paviaU", 64, full=not full)
                    if mutation == "out": bad["cases"][0]["out"] = str(root / "outside-case")
                    if mutation == "context": bad["cases"][0]["source_context"]["target_payload_accessed"] = True
                    if mutation == "initial": bad["cases"][0]["initial_state_sha256"] = "b" * 64
                    if mutation == "study": bad["study_file_sha256"] = "b" * 64
                    with self.subTest(full=full, mutation=mutation), self.assertRaises(ValueError):
                        select_bound_case(bad, producer, "xuzhou", 64, full=full)
                with self.assertRaises(ValueError):
                    select_bound_case(launch, producer, "xuzhou", 64, full=not full)

    def test_real_runtime_binding_resume_evidence_and_initial_mismatch(self):
        settings = options("xuzhou")
        data = {n: i for n, i in items().items() if n in settings["source_tasks"]}
        counts = {n: i["class_num"] for n, i in data.items()}
        init = initial_binding(settings, counts)
        case = {"name": "xuzhou-s64", "seed": 64, "held_out": "xuzhou", "settings": settings,
                "graph_binding": {}, **init}
        launch = {"code": SOURCE, "profile": "short", "study_file_sha256": file_hash(STUDY_PATH),
                  "execution_device": DEVICE}
        binding = launch_binding(launch, {**case, "source_context": PROVENANCE}, "f" * 64)
        bad = copy.deepcopy(binding)
        bad["initial_state_sha256"] = "0" * 64
        from model.research_config import ResearchConfig
        with self.assertRaisesRegex(ValueError, "before first update"):
            BoundNoneSourceRuntime(bad).build_model(ResearchConfig.from_dict(settings["config"]), counts)
        runtime = BoundNoneSourceRuntime(binding)
        provenance = {**PROVENANCE, "none_source_launch": binding}
        context = SimpleNamespace(record=PROVENANCE, sources=tuple(data), items=data, class_counts=counts)
        with tempfile.TemporaryDirectory() as temp:
            case["out"] = str(Path(temp) / "case")
            run_training(settings, data, case["out"], provenance, SOURCE, device=DEVICE,
                         stop_after=2, runtime_adapter=runtime)
            result = run_training(settings, data, case["out"], provenance, SOURCE, device=DEVICE,
                                  resume=True, runtime_adapter=runtime)
            self.assertEqual(result["updates_this_call"], 6)
            before = hashes(case["out"])
            row = audit_case(launch, case, context, "f" * 64, device=DEVICE)
            self.assertEqual(row["metrics"], 12)
            self.assertEqual(row["positive_graph_gradient_steps"], 8)
            self.assertEqual(hashes(case["out"]), before)
            saved = torch.load(Path(case["out"]) / "primary.pt", map_location="cpu", weights_only=True)
            self.assertTrue(saved)


if __name__ == "__main__":
    unittest.main()

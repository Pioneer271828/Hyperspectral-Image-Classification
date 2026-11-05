"""Synthetic complete-provenance fixtures; never evidence of source training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from g4_none_ordered_source import EXPECTED, options
from g4_none_source_launch import REFERENCE_LAUNCH, STUDY_PATH, grid, launch_binding
from joint_training import load_plan
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import POLICY
from model.checkpoint_io import save_model_checkpoint
from model.research_config import ResearchConfig
from none_full_transfer import SOURCE_PRODUCER, full_none_source, initialize_none_full_target
from research_runtime import code_snapshot
from target_initialization import METHODS, domain_keys, state_id


def fixture(root):
    # Simulated receipt/catalogue for initialization tests only. Real training
    # consumes the independently accepted LXC batch, not this fixture.
    root.mkdir(parents=True)
    current = code_snapshot()
    producer = {**current, "commit": SOURCE_PRODUCER, "dirty": False, "fixture": True}
    cases = []
    for held, seed in grid(True):
        settings = options(held, seed, full=True)
        name = held + "-s" + str(seed)
        context = {"policy": "selective_acc1p_source_v1", "source_identity": "c" * 64,
                   "held_out": held, "sources": settings["source_tasks"], "target_payload_accessed": False}
        cases.append({"held_out": held, "seed": seed, "name": name, "settings": settings,
            "out": str(root / "cases" / name), "source_context": context,
            "graph_binding": {n: {"Q": "d" * 64, "A": "e" * 64} for n in settings["source_tasks"]},
            "initial_state_sha256": "a" * 64, "common_initial_sha256": dict.fromkeys(("capacity_control", "x_residual"), "a" * 64),
            "common_state_tensors": 1, "shared_initialization_equal": True})
    launch = {"code": producer, "out_dir": str(root), "profile": "full", "study": EXPECTED,
              "study_file_sha256": file_hash(STUDY_PATH), "expected_updates": 9018, "execution_device": "cuda",
              "reference_launch_sha256": REFERENCE_LAUNCH, "cases": cases, "test_scoring": False,
              "query_labels_accessed": False, "prepared_manifest_sha256": load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"]}
    def write(path, value):
        path.write_text(json.dumps(value))
    write(root / "launch.json", launch)
    case = next(c for c in cases if (c["held_out"], c["seed"]) == ("xuzhou", 64))
    folder = Path(case["out"])
    folder.mkdir(parents=True)
    source = ResearchConfig.from_dict(case["settings"]["config"]).build_model({n: DATASETS[n][2] for n in case["settings"]["source_tasks"]})
    with torch.no_grad():
        for p in source.parameters():
            p.add_(.01)
        for name, value in source.backbone.state_dict().items():
            if name in domain_keys(source.backbone):
                value.add_(.5 if value.is_floating_point() else 3)
    binding = launch_binding(launch, case, file_hash(root / "launch.json"))
    provenance = {**case["source_context"], "none_source_launch": binding}
    source.training_config = {"joint_runtime": {**case["settings"], "executed_optimizer_steps": 1002, "selected_step": 1002},
        "training_code": producer, "source_validation_arithmetic": POLICY, "none_source_launch": binding,
        "provenance": provenance, "research_config": case["settings"]["config"], "checkpoint_optimizer_step": 1002}
    save_model_checkpoint(source, folder / "primary.pt")
    result = {"status": "completed", "start_code": producer, "optimizer_steps": 1002, "selected_step": 1002,
              "settings": case["settings"], "identity": {"prepared": provenance},
              "artifacts": {n: file_hash(folder / n) for n in ("primary.pt", "primary.pt.json")}}
    write(folder / "result.json", result)
    rows = [{"name": c["name"], "updates": 1002, "selected_step": 1002, "exact_reload_repeats": 3,
             "completed_resume_updates": 0, "metrics": 12,
             "recalls": 6 * sum(DATASETS[n][2] for n in c["settings"]["source_tasks"]),
             "result_sha256": file_hash(folder / "result.json"), "primary_sha256": file_hash(folder / "primary.pt"),
             "sidecar_sha256": file_hash(folder / "primary.pt.json")} for c in cases]
    accepted = {"code": producer, "status": "passed", "optimizer_updates": 9018,
                "launch_sha256": file_hash(root / "launch.json"), "cases": rows,
                "query_labels_accessed": False, "test_scoring": False}
    write(root / "training-acceptance.json", accepted)
    write(root / "protected-files.json", {"fixture": True})
    audit = {"status": "passed", "producer": producer, "acceptance_sha256": file_hash(root / "training-acceptance.json"),
             "launch_sha256": file_hash(root / "launch.json"), "protected_manifest_sha256": file_hash(root / "protected-files.json"),
             "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False, "cases": rows}
    write(root / "independent.json", audit)
    spec = {"source_path": str(folder / "primary.pt"), "source_weight_sha256": file_hash(folder / "primary.pt"),
            "source_sidecar_sha256": file_hash(folder / "primary.pt.json")}
    for key, path in {"source_result": folder / "result.json", "source_launch": root / "launch.json",
                      "source_acceptance": root / "training-acceptance.json", "source_protected_manifest": root / "protected-files.json",
                      "source_independent_audit": root / "independent.json"}.items():
        spec[key], spec[key + "_sha256"] = str(path), file_hash(path)
    return source, spec


class NoneFullTransferTests(unittest.TestCase):
    def test_shared_copy_fresh_heads_domains_and_role_hash_barriers(self):
        with tempfile.TemporaryDirectory() as temp:
            source, spec = fixture(Path(temp) / "source")
            before = state_id(source)
            scratch, _ = initialize_none_full_target("xuzhou", METHODS[0], model_seed=64, head_seed=1064)
            model, record = initialize_none_full_target("xuzhou", METHODS[1], model_seed=64, head_seed=1064, source_spec=spec)
            fresh = domain_keys(model.backbone)
            for name, value in model.backbone.state_dict().items():
                expected = scratch.backbone.state_dict()[name] if name in fresh else source.backbone.state_dict()[name]
                self.assertTrue(torch.equal(value, expected), name)
            for family in ("heads", "hgnn_heads", "cnn_heads"):
                for name, value in getattr(model, family).state_dict().items():
                    self.assertTrue(torch.equal(value, getattr(scratch, family).state_dict()[name]))
            self.assertEqual(before, state_id(source))
            self.assertFalse(record["source_heads_copied"])
            self.assertFalse(record["source_domain_state_copied"])
            for target, seed in (("paviaU", 64), ("xuzhou", 512)):
                with self.assertRaises(ValueError): full_none_source(spec, target, seed)
            bad = copy.deepcopy(spec)
            bad["source_protected_manifest_sha256"] = "0" * 64
            with self.assertRaises(ValueError): full_none_source(bad, "xuzhou", 64)
            with self.assertRaises(ValueError):
                initialize_none_full_target("xuzhou", METHODS[0], model_seed=64, head_seed=1064, source_spec=spec)
            audit = json.loads(Path(spec["source_independent_audit"]).read_text())
            audit["new_optimizer_updates"] = 1
            other = Path(temp) / "bad-audit.json"
            other.write_text(json.dumps(audit))
            bad = {**spec, "source_independent_audit": str(other), "source_independent_audit_sha256": file_hash(other)}
            with self.assertRaises(ValueError): full_none_source(bad, "xuzhou", 64)


if __name__ == "__main__":
    unittest.main()

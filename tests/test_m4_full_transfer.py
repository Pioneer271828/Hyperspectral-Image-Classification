"""Synthetic complete-provenance fixtures for full-source initialization only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from loadData.dataset_registry import DATASETS, file_hash
from m4_full_transfer import full_source, initialize_full_target, SOURCE_PRODUCER
from m4_ordered_source import EXPECTED, POLICY, full_settings
from model.checkpoint_io import save_model_checkpoint
from model.research_config import ResearchConfig
from target_initialization import METHODS, domain_keys, state_id


def fixture(root, mode="x_residual"):
    # These files test initialization contracts; they are never experiment results.
    cases = []
    for target in EXPECTED["held_targets"]:
        for seed in EXPECTED["training_seeds"]:
            for mode in EXPECTED["modes"]:
                settings = full_settings(target, seed, mode)
                binding = {n: {"Q": "a" * 64, "A": "b" * 64} for n in settings["source_tasks"]}
                context = {"policy": "selective_acc1p_source_v1", "source_identity": "c" * 64,
                           "held_out": target, "sources": settings["source_tasks"], "target_payload_accessed": False}
                name = target + "-s" + str(seed) + "-" + mode
                cases.append({"name": name, "held_out": target, "seed": seed, "mode": mode,
                              "settings": settings, "out": str(root / "cases" / name),
                              "graph_binding": binding, "source_context": context,
                              "provenance": {**context, "m4_mode": mode, "m4_graph_binding": binding}})
    producer = {"commit": SOURCE_PRODUCER, "dirty": False, "fixture": True}
    launch = {"code": producer, "study": EXPECTED, "test_scoring": False, "expected_updates": 18036,
              "out_dir": str(root), "cases": cases}
    case = next(c for c in cases if (c["held_out"], c["seed"], c["mode"]) == ("xuzhou", 64, mode))
    folder = Path(case["out"])
    folder.mkdir(parents=True)
    source = ResearchConfig.from_dict(case["settings"]["config"]).build_model(
        {n: DATASETS[n][2] for n in case["settings"]["source_tasks"]})
    source.training_config = {"joint_runtime": {**case["settings"], "executed_optimizer_steps": 1002, "selected_step": 1002},
        "training_code": producer, "source_validation_arithmetic": POLICY, "provenance": case["provenance"],
        "research_config": case["settings"]["config"], "checkpoint_optimizer_step": 1002}
    save_model_checkpoint(source, folder / "primary.pt")
    def write(path, value):
        path.write_text(json.dumps(value))
    result = {"status": "completed", "start_code": producer, "optimizer_steps": 1002,
              "selected_step": 1002, "settings": case["settings"], "identity": {"prepared": case["provenance"]},
              "artifacts": {n: file_hash(folder / n) for n in ("primary.pt", "primary.pt.json")}}
    write(folder / "result.json", result)
    write(root / "launch.json", launch)
    accepted = {"code": producer, "status": "passed", "optimizer_updates": 18036,
                "launch_sha256": file_hash(root / "launch.json"),
                "cases": [{"name": c["name"], "result_sha256": file_hash(folder / "result.json")} for c in cases]}
    write(root / "training-acceptance.json", accepted)
    audit = {"status": "passed", "producer": producer, "acceptance_sha256": file_hash(root / "training-acceptance.json"),
             "launch_sha256": file_hash(root / "launch.json"), "cases": [{"name": c["name"]} for c in cases]}
    write(root / "independent.json", audit)
    spec = {"source_path": str(folder / "primary.pt"), "source_weight_sha256": file_hash(folder / "primary.pt"),
            "source_sidecar_sha256": file_hash(folder / "primary.pt.json")}
    for key, path in {"source_result": folder / "result.json", "source_launch": root / "launch.json",
                      "source_acceptance": root / "training-acceptance.json", "source_independent_audit": root / "independent.json"}.items():
        spec[key], spec[key + "_sha256"] = str(path), file_hash(path)
    return source, spec


class FullTransferTests(unittest.TestCase):
    def test_copy_shared_parameters_keep_fresh_heads_and_domains(self):
        with tempfile.TemporaryDirectory() as temp:
            source, spec = fixture(Path(temp))
            before = state_id(source)
            scratch, _ = initialize_full_target("xuzhou", METHODS[0], "x_residual", model_seed=64, head_seed=1064)
            target, record = initialize_full_target("xuzhou", METHODS[1], "x_residual",
                                                   model_seed=64, head_seed=1064, source_spec=spec)
            fresh = domain_keys(target.backbone)
            for name, value in target.backbone.state_dict().items():
                expected = scratch.backbone.state_dict()[name] if name in fresh else source.backbone.state_dict()[name]
                self.assertTrue(value.equal(expected), name)
            for family in ("heads", "hgnn_heads", "cnn_heads"):
                for name, value in getattr(target, family).state_dict().items():
                    self.assertTrue(value.equal(getattr(scratch, family).state_dict()[name]))
            self.assertEqual(state_id(source), before)
            self.assertEqual(record["method"], METHODS[1])
            for target_name, seed, mode in (("paviaU", 64, "x_residual"), ("xuzhou", 512, "x_residual"),
                                             ("xuzhou", 64, "capacity_control")):
                with self.assertRaises(ValueError):
                    full_source(spec, target_name, seed, mode)
            bad = copy.deepcopy(spec)
            bad["source_result_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                full_source(bad, "xuzhou", 64, "x_residual")
            with self.assertRaises(ValueError):
                initialize_full_target("xuzhou", METHODS[0], "x_residual", model_seed=64, head_seed=1064, source_spec=spec)


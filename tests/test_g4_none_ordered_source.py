"""Same none architecture and shared initialization; explicit ordered checkpoints."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from g4_none_ordered_source import EXPECTED, NoneOrderedSourceRuntime, load_study, options, select_case
from joint_baseline import NAMES
from joint_training import _tree_fingerprint, run_training, schedule_for
from loadData.dataset_registry import DATASETS
from m4_ordered_source import full_settings as active_settings
from model.research_config import ResearchConfig
from source_pretrain import full_settings as native_settings
from target_initialization import state_id
from test_calibrated_selection import items
from test_joint_training import PROVENANCE, SOURCE, checkpoints, hashes

DEVICE = os.environ.get("G4_NONE_DEVICE", "cpu")


class NoneOrderedSourceTests(unittest.TestCase):
    def test_profiles_nine_case_grid_and_unchanged_architecture_loss_schedule(self):
        self.assertEqual(load_study(), EXPECTED)
        producer = {"dirty": False, "commit": "fixture"}
        cases = []
        for target in NAMES:
            for seed in (64, 512, 2048):
                short, full = options(target, seed), options(target, seed, full=True)
                self.assertEqual(short["optimizer_steps"], 8)
                self.assertEqual(full["optimizer_steps"], 1002)
                self.assertEqual(full["config"], native_settings(target, seed)["config"])
                self.assertEqual(full["scheduler"], native_settings(target, seed)["scheduler"])
                self.assertEqual(schedule_for(full, task_names=full["source_tasks"]),
                                 schedule_for(native_settings(target, seed), task_names=full["source_tasks"]))
                self.assertNotIn(target, full["source_tasks"])
                NoneOrderedSourceRuntime().validate(full)
                cases.append({"held_out": target, "seed": seed, "settings": full, "out": target + str(seed)})
        launch = {"code": producer, "study": copy.deepcopy(EXPECTED), "cases": cases,
                  "expected_updates": 9018, "test_scoring": False}
        self.assertEqual(select_case(launch, producer, "xuzhou", 512)["seed"], 512)
        bad = copy.deepcopy(launch)
        bad["cases"].pop()
        with self.assertRaises(ValueError):
            select_case(bad, producer, "xuzhou", 512)
        for key in ("optimizer_steps", "source_validation_arithmetic", "config"):
            bad = options("xuzhou", full=True)
            if key == "config":
                bad[key]["model"]["region_residual_mode"] = "x_residual"
            else:
                bad[key] = None
            with self.assertRaises(ValueError):
                NoneOrderedSourceRuntime().validate(bad)

    def test_common_initial_parameters_match_both_active_source_controls_all_nine_roles(self):
        for target in NAMES:
            for seed in (64, 512, 2048):
                opts = options(target, seed, full=True)
                counts = {n: DATASETS[n][2] for n in opts["source_tasks"]}
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    baseline = ResearchConfig.from_dict(opts["config"]).build_model(counts)
                common = baseline.state_dict()
                for mode in ("capacity_control", "x_residual"):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(seed)
                        active = ResearchConfig.from_dict(active_settings(target, seed, mode)["config"]).build_model(counts)
                    state = active.state_dict()
                    self.assertTrue(all(k in state and torch.equal(v, state[k]) for k, v in common.items()))
                    del active
                del baseline

    def test_synthetic_eight_step_resume_exact_reload_and_readonly_completion(self):
        opts = options("xuzhou")
        data = {n: i for n, i in items().items() if n in opts["source_tasks"]}
        moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()}
                 for n, item in data.items()}
        runtime = NoneOrderedSourceRuntime()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full, split = root / "full", root / "split"
            result = run_training(opts, data, full, PROVENANCE, SOURCE, device=DEVICE, runtime_adapter=runtime)["result"]
            run_training(opts, data, split, PROVENANCE, SOURCE, device=DEVICE, runtime_adapter=runtime, stop_after=3)
            resumed = run_training(opts, data, split, PROVENANCE, SOURCE, device=DEVICE, runtime_adapter=runtime, resume=True)
            self.assertEqual(resumed["updates_this_call"], 5)
            for key in ("model", "optimizer", "scheduler", "rng", "trace", "history", "best_model", "best_probabilities"):
                self.assertEqual(_tree_fingerprint(checkpoints(full)[key]), _tree_fingerprint(checkpoints(split)[key]), key)
            before = hashes(split)
            self.assertEqual(run_training(opts, data, split, PROVENANCE, SOURCE, device=DEVICE,
                runtime_adapter=runtime, resume=True)["updates_this_call"], 0)
            self.assertEqual(before, hashes(split))
            for _ in range(3):
                model = runtime.load_selected(full / "primary.pt", moved,
                    full / f'raw-validation-{result["selected_step"]:06d}.pt', device=DEVICE)
                original = state_id(model)
                values = runtime.validation_snapshot(model, moved)[1]
                self.assertEqual(_tree_fingerprint(values), _tree_fingerprint(checkpoints(full)["best_probabilities"]))
                self.assertEqual(original, state_id(model))
            self.assertTrue(all(row["gradient_norms"]["graph"] > 0 and row["graph_changed_elements"] > 0
                                for row in checkpoints(full)["trace"]))


if __name__ == "__main__":
    unittest.main()

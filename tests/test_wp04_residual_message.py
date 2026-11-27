"""Verify the new composition, selected loader and persisted bounded recovery."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from g4_residual_message_target import ARMS, KEY, MESSAGE, RESIDUAL, SEEDS, VERSION, initialize, prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from target_initialization import state_id
from test_g4_branch_local_target_runtime import fixture
from test_joint_training import checkpoints


class ResidualMessageTests(unittest.TestCase):
    def test_exact_cap_and_existing_factory_initialization(self):
        torch.set_num_threads(1)
        for seed in SEEDS:
            rng = torch.get_rng_state().clone()
            models = {arm: initialize(arm, model_seed=seed, head_seed=seed+1000) for arm in ARMS}
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            bindings = {arm: binding for arm, (_,binding) in models.items()}
            for key in ("shared_sha256", "message_sha256"):
                self.assertEqual(len({binding[key] for binding in bindings.values()}), 1)
            for prefix in ("R","Bcap"):
                self.assertEqual(bindings[prefix+"-cross"]["residual_sha256"], bindings[prefix+"-local"]["residual_sha256"])
            for arm, (model, _) in models.items():
                self.assertEqual(sum(p.numel() for p in model.parameters()), 128450)
                self.assertEqual(VERSION in model.state_dict(), arm.endswith("-local"))
                self.assertEqual(model.backbone.graph_mode, "static")
                self.assertIsNone(model.backbone.edge_updater)

    def test_twelve_cases_restore_and_compound_checkpoint_refuses_wrong_arm(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, _ = fixture(root)
            checksum = file_hash(manifest)
            for seed in SEEDS:
                for arm in ARMS:
                    case = root / f"{seed}-{arm}"
                    a,b = case / "whole", case / "split"

                    def bounded(directory):
                        return prepare(manifest, checksum, directory, arm=arm, model_seed=seed)

                    def launch(directory, **kwargs):
                        opt, items, provenance, code, runtime, guard = bounded(directory)
                        return run_training(opt, items, directory, provenance, code, device="cpu",
                                            runtime_adapter=runtime, finish_guard=guard, **kwargs)

                    whole = launch(a)
                    self.assertEqual(whole["updates_this_call"], 12)
                    self.assertEqual(launch(b, stop_after=2)["updates_this_call"], 2)
                    self.assertEqual(launch(b, resume=True)["updates_this_call"], 10)
                    self.assertEqual(launch(b, resume=True)["updates_this_call"], 0)
                    x,y = checkpoints(a),checkpoints(b)
                    for key in ("model","optimizer","scheduler","rng","trace","history","best_probabilities"):
                        self.assertEqual(_tree_fingerprint(x[key]), _tree_fingerprint(y[key]), (seed,arm,key))
                    for prefix in (MESSAGE, RESIDUAL):
                        self.assertGreater(max(r[KEY][prefix]["gradient_norm"] for r in x["trace"]), 0)
                        self.assertGreater(sum(r[KEY][prefix]["changed_elements"] for r in x["trace"]), 0)
                    self.assertEqual(whole["result"]["selected_probability_max_abs_error"], 0)
                    self.assertEqual(x["best_step"], min(x["history"], key=lambda row:row["val_loss"])["step"])
            selected = root / "64-R-cross" / "whole"
            _,items,_,_,runtime,_ = prepare(manifest, checksum, root / "unused", arm="R-cross", model_seed=64)
            step = checkpoints(selected)["best_step"]
            path = selected / "primary.pt"
            raw = selected / f"raw-validation-{step:06d}.pt"
            model = runtime.load_selected(path, items, raw)
            self.assertEqual(state_id(model), model.training_config["target_bn_calibration"]["state_sha256"])
            _,items,_,_,wrong,_ = prepare(manifest, checksum, root / "unused", arm="Bcap-local", model_seed=64)
            with self.assertRaises(ValueError):
                wrong.load_selected(path, items, raw)
            invalid = copy.deepcopy(runtime.settings)
            invalid["optimizer_steps"] = 13
            with self.assertRaises(ValueError):
                runtime.validate(invalid)


if __name__ == "__main__":
    unittest.main()

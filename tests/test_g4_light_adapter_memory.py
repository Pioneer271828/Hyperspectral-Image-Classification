"""Numerical and recovery checks for the opt-in light-target memory policy."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import g4_light_adapter_memory as memory
from g4_light_memory_probe import backward_snapshot
from g4_light_target import ADAPTER_HEADS, HEAD_ONLY, LightTargetRuntime, apply_parameters, prepare
from g4_support_target import METHODS
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from target_training import target_settings
from test_g4_support_context import prepared
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as none_source


def honghu_fixture(device):
    """Compact synthetic scene with HongHu bands, heads and base97 backbone."""
    target, bands, rows, columns, regions, classes = "whu-honghu", 270, 12, 24, 6, 22
    config = ResearchConfig.from_dict(target_settings(target, METHODS[1], device=device)["config"])
    config = replace(config, model=replace(config.model, adapter_chunk_pixels=64))
    torch.manual_seed(64)
    model = config.build_model({target: classes}).to(device)
    item = {
        "bands": bands,
        "net_input": torch.randn(rows, columns, bands + model.backbone.extra_channels, device=device),
        "Q": torch.nn.functional.one_hot(torch.arange(rows * columns, device=device) % regions, regions).float(),
        "A": torch.eye(regions, device=device),
        "wavelengths": None,
        "fwhm": None,
        "superpixel_scale": 100,
        "train_indices": torch.arange(classes, device=device) * 7,
        "train_labels": torch.arange(classes, device=device),
    }
    return model, item, config


def paired_backward(device, *, trace=False, include_offload=False):
    model, item, config = honghu_fixture(device)
    runtime = LightTargetRuntime.__new__(LightTargetRuntime)
    runtime.settings = {"target": "whu-honghu", "method": ADAPTER_HEADS, "optimizer_steps": 12}
    runtime.parameter_contract = apply_parameters(model, ADAPTER_HEADS)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    reports, states = {}, {}
    variants = ["native", "native_repeat", "recompute"]
    if include_offload:
        variants.append("cpu_offload")
    for name in variants:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        if devices:
            torch.cuda.empty_cache()
        optimizer = runtime.optimizer(model, config)
        with torch.random.fork_rng(devices=devices):
            report = backward_snapshot(runtime, model, item, config, offload=name == "cpu_offload",
                                       recompute=name == "recompute", capture_projection=trace)
            rng = [torch.get_rng_state(), *[torch.cuda.get_rng_state(card) for card in devices]]
        gradients = {key: p.grad.detach().cpu().clone() for key, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        frozen = {key: model.state_dict()[key] for key in initial if key not in gradients}
        if any(not torch.equal(value, initial[key]) for key, value in frozen.items()):
            raise AssertionError("The optimizer changed frozen state")
        updated = {key: p.detach().cpu().clone() for key, p in model.named_parameters() if key in gradients}
        states[name] = (gradients, updated, _tree_fingerprint(optimizer.state_dict()), _tree_fingerprint(rng))
        if name != "native":
            native = states["native"]
            report.update(
                gradient_max_abs=max(float((value - native[0][key]).abs().max()) for key, value in gradients.items()),
                update_max_abs=max(float((value - native[1][key]).abs().max()) for key, value in updated.items()),
                adam_equal=states[name][2] == native[2], rng_equal=states[name][3] == native[3],
                outputs_equal=report["output_sha256"] == reports["native"]["output_sha256"],
                losses_equal=report["losses"] == reports["native"]["losses"],
            )
        reports[name] = report
        del optimizer
    return reports


class AdapterMemoryTests(unittest.TestCase):
    def test_honghu_geometry_save_reduction_and_one_update(self):
        rows = paired_backward("cpu", trace=True)
        recompute = rows["recompute"]
        self.assertEqual(recompute["gradient_max_abs"], 0)
        self.assertEqual(recompute["update_max_abs"], 0)
        for key in ("adam_equal", "rng_equal", "outputs_equal", "losses_equal"):
            self.assertTrue(recompute[key], key)
        self.assertGreater(rows["native"]["saved_value_hidden_storage_bytes"], 0)
        self.assertEqual(recompute["saved_value_hidden_storage_bytes"], 0)
        self.assertIsNone(recompute["peak_cuda_allocated_bytes"])
        self.assertIsNone(recompute["peak_cuda_reserved_bytes"])

    def test_chunk_boundaries_no_grad_and_exception_cleanup(self):
        model, _, _ = honghu_fixture("cpu")
        adapter = model.backbone.adapter.eval()
        for size, chunks in ((63, 1), (64, 1), (65, 2), (128, 2), (129, 3)):
            with self.subTest(size=size):
                source = torch.randn(size, 270)
                attention = torch.randn(adapter.latent_channels, 270)
                expected = adapter._project_to_latent_chunked(source, attention)
                with patch.object(memory, "checkpoint", wraps=memory.checkpoint) as checkpoint:
                    with memory.recompute_value_projection(adapter):
                        actual = adapter._project_to_latent_chunked(source, attention)
                        with torch.no_grad():
                            self.assertTrue(torch.equal(expected, adapter._project_to_latent_chunked(source, attention)))
                    self.assertEqual(checkpoint.call_count, chunks)
                self.assertTrue(torch.equal(expected, actual))
                self.assertNotIn("_project_to_latent", adapter.__dict__)
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with memory.recompute_value_projection(adapter):
                raise RuntimeError("test exception")
        self.assertNotIn("_project_to_latent", adapter.__dict__)

    def test_policy_binding_and_full_split_completed_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, catalog, _ = prepared(root)
            manifest = catalog / "manifest.json"
            _, spec = none_source(root / "source")

            def setup(name, *, recompute, method=ADAPTER_HEADS):
                return prepare(manifest, file_hash(manifest), root / name, target="xuzhou", method=method,
                               mode="none", model_seed=64, support_seed=512, per_class=1, source_spec=spec,
                               adapter_recompute=recompute)

            def run(name, *, recompute=True, **kwargs):
                opts, items, provenance, code, runtime, guard = setup(name, recompute=recompute)
                return run_training(opts, items, root / name, provenance, code, runtime_adapter=runtime,
                                    finish_guard=guard, **kwargs)

            with self.assertRaisesRegex(ValueError, "adapter"):
                setup("head", recompute=True, method=HEAD_ONLY)
            self.assertEqual(run("native", recompute=False)["updates_this_call"], 12)
            full = run("full")["result"]
            self.assertEqual(run("split", stop_after=2)["updates_this_call"], 2)
            partial = hashes(root / "split")
            with self.assertRaisesRegex(ValueError, "Resume identity"):
                run("split", recompute=False, resume=True)
            self.assertEqual(hashes(root / "split"), partial)
            split = run("split", resume=True)
            self.assertEqual(split["updates_this_call"], 10)
            states = [checkpoints(root / name) for name in ("native", "full", "split")]
            for key in ("model", "optimizer", "scheduler", "rng", "trace", "history", "best_probabilities"):
                self.assertEqual(len({_tree_fingerprint(state[key]) for state in states}), 1, key)
            self.assertEqual(full["selected_step"], split["result"]["selected_step"])
            history = states[-1]["history"]
            self.assertEqual([event["step"] for event in history], [1, 4, 7, 10, 12])
            self.assertEqual(full["selected_step"], min(history, key=lambda event: event["val_loss"])["step"])
            self.assertTrue(all(row["frozen_parameters_unchanged"] and row["frozen_buffers_unchanged"]
                                and row["adapter_changed_elements"] > 0 for row in states[-1]["trace"]))
            self.assertFalse((root / "split" / "feature-bank.pt").exists())
            opts, items, _, _, runtime, _ = setup("split", recompute=True)
            self.assertEqual(opts["adapter_backward_policy"], memory.RECOMPUTE)
            artifact = root / "split"
            raw = artifact / f"raw-validation-{full['selected_step']:06d}.pt"
            reloaded = runtime.load_selected(artifact / "primary.pt", items, raw)
            self.assertEqual(reloaded.training_config["adapter_backward_policy"], memory.RECOMPUTE)
            before = hashes(artifact)
            self.assertEqual(run("split", resume=True)["updates_this_call"], 0)
            self.assertEqual(hashes(artifact), before)


if __name__ == "__main__":
    unittest.main()

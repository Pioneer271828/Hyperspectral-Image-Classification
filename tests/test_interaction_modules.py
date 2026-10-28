"""CPU regression and integration checks; no real dataset or long training run needed."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCN
from model.region_pixel_alignment import RegionPixelAlignment, pool_regions, broadcast_regions
from model.cross_branch_interaction import BidirectionalSelectiveInteraction, SelectiveMessage
from model.graph_edge_update import GraphEdgeUpdater
from model.interaction_config import add_interaction_arguments, interaction_kwargs
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint, dataset_seed
from interaction_ablation_presets import PRESETS, build_interaction_experiments


torch.set_num_threads(1)


def make_model(graph_mode="learned", **kwargs):
    return MultiDatasetVCASAHGCN(
        {"demo": 3, "other": 5}, latent_channels=8, adapter_embed_dim=8,
        locality_window=None, norm_type="group_norm", graph_mode=graph_mode, **kwargs,
    )


def make_input(bands=12):
    return torch.randn(6, 7, bands + 2), torch.linspace(450, 950, bands)


def static_graph():
    ids = torch.arange(42) % 6
    q = torch.nn.functional.one_hot(ids, 6).float().to_sparse().coalesce()
    a = (torch.eye(6) + torch.roll(torch.eye(6), 1, 1) + torch.roll(torch.eye(6), -1, 1)) / 3
    return q, a


def synthetic_item(name="demo", graph_mode="learned"):
    x, wavelength = make_input()
    q, a = static_graph() if graph_mode == "static" else (None, None)
    item = dict(name=name, height=6, width=7, bands=12, class_num=3, net_input=x,
                wavelengths=wavelength, fwhm=None, Q=q, A=a, superpixel_scale=9, zeros=torch.zeros(42))
    labels = torch.arange(42) % 3 + 1
    for split, start, end in (("train", 0, 12), ("val", 12, 24), ("test", 24, 42)):
        selected = torch.zeros(42, dtype=torch.bool)
        selected[start:end] = True
        item[split + "_samples_gt"] = labels * selected
        item[split + "_gt_onehot"] = torch.nn.functional.one_hot(labels - 1, 3).float() * selected[:, None]
        item[split + "_label_mask"] = selected[:, None].expand(-1, 3).float()
    return item


class AlignmentTests(unittest.TestCase):
    def test_dense_sparse_pool_and_empty_region(self):
        q = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        x = torch.tensor([[1., 2.], [3., 4.], [8., 9.]])
        expected = torch.tensor([[2., 3.], [8., 9.], [0., 0.]])
        for assignment in (q, q.to_sparse()):
            torch.testing.assert_close(pool_regions(assignment, x), expected)
            torch.testing.assert_close(broadcast_regions(assignment, expected), expected[[0, 0, 1]])

    def test_soft_assignment_gradient_and_constant_preservation(self):
        logits = torch.randn(7, 3, requires_grad=True)
        q = logits.softmax(-1).to_sparse()
        x = torch.randn(7, 4, requires_grad=True)
        y = broadcast_regions(q, pool_regions(q, x))
        y.square().sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(broadcast_regions(q, pool_regions(q, torch.ones(7, 4))), torch.ones(7, 4))

    def test_global_control_is_independent_of_correspondence(self):
        q, _ = static_graph()
        pixels, regions = torch.randn(42, 4), torch.randn(6, 4)
        align = RegionPixelAlignment("global")
        a, b = align(pixels, regions, q)
        torch.testing.assert_close(a, regions.mean(0).expand(42, -1))
        torch.testing.assert_close(b, pixels.mean(0).expand(6, -1))


class InteractionTests(unittest.TestCase):
    def test_none_and_zero_strength_are_identity(self):
        p, r = torch.randn(7, 4), torch.randn(3, 4)
        for mode, strength in (("none", 0.1), ("gated", 0.0)):
            module = BidirectionalSelectiveInteraction(4, mode, strength=strength)
            out_p, out_r, _ = module(p, r, torch.randn_like(p), torch.randn_like(r))
            torch.testing.assert_close(out_p, p, rtol=0, atol=0)
            torch.testing.assert_close(out_r, r, rtol=0, atol=0)

    def test_direction_and_gate_gradients(self):
        for direction in ("cnn_to_graph", "graph_to_cnn", "both"):
            with self.subTest(direction=direction):
                module = BidirectionalSelectiveInteraction(4, "gated", direction, chunk_size=2)
                p, r = torch.randn(7, 4), torch.randn(3, 4)
                out_p, out_r, info = module(p, r, torch.randn_like(p), torch.randn_like(r), True, True)
                (out_p.square().sum() + out_r.square().sum()).backward()
                if direction == "cnn_to_graph":
                    torch.testing.assert_close(out_p, p, rtol=0, atol=0)
                if direction == "graph_to_cnn":
                    torch.testing.assert_close(out_r, r, rtol=0, atol=0)
                for name, value in module.named_parameters():
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
                for key, value in info.items():
                    if key.endswith("_gate"):
                        self.assertTrue(((value >= 0) & (value <= 1)).all())

    def test_chunked_checkpoint_matches_outputs_and_gradients(self):
        unchunked = SelectiveMessage(4, 3, True, 0.1, 0)
        chunked = SelectiveMessage(4, 3, True, 0.1, 2)
        chunked.load_state_dict(unchunked.state_dict())
        p, r = torch.randn(7, 4), torch.randn(7, 4)
        a, ag = unchunked(p, r)
        b, bg = chunked(p, r)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(ag, bg)
        a.square().sum().backward()
        b.square().sum().backward()
        for pa, pb in zip(unchunked.parameters(), chunked.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-6)


class EdgeUpdateTests(unittest.TestCase):
    def test_sparse_support_normalization_and_gradients(self):
        _, a = static_graph()
        for mode in ("graph", "joint"):
            module = GraphEdgeUpdater(4, mode, 3)
            regions = torch.randn(6, 4, requires_grad=True)
            pixels = torch.randn(6, 4, requires_grad=True)
            updated, info = module(regions, a.to_sparse(), pixels, True, True)
            torch.testing.assert_close(updated.indices(), a.to_sparse().coalesce().indices())
            torch.testing.assert_close(torch.sparse.sum(updated, 1).to_dense(), torch.ones(6))
            self.assertTrue((updated.values() >= 0).all())
            (updated.values() * torch.arange(updated._nnz())).sum().backward()
            self.assertGreater(module.query.weight.grad.abs().sum().item(), 0)
            self.assertGreater(regions.grad.abs().sum().item(), 0)
            if mode == "joint":
                self.assertGreater(pixels.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(pixels.grad)

    def test_isolated_nodes_and_zero_mix(self):
        module = GraphEdgeUpdater(4, "graph", mix=0)
        a = torch.tensor([[2., 1., 0.], [1., 2., 0.], [0., 0., 0.]])
        updated, _ = module(torch.randn(3, 4), a)
        torch.testing.assert_close(updated.to_dense(), torch.tensor([[2/3, 1/3, 0.], [1/3, 2/3, 0.], [0., 0., 1.]]))
        single, _ = module(torch.randn(1, 4), torch.ones(1, 1))
        torch.testing.assert_close(single.to_dense(), torch.ones(1, 1))

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            GraphEdgeUpdater(4, "graph")(torch.randn(2, 4), -torch.eye(2))


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_all_presets_static_learned_variable_bands_backward(self):
        for graph_mode in ("static", "learned"):
            for name, overrides in PRESETS.items():
                with self.subTest(graph=graph_mode, preset=name):
                    model = make_model(graph_mode, interaction_chunk_pixels=11, **overrides).train()
                    q, a = static_graph() if graph_mode == "static" else (None, None)
                    for dataset, bands, classes in (("demo", 12, 3), ("other", 17, 5)):
                        x, wavelengths = make_input(bands)
                        outputs = model(dataset, x, q, a, wavelengths=wavelengths, superpixel_scale=9, return_graph_info=True)
                        for y in outputs[:3]:
                            self.assertEqual(y.shape, (42, classes))
                            self.assertTrue(torch.isfinite(y).all())
                            torch.testing.assert_close(y.sum(-1), torch.ones(42))
                        loss = sum(-y[:, 0].clamp_min(1e-8).log().mean() for y in outputs[:3])
                        model.zero_grad(set_to_none=True)
                        loss.backward()
                        for module_name in ("interaction", "edge_updater", "learnable_graph"):
                            module = getattr(model.backbone, module_name)
                            if module is not None:
                                gradients = [p.grad for p in module.parameters()]
                                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), module_name)
                                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, module_name)

    def test_default_keys_and_initialization_preserved(self):
        torch.manual_seed(9)
        baseline = make_model()
        torch.manual_seed(9)
        full = make_model(interaction_mode="gated", graph_update_mode="joint")
        self.assertFalse(any("interaction." in key or "edge_updater." in key for key in baseline.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], rtol=0, atol=0)

    def test_zero_interaction_matches_original_forward(self):
        for graph_mode in ("static", "learned"):
            baseline = make_model(graph_mode).eval()
            staged = make_model(graph_mode, interaction_mode="gated", interaction_strength=0).eval()
            result = staged.load_state_dict(baseline.state_dict(), strict=False)
            self.assertFalse(result.unexpected_keys)
            self.assertTrue(all(key.startswith("backbone.interaction.") for key in result.missing_keys))
            q, a = static_graph() if graph_mode == "static" else (None, None)
            x, wavelength = make_input()
            with torch.no_grad():
                before = baseline("demo", x, q, a, wavelengths=wavelength, superpixel_scale=9)
                after = staged("demo", x, q, a, wavelengths=wavelength, superpixel_scale=9)
            for expected, actual in zip(before, after):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_legacy_loading(self):
        x, wavelength = make_input()
        for overrides in ({}, dict(interaction_mode="gated", graph_update_mode="joint", alignment_mode="global")):
            model = make_model(**overrides).eval()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.pt"
                save_model_checkpoint(model, path)
                loaded = load_model_checkpoint(path)
                self.assertEqual(model.export_config(), loaded.export_config())
                for a, b in zip(model("demo", x, wavelengths=wavelength, superpixel_scale=9),
                                loaded("demo", x, wavelengths=wavelength, superpixel_scale=9)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                legacy_path = Path(folder) / "legacy.pt"
                torch.save(model.state_dict(), legacy_path)
                legacy = load_model_checkpoint(legacy_path, config=model.backbone_kwargs)
                self.assertEqual(list(legacy.heads), ["demo", "other"])
                if overrides:
                    with self.assertRaises(RuntimeError):
                        load_model_checkpoint(legacy_path, config={**model.backbone_kwargs, "interaction_mode": "none"})

    def test_single_dataset_wrapper_and_domainbn(self):
        model = VCASAHGCN(6, 7, 14, 3, None, None, 0.8, latent_channels=8, adapter_embed_dim=8,
                          graph_mode="learned", norm_type="domain_bn", interaction_mode="gated",
                          graph_update_mode="joint", superpixel_scale=9)
        x, _ = make_input()
        y = model(x, return_graph_info=True, return_diagnostics=True)
        self.assertEqual(y[0].shape, (42, 3))
        self.assertIn("edge_weights_after", y[3]["interaction"])
        (-y[0][:, 0].clamp_min(1e-8).log().mean()).backward()


class PipelineTests(unittest.TestCase):
    def test_cli_prefix_and_presets(self):
        parser = argparse.ArgumentParser()
        add_interaction_arguments(parser, prefix="vca-")
        args = parser.parse_args(["--vca-interaction-mode", "gated", "--vca-graph-update-mode", "joint"])
        self.assertEqual(interaction_kwargs(args, "vca_")["graph_update_mode"], "joint")
        args = argparse.Namespace(base_latent_channels=8, base_adapter_embed_dim=8,
                                  interaction_norm_type="group_norm", interaction_graph_mode="learned")
        experiments = build_interaction_experiments(args)
        self.assertEqual(len(experiments), len(PRESETS))
        self.assertEqual(len({item["name"] for item in experiments}), len(PRESETS))

    def test_fixed_count_splits_are_disjoint(self):
        from loadData.split_data import split_data
        gt = np.repeat([1, 2, 3], 30)
        train, val, test = split_data(gt, 3, .1, .1, 5, 4, "number")
        self.assertEqual((len(train), len(val), len(test)), (15, 12, 63))
        self.assertFalse(set(train) & set(val) or set(val) & set(test) or set(train) & set(test))

    def test_recorded_dataset_seed_survives_skipped_domains(self):
        config = {"seed": 10, "dataset_seeds": {"demo": 12}}
        self.assertEqual(dataset_seed(config, "demo", ["demo"]), 12)
        self.assertEqual(dataset_seed({"seed": 10}, "other", ["demo", "other"]), 11)

    def test_synthetic_train_save_reload_and_export(self):
        import train_vca_multidataset as train
        from export_interaction_diagnostics import export_dataset
        from run_paper_ablation import collect_rows, summarize_rows
        for graph_mode in ("learned", "static"):
            with self.subTest(graph=graph_mode), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "full.pt"
                metrics_path = Path(folder) / "metrics.json"
                item = synthetic_item(graph_mode=graph_mode)
                argv = ["train", "--device", "cpu", "--datasets", "demo", "--epochs", "0",
                        "--steps-per-epoch", "1", "--val-interval", "1", "--latent-channels", "8",
                        "--adapter-embed-dim", "8", "--norm-type", "group_norm", "--graph-mode", graph_mode,
                        "--interaction-mode", "gated", "--graph-update-mode", "joint", "--no-progress",
                        "--save-path", str(path), "--metrics-json", str(metrics_path)]
                with patch("sys.argv", argv), patch.object(train, "prepare_dataset", return_value=item), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["config"]["interaction_mode"], "gated")
                self.assertIn("edge_update_l1", payload["interaction_diagnostics"]["demo"])
                self.assertTrue(summarize_rows(collect_rows(Path(folder))))
                loaded = load_model_checkpoint(path)
                from export_best_classification_maps import build_model as build_best
                from export_table_palette_classification_maps import build_model as build_palette
                for restored in (build_best(payload, torch.device("cpu")),
                                 build_palette({"demo": 3}, {"checkpoint": str(path)}, torch.device("cpu"))):
                    self.assertEqual(restored.export_config(), loaded.export_config())
                export_dataset(loaded, "demo", item, Path(folder) / "diagnostics")
                with np.load(Path(folder) / "diagnostics" / "interaction_arrays.npz") as arrays:
                    self.assertEqual(arrays["prediction"].shape, (6, 7))
                    self.assertIn("edge_weights_after", arrays.files)
                self.assertTrue((Path(folder) / "diagnostics" / "graph_to_cnn_gate.png").exists())


if __name__ == "__main__":
    unittest.main()

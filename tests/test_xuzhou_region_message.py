"""Matched message weights, independent operators and registered spatial recovery."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from tests.test_curriculum_exposure import freeze
from tests.test_int8_model import fixture
from tests.test_region_message_attention import identity_attention
from transfer.development_training import prepare_scene
from transfer.g3_training import _forward, state_hash
from transfer.region_message_attention import ATTENTION_MODES
from transfer.region_message_run import DIRECTIONS, RegionMessageRun, initialize, initialization_evidence
from transfer.xuzhou_multiscale import execute_stage
from transfer.xuzhou_multiscale_audit import _exact
from transfer import xuzhou_region_message as campaign


def spatial_fixture(classes=9):
    _, prior, _ = fixture()
    plan = make_tile_plan((84, 84), 28, 1, 2026)
    labels = np.resize(np.arange(1, classes+1), (84, 84))
    record = make_development(labels, plan, list(range(1, classes+1)), train_per_class=5)
    cube = np.random.default_rng(12).normal(size=(84, 84, 8)).astype(np.float32)
    scene = prepare_scene(cube, prior.bands, plan, record)
    torch.manual_seed(64)
    base = MultiDatasetVCASAHGCN({scene.dataset: classes}, latent_channels=8, adapter_embed_dim=8,
                                locality_window=None, graph_mode="learned", norm_type="group_norm").eval()
    base.backbone.adapter.cache_attention = False
    return base, scene, record


def _reference(layer, target, source, valid, cells):
    ids = valid.nonzero().flatten()
    q = target[ids] @ layer.query.weight.T
    k = source[ids] @ layer.key.weight.T
    v = source[ids] @ layer.value.weight.T
    groups = sorted(set(cells))
    landmarks = torch.stack([q[[j for j, value in enumerate(cells) if value == group]].mean(0) for group in groups])
    probe = torch.stack([torch.stack([left.dot(right) * layer.scale for right in k]) for left in landmarks])
    compression = torch.stack([(weights.softmax(0).unsqueeze(0) @ v).squeeze(0) for weights in probe])
    rows = []
    for i, query in enumerate(q):
        if layer.mode == "local":
            row = (query + k[i]).tanh() * v[i]
        else:
            selected_keys, selected_values = k, v
            if layer.mode == "compress":
                selected_keys, selected_values = landmarks, compression
            if layer.mode in ("route", "combined"):
                chosen = max(range(len(groups)), key=lambda j: float(query.detach().dot(landmarks[j].detach())))
                order = sorted(range(len(ids)), key=lambda j: (-float(probe[chosen, j].detach()), j))
                selected = order[:min(len(ids), layer.top_k)]
                selected_keys, selected_values = k[selected], v[selected]
                if layer.mode == "combined":
                    selected_keys = torch.cat((landmarks, selected_keys))
                    selected_values = torch.cat((compression, selected_values))
            score = torch.stack([query.dot(key) * layer.scale for key in selected_keys])
            row = sum(weight * value for weight, value in zip(score.softmax(0), selected_values, strict=True))
        rows.append(row @ layer.output.weight.T)
    return target.new_zeros(target.shape).index_copy(0, ids, torch.stack(rows))


class XuzhouRegionMessageTests(unittest.TestCase):
    def test_all_modes_independent_output_gradient_and_chunking(self):
        torch.manual_seed(94)
        target = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
        source = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
        positions = torch.tensor([[.1,.1],[.3,.2],[.5,.8],[.8,.8],[.9,.7]], dtype=torch.float64)
        valid = torch.tensor([True, True, False, True, True])
        for mode in ATTENTION_MODES:
            with self.subTest(mode=mode):
                layer = identity_attention(mode, top_k=3, chunk=1)
                inputs = (target, source, *layer.parameters())
                actual = layer(target, source, positions, valid)
                reference = _reference(layer, target, source, valid, [0,0,1,1])
                torch.testing.assert_close(actual, reference)
                actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
                reference_grad = torch.autograd.grad(reference.square().sum(), inputs)
                for a, b in zip(actual_grad, reference_grad, strict=True):
                    torch.testing.assert_close(a, b)
                layer.query_chunk = 16
                wide = layer(target, source, positions, valid)
                torch.testing.assert_close(actual, wide)
                for a, b in zip(actual_grad, torch.autograd.grad(wide.square().sum(), inputs), strict=True):
                    torch.testing.assert_close(a, b)

    def test_equal_scores_masked_and_invalid_inputs(self):
        layer = identity_attention("route", top_k=2)
        with torch.no_grad():
            layer.query.weight.zero_()
            layer.key.weight.zero_()
        values = torch.tensor([[1., 1.], [3., 3.], [9., 9.]], dtype=torch.double)
        positions = torch.tensor([[.1,.1],[.5,.5],[.9,.9]], dtype=torch.double)
        valid = torch.ones(3, dtype=torch.bool)
        trace = {}
        expected = torch.full_like(values, 2.)
        torch.testing.assert_close(layer(values, values, positions, valid, trace=trace), expected)
        self.assertEqual((trace["k"], trace["raw_query_kv_pairs"], trace["landmarks"]), (2,6,2))
        for bad in ((values.float(), values, positions, valid), (values, values, positions, valid.float()),
                    (values, values, positions+1, valid), (values, values*float("nan"), positions, valid)):
            with self.assertRaises(ValueError):
                layer(*bad)
        for mode in ATTENTION_MODES:
            torch.testing.assert_close(identity_attention(mode)(values, values, positions, ~valid),
                                       torch.zeros_like(values))

    def test_mapped_weights_simultaneous_exchange_and_block_isolation(self):
        base, scene, record = spatial_fixture()
        aligned = initialize(base, "aligned", 64)
        mapped = initialization_evidence(aligned)
        attention = [initialize(base, mode, 64) for mode in ATTENTION_MODES]
        fingerprints = [initialization_evidence(model)["fingerprints"] for model in attention]
        self.assertEqual(len({v["message_projection"] for v in fingerprints}), 1)
        self.assertEqual({v["outer_exchange"] for v in fingerprints}, {mapped["fingerprints"]["outer_exchange"]})
        self.assertEqual({v["shared"] for v in fingerprints}, {mapped["fingerprints"]["shared"]})
        model = attention[-1]
        exchange = model.backbone.interaction
        original = exchange.forward_aligned
        received = {}

        def capture(pixels, regions, assignment, alignment, **kwargs):
            received["pixels"] = alignment.to_regions(pixels, assignment).detach().clone()
            received["regions"] = regions.detach().clone()
            return original(pixels, regions, assignment, alignment, **kwargs)

        def capture_branch(name):
            def hook(_, args):
                received[name] = tuple(t.detach().clone() for t in args[:2])
            return hook

        tiles = [scene.plan["tiles"][row["id"]] for row in record["roles"] if row["role"] == "devtrain"][:2]
        hooks = [exchange.to_pixels.register_forward_pre_hook(capture_branch("to_pixels")),
                 exchange.to_regions.register_forward_pre_hook(capture_branch("to_regions"))]
        try:
            with torch.no_grad(), patch.object(exchange, "forward_aligned", side_effect=capture):
                first = _forward(model, scene, tiles[0])[:3]
                torch.testing.assert_close(received["to_pixels"][0], received["pixels"])
                torch.testing.assert_close(received["to_pixels"][1], received["regions"])
                torch.testing.assert_close(received["to_regions"][0], received["regions"])
                torch.testing.assert_close(received["to_regions"][1], received["pixels"])
        finally:
            for hook in hooks:
                hook.remove()
        with torch.no_grad():
            _forward(model, scene, tiles[1])
            self.assertTrue(_exact(first, _forward(model, scene, tiles[0])[:3]))
            for direction in DIRECTIONS:
                getattr(exchange.exchange, direction).strength = 0.
            self.assertTrue(_exact(_forward(base, scene, tiles[0])[:3], _forward(model, scene, tiles[0])[:3]))

    def test_9_and_22_class_three_heads_and_actual_gradient_updates(self):
        for classes in (9,22):
            with self.subTest(classes=classes):
                base, scene, record = spatial_fixture(classes)
                plan = freeze(scene, record, updates=64)
                before = state_hash(base)
                for arm in campaign.ARMS:
                    run = campaign.new_run(base, scene, record, plan, arm)
                    row = run.step()
                    run._validate_history_extensions(row)
                    self.assertEqual(row["validation"]["count"], 5*classes)
                    self.assertEqual(len(run.timings["train_backward"]), 1)
                    self.assertGreater(row["changed_parameters"]["heads"], 0)
                    for name, evidence in row["message_parameters"].items():
                        self.assertIsNotNone(evidence["gradient_l1"], name)
                        self.assertGreater(evidence["gradient_l1"], 0, name)
                        self.assertGreater(evidence["changed"], 0, name)
                self.assertEqual(state_hash(base), before)

    def test_registered_recovery_and_scope_rejects_altered_inputs(self):
        base, scene, record = spatial_fixture()
        plan = freeze(scene, record, updates=24)
        run = campaign.new_run(base, scene, record, plan, "combined")
        run.advance(3)
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / "checkpoint.pt"
            options = dict(plan=plan, trial="combined", selector_seed=64, expected_sha256=run.save(path),
                           expected_contract_hash=run.contract_hash)
            restored = RegionMessageRun.load(base, scene, record, path, **options)
            self.assertEqual(restored.step(), run.step())
            self.assertTrue(_exact(restored.optimizer.state_dict(), run.optimizer.state_dict()))
            for edit in (dict(trial="route"), dict(expected_sha256="0"*64), dict(selector_seed=63),
                         dict(plan=freeze(scene, record, updates=25))):
                with self.assertRaises(ValueError):
                    RegionMessageRun.load(base, scene, record, path, **{**options, **edit})
            poisoned = deepcopy(scene)
            tile = next(row for row in record["roles"] if row["role"] == "devtrain")
            y, _, x, _ = tile["box"]
            poisoned.cube[y,x,0] += 1
            with self.assertRaises(ValueError):
                RegionMessageRun.load(base, poisoned, record, path, **options)
            with torch.no_grad():
                next(base.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "source state"):
                RegionMessageRun.load(base, scene, record, path, **options)

    def test_stage_duplicate_cross_arm_and_wrong_budget(self):
        base, scene, record = spatial_fixture()
        plan = freeze(scene, record, updates=24)
        run = campaign.new_run(base, scene, record, plan, "local")
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            registration = {"path": str(root / "registration.json"), "sha256": "a"*64}
            stages = {"full": (0,24,None), "prefix": (0,7,None), "resume": (7,24,"prefix"),
                      "completed": (24,24,"resume")}
            kwargs = dict(value=dict(budget=24, contracts={"local": run.contract_hash}), baseline=base,
                          scene=scene, record=record, plan=plan, arms=campaign.ARMS, stages=stages,
                          run_factory=campaign.new_run, run_type=RegionMessageRun)
            prefix = execute_stage(root, registration, "local", "prefix", None, **kwargs)
            with self.assertRaisesRegex(FileExistsError, "consumed"):
                execute_stage(root, registration, "local", "prefix", None, **kwargs)
            with self.assertRaisesRegex(ValueError, "Cross-arm"):
                execute_stage(root, registration, "route", "resume", prefix, **kwargs)
            with self.assertRaisesRegex(ValueError, "budget"):
                execute_stage(root, registration, "local", "resume", prefix,
                              **{**kwargs, "value": {"budget": 25}})


if __name__ == "__main__":
    unittest.main()

"""Independent scalar/gradient and fail-before-mutation checks for released labels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.task_losses import probability_loss_components
from scripts.run_native_released_points import check_real_route
from transfer import native_released_points as points, native_released_training as sparse
from transfer import native_scale_training as training
from transfer.native_released_preflight import open_bank
from transfer.native_released_real import RealReleasedRun
from transfer.native_scale_fixture import Fixture
from transfer.native_scale_observation import Lattice
from transfer.native_scale_supervision import loss
from transfer.xuzhou_multiscale_audit import _exact


class ReleasedPointTests(unittest.TestCase):
    def material(self, classes, factor):
        fixture = Fixture(classes, "devtrain")
        spec = {"tile": fixture.tiles[0], "factor": factor, "band": "even_retained_positions", "profile": "a"}
        return sparse.materialize(fixture, spec, "oracle")

    def test_count_loss_matches_independent_points_and_gradient(self):
        for classes in (9, 22):
            for factor in (1, 2, 4):
                with self.subTest(classes=classes, factor=factor):
                    actual, labels, target = self.material(classes, factor)
                    values, mask = target.weights(points.MODE)
                    generator = torch.Generator().manual_seed(classes)
                    logits = [torch.randn(len(actual.lattice.cells), classes, generator=generator,
                                          dtype=torch.float64, requires_grad=True) for _ in range(3)]
                    p = tuple(value.softmax(1) for value in logits)
                    observed = loss(p, values, mask)
                    y0, _, x0, _ = actual.lattice.box
                    w = actual.lattice.scene_shape[1]
                    width = actual.lattice.shape[1]
                    remap = [(((pixel//w-y0)//factor)*width+(pixel%w-x0)//factor, label-1)
                             for pixel, label in target.points]
                    oracle = [-sum(head[i, label].clamp(1e-8, 1.).log() for i, label in remap) / len(remap)
                              for head in p]
                    for name, expected in zip(points.HEADS, oracle, strict=True):
                        torch.testing.assert_close(observed[name+"_supervised"], expected, rtol=5e-8, atol=1e-12)
                    torch.testing.assert_close(observed["total"], sum(oracle), rtol=5e-8, atol=1e-12)
                    native = torch.autograd.grad(observed["total"], logits, retain_graph=True)
                    independent = torch.autograd.grad(sum(oracle), logits)
                    for a, b in zip(native, independent, strict=True):
                        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-10)
                    if factor == 1:
                        y = np.zeros_like(labels)
                        for pixel, label in target.points:
                            y[pixel//w-y0, pixel%w-x0] = label
                        flat = torch.as_tensor(y.ravel())
                        base = torch.nn.functional.one_hot(flat, classes+1)[:, 1:]
                        keep = (flat > 0)[:, None].expand_as(base)
                        expected = probability_loss_components(p, base.to(p[0]), keep.to(p[0]))
                        self.assertTrue(torch.equal(expected["total"], observed["total"]))
                    else:
                        self.assertGreater(target.summary()["conflicting_cells"], 0)
                    scored = points.scores(p, target, detailed=True)
                    for row in scored.values():
                        self.assertEqual(row["point_confusion"], row["coarse_known_count_confusion"])
                        self.assertEqual(sum(row["class_support"]), len(remap))
                        self.assertLess(row["coarse_known_cells"], row["count"]+int(factor==1))
                    self.assertGreater(target.summary()["unknown_pixels"], 0)
                    self.assertIsNone(target.summary()["complete_purity"])

    def test_tampered_geometry_role_labels_and_legacy_restore_rejected(self):
        actual, labels, target = self.material(9, 4)
        role = "devtrain"
        row = {"id": actual.receipt["source"]["tile"], "role": role,
               "box": actual.lattice.box, "samples": list(target.points)}
        for update, text in (({"samples": [*row["samples"], row["samples"][-1]]}, "unique"),
                             ({"role": "devval"}, "geometry or role"),
                             ({"box": (0, 1, 0, 1)}, "geometry or role")):
            with self.assertRaisesRegex(ValueError, text):
                points.bind(actual, {**row, **update}, "oracle", role, synthetic_labels=labels)
        other_lattice = Lattice.make(actual.lattice.box, actual.lattice.scene_shape, 2)
        with self.assertRaisesRegex(ValueError, "grid"):
            points.bind(replace(actual, lattice=other_lattice), row, "oracle", role, synthetic_labels=labels)
        with self.assertRaisesRegex(ValueError, "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"):
            open_bank(None, "paviaU")
        spec = sparse.contract(9, 64, "joint", 4, "oracle")
        run = training.Run(spec, "oracle", released=sparse)
        for mutation in ("labels_root", "edge", "objective", "initialization"):
            with self.assertRaisesRegex(ValueError, "registration"):
                training.Run({**spec, mutation: None}, "oracle", released=sparse)
        run.start()
        run.update(spec["training"][0])
        current = run.pack()
        with self.assertRaisesRegex(ValueError, "registration"):
            run.restore({**current, "format": training.FORMAT}, 1)
        wrong = deepcopy(current)
        wrong["history"][0]["released"]["points"] = 0
        with self.assertRaisesRegex(ValueError, "released point input"):
            run.restore(wrong, 1)
        self.assertTrue(_exact(current, run.pack()))

    def test_real_execution_requires_separately_reviewed_release_before_bank_access(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch("transfer.native_released_real.require_frozen_code"), \
                 patch("transfer.native_released_real.open_bank") as bank:
                with self.assertRaisesRegex(ValueError, "separately reviewed"):
                    RealReleasedRun(root, None, "case", None, root / "run", {})
                bank.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_real_path_cannot_spend_the_same_budget_twice(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            self.assertIsNone(check_real_route(out, "full"))
            (out / "prefix.intent.json").touch()
            self.assertEqual(check_real_route(out, "resume"), "prefix")
            with self.assertRaisesRegex(ValueError, "one training path"):
                check_real_route(out, "full")
            (out / "resume.intent.json").touch()
            self.assertEqual(check_real_route(out, "completed"), "resume")
            for phase in ("resume", "prefix"):
                with self.assertRaisesRegex(ValueError, "one training path"):
                    check_real_route(out, phase)

    def test_empty_block_rejected_before_modes_gradient_and_rng(self):
        spec = sparse.contract(22, 512, "mean", 2, "oracle")
        run = training.Run(spec, "oracle", released=sparse)
        run.start()
        actual, labels, target = sparse.materialize(run._train, spec["training"][0], "oracle")
        row = {"id": actual.receipt["source"]["tile"], "role": "devtrain",
               "box": actual.lattice.box, "samples": []}
        empty = points.bind(actual, row, "oracle", "devtrain", synthetic_labels=labels)
        current = run.pack()
        with patch.object(run, "_materialize", return_value=(actual, labels, empty)),              patch.object(run.model, "train", wraps=run.model.train) as mode,              patch.object(run.optimizer, "zero_grad", wraps=run.optimizer.zero_grad) as zero:
            with self.assertRaisesRegex(ValueError, "No released points"):
                run.update(spec["training"][0])
            mode.assert_not_called()
            zero.assert_not_called()
        self.assertTrue(_exact(current, run.pack()))
        event = spec["training"][0]
        wrong = deepcopy(spec)
        first = next(k for k in wrong["inputs"] if k.startswith("devtrain"))
        wrong["inputs"][first]["bands"] = "wrong"
        with self.assertRaisesRegex(ValueError, "retained band"):
            sparse.validate_observation(wrong, actual, labels, target, event)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

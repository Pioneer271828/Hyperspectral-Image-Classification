"""Run only in the authorized LXC; external author root is explicit."""
import copy
import io
import os
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import Adapter, first_minimum, load_author_model, objective, patches


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.author = load_author_model(os.environ["PAPER1C_DSNET"])

    def test_original_forward_gradient_update_and_reload(self):
        for classes in (3, 9, 22):
            torch.manual_seed(64)
            original = self.author.DSNet(8, classes, 7, "conv2d_unmix")
            adapted = Adapter(self.author, 8, classes)
            adapted.model.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
            inputs = torch.rand(3, 8, 7, 7) * .8 + .1
            target = torch.arange(3) % classes
            before = copy.deepcopy(original.state_dict())
            direct = original(inputs)
            wrapped = adapted(inputs)
            for a, b in zip(direct, wrapped):
                self.assertTrue(torch.equal(a, b))
            # Reference expression follows demo.py's original operations.
            nonlinear, linear, logits = direct
            reconstructed = nonlinear + (linear[:, :8] + linear[:, 8:16])
            sad = torch.mean(torch.acos(torch.sum(inputs * reconstructed, dim=1) / (
                torch.norm(reconstructed, dim=1, p=2) * torch.norm(inputs, dim=1, p=2))))
            reference = torch.nn.functional.cross_entropy(logits, target) + sad
            actual = objective(wrapped, inputs, target)
            self.assertTrue(torch.equal(reference, actual))
            reference.backward()
            actual.backward()
            for a, b in zip(original.parameters(), adapted.model.parameters()):
                self.assertTrue(torch.equal(a.grad, b.grad))
            aopt = torch.optim.Adam(original.parameters(), lr=1e-4)
            bopt = torch.optim.Adam(adapted.model.parameters(), lr=1e-4)
            aopt.step()
            bopt.step()
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, adapted.model.state_dict()[name]))
            self.assertTrue(any(not torch.equal(value, before[name]) for name, value in original.state_dict().items()))
            adapted.eval()
            stream = io.BytesIO()
            torch.save(adapted.state_dict(), stream)
            stream.seek(0)
            clone = Adapter(self.author, 8, classes).eval()
            clone.load_state_dict(torch.load(stream, weights_only=True))
            for a, b in zip(adapted(inputs), clone(inputs)):
                self.assertTrue(torch.equal(a, b))

    def test_tile_local_padding_and_rejections(self):
        x = torch.arange(2*3*4, dtype=torch.float32).reshape(2, 3, 4)
        result = patches(x, [(0, 0), (1, 2)])
        self.assertEqual(tuple(result.shape), (2, 4, 7, 7))
        self.assertTrue(torch.equal(result[0, :, 0, 0], x[0, 0]))
        self.assertTrue(torch.equal(result[1, :, -1, -1], x[1, 2]))
        for points in ([(2, 0)], [(0, 0), (0, 0)], []):
            with self.assertRaises(ValueError):
                patches(x, points)
        bad = x.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            patches(bad, [(1, 1)])

    def test_selection_is_frozen_and_nonfinite_is_failure(self):
        self.assertEqual(first_minimum([(0, 2.), (5, 1.), (10, 1.)]), 5)
        for rows in ([], [(0, float("nan"))], [(1, 2.), (0, 1.)]):
            with self.assertRaises(ValueError):
                first_minimum(rows)
        model = Adapter(self.author, 8, 3)
        inputs = torch.zeros(2, 8, 7, 7)
        with self.assertRaises(FloatingPointError):
            objective(model(inputs), inputs, torch.zeros(2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

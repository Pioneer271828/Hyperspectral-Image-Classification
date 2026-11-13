"""Buffer scopes, failure restoration and read-only moment probes."""
from unittest import TestCase

import torch
from torch import nn

from evaluation.p4_bn_factors import Moments, buffer_reset, fingerprint, groups
from transfer.g3_training import state_hash


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        for part in ("CNN_denoise", "CNN_Branch", "HGNN_Branch"):
            holder = nn.Module()
            holder.norms = nn.ModuleDict({"synthetic": nn.BatchNorm1d(2)})
            setattr(self.backbone, part, holder)


class FactorTests(TestCase):
    def test_group_scope_and_cleanup_under_failure(self):
        model = Fixture().eval()
        init = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.running_mean.fill_(7)
        frozen = state_hash(model)
        with self.assertRaisesRegex(RuntimeError, "fixture error"):
            with buffer_reset(model, init, ("front",)) as changed:
                self.assertEqual(len(changed), 3)
                for module, identity in groups(model).items():
                    tensor = model.get_submodule(module).running_mean
                    self.assertTrue(torch.all(tensor == (0 if identity == "front" else 7)))
                raise RuntimeError("fixture error")
        self.assertEqual(state_hash(model), frozen)

    def test_population_moments_do_not_write_buffers(self):
        layer = nn.BatchNorm1d(2).eval()
        mom = Moments()
        mom.tile = 7
        initial, params = state_hash(layer), fingerprint(layer)
        hook = layer.register_forward_pre_hook(mom.observe)
        try:
            layer(torch.tensor([[1., 2.], [3., 6.], [5., 4.]]))
        finally:
            hook.remove()
        values = mom.compare(layer)
        self.assertEqual(values["context_mean"], [3., 4.])
        for value in values["context_var"]:
            self.assertAlmostEqual(value, 8/3)
        self.assertEqual(state_hash(layer), initial)
        self.assertEqual(fingerprint(layer), params)

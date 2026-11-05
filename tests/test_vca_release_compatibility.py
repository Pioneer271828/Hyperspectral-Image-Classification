"""Main-based repair: unchanged fresh mathematics, live gradients after warmup."""
import os
import subprocess
import unittest

import torch
from model.variable_channel_adapter import VariableChannelAdapter

DEVICE = os.environ.get('REPAIR_DEVICE', 'cpu')
BASE = '4d2c7dbe4a081d671b1367bc4fb3eabf5e929ccc'
torch.set_num_threads(1)


class ReleaseCompatibilityTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(64)
        model = VariableChannelAdapter(latent_channels=8, embed_dim=8).to(DEVICE)
        x = torch.randn(2, 3, 6, device=DEVICE)
        wave = torch.linspace(450, 900, 6, device=DEVICE)
        return model, x, wave

    def test_fresh_training_matches_immutable_main_output_and_gradients(self):
        source = subprocess.check_output(
            ['git', 'show', BASE + ':model/variable_channel_adapter.py'], text=True)
        namespace = {}
        exec(compile(source, 'immutable-main-adapter', 'exec'), namespace)
        model, x, wave = self.fixture()
        old = namespace['VariableChannelAdapter'](latent_channels=8, embed_dim=8).to(DEVICE)
        old.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(old.state_dict()))
        left = model(x, wave)
        right = old(x, wave)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        for (name, p), (other, q) in zip(model.named_parameters(), old.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_warm_eval_retains_attention_parameter_gradients(self):
        model, x, wave = self.fixture()
        model.eval()
        with torch.no_grad():
            model(x, wave)
        model(x, wave).square().sum().backward()
        for p in (model.query_encoder[0].weight, model.key_encoder[0].weight,
                  model.anchor_offsets):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_inplace_weight_change_and_load_invalidate_inference_cache(self):
        model, x, wave = self.fixture()
        model.eval()
        with torch.no_grad():
            model(x, wave)
            model.query_encoder[0].weight.add_(.25)
            cached = model(x, wave)
            model.clear_attention_cache()
            torch.testing.assert_close(cached, model(x, wave), rtol=0, atol=0)
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model(x, wave)
            model.load_state_dict(saved)
            self.assertFalse(model._eval_attention_cache)


if __name__ == '__main__':
    unittest.main()

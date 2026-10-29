"""M8 fixed fusion, logits and unchanged masked probability supervision."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCN
from model.branch_fusion import fixed_fusion
from model.task_losses import probability_loss_components, detached_loss_values
from model.research_config import ResearchConfig
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from train_vca_multidataset import supervised_loss, build_parser

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b2809302a44d1ff454004361a42d34001235616d'
torch.set_num_threads(1)


def legacy_module(path, name):
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':' + path], text=True)
    module = types.ModuleType(name)
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def model(**kwargs):
    return MultiDatasetVCASAHGCN({'a': 3, 'b': 5}, latent_channels=8, adapter_embed_dim=8,
                                locality_window=None, graph_mode='learned', norm_type='group_norm', **kwargs)


def supervision():
    labels = torch.arange(42) % 3
    mask = (torch.arange(42) % 4 != 0)[:, None].expand(-1, 3).float()
    return {'train_gt_onehot': torch.nn.functional.one_hot(labels, 3).float() * mask,
            'train_label_mask': mask}


class FusionLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_default_old_outputs_loss_and_all_gradients_exact(self):
        old_models = legacy_module('model/VCASAHGCN.py', 'old_m8_models')
        old_trainer = legacy_module('train_vca_multidataset.py', 'old_m8_trainer')
        weights = {'compactness': .01, 'entropy': .001, 'balance': .01}
        for options in ({}, {'interaction_mode': 'gated', 'graph_update_mode': 'joint'}):
            old = old_models.MultiDatasetVCASAHGCN({'a': 3, 'b': 5}, latent_channels=8, adapter_embed_dim=8,
                    locality_window=None, graph_mode='learned', norm_type='group_norm', **options)
            current = model(**options)
            current.load_state_dict(old.state_dict(), strict=True)
            self.assertEqual(set(old.state_dict()), set(current.state_dict()))
            old_x = torch.randn(6, 7, 14, requires_grad=True)
            new_x = old_x.detach().clone().requires_grad_()
            expected = old('a', old_x, superpixel_scale=9, return_graph_info=True)
            actual = current('a', new_x, superpixel_scale=9, return_graph_info=True)
            for before, after in zip(expected[:3], actual[:3]):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            old_loss = old_trainer.supervised_loss(expected, supervision(), 'train', True, weights, .3, .7)
            new_loss = supervised_loss(actual, supervision(), 'train', True, weights, .3, .7)
            torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
            old_loss.backward(); new_loss.backward()
            torch.testing.assert_close(old_x.grad, new_x.grad, rtol=0, atol=0)
            for (name, before), (_, after) in zip(old.named_parameters(), current.named_parameters()):
                if before.grad is None:
                    self.assertIsNone(after.grad, name)
                else:
                    torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0, msg=name)

    def test_logits_probability_and_fuse_predict_correspondence(self):
        net = model().eval()
        for name, classes, bands in (('a', 3, 12), ('b', 5, 17)):
            x = torch.randn(6, 7, bands + 2)
            with torch.no_grad():
                probs = net(name, x, superpixel_scale=9, return_graph_info=True)
                logits = net(name, x, superpixel_scale=9, return_graph_info=True, return_logits=True)
                features = net.backbone.forward_features(x, dataset_name=name, superpixel_scale=9)
                direct = net.fuse_and_predict(features[2], features[1], name)
            self.assertEqual(len(logits), 4)
            for probability, raw, prediction in zip(probs[:3], logits[:3], direct):
                self.assertEqual(tuple(raw.shape), (42, classes))
                torch.testing.assert_close(raw.softmax(-1), probability, rtol=0, atol=0)
                torch.testing.assert_close(prediction, probability, rtol=0, atol=0)
                torch.testing.assert_close(probability.sum(-1), torch.ones(42))
        wrapper = VCASAHGCN(6, 7, 14, 3, None, None, .8, latent_channels=8,
                           adapter_embed_dim=8, graph_mode='learned', norm_type='group_norm').eval()
        x = torch.randn(6, 7, 14)
        with torch.no_grad():
            for probability, raw in zip(wrapper(x), wrapper(x, return_logits=True)):
                torch.testing.assert_close(raw.softmax(-1), probability, rtol=0, atol=0)

    def test_fixed_fusion_endpoints_and_input_gradients(self):
        cnn, graph = torch.randn(5, 64, requires_grad=True), torch.randn(5, 64, requires_grad=True)
        for alpha, expected in ((0., graph), (1., cnn)):
            actual = fixed_fusion(cnn, graph, alpha)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fixed_fusion(cnn, graph, .8).sum().backward()
        torch.testing.assert_close(cnn.grad, torch.full_like(cnn, .8))
        torch.testing.assert_close(graph.grad, torch.full_like(graph, .2))
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): model(alph=alpha)
        with self.assertRaises(ValueError): fixed_fusion(cnn, graph[:, :4])

    def test_mask_zero_aux_weights_and_empty_supervision(self):
        probabilities = [torch.rand(42, 3, requires_grad=True) for _ in range(3)]
        item = supervision()
        loss = supervised_loss(probabilities, item, 'train', cnn_loss_weight=0., graph_branch_loss_weight=0.)
        loss.backward()
        masked = item['train_label_mask'][:, 0] == 0
        torch.testing.assert_close(probabilities[0].grad[masked], torch.zeros_like(probabilities[0].grad[masked]))
        self.assertGreater(probabilities[0].grad[~masked].abs().sum().item(), 0)
        for branch in probabilities[1:]: torch.testing.assert_close(branch.grad, torch.zeros_like(branch.grad))
        changed = [value.detach().clone() for value in probabilities]
        for value in changed: value[masked] = .999
        torch.testing.assert_close(loss.detach(), supervised_loss(changed, item, 'train', cnn_loss_weight=0., graph_branch_loss_weight=0.))
        zero_item = {key: torch.zeros_like(value) for key, value in item.items()}
        for balanced in (False, True):
            value = supervised_loss(probabilities, zero_item, 'train', class_balance=balanced)
            self.assertEqual(value.item(), 0.)
            self.assertTrue(torch.isfinite(value))

    def test_named_weighted_terms_and_regularizer_gradients(self):
        outputs = [torch.rand(42, 3, requires_grad=True) for _ in range(3)]
        losses = [torch.tensor(v, requires_grad=True) for v in (2., 3., 4.)]
        outputs.append(dict(zip(('compactness_loss', 'assignment_entropy_loss', 'assignment_balance_loss'), losses)))
        weights = {'compactness': .2, 'entropy': 0., 'balance': .4}
        terms = supervised_loss(outputs, supervision(), 'train', graph_loss_weights=weights,
                                cnn_loss_weight=.3, graph_branch_loss_weight=.7, return_components=True)
        self.assertEqual(len(terms), 13)
        total = (terms['weighted_fused_supervised'] + terms['weighted_graph_supervised'] + terms['weighted_cnn_supervised']
                 + terms['weighted_graph_compactness'] + terms['weighted_graph_entropy'] + terms['weighted_graph_balance'])
        torch.testing.assert_close(terms['total'], total, rtol=0, atol=0)
        terms['total'].backward()
        for loss, expected in zip(losses, (.2, 0., .4)):
            torch.testing.assert_close(loss.grad, torch.tensor(expected))
        detached = detached_loss_values(terms)
        self.assertTrue(all(isinstance(value, float) for value in detached.values()))
        # Missing graph info/weights remains the original supervised-only objective.
        onehot, mask = supervision().values()
        plain = probability_loss_components(outputs[:3], onehot, mask)
        for name in ('compactness', 'entropy', 'balance'):
            self.assertEqual(plain['weighted_graph_' + name].item(), 0.)

    def test_class_balance_clamping_and_absent_classes_match_legacy(self):
        old = legacy_module('train_vca_multidataset.py', 'old_m8_loss')
        outputs = [torch.tensor([[0., .4, 1.1], [.2, .3, .5], [.4, .5, .1]], requires_grad=True) for _ in range(3)]
        item = {'train_gt_onehot': torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                'train_label_mask': torch.ones(3, 3)}
        for balanced in (False, True):
            torch.testing.assert_close(supervised_loss(outputs, item, 'train', balanced),
                                       old.supervised_loss(outputs, item, 'train', balanced), rtol=0, atol=0)

    def test_alpha_config_cli_and_new_old_checkpoint_roundtrip(self):
        base = ResearchConfig.from_dict({})
        changed = ResearchConfig.from_dict({'model': {'alph': .35}})
        self.assertNotEqual(base.config_hash, changed.config_hash)
        args = build_parser().parse_args(['--alph', '.35'])
        self.assertEqual(ResearchConfig.from_legacy(args).model.alph, .35)
        for alpha in (-1, 2, float('nan')):
            with self.assertRaises(ValueError): ResearchConfig.from_dict({'model': {'alph': alpha}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for alpha in (.35, .8):
                net = model(alph=alpha).eval()
                save_model_checkpoint(net, path)
                if alpha == .8:
                    metadata_path = Path(str(path) + '.json')
                    metadata = json.loads(metadata_path.read_text())
                    metadata['backbone_kwargs'].pop('alph')
                    metadata_path.write_text(json.dumps(metadata))
                restored = load_model_checkpoint(path)
                self.assertEqual(restored.backbone.alph, alpha)
                x = torch.randn(6, 7, 14)
                with torch.no_grad():
                    for actual, expected in zip(restored('a', x), net('a', x)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

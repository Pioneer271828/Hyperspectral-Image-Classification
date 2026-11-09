"""Coordinate transformations, cleanup and end-to-end validation support."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import torch
from scripts.intervene_screening_coordinates import transform_input, coordinate_condition, run_validation, CONDITIONS
from loadData.spatial_tiles import make_tile_plan, fit_tile_statistics
from train_spatial import SpatialConfig

class CoordinateTests(unittest.TestCase):
    def test_coordinates_preserve_spectral_and_source(self):
        x = torch.randn(3, 5, 8)
        before = x.clone()
        self.assertIs(transform_input(x, 'original'), x)
        for mode in ('zero', 'relative'):
            y = transform_input(x, mode)
            torch.testing.assert_close(y[..., :-2], before[..., :-2], rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)
            if mode == 'zero': self.assertEqual(y[..., -2:].count_nonzero(), 0)
            else:
                torch.testing.assert_close(y[:, 0, -2], torch.tensor([0., .5, 1.]))
                torch.testing.assert_close(y[0, :, -1], torch.linspace(0, 1, 5))
        self.assertTrue(torch.isfinite(transform_input(torch.ones(1, 1, 3), 'relative')).all())
        with self.assertRaises(ValueError): transform_input(x, 'invalid')

    def test_exception_cleanup_and_reject_training(self):
        class Echo(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = SimpleNamespace(extra_channels=2)
            def forward(self, dataset, x): return x
        model = Echo().eval(); x = torch.ones(2, 3, 4)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with coordinate_condition(model, 'zero'):
                self.assertEqual(model('x', x)[..., -2:].count_nonzero(), 0)
                raise RuntimeError('injected')
        self.assertFalse(model._forward_pre_hooks)
        self.assertIs(model('x', x), x)
        model.train()
        with self.assertRaises(ValueError):
            with coordinate_condition(model, 'zero'): pass

    def test_real_model_collection_validation_only(self):
        torch.manual_seed(4); torch.set_num_threads(1)
        config = SpatialConfig({'name':'coordinate_fixture', 'dataset':'paviaU',
            'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm'},
            'interaction':{'interaction_mode':'gated'}, 'training':{'epochs':0,'steps_per_epoch':2},
            'tile_size':12,'gap':1})
        model = config.research.build_model({'paviaU':9}).eval()
        cube = np.random.default_rng(4).normal(size=(60,60,6)).astype(np.float32)
        before = cube.copy()
        labels = np.tile(np.arange(1,10),400).reshape(60,60)
        plan = make_tile_plan((60,60),12,1); stats = fit_tile_statistics(cube,plan)
        from scripts.intervene_screening_coordinates import collect_partition
        partitions = []
        def guarded(*args, **kwargs):
            partitions.append(args[5]); self.assertEqual(args[5], 'val')
            return collect_partition(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('scripts.intervene_screening_coordinates.collect_partition', side_effect=guarded):
            summaries, _, checks = run_validation(model,cube,labels,plan,stats,None,50,list(range(1,10)),Path(directory))
        self.assertEqual(partitions, ['val']*4)
        self.assertEqual(set(summaries), set(CONDITIONS))
        self.assertIn('original_after_restoration', checks)
        np.testing.assert_array_equal(cube,before)
        self.assertFalse(model._forward_pre_hooks)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

if __name__ == '__main__': unittest.main()

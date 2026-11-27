"""Check independent image-to-feature reconstruction on generated NPY tiles only."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from c8_bridge import digest, write, patch_batch
from features import transform
from audit_features import audit_original_features


class OriginalFeatureReconstruction(unittest.TestCase):
    def test_seventeen_synthetic_tiles_both_views_and_cache_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, experiment = root/'inputs', root/'experiment'
            inputs.mkdir()
            experiment.mkdir()
            (experiment/'evaluation01').mkdir()
            write(experiment/'fitting-complete.json', dict(all_fitting_complete_before_any_development_array=True))
            write(experiment/'evaluation01/evaluation.json', {})
            rng = np.random.default_rng(904)
            banks, selected, frames = {}, {}, {}
            for scene, counts in [('xuzhou', (7, 3)), ('whu-honghu', (5, 2))]:
                (inputs/scene).mkdir()
                rows, samples, roles = {}, {}, {}
                for role, count in zip(('devtrain', 'devval'), counts):
                    by_tile = {i+(0 if role == 'devtrain' else 100): [] for i in range(count)}
                    for index in range(10):
                        tile = list(by_tile)[index % count]
                        y0 = tile*4
                        local_y, local_x = divmod(index//count, 3)
                        pid = (y0+local_y)*100+local_x
                        by_tile[tile].append(dict(point_id=pid, parent_id=tile, original_label=1 if index < 5 else 2, common_id=int(index >= 5)))
                    role_frame = {k: [] for k in ('raw_distribution', 'legacy_scene_distribution', 'y', 'point_ids', 'parent_ids')}
                    for tile, points in by_tile.items():
                        box = [tile*4, tile*4+3, 0, 3]
                        rows[tile] = dict(box=box, role=role, parent_tile=tile)
                        samples[tile] = [[p['point_id'], p['original_label']] for p in points]
                        raw = rng.uniform(-1, 2, (3, 3, 17)).astype(np.float32)
                        with (inputs/scene/f'tile-{tile}.npy').open('xb') as stream:
                            np.lib.format.write_array(stream, raw, version=(1, 0), allow_pickle=False)
                        low = np.array([0]+[-1]*16, np.float32)
                        high = np.array([0]+[3]*16, np.float32)
                        normalized = (raw-low)/np.where(high > low, high-low, np.float32(1))
                        ids = [p['point_id'] for p in points]
                        for rep, image in [('raw_distribution', raw), ('legacy_scene_distribution', normalized)]:
                            patches, _ = patch_batch(image, ids, box, 100)
                            role_frame[rep].extend(transform(patches))
                        role_frame['point_ids'].extend(ids)
                        role_frame['parent_ids'].extend([tile]*len(ids))
                        role_frame['y'].extend(p['common_id'] for p in points)
                    roles[role] = by_tile
                    frames[scene, role] = {key: np.asarray(values) for key, values in role_frame.items()}
                banks[scene] = SimpleNamespace(rows=rows, samples=samples, pin={'shape': [1000, 100, 17]}, registration={'statistics': {'minimum': [0]+[-1]*16, 'maximum': [0]+[3]*16}})
                selected[scene] = roles
            def batch(phase, tile):
                scene = 'xuzhou' if phase.startswith('source_') else 'whu-honghu'
                return {'original_npy_sha256': digest(inputs/scene/f'tile-{tile}.npy')}
            gate = SimpleNamespace(banks=banks, _selected=selected, batch=batch)
            prefixes = []
            for source, target in [('xuzhou', 'whu-honghu'), ('whu-honghu', 'xuzhou')]:
                for stage, scene in [('source', source), ('target', target)]:
                    folder = experiment/f'{stage}-{source}-to-{target}'
                    folder.mkdir()
                    prefixes.append((folder/'features', scene, 'devtrain'))
            prefixes += [(experiment/'evaluation01'/f'features-{s}', s, 'devval') for s in banks]
            for prefix, scene, role in prefixes:
                with prefix.with_suffix('.npz').open('xb') as stream:
                    np.savez(stream, **frames[scene, role])
                write(prefix.with_suffix('.json'), dict(cache_name=prefix.with_suffix('.npz').name, cache_sha256=digest(prefix.with_suffix('.npz')), scene=scene, original_role=role))
            with patch('audit_features.PairInputs', return_value=gate):
                result = audit_original_features(None, experiment, None, inputs, None)
                self.assertEqual(result['selected_tiles'], 17)
                self.assertEqual(result['feature_caches_checked'], 6)
                path = prefixes[0][0].with_suffix('.npz')
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError, 'digest'):
                    audit_original_features(None, experiment, None, inputs, None)


if __name__ == '__main__':
    unittest.main()

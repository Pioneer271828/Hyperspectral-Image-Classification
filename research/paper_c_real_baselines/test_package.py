"""Complete synthetic artifact package, independent replay/export and restoration.

All examples are generated here. These checks do not read the admitted real
images, release labels or run any real-data optimization.
"""
import csv
import hashlib
import hmac
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from c8_bridge import protocol, canonical, sha, write, read, digest
from models import fit, centroid, predict
from metrics import measures, sensitivity
from evaluate import comparison_rows, direction_rows
from audit import validate_package
from export_tables import export


PRODUCER = '0'*40


def fixture(root):
    """Build the full frozen registry on 40 generated, explicitly synthetic points."""
    root.mkdir()
    cfg = protocol()
    ph = sha(canonical(cfg))
    rng = np.random.default_rng(903)
    data = {}
    for scene in ('xuzhou', 'whu-honghu'):
        for role in ('devtrain', 'devval'):
            data[scene, role] = dict(raw_distribution=rng.normal(0, .3, (10, 8)), legacy_scene_distribution=rng.normal(0, .3, (10, 8)), y=np.array([0]*5+[1]*5), point_ids=np.arange(10)+(0 if role == 'devtrain' else 100), parent_ids=np.array(([1]*3+[2]*4+[3]*3) if scene == 'xuzhou' else ([1]*5+[2]*5)))
    def save_cache(prefix, scene, role, phase):
        path = prefix.with_suffix('.npz')
        with path.open('xb') as stream:
            np.savez(stream, **data[scene, role])
        meta = dict(cache_name=path.name, cache_sha256=digest(path), phase=phase, original_role=role, scene=scene)
        write(prefix.with_suffix('.json'), meta)
        return meta
    for stage in ('source', 'target'):
        for source, target in cfg['directions']:
            folder = root/f'{stage}-{source}-to-{target}'
            folder.mkdir()
            scene = source if stage == 'source' else target
            phase = 'source_fit' if stage == 'source' else 'target_adapt'
            frame = save_cache(folder/'features', scene, 'devtrain', phase)
            entries = []
            def save(model, rep, seed, arm, ancestor=None):
                model.update(source=PRODUCER, protocol_sha256=ph, direction=[source, target], fit_role='devtrain', fit_phase=phase, fit_points=10, fit_scene=scene, representation=rep, arm=arm, feature_cache_sha256=frame['cache_sha256'], selection='closed_form_no_development_selection' if seed is None else 'fixed_terminal_no_development_selection', parent_source=ancestor)
                name = f'{rep}-centroid.json' if seed is None else f'{rep}-s{seed}-{arm}.json'
                write(folder/name, model)
                entries.append(dict(file=name, sha256=digest(folder/name), representation=rep, seed=seed, arm=arm, kind=model['kind'], optimizer_updates=model['optimizer_updates']))
            X, y = data[scene, 'devtrain'], data[scene, 'devtrain']['y']
            for rep in ('raw_distribution', 'legacy_scene_distribution'):
                for seed in cfg['seeds']:
                    if stage == 'source':
                        save(fit(X[rep], y, seed=seed, steps=64), rep, seed, 'source_fit')
                    else:
                        parent = root/f'source-{source}-to-{target}'/f'{rep}-s{seed}-source_fit.json'
                        save(fit(X[rep], y, seed=seed, steps=64, start=read(parent)['theta']), rep, seed, 'target_adapt', dict(file=parent.name, sha256=digest(parent)))
                        for steps in (64, 128):
                            save(fit(X[rep], y, seed=seed, steps=steps), rep, seed, f'target_scratch{steps}')
                if stage == 'source':
                    save(centroid(X[rep], y), rep, None, 'centroid')
            write(folder/'manifest.json', dict(complete=True, source=PRODUCER, protocol_sha256=ph, direction=[source, target], stage=stage, models=entries, optimizer_updates=sum(e['optimizer_updates'] for e in entries), denied_opens=[], no_development_array_phase_available=True))
    write(root/'fitting-complete.json', dict(source=PRODUCER, protocol_sha256=ph, all_fitting_complete_before_any_development_array=True))
    evaluation = root/'evaluation01'
    evaluation.mkdir()
    key = bytes(range(32))  # Public, synthetic fixture key, not a real alias key.
    (evaluation/'pseudonymization.key').write_bytes(key)
    for scene in ('xuzhou', 'whu-honghu'):
        save_cache(evaluation/f'features-{scene}', scene, 'devval', 'source_development' if scene == 'xuzhou' else 'target_development')
    rows = []
    for source, target in cfg['directions']:
        for stage in ('source', 'target'):
            folder = root/f'{stage}-{source}-to-{target}'
            for entry in read(folder/'manifest.json')['models']:
                model = read(folder/entry['file'])
                cases = [(target, entry['arm'])] if stage == 'target' else [(source, 'source_same_device' if entry['kind'] == 'logistic' else 'centroid_same_device'), (target, 'source_fit_transfer' if entry['kind'] == 'logistic' else 'centroid_transfer')]
                for scene, method in cases:
                    frame = data[scene, 'devval']
                    z, p, pred = predict(model, frame[entry['representation']])
                    alias = lambda kind, values: [hmac.new(key, canonical([kind, scene, int(v)]), hashlib.sha256).hexdigest()[:20] for v in values]
                    row = dict(id=f'{source}_to_{target}/{entry["representation"]}/{entry["seed"]}/{method}', direction=[source, target], scene=scene, method=method, representation=entry['representation'], seed=entry['seed'], model_file=str(folder/entry['file']), model_sha256=entry['sha256'], fit_scene=model['fit_scene'], fit_labeled_points=10, point_ids=frame['point_ids'].tolist(), parent_ids=frame['parent_ids'].tolist(), truth=frame['y'].tolist(), logits=z.tolist(), probabilities=p.tolist(), predictions=pred.tolist(), point_aliases=alias('point', frame['point_ids']), parent_aliases=alias('parent', frame['parent_ids']), metrics=measures(frame['y'], z, p, pred))
                    row['parent_sensitivity'] = sensitivity(row)
                    rows.append(row)
    result = dict(evidence_kind='synthetic_test', producer=PRODUCER, protocol_sha256=ph, selection_used_target_development=False, query_reads=0, rows=rows, paired_comparisons=comparison_rows(rows), direction_differences=direction_rows(rows))
    write(evaluation/'evaluation.json', result)
    return result


class SyntheticPackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.root = cls.base/'original'
        fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_52_checkpoint_68_evaluation_independent_replay(self):
        result = validate_package(self.root, PRODUCER, replay_fits=True)
        self.assertEqual(result['independent_replay_updates'], 3840)
        self.assertEqual(result['prediction_presentations'], 680)
        self.assertEqual(result['argmax_changes'], 0)
        self.assertFalse(result['original_image_feature_provenance_checked'])

    def test_export_field_whitelist_keeps_weights_IDs_paths_and_key_private(self):
        out = self.base/'tables'
        result = export(self.root, PRODUCER, out)
        self.assertEqual(result['evidence_kind'], 'synthetic_test')
        self.assertEqual(result['per_point_rows'], 680)
        for path in out.glob('*.csv'):
            with path.open(newline='', encoding='utf-8') as stream:
                fields = csv.DictReader(stream).fieldnames
            self.assertTrue(set(fields).isdisjoint({'theta', 'point_ids', 'parent_ids', 'model_file', 'pseudonymization.key', 'features'}))
        self.assertEqual(len(list(out.iterdir())), 7)

    def test_copy_restoration_matches_every_file_and_predictions_without_refit(self):
        restored = self.base/'restored'
        shutil.copytree(self.root, restored)
        originals = {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob('*') if p.is_file()}
        copies = {str(p.relative_to(restored)): digest(p) for p in restored.rglob('*') if p.is_file()}
        self.assertEqual(originals, copies)
        result = validate_package(restored, PRODUCER, replay_fits=False)
        self.assertEqual(result['independent_replay_updates'], 0)
        self.assertEqual(result['argmax_changes'], 0)

    def test_tampered_metric_and_checkpoint_both_fail_closed(self):
        root = self.base/'tampered'
        shutil.copytree(self.root, root)
        path = root/'evaluation01/evaluation.json'
        original = path.read_bytes()
        data = read(path)
        data['rows'][0]['metrics']['NLL'] += .01
        path.write_text(__import__('json').dumps(data))
        with self.assertRaisesRegex(ValueError, 'NLL'):
            validate_package(root, PRODUCER)
        path.write_bytes(original)
        folder = root/'source-xuzhou-to-whu-honghu'
        model = folder/read(folder/'manifest.json')['models'][0]['file']
        model.write_bytes(model.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'digest'):
            validate_package(root, PRODUCER)


if __name__ == '__main__':
    unittest.main()

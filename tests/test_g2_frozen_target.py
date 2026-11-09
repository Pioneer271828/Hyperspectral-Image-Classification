import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from loadData.spatial_tiles import make_tile_plan
from loadData.transfer_protocol import _pool
from research_runtime import hash_file, write_json
import test_g2_source_runner as source_fixture
from transfer.g2_frozen_target import POLICY, infer_target, pack_target_observations, score_target


class FrozenTargetTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture.SourceRunnerTests()
        self.addCleanup(self.source.doCleanups)
        self.source.setUp()
        self.source.execute('source')
        self.root = self.source.root
        self.commit = self.source.commit
        reg = self.source.record
        self.target = self.source.contract['target']
        catalog = json.loads(Path(reg['catalog']).read_text())
        artifact = catalog['scenes'][self.target]['artifact_sha256']
        self.plan = make_tile_plan((24, 24), tile_size=8, gap=0)
        self.plan_path = self.root/'target-plan.json'
        write_json(self.plan_path, self.plan)
        source_wave = np.linspace(420, 850, 6, dtype=np.float32)
        self.wave = np.array([source_wave[5], 1100, source_wave[0], source_wave[2]], dtype=np.float32)
        self.cube = np.random.default_rng(91).random((24, 24, 4), dtype=np.float32)
        observations = self.root/'target.npz'
        observations.write_bytes(pack_target_observations(self.target, artifact, self.plan, self.cube, self.wave))
        self.evidence = dict(policy='verified_exact_centers_v1', source_reference='paviaU',
                             source_artifact_sha256=catalog['scenes']['paviaU']['artifact_sha256'],
                             target_artifact_sha256=artifact, source_band_indices=[0, 2, 5],
                             target_band_indices=[2, 3, 0], wavelength_quality='verified',
                             radiometry='verified_same_physical_quantity_and_units', provenance_sha256='f'*64)
        self.evidence_path = self.root/'spectral-evidence.json'
        self.release = dict(scene=self.target, partition='test', plan_hash=self.plan['plan_hash'],
                            observation_sha256=hash_file(observations), indices=_pool(self.plan,'test')[:3].tolist(),
                            labels=[1, 2, 3])
        self.labels = self.root/'labels-not-present-during-inference.json'
        self.registration = self.root/'target-registration.json'
        self.record = dict(policy=POLICY, catalog=reg['catalog'], catalog_sha256=reg['catalog_sha256'],
                           contract=reg['contract'], contract_sha256=reg['contract_sha256'],
                           source_result=str(self.root/'source/result.json'),
                           source_result_sha256=hash_file(self.root/'source/result.json'),
                           source_commit=self.commit, source_registration=str(self.root/'registration.json'),
                           spectral_evidence=str(self.evidence_path), target_plan=str(self.plan_path),
                           target_plan_sha256=hash_file(self.plan_path), target_observations=str(observations),
                           target_observations_sha256=hash_file(observations), superpixel_scale=16)

    def register(self):
        write_json(self.evidence_path, self.evidence)
        self.record['spectral_evidence_sha256'] = hash_file(self.evidence_path)
        label_bytes = json.dumps(self.release).encode()
        self.record['target_labels_sha256'] = hashlib.sha256(label_bytes).hexdigest()
        write_json(self.registration, self.record)
        self.sha = hash_file(self.registration)
        return label_bytes

    def infer(self, name='predictions'):
        return infer_target(self.registration, self.sha, self.commit, self.root/name)

    def score(self, inference_sha, name='score'):
        return score_target(self.registration, self.sha, self.commit, self.root/'predictions',
                            inference_sha, self.labels, self.root/name)

    def test_frozen_inference_and_independent_closed_set_metrics(self):
        label_bytes = self.register()
        self.assertFalse(self.labels.exists())
        entry = [sys.executable, '-m', 'transfer.g2_frozen_target']
        args = ['--registration', str(self.registration), '--registration-sha256', self.sha,
                '--expected-commit', self.commit]
        prediction = self.root/'predictions'
        subprocess.run([*entry, 'infer', *args, '--out', str(prediction)], check=True)
        frozen = json.loads((prediction/'receipt.json').read_text())
        self.assertFalse(frozen['target_label_access'])
        self.assertFalse(self.labels.exists())
        self.assertEqual(frozen['optimizer_updates'], 0)
        reference_checkpoint = self.root/'source'/json.loads((self.root/'source/result.json').read_text())['selected_checkpoint']
        self.assertEqual(frozen['checkpoint_sha256'], hash_file(reference_checkpoint))
        self.labels.write_bytes(label_bytes)
        score = self.root/'score'
        subprocess.run([*entry, 'score', *args, '--inference-dir', str(prediction),
                        '--inference-sha256', hash_file(prediction/'receipt.json'),
                        '--label-path', str(self.labels), '--out', str(score)], check=True)
        scored = json.loads((score/'result.json').read_text())
        self.assertEqual(scored['excluded_unmapped'], 1)
        self.assertEqual(scored['optimizer_updates'], 0)
        with np.load(self.root/'predictions/predictions.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['global_ids'], _pool(self.plan, 'test'))
            for name, row in scored['metrics'].items():
                probs = archive[name][:2]
                expected = [[0, 0], [0, 0]]
                for i, prob in enumerate(probs):
                    expected[i][int(prob.argmax())] += 1
                self.assertEqual(row['confusion_matrix'], expected)
                self.assertAlmostEqual(row['nll'], float(np.mean(-np.log(np.diag(probs)))), places=6)

    def test_bad_radiometry_and_nonmatching_band_stop_inference(self):
        self.evidence['radiometry'] = 'unverified'
        self.register()
        with self.assertRaisesRegex(ValueError, 'radiometric evidence'):
            self.infer()
        self.assertFalse((self.root/'predictions').exists())
        self.evidence['radiometry'] = 'verified_same_physical_quantity_and_units'
        self.evidence['target_band_indices'][0] = 1
        self.register()
        with self.assertRaisesRegex(ValueError, 'centers must match'):
            self.infer()

    def test_changed_predictions_rejected_before_label_file_access(self):
        self.register()
        self.infer()
        path = self.root/'predictions/predictions.npz'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'Frozen predictions changed'):
            self.score(hash_file(self.root/'predictions/receipt.json'))
        self.assertFalse((self.root/'score').exists())

    def test_train_pixel_label_rejected_even_when_release_is_registered(self):
        self.release['indices'] = _pool(self.plan,'train')[:3].tolist()
        self.labels.write_bytes(self.register())
        self.infer()
        with self.assertRaisesRegex(ValueError, 'escape the frozen test role'):
            self.score(hash_file(self.root/'predictions/receipt.json'))
        self.assertFalse((self.root/'score').exists())

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from loadData.spatial_tiles import make_tile_plan
from model.checkpoint_io import load_model_checkpoint
from research_runtime import hash_file, write_json
from test_g2_contract import fixture
from transfer.g2_contract import validate_lineage
from transfer.g2_observations import pack_source_observations
from transfer.g2_source_runner import SELECTION, run_source
from transfer.g3_training import head_hash, state_hash


class SourceRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        catalog, self.contract, _, _ = fixture()
        self.catalog_path = self.root / 'catalog.json'
        self.contract_path = self.root / 'contract.json'
        write_json(self.catalog_path, catalog)
        self.contract['catalog_sha256'] = hash_file(self.catalog_path)
        plan = make_tile_plan((24, 24), tile_size=8, gap=0)
        plan_path = self.root / 'plan.json'
        write_json(plan_path, plan)
        generator = np.random.default_rng(8)
        waves = np.linspace(420, 850, 6, dtype=np.float32)
        sources = {}
        for scene in self.contract['sources']:
            entry = dict(plan=str(plan_path), plan_sha256=hash_file(plan_path), superpixel_scale=16)
            cube = generator.random((24, 24, 6), dtype=np.float32)
            artifact = catalog['scenes'][scene]['artifact_sha256']
            for role in ('train', 'val'):
                tile = next(t for t in plan['tiles'] if t['partition'] == role)
                y, _, x, _ = tile['box']
                i = y * plan['shape'][1] + x
                obs = self.root / f'{scene}-{role}.npz'
                obs.write_bytes(pack_source_observations(scene, artifact, plan, role, cube, waves))
                labels = self.root / f'{scene}-{role}.json'
                write_json(labels, dict(scene=scene, partition=role, plan_hash=plan['plan_hash'],
                                       observation_sha256=hash_file(obs), indices=[i, i+1, i+2], labels=[1, 2, 3]))
                self.contract['source_releases'][scene][role] = dict(observation_sha256=hash_file(obs),
                                                                    labels_sha256=hash_file(labels))
                entry.update({f'{role}_observations': str(obs), f'{role}_labels': str(labels)})
            sources[scene] = entry
        write_json(self.contract_path, self.contract)
        self.record = dict(policy='g2_source_cpu_engineering_v1', catalog=str(self.catalog_path),
                           catalog_sha256=hash_file(self.catalog_path), contract=str(self.contract_path),
                           contract_sha256=hash_file(self.contract_path), sources=sources, model_seed=64,
                           sampler_seed=42, steps_per_source=2, val_every_rounds=1, selection=SELECTION,
                           base={'model': {'graph_mode': 'learned', 'norm_type': 'group_norm'},
                                 'training': {'steps_per_epoch': 1}})

    def registration(self):
        path = self.root / 'registration.json'
        write_json(path, self.record)
        return path

    def execute(self, out):
        registration = self.registration()
        return run_source(registration, hash_file(registration), self.commit, self.root / out)

    def test_two_sources_budget_reload_and_lineage(self):
        result = self.execute('accepted')
        self.assertEqual(result['optimizer_updates'], 4)
        self.assertEqual(result['updates_per_source'], {'paviaU': 2, 'xuzhou': 2})
        self.assertEqual([event['scene'] for event in result['trace']], ['paviaU', 'xuzhou'] * 2)
        self.assertEqual([event['step'] for event in result['history']], [2, 4])
        for event in result['history']:
            for scene in event['scenes']:
                for head in scene['metrics'].values():
                    self.assertEqual(head['count'], 2)
            self.assertAlmostEqual(event['macro_fused_nll'],
                                   sum(row['metrics']['fused']['nll'] for row in event['scenes']) / 2)
        expected = min(result['history'], key=lambda event: event['macro_fused_nll'])
        self.assertEqual(result['selected_step'], expected['step'])
        self.assertNotEqual(result['initial_state_sha256'], result['selected_state_sha256'])
        out = self.root / 'accepted'
        checkpoint = out / result['selected_checkpoint']
        self.assertEqual(hash_file(checkpoint), result['checkpoint_sha256'])
        model = load_model_checkpoint(checkpoint, 'cpu')
        self.assertEqual(state_hash(model), result['selected_state_sha256'])
        lineage = json.loads((out / 'lineage.json').read_text())
        self.assertEqual(validate_lineage(self.contract, lineage, self.record['contract_sha256']),
                         {result['checkpoint_sha256']})
        row = lineage['records'][lineage['selected']]
        self.assertEqual(row['parents'], [])
        self.assertEqual(row['head']['state_sha256'], head_hash(model))
        self.assertEqual({(v['scene'], v['phase']) for v in row['access']},
                         {(scene, phase) for scene in self.contract['sources']
                          for phase in ('source_train', 'train_graph', 'fit_statistics',
                                        'validation_graph', 'source_select')})
        replay = self.execute('replay')
        for field in ('history', 'trace', 'selected_step', 'selected_state_sha256'):
            self.assertEqual(replay[field], result[field])
        prior = hash_file(out / 'result.json')
        with self.assertRaises(FileExistsError):
            self.execute('accepted')
        self.assertEqual(hash_file(out / 'result.json'), prior)

    def test_wrong_role_and_extra_source_rejected_before_updates(self):
        source = self.record['sources']['paviaU']
        source['val_labels'] = source['train_labels']
        with self.assertRaisesRegex(ValueError, 'label release hash'):
            self.execute('wrong-role')
        self.assertFalse((self.root / 'wrong-role').exists())
        self.record['sources']['whu-honghu'] = {}
        with self.assertRaisesRegex(ValueError, 'registered sources exactly'):
            self.execute('extra-source')
        self.assertFalse((self.root / 'extra-source').exists())

    def test_registration_hash_checked(self):
        path = self.registration()
        with self.assertRaisesRegex(ValueError, 'Pinned document changed'):
            run_source(path, '0' * 64, self.commit, self.root / 'wrong-registration')
        self.assertFalse((self.root / 'wrong-registration').exists())

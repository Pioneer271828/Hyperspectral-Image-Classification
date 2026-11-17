"""Paired guarded HongHu coverage engineering on synthetic released labels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch

from joint_training import _tree_fingerprint
from loadData.data_read import read_band_metadata
from loadData.dataset_registry import DATASETS
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.checkpoint_io import save_model_checkpoint
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, hash_file, tensor_hash, write_json
from train_spatial import SpatialConfig
from transfer import p4_coverage as coverage, p4_exposure, p4_guarded as guarded, p4_spatial as spatial
from transfer.g3_training import Scene, _forward
from transfer.guarded_spatial_roles import prepare_labels


CLASSES = 22
STRIDE = {'mode': 'uniform_stride', 'stride': 2, 'offset': 0}


def fixture(root):
    plan = make_tile_plan((96, 96), tile_size=32, gap=2)
    focus, left, right, *_ = [tile for tile in plan['tiles'] if tile['partition'] == 'train']
    y0, y1, x0, x1 = focus['box']
    cut = (x0 + x1 - plan['gap']) // 2
    first, second = [y0, y1, x0, cut], [y0, y1, cut + plan['gap'], x1]
    blocker = {'only_tile': focus['id'], 'class_id': CLASSES, 'available': 10}
    witness = {'status': 'feasible', 'axis': 'column', 'cut': cut,
               'first_box': first, 'second_box': second, 'validation_blocks': ['second'],
               'train_counts': [5] * CLASSES, 'validation_counts': [5] * CLASSES}
    labels = np.zeros(plan['shape'], dtype=np.int16)
    for box, classes in ((left['box'], range(1, 12)), (right['box'], range(12, CLASSES)),
                         (first, [CLASSES]), (second, range(1, CLASSES + 1))):
        yy, xx = np.mgrid[box[0]:box[1], box[2]:box[3]]
        values = np.repeat(list(classes), 5)
        labels[yy.ravel()[:len(values)], xx.ravel()[:len(values)]] = values
    roles, supports, validation = prepare_labels(labels, plan, CLASSES, blocker, witness)
    registration = {'pins': {}, 'cases': [], 'config': {
        'dataset': guarded.SCENE, 'model': {'graph_mode': 'learned', 'latent_channels': 16,
                                             'adapter_embed_dim': 8, 'learned_assignment_dim': 8},
        'training': {'steps_per_epoch': 1}, 'device': 'cpu'}}
    pins = registration['pins']

    def put(name, value):
        path = root / (name + '.json')
        write_json(path, value)
        pins[str(path)] = hash_file(path)
        return str(path)

    registration['plan'] = put('plan', plan)
    scan = {'scene': guarded.SCENE, 'blocker': blocker, 'plan_sha256': pins[registration['plan']]}
    scan_path = put('scan', scan)
    audit = {'status': 'passed', 'chosen_geometry_for_followup': witness,
             'protected_files': {scan_path: pins[scan_path]}}
    audit_path = put('audit', audit)
    geometry = {'schema_version': 1, 'kind': guarded.KIND, 'scene': guarded.SCENE,
                'parent_plan_hash': plan['plan_hash'], 'shape': plan['shape'], 'roles': roles,
                'audit_sha256': pins[audit_path]}
    geometry['development_roles_hash'] = digest(geometry)
    geometry_path = registration['development'] = put('geometry', geometry)
    runtime, allowed = guarded.validate_geometry(plan, geometry, scan, audit)
    receipt = {'status': 'prepared', 'roles_sha256': pins[geometry_path], 'releases': []}
    for (seed, k), support in supports.items():
        release = {'scene': guarded.SCENE, 'development_roles_hash': runtime['plan_hash'],
                   'support': support, 'validation': validation}
        path = put(f'release-{seed}-{k}', dict(release, release_hash=digest(release)))
        receipt['releases'].append({'file': Path(path).name, 'seed': seed, 'k': k, 'sha256': pins[path]})
        registration['cases'].extend({'release': path, 'support_seed': seed, 'k': k, 'method': method}
                                     for method in ('scratch', 'shared_transfer'))
    registration['guarded'] = {'scan_registration': scan_path, 'cut_audit': audit_path,
                               'release_receipt': put('receipt', receipt)}
    cube = np.random.default_rng(219).normal(size=(*runtime['shape'], 12)).astype(np.float32)
    stats = spatial.fit_role_statistics(cube, runtime, allowed['train'], runtime['plan_hash'])
    registration['statistics'] = put('statistics', stats)
    variable = DATASETS[guarded.SCENE][1]
    # Even the synthetic MAT includes a GT variable that the runtime must never request.
    image = str(root / 'image.mat')
    savemat(image, {variable: cube, variable + '_gt': np.full(runtime['shape'], -999)})
    registration['image'] = image
    pins[image] = hash_file(image)
    model_config = dict(registration['config']['model'], graph_mode='static')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        source = ResearchConfig.from_dict({'model': model_config}).build_model({'paviaU': 9, 'xuzhou': 9})
    weight = str(root / 'source.pt')
    save_model_checkpoint(source, weight)
    for path in (weight, weight + '.json'):
        pins[path] = hash_file(path)
    registration['source'] = {'weight': weight, 'result': put('source-result', {
        'status': 'completed', 'per_dataset_updates': {'paviaU': 2, 'xuzhou': 2}})}

    train_witness = [left['id'], right['id'], len(plan['tiles'])]
    floor = {'status': 'passed', 'protected_files': {geometry_path: pins[geometry_path]},
             'rows': []}
    for row in receipt['releases']:
        path = str(root / row['file'])
        floor['protected_files'][path] = pins[path]
        floor['rows'].append({'seed': row['seed'], 'k': row['k'], 'witness_tiles': train_witness,
                              'minimum_tiles_to_see_all_classes': len(train_witness)})
    floor_path = put('floor', floor)
    options = {'steps': 4, 'events': [3, 4], 'seed': 64}
    rows = []
    for case in registration['cases'][::2]:
        role = spatial.load_release(spatial.read(case['release']), plan, case['support_seed'], case['k'],
                                    geometry, guarded_tiles=allowed)
        rows.append(p4_exposure.cohort(floor, registration, case, role['train'], **options))
    exposure = {'policy': p4_exposure.POLICY, 'steps': options['steps'], 'validation_steps': options['events'],
                'sampler_seed': options['seed'], 'geometry_sha256': pins[geometry_path],
                'floor_sha256': pins[floor_path], 'cohorts': rows}
    registration['exposure'] = {'floor': floor_path, 'plan': put('exposure', dict(exposure, plan_hash=digest(exposure)))}
    parent = root / 'parent.json'
    write_json(parent, registration)
    bands = read_band_metadata(guarded.SCENE, cube.shape[-1])
    return parent, Scene(guarded.SCENE, cube, runtime, bands, CLASSES, stats), allowed


class MatchedCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = code_snapshot()

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parent, self.scene, self.allowed = fixture(self.root)

    def prepare(self, observation=STRIDE, metadata='nominal'):
        return coverage.prepare(self.parent, hash_file(self.parent), commit=self.code['commit'],
                                observation=observation, seed=3031, metadata=metadata)

    def run_case(self, registration, case, output, until, arm=None):
        return spatial.run_case(registration, case, output, until, self.code, coverage_arm=arm)

    def test_five_arms_share_exposure_and_exact_recovery_with_both_initializations(self):
        registration = self.prepare()
        initialization = {}
        coverage_initializations = set()
        for case in (0, 1):
            pairs = set()
            for arm in coverage.ARMS:
                with self.subTest(case=case, arm=arm):
                    full, split = [self.root / f'{case}-{arm}-{suffix}' for suffix in ('full', 'split')]
                    self.assertEqual(self.run_case(registration, case, full, 4, arm)['updates_this_call'], 4)
                    self.assertEqual(self.run_case(registration, case, split, 2, arm)['updates_this_call'], 2)
                    partial = torch.load(split / 'state.pt', weights_only=True)
                    self.assertEqual(partial['history'], [])
                    self.assertIsNone(partial['best'])
                    self.assertEqual(self.run_case(registration, case, split, 4, arm)['updates_this_call'], 2)
                    state = torch.load(full / 'state.pt', weights_only=True)
                    resumed = torch.load(split / 'state.pt', weights_only=True)
                    self.assertEqual(_tree_fingerprint(state), _tree_fingerprint(resumed))
                    paths = [split / name for name in ('state.pt', 'result.json')]
                    hashes = [hash_file(p) for p in paths]
                    self.assertEqual(self.run_case(registration, case, split, 4, arm)['updates_this_call'], 0)
                    self.assertEqual([hash_file(p) for p in paths], hashes)
                    result = spatial.read(full / 'result.json')
                    frozen = registration['coverage']
                    for field in ('records', 'steps', 'validation_steps', 'selection'):
                        self.assertEqual(result['coverage'][field], frozen[field])
                    pairs.add(result['coverage']['pair_sha256'])
                    self.assertFalse(result['query_accessed'])
                    self.assertFalse(result['raw_gt_loaded'])
                    self.assertEqual(result['exposure']['first_all_classes_step'], 3)
                    self.assertEqual([row['step'] for row in state['history']], [3, 4])
                    self.assertEqual(result['best_step'],
                                     min(state['history'], key=lambda row: row['metrics']['fused']['nll'])['step'])
                    exposure = spatial.read(registration['exposure']['plan'])['cohorts'][0]
                    self.assertEqual([row['tile'] for row in state['trace']], exposure['tile_sequence'])
                    self.assertEqual(result['exposure']['label_exposure_at_events'], exposure['label_exposure_at_events'])
                    self.assertEqual({row['coverage']['bands'] for row in state['trace']}, {6})
                    self.assertEqual({row['coverage']['sampling_sha256'] for row in state['trace']},
                                     {frozen['records']['train']['record_hash']})
                    self.assertEqual({row['coverage']['metadata_sha256'] for row in state['trace']},
                                     {frozen['forward_metadata_sha256']})
                    init = result['initialization']
                    if arm == 'none':
                        initialization[case] = init
                    for key in ('base_initial_state_sha256', 'copied_keys', 'graph_sha256',
                                'fresh_domain_sha256', 'fresh_heads_sha256'):
                        self.assertEqual(init[key], initialization[case][key])
                    added = init['coverage']
                    self.assertEqual(added['parameter_count'], 0 if arm == 'none' else 194)
                    if arm != 'none':
                        coverage_initializations.add(added['state_sha256'])
                        updates = [row['coverage'] for row in state['trace']]
                        self.assertTrue(all(row['usable_anchors'] > 0 for row in updates))
                        for key in added['parameters']:
                            self.assertTrue(any(row['parameters'][key]['gradient_l1'] > 0 for row in updates))
                            self.assertTrue(any(row['parameters'][key]['changed_elements'] > 0 for row in updates))
            self.assertEqual(len(pairs), 1)
        self.assertEqual(len(coverage_initializations), 1)
        self.assertEqual(len(initialization[0]['copied_keys']), 0)
        self.assertEqual(len(initialization[1]['copied_keys']), 29)

    def test_common_initialization_and_zero_output_with_unchanged_rng(self):
        registration = self.prepare()
        tile = self.allowed['val'][0]
        view = coverage.observation_provider(self.scene, self.allowed, registration['coverage'])(tile)
        config = SpatialConfig(registration['config'])
        source = registration['source']
        for args in ((), (source['weight'], source['result'])):
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(351)
                base, _ = spatial.initialize(config, *args)
                expected_rng = torch.get_rng_state()
                expected = _forward(base.eval(), self.scene, tile, observation_view=view)
                for arm in coverage.ARMS:
                    torch.manual_seed(351)
                    model, _ = spatial.initialize(config, *args, coverage_arm=arm)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    for key, value in base.state_dict().items():
                        self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                    actual = _forward(model.eval(), self.scene, tile, observation_view=view)
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected[:3], actual[:3])), arm)

    def test_retained_values_graph_and_predictions_ignore_deleted_columns_and_forbidden_pixels(self):
        registration = self.prepare()
        frozen = registration['coverage']
        record = frozen['records']['train']
        poison = np.full_like(self.scene.cube, np.nan)
        positions = record['selected_positions']
        for tiles in self.allowed.values():
            for tile in tiles:
                poison[tile_slice(tile)][:, :, positions] = self.scene.cube[tile_slice(tile)][:, :, positions]
        scene = replace(self.scene, cube=poison)
        self.assertEqual(spatial.fit_role_statistics(scene.cube, scene.plan, self.allowed['train'],
                                                    scene.plan['plan_hash'], positions=positions)['minimum'],
                         [scene.stats['minimum'][i] for i in positions])
        a = coverage.observation_provider(self.scene, self.allowed, frozen)
        b = coverage.observation_provider(scene, self.allowed, frozen)
        config = SpatialConfig(registration['config'])
        model, _ = spatial.initialize(config, coverage_arm='continuous')
        model.eval()
        for role in ('train', 'val'):
            tile = self.allowed[role][0]
            np.testing.assert_array_equal(a(tile).observation.spectral, b(tile).observation.spectral)
            self.assertEqual(a(tile).observation.spectral.shape[-1], len(positions))
            # Hash Q and A from the real learned graph, before the three heads.
            graph = model.backbone.learnable_graph
            build = graph.build_graph
            hashes = []

            def capture(*args, **kwargs):
                state = build(*args, **kwargs)
                hashes.append((tensor_hash(state.Q), tensor_hash(state.A0)))
                return state

            graph.build_graph = capture
            try:
                with torch.no_grad():
                    original = _forward(model, self.scene, tile, observation_view=a(tile))
                    poisoned = _forward(model, scene, tile, observation_view=b(tile))
            finally:
                del graph.build_graph
            self.assertEqual(len(hashes), 2)
            self.assertEqual(hashes[0], hashes[1])
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(original[:3], poisoned[:3])))
        forbidden = next(tile for tile in scene.plan['tiles'] if tile['partition'] == 'test')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            b(forbidden)

    def test_nominal_missing_and_shifted_metadata_contract(self):
        variants = {mode: self.prepare(metadata=mode) for mode in coverage.METADATA_MODES}
        nominal, missing, shifted = [variants[mode]['coverage'] for mode in coverage.METADATA_MODES]
        self.assertEqual(nominal['records'], missing['records'])
        self.assertEqual(nominal['records'], shifted['records'])
        self.assertIsNone(missing['forward_metadata_sha256'])
        self.assertNotEqual(nominal['forward_metadata_sha256'], shifted['forward_metadata_sha256'])
        tile = self.allowed['val'][0]
        a, b, c = [coverage.observation_provider(self.scene, self.allowed, v)(tile)
                   for v in (nominal, missing, shifted)]
        np.testing.assert_array_equal(a.observation.spectral, b.observation.spectral)
        np.testing.assert_array_equal(a.observation.spectral, c.observation.spectral)
        self.assertEqual(a.bands.band_ids, c.bands.band_ids)
        self.assertEqual(a.bands.wavelength_quality, c.bands.wavelength_quality)
        np.testing.assert_allclose(np.asarray(c.bands.wavelengths) - a.bands.wavelengths, 10)
        config = SpatialConfig(variants['nominal']['config'])
        none, _ = spatial.initialize(config)
        active, _ = spatial.initialize(config, coverage_arm='continuous')
        none.eval()
        active.eval()
        with torch.no_grad():
            active.backbone.coverage_modulation.generator[-1].bias.fill_(0.25)
            ref = _forward(none, self.scene, tile, observation_view=b, metadata_missing=True)
            fallback = _forward(active, self.scene, tile, observation_view=b, metadata_missing=True)
            perturbed = _forward(active, self.scene, tile, observation_view=c)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ref[:3], fallback[:3])))
        self.assertEqual(fallback[3]['coverage']['fallback_reason'], 'missing_metadata')
        self.assertEqual(perturbed[3]['coverage']['metadata_hash'], c.bands.fingerprint)
        output = self.root / 'missing'
        self.run_case(variants['missing'], 0, output, 4, 'shuffled')
        state = torch.load(output / 'state.pt', weights_only=True)
        for row in state['trace']:
            self.assertEqual(row['coverage']['fallback_reason'], 'missing_metadata')
            self.assertEqual(row['coverage']['usable_anchors'], 0)
            self.assertTrue(all(v == {'gradient_l1': None, 'changed_elements': 0}
                                for v in row['coverage']['parameters'].values()))

    def test_wrong_arm_budget_metadata_and_label_release_are_rejected(self):
        registration = self.prepare()
        output = self.root / 'partial'
        with self.assertRaisesRegex(ValueError, 'explicitly registered'):
            self.run_case(registration, 0, output, 2)
        with self.assertRaisesRegex(ValueError, 'exposure budget'):
            self.run_case(registration, 0, output, 5, 'none')
        self.run_case(registration, 0, output, 2, 'continuous')
        state_path = output / 'state.pt'
        prior = hash_file(state_path)
        with self.assertRaisesRegex(ValueError, 'Resume identity'):
            self.run_case(registration, 0, output, 4, 'shuffled')
        with self.assertRaisesRegex(ValueError, 'Resume identity'):
            self.run_case(registration, 1, output, 4, 'continuous')
        with self.assertRaisesRegex(ValueError, 'Resume identity'):
            self.run_case(self.prepare(observation=dict(STRIDE, offset=1)), 0, output, 4, 'continuous')
        with self.assertRaisesRegex(ValueError, 'Resume identity'):
            self.run_case(self.prepare(metadata='shift10nm'), 0, output, 4, 'continuous')
        for field, value in (('steps', 5), ('forward_metadata_sha256', 'a' * 64),
                             ('arms', ['none', 'continuous'])):
            changed = deepcopy(registration)
            changed['coverage'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'metadata or exposure budget'):
                self.run_case(changed, 0, output, 4, 'continuous')
        self.assertEqual(hash_file(state_path), prior)

        case = registration['cases'][0]
        release_path = Path(case['release'])
        release = spatial.read(release_path)
        tile = next(tile for tile in self.scene.plan['tiles'] if tile['partition'] == 'test')
        y, x = tile['box'][0], tile['box'][2]
        release['support']['indices'][0] = y * self.scene.plan['shape'][1] + x
        release['support']['indices'].sort()
        release['release_hash'] = digest({key: value for key, value in release.items() if key != 'release_hash'})
        write_json(release_path, release)
        with self.assertRaisesRegex(ValueError, 'frozen input changed'):
            self.run_case(registration, 0, output, 4, 'continuous')
        registration['pins'][case['release']] = hash_file(release_path)
        receipt_path = registration['guarded']['release_receipt']
        receipt = spatial.read(receipt_path)
        receipt['releases'][0]['sha256'] = hash_file(release_path)
        write_json(receipt_path, receipt)
        registration['pins'][receipt_path] = hash_file(receipt_path)
        with self.assertRaisesRegex(ValueError, 'escapes the registered spatial role'):
            self.run_case(registration, 0, output, 4, 'continuous')
        self.assertEqual(hash_file(state_path), prior)

    def test_full_nominal_none_matches_previous_exposure_path(self):
        legacy, new = [self.root / name for name in ('previous', 'matched')]
        self.run_case(spatial.read(self.parent), 0, legacy, 4)
        with patch.object(spatial, 'loadmat', wraps=spatial.loadmat) as loader:
            self.run_case(self.prepare(observation={'mode': 'none'}), 0, new, 4, 'none')
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs['variable_names'], [DATASETS[guarded.SCENE][1]])
        a, b = [torch.load(folder / 'state.pt', weights_only=True) for folder in (legacy, new)]
        for key in ('model', 'optimizer', 'scheduler', 'rng', 'history', 'best'):
            self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(b[key]), key)
        self.assertEqual(a['trace'], [{key: value for key, value in row.items() if key != 'coverage'}
                                      for row in b['trace']])

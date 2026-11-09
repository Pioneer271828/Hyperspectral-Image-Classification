"""Frozen, role-bound real-band schedules for opt-in development training (M1-B)."""
from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np

from loadData.observation_sampling import BandMetadata, sample, replay, subset_statistics
from loadData.spatial_protocol import PartitionObservation
from loadData.spatial_tiles import observe_tile
from model.research_config import digest


def require(condition, message):
    if not condition:
        raise ValueError('Observation policy: ' + message)


def scene_identity(scene, record):
    from transfer.development_training import validate
    validate(record, scene.plan)
    require(type(scene.classes) is int and scene.classes==len(record['classes']), 'scene/record class counts differ')
    require(scene.stats.get('fit_role') == 'devtrain'
            and scene.stats.get('development_hash') == record['validation_hash']
            and scene.stats.get('statistics_hash') == digest({k: v for k, v in scene.stats.items() if k != 'statistics_hash'}),
            'requires unchanged devtrain-only normalization')
    require(scene.bands.wavelength_unit == 'nm' and scene.bands.wavelength_quality != 'index_only',
            'this integration requires explicit physical centers; approximate stays approximate')
    return {'plan_hash': scene.plan['plan_hash'], 'development_hash': record['validation_hash'],
            'statistics_hash': scene.stats['statistics_hash'], 'metadata_hash': scene.bands.fingerprint,
            'dataset': scene.dataset, 'classes': scene.classes}


def default_conditions(bands):
    """Development perturbations, not identities of real alternative sensors."""
    wave = np.asarray(bands.wavelengths)[np.asarray(bands.valid)]
    low, high = float(wave.min()), float(wave.max())
    return {'full': {'mode': 'none'},
            **{f'random_{p}': {'mode': 'random_subset', 'keep_fraction': p / 100} for p in (75, 50, 25)},
            'gap': {'mode': 'contiguous_gap', 'lower': low + .375 * (high-low),
                    'upper': low + .625 * (high-low), 'unit': bands.wavelength_unit},
            'stride': {'mode': 'uniform_stride', 'stride': 2, 'offset': 0}}


class ObservationPolicy:
    """No RNG at replay; private copied records prevent accidental caller mutation."""
    def __init__(self, value):
        self._value = deepcopy(value)
        v = self._value
        require(v.get('policy_hash') == digest({k: x for k, x in v.items() if k != 'policy_hash'}), 'hash mismatch')
        require(v.get('schema_version') == 1 and type(v.get('updates')) is int and v['updates'] > 0,
                'invalid schema/budget')
        require(len(v['training']) == v['updates'] and 'full' in v['validation']
                and v['validation']['full']['config'] == {'mode': 'none'}, 'missing full validation or schedule')
        bands = BandMetadata(**v['bands'])
        require(self.training_application in ('real_subset', 'support_zero'), 'unknown training application')
        require(self.training_application != 'support_zero' or all(bands.valid), 'zero control requires all-valid source columns')
        require(bands.fingerprint == v['identity']['metadata_hash'], 'metadata identity mismatch')
        stub = self._stub(bands, v['identity']['plan_hash'])
        for r in [*v['training'], *v['validation'].values()]:
            replay(stub, bands, r)

    @staticmethod
    def _stub(bands, plan_hash):
        return PartitionObservation('train', np.zeros((1, 1, len(bands.band_ids)), dtype=np.float32),
                                    np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int64), plan_hash)

    @classmethod
    def freeze(cls, scene, record, *, updates, training_configs, train_seed, validation_seed, conditions=None, training_application='real_subset'):
        require(training_application in ('real_subset', 'support_zero'), 'unknown training application')
        require(type(updates) is int and updates > 0 and training_configs, 'positive budget and explicit training configs required')
        require(all(type(s) is int and 0 <= s < 2**32 for s in (train_seed, validation_seed))
                and train_seed != validation_seed, 'separate uint32 train/validation seeds required')
        identity = scene_identity(scene, record)
        stub = cls._stub(scene.bands, scene.plan['plan_hash'])
        # Separate local generators; neither touches model or spatial sampler randomness.
        training_rng = np.random.default_rng(train_seed)
        training = [sample(stub, scene.bands, training_configs[i % len(training_configs)], training_rng)[1]
                    for i in range(updates)]
        configs = default_conditions(scene.bands) if conditions is None else deepcopy(conditions)
        require(isinstance(configs, dict) and configs.get('full') == {'mode': 'none'}, 'full condition required')
        require(all(isinstance(k, str) and k for k in configs), 'named validation conditions required')
        validation = {}
        for name, config in sorted(configs.items()):
            # Each named condition is independent of train budget and other conditions/order.
            seed = int(digest({'seed': validation_seed, 'name': name})[:16], 16)
            validation[name] = sample(stub, scene.bands, config, np.random.default_rng(seed))[1]
        value = {'schema_version': 1, 'identity': identity, 'bands': asdict(scene.bands), 'updates': updates,
                 'train_seed': train_seed, 'validation_seed': validation_seed,
                 'training_configs': deepcopy(training_configs), 'training': training, 'validation': validation,
                 'selection_condition': 'full', 'schedule_rule': 'cycle explicit configs; per-update frozen indices'}
        # Omit the default field to preserve the exact accepted M1-B policy hashes.
        if training_application != 'real_subset': value['training_application'] = training_application
        value['policy_hash'] = digest(value)
        return cls(value)

    @property
    def policy_hash(self): return self._value['policy_hash']

    @property
    def conditions(self): return tuple(self._value['validation'])

    @property
    def training_application(self): return self._value.get('training_application', 'real_subset')

    def to_dict(self): return deepcopy(self._value)

    def validate(self, scene, record, updates=None):
        require(scene_identity(scene, record) == self._value['identity'], 'scene/roles/statistics differ')
        require(updates is None or updates == self._value['updates'], 'budget differs')

    def view(self, scene, record, tile, *, phase, step=None, condition='full'):
        require(phase in ('training', 'validation'), 'only development train/validation roles allowed')
        require(scene.bands.fingerprint == self._value['identity']['metadata_hash']
                and scene.stats['statistics_hash'] == self._value['identity']['statistics_hash']
                and record['validation_hash'] == self._value['identity']['development_hash'], 'scene identity differs')
        require(type(tile.get('id')) is int and 0 <= tile['id'] < len(record['roles']), 'unknown tile')
        require(record['roles'][tile['id']]['role'] == ('devtrain' if phase == 'training' else 'devval'), 'role access denied')
        if phase == 'training':
            require(type(step) is int and 1 <= step <= self._value['updates'], 'update outside frozen schedule')
            frozen = self._value['training'][step - 1]
        else:
            require(step is None and condition in self.conditions, 'unknown validation condition')
            frozen = self._value['validation'][condition]
        obs = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        if phase == 'training' and self.training_application == 'support_zero':
            # Matched historical mechanism: only released train-support pixels are
            # zeroed; all context pixels, B, centers and coordinates remain present.
            full, _ = sample(obs, scene.bands, {'mode': 'none'})
            values = np.array(obs.spectral, copy=True)
            rows = np.flatnonzero(np.isin(obs.global_ids.reshape(-1), record['train_support']['indices']))
            dropped = np.setdiff1d(np.arange(len(scene.bands.band_ids)), frozen['selected_positions'])
            values.reshape(-1, values.shape[-1])[np.ix_(rows, dropped)] = 0.
            values.setflags(write=False)
            return replace(full, observation=replace(obs, spectral=values))
        return replay(obs, scene.bands, frozen)

    def training_record(self, step):
        require(type(step) is int and 1 <= step <= self._value['updates'], 'update outside frozen schedule')
        r = self._value['training'][step - 1]
        result = {'policy_hash': self.policy_hash, 'record_hash': r['record_hash'],
                  'selected_band_count': len(r['selected_positions'])}
        if self.training_application == 'support_zero':
            result.update(training_application='support_zero', selected_band_count=r['source_band_count'],
                          retained_band_count=len(r['selected_positions']))
        return result

    def selected_statistics(self, scene, *, step):
        """Derived slices only; never pass them back as full-scene normalization."""
        require(self.training_application == 'real_subset', 'zero control retains full training statistics')
        self.training_record(step)
        return subset_statistics(scene.stats, scene.bands, self._value['training'][step - 1])


def evaluate_conditions(model, scene, record, policy):
    """Fixed released-devval, three-head readouts; never use for automatic selection."""
    import torch
    from evaluation.classification_metrics import metrics_from_confusion
    from loadData.spatial_tiles import tile_slice
    from transfer.development_training import sparse_labels, development_counts
    from transfer.g3_training import _forward, state_hash
    policy.validate(scene, record)
    _,validation_count=development_counts(record)
    expected = getattr(model, 'training_config', {}).get('observation_policy_hash')
    require(expected is None or expected == policy.policy_hash, 'checkpoint policy differs')
    labels = sparse_labels(record, scene.plan, 'validation_support'); before = state_hash(model)
    results = {}; model.eval()
    with torch.no_grad():
        for condition in policy.conditions:
            matrices = [np.zeros((scene.classes, scene.classes), dtype=np.int64) for _ in range(3)]
            nll = np.zeros(3); both = np.zeros(scene.classes, dtype=np.int64)
            for tile in scene.plan['tiles']:
                if record['roles'][tile['id']]['role'] != 'devval': continue
                target = labels[tile_slice(tile)].reshape(-1); valid = target > 0
                if not valid.any(): continue
                truth = target[valid] - 1
                view = policy.view(scene, record, tile, phase='validation', condition=condition)
                outputs = _forward(model, scene, tile, observation_view=view); predictions = []
                for i, output in enumerate(outputs[:3]):
                    probs = output.detach().cpu().numpy()[valid]; prediction = probs.argmax(1); predictions.append(prediction)
                    matrices[i] += np.bincount(truth*scene.classes + prediction, minlength=scene.classes**2).reshape(scene.classes, scene.classes)
                    nll[i] -= np.log(np.maximum(probs[np.arange(len(truth)), truth], 1e-8)).sum()
                wrong = (predictions[1] != truth) & (predictions[2] != truth)
                both += np.bincount(truth[wrong], minlength=scene.classes)
            require(all(m.sum() == validation_count and m.sum(1).tolist() == [5]*scene.classes for m in matrices), 'development support differs')
            results[condition] = {h: {**metrics_from_confusion(m, list(range(1, scene.classes+1))), 'nll': float(nll[i]/validation_count)}
                                  for i, (h, m) in enumerate(zip(('fused', 'graph', 'cnn'), matrices))}
            results[condition]['both_wrong_by_class'] = (both/5).tolist()
    require(before == state_hash(model), 'evaluation mutated model state')
    return {'policy_hash': policy.policy_hash, 'selection_condition': 'full', 'test_evaluated': False,
            'scope': 'frozen development perturbations, not real alternative sensors', 'conditions': results}

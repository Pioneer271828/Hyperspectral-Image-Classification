"""Exact-position constructor/provider adapter; the existing native lifecycle remains authoritative."""
from copy import deepcopy
from itertools import product

import torch

from model.research_config import digest
from transfer import wp07_condition_run as previous
from transfer import wp07_fine16_training as parent
from transfer import wp07_graph_range as graph
from transfer import wp07_exact_positions as positions
from transfer.development_bank_training import _initial
from transfer.g3_training import state_hash
from transfer.native_scale_training import baseline
from transfer import native_released_training

FORMAT = 'wp07_exact_positions_synthetic_continuation_v1'
ARMS = tuple('both.' + mode for mode in ('mean', 'content', 'metadata', 'joint'))


class FixtureProvider(previous.FixtureProvider):
    def __init__(self, classes, representation):
        graph.require(representation in positions.REPRESENTATIONS, 'Unknown representation')
        self.classes = classes
        self.representation = representation
        self.fixtures = {role: positions.ControlledFixture(classes, role) for role in ('devtrain', 'devval')}
        self.catalog = digest(positions.catalog(classes))
        self.override = 'matched'

    def events(self):
        train = []
        for tile in self.fixtures['devtrain'].tiles:
            for band in positions.POSITION_SETS:
                train.append(dict(step=len(train) + 1, tile=tile, factor=1,
                                  band=band, profile='a', condition='matched'))
        val = [dict(tile=tile, factor=1, band=band, profile='b', condition='matched')
               for tile, band in product(self.fixtures['devval'].tiles, positions.POSITION_SETS)]
        return train, val, [0, 12, 24]

    def materialize(self, role, event, producer):
        graph.require(role in self.fixtures, 'No held-out or query role')
        return native_released_training.materialize(self.fixtures[role], event, producer)

    def selection_condition(self, actual, spec):
        return positions.bind(actual, self.override)


def cache(provider, producer):
    # Capture each derived observation's own graph; never transplant the parent's support.
    return previous.cache(provider, producer)


def contract(provider, seed, arm, producer, numeric, cache_pin, authority):
    graph.require(provider.synthetic and type(provider) is FixtureProvider
                  and seed in (64, 512, 2048) and arm in ARMS,
                  'Only preregistered synthetic position integration may update')
    source = provider.baseline(seed)
    initial = positions.initialize(source, seed + 7321)
    train, val, events = provider.events()
    inputs = {}
    for role, roster in (('devtrain', train), ('devval', val)):
        for event in roster:
            actual, labels, points = provider.materialize(role, event, producer)
            inputs[previous.key(role, event)] = provider.evidence(actual, labels, points)
    return dict(format=FORMAT, producer=producer, classes=provider.classes, seed=seed, arm=arm,
                representation=provider.representation, supervision=parent.MODE, objective=parent.LOSS,
                training=train, validation=val, steps=24, prefix=12, events=events,
                numeric=deepcopy(numeric), module_seed=seed + 7321, rng_seed=seed + 15000,
                learning_rate=.0001, initialization=_initial(initial, source), source_state=state_hash(source),
                parameters=sum(p.numel() for p in initial.parameters()), registered_graph_scale=16,
                cache=cache_pin, authority=authority, synthetic=True, inputs=inputs,
                condition_catalog=provider.catalog, selection='first_min_released_point_fused_nll',
                common_candidates={'cnn': [1, 2, 3], 'graph': [1, 2]},
                condition_semantics='declared_processed_columns_not_authenticated_sensor_mapping')


class PositionRun(previous.ConditionRun):
    def __init__(self, spec, provider, entries, producer, execution):
        expected = contract(provider, spec['seed'], spec['arm'], producer, execution.numeric,
                            spec['cache'], spec['authority'])
        graph.require(spec == expected, 'Position registration, source, representation or budget changed')
        self._contract, self.producer, self.execution = deepcopy(spec), producer, execution
        self._released, self._entries = provider, entries
        self.device = execution.device
        self.model = positions.initialize(provider.baseline(spec['seed']), spec['module_seed']).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=spec['learning_rate'], foreach=False, fused=False)
        self._train, self._val = 'devtrain', 'devval'
        self._history, self._validation, self._selected = [], [], None
        self._fresh = tuple(spec['initialization']['fresh'])
        self._capacity, self._region_action = None, None
        self._selection_override, self._policy_override = None, None
        self.timing = dict(train_update_seconds=0., validation_seconds=0., transform_seconds=0.,
                           forward_seconds=0., graph_seconds=0.)

    def _forward(self, actual, role, intervention):
        with positions.representation(self.model, self._contract['representation']):
            output, trace, condition = super()._forward(actual, role, intervention)
        trace['position_representation'] = self._contract['representation']
        trace['condition_semantics'] = self._contract['condition_semantics']
        return output, trace, condition

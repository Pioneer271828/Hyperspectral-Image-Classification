"""Private processed-index representations and label-blind controlled synthetic views."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from loadData.observation_sampling import _apply
from model.research_config import digest
from transfer import wp07_observed_conditions as prior
from transfer.native_scale_fixture import Fixture, registration
from transfer.native_scale_observation import _materialize
from transfer.native_scale_supervision import complete_targets
from transfer.wp07_graph_range import require

MAX_BANDS = 436
REPRESENTATIONS = ('summary8', 'exact_mask')
POSITION_SETS = {
    'shift_a': (0, 2, 4, 6, 8, 10, 12, 14),
    'shift_b': (1, 3, 5, 7, 9, 11, 13, 15),
    'interior_a': (0, 1, 2, 3, 4, 12, 17, 19),
    'interior_b': (0, 1, 2, 3, 5, 10, 18, 19),
}


def mask_features(positions, master_count):
    prior.index_features(positions, master_count)
    require(master_count <= MAX_BANDS and tuple(positions) == tuple(sorted(positions)),
            'Ordered processed positions exceed bounded mask namespace')
    present = set(positions)
    return (master_count / MAX_BANDS, *(float(i in present) for i in range(MAX_BANDS)))


class PositionEncoder(prior.IndexConditionEncoder):
    def __init__(self, old):
        # Retain exactly the already initialized physical and summary tensors.
        nn.Module.__init__(self)
        self.physical = old.physical
        self.indices = old.indices
        self.exact_mask = nn.Sequential(nn.Linear(MAX_BANDS + 1, 32), nn.GELU())
        self.representation = 'summary8'

    def forward(self, packet, coverage_fraction, gsd_m, reference):
        require(self.representation in REPRESENTATIONS, 'Unknown position representation')
        if packet is None:
            return self.physical(None, coverage_fraction, gsd_m, reference)
        require(isinstance(packet, prior.IndexPacket)
                and not packet.original_physical_mapping_authenticated,
                'Only declared processed-index packets are admitted')
        features = prior.index_features(packet.positions, packet.master_count)
        require(coverage_fraction is None or coverage_fraction == features[0], 'Coverage differs')
        physical = self.physical(packet.bands, None, gsd_m, reference)
        if self.representation == 'summary8':
            observed = self.indices(reference.new_tensor(features))
        else:
            observed = self.exact_mask(reference.new_tensor(mask_features(packet.positions, packet.master_count)))
        return observed if physical is None else observed + physical


def encoders(model):
    for owner, name in ((model.backbone.CNN_Branch[0], 'condition_encoder'),
                        (model.backbone, 'range_encoder')):
        value = getattr(owner, name, None)
        if isinstance(value, PositionEncoder):
            yield value


def extend(model, seed):
    """Extend existing index encoders without reseeding CUDA or changing old tensors."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for owner, name in ((model.backbone.CNN_Branch[0], 'condition_encoder'),
                            (model.backbone, 'range_encoder')):
            old = getattr(owner, name, None)
            if old is not None:
                require(type(old) is prior.IndexConditionEncoder, 'Expected one unextended index encoder')
                setattr(owner, name, PositionEncoder(old))
    return model.to(next(model.parameters()).device)


def initialize(baseline, seed):
    return extend(prior.initialize(baseline, seed), seed + 27183)


@contextmanager
def representation(model, value):
    require(value in REPRESENTATIONS, 'Unknown position representation')
    layers = list(encoders(model))
    require(layers, 'No registered position encoders')
    before = [layer.representation for layer in layers]
    try:
        for layer in layers:
            layer.representation = value
        yield
    finally:
        for layer, previous in zip(layers, before, strict=True):
            layer.representation = previous


def catalog(classes):
    require(classes in (9, 22), 'Only registered synthetic class counts')
    master = 20 if classes == 9 else 24
    return dict(schema='wp07_controlled_processed_positions_v1', classes=classes,
                parent_fixture=registration(classes), master_count=master,
                positions={key: list(value) for key, value in POSITION_SETS.items()},
                physical_metadata='withheld identically from selector, no synthetic physical cue',
                roles=['devtrain', 'devval'], factor=1)


@dataclass(frozen=True)
class PositionCondition:
    catalog: str
    observation: str
    master_count: int
    intervention: str
    packet: prior.IndexPacket | None
    source_key: tuple
    gsd_m: None = None

    def values(self):
        return self.packet, None, None

    def validate(self, actual, receipt):
        require(self == bind(actual, self.intervention), 'Controlled descriptor or source changed')


def bind(actual, intervention='matched'):
    info = actual.receipt
    source = info['source']
    config = catalog(actual.classes)
    require(info['synthetic_profile'] is not None and source['position_catalog'] == digest(config)
            and source['positions'] == config['positions'][source['band_view']],
            'Controlled view lacks registered synthetic input identity')
    require(intervention in ('matched', 'paired', 'missing'), 'Unknown position intervention')
    name = source['band_view']
    donor = name[:-1] + ('b' if name.endswith('a') else 'a') if intervention == 'paired' else name
    packet = None if intervention == 'missing' else prior.IndexPacket(None, POSITION_SETS[donor], config['master_count'])
    return PositionCondition(digest(config), info['observation'], config['master_count'], intervention,
                             packet, (digest(config), intervention, donor))


class ControlledFixture:
    def __init__(self, classes, role):
        require(role in ('devtrain', 'devval'), 'No confirmation or query role')
        self.config = catalog(classes)
        self.parent = Fixture(classes, role)
        # This is a separately versioned synthetic generator, never a real bank release.
        self.parent._registration = deepcopy(self.parent._registration)
        self.parent._registration['count'] = self.config['master_count']
        self.parent._registration['statistics']['minimum'] = [-1.] * self.config['master_count']
        self.parent._registration['statistics']['maximum'] = [2.] * self.config['master_count']
        self.parent._plan = digest(self.parent._registration)

    @property
    def tiles(self):
        return self.parent.tiles

    def materialize(self, event, producer):
        require(event['band'] in POSITION_SETS and event['factor'] == 1, 'Unregistered controlled view')
        original, labels, _ = self.parent.materialize(event['tile'], 1, 'full', event['profile'], producer)
        positions = POSITION_SETS[event['band']]
        view = _apply(original.view.observation, original.view.bands, positions)
        source = {**original.receipt['source'], 'positions': list(positions),
                  'parent_observation': original.receipt['observation'],
                  'position_catalog': digest(self.config), 'band_view': event['band'],
                  'selection': digest(dict(parent=original.receipt['observation'], positions=positions))}
        physical = self.parent._physical[event['profile']]
        actual = _materialize(original.dataset, original.classes, view, original.lattice,
                              source, original.receipt['role'], producer, profile=physical)
        # Complete synthetic labels remain engineering-only and use the existing native mapper.
        return actual, labels, complete_targets(actual, labels)

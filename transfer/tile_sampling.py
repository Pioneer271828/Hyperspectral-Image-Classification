"""Support-only tile sampling; no image, validation or test labels are accepted."""
import numpy as np
from loadData.spatial_tiles import tile_slice
from model.research_config import digest


class SupportTileSampler:
    MODES = ('uniform', 'class_first')

    def __init__(self, plan, roles, labels, classes, mode='uniform'):
        if mode not in self.MODES:
            raise ValueError('Unknown support tile sampler')
        self.mode = mode
        self.classes = list(classes)
        if self.classes != list(range(1, len(self.classes) + 1)) or not self.classes:
            raise ValueError('Require contiguous positive classes')
        self.tiles = [plan['tiles'][t['id']] for t in roles
                      if t['role'] == 'devtrain' and (labels[tile_slice(t)] > 0).any()]
        self.counts = np.asarray([
            [int((labels[tile_slice(t)] == c).sum()) for c in self.classes]
            for t in self.tiles], dtype=np.int64)
        if not self.tiles or (self.counts.sum(axis=0) == 0).any():
            raise ValueError('Every class needs released training support')
        self.by_class = [np.flatnonzero(self.counts[:, i]) for i in range(len(self.classes))]
        probabilities = np.full(len(self.tiles), 1 / len(self.tiles))
        if mode == 'class_first':
            probabilities = np.zeros(len(self.tiles))
            for indices in self.by_class:
                probabilities[indices] += 1 / (len(self.classes) * len(indices))
        self.metadata = {
            'mode': mode, 'classes': self.classes,
            'rule': 'uniform tile' if mode == 'uniform' else 'uniform class then uniform tile containing that class',
            'supervision': 'all released training labels in sampled tile',
            'tile_ids': [t['id'] for t in self.tiles],
            'tile_class_counts': self.counts.tolist(),
            'tile_probabilities': probabilities.tolist(),
            'expected_class_participation_per_update': (probabilities @ (self.counts > 0)).tolist(),
            'expected_label_exposure_per_update': (probabilities @ self.counts).tolist(),
        }
        self.metadata['sampler_hash'] = digest(self.metadata)

    def draw(self, rng):
        if self.mode == 'uniform':
            # One integer draw preserves the legacy uniform sampler sequence.
            return self.tiles[int(rng.integers(len(self.tiles)))], None
        index = int(rng.integers(len(self.classes)))
        choices = self.by_class[index]
        return self.tiles[int(choices[int(rng.integers(len(choices)))])], self.classes[index]

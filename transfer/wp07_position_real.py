"""Read-only bank-to-provider gate. API slicing capability is not a new view release."""
from copy import deepcopy

from transfer.wp07_fine16_training import BankProvider
from transfer.wp07_graph_range import require


class RegisteredPositionProvider:
    synthetic = False

    def __init__(self, root, preflight, seed):
        self.provider = BankProvider(root, preflight, seed)
        self.preflight = deepcopy(preflight)
        self.classes = self.provider.classes
        self._registry = {(row['role'], row['tile'], row['band'], row['factor']): row
                          for row in self.preflight['views']}

    def request(self, role, event, producer, *, positions=None):
        key = (role, event.get('tile'), event.get('band'), event.get('factor'))
        require(key in self._registry, 'Position view is not in the admitted real preflight')
        reference = self._registry[key]
        if positions is not None:
            require(list(positions) == reference['source']['positions'],
                    'New real position set requires separate view admission, before payload access')
        actual, labels, points = self.provider.materialize(role, event, producer)
        self.provider.validate_observation(None, actual, labels, points, event)
        return actual, labels, points

    def materialize(self, role, event, producer):
        return self.request(role, event, producer)

    def contract(self, *args, **kwargs):
        raise ValueError('Read-only real position provider has no optimizer release')

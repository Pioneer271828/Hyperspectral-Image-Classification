"""Equal-budget restart from the original source with frozen deletion topology.

The selected donor proposes topology only; its optimization cost is separate.
Edge values and assignments remain trainable. No query or implicit launch.
"""
from copy import deepcopy
from evaluation.graph_edge_intervention import _selected, _probe
from evaluation.matched_edge_controls import freeze, verify
from evaluation.matched_edge_review import coordinates
from model.research_config import digest
from transfer.graph_route_context import RouteContextBank, _context, route_state_hash
from transfer.graph_route_training import GraphRouteRun


class MatchedRetrainingPlan:
    def __init__(self, value, *, expected_hash):
        self._value = deepcopy(value)
        if (set(value) != {'version', 'donor_contract', 'source_state', 'selected_state',
                           'selected_step', 'base_bank', 'contexts', 'names', 'plan_hash'}
                or value['version'] != 1 or value['plan_hash'] != expected_hash
                or digest({k:v for k,v in value.items() if k != 'plan_hash'}) != expected_hash):
            raise ValueError('Matched retraining plan identity differs')
        self.plan_hash = expected_hash

    def to_dict(self): return deepcopy(self._value)

    @classmethod
    def freeze(cls, donor, *, per_row, distance_bins, seeds, time_limit=30.):
        model = _selected(donor)
        if donor.bank is not None: raise ValueError('Dynamic donor required; fixed donors need a separate protocol')
        state = route_state_hash(model)
        ids = [t['id'] for t in donor.record['roles'] if t['role'] in ('devtrain', 'devval')]
        bank = RouteContextBank.freeze(model, donor.scene, donor.record, tile_ids=ids,
                                       step=0, expected_source_state=state)
        rows = []
        for i in ids:
            context = _context(donor.scene, donor.record, i)
            _, (graph, _) = _probe(donor, model, i)
            control = freeze(graph, coordinates(context), per_row=per_row,
                             distance_bins=distance_bins, seeds=seeds, time_limit=time_limit)
            rows.append({'context':context, 'control_plan':control})
        donor._check()
        if route_state_hash(model) != state: raise ValueError('Donor changed during freeze')
        value = dict(version=1, donor_contract=donor.contract_hash, source_state=route_state_hash(donor.source),
                     selected_state=state, selected_step=donor.selected_step, base_bank=bank.to_dict(),
                     contexts=rows, names=['none','important','low_weight',*[f'random:{s}' for s in seeds]])
        value['plan_hash'] = digest(value)
        return cls(value, expected_hash=value['plan_hash'])

    def _validate(self, donor, expected_hash):
        value = MatchedRetrainingPlan(self.to_dict(), expected_hash=expected_hash).to_dict()
        model = _selected(donor)
        if (donor.bank is not None or value['donor_contract'] != donor.contract_hash
                or value['source_state'] != route_state_hash(donor.source)
                or value['selected_state'] != route_state_hash(model)
                or value['selected_step'] != donor.selected_step):
            raise ValueError('Donor/source/selected state differs')
        contexts = [_context(donor.scene, donor.record, t['id']) for t in donor.record['roles']
                    if t['role'] in ('devtrain','devval')]
        if [r['context'] for r in value['contexts']] != contexts:
            raise ValueError('Complete training and validation context population required')
        original = RouteContextBank.freeze(model, donor.scene, donor.record,
                    tile_ids=[c['tile_id'] for c in contexts], step=0, expected_source_state=value['selected_state'])
        if value['base_bank'] != original.to_dict(): raise ValueError('Donor topology differs')
        settings = None
        for row in value['contexts']:
            p = row['control_plan'];names = ['none','important','low_weight',*[f'random:{s}' for s in p['seeds']]]
            if value['names'] != names or set(p['controls']) != set(names[2:]): raise ValueError('Control population differs')
            current = (p['per_row'],p['distance_bins'],p['seeds'])
            if settings is not None and settings != current: raise ValueError('Control settings differ across blocks')
            settings = current
            _, (graph, _) = _probe(donor, model, row['context']['tile_id'],expected_graph_hash=p['adjacency_sha256'])
            verify(p, graph, coordinates(row['context']), expected_hash=p['plan_hash'])
        donor._check()
        return value

    def bank(self, donor, name, *, expected_hash):
        value = self._validate(donor, expected_hash)
        if name not in value['names']: raise ValueError('Unregistered control')
        return self._materialize(value, name)

    @staticmethod
    def _materialize(value, name):
        base = deepcopy(value['base_bank'])
        for row in value['contexts']:
            p = row['control_plan']
            if name not in ('none','important') and p['controls'][name]['status'] != 'optimal':
                raise ValueError('Whole control unavailable: at least one block lacks a matched mask')
            removed = [] if name == 'none' else p['baseline']['removed'] if name == 'important' else p['controls'][name]['removed']
            fixed = base['contexts'][digest(row['context'])]['fixed']
            fixed['selected'] = sorted(set(fixed['selected']) - set(removed))
        base['bank_hash'] = digest({k:v for k,v in base.items() if k != 'bank_hash'})
        return RouteContextBank(base, expected_hash=base['bank_hash'])

    def banks(self, donor, *, expected_hash):
        """Export all feasible arms after a single full-population validation."""
        value = self._validate(donor, expected_hash)
        result, unavailable = {}, {}
        for name in value['names']:
            reasons = [{"tile_id":row["context"]["tile_id"],**row["control_plan"]["controls"][name]}
                       for row in value["contexts"] if name in row["control_plan"]["controls"]
                       and row["control_plan"]["controls"][name]["status"]!="optimal"]
            if reasons:
                unavailable[name] = reasons
            else:
                result[name] = self._materialize(value,name)
        return result, unavailable

    def start(self, donor, name, *, expected_hash):
        bank = self.bank(donor, name, expected_hash=expected_hash)
        return GraphRouteRun(donor.source, donor.scene, donor.record, plan=donor.plan,
                    bank=bank, expected_bank_hash=bank.bank_hash, lr=donor.spec['lr'],
                    torch_seed=donor.spec['torch_seed'], expected_source_state=donor.spec['source_state'],
                    expected_plan_hash=donor.plan.plan_hash)

    def restore(self, donor, name, path, *, expected_hash, expected_sha256):
        reference = self.start(donor, name, expected_hash=expected_hash)
        return GraphRouteRun.load(donor.source, donor.scene, donor.record, path, plan=donor.plan,
                    bank=reference.bank, expected_bank_hash=reference.bank.bank_hash,
                    expected_sha256=expected_sha256, expected_contract_hash=reference.contract_hash)

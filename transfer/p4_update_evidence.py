"""Observe Adam's actual parameter participation and verify saved update identities."""
from collections import Counter
from time import perf_counter

import torch

from model.research_config import digest
from research_runtime import tensor_hash
from transfer import p4_coverage, p4_runtime_device as device


SCHEMA = 'p4_adam_named_step_evidence_v1'
BASE_FIELDS = ('model', 'optimizer', 'scheduler', 'rng', 'trace', 'history', 'best')
MOMENTS = ('step', 'exp_avg', 'exp_avg_sq')
PRESENT, NONZERO, CHANGED = 1, 2, 4
VALID_FLAGS = {0, PRESENT, PRESENT | NONZERO, PRESENT | CHANGED, PRESENT | NONZERO | CHANGED}


def fields(reg):
    return (*BASE_FIELDS, 'parameter_audit') if reg.get('parameter_audit') == SCHEMA else BASE_FIELDS


def category(name):
    if name.startswith(p4_coverage.PREFIX):
        return 'coverage'
    if name.startswith('backbone.learnable_graph.'):
        return 'learned_graph'
    if name.startswith(('heads.', 'hgnn_heads.', 'cnn_heads.')):
        return 'target_heads'
    return 'target_domain_norm' if '.norms.' in name else 'shared_backbone'


def layout(model, optimizer):
    named = dict(model.named_parameters())
    names = {id(value): name for name, value in named.items()}
    saved_groups = optimizer.state_dict()['param_groups']
    slots = {}
    for group, (live, saved) in enumerate(zip(optimizer.param_groups, saved_groups, strict=True)):
        for offset, (parameter, index) in enumerate(zip(live['params'], saved['params'], strict=True)):
            name = names.get(id(parameter))
            device.require(name is not None and name not in slots and parameter.requires_grad,
                           'Adam contains an unknown, duplicate or frozen parameter')
            slots[name] = (group, offset, index)
    device.require(set(slots) == {name for name, value in named.items() if value.requires_grad}
                   and len({slot[2] for slot in slots.values()}) == len(slots),
                   'Adam does not bind each trainable parameter exactly once')
    rows = []
    for name, value in named.items():
        group, offset, index = slots.get(name, (None, None, None))
        part = category(name)
        rule = ('frozen_by_configuration' if index is None else
                'when_coverage_has_usable_anchors' if part == 'coverage' else 'every_step')
        rows.append({'name': name, 'group': group, 'offset': offset, 'index': index,
                     'shape': list(value.shape), 'dtype': str(value.dtype), 'trainable': value.requires_grad,
                     'category': part, 'participation': rule})
    return rows


def bind_names(optimizer, rows):
    for group, live in enumerate(optimizer.param_groups):
        members = sorted((row for row in rows if row['group'] == group), key=lambda row: row['offset'])
        live['param_names'] = [row['name'] for row in members]


def moment_binding(row, entry):
    return digest({'parameter': row, 'moments': {key: tensor_hash(entry[key]) for key in MOMENTS}})


def sum_flags(flags):
    return [[int(bool(flag & bit)) for bit in (PRESENT, NONZERO, CHANGED)] for flag in flags]


def add_counts(left, right):
    return [[a + b for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


class Recorder:
    def __init__(self, model, optimizer, rows):
        self.rows = rows
        named = dict(model.named_parameters())
        self.params = [named[row['name']] for row in rows]
        self.optimizer = optimizer
        self.counts = [[0, 0, 0] for _ in rows]
        self.pending = None
        self.flags = None
        self.before_hook = optimizer.register_step_pre_hook(self.before_step)
        self.after_hook = optimizer.register_step_post_hook(self.after_step)

    @torch.no_grad()
    def before_step(self, optimizer, args, kwargs):
        device.require(self.pending is None and self.flags is None, 'Unexpected repeated Adam step')
        self.pending = []
        for parameter, row in zip(self.params, self.rows, strict=True):
            if row['trainable']:
                grad = parameter.grad
                flag = 0 if grad is None else (PRESENT + torch.any(grad != 0).to(torch.int8) * NONZERO
                                                + (~torch.isfinite(grad).all()).to(torch.int8) * 8)
                self.pending.append((parameter.detach().clone(), flag))

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        device.require(self.pending is not None and self.flags is None, 'Missing Adam pre-step evidence')
        readings = []
        prior = iter(self.pending)
        for parameter, row in zip(self.params, self.rows, strict=True):
            if not row['trainable']:
                readings.append(parameter.new_zeros((), dtype=torch.int8))
                continue
            before, flag = next(prior)
            changed = torch.any(parameter.detach() != before).to(torch.int8) * CHANGED
            readings.append(changed + flag)
        flags = torch.stack(readings).cpu().tolist()
        device.require(set(flags) <= VALID_FLAGS, 'Adam changed a missing-gradient parameter or saw nonfinite gradient')
        self.counts = add_counts(self.counts, sum_flags(flags))
        self.flags = bytes(flags).hex()
        self.pending = None

    def take_step(self):
        device.require(self.flags is not None and self.pending is None, 'Expected exactly one observed Adam step')
        flags, self.flags = self.flags, None
        return flags

    def snapshot(self):
        named = {row['name']: moment_binding(row, self.optimizer.state[p])
                 for p, row, counts in zip(self.params, self.rows, self.counts, strict=True) if counts[0]}
        return {'schema': SCHEMA, 'layout': self.rows, 'counts': self.counts, 'moment_bindings': named}


def check(model, optimizer, state, expected_layout_sha):
    """Recount individual parameter steps from trace flags and bind each Adam state slot."""
    evidence = state.get('parameter_audit', {})
    expected = layout(model, optimizer)
    device.require(evidence.get('schema') == SCHEMA and evidence.get('layout') == expected
                   and digest(expected) == expected_layout_sha, 'Registered Adam parameter identity differs')
    bind_names(optimizer, expected)
    actual = state['optimizer']
    expected_groups = optimizer.state_dict()['param_groups']
    device.require(len(actual['param_groups']) == len(expected_groups) and all(
        a['params'] == e['params'] and a.get('param_names') == e['param_names']
        for a, e in zip(actual['param_groups'], expected_groups)), 'Adam group, parameter index or name differs')
    trace, cursor = state['trace'], state['cursor']
    device.require([event['step'] for event in trace] == list(range(1, cursor + 1)),
                   'Parameter evidence is not aligned to the optimizer cursor')
    counts = [[0, 0, 0] for _ in expected]
    for event in trace:
        encoded = event.get('parameter_flags')
        device.require(isinstance(encoded, str), 'Missing per-step Adam flags')
        flags = bytes.fromhex(encoded)
        device.require(flags.hex() == encoded and len(flags) == len(expected) and set(flags) <= VALID_FLAGS,
                       'Invalid per-step Adam flags')
        coverage = event.get('coverage', {})
        graph_flags = []
        for row, flag in zip(expected, flags, strict=True):
            rule = row['participation']
            present = rule == 'every_step' or (rule == 'when_coverage_has_usable_anchors'
                                               and coverage['usable_anchors'] > 0)
            device.require(bool(flag & PRESENT) == present,
                           f"Unexpected gradient presence for {row['name']} at step {event['step']}")
            if row['category'] == 'learned_graph':
                graph_flags.append(flag)
            if rule == 'when_coverage_has_usable_anchors':
                detail = coverage['parameters'][row['name']]
                device.require((detail['gradient_l1'] is not None) == present
                               and bool(flag & NONZERO) == bool(detail['gradient_l1'])
                               and bool(flag & CHANGED) == bool(detail['changed_elements']),
                               'Adam flags differ from coverage gradient/update evidence')
        if graph_flags:
            device.require(any(flag & NONZERO for flag in graph_flags) == bool(event['graph_gradient_l1'])
                           and any(flag & CHANGED for flag in graph_flags) == bool(event['graph_changed_elements']),
                           'Adam flags differ from learned graph gradient/update evidence')
        counts = add_counts(counts, sum_flags(flags))
    device.require(evidence.get('counts') == counts, 'Cumulative Adam participation differs from per-step flags')
    active = {row['index']: (row, count[0]) for row, count in zip(expected, counts, strict=True) if count[0]}
    device.require(set(actual['state']) == set(active), 'Missing or unexpected named Adam moment slots')
    bindings = {}
    for index, (row, steps) in active.items():
        entry = actual['state'][index]
        device.require(set(entry) == set(MOMENTS) and isinstance(entry['step'], torch.Tensor)
                       and entry['step'].ndim == 0 and entry['step'].is_floating_point()
                       and float(entry['step']) == steps,
                       f"Adam step differs from observed participation for {row['name']}")
        device.require(all(isinstance(entry[key], torch.Tensor) and list(entry[key].shape) == row['shape']
                           and str(entry[key].dtype) == row['dtype'] and bool(torch.isfinite(entry[key]).all())
                           for key in MOMENTS[1:]), f"Invalid Adam moments for {row['name']}")
        device.require(bool((entry['exp_avg_sq'] >= 0).all()), f"Negative Adam second moment for {row['name']}")
        bindings[row['name']] = moment_binding(row, entry)
    device.require(evidence.get('moment_bindings') == bindings, 'Named Adam moment binding differs')
    totals = {}
    for row, count in zip(expected, counts, strict=True):
        group = totals.setdefault(row['category'], Counter())
        group.update({'parameters': 1, 'gradient_present': count[0], 'gradient_zero': count[0] - count[1],
                      'gradient_none': cursor - count[0], 'parameter_changed': count[2]})
    return {'schema': SCHEMA, 'parameter_layout_sha256': expected_layout_sha,
            'parameters': [{**row, 'adam_steps': count[0], 'nonzero_gradient_steps': count[1],
                            'changed_steps': count[2]} for row, count in zip(expected, counts, strict=True)],
            'by_category': {group: dict(values) for group, values in totals.items()}}


class Timing:
    """Keep synchronized measurements out of the scientific checkpoint fingerprints."""

    def __init__(self, target):
        self.target = target
        self.started = perf_counter()
        self.steps = []
        self.startup_seconds = None
        self.reload_seconds = 0.0

    def mark(self):
        if self.target.type == 'cuda':
            torch.cuda.synchronize(self.target)
        return perf_counter()

    def begin_update(self):
        if self.target.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.target)
        return self.mark()

    def update_memory(self):
        if self.target.type == 'cuda':
            return {'allocated_bytes': torch.cuda.max_memory_allocated(self.target),
                    'reserved_bytes': torch.cuda.max_memory_reserved(self.target)}
        return None

    def result(self):
        return {'startup_seconds': self.startup_seconds, 'steps': self.steps, 'reload_seconds': self.reload_seconds,
                'call_seconds': self.mark() - self.started}

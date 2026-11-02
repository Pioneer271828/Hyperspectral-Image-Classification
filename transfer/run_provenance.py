"""Effective G3 stage settings and strict, read-only artifact consistency checks."""
import copy
import json
from pathlib import Path
import numpy as np

from model.research_config import digest
from research_runtime import hash_file
from train_spatial import SpatialConfig


def _require(condition, message):
    if not condition:
        raise ValueError('Run provenance mismatch: ' + message)


def _seed(value, name):
    _require(type(value) is int and 0 <= value < 2**32, name + ' must be a uint32 seed')
    return value


def resolve_budget(config, steps, val_every=None):
    """Keep SpatialConfig loadable; its legacy inclusive epoch represents one stage."""
    _require(type(steps) is int and steps > 0, 'positive update budget required')
    value = config.to_dict()
    value['training'].update(epochs=0, steps_per_epoch=steps)
    if val_every is not None:
        _require(type(val_every) is int and val_every > 0, 'positive validation interval required')
        value['training'].update(val_interval=val_every, primary_checkpoint='val_loss')
    return SpatialConfig(value)


def run_fields(config, *, phase, method, head_seed, sampler_seed, val_every,
               selection, sampling, optimizer, scheduler=None):
    _require(phase in ('source', 'target', 'development'), 'unknown stage')
    model_seed = _seed(config.training.seed, 'model seed')
    _seed(sampler_seed, 'sampler seed')
    if head_seed is not None:
        _seed(head_seed, 'head seed')
    resolved = config.to_dict()
    frozen = method == 'source_pretrained_head_only'
    roles = ['heads'] if frozen else ['backbone', 'heads']
    groups = []
    for role, group in zip(roles, optimizer.param_groups):
        expected_lr = config.training.head_lr if role == 'heads' else config.training.lr
        _require(group['lr'] == expected_lr and group['weight_decay'] == config.training.weight_decay,
                 'optimizer differs from resolved configuration')
        groups.append({'role': role, 'lr': group['lr'], 'weight_decay': group['weight_decay'],
                       'betas': list(group['betas']), 'eps': group['eps'],
                       'parameters': sum(p.numel() for p in group['params'])})
    _require(len(groups) == len(roles) == len(optimizer.param_groups), 'optimizer group count')
    runtime = {
        'schema_version': 1, 'phase': phase, 'method': method,
        'resolved_config_hash': digest(resolved),
        'optimizer_steps': config.budget['optimizer_steps'],
        'seeds': {'model': model_seed, 'head': head_seed, 'sampler': sampler_seed,
                  'training_torch': model_seed},
        'validation_every_updates': val_every, 'selection': selection,
        'sampling': copy.deepcopy(sampling), 'frozen_backbone': frozen,
        'optimizer': {'name': 'Adam', 'initial_groups': groups},
        'scheduler': None if scheduler is None else {
            'name': type(scheduler).__name__, 'mode': scheduler.mode,
            'factor': scheduler.factor, 'patience': scheduler.patience},
        # These template fields do not construct the supplied scene/plan.
        'inactive_resolved_fields': ['tile_size', 'gap', 'split_seed'] + (
            ['training.val_interval', 'training.primary_checkpoint', 'training.eval_checkpoint']
            if phase == 'target' else []),
    }
    return {'resolved_config': resolved, 'runtime_config': runtime,
            'runtime_config_hash': digest(runtime)}


def verify_run_artifacts(folder):
    """Verify new records without editing files; legacy records require explicit audit."""
    folder = Path(folder)
    result = json.loads((folder / 'result.json').read_text())
    sidecar = json.loads((folder / 'model.pt.json').read_text())
    info = sidecar['training_config']
    keys = ('resolved_config', 'runtime_config', 'runtime_config_hash')
    _require(all(key in record for record in (result, info) for key in keys),
             'legacy/missing provenance; audit original trace and sidecar, do not rewrite history')
    _require(all(result[key] == info[key] for key in keys), 'result/sidecar settings differ')
    runtime = result['runtime_config']
    _require(result['runtime_config_hash'] == digest(runtime), 'runtime hash')
    _require(runtime['schema_version'] == 1, 'runtime schema')
    _require(runtime['resolved_config_hash'] == digest(result['resolved_config']), 'resolved config hash')
    config = SpatialConfig(result['resolved_config'])
    steps = runtime['optimizer_steps']; phase = runtime['phase']
    _require(type(steps) is int and steps > 0 and steps == config.budget['optimizer_steps'], 'update budget')
    _require(config.training.epochs == 0, 'one-stage epoch representation')
    seeds = runtime['seeds']
    for role, seed in seeds.items():
        if role != 'head' or seed is not None:
            _seed(seed, role)
    _require(seeds['model'] == config.training.seed == seeds['training_torch'], 'model/training seed')
    _require(phase in ('source', 'target', 'development') and info['phase'] == phase, 'phase')
    selection = runtime['selection']; interval = runtime['validation_every_updates']
    _require(result['status'] == 'completed' and result['selection'] == info['selection'] == selection, 'selection')
    _require(result.get('optimizer_steps', result.get('steps')) == steps, 'result update count')
    for key, expected in {'optimizer_budget': steps, 'optimizer_steps': steps, 'model_seed': seeds['model'],
                          'head_seed': seeds['head'], 'sampler_seed': seeds['sampler']}.items():
        if key in info:
            _require(info[key] == expected, 'sidecar ' + key)
    _require(runtime['frozen_backbone'] == (runtime['method'] == 'source_pretrained_head_only'), 'freeze policy')
    if phase != 'source':
        _require(result['method'] == info['method'] == runtime['method'], 'method')
    optimizer = runtime['optimizer']
    roles = ['heads'] if runtime['frozen_backbone'] else ['backbone', 'heads']
    _require(optimizer['name'] == 'Adam' and [g['role'] for g in optimizer['initial_groups']] == roles, 'optimizer roles')
    for group in optimizer['initial_groups']:
        lr = config.training.head_lr if group['role'] == 'heads' else config.training.lr
        _require(group['lr'] == lr and group['weight_decay'] == config.training.weight_decay, 'optimizer settings')
    expected_trace = list(range(1, steps + 1)) if phase == 'development' else [
        step for step in range(1, steps + 1) if step == 1 or step % 10 == 0 or step == steps]
    _require([row['step'] for row in result['trace']] == expected_trace, 'training trace steps')
    for row in result['trace']:
        if 'budget' in row:
            _require(row['budget'] == steps, 'trace budget')
    sampling = runtime['sampling']; tile_ids = sampling['tile_ids']
    _require(tile_ids and len(tile_ids) == len(set(tile_ids)), 'eligible sampling tiles')
    _require(sampling['mode'] in ('uniform', 'class_first'), 'sampling mode')
    if 'sampler_hash' in sampling:
        _require(sampling['sampler_hash'] == digest({k: v for k, v in sampling.items() if k != 'sampler_hash'}),
                 'sampler hash')
    rng = np.random.default_rng(seeds['sampler'])
    trace = {row['step']: row for row in result['trace']}
    for step in range(1, steps + 1):
        anchor = None
        if sampling['mode'] == 'class_first':
            column = int(rng.integers(len(sampling['classes'])))
            choices = np.flatnonzero(np.asarray(sampling['tile_class_counts'])[:, column])
            _require(len(choices) > 0, 'sampling class has no eligible tile')
            index = int(choices[int(rng.integers(len(choices)))])
            anchor = sampling['classes'][column]
        else:
            index = int(rng.integers(len(tile_ids)))
        if step in trace:
            _require(trace[step]['tile_id'] == tile_ids[index], 'sampler seed/trace replay')
            _require(trace[step].get('anchor_class') == anchor, 'anchor-class replay')
    if phase == 'target':
        _require(interval is None and selection == 'fixed_final_step' and runtime['scheduler'] is None,
                 'fixed-final target policy')
        selected = steps
    else:
        _require(type(interval) is int and interval > 0 and config.training.val_interval == interval,
                 'validation interval')
        _require(config.training.primary_checkpoint == 'val_loss', 'validation-NLL selection config')
        history = result['history'] if phase == 'source' else result['validation']
        expected_validation = list(range(interval, steps + 1, interval))
        if not expected_validation or expected_validation[-1] != steps:
            expected_validation.append(steps)
        _require([row['step'] for row in history] == expected_validation, 'validation trace steps')
        selected = min(history, key=lambda row: row['val']['nll'] if phase == 'source' else row['nll'])['step']
        _require(selection == ('source_val_nll' if phase == 'source' else 'minimum_validation_nll_then_earliest_step'),
                 'stage selection policy')
    _require(result['selected_step'] == info['selected_step'] == selected, 'selected checkpoint step')
    if phase == 'development':
        _require(result['sampler'] == info['sampler'] == runtime['sampling'], 'sampling metadata')
    for name, key in (('model.pt', 'checkpoint_sha256'), ('model.pt.json', 'sidecar_sha256')):
        _require(hash_file(folder / name) == result[key], name + ' hash')
    return {'status': 'verified', 'phase': phase, 'optimizer_steps': steps,
            'selected_step': selected, 'runtime_config_hash': result['runtime_config_hash']}

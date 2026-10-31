"""Paired B0 coordinate retraining; fixed full-budget protocol (Issue 33)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('absolute', 'relative', 'zero')


def state_digest(state):
    from model.research_config import digest
    from research_runtime import tensor_hash
    return digest({key: tensor_hash(value) for key, value in sorted(state.items())})


def initial_state(config):
    import torch
    from loadData.dataset_registry import DATASETS
    from train_spatial import SpatialConfig
    value = config.to_dict()
    value['interaction']['interaction_mode'] = 'none'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        model = SpatialConfig(value).research.build_model({value['dataset']: DATASETS[value['dataset']][2]})
    return {key: item.detach().cpu().clone() for key, item in model.state_dict().items()}


def paired_initializer(state, expected_digest, seed):
    import torch
    if state_digest(state) != expected_digest:
        raise ValueError('Shared initialization changed')
    def initialize(model):
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith('backbone.interaction.') for key in result.missing_keys):
            raise ValueError('Shared initialization does not match model')
        actual = {key: model.state_dict()[key] for key in state}
        if state_digest(actual) != expected_digest:
            raise AssertionError('Shared parameters/buffers not identical')
        # Constructor RNG consumption differs by variant; training RNG starts identically.
        torch.manual_seed(seed)
        return {'policy': 'same_noninteraction_state_and_training_seed_v1',
                'state_digest': expected_digest, 'seed': seed, 'shared_tensor_count': len(state),
                'variant_only_keys': result.missing_keys}
    return initialize


@contextmanager
def campaign_lock(root):
    lock = root / '.campaign.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as stream: stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def prepare(root, code):
    import torch
    from research_runtime import hash_file, write_json
    from train_spatial import SpatialConfig
    configs = {name: SpatialConfig(json.loads((ROOT / 'configs/experiments' / ('screen_coordinates_' + name + '.json')).read_text(encoding='utf-8-sig')))
               for name in VARIANTS}
    comparable = []
    for name, config in configs.items():
        value = config.to_dict(); value.pop('name'); value.pop('coordinate_mode')
        comparable.append(value)
        interaction = config.research.interaction
        if (interaction.interaction_mode != 'none' or config.values['coordinate_mode'] != name or interaction.alignment_mode != 'region'
                or interaction.interaction_direction != 'both' or interaction.graph_update_mode != 'none'
                or config.budget['optimizer_steps'] != 10000):
            raise ValueError('Unexpected screening modes or budget')
    if comparable[1:] != comparable[:-1]: raise ValueError('Unpaired configurations')
    state = initial_state(configs['absolute'])
    initialization = root / 'shared_initial_state.pt'
    manifest_path = root / 'campaign.json'
    expected = {'schema_version': 1, 'source_hash': code['source_hash'], 'commit': code['commit'],
                'configs': {name: config.config_hash for name, config in configs.items()},
                'shared_state_digest': state_digest(state), 'order': list(VARIANTS),
                'budget_per_variant': 10000, 'seed': configs['absolute'].training.seed}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if {key: previous[key] for key in expected} != expected:
            raise ValueError('Campaign identity changed; use a new output root')
        if not initialization.is_file() or hash_file(initialization) != previous['initialization_file_sha256']:
            raise ValueError('Shared initialization file missing or changed')
        if state_digest(torch.load(initialization, weights_only=True, map_location='cpu')) != expected['shared_state_digest']:
            raise ValueError('Initialization state mismatch')
    else:
        if initialization.exists(): raise FileExistsError('Untracked initialization file; inspect this campaign root')
        torch.save(state, initialization)
        write_json(manifest_path, {**expected, 'initialization_file_sha256': hash_file(initialization)})
    return configs, state, expected['shared_state_digest']


def run_one(variant, config, state, shared_digest, root, code, loader, resume=False):
    from research_runtime import write_json
    from train_spatial import execute
    started = time.perf_counter()
    train_times = []
    last_step_at = None
    def progress(event):
        nonlocal last_step_at
        stamp = time.perf_counter()
        if event['phase'] == 'train':
            if last_step_at is not None: train_times.append(stamp - last_step_at)
            last_step_at = stamp
        else:
            last_step_at = None  # Validation time must not count as a training step.
        if event['phase'] == 'train' and event['optimizer_steps'] % 10:
            return
        payload = {**event, 'variant': variant, 'elapsed_seconds': stamp - started,
                   'updated_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(root / 'progress.json', payload)
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with (root / 'progress.jsonl').open('a', encoding='utf-8') as stream: stream.write(line + '\n')
        print(line, flush=True)
    # execute redirects stdout while training; each variant has its own detailed training.log.
    result = execute(config, root / 'training', code, loader, resume=resume,
        model_initializer=paired_initializer(state, shared_digest, config.training.seed), progress_callback=progress)
    if result['status'] == 'skipped':
        attempt = root / 'training' / result['run_id'] / 'attempts' / result['verified_attempt']
        original = json.loads((attempt / 'manifest.json').read_text(encoding='utf-8-sig'))
        if original.get('initialization', {}).get('state_digest') != shared_digest:
            raise ValueError('Previous completed run used a different initialization')
    elapsed = time.perf_counter() - started
    summary = {'variant': variant, 'status': result['status'], 'run_id': result['run_id'],
               'attempt': result['attempt'], 'elapsed_seconds': elapsed, 'shared_state_digest': shared_digest,
               'optimizer_steps_budget': config.budget['optimizer_steps'],
               'median_train_step_seconds': statistics.median(train_times[4:]) if len(train_times) > 4 else None}
    write_json(root / (variant + '-result.json'), summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'benchmark', 'run'))
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists(): parser.error('Run using ssh rmvision in the training container')
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
    from train_spatial import SpatialConfig
    code = code_snapshot()
    if (not re.fullmatch('[0-9a-f]{40}', args.expected_commit) or code['commit'] != args.expected_commit or code['dirty']):
        parser.error('Require exact full SHA and clean checkout')
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): parser.error('CUDA is required; no silent fallback')
    root = external_directory(args.out_root); root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        configs, state, shared_digest = prepare(root, code)
        print(json.dumps({'action': args.action, 'commit': code['commit'], 'gpu': torch.cuda.get_device_name(0),
                          'budget_per_variant': 10000, 'shared_state_digest': shared_digest}), flush=True)
        if args.action == 'prepare': return
        record, labels = inspect_dataset(args.data_root, 'paviaU')
        _, key, _, _ = DATASETS['paviaU']
        cube = loadmat(args.data_root / record['image_path'], variable_names=[key])[key]
        wave, _ = readWavelengths('paviaU', cube.shape[2])
        loader = lambda: (cube, labels, record, wave)
        if args.action == 'benchmark':
            value = configs['absolute'].to_dict()
            value['name'] = 'screen_coordinates_timing_only'
            value['training'].update(epochs=0, steps_per_epoch=30)
            timing_root = root / 'benchmark'; timing_root.mkdir(exist_ok=True)
            summary = run_one('timing', SpatialConfig(value), state, shared_digest, timing_root, code, loader, args.resume)
            write_json(root / 'timing.json', summary)
        else:
            for name, config in configs.items():
                require_frozen_code(code)
                result = run_one(name, config, state, shared_digest, root, code, loader, args.resume)
                if result['status'] not in ('completed', 'skipped'): raise RuntimeError('Training not complete')
        require_frozen_code(code)


if __name__ == '__main__': main()


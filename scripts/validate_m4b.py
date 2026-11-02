"""Remote M4-B1 acceptance with optional released-development two-update check."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = 'ca78fff9c8f6d032de4dc8985603e533485035fa'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def file_hash(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().rstrip('\0').split('\0')
    return {p: file_hash(ROOT / p) for p in paths
            if p.endswith('.py') or (p.startswith('configs/') and p.endswith('.json'))}


def real_development_check(development_root, protocol_root, data_root, out):
    """Load image and released train/devval supports; never load raw target GT."""
    import torch
    from scipy.io import loadmat
    from loadData.data_read import readWavelengths
    from loadData.dataset_registry import DATASETS
    from loadData.observation_sampling import BandMetadata
    from model.checkpoint_io import load_model_checkpoint
    from transfer import development_training as development
    from transfer.run_provenance import verify_run_artifacts

    sources = [development_root / 'manifest.json', development_root / 'development.json',
               development_root / 'parent-target-plan.json', protocol_root / 'task.json']
    before = {str(p): file_hash(p) for p in sources}
    manifest, record, plan, task = [json.loads(p.read_text()) for p in sources]
    require(manifest['status'] == 'passed' and manifest['validation_hash'] == record['validation_hash'],
            'unaccepted development protocol')
    require(manifest['parent_task_hash'] == task['task_hash'] and plan['plan_hash'] == task['target_plan_hash'],
            'development/parent task mismatch')
    identity = task['target_identity']
    require(identity['name'] == 'xuzhou', 'this short real check requires the frozen Xuzhou development task')
    data_root = data_root.resolve(strict=True)
    image = (data_root / identity['image_path']).resolve(strict=True)
    require(image.is_relative_to(data_root) and file_hash(image) == identity['image_sha256'], 'image identity mismatch')
    cube = loadmat(image, variable_names=[DATASETS['xuzhou'][1]])[DATASETS['xuzhou'][1]]
    wave, _ = readWavelengths('xuzhou', cube.shape[-1])
    bands = BandMetadata(identity['image_sha256'], tuple(range(len(wave))), tuple(wave),
                         'nm', 'approximate', (True,) * len(wave))
    scene = development.prepare_scene(cube, bands, plan, record)
    base = json.loads((ROOT / 'configs/experiments/screen_m6_none.json').read_text())
    base['model']['region_residual_mode'] = 'x_residual'
    trace = {'train_calls': 0, 'eval_calls': 0, 'train_input_max_changes': [],
             'projection_gradients': [], 'q_gradients': []}
    handles = []
    retained = {}
    original_builder = development.build_target

    def gradient_summary(tensor):
        values = tensor.coalesce().values() if tensor.is_sparse else tensor
        return {'finite': bool(torch.isfinite(values).all()), 'max_abs': float(values.abs().max())}

    def builder(*args, **kwargs):
        model, config = original_builder(*args, **kwargs)
        layer = model.backbone.region_residual
        retained['model'] = model
        retained['initial_weight'] = layer.projection.weight.detach().clone()
        handles.append(layer.projection.weight.register_hook(
            lambda grad: trace['projection_gradients'].append(gradient_summary(grad))))
        def observed(module, arguments, output):
            key = 'train_calls' if module.training else 'eval_calls'
            trace[key] += 1
            if module.training:
                with torch.no_grad():
                    trace['train_input_max_changes'].append(float((output.cnn_input - arguments[0]).abs().max()))
                if arguments[1].requires_grad:
                    arguments[1].register_hook(lambda grad: trace['q_gradients'].append(gradient_summary(grad)))
        handles.append(layer.register_forward_hook(observed))
        return model, config

    start = time.monotonic()
    try:
        with patch.object(development, 'build_target', side_effect=builder):
            result = development.adapt(scene, record, base, out / 'real-two-update',
                steps=2, val_every=1, device='cuda', seed=64, sampling='class_first')
    finally:
        for handle in handles:
            handle.remove()
    verified = verify_run_artifacts(out / 'real-two-update')
    require(verified['optimizer_steps'] == 2 and result['test_evaluated'] is False, 'short-check budget/test policy')
    require(result['resolved_config']['model']['region_residual_mode'] == 'x_residual', 'saved M4 mode mismatch')
    require(trace['train_calls'] == 2 and trace['eval_calls'] > 0, 'M4 did not execute in training/validation')
    for name in ('projection_gradients', 'q_gradients'):
        require(len(trace[name]) == 2 and all(g['finite'] and g['max_abs'] > 0 for g in trace[name]),
                'missing finite nonzero ' + name)
    weight_delta = float((retained['model'].backbone.region_residual.projection.weight.detach()
                          - retained['initial_weight']).abs().max())
    require(weight_delta > 0 and trace['train_input_max_changes'][1] > 0, 'M4 did not update/change CNN input')
    restored = load_model_checkpoint(out / 'real-two-update/model.pt', 'cuda')
    require(restored.backbone.region_residual_mode == 'x_residual', 'restored M4 mode mismatch')
    require(all(file_hash(Path(p)) == h for p, h in before.items())
            and file_hash(image) == identity['image_sha256'], 'input artifacts changed')
    review = {'status': 'passed', 'scope': 'two-update engineering; not accuracy/convergence evidence',
              'seconds': time.monotonic() - start, 'model_seed': 64, 'optimizer_steps': 2,
              'development_hash': record['validation_hash'], 'training_labels': 90, 'validation_labels': 45,
              'raw_target_label_file_loaded': False, 'test_evaluated': False,
              'input_hashes': before, 'image_sha256': identity['image_sha256'],
              'trace': trace, 'projection_weight_max_change': weight_delta,
              'selected_step': result['selected_step'], 'runtime_config_hash': result['runtime_config_hash'],
              'decision': 'keep integration; preregister fair M4-B2 before any accuracy claim'}
    (out / 'real-training-review.json').write_text(json.dumps(review, indent=2))
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--development-root', type=Path)
    parser.add_argument('--protocol-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--legacy-run-root', type=Path)
    args = parser.parse_args()
    require(all((args.development_root, args.protocol_root, args.data_root))
            or not any((args.development_root, args.protocol_root, args.data_root)), 'supply all three real-data roots or none')
    out = args.out_dir.resolve()
    require(not out.is_relative_to(ROOT) and not ROOT.is_relative_to(out), 'output overlaps checkout')
    for path in (args.development_root, args.protocol_root, args.data_root, args.legacy_run_root):
        if path is not None:
            source = path.resolve(strict=True)
            require(not source.is_relative_to(out) and not out.is_relative_to(source), 'output overlaps input artifacts')
    out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'expected_commit': args.expected_commit, 'checks': [],
              'scope': 'M4-B1 integration engineering; no complete accuracy experiment',
              'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        require(sys.platform == 'linux' and __debug__, 'Linux container with assertions required')
        container = subprocess.run(['systemd-detect-virt', '--container'], capture_output=True, text=True)
        require(Path('/.dockerenv').exists() or container.returncode == 0, 'training container not confirmed')
        require(re.fullmatch('[0-9a-f]{40}', args.expected_commit) is not None, 'expected full SHA required')
        require(git('rev-parse', 'HEAD') == args.expected_commit, 'unexpected checkout SHA')
        require(not git('status', '--porcelain'), 'dirty checkout')
        require(git('rev-parse', '--is-shallow-repository') == 'false', 'full history required')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
        paths = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', '-z', BASE, 'HEAD']).decode().rstrip('\0')
        allowed = {'model/VCASAHGCN.py', 'model/research_config.py', 'model/checkpoint_io.py',
                   'model/region_residual_decomposition.py',
                   'train_vca_multidataset.py', 'transfer/g3_training.py', 'tests/test_region_residual_integration.py',
                   'scripts/validate_m4b.py', 'README.md', 'docs/architecture/module-map.json',
                   'docs/guides/region-residual-m4.md', 'docs/guides/g3-training.md'}
        require(all(p in allowed or p.startswith('research/') for p in paths.split('\0') if p),
                'changes outside M4-B1 scope')
        before = source_files()
        import torch
        require(torch.cuda.is_available(), 'CUDA acceptance required')
        torch.set_num_threads(1)
        record.update(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                      checkout=str(ROOT), base=BASE, source_files=before,
                      environment={'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                   'container': container.stdout.strip(), 'gpu': torch.cuda.get_device_name(0),
                                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   M4_DEVICE='cpu', G3_DEVICE='cpu', GRAPH_CACHE_DEVICE='cpu', DYNAMIC_HEAD_DEVICE='cpu')
        def test(pattern): return ['-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v']
        checks = [('focused-cpu', test('test_region_residual_integration.py'), {}),
                  ('full-cpu', test('test_*.py'), {}),
                  ('focused-cuda', test('test_region_residual_integration.py'), {'M4_DEVICE': 'cuda'}),
                  ('g3-transfer-cuda', test('test_transfer_training.py'), {'G3_DEVICE': 'cuda'}),
                  ('g3-development-cuda', test('test_development_training.py'), {'G3_DEVICE': 'cuda'}),
                  ('g3-provenance-cuda', test('test_run_provenance.py'), {'G3_DEVICE': 'cuda'}),
                  ('smoke-cpu', ['smoke_test.py', '--device', 'cpu'], {}),
                  ('smoke-cuda', ['smoke_test.py', '--device', 'cuda'], {})]
        for name, command, overrides in checks:
            start = time.monotonic()
            item = {'name': name, 'command': [sys.executable, *command],
                    'environment': {**{k: env[k] for k in ('M4_DEVICE', 'G3_DEVICE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, **overrides}}
            record['checks'].append(item)
            with (out / (name + '.log')).open('w') as log:
                result = subprocess.run(item['command'], cwd=ROOT, env={**env, **overrides},
                                        stdout=log, stderr=subprocess.STDOUT, timeout=300)
            item.update(returncode=result.returncode, seconds=time.monotonic() - start)
            require(result.returncode == 0, name + ' failed; see preserved log')
        if args.development_root is not None:
            record['real_check'] = real_development_check(args.development_root.resolve(),
                args.protocol_root.resolve(), args.data_root.resolve(), out)
        else:
            record['real_check'] = {'status': 'not_run', 'reason': 'explicit artifact roots not supplied'}
        if args.legacy_run_root is not None:
            from model.checkpoint_io import load_model_checkpoint
            from transfer.run_provenance import verify_run_artifacts
            legacy = args.legacy_run_root.resolve(strict=True)
            hashes = {p.name: file_hash(p) for p in legacy.iterdir() if p.is_file()}
            checked = verify_run_artifacts(legacy)
            old_model = load_model_checkpoint(legacy / 'model.pt', 'cuda')
            require(old_model.backbone.region_residual_mode == 'none', 'legacy run must default to no M4')
            require({p.name: file_hash(p) for p in legacy.iterdir() if p.is_file()} == hashes,
                    'legacy artifacts changed')
            record['legacy_check'] = {'status': 'passed', 'root': str(legacy),
                                      'mode': 'none', 'verified': checked, 'unchanged_hashes': hashes}
        else:
            record['legacy_check'] = {'status': 'not_run', 'reason': 'explicit legacy run root not supplied'}
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--check', BASE, 'HEAD'], check=True)
        require(git('rev-parse', 'HEAD') == args.expected_commit and not git('status', '--porcelain'),
                'checkout changed during acceptance')
        require(source_files() == before, 'source fingerprints changed')
        record.update(status='passed', dirty_after=False)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        (out / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != 'source_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

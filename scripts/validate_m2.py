"""Frozen CPU acceptance of M2; execute only in the remote training container."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = '82d56ebaf6da3a5c391f1e1b78cd5323655bdf19'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def legacy_equivalence():
    import torch
    from model.variable_channel_adapter import VariableChannelAdapter
    legacy = types.ModuleType('legacy_vca_m2')
    exec(compile(git('show', BASE + ':model/variable_channel_adapter.py'), 'legacy_vca_m2', 'exec'), legacy.__dict__)
    results = []
    for mode in ('wavelength', 'index', 'uniform'):
        for bands in (12, 20, 103):
            torch.manual_seed(2026)
            options = dict(latent_channels=8, embed_dim=8, attention_mode=mode)
            original = legacy.VariableChannelAdapter(**options)
            current = VariableChannelAdapter(**options)
            current.load_state_dict(original.state_dict(), strict=True)
            if set(current.state_dict()) != set(original.state_dict()):
                raise AssertionError('Checkpoint keys changed')
            x = torch.randn(4, 5, bands)
            wave = torch.linspace(400, 2500, bands)
            fwhm = torch.linspace(3, 20, bands)
            before = original(x, wave, fwhm)
            after = current(x, wave, fwhm)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            before.square().sum().backward()
            after.square().sum().backward()
            for (name, a), (_, b) in zip(original.named_parameters(), current.named_parameters()):
                if a.grad is None:
                    if b.grad is not None:
                        raise AssertionError('Changed unused parameter: ' + name)
                else:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            original.eval()
            current.eval()
            with torch.no_grad():
                torch.testing.assert_close(original(x, wave, fwhm), current(x, wave, fwhm), rtol=0, atol=0)
            results.append({'mode': mode, 'bands': bands, 'output_and_gradient_exact': True})
    return results


def real_band_checks(data_root, device):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, file_hash
    from loadData.data_read import readWavelengths
    from model.variable_channel_adapter import VariableChannelAdapter
    results = []
    for name in ('paviaU', 'xuzhou', 'whu-honghu'):
        stem, key, _, _ = DATASETS[name]
        path = (data_root / (stem + '.mat')).resolve(strict=True)
        if data_root not in path.parents:
            raise ValueError('Data path escapes root')
        cube = loadmat(path, variable_names=[key])[key]
        shape = list(cube.shape)
        # Scaling this tiny fixture is only a numerical check, not a training protocol.
        patch = np.asarray(cube[:12, :12, :], dtype=np.float32).copy()
        del cube
        patch = (patch - patch.mean((0, 1))) / np.maximum(patch.std((0, 1)), 1e-6)
        x = torch.from_numpy(patch).to(device)
        wavelength_values, _ = readWavelengths(name, shape[2])
        wave = torch.as_tensor(wavelength_values, dtype=torch.float32, device=device)
        record = {'dataset': name, 'image_sha256': file_hash(path), 'raw_shape': shape,
                  'patch_shape': list(x.shape), 'wavelength_quality': 'estimated_not_verified',
                  'fwhm_available': False, 'device': str(device), 'modes': []}
        for mode in ('wavelength', 'index', 'uniform'):
            torch.manual_seed(2026)
            adapter = VariableChannelAdapter(latent_channels=64, embed_dim=32, attention_mode=mode).to(device)
            optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
            adapter.eval()
            with torch.no_grad():
                adapter(x, wave)
            adapter.train()
            y, info = adapter(x, wave, return_attention=True, wavelength_unit='nm')
            y.square().mean().backward()
            gradients = {n: float(p.grad.abs().sum()) for n, p in adapter.named_parameters() if p.grad is not None}
            if not torch.isfinite(y).all() or not all(np.isfinite(v) for v in gradients.values()):
                raise AssertionError('Nonfinite output/gradient')
            if gradients.get('value_proj.0.weight', 0) <= 0:
                raise AssertionError('No value gradient')
            if mode != 'uniform' and any(gradients.get(n, 0) <= 0 for n in ('query_encoder.0.weight', 'key_encoder.0.weight', 'anchor_offsets')):
                raise AssertionError('No attention gradient')
            optimizer.step()
            adapter.eval()
            with torch.no_grad():
                first = adapter(x, wave)
                cached = adapter(x, wave)
                adapter.clear_attention_cache()
                fresh = adapter(x, wave)
            torch.testing.assert_close(first, fresh, rtol=0, atol=0)
            torch.testing.assert_close(cached, fresh, rtol=0, atol=0)
            record['modes'].append({'mode': mode, 'output_shape': list(y.shape),
                                    'valid_anchors': int(info['anchor_valid'].sum()),
                                    'gradient_l1': gradients, 'cached_vs_fresh_max_diff': float((cached - fresh).abs().max()),
                                    'optimizer_steps': 1})
        results.append(record)
    return results


def cuda_cache_lifecycle():
    import copy
    import torch
    from model.variable_channel_adapter import VariableChannelAdapter
    torch.manual_seed(7)
    adapter = VariableChannelAdapter(latent_channels=8, embed_dim=8, locality_window=None).eval()
    x = torch.randn(4, 5, 12)
    wave = torch.linspace(400, 2500, 12)
    with torch.no_grad():
        adapter(x, wave)
    adapter.cuda()
    if adapter._eval_attention_cache:
        raise AssertionError('Device transfer retained CPU cache')
    x, wave = x.cuda(), wave.cuda()
    with torch.no_grad():
        adapter(x, wave)
    adapter(x, wave).square().mean().backward()
    for parameter in (adapter.query_encoder[0].weight, adapter.key_encoder[0].weight, adapter.anchor_offsets):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() <= 0:
            raise AssertionError('Warm CUDA eval lost attention gradient')
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.05)
    with torch.no_grad():
        adapter(x, wave)
    optimizer.step()
    with torch.no_grad():
        updated = adapter(x, wave)
        adapter.clear_attention_cache()
        torch.testing.assert_close(updated, adapter(x, wave), rtol=0, atol=0)
    parent = torch.nn.Sequential(adapter)
    donor = copy.deepcopy(parent)
    with torch.no_grad():
        donor[0].anchor_offsets.add_(1.)
    parent.load_state_dict(donor.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(adapter(x, wave), donor[0](x, wave), rtol=0, atol=0)
        for dtype in (torch.float16, torch.bfloat16):
            with torch.autocast('cuda', dtype=dtype):
                actual = adapter(x, wave)
                adapter.clear_attention_cache()
                torch.testing.assert_close(actual, adapter(x, wave), rtol=0, atol=0)
        adapter.cpu()
        if adapter._eval_attention_cache:
            raise AssertionError('Device transfer retained CUDA cache')
        if not torch.isfinite(adapter(x.cpu(), wave.cpu())).all():
            raise AssertionError('Invalid output after transfer to CPU')
    return {'cpu_cuda_cpu_transfer': 'passed', 'warm_eval_gradient': 'passed',
            'eval_optimizer_step': 'passed', 'nested_checkpoint_load': 'passed',
            'autocast_float16_bfloat16': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists():
        parser.error('Use ssh rmvision in the training container')
    if len(args.expected_commit) != 40 or git('rev-parse', 'HEAD') != args.expected_commit:
        parser.error('Expected full SHA must match HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and enabled assertions')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('Output must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    manifest = {'module': 'M2', 'commit': args.expected_commit, 'branch': git('branch', '--show-current'),
                'baseline_commit': BASE, 'python': sys.version, 'torch': torch.__version__,
                'hostname': platform.node(), 'device': args.device, 'code_directory': str(ROOT),
                'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'checks': []}

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save()
    try:
        commands = [
            ('unit', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']),
            ('smoke', [sys.executable, 'smoke_test.py', '--device', 'cpu']),
        ]
        if args.device == 'cuda':
            commands.append(('smoke_cuda', [sys.executable, 'smoke_test.py', '--device', 'cuda']))
        for name, command in commands:
            started = time.monotonic()
            with (out / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            manifest['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode,
                                       'seconds': round(time.monotonic() - started, 3)})
            save()
            if result.returncode:
                raise RuntimeError(name + ' failed; retain log and retry in a new output directory')
        manifest['legacy_equivalence'] = legacy_equivalence()
        if args.device == 'cuda':
            manifest['cuda_cache_lifecycle'] = cuda_cache_lifecycle()
        save()
        manifest['real_band_checks'] = real_band_checks(args.data_root.resolve(strict=True), torch.device(args.device))
        if args.device == 'cuda':
            torch.cuda.synchronize()
            manifest['real_check_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['real_check_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        if git('rev-parse', 'HEAD') != args.expected_commit or git('status', '--porcelain', '--untracked-files=all'):
            raise RuntimeError('Checkout changed during validation')
        manifest['status'] = 'passed'
    except Exception:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()

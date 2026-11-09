"""Frozen M3 validation in the remote container, with GPU spatial-block checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = 'd9007c36ec54cf3c882e36e64cd2eeac55f40168'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def legacy_equivalence():
    import torch
    from model.learnable_superpixel_graph import LearnableSuperpixelGraph
    from model.region_pixel_alignment import pool_regions, broadcast_regions
    legacy = types.ModuleType('legacy_graph_m3')
    exec(compile(git('show', BASE + ':model/learnable_superpixel_graph.py'), 'legacy_graph_m3', 'exec'), legacy.__dict__)
    records = []
    for h, w, scale in ((7, 9, 9), (12, 12, 16), (3, 7, 4)):
        torch.manual_seed(2026)
        original = legacy.LearnableSuperpixelGraph(6, assignment_dim=8)
        current = LearnableSuperpixelGraph(6, assignment_dim=8)
        current.load_state_dict(original.state_dict(), strict=True)
        if set(current.state_dict()) != set(original.state_dict()):
            raise AssertionError('State dict keys changed')
        old_x = torch.randn(h, w, 6, requires_grad=True)
        new_x = old_x.detach().clone().requires_grad_()
        Q, A, old_info = original(old_x, scale)
        state = current.build_graph(new_x, scale)
        for old, new in ((Q, state.Q), (A, state.A0)):
            torch.testing.assert_close(old.indices(), new.indices(), rtol=0, atol=0)
            torch.testing.assert_close(old.values(), new.values(), rtol=0, atol=0)
        def loss(q, a, x, losses):
            regions = pool_regions(q, x.reshape(-1, 6))
            return broadcast_regions(q, torch.sparse.mm(a, regions)).square().mean() + sum(losses)
        old_losses = [v for k, v in old_info.items() if k.endswith('_loss')]
        for name, value in state.regularizers.items():
            torch.testing.assert_close(value, old_info[name], rtol=0, atol=0)
        loss(Q, A, old_x, old_losses).backward()
        loss(state.Q, state.A0, new_x, state.regularizers.values()).backward()
        torch.testing.assert_close(old_x.grad, new_x.grad, rtol=0, atol=0)
        for (name, old), (_, new) in zip(original.named_parameters(), current.named_parameters()):
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
        records.append({'shape': [h, w, 6], 'scale': scale, 'Q_A_losses_gradients_exact': True})
    return records


def real_checks(data_root, device):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, file_hash
    from loadData.data_read import readWavelengths
    from loadData.spatial_tiles import make_tile_plan, fit_tile_statistics, observe_tile
    from model.variable_channel_adapter import VariableChannelAdapter
    from model.learnable_superpixel_graph import LearnableSuperpixelGraph
    from model.region_pixel_alignment import pool_regions, broadcast_regions
    results = []
    for name in ('paviaU', 'xuzhou', 'whu-honghu'):
        stem, key, _, _ = DATASETS[name]
        path = (data_root / (stem + '.mat')).resolve(strict=True)
        if data_root not in path.parents:
            raise ValueError('Dataset path escapes root')
        cube = loadmat(path, variable_names=[key])[key]
        plan = make_tile_plan(tuple(cube.shape[:2]), tile_size=32, gap=2, seed=2026)
        stats = fit_tile_statistics(cube, plan)
        selected = {p: next(tile for tile in plan['tiles'] if tile['partition'] == p) for p in ('train', 'val', 'test')}
        observations = {p: observe_tile(cube, plan, stats, tile) for p, tile in selected.items()}
        wave_values, _ = readWavelengths(name, cube.shape[2])
        wave = torch.as_tensor(wave_values, dtype=torch.float32, device=device)
        record = {'dataset': name, 'raw_shape': list(cube.shape), 'image_sha256': file_hash(path),
                  'plan_hash': plan['plan_hash'], 'statistics_hash': stats['statistics_hash'],
                  'wavelength_quality': 'not_verified', 'fwhm_available': False, 'controls': []}
        del cube
        ids = {p: set(obs.global_ids.reshape(-1).tolist()) for p, obs in observations.items()}
        if any(ids[a] & ids[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise AssertionError('Spatial blocks overlap')
        torch.manual_seed(2026)
        adapter = VariableChannelAdapter(latent_channels=64, embed_dim=32).to(device).eval()
        with torch.no_grad():
            latent = {p: adapter(torch.tensor(obs.spectral, device=device), wave, wavelength_unit='nm')
                      for p, obs in observations.items()}
        for q_mode, a_mode in (('fixed_grid', 'fixed_local'), ('fixed_grid', 'learned'), ('learned', 'learned')):
            torch.manual_seed(2026)
            builder = LearnableSuperpixelGraph(64, assignment_dim=32).to(device)
            optimizer = torch.optim.Adam(builder.parameters(), lr=1e-3)
            x = latent['train'].detach().clone().requires_grad_()
            state = builder.build_graph(x, 16, assignment_mode=q_mode, adjacency_mode=a_mode)
            regions = pool_regions(state.Q, x.reshape(-1, 64))
            output = broadcast_regions(state.Q, torch.sparse.mm(state.A0, regions))
            # A label-free numerical probe, not a classifier or accuracy objective.
            loss = output.square().mean() + .001 * sum(state.regularizers.values())
            loss.backward()
            gradients = {n: float(p.grad.abs().sum()) for n, p in builder.named_parameters() if p.grad is not None}
            if not torch.isfinite(output).all() or not torch.isfinite(x.grad).all() or not all(np.isfinite(v) for v in gradients.values()):
                raise AssertionError('Nonfinite graph output/gradient')
            for scorer in ('pixel_query', 'seed_key', 'graph_query', 'graph_key'):
                expected = q_mode == 'learned' if scorer in ('pixel_query', 'seed_key') else a_mode == 'learned'
                value = gradients.get(scorer + '.weight')
                if expected and (value is None or value <= 0):
                    raise AssertionError('Missing scorer gradient: ' + scorer)
                if not expected and value is not None:
                    raise AssertionError('Fixed control received scorer gradient: ' + scorer)
            steps = int(bool(gradients))
            if steps:
                optimizer.step()
            builder.eval()
            with torch.no_grad():
                before = builder.build_graph(latent['train'], 16, assignment_mode=q_mode, adjacency_mode=a_mode)
                partition_records = []
                for partition in ('train', 'val', 'test'):
                    evaluated = builder.build_graph(latent[partition], 16, assignment_mode=q_mode, adjacency_mode=a_mode)
                    partition_records.append({'partition': partition, 'tile': selected[partition],
                                              'input_shape': list(latent[partition].shape), **evaluated.diagnostics})
                after = builder.build_graph(latent['train'], 16, assignment_mode=q_mode, adjacency_mode=a_mode)
                diffs = {}
                for field in ('Q', 'A0'):
                    a, b = getattr(before, field), getattr(after, field)
                    torch.testing.assert_close(a.indices(), b.indices(), rtol=0, atol=0)
                    torch.testing.assert_close(a.values(), b.values(), rtol=1e-5, atol=1e-6)
                    diffs[field] = float((a.values() - b.values()).abs().max())
            record['controls'].append({'assignment_mode': q_mode, 'adjacency_mode': a_mode,
                                       'train_optimizer_steps': steps, 'gradient_l1': gradients,
                                       'partition_checks': partition_records, 'train_revisit_max_diff': diffs})
        record['spatial_global_ids_disjoint'] = True
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists():
        parser.error('Run in the remote training container through ssh rmvision')
    if len(args.expected_commit) != 40 or git('rev-parse', 'HEAD') != args.expected_commit:
        parser.error('Full SHA must match HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and assertions enabled')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('Write outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from research_runtime import code_snapshot
    torch.set_num_threads(1)
    manifest = {'module': 'M3', 'commit': args.expected_commit, 'branch': git('branch', '--show-current'),
                'baseline_commit': BASE, 'source_hash': code_snapshot()['source_hash'],
                'python': sys.version, 'torch': torch.__version__, 'device': args.device,
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'checks': []}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            manifest['gpu'] = torch.cuda.get_device_name(0)
        commands = [('unit', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']),
                    ('smoke_cpu', ['smoke_test.py', '--device', 'cpu'])]
        if args.device == 'cuda':
            commands.append(('smoke_cuda', ['smoke_test.py', '--device', 'cuda']))
        for name, command in commands:
            command = [sys.executable, *command]
            with (out / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            manifest['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(name + ' failed; retain logs and retry in a new directory')
        manifest['legacy_equivalence'] = legacy_equivalence()
        save()
        manifest['real_checks'] = real_checks(args.data_root.resolve(strict=True), torch.device(args.device))
        if args.device == 'cuda':
            torch.cuda.synchronize()
            manifest['real_check_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['real_check_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        snapshot = code_snapshot()
        if snapshot['commit'] != args.expected_commit or snapshot['dirty'] or snapshot['source_hash'] != manifest['source_hash']:
            raise RuntimeError('Source changed during validation')
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

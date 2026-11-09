"""Two isolated GPU queues; freeze all nine cases before the first update."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
from joint_training import import_preparation, load_plan, schedule_for
from joint_adapter_study import load_study, case_settings
from scripts.train_joint_m4 import graph_binding
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, require_frozen_code, write_json, external_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit', 'prepared-dir', 'data-root', 'input-manifest', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--gpus', required=True, type=int, nargs=2)
    a = p.parse_args(); source = code_snapshot()
    if sys.platform != 'linux' or source['dirty'] or source['commit'] != a.expected_commit:
        raise RuntimeError('Clean expected remote source required')
    if len(set(a.gpus)) != 2 or min(a.gpus) < 0: raise ValueError('Two distinct GPU indices required')
    out = external_directory(a.out_dir); out.mkdir(exist_ok=False)
    for name in ('logs', 'reviews', 'cases'): (out/name).mkdir()
    load = lambda path: json.loads(Path(path).read_text())
    started = time.monotonic(); study = load_study(); plan = load_plan('configs/experiments/base97_training.json')
    base, items, prov = import_preparation(a.prepared_dir, a.data_root, plan, source)
    prov = {**prov, 'm4_graph_binding': graph_binding(items)}
    inputs = load(a.input_manifest)['input_files']
    assert len(inputs) == 13 and all(file_hash(p) == h for p, h in inputs.items())
    cases = []
    for seed in study['training_seeds']:
        for mode in study['modes']:
            name = f'{mode}-{seed}'
            cases.append({'name': name, 'mode': mode, 'seed': seed,
                          'out': str(out/'cases'/name),
                          'settings': case_settings(study, plan, base, mode, seed)})
    counts = {n: np.bincount(i['train_samples_gt'].cpu().numpy().astype(int).ravel(), minlength=i['class_num']+1)[1:].tolist()
              for n, i in items.items()}
    launch = {'code': source, 'study': study, 'prepared': a.prepared_dir, 'data_root': a.data_root,
              'provenance': prov, 'inputs': inputs, 'cases': cases, 'gpus': a.gpus,
              'supervisor_sha256': file_hash(__file__), 'test_scoring': False,
              'expected_updates': 13527, 'train_class_counts': counts}
    write_json(out/'launch.json', launch); launch_hash = file_hash(out/'launch.json'); del items
    jobs = []
    for index, case in enumerate(cases):
        command = [sys.executable, str(ROOT/'scripts/train_joint_adapter_full.py'),
                   '--expected-commit', a.expected_commit, '--launch-plan', str(out/'launch.json'),
                   '--launch-sha256', launch_hash, '--mode', case['mode'],
                   '--seed', str(case['seed'])]
        jobs.append({k: case[k] for k in ('name', 'mode', 'seed', 'out')})
        jobs[-1].update(command=command, gpu=a.gpus[index%2], status='queued')
    lock = threading.RLock(); stop = threading.Event()

    def publish(final=None):
        with lock:
            write_json(out/'status.json', {'status': final or ('failed' if stop.is_set() else 'running'),
                       'elapsed_seconds': time.monotonic()-started, 'jobs': jobs,
                       'test_scoring': False, 'launch_sha256': launch_hash})

    def update(job, **values):
        with lock: job.update(**values); publish()

    def invoke(command, env, log):
        with log.open('x') as stream:
            return subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode

    def review(job, env):
        folder = Path(job['out']); result = load(folder/'result.json')
        trace, history = load(folder/'trace.json'), load(folder/'history.json')
        case = next(c for c in cases if c['name'] == job['name'])
        assert result['settings'] == case['settings'] and result['start_code'] == source
        assert result['status'] == 'completed' and result['optimizer_steps'] == len(trace) == 1503 and len(history) == 51
        assert [t['dataset'] for t in trace] == schedule_for(case['settings'])
        assert [t['step'] for t in trace] == list(range(1,1504))
        assert result['selected_step'] == max(history, key=lambda e: e['macro_oa'])['step']
        assert all(np.isfinite(v) for t in trace for v in t['loss'].values())
        assert all(np.isfinite(v) for t in trace for v in t['gradient_norms'].values())
        assert sum(t['graph_changed_elements'] for t in trace) > 0
        for phases in result['selected_validation']['metrics'].values():
            for metrics in phases.values(): assert set(metrics) == {'fused','graph','cnn'}
        before = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        log = out/'logs'/(job['name']+'-resume.log')
        assert invoke(job['command']+['--resume'], env, log) == 0
        assert json.loads(log.read_text().strip().splitlines()[-1])['updates_this_call'] == 0
        assert before == {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        receipt = {'code': source, 'name': job['name'], 'updates': 1503, 'selected_step': result['selected_step'],
                   'loss_first30': float(np.mean([t['loss']['total'] for t in trace[:30]])),
                   'loss_last30': float(np.mean([t['loss']['total'] for t in trace[-30:]])),
                   'last50_epoch_validation_gain': history[-1]['macro_oa']-history[-6]['macro_oa'],
                   'selected_validation': result['selected_validation'], 'artifacts': before, 'test_scoring': False,
                   'positive_gradient_steps': {key: sum(t['gradient_norms'][key] > 0 for t in trace)
                                              for key in sorted(set.intersection(*(set(t['gradient_norms']) for t in trace)))},
                   'finalization_policy': result.get('finalization_policy', 'original_strict')}
        write_json(out/'reviews'/(job['name']+'.json'), receipt)

    def worker(gpu):
        for job in [j for j in jobs if j['gpu'] == gpu]:
            if stop.is_set(): return
            try:
                require_frozen_code(source)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OPENBLAS_NUM_THREADS': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                while True:
                    free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip().split(','))
                    if free >= 11264 and util <= 60: break
                    update(job, status='waiting_resource'); time.sleep(15)
                    if stop.is_set(): return
                log = out/'logs'/(job['name']+'.log'); update(job, status='running', log=str(log)); t = time.monotonic()
                code = invoke(job['command'], env, log)
                update(job, exit_code=code, training_seconds=time.monotonic()-t)
                if code:
                    folder = Path(job['out']); progress = load(folder/'progress.json')
                    assert progress['cursor'] == 1503 and all((folder/n).is_file() for n in ('primary.pt','last.pt','selected-probabilities.pt'))
                    update(job, status='zero_update_finalization', original_failure_preserved=True)
                    if invoke(job['command']+['--finalize-only'], env, out/'logs'/(job['name']+'-finalize.log')):
                        raise RuntimeError('Zero-update finalization failed')
                update(job, status='reviewing'); review(job, env); update(job, status='completed')
                print('CASE_COMPLETE', job['name'], job['training_seconds'], flush=True)
            except Exception as error:
                stop.set(); update(job, status='failed', error=repr(error)); traceback.print_exc(); return
    publish()
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(worker, a.gpus))
    require_frozen_code(source); assert all(file_hash(p) == h for p, h in inputs.items())
    state = 'training_completed' if all(j['status'] == 'completed' for j in jobs) else 'failed'
    publish(state); print('CAMPAIGN_FINISHED', state, flush=True)


if __name__ == '__main__': main()

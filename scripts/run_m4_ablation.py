"""Prepare/execute the frozen four-arm M4 screen; one seed group per explicit GPU."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT, code_snapshot, external_directory, write_json
from transfer.container_runtime import training_container_kind
from transfer.m4_ablation import (freeze_plan, input_identity, load_scene, run_case,
                                  summarize, verify_plan, require)


def schedule(seeds, gpus):
    require(gpus and len(gpus) == len(set(gpus)) and len(gpus) <= len(seeds), 'distinct GPUs required')
    require(len(seeds) == len(set(seeds)), 'duplicate seed groups')
    return [{'seed': seed, 'gpu': gpus[index % len(gpus)]} for index, seed in enumerate(seeds)]


def execute(plan, gpus, *, resume=False):
    verify_plan(plan); root = Path(plan['output'])
    require(all(re.fullmatch(r'GPU-[0-9a-fA-F-]+', gpu) for gpu in gpus), 'use explicit GPU UUIDs')
    jobs = schedule(plan['seeds'], gpus)
    lock = root / '.controller.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    execution = root / 'executions' / str(time.time_ns())
    active = {}; state = {'status': 'running', 'plan_hash': plan['plan_hash'], 'jobs': jobs,
                           'pid': os.getpid(), 'started_unix': time.time(), 'resume': resume}
    execution.mkdir(parents=True, exist_ok=False)
    def save(): write_json(execution / 'campaign.json', state)
    try:
        snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                            '--format=csv,noheader'], text=True)
        (execution / 'gpu-before.csv').write_text(snapshot)
        available = {row.split(',')[1].strip(): int(row.split(',')[2].strip().split()[0]) for row in snapshot.splitlines()}
        require(all(gpu in available and available[gpu] >= 8192 for gpu in gpus), 'selected GPU lacks 8GiB free')
        pending = list(jobs); failed = False; save()
        while pending or active:
            busy = set(active)
            if not failed:
                for job in list(pending):
                    if job['gpu'] in busy: continue
                    log = (execution / ('seed-' + str(job['seed']) + '.log')).open('xb')
                    command = [sys.executable, '-u', str(ROOT / 'scripts/run_m4_ablation.py'), 'worker',
                               '--plan', str(root / 'plan.json'), '--expected-commit', plan['code']['commit'],
                               '--seed', str(job['seed'])] + (['--resume'] if resume else [])
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               PYTHONDONTWRITEBYTECODE='1')
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    job.update(status='running', command=command, pid=process.pid)
                    active[job['gpu']] = (process, log, job); busy.add(job['gpu']); pending.remove(job); save()
            for gpu, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close(); job.update(status='passed' if code == 0 else 'failed', returncode=code)
                    failed |= code != 0; del active[gpu]; save()
            if failed and not active:
                for job in pending: job['status'] = 'not_started_after_failure'
                break
            if active: time.sleep(2)
        require(not failed, 'worker failed; retained artifacts; pending groups not launched')
        report = summarize(plan); write_json(root / 'paired-results.json', report)
        state.update(status='completed', target_updates=report['target_updates'], decision=report['decision'])
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        # Ordinary worker failures are polled to completion; never signal other projects.
        for process, log, job in active.values():
            job['returncode'] = process.wait(); log.close()
        state['finished_unix'] = time.time(); save()
        if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'worker', 'summarize'))
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--development-root', type=Path)
    parser.add_argument('--protocol-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--gpus', nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); code = code_snapshot()
    require(sys.platform == 'linux' and training_container_kind() and __debug__
            and not code['dirty'] and code['commit'] == args.expected_commit, 'clean exact SHA in training container required')
    if args.command == 'prepare':
        require(all((args.out_dir, args.development_root, args.protocol_root, args.data_root)), 'input/output roots required')
        out = external_directory(args.out_dir)
        for source in (args.development_root, args.protocol_root, args.data_root):
            source = source.resolve(strict=True)
            require(not out.is_relative_to(source) and not source.is_relative_to(out), 'output overlaps frozen inputs')
        identity = input_identity(args.development_root, args.protocol_root, args.data_root)
        plan = freeze_plan(code, identity, out, smoke=args.smoke)
        out.mkdir(parents=True, exist_ok=False); write_json(out / 'plan.json', plan)
        print(json.dumps({'status': 'frozen', 'scope': plan['scope'], 'plan_hash': plan['plan_hash'],
                          'cases': len(plan['seeds']) * len(plan['modes']), 'updates_per_case': plan['updates']}, indent=2))
        return
    require(args.plan is not None, '--plan required')
    plan = json.loads(args.plan.read_text()); verify_plan(plan)
    require(plan['code']['commit'] == args.expected_commit, 'plan belongs to a different commit')
    require(args.plan.resolve() == Path(plan['output']) / 'plan.json', 'plan path differs from frozen output')
    if args.command == 'worker':
        import torch
        require(args.seed in plan['seeds'] and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'one visible CUDA GPU and registered seed required')
        torch.set_num_threads(1); scene, record = load_scene(plan)
        operations = []
        for mode in plan['modes']:
            result = run_case(plan, scene, record, args.seed, mode, resume=args.resume)
            operations.append({'mode': mode, 'operation': result['operation'], 'experiment_status': result['status'],
                               'verified_artifact_dir': result['artifact_dir']})
            write_json(Path(plan['output']) / ('seed-' + str(args.seed) + '-operations.json'), operations)
        return
    if args.command == 'run':
        require(args.gpus is not None, 'explicit GPU UUIDs required')
        result = execute(plan, args.gpus, resume=args.resume)
    else:
        result = summarize(plan); write_json(Path(plan['output']) / 'paired-results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

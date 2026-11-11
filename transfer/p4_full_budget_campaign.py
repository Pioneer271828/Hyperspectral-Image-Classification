"""Prepare, execute and accept one new P4 full-budget campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Lock
from time import perf_counter

from research_runtime import ROOT, code_snapshot, hash_file, require_frozen_code, write_json
from transfer import p4_coverage_batch as batch, p4_full_budget_audit as audit
from transfer import p4_full_budget_registration as registration, p4_runtime_device as device
from transfer.g2_contract import read_pinned
from transfer.p4_spatial import run_case


def read(path):
    return read_pinned(path, hash_file(path))


def owner(out, scope_path, sha):
    return {'policy': registration.POLICY, 'scope': str(Path(scope_path).resolve()), 'scope_sha256': sha}


def bound_folder(out, case):
    path = out / 'cases' / case['id']
    device.require(path.resolve() == path, 'Case output cannot traverse symlinks')
    return path


def accepted(folder, scope, case, sha):
    path = folder / 'audit.json'
    row = read(path)
    steps = scope['budget']['steps']
    case_files = {str(folder / name) for name in ('state.pt', 'result.json', 'completed-resume.json')}
    case_files |= {str(p) for p in (folder / 'calls').glob('*.json')}
    device.require(row['status'] == 'passed' and row['case'] == case and row['scope_sha256'] == sha
        and row['optimizer_updates'] == steps and row['completed_resume_updates'] == 0
        and row['audit_optimizer_updates'] == 0 and row['query_accessed'] is row['raw_gt_loaded'] is False
        and row['numeric']['physical_uuid'] == case['physical_uuid'] and row['numeric']['protocol'] == case['protocol']
        and row['validation_steps'] == scope['budget']['validation_steps']
        and row['selected_step'] in row['validation_steps'] and set(row['artifact_pins']) == case_files,
        'Case audit does not match the frozen scope')
    batch.check_pins(row['artifact_pins'])
    cursor, calls = audit.ledger(folder, case, sha)
    result = read(folder / 'result.json')
    proof = read(folder / 'completed-resume.json')
    device.require(cursor == steps and all(call['numeric'] == row['numeric'] for call in calls)
        and result['best_step'] == row['selected_step'] and result['identity'] == proof['identity']
        and result['runtime'] == {'scope_sha256': sha, 'case': case, 'numeric': row['numeric']}
        and proof['cursor'] == steps and proof['new_optimizer_updates'] == 0,
        'Completed audit differs from worker checkpoint ledger')
    return row


def worker(scope_path, sha, out, case_id, operation, until, code):
    scope, reg = registration.load(scope_path, sha, code, check_files=False)
    out = batch.sibling(out, scope_path)
    device.require(read(out / 'owner.json') == owner(out, scope_path, sha), 'Unowned campaign directory')
    matches = [case for case in scope['cases'] if case['id'] == case_id]
    device.require(len(matches) == 1, 'Unregistered case')
    case, = matches
    folder = bound_folder(out, case)
    # Do not claim the case or initialize CUDA while the device is ineligible.
    with device.arithmetic(case['protocol'], case['physical_uuid']) as numeric:
        batch.check_pins(scope['protected_files'])
        folder.mkdir(exist_ok=True)
        with device.lease(folder / '.operation.lock'):
            if (folder / 'audit.json').exists():
                accepted(folder, scope, case, sha)
                device.require(operation != 'advance', 'Accepted case cannot be rerun')
                return {'status': 'accepted', 'case': case_id, 'updates': 0}
            cursor, calls = audit.ledger(folder, case, sha)
            steps = scope['budget']['steps']
            device.require(all(call['numeric'] == numeric for call in calls), 'Cannot change the numerical resume device')
            runtime = {'scope_sha256': sha, 'case': case, 'numeric': numeric}
            if operation == 'audit':
                device.require(until is None and cursor == steps, 'Audit requires the completed full budget')
                row = audit.case_audit(folder, scope, reg, case, sha, numeric)
                write_json(folder / 'audit.json', row)
                return {'status': 'accepted', 'case': case_id, 'updates': 0}
            if operation == 'completed':
                device.require(until is None and cursor == steps, 'Completed recovery requires the full budget')
                protected = {folder / name: hash_file(folder / name) for name in ('state.pt', 'result.json')}
                result = run_case(reg, case['index'], folder, steps, code, coverage_arm=case['arm'], runtime=runtime)
                batch.check_pins(protected)
                device.require(result['cursor'] == steps and result['updates_this_call'] == 0,
                               'Completed recovery changed the update budget')
                proof = {'scope_sha256': sha, 'identity': result['identity'], 'case': case_id,
                    'new_optimizer_updates': 0, 'cursor': steps, 'numeric': numeric,
                    'state_sha256': protected[folder / 'state.pt'], 'result_sha256': protected[folder / 'result.json']}
                path = folder / 'completed-resume.json'
                if path.exists():
                    device.require(read(path) == proof, 'Completed recovery evidence changed')
                else:
                    write_json(path, proof)
                return {'status': 'verified_completed', 'case': case_id, 'updates': 0}
            device.require(operation == 'advance' and type(until) is int and cursor < until <= steps,
                           'Advance must consume a positive registered update interval')
            started, tick = batch.now(), perf_counter()
            result = run_case(reg, case['index'], folder, until, code, coverage_arm=case['arm'], runtime=runtime)
            elapsed = perf_counter() - tick
            device.require(result['cursor'] == until and result['updates_this_call'] == until - cursor,
                           'Worker consumed an unexpected number of updates')
            folder.joinpath('calls').mkdir(exist_ok=True)
            path = folder / 'calls' / f'{cursor:06d}-{until:06d}.json'
            device.require(not path.exists(), 'Update receipt already exists')
            write_json(path, {'case': case_id, 'scope_sha256': sha, 'identity': result['identity'],
                'numeric': numeric, 'start': cursor, 'until': until, 'new_optimizer_updates': until - cursor,
                'state_sha256': hash_file(folder / 'state.pt'), 'started_utc': started,
                'finished_utc': batch.now(), 'wall_seconds': elapsed})
            return {'status': 'advanced', 'case': case_id, 'updates': until - cursor}


def invoke(command, log_dir, case, operation, locks):
    gpu = case['physical_uuid']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu or '', 'CUBLAS_WORKSPACE_CONFIG': device.WORKSPACE,
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with tempfile.NamedTemporaryFile(dir=log_dir, prefix=f"{case['id']}-{operation}-",
                                    suffix='.log', delete=False) as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 pass_fds=tuple(lock.fileno() for lock in locks))
    if process.returncode:
        raise RuntimeError(f'{case["id"]} {operation} exited {process.returncode}, evidence {log.name}')
    return json.loads(Path(log.name).read_text(encoding='utf-8').splitlines()[-1])


def invoke_operation(scope_path, sha, out, code, case, operation, until, locks, launcher):
    # Check again in the isolated subprocess before touching CUDA.
    if case['physical_uuid'] is not None:
        device.gate(case['physical_uuid'])
    command = [sys.executable, '-m', 'transfer.p4_full_budget_campaign', 'worker',
               '--expected-commit', code['commit'], '--scope', str(scope_path), '--scope-sha256', sha,
               '--out', str(out), '--case', case['id'], '--operation', operation]
    if until is not None:
        command.extend(['--until', str(until)])
    result = launcher(command, out / 'logs', case, operation, locks)
    if result['status'] == 'waiting_for_resources':
        raise device.ResourceUnavailable(result['reason'])
    return result


def run(scope_path, sha, out, code, *, resume=False, segment=None, launcher=invoke):
    scope, _ = registration.load(scope_path, sha, code)
    out = batch.sibling(out, scope_path)
    steps = scope['budget']['steps']
    segment = steps if segment is None else segment
    device.require(type(segment) is int and 1 <= segment <= steps, 'Invalid controlled segment size')
    if not resume:
        out.mkdir(exist_ok=False)
    else:
        device.require(out.is_dir(), 'Resume requires an existing owned output')
    with device.lease(out / '.coordinator.lock') as coordinator:
        ownership = owner(out, scope_path, sha)
        status_path = out / 'status.json'
        if resume:
            device.require(read(out / 'owner.json') == ownership and read(status_path)['status'] != 'failed',
                           'Campaign ownership differs or a recorded failure requires diagnosis')
        else:
            write_json(out / 'owner.json', ownership)
            for name in ('cases', 'logs'):
                (out / name).mkdir()
        cases = scope['cases']
        completed = {}
        for case in cases:
            folder = bound_folder(out, case)
            if (folder / 'audit.json').exists():
                completed[case['id']] = accepted(folder, scope, case, sha)
        if len(completed) == len(cases):
            result = read(status_path)
            if result['status'] != 'completed':
                result = {'status': 'completed', 'scope_sha256': sha, 'cases': len(cases),
                          'optimizer_updates': scope['budget']['planned_optimizer_updates'],
                          'optimizer_updates_this_invocation': 0, 'slots': {}, 'errors': {}}
                write_json(status_path, result)
            return {**result, 'optimizer_updates_this_invocation': 0}
        stop, mutex = Event(), Lock()
        errors, states = {}, {}
        new_updates = 0
        slots = device.slots(scope['protocol'], scope['gpus'])
        if scope['gpus']:
            leases = out.parent / '.p4_gpu_leases'
            leases.mkdir(exist_ok=True)

        def publish(status, slot, case=None, reason=None):
            with mutex:
                states[str(slot)] = {'status': status, 'case': case, 'reason': reason}
                write_json(status_path, {'status': 'draining_after_failure' if errors else 'running',
                    'scope_sha256': sha, 'completed_cases': len(completed), 'planned_cases': len(cases),
                    'optimizer_updates_this_invocation': new_updates, 'slots': states, 'errors': errors})

        def queue(slot):
            nonlocal new_updates
            active = None
            try:
                lock_path = leases / (slot + '.lock') if slot is not None else out / '.cpu.lock'
                with device.lease(lock_path) as lock:
                    for case in cases:
                        active = case['id']
                        if stop.is_set():
                            break
                        if case['physical_uuid'] != slot or active in completed:
                            continue
                        folder = bound_folder(out, case)
                        cursor, _ = audit.ledger(folder, case, sha)
                        while cursor < steps and not stop.is_set():
                            until = min(cursor + segment, steps)
                            publish('advancing', slot, active)
                            got = invoke_operation(scope_path, sha, out, code, case, 'advance', until,
                                                   (coordinator, lock), launcher)
                            device.require(got == {'status': 'advanced', 'case': active, 'updates': until - cursor},
                                           'Coordinator update accounting differs')
                            with mutex:
                                new_updates += got['updates']
                            cursor = until
                        if stop.is_set():
                            break
                        if not (folder / 'completed-resume.json').is_file():
                            publish('completed_recovery', slot, active)
                            proof = invoke_operation(scope_path, sha, out, code, case, 'completed', None,
                                                     (coordinator, lock), launcher)
                            device.require(proof == {'status': 'verified_completed', 'case': active, 'updates': 0},
                                           'Unexpected completed recovery budget')
                        if stop.is_set():
                            break
                        publish('auditing', slot, active)
                        proof = invoke_operation(scope_path, sha, out, code, case, 'audit', None,
                                                 (coordinator, lock), launcher)
                        device.require(proof == {'status': 'accepted', 'case': active, 'updates': 0},
                                       'Unexpected independent audit result')
                        row = accepted(folder, scope, case, sha)
                        with mutex:
                            completed[active] = row
                publish('finished', slot)
            except device.ResourceUnavailable as error:
                publish('waiting_for_resources', slot, active, str(error))
            except Exception as error:
                stop.set()
                with mutex:
                    errors[str(slot)] = {'case': active, 'reason': repr(error)}
                publish('failed', slot, active, repr(error))

        for slot in slots:
            publish('queued', slot)
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            tasks = [pool.submit(queue, slot) for slot in slots]
            try:
                for task in tasks:
                    task.result()
            except BaseException:
                stop.set()
                raise
        status = 'failed' if errors else 'completed' if len(completed) == len(cases) else 'waiting_for_resources'
        consumed = sum(audit.ledger(bound_folder(out, case), case, sha)[0] for case in cases) if not errors else None
        result = {'status': status, 'scope_sha256': sha, 'cases': len(completed), 'optimizer_updates': consumed,
                  'optimizer_updates_this_invocation': new_updates, 'slots': states, 'errors': errors}
        write_json(status_path, result)
        if errors:
            raise RuntimeError(f'Campaign stopped dispatch after failure, see {status_path}')
        require_frozen_code(code)
        return result


def accept(scope_path, sha, out, code):
    scope, _ = registration.load(scope_path, sha, code)
    out = batch.sibling(out, scope_path)
    device.require(read(out / 'owner.json') == owner(out, scope_path, sha), 'Unowned campaign directory')
    with device.lease(out / '.coordinator.lock'):
        status = read(out / 'status.json')
        cases = scope['cases']
        device.require(status['status'] == 'completed' and status['scope_sha256'] == sha
            and status['cases'] == len(cases) and status['optimizer_updates'] == scope['budget']['planned_optimizer_updates'],
            'Incomplete full matrix cannot be accepted')
        case_pins, all_artifacts, pairs = {}, {}, {}
        for case in cases:
            folder = bound_folder(out, case)
            row = accepted(folder, scope, case, sha)
            path = folder / 'audit.json'
            case_pins[str(path)] = hash_file(path)
            all_artifacts.update(row['artifact_pins'])
            pairs.setdefault(case['pair_sha256'], []).append(case['arm'])
        device.require(len(pairs) == 24 and all(arms == list(registration.coverage.ARMS) for arms in pairs.values()),
                       'Missing matched coverage arm at acceptance')
        pins = {**scope['protected_files'], str(Path(scope_path).resolve()): sha, **case_pins, **all_artifacts}
        batch.check_pins(pins)
        require_frozen_code(code)
        path = out / 'acceptance.json'
        result = {'status': 'passed', 'policy': registration.POLICY, 'scope_sha256': sha, 'producer_commit': code['commit'],
            'cases': len(cases), 'pairs': len(pairs), 'optimizer_updates': status['optimizer_updates'],
            'nonselected_event_probability_replays': 0, 'audit_optimizer_updates': 0, 'raw_gt_loaded': False,
            'query_accessed': False, 'case_audits': case_pins, 'protected_files': pins}
        if path.exists():
            device.require(read(path) == result, 'Published full-budget acceptance differs')
        else:
            write_json(path, result)
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--parent', type=Path, required=True)
    prep.add_argument('--prior-receipt', type=Path, required=True)
    prep.add_argument('--protocol', choices=(device.CPU, device.CUDA), required=True)
    prep.add_argument('--gpus', nargs='*', default=[])
    execute = modes.add_parser('run')
    execute.add_argument('--resume', action='store_true')
    execute.add_argument('--segment', type=int)
    work = modes.add_parser('worker')
    work.add_argument('--case', required=True)
    work.add_argument('--operation', choices=('advance', 'completed', 'audit'), required=True)
    work.add_argument('--until', type=int)
    for mode in (execute, work, modes.add_parser('accept')):
        mode.add_argument('--scope', type=Path, required=True)
        mode.add_argument('--scope-sha256', required=True)
    for mode in modes.choices.values():
        mode.add_argument('--out', type=Path, required=True)
        mode.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    code = code_snapshot()
    device.require(args.expected_commit == code['commit'], 'Unexpected full-budget source commit')
    if args.mode == 'prepare':
        path = registration.prepare(args.parent, batch.PARENT_SHA, args.prior_receipt, batch.PRIOR_SHA,
                                    args.out, code, protocol=args.protocol, gpus=args.gpus)
        result = {'path': str(path), 'sha256': hash_file(path)}
    elif args.mode == 'worker':
        try:
            result = worker(args.scope, args.scope_sha256, args.out, args.case, args.operation, args.until, code)
        except device.ResourceUnavailable as error:
            result = {'status': 'waiting_for_resources', 'case': args.case, 'updates': 0, 'reason': str(error)}
    elif args.mode == 'accept':
        path = accept(args.scope, args.scope_sha256, args.out, code)
        result = {'path': str(path), 'sha256': hash_file(path)}
    else:
        result = run(args.scope, args.scope_sha256, args.out, code, resume=args.resume, segment=args.segment)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

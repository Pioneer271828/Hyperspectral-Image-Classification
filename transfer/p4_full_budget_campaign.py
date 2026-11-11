"""Prepare, execute and accept one new P4 full-budget campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from fractions import Fraction
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
    path = Path(scope_path).resolve()
    return {'policy': read_pinned(path, sha)['policy'], 'scope': str(path), 'scope_sha256': sha}


def bound_folder(out, case, path_name=None):
    path = out / 'cases' / case['id']
    if path_name is not None:
        device.require(path_name in registration.ADMISSION_PATHS, 'Unregistered case path')
        path = path / path_name
    device.require(path.resolve() == path, 'Case output cannot traverse symlinks')
    return path


def case_paths(scope, folder):
    if not scope.get('paths'):
        return {None: (folder, None)}
    paths = {name: (folder / name, stops) for name, stops in registration.ADMISSION_PATHS.items()}
    device.require(all(path.resolve() == path for path, _ in paths.values()), 'Case path cannot traverse symlinks')
    return paths


def case_progress(scope, out, case, sha):
    return sum(audit.ledger(path, case, sha, path_name=name, stops=stops)[0]
               for name, (path, stops) in case_paths(scope, bound_folder(out, case)).items())


def accepted(folder, scope, case, sha):
    path = folder / 'audit.json'
    row = read(path)
    steps = scope['budget']['steps']
    paths = case_paths(scope, folder)
    case_files = {str(file) for path, _ in paths.values() for file in audit.case_files(path)}
    device.require(row['status'] == 'passed' and row['case'] == case and row['scope_sha256'] == sha
        and row['optimizer_updates'] == steps * len(paths) and row['completed_resume_updates'] == 0
        and row['audit_optimizer_updates'] == 0 and row['query_accessed'] is row['raw_gt_loaded'] is False
        and row['numeric']['physical_uuid'] == case['physical_uuid'] and row['numeric']['protocol'] == case['protocol']
        and row['validation_steps'] == scope['budget']['validation_steps']
        and row['selected_step'] in row['validation_steps'] and set(row['artifact_pins']) == case_files,
        'Case audit does not match the frozen scope')
    batch.check_pins(row['artifact_pins'])
    if scope.get('parameter_audit'):
        device.require(row['parameter_audit']['schema'] == scope['parameter_audit']
            and row['parameter_audit']['parameter_layout_sha256'] == case['parameter_layout_sha256'],
            'Accepted parameter audit differs from registered layout')
    if scope.get('paths'):
        fingerprints = row.get('exact_path_fingerprints', {})
        device.require(set(fingerprints) == set(paths) and fingerprints['full'] == fingerprints['split']
            and 'parameter_audit' in fingerprints['full'], 'Paired state and update evidence must match exactly')
    for name, (path, stops) in paths.items():
        cursor, calls = audit.ledger(path, case, sha, path_name=name, stops=stops)
        result = read(path / 'result.json')
        proof = read(path / 'completed-resume.json')
        device.require(cursor == steps and all(call['numeric'] == row['numeric'] for call in calls)
            and result['best_step'] == row['selected_step'] and result['identity'] == proof['identity']
            and result['runtime'] == audit.runtime_contract(sha, case, row['numeric'], name)
            and proof['cursor'] == steps and proof['new_optimizer_updates'] == 0,
            'Completed audit differs from worker checkpoint ledger')
    return row


def slot_lock(out, gpu):
    return out / '.cpu.lock' if gpu is None else out.parent / '.p4_gpu_leases' / (gpu + '.lock')


def worker(scope_path, sha, out, case_id, operation, until, code, *, lease_fds=None, path_name=None):
    scope, reg = registration.load(scope_path, sha, code, check_files=False)
    out = batch.sibling(out, scope_path)
    device.require(read(out / 'owner.json') == owner(out, scope_path, sha), 'Unowned campaign directory')
    matches = [case for case in scope['cases'] if case['id'] == case_id]
    device.require(len(matches) == 1, 'Unregistered case')
    case, = matches
    if lease_fds is not None or not scope['synthetic']:
        device.require(lease_fds is not None, 'Production worker requires the coordinator and slot leases')
        coordinator_fd, slot_fd = lease_fds
        device.check_inherited_lease(coordinator_fd, out / '.coordinator.lock')
        device.check_inherited_lease(slot_fd, slot_lock(out, case['physical_uuid']))
    case_folder = bound_folder(out, case)
    paths = case_paths(scope, case_folder)
    device.require((operation == 'audit' and path_name is None)
                   or (operation != 'audit' and path_name in paths), 'Operation requires its registered case path')
    folder = bound_folder(out, case, path_name)
    # Do not claim the case or initialize CUDA while the device is ineligible.
    with device.arithmetic(case['protocol'], case['physical_uuid']) as numeric:
        batch.check_pins(scope['protected_files'])
        qualification = scope.get('qualifications', {}).get(case['physical_uuid'])
        if qualification:
            receipt = read_pinned(Path(qualification), scope['protected_files'][qualification])
            device.require(receipt['numeric'] == numeric, 'CUDA identity differs from qualified hardware and runtime')
        case_folder.mkdir(exist_ok=True)
        with device.lease(case_folder / '.operation.lock'):
            if (case_folder / 'audit.json').exists():
                accepted(case_folder, scope, case, sha)
                device.require(operation != 'advance', 'Accepted case cannot be rerun')
                return {'status': 'accepted', 'case': case_id, 'updates': 0}
            steps = scope['budget']['steps']
            if operation == 'audit' and scope.get('paths'):
                device.require(until is None, 'Paired audit does not take a stop')
                row = audit.paired_case_audit(case_folder, scope, reg, case, sha, numeric)
                write_json(case_folder / 'audit.json', row)
                return {'status': 'accepted', 'case': case_id, 'updates': 0}
            stops = paths[path_name][1]
            cursor, calls = audit.ledger(folder, case, sha, path_name=path_name, stops=stops)
            device.require(all(call['numeric'] == numeric for call in calls), 'Cannot change the numerical resume device')
            runtime = audit.runtime_contract(sha, case, numeric, path_name)
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
                if path_name is not None:
                    proof['path'] = path_name
                if path.exists():
                    device.require(audit.path_proof(read(path)) == proof, 'Completed recovery evidence changed')
                else:
                    if scope.get('parameter_audit'):
                        proof['timing'] = result['timing']
                        audit.check_timing(proof['timing'], steps, steps, scope['budget']['validation_steps'], numeric)
                    write_json(path, proof)
                return {'status': 'verified_completed', 'case': case_id, 'updates': 0}
            device.require(operation == 'advance' and type(until) is int and cursor < until <= steps,
                           'Advance must consume a positive registered update interval')
            device.require(stops is None or (len(calls) < len(stops) and until == stops[len(calls)]),
                           'Advance must follow the registered paired stop schedule')
            started, tick = batch.now(), perf_counter()
            result = run_case(reg, case['index'], folder, until, code, coverage_arm=case['arm'], runtime=runtime)
            elapsed = perf_counter() - tick
            device.require(result['cursor'] == until and result['updates_this_call'] == until - cursor,
                           'Worker consumed an unexpected number of updates')
            folder.joinpath('calls').mkdir(exist_ok=True)
            path = folder / 'calls' / f'{cursor:06d}-{until:06d}.json'
            device.require(not path.exists(), 'Update receipt already exists')
            receipt = {'case': case_id, 'scope_sha256': sha, 'identity': result['identity'],
                'numeric': numeric, 'start': cursor, 'until': until, 'new_optimizer_updates': until - cursor,
                'state_sha256': hash_file(folder / 'state.pt'), 'started_utc': started,
                'finished_utc': batch.now(), 'wall_seconds': elapsed}
            if path_name is not None:
                receipt['path'] = path_name
            if scope.get('parameter_audit'):
                receipt['timing'] = result['timing']
                audit.check_timing(receipt['timing'], cursor, until, scope['budget']['validation_steps'], numeric)
            write_json(path, receipt)
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


def invoke_operation(scope_path, sha, out, code, case, operation, until, locks, launcher, *, path_name=None):
    # Check again in the isolated subprocess before touching CUDA.
    if case['physical_uuid'] is not None:
        device.gate(case['physical_uuid'])
    command = [sys.executable, '-m', 'transfer.p4_full_budget_campaign', 'worker',
               '--expected-commit', code['commit'], '--scope', str(scope_path), '--scope-sha256', sha,
               '--out', str(out), '--case', case['id'], '--operation', operation,
               '--coordinator-fd', str(locks[0].fileno()), '--slot-fd', str(locks[1].fileno())]
    if until is not None:
        command.extend(['--until', str(until)])
    if path_name is not None:
        command.extend(['--path', path_name])
    result = launcher(command, out / 'logs', case, operation, locks)
    if result['status'] == 'waiting_for_resources':
        raise device.ResourceUnavailable(result['reason'])
    return result


def run(scope_path, sha, out, code, *, resume=False, segment=None, launcher=invoke):
    scope, _ = registration.load(scope_path, sha, code)
    out = batch.sibling(out, scope_path)
    steps = scope['budget']['steps']
    paired = scope.get('paths') is not None
    device.require(not paired or segment is None, 'Paired admission has fixed uninterrupted and split stops')
    segment = steps if segment is None else segment
    device.require(type(segment) is int and 1 <= segment <= steps, 'Invalid controlled segment size')
    if not resume:
        out.mkdir(exist_ok=False)
    else:
        device.require(out.is_dir(), 'Resume requires an existing owned output')
    with device.lease(out / '.coordinator.lock') as coordinator, ExitStack() as starting_leases:
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
        slots = device.slots(scope['protocol'], scope['gpus'], recovery=scope.get('purpose') == registration.RECOVERY)
        if scope['gpus']:
            slot_lock(out, scope['gpus'][0]).parent.mkdir(exist_ok=True)
        event_path = out / 'queue-events.jsonl'

        def record(event):
            if paired:
                with event_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'utc': batch.now(), **event}) + '\n')

        prelocked = {}
        if paired and not scope['synthetic'] and not any(case_progress(scope, out, case, sha) for case in cases):
            try:
                prelocked = {gpu: starting_leases.enter_context(device.lease(slot_lock(out, gpu))) for gpu in slots}
                inventory = device.simultaneous_inventory(slots)
                record({'status': 'simultaneous_inventory', 'slots': inventory})
                if not all(device.eligible(row) for row in inventory.values()):
                    raise device.ResourceUnavailable('Both registered GPUs must qualify in the same inventory snapshot')
            except device.ResourceUnavailable as error:
                result = {'status': 'waiting_for_resources', 'scope_sha256': sha, 'cases': 0,
                          'optimizer_updates': 0, 'optimizer_updates_this_invocation': 0, 'slots': {},
                          'errors': {}, 'reason': str(error)}
                record({'status': 'waiting_for_resources', 'reason': str(error)})
                write_json(status_path, result)
                return result
            admission = out / 'first-dispatch-admission.json'
            if not admission.exists():
                write_json(admission, {'scope_sha256': sha, 'utc': batch.now(), 'slots': inventory})

        def publish(status, slot, case=None, reason=None, path_name=None):
            with mutex:
                states[str(slot)] = {'status': status, 'case': case, 'reason': reason}
                record({'status': status, 'slot': slot, 'case': case, 'path': path_name, 'reason': reason})
                write_json(status_path, {'status': 'draining_after_failure' if errors else 'running',
                    'scope_sha256': sha, 'completed_cases': len(completed), 'planned_cases': len(cases),
                    'optimizer_updates_this_invocation': new_updates, 'slots': states, 'errors': errors})

        def queue(slot):
            nonlocal new_updates
            active, path_name = None, None
            try:
                lease = nullcontext(prelocked[slot]) if slot in prelocked else device.lease(slot_lock(out, slot))
                with lease as lock:
                    for case in cases:
                        active = case['id']
                        if stop.is_set():
                            break
                        if case['physical_uuid'] != slot or active in completed:
                            continue
                        folder = bound_folder(out, case)
                        for path_name, (path, stops) in case_paths(scope, folder).items():
                            cursor, _ = audit.ledger(path, case, sha, path_name=path_name, stops=stops)
                            while cursor < steps and not stop.is_set():
                                until = next(n for n in stops if n > cursor) if stops else min(cursor + segment, steps)
                                publish('advancing', slot, active, path_name=path_name)
                                got = invoke_operation(scope_path, sha, out, code, case, 'advance', until,
                                                       (coordinator, lock), launcher, path_name=path_name)
                                device.require(got == {'status': 'advanced', 'case': active, 'updates': until - cursor},
                                               'Coordinator update accounting differs')
                                with mutex:
                                    new_updates += got['updates']
                                cursor = until
                            if stop.is_set():
                                break
                            if not (path / 'completed-resume.json').is_file():
                                publish('completed_recovery', slot, active, path_name=path_name)
                                proof = invoke_operation(scope_path, sha, out, code, case, 'completed', None,
                                                         (coordinator, lock), launcher, path_name=path_name)
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
                publish('waiting_for_resources', slot, active, str(error), path_name)
            except Exception as error:
                stop.set()
                with mutex:
                    errors[str(slot)] = {'case': active, 'reason': repr(error)}
                publish('failed', slot, active, repr(error), path_name)

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
        consumed = sum(case_progress(scope, out, case, sha) for case in cases) if not errors else None
        result = {'status': status, 'scope_sha256': sha, 'cases': len(completed), 'optimizer_updates': consumed,
                  'optimizer_updates_this_invocation': new_updates, 'slots': states, 'errors': errors}
        write_json(status_path, result)
        if errors:
            raise RuntimeError(f'Campaign stopped dispatch after failure, see {status_path}')
        require_frozen_code(code)
        return result


def admission_validation(rows):
    """Report paired short-admission metrics while retaining class protection failures."""
    references = {row['case']['pair_sha256']:row for row in rows if row['case']['arm']=='none'}
    report = []
    for row in rows:
        current = row['fit']['selected']['val']['heads']
        baseline = references[row['case']['pair_sha256']]['fit']['selected']['val']['heads']
        heads = {}
        for head, score in current.items():
            other = baseline[head]
            supports = [sum(line) for line in score['confusion']]
            device.require(len(supports)==22 and supports == [5]*22
                and [sum(line) for line in other['confusion']] == supports,
                'Short admission must retain all 22 validation classes')
            recalls = [Fraction(score['confusion'][i][i]-other['confusion'][i][i], support)
                       for i,support in enumerate(supports)]
            heads[head] = {'oa':score['oa'], 'aa':score['aa'], 'kappa':score['kappa'],
                'macro_f1':score['macro_f1'], 'nll':score['nll'], 'class_support':supports,
                'recall':score['recall'], 'zero_recall_classes':[i for i,v in enumerate(score['recall'],1) if v==0],
                'recall_delta_vs_none':[float(value) for value in recalls],
                'recall_drop_over_2pp_classes':[i for i,value in enumerate(recalls,1) if value < Fraction(-2,100)]}
        report.append({'case':row['case']['id'],'baseline':references[row['case']['pair_sha256']]['case']['id'],
                       'selected_step':row['selected_step'],'heads':heads})
    return report


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
        case_pins, all_artifacts, pairs, audits = {}, {}, {}, []
        for case in cases:
            folder = bound_folder(out, case)
            row = accepted(folder, scope, case, sha)
            audits.append(row)
            path = folder / 'audit.json'
            case_pins[str(path)] = hash_file(path)
            all_artifacts.update(row['artifact_pins'])
            pairs.setdefault(case['pair_sha256'], []).append(case['arm'])
        expected_arms = ('none',) if scope.get('purpose') == registration.RECOVERY else registration.coverage.ARMS
        device.require(len(pairs) * len(expected_arms) == len(cases)
                       and all(arms == list(expected_arms) for arms in pairs.values()),
                       'Missing matched coverage arm at acceptance')
        pins = {**scope['protected_files'], str(Path(scope_path).resolve()): sha, **case_pins, **all_artifacts}
        if scope.get('paths'):
            events = out / 'queue-events.jsonl'
            pins[str(events)] = hash_file(events)
            if not scope['synthetic']:
                admission = out / 'first-dispatch-admission.json'
                row = read(admission)
                device.require(row['scope_sha256'] == sha and set(row['slots']) == set(scope['gpus'])
                               and all(device.eligible(slot) for slot in row['slots'].values()),
                               'Missing simultaneous GPU admission evidence')
                pins[str(admission)] = hash_file(admission)
        batch.check_pins(pins)
        require_frozen_code(code)
        path = out / 'acceptance.json'
        result = {'status': 'passed', 'policy': scope['policy'], 'scope_sha256': sha, 'producer_commit': code['commit'],
            'cases': len(cases), 'pairs': len(pairs), 'optimizer_updates': status['optimizer_updates'],
            'nonselected_event_probability_replays': 0, 'audit_optimizer_updates': 0, 'raw_gt_loaded': False,
            'query_accessed': False, 'case_audits': case_pins, 'protected_files': pins}
        if scope.get('parameter_audit'):
            result.update(purpose=scope['purpose'], parameter_audit=scope['parameter_audit'], paths=scope['paths'],
                          synthetic=scope['synthetic'])
        if scope.get('purpose') == registration.ADMISSION and scope['protocol'] == device.CUDA_FP64_WEIGHT:
            result['validation_diagnostics'] = admission_validation(audits)
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
    prep.add_argument('--protocol', choices=(device.CPU, *device.CUDA_PROTOCOLS), required=True)
    prep.add_argument('--gpus', nargs='*', default=[])
    prep.add_argument('--purpose', choices=registration.POLICIES, default=registration.LEGACY)
    prep.add_argument('--qualification', type=Path, action='append', default=[])
    prep.add_argument('--recovery-receipt', type=Path)
    execute = modes.add_parser('run')
    execute.add_argument('--resume', action='store_true')
    execute.add_argument('--segment', type=int)
    work = modes.add_parser('worker')
    work.add_argument('--case', required=True)
    work.add_argument('--operation', choices=('advance', 'completed', 'audit'), required=True)
    work.add_argument('--until', type=int)
    work.add_argument('--path', choices=registration.ADMISSION_PATHS)
    work.add_argument('--coordinator-fd', type=int, required=True)
    work.add_argument('--slot-fd', type=int, required=True)
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
                                    args.out, code, protocol=args.protocol, gpus=args.gpus, purpose=args.purpose,
                                    qualifications=args.qualification, recovery_receipt=args.recovery_receipt)
        result = {'path': str(path), 'sha256': hash_file(path)}
    elif args.mode == 'worker':
        try:
            result = worker(args.scope, args.scope_sha256, args.out, args.case, args.operation, args.until, code,
                            lease_fds=(args.coordinator_fd, args.slot_fd), path_name=args.path)
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

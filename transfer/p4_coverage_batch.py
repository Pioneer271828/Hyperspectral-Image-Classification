"""Freeze and run the guarded HongHu five-arm short comparison."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import perf_counter

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer import p4_coverage as coverage
from transfer.g2_contract import read_pinned


POLICY = 'wp05_honghu_five_arm_short_v1'
PLAN_SHA = '27cf2ecfb99103cb32fac09b18219c871f9cc902631506aa428cdffca4141c9c'
PARENT_SHA = '3de2b10854b41901674c1c99dfaac9dc6c3af3eff7f5b08529953d368560a2a2'
PRIOR_SHA = '7a689279abd26c568a96191c093cd2bfd5137328852d1d9f56fc4e024ba58440'
OBSERVATION = {'mode': 'uniform_stride', 'stride': 2, 'offset': 0}
METHODS = ('scratch', 'shared_transfer')
COHORTS = ((512, 1), (512, 5), (2048, 1), (2048, 5))
FIELDS = ('model', 'optimizer', 'scheduler', 'rng', 'trace', 'history', 'best')


def check_pins(pins):
    for path, expected in pins.items():
        if hash_file(path) != expected:
            raise ValueError('Pinned input or artifact changed: ' + str(path))


def now():
    return datetime.now(timezone.utc).isoformat()


def manifest(registration, *, synthetic):
    cases = registration['cases']
    keys = [(case['support_seed'], case['k'], case['method']) for case in cases]
    expected = {(seed, k, method) for seed, k in COHORTS for method in METHODS}
    frozen = registration['coverage']
    steps, events = frozen['steps'], frozen['validation_steps']
    if len(keys) != len(expected) or set(keys) != expected or frozen['arms'] != list(coverage.ARMS):
        raise ValueError('Expected all 40 unique support/K/method/coverage cases')
    if (frozen['policy'] != coverage.POLICY or frozen['selection'] != coverage.SELECTION
            or frozen['observation'] != OBSERVATION or frozen['metadata'] != 'nominal'
            or frozen['observation_seed'] != 3031 or not 2 < events[0] <= steps or events[-1] != steps
            or (not synthetic and (steps, events, frozen['exposure_plan_sha256']) != (24, [22, 24], PLAN_SHA))):
        raise ValueError('Observation, validation events or registered update budget differs')
    rows = [{'id': f's{seed}-k{k}-{method}-{arm}', 'index': index, 'support_seed': seed, 'k': k,
             'method': method, 'arm': arm}
            for index, (seed, k, method) in enumerate(keys) for arm in coverage.ARMS]
    return {'policy': POLICY, 'synthetic': synthetic, 'commit': registration['commit'], 'cases': rows,
            'steps': steps, 'split_after': 2, 'validation_steps': events, 'selection': frozen['selection'],
            'optimizer_updates': len(rows) * steps * 2}


def sibling(out, anchor):
    out = external_directory(out)
    if out.parent != Path(anchor).resolve().parent.parent:
        raise ValueError('Stage output must be a new sibling directory')
    return out


def prepare(parent, parent_sha, prior, prior_sha, out, code, *, synthetic=False):
    parent, prior = Path(parent).resolve(), Path(prior).resolve()
    require_frozen_code(code)
    if not synthetic and (parent_sha, prior_sha) != (PARENT_SHA, PRIOR_SHA):
        raise ValueError('Unexpected accepted exposure registration or prior receipt SHA')
    accepted = read_pinned(prior, prior_sha)
    if (accepted['status'] != 'passed' or accepted['protected_files'].get(str(parent)) != parent_sha
            or accepted['query_accessed'] is not False):
        raise ValueError('Prior audit does not authenticate the exposure registration')
    registration = coverage.prepare(parent, parent_sha, commit=code['commit'], observation=OBSERVATION,
                                    seed=3031, metadata='nominal')
    scope = manifest(registration, synthetic=synthetic)
    pins = {**registration['pins'], str(parent): parent_sha, str(prior): prior_sha}
    plan = read_pinned(Path(registration['exposure']['plan']), registration['coverage']['exposure_plan_sha256'])
    if not synthetic and plan['sampler_seed'] != 64:
        raise ValueError('Historical sampler seed differs')
    check_pins(pins)
    out = sibling(out, parent)
    out.mkdir(exist_ok=False)
    reg_path = out / 'registration.json'
    write_json(reg_path, registration)
    scope.update(registration=str(reg_path), registration_sha256=hash_file(reg_path),
                 parent_registration=str(parent), prior_receipt=str(prior), protected_files=pins)
    scope['scope_hash'] = digest(scope)
    path = out / 'scope.json'
    write_json(path, scope)
    require_frozen_code(code)
    return path


def load_scope(path, sha, code):
    require_frozen_code(code)
    scope = read_pinned(Path(path), sha)
    if (scope['scope_hash'] != digest({k: v for k, v in scope.items() if k != 'scope_hash'})
            or scope['commit'] != code['commit']):
        raise ValueError('Scope code or identity differs')
    reg = read_pinned(Path(scope['registration']), scope['registration_sha256'])
    if any(scope[k] != v for k, v in manifest(reg, synthetic=scope['synthetic']).items()):
        raise ValueError('Scope case matrix or budget differs')
    parent, prior = (scope[k] for k in ('parent_registration', 'prior_receipt'))
    pins = scope['protected_files']
    if (set(pins) != set(reg['pins']) | {parent, prior} or any(pins[k] != v for k, v in reg['pins'].items())
            or (not scope['synthetic'] and (pins[parent], pins[prior]) != (PARENT_SHA, PRIOR_SHA))):
        raise ValueError('Scope input pins differ')
    check_pins(pins)
    original = read_pinned(Path(parent), pins[parent])
    expected = {**original, 'commit': code['commit']}
    if {k: v for k, v in reg.items() if k not in ('coverage', 'geometry_hash')} != expected:
        raise ValueError('Batch registration differs from the authenticated parent')
    return scope, reg


def phases(scope):
    end, split = scope['steps'], scope['split_after']
    return (('full', 'full', end, end), ('first', 'split', split, split),
            ('resume', 'split', end, end - split), ('completed', 'split', end, 0))


def invocation(scope, case, out, path, until):
    return [sys.executable, '-m', 'transfer.p4_spatial', '--registration', scope['registration'],
            '--sha256', scope['registration_sha256'], '--case', str(case['index']), '--coverage-arm',
            case['arm'], '--out', str(out / case['id'] / path), '--until', str(until)]


def launch(command, log):
    with log.open('xb') as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def run_one(scope, case, out, stop, runner):
    root = out / case['id']
    root.mkdir(exist_ok=False)
    calls = []
    identity = None
    for name, path, until, updates in phases(scope):
        if stop.is_set():
            return None
        artifact = root / path
        before = ({n: hash_file(artifact / n) for n in ('state.pt', 'result.json')}
                  if name == 'completed' else None)
        command = invocation(scope, case, out, path, until)
        log = root / f'{name}.log'
        started, tick = now(), perf_counter()
        status = runner(command, log)
        event = {'phase': name, 'argv': command, 'exit_code': status, 'started_utc': started,
                 'finished_utc': now(), 'wall_seconds': perf_counter() - tick, 'log_sha256': hash_file(log)}
        write_json(root / f'{name}.json', event)
        if status:
            raise RuntimeError(f"{case['id']} {name} failed, see {log}")
        result = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
        identity = result['identity'] if identity is None else identity
        if result != {'cursor': until, 'updates_this_call': updates, 'identity': identity}:
            raise ValueError(f"{case['id']} {name} update count or identity differs")
        if before is not None:
            check_pins({artifact / n: sha for n, sha in before.items()})
        if name == 'first':
            import torch

            state = torch.load(artifact / 'state.pt', map_location='cpu', weights_only=True)
            partial = {'cursor': state['cursor'], 'validation_events': len(state['history']),
                       'has_selected_state': state['best'] is not None}
            if (partial != {'cursor': until, 'validation_events': 0, 'has_selected_state': False}
                    or (artifact / 'result.json').exists()):
                raise ValueError('Split path selected a model before first validation')
            event['partial'] = partial
        event.update(result=result, state_sha256=hash_file(artifact / 'state.pt'))
        write_json(root / f'{name}.json', event)
        calls.append(event)
    return {'case': case, 'calls': calls, 'updates': sum(e['result']['updates_this_call'] for e in calls)}


def run(scope_path, scope_sha, out, code, *, workers=2, runner=launch):
    scope, _ = load_scope(scope_path, scope_sha, code)
    if workers not in (1, 2):
        raise ValueError('Use one or two CPU case queues')
    out = sibling(out, scope_path)
    out.mkdir(exist_ok=False)
    started, tick = now(), perf_counter()
    stop = Event()
    rows, errors = {}, []

    def worker(case):
        try:
            return run_one(scope, case, out, stop, runner)
        except Exception:
            stop.set()
            raise

    remaining = iter(scope['cases'])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(worker, c): c for c in (next(remaining) for _ in range(workers))}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for task in done:
                case = pending.pop(task)
                try:
                    row = task.result()
                    if row is not None:
                        rows[case['id']] = row
                except Exception as error:
                    errors.append({'case': case['id'], 'error': repr(error)})
            for _ in done:
                following = None if stop.is_set() else next(remaining, None)
                if following is not None:
                    pending[pool.submit(worker, following)] = following
    if errors or len(rows) != len(scope['cases']):
        write_json(out / 'failure.json', {'status': 'failed', 'scope_sha256': scope_sha,
                   'started_utc': started, 'finished_utc': now(), 'wall_seconds': perf_counter() - tick,
                   'completed_cases': list(rows), 'errors': errors})
        raise RuntimeError(f'Batch stopped before all cases completed, see {out / "failure.json"}')
    rows = [rows[c['id']] for c in scope['cases']]
    updates = sum(row['updates'] for row in rows)
    if updates != scope['optimizer_updates']:
        raise ValueError('Total actual update budget differs')
    artifacts = {str(p.relative_to(out)): hash_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    check_pins({**scope['protected_files'], str(scope_path): scope_sha,
                scope['registration']: scope['registration_sha256']})
    require_frozen_code(code)
    result = {'status': 'completed', 'commit': code['commit'], 'scope_sha256': scope_sha,
              'started_utc': started, 'finished_utc': now(), 'wall_seconds': perf_counter() - tick,
              'workers': workers, 'rows': rows, 'optimizer_updates': updates, 'artifacts': artifacts,
              'raw_gt_loaded': False, 'query_accessed': False}
    path = out / 'execution.json'
    write_json(path, result)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--parent', type=Path, required=True)
    prep.add_argument('--prior-receipt', type=Path, required=True)
    execute = modes.add_parser('run')
    execute.add_argument('--scope', type=Path, required=True)
    execute.add_argument('--scope-sha256', required=True)
    execute.add_argument('--workers', type=int, choices=(1, 2), default=2)
    for mode in (prep, execute):
        mode.add_argument('--expected-commit', required=True)
        mode.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected batch commit')
    if args.mode == 'prepare':
        path = prepare(args.parent, PARENT_SHA, args.prior_receipt, PRIOR_SHA, args.out, code)
    else:
        path = run(args.scope, args.scope_sha256, args.out, code, workers=args.workers)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}), flush=True)


if __name__ == '__main__':
    main()

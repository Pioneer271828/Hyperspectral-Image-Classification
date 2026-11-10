"""Authenticated external curriculum registration and bounded CPU attempts.

Transport and model arithmetic are reused unchanged. A registration permits only
the released development package, not loading a dataset or its original GT.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from model.research_config import canonical_json
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.observation_curriculum import CurriculumPolicy
from transfer.observation_policy import ObservationPolicy
from transfer.registered_development import _read, _json_new, _load_source, load_development
from transfer.sample_curriculum import SampleCurriculum


FORMAT = 'vca-registered-curriculum-v1'
INPUTS = {'source', 'metadata', 'development', 'exposure', 'sample_policy', 'observation_policy'}


def _code(expected_commit):
    code = code_snapshot()
    if (not isinstance(expected_commit, str) or len(expected_commit) != 40
            or any(c not in '0123456789abcdef' for c in expected_commit)
            or not training_container_kind() or sys.flags.optimize
            or code['dirty'] or code['commit'] != expected_commit):
        raise ValueError('Clean exact registered code in Linux container required')
    return code


def _build(inputs, lr, torch_seed):
    if not isinstance(inputs, dict) or set(inputs) != INPUTS:
        raise ValueError('Explicit curriculum input schema required')
    if inputs['sample_policy'] is not None and inputs['observation_policy'] is not None:
        raise ValueError('Sample and observation curricula require separate experiments')
    # Authenticate all top-level inputs before deserializing any source weight.
    raw = {k: None if v is None else _read(v) for k, v in inputs.items()}
    if any(raw[k] is None for k in ('source', 'metadata', 'development', 'exposure')):
        raise ValueError('Source, metadata, development and exposure are required')
    scene, record, files = load_development(inputs['development'])
    value = json.loads(raw['exposure'])
    plan = ExposurePlan(scene, record, value, expected_plan_hash=value['plan_hash'])
    plan.require_eligible()
    policy = None if raw['sample_policy'] is None else SampleCurriculum(json.loads(raw['sample_policy']))
    observation = None
    if raw['observation_policy'] is not None:
        value = json.loads(raw['observation_policy'])
        observation = (CurriculumPolicy if 'curriculum' in value else ObservationPolicy)(value)
    source = _load_source(inputs['source'], inputs['metadata'])
    run = CurriculumRun(source, scene, record, plan=plan, policy=policy, lr=lr, torch_seed=torch_seed,
        expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
        expected_policy_hash=None if policy is None else policy.policy_hash,
        observation_policy=observation,
        expected_observation_hash=None if observation is None else observation.policy_hash)
    files += [deepcopy(v) for k, v in inputs.items() if v is not None and k != 'development']
    return run, files


def register_curriculum(inputs, *, out, lr, torch_seed, expected_commit):
    """Bind independently supplied file hashes; execute zero optimizer steps."""
    code = _code(expected_commit)
    run, files = _build(inputs, lr, torch_seed)
    out = external_directory(out)
    if any(out == Path(pin['path']).resolve() for pin in files):
        raise ValueError('Registration cannot overwrite an input')
    for pin in files:
        _read(pin)
    require_frozen_code(code)
    value = dict(format=FORMAT, expected_commit=expected_commit, inputs=deepcopy(inputs),
                 lr=lr, torch_seed=torch_seed, contract_hash=run.contract_hash, spec=run.spec,
                 query=0, updates_at_registration=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    return _json_new(out, value)


def load_registration(pin, expected_commit):
    value = json.loads(_read(pin))
    keys = {'format', 'expected_commit', 'inputs', 'lr', 'torch_seed', 'contract_hash',
            'spec', 'query', 'updates_at_registration'}
    if (not isinstance(value, dict) or set(value) != keys or value['format'] != FORMAT
            or value['expected_commit'] != expected_commit
            or type(value['query']) is not int or value['query'] != 0
            or type(value['updates_at_registration']) is not int or value['updates_at_registration'] != 0):
        raise ValueError('Unsupported registration or code identity')
    run, files = _build(value['inputs'], value['lr'], value['torch_seed'])
    if run.contract_hash != value['contract_hash'] or canonical_json(run.spec) != canonical_json(value['spec']):
        raise ValueError('Registered execution contract changed')
    return run, [deepcopy(pin), *files]


def execute_curriculum(registration, *, out, to_step, expected_commit, resume=None):
    """Save each accepted prefix in a new attempt; never auto-retry or reselect."""
    code = _code(expected_commit)
    if resume is not None:
        _read(resume)
    run, files = load_registration(registration, expected_commit)
    if type(to_step) is not int or not 0 <= to_step <= run.plan.updates:
        raise ValueError('Absolute target exceeds frozen budget')
    if resume is not None:
        files.append(deepcopy(resume))
        run = CurriculumRun.load(run.source, run.scene, run.record, resume['path'], plan=run.plan,
            policy=run.policy, observation_policy=run.observation_policy,
            expected_sha256=resume['sha256'], expected_contract_hash=run.contract_hash)
    if to_step < run.completed:
        raise ValueError('Cannot rewind an accepted prefix')
    out = external_directory(out)
    for pin in files:
        parent = Path(pin['path']).resolve().parent
        if out == parent or out in parent.parents or parent in out.parents:
            raise ValueError('Attempt output overlaps an input/history directory')
    out.mkdir(parents=True, exist_ok=False)
    initial = run.completed
    started = time.monotonic()
    status = dict(format=FORMAT, status='running', code=code, registration=deepcopy(registration),
        contract_hash=run.contract_hash, initial_updates=initial, completed_updates=initial,
        target_updates=to_step, total_budget=run.plan.updates, new_updates=0, checkpoint=None,
        query=0, independent_acceptance=False, inputs=files, command=list(sys.argv))

    def save_status():
        temp = out/'status.tmp'
        with temp.open('xb') as f:
            f.write((canonical_json(status)+'\n').encode())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, out/'status.json')

    def checkpoint():
        path = out/f'step-{run.completed}.pt'
        status.update(checkpoint=dict(path=str(path), sha256=run.save(path)),
                      completed_updates=run.completed, new_updates=run.completed-initial)
        save_status()

    try:
        checkpoint()
        while run.completed < to_step:
            run.step()
            checkpoint()
        for pin in files:
            _read(pin)
        require_frozen_code(code)
        status.update(status='completed' if run.completed == run.plan.updates else 'partial',
                      acceptance=run.finish() if run.completed == run.plan.updates else None,
                      elapsed_seconds=time.monotonic()-started)
        save_status()
        return deepcopy(status)
    except BaseException as error:
        if not run.failed:
            run.failed = True
            run.failure = dict(type=type(error).__name__, message=str(error), attempted_step=run.completed+1)
        path = out/'failure.pt'
        status.update(status='failed', failure=dict(path=str(path), sha256=run.save_failure(path)),
                      error=repr(error), elapsed_seconds=time.monotonic()-started)
        save_status()
        raise

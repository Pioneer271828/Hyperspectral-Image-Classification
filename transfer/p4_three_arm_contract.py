"""The bounded three-arm HongHu fitting registration contract."""
from pathlib import Path

from model.research_config import digest
from research_runtime import hash_file
from transfer import p4_coverage as coverage, p4_coverage_batch as batch, p4_runtime_device as device
from transfer.g2_contract import read_pinned

PURPOSE = 'honghu_three_arm_fit'
POLICY = 'wp05_honghu_three_arm_fit_v1'
ARMS = ('none', 'continuous', 'anchor_only')
SEEDS = (64, 512, 2048)
ROLE = (2048, 5, 'scratch')
UNSEEN = ('odd_stride', 'stride4', 'low_half', 'high_half', 'random50')
CONDITIONS = ('training_stride', 'odd_stride', 'stride4', 'full', 'low_half', 'high_half', 'random50')
STEPPING = {'real': (1500, 30, (2, 30, 1500)),
            'bounded_synthetic': (6, 3, (2, 3, 6)),
            'long_synthetic': (1500, 30, (2, 30, 1500))}
ADMISSION_COMMIT = '3a6381cb4b0c089a10123292bc831a3a613e6106'
ADMISSION_SOURCE = '1c4251f86d0c725323d0b00df80b516064e2ae32bc4436abe5203cd9dd5a495d'
ADMISSION_SHA = 'a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5'
LONG_PREFLIGHT_SHA = '358b50aee1bf51feeb80fe7ef1e603b42662da3c6bf6886fa4b636155ca391fd'
EXPECTED_RELEASE_SHA = 'c0ce157d9e69854940bd9f245bb82309346690fd9ad46bf57f487b606cbf033f'
EXPECTED_SEQUENCE_SHA = '4d3d89847c08eb1d0aef198349195895a7dc18bfecf81b1d4d898aa7eacec4db'
PROTECTION = {'max_oa_aa_drop_pp': .5, 'max_small_recall_drop_pp': 2,
              'max_small_train_support': 25, 'minimum_positive_seeds': 2}


def contract(variant, classes=None):
    device.require(variant in STEPPING, 'Unknown HongHu fitting variant')
    synthetic = variant != 'real'
    if variant == 'bounded_synthetic':
        device.require(type(classes) is int and classes in (9, 22), 'Expected 9 or 22 synthetic classes')
    else:
        device.require(classes in (None, 22), 'The full HongHu fitting clock requires 22 classes')
        classes = 22
    return {'variant': variant, 'classes': classes, 'stops': list(STEPPING[variant][2]),
            'conditions': list(CONDITIONS), 'primary_conditions': list(UNSEEN),
            'selection_condition': 'training_stride', 'protection': dict(PROTECTION),
            'research_inference': not synthetic, 'release': None, 'execution': None,
            'normalization': 'frozen_devtrain_minmax_selected_columns',
            'graph': 'learned_from_each_retained_view_with_original_coordinates'}


def validate_budget(fit, protocol, gpus, synthetic, steps, interval):
    variant = fit['variant']
    expected = contract(variant, fit['classes'])
    destination = Path(fit['execution'])
    device.require(destination.is_absolute() and destination == destination.resolve(),
                   'The fitting execution directory must have a fixed absolute identity')
    device.require({**fit, 'release': None, 'execution': None} == expected and synthetic is (variant != 'real')
                   and (steps, interval) == STEPPING[variant][:2],
                   'Three-arm fitting variant, budget, calendar or role differs')
    if synthetic:
        device.require(fit['release'] is None and (protocol, len(gpus)) in
                       ((device.CPU, 0), (device.CUDA_FP64_WEIGHT, 1)),
                       'Synthetic fitting requires CPU or one physical v4 GPU')
    else:
        device.require(protocol == device.CUDA_FP64_WEIGHT and len(gpus) in (2, 3),
                       'Real fitting requires two or three qualified physical v4 GPUs')
    device.slots(protocol, gpus, recovery=synthetic and bool(gpus))


def matrix(reg, keys, protocol, gpus, fit):
    index = keys.index(ROLE)
    variant = fit['variant']
    slots = device.slots(protocol, gpus, recovery=variant != 'real' and bool(gpus))
    roles = ((512, 'continuous'),) if variant == 'long_synthetic' else (
        (seed, arm) for seed in SEEDS for arm in ARMS)
    seed_slots = {seed: slots[i % len(slots)] for i, seed in enumerate(SEEDS)}
    support_seed, k, method = ROLE
    return [{'id': f's{support_seed}-k{k}-{method}-m{seed}-{arm}', 'index': index,
             'support_seed': support_seed, 'k': k, 'method': method, 'model_seed': seed,
             'sampler_seed': 64, 'arm': arm, 'physical_uuid': seed_slots[seed], 'protocol': protocol,
             'observation_sha256': reg['coverage']['records']['train']['record_hash']}
            for seed, arm in roles]


def check_initializations(cases, fit):
    initializations = {}
    for case in cases:
        init = case['initialization']
        details = init['coverage']
        none = case['arm'] == 'none'
        device.require(details['arm'] == case['arm'] and details['parameter_count'] == (0 if none else 194),
                       'The two active arms require the same 194 E1 parameters')
        group = initializations.setdefault(case['model_seed'], [])
        group.append(case)
    for group in initializations.values():
        device.require(len({c['pair_sha256'] for c in group}) == 1, 'Seed pair initialization differs')
        if fit['variant'] != 'long_synthetic':
            _, continuous, anchor = (c['initialization']['coverage'] for c in group)
            device.require(continuous['parameters'] == anchor['parameters']
                           and continuous['state_sha256'] == anchor['state_sha256'],
                           'Active arms do not share the same identity initialization')
    device.require(len({g[0]['initialization']['base_initial_state_sha256']
                        for g in initializations.values()}) == len(initializations),
                   'Distinct model seeds require distinct common initial states')


def check_observations(conditions):
    device.require(tuple(conditions) == CONDITIONS
                   and all(row['status'] == 'ready' for row in conditions.values()),
                   'Expected seven ready fixed HongHu conditions')


def check_reference(reg, cases, evidence, conditions, reference):
    """Compare the new release and view records against the frozen 1500-step preflight."""
    check_observations(conditions)
    device.require(evidence['s2048-k5']['release_sha256'] == EXPECTED_RELEASE_SHA
                   and evidence['s2048-k5']['sequence_sha256'] == EXPECTED_SEQUENCE_SHA
                   and evidence['s2048-k5'] == reference['exposure']['s2048-k5']
                   and conditions == reference['observation_diagnostics'],
                   'Released HongHu support, sequence or seven conditions changed')
    old = {(c['model_seed'], c['arm']): c['initialization'] for c in reference['cases']
           if (c['support_seed'], c['k'], c['method']) == ROLE}
    device.require(all(old[c['model_seed'], c['arm']] == c['initialization'] for c in cases)
                   and reg['coverage']['records']['val'] == conditions['training_stride']['sampling'],
                   'Shared initialization or training-stride selection view differs')


def release_pins(release, code, gpus, qualifications):
    """Consume a separately authorized execution release and the old accepted 2400 evidence."""
    path = release['path']
    approval = read_pinned(Path(path), release['sha256'])
    device.require(set(approval) == {'schema', 'decision', 'source_commit', 'source_hash', 'research_issue',
                                     'gpus', 'qualification_sha256', 'admission_scope',
                                     'admission_acceptance', 'planned_optimizer_updates'}
                   and approval['schema'] == 'wp05_three_arm_execution_release_v1'
                   and approval['decision'] == 'approved'
                   and (approval['source_commit'], approval['source_hash']) ==
                       (code['commit'], code['source_hash'])
                   and type(approval['research_issue']) is int and approval['research_issue'] > 0
                   and approval['gpus'] == list(gpus)
                   and approval['qualification_sha256'] == qualifications
                   and approval['planned_optimizer_updates'] == 13500,
                   'Separately reviewed 13500-step release required')
    pins = {path: release['sha256']}
    old_scope, receipt = [approval[key] for key in ('admission_scope', 'admission_acceptance')]
    device.require(old_scope['sha256'] == ADMISSION_SHA, 'Expected the frozen P4 2400 admission scope')
    scope = read_pinned(Path(old_scope['path']), ADMISSION_SHA)
    result = read_pinned(Path(receipt['path']), receipt['sha256'])
    device.require(scope['commit'] == ADMISSION_COMMIT and scope['source_hash'] == ADMISSION_SOURCE
                   and scope['purpose'] == 'real_admission' and scope['synthetic'] is False
                   and result['status'] == 'passed' and result['purpose'] == 'real_admission'
                   and result['producer_commit'] == ADMISSION_COMMIT
                   and result['scope_sha256'] == ADMISSION_SHA and result['cases'] == 20
                   and result['optimizer_updates'] == 2400 and result['audit_optimizer_updates'] == 0
                   and result['query_accessed'] is result['raw_gt_loaded'] is False
                   and result['synthetic'] is False
                   and result['protected_files'].get(old_scope['path']) == ADMISSION_SHA
                   and len(result['case_audits']) == 20
                   and all(result['protected_files'].get(k) == v
                           for k, v in {**scope['protected_files'], **result['case_audits']}.items()),
                   'The full accepted P4 2400 evidence is required')
    admitted = str(Path(receipt['path']).parent / 'first-dispatch-admission.json')
    resources = read_pinned(Path(admitted), result['protected_files'][admitted])
    device.require(resources['scope_sha256'] == ADMISSION_SHA
                   and set(resources['slots']) == set(scope['gpus'])
                   and all(device.eligible(row) for row in resources['slots'].values()),
                   'Missing paired P4 admission inventory')
    audited = {}
    for name, checksum in result['case_audits'].items():
        case = read_pinned(Path(name), checksum)
        device.require(case['status'] == 'passed' and case['scope_sha256'] == ADMISSION_SHA
                       and case['optimizer_updates'] == 120
                       and all(len(head['confusion']) == 22 and [sum(r) for r in head['confusion']] == [5]*22
                               for head in case['fit']['selected']['val']['heads'].values())
                       and case['completed_resume_updates'] == case['audit_optimizer_updates'] == 0
                       and case['exact_path_fingerprints']['full'] == case['exact_path_fingerprints']['split'],
                       'P4 admission case recovery or independent audit differs')
        audited[case['case']['id']] = case['case']
    device.require(audited == {case['id']: case for case in scope['cases']},
                   'The prerequisite P4 20-case matrix differs')
    return {**result['protected_files'], **pins, receipt['path']: receipt['sha256']}


def registered_for_dispatch(scope, out):
    fit = scope.get('fit')
    if fit is not None:
        device.require(Path(out).resolve() == Path(fit['execution']),
                       'Fitting scope has a different registered execution directory')
        if fit['variant'] == 'real':
            device.require(fit['release'] is not None,
                           'A separate real fitting execution release is required; preflight is read-only')


def scope_stops(scope):
    return scope.get('fit', {}).get('stops')

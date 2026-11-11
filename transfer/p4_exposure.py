"""Freeze one guarded support-block sequence per cohort for paired training."""
import argparse
from pathlib import Path

import numpy as np

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer import p4_guarded
from transfer.g2_contract import read_pinned


POLICY = 'p4_guarded_witness_exposure_v1'
CLASSES = 22


def schedule(training_tiles, witness, *, steps, events, seed, classes=CLASSES):
    """Use every audited witness block once before seeded uniform block sampling."""
    if (type(steps) is not int or steps < 1 or type(seed) is not int or not 0 <= seed < 2**32
            or not isinstance(events, list) or not events or any(type(i) is not int for i in events)
            or events != sorted(set(events)) or events[-1] != steps):
        raise ValueError('Expected fixed positive budget, seed and ordered validation events')
    counts = {tile['id']: np.bincount(labels.reshape(-1), minlength=classes + 1)[1:]
              for tile, labels in training_tiles}
    if (not counts or not witness or len(witness) != len(set(witness))
            or not set(witness) <= counts.keys()):
        raise ValueError('Exposure witness must contain unique released training tiles')
    if len(witness) > events[0] or np.any(np.sum([counts[i] for i in witness], axis=0) == 0):
        raise ValueError('Exposure witness must cover all classes before first validation')
    rng = np.random.default_rng(seed)
    sequence = rng.permutation(sorted(witness)).tolist()
    sequence.extend(rng.choice(sorted(counts), size=steps - len(sequence), replace=True).tolist())
    cumulative = np.cumsum([counts[tile] for tile in sequence], axis=0)
    first = int(np.flatnonzero((cumulative > 0).all(axis=1))[0]) + 1
    return {'tile_sequence': sequence, 'first_all_classes_step': first,
            'label_exposure_at_events': {str(i): cumulative[i - 1].tolist() for i in events}}


def cohort(floor, registration, case, training_tiles, *, steps, events, seed):
    key = case['support_seed'], case['k']
    matches = [row for row in floor['rows'] if (row['seed'], row['k']) == key]
    if (floor['status'] != 'passed' or len(matches) != 1
            or any(floor['protected_files'].get(path) != registration['pins'][path]
                   for path in (registration['development'], case['release']))):
        raise ValueError('Exposure proof does not bind the guarded support cohort')
    proof = matches[0]
    witness = proof['witness_tiles']
    if proof['minimum_tiles_to_see_all_classes'] != len(witness):
        raise ValueError('Exposure proof witness size differs')
    return {'support_seed': key[0], 'k': key[1], 'release_sha256': registration['pins'][case['release']],
            'witness_tiles': witness, **schedule(training_tiles, witness, steps=steps, events=events, seed=seed)}


def bind(registration, case, training_tiles):
    paths = registration['exposure']
    if set(paths) != {'plan', 'floor'}:
        raise ValueError('Expected pinned exposure plan and floor proof')
    frozen, floor = [read_pinned(Path(paths[name]), registration['pins'][paths[name]])
                     for name in ('plan', 'floor')]
    fields = {k: v for k, v in frozen.items() if k != 'plan_hash'}
    if (frozen['policy'] != POLICY or digest(fields) != frozen['plan_hash']
            or frozen['geometry_sha256'] != registration['pins'][registration['development']]
            or frozen['floor_sha256'] != registration['pins'][paths['floor']]):
        raise ValueError('Guarded exposure plan identity differs')
    options = {'steps': frozen['steps'], 'events': frozen['validation_steps'], 'seed': frozen['sampler_seed']}
    expected = cohort(floor, registration, case, training_tiles, **options)
    stored = [r for r in frozen['cohorts'] if (r['support_seed'], r['k']) == (case['support_seed'], case['k'])]
    if stored != [expected]:
        raise ValueError('Frozen exposure sequence or counts differ from the released support')
    return frozen, expected


def main():
    from transfer.p4_spatial import load_release

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'floor', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--floor-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--validation-steps', type=int, nargs='+', required=True)
    parser.add_argument('--sampler-seed', type=int, default=64)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected exposure planner commit')
    require_frozen_code(code)
    floor = read_pinned(args.floor, args.floor_sha256)
    original = args.registration.resolve()
    registration = read_pinned(original, floor['protected_files'][str(original)])
    if registration['config']['dataset'] != p4_guarded.SCENE or 'exposure' in registration:
        raise ValueError('Expected the accepted guarded baseline registration')
    inputs = {path: registration['pins'][path] for path in (
        registration['plan'], registration['development'], *registration['guarded'].values(),
        *(case['release'] for case in registration['cases']))}
    if any(hash_file(path) != sha for path, sha in inputs.items()):
        raise ValueError('Guarded support input changed')
    plan = read_pinned(Path(registration['plan']), inputs[registration['plan']])
    geometry = read_pinned(Path(registration['development']), inputs[registration['development']])
    _, allowed = p4_guarded.bind_inputs(registration, plan, geometry)
    rows = []
    for case in registration['cases']:
        if case['method'] == 'scratch':
            release = read_pinned(Path(case['release']), inputs[case['release']])
            role = load_release(release, plan, case['support_seed'], case['k'], geometry,
                                guarded_tiles=allowed)
            rows.append(cohort(floor, registration, case, role['train'], steps=args.steps,
                               events=args.validation_steps, seed=args.sampler_seed))
    if any(hash_file(path) != sha for path, sha in inputs.items()):
        raise RuntimeError('Guarded support input changed during planning')
    require_frozen_code(code)
    record = {'policy': POLICY, 'geometry_sha256': inputs[registration['development']],
              'floor_sha256': args.floor_sha256, 'steps': args.steps, 'validation_steps': args.validation_steps,
              'sampler_seed': args.sampler_seed, 'cohorts': rows}
    record['plan_hash'] = digest(record)
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', record)


if __name__ == '__main__':
    main()

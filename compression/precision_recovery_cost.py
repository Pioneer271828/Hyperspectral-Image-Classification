"""Benchmark the independently audited WP10 packages with the original CPU methods."""
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch

from compression.g4_cpu_cost import _p95
from compression.precision_cost import benchmark_precision_cost, prepare_resident_precision_tiles
from compression.precision_recovery import ARMS, _kernel_hook
from compression.precision_recovery_audit import load_package
from compression.precision_startup import measure_precision_startup
from compression.registered_precision_recovery import load_registration
from research_runtime import ROOT, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev


def _cost_inputs(registration, evidence, code):
    root, reg, _, scene, _, _ = load_registration(registration, code)
    audit = admission.read(evidence)
    if (evidence['path'] != str(root / 'audit.json') or audit['registration'] != registration
            or audit['status'] != 'passed_precision_recovery_independent'
            or audit['actual_updates'] != reg['maximum_updates'] or audit['independent_updates'] != 0):
        raise ValueError('Cost requires the completed independent precision audit')
    cases = {'fp32': dict(arm='fp32', passed=True, format='fp32')}
    for family, key in (('qat', 'ptq'), ('ternary', 'ternary')):
        chosen = audit['decisions']['selected'][family]['arm']
        # A fixed first-layer example remains measurable if the family fell back.
        cases[key] = dict(arm=chosen if chosen != 'fp32' else family + '-first',
                          passed=chosen != 'fp32', format='qat_integer' if family == 'qat' else 'ternary_packed')
    for case in cases.values():
        item = audit['report'][case['arm']]['selected']
        package = item['package']
        if (package['format'] != case['format']
                or package['weights']['path'] != str(root / 'deployment' / case['arm'] / 'selected.pt')):
            raise ValueError('Wrong independently packaged selected endpoint')
        case['package'] = package
    tiles = prepare_resident_precision_tiles(scene, reg['validation']['tiles'])
    return root, reg, scene.dataset, cases, tiles


def startup(registration, evidence, key, code):
    _, _, dataset, cases, tiles = _cost_inputs(registration, evidence, code)
    if key not in cases:
        raise ValueError('Unknown registered startup package')
    case = cases[key]
    p = case['package']
    return measure_precision_startup(key, p['weights']['path'], tiles, dataset, expected_sha256=p['weights']['sha256'],
        expected_sidecar_sha256=None if p['sidecar'] is None else p['sidecar']['sha256'], expected_recovered_state=p['state'])


@torch.no_grad()
def _kernel_evidence(model, arm, tiles, dataset):
    mode, paths = ARMS[arm]
    counts = dict.fromkeys(paths, 0)
    with ExitStack() as stack:
        for path in paths:
            module = model.get_submodule(path)
            target = module.kernel if mode == 'qat' else module
            stack.enter_context(target.register_forward_hook(_kernel_hook(mode == 'qat', path, counts)))
        for tile in tiles:
            model(dataset, tile.image, wavelengths=tile.wavelengths,
                  superpixel_scale=50 if dataset == 'paviaU' else 100)
    if counts != dict.fromkeys(paths, len(tiles)):
        raise ValueError('The selected actual package did not execute each declared precision operator')
    return dict(calls=counts, kind={'fp32': 'fp32', 'qat': 'int8_convolution', 'ternary': 'packed_decode_fp32'}[mode])


def cost(registration, evidence, code):
    root, reg, dataset, cases, tiles = _cost_inputs(registration, evidence, code)
    out = root / 'cost'
    out.mkdir(exist_ok=False)
    models = {name: load_package(case['package']) for name, case in cases.items()}
    proof = {key: _kernel_evidence(model, cases[key]['arm'], tiles, dataset) for key, model in models.items()}
    restored = dict(common_dense_state=reg['initial']['fp32']['evidence']['dense_state'],
                    package_states={key: case['package']['state'] for key, case in cases.items()})
    started = perf_counter()
    resident = benchmark_precision_cost(models, tiles, dataset, recovered=restored)
    resident_seconds = perf_counter() - started
    pin_resident = dev._json_new(out / 'resident.json', resident)
    commands = []
    workers = {key: [] for key in cases}
    program = str(ROOT / 'scripts' / 'run_precision_recovery.py')
    sequence = list(cases)
    started = perf_counter()
    for repeat in range(3):
        for key in sequence[repeat:] + sequence[:repeat]:
            command = [sys.executable, program, '--expected-commit', code['commit'], 'startup',
                       '--registration', registration['path'], registration['sha256'],
                       '--audit', evidence['path'], evidence['sha256'], '--arm', key]
            log = out / f'startup-{repeat}-{key}.log'
            errors = out / f'startup-{repeat}-{key}.stderr.log'
            with log.open('x') as stdout, errors.open('x') as stderr:
                subprocess.run(command, stdout=stdout, stderr=stderr, check=True, cwd=ROOT)
            commands.append(command)
            row = json.loads(log.read_text())
            if (row['output_hashes'] != resident['output_hashes'][key]
                    or row['package_bytes'] != cases[key]['package']['bytes']):
                raise ValueError('Independent worker disagrees with the resident package')
            workers[key].append(row)
    startup_seconds = perf_counter() - started
    pids = {row['pid'] for rows in workers.values() for row in rows}
    if len(pids) != 9:
        raise ValueError('Expected one isolated worker process per first sweep')
    summary = {}
    for key, rows in workers.items():
        measures = dict(load_ms=[r['load_ms'] for r in rows], first_sweep_ms=[r['first_sweep_ms'] for r in rows],
                        peak_rss_kib=[r['after_first_sweep']['peak_rss_kib'] for r in rows])
        summary[key] = {name: dict(p50=median(x), p95=_p95(x), min=min(x), max=max(x)) for name, x in measures.items()}
    admission.check_pins([registration, evidence, *[row['package']['weights'] for row in cases.values()],
                          cases['fp32']['package']['sidecar']])
    require_frozen_code(code)
    return dev._json_new(out / 'receipt.json', dict(status='passed_precision_recovery_cpu_cost',
        registration=registration, audit=evidence, cases=cases, actual_kernel_evidence=proof,
        resident=pin_resident, workers=workers, startup_summary=summary, commands=commands,
        resident_seconds=resident_seconds, startup_seconds=startup_seconds, optimizer_updates=0, gt=0, query=0))

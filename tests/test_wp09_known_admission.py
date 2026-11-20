"""CLI integration, authorization denial tests and independent scoring oracle."""
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluate_tta import HEADS, _Session
from evaluation.known_development_authority import AUTHORITY_KIND
from evaluation.known_source_admission import admit_source
from evaluation.online_tta import _labels, _paired, score
from evaluation.open_set_producer import ROLES
from loadData.observation_sampling import BandMetadata
from model.checkpoint_io import load_model_checkpoint
from model.research_config import canonical_json
from research_runtime import ROOT, code_snapshot, hash_file
from transfer.g3_training import state_hash
from transfer.online_tta_run import execute, runtime_identity
from transfer.online_tta_stream import ORDERS, VERSION, StreamRegistration, frame_budget
from transfer.registered_development import _json_new, _pin


CLASSES = [2, 7, 19]
WINDOWS = {'A': [0, 4, 0, 3], 'B': [0, 4, 3, 6]}
FRAMES = [('A1', 'A'), ('B1', 'B'), ('A2', 'A'), ('B2', 'B')]
MODES = ('frozen', 'affine', 'adapter')
DEVELOPMENT_INDICES = np.arange(0, 24, 2, dtype=np.int32)
DEVELOPMENT_LABELS = np.array([2, 7, 2, 0, 7, 2, 2, 7, 0, 7, 2, 7], dtype=np.int64)
SENTINEL = """import os
import sys

target = os.environ.get('WP09_WATCHED_LABEL')
guard = os.environ.get('WP09_ALLOW_LABEL')
trace = os.environ.get('WP09_LABEL_OPEN_LOG')

def audit(event, args):
    if event == 'open' and isinstance(args[0], str) and os.path.abspath(args[0]) == target:
        with open(trace, 'a') as log:
            log.write(guard + '\\n')
        if guard != 'allow':
            raise RuntimeError('Premature development label payload read')

if target:
    sys.addaudithook(audit)
"""


def _read(pin_or_path):
    path = Path(pin_or_path['path']) if isinstance(pin_or_path, dict) else pin_or_path
    return json.loads(path.read_bytes())


def _args(name, pin):
    return [f'--{name}', pin['path'], f'--{name}-sha256', pin['sha256']]


def _fixture(root, sha):
    inputs = root / 'inputs'
    inputs.mkdir()
    rng = np.random.default_rng(192)
    waves = np.linspace(450, 900, 7, dtype=np.float32)
    dev_indices, targets = DEVELOPMENT_INDICES, DEVELOPMENT_LABELS
    observations = []
    for i, role in enumerate(ROLES):
        shape = (8, 12) if i == 0 else (4, 6)
        x = rng.normal(0.2 * i, 0.7, (*shape, 7)).astype(np.float32)
        x[..., 0] = np.float32(3 + 0.1 * i)
        indices = np.arange(57, dtype=np.int32) if i == 0 else dev_indices
        arrays = dict(spectral=x, wavelengths_nm=waves, indices=indices)
        if i == 0:
            labels = np.resize(CLASSES, shape[0] * shape[1]).astype(np.uint8)
            labels[:57] = np.repeat(CLASSES, [26, 25, 6])
            arrays['context_labels'] = labels.reshape(shape)
        if i == 1:
            arrays['context_labels'] = np.resize(CLASSES, 24).astype(np.uint8).reshape(shape)
        path = inputs / f'{role}.npz'
        np.savez(path, **arrays)
        observations.append(dict(role=role, scene_uid='fixture-scene', scene_shape=[30, 30],
                                 origin=[i * 10, 0], superpixel_scale=4, input=_pin(path)))
    sample_ids = [json.dumps(['fixture-scene', 20 + int(index) // 6, int(index) % 6], separators=(',', ':'))
                  for index in dev_indices]
    release = dict(version=1, role=ROLES[-1], sample_ids=sample_ids, labels=targets.tolist())
    label_bytes = (canonical_json(release) + '\n').encode()
    label_path = inputs / 'deferred-development-labels.json'
    context = _json_new(inputs / 'context.json', dict(version=1, kind='known_full_context_preparation_v1',
        known=[dict(id=label, name=f'known-{label}') for label in CLASSES], unknown=[dict(id=31, name='held-out')],
        dataset='xuzhou', observations=observations,
        labels=dict(path=str(label_path), sha256=sha256(label_bytes).hexdigest())))
    bands = BandMetadata('synthetic-fixture-192', tuple(range(7)), tuple(map(float, waves)),
                         'nm', 'approximate', (True,) * 7)
    evidence = _json_new(inputs / 'fixture-origin.json', dict(scope='synthetic', seed=192, scenario='known-only-rectangles'))
    authority = _json_new(inputs / 'authority.json', dict(kind=AUTHORITY_KIND, scope='synthetic',
        context_registration=context, dataset_evidence=evidence, preparation_commit=sha, training_commit=sha,
        bands=asdict(bands), windows=WINDOWS))
    config = _read(ROOT / 'configs/experiments/screen_m6_none.json')
    config['model'].update(latent_channels=8, adapter_embed_dim=8)
    config = _json_new(inputs / 'config.json', config)
    return authority, context, config, label_path, label_bytes, dev_indices, targets


class Pipeline:
    def __init__(self, root):
        self.root = root
        self.code = code_snapshot()
        self.sha = self.code['commit']
        self.authority, self.context, self.config, self.label_path, self.label_bytes, self.indices, self.targets = (
            _fixture(root, self.sha))
        self.log = root / 'commands'
        self.log.mkdir()
        self.watcher = root / 'watcher'
        self.watcher.mkdir()
        (self.watcher / 'sitecustomize.py').write_text(SENTINEL)
        self.cases, self.commands, self.denials = [], [], []

    def cli(self, tag, entry, *args, allow=False):
        marker = self.log / f'{tag}.label-opens'
        env = dict(os.environ, PYTHONPATH=str(self.watcher) + os.pathsep + str(ROOT),
                   WP09_WATCHED_LABEL=str(self.label_path), WP09_ALLOW_LABEL='allow' if allow else 'deny',
                   WP09_LABEL_OPEN_LOG=str(marker))
        command = [sys.executable, *entry, *args]
        start = perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (self.log / f'{tag}.log').write_text(result.stdout + result.stderr)
        self.commands.append(dict(tag=tag, argv=command, exit_code=result.returncode, seconds=perf_counter() - start))
        _json_new(self.log / f'{tag}.command.json', self.commands[-1])
        if result.returncode:
            raise AssertionError(f'{tag}: {result.stderr}')
        if allow:
            assert marker.read_text().splitlines() == ['allow']
        else:
            assert not marker.exists(), tag
        return json.loads(result.stdout) if result.stdout else None

    def train_and_admit(self):
        prepared_out = self.root / 'prepared'
        prepared = self.cli('prepare', ['scripts/prepare_open_set_contexts.py'], *_args('registration', self.context),
                            '--expected-commit', self.sha, '--out', str(prepared_out))
        trial = _json_new(self.root / 'inputs' / 'train.json', dict(version=2, kind='known_source_authorized_v2',
            authority=self.authority, prepared=prepared, config=self.config, seed=64, updates=2))
        context, packet = _read(self.context), _read(prepared)
        withheld = [Path(row['input']['path']) for value in (context, packet) for row in value['observations'][1:]]
        for path in withheld:
            path.rename(path.with_suffix('.withheld'))
        try:
            self.source = self.cli('train', ['scripts/train_known_source.py'], *_args('registration', trial),
                '--expected-commit', self.sha, '--out', str(self.root / 'trained'))
        finally:
            for path in withheld:
                path.with_suffix('.withheld').rename(path)
        receipt = _read(self.source)
        assert receipt['optimizer_updates'] == receipt['selected_step'] == 2
        assert receipt['source_annotated_pixels'] == 96 and receipt['source_supervised_pixels'] == 57
        model = load_model_checkpoint(receipt['checkpoint']['path'], 'cpu')
        assert state_hash(model) == receipt['final_state_sha256'] != receipt['initial_state_sha256']
        assert model.training_config['registration'] == trial
        self.admission = self.cli('admit', ['-m', 'evaluation.known_source_admission'],
            *_args('authority', self.authority), *_args('source-receipt', self.source),
            '--expected-commit', self.sha, '--out', str(self.root / 'admitted'))
        admitted = _read(self.admission)
        assert admitted['class_ids'] == CLASSES and admitted['training_class_counts'] == [26, 25, 6]
        assert str(self.label_path) not in Path(self.admission['path']).read_text()
        self.spec = dict(version=VERSION, commit=self.sha, source_hash=self.code['source_hash'],
            runtime=runtime_identity(), source=admitted['source'], provenance=self.admission,
            class_ids=CLASSES, bands=_read(self.authority)['bands'], observations=admitted['observations'],
            frames=[dict(frame_id=key, scene='fixture-scene', segment=window, observation=window)
                    for key, window in FRAMES])

    def run_streams(self):
        registrations = self.root / 'registrations'
        registrations.mkdir()
        actual = dict(predictions=0, optimizer_updates=0, statistics_updates=0)
        for mode in MODES:
            for order in ORDERS:
                tag = f'{mode}-{order}'
                reg = _json_new(registrations / f'{tag}.json', dict(self.spec, order=order,
                    settings=dict(task='xuzhou', mode=mode, reset='continual', scale=4, lr=.05, rank=2, seed=21),
                    budget=frame_budget(mode, 4)))
                args = [*_args('registration', reg), *_args('admission', self.admission)]
                run = self.root / f'run-{tag}'
                result = self.cli(f'{tag}-full', ['-m', 'run_online_tta'], *args, '--out', str(run))
                assert result['added'] == 4 and result['actual'] == frame_budget(mode, 4)
                for key in actual:
                    actual[key] += result['actual'][key]
                retry_run = run
                if order == 'original':
                    retry_run = self.root / f'split-{tag}'
                    prefix = self.cli(f'{tag}-prefix', ['-m', 'run_online_tta'], *args, '--out', str(retry_run), '--to-frame', '2')
                    resumed = self.cli(f'{tag}-resume', ['-m', 'run_online_tta'], *args, '--out', str(retry_run))
                    assert prefix['added'] == resumed['added'] == 2 and resumed['tip'] == result['tip']
                    assert prefix['actual'] == resumed['actual'] == frame_budget(mode, 2)
                    for key in actual:
                        actual[key] += prefix['actual'][key] + resumed['actual'][key]
                    for i in range(1, 5):
                        left = run / f'frame-{i:06d}' / 'event.json'
                        right = retry_run / f'frame-{i:06d}' / 'event.json'
                        assert hash_file(left) == hash_file(right)
                retry = self.cli(f'{tag}-complete', ['-m', 'run_online_tta'], *args, '--out', str(retry_run))
                assert retry['added'] == 0 and retry['actual'] == frame_budget(mode, 0)
                anchor = self.root / f'anchor-{tag}'
                self.cli(f'{tag}-replay', ['-m', 'transfer.online_tta_replay'], *args, '--run', str(run), '--out', str(anchor))
                self.cases.append(dict(registration=reg, directory=str(run), replay=_pin(anchor / 'replay.json')))
        assert actual == dict(predictions=48, optimizer_updates=32, statistics_updates=0)
        return actual

    def offline(self):
        assert not self.label_path.exists()
        request = _json_new(self.root / 'registrations' / 'freeze-plan.json', dict(version=2, runs=self.cases))
        self.release = self.root / 'label-release.json'
        self.trust = dict(authority=self.authority, admission=self.admission, source_receipt=self.source)
        flags = [*_args('authority', self.authority), *_args('admission', self.admission),
                 *_args('source-receipt', self.source)]
        self.cli('freeze-label-release', ['-m', 'evaluation.online_tta'], *_args('plan', request), *flags,
                 '--freeze-release', '--out', str(self.release))
        self.plan = _json_new(self.root / 'registrations' / 'score-plan.json',
                              dict(version=2, runs=self.cases, label_release=_pin(self.release)))
        assert not self.label_path.exists()
        self.label_path.write_bytes(self.label_bytes)
        out = self.root / 'scores.json'
        self.cli('score', ['-m', 'evaluation.online_tta'], *_args('plan', self.plan), *flags,
                 '--out', str(out), allow=True)
        report = _read(out)
        assert report['label_scope'] == 'synthetic' and report['small_classes'] == [7, 19]
        assert report['annotation_costs'] == dict(source_pixels=96, calibration_pixels=24,
            source_supervised=57, calibration_selected=12, development_selected=12)
        return report

    def deny(self, name, fn, match):
        path_read = Path.read_bytes

        def guard(path):
            if path.resolve() == self.label_path:
                raise AssertionError(f'{name} opened target label payload')
            return path_read(path)

        with (
            patch.object(Path, 'read_bytes', guard),
            patch.object(_Session, 'adapt', side_effect=AssertionError('adapt')),
            patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('train')),
            patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('SGD')),
        ):
            try:
                fn()
            except (ValueError, RuntimeError) as error:
                assert match in str(error), (name, str(error))
                self.denials.append(dict(case=name, error=str(error), target_label_reads=0, optimizer_updates=0))
            else:
                raise AssertionError(f'{name} was accepted')

    def failures(self, directory='probes'):
        probes = self.root / directory
        probes.mkdir()
        original = _read(self.cases[0]['registration'])
        label_pin = _read(self.context)['labels']
        for name, edits, error in (
            ('class-order', {'class_ids': [7, 2, 19]}, 'Head class order'),
            ('band-order', {'bands': dict(original['bands'], band_ids=list(reversed(original['bands']['band_ids'])))},
             'Source or observation provenance'),
            ('frame-region', {'frames': [dict(original['frames'][0], scene='other-scene'), *original['frames'][1:]]},
             'admitted development scene'),
            ('weight-alias', {'source': dict(original['source'], weights=label_pin)},
             'Source or observation provenance'),
            ('groupnorm-statistics', {'settings': dict(original['settings'], mode='statistics'),
                                      'budget': frame_budget('statistics', 4)}, 'GroupNorm'),
        ):
            reg = _json_new(probes / f'{name}.json', dict(original, **edits))
            out = probes / f'output-{name}'
            self.deny(name, lambda: execute(reg, out, admission=self.admission), error)
            assert not out.exists()
        self.deny('missing-independent-admission',
            lambda: execute(self.cases[0]['registration'], probes / 'untrusted'), 'operator-approved')
        fabricated = _json_new(probes / 'self-declared-real.json', dict(kind='registered_development', origin_hash='f' * 64))
        reg = _json_new(probes / 'forged.json', dict(original, provenance=fabricated))
        self.deny('self-declared-real', lambda: StreamRegistration(reg), 'Unsupported source provenance')
        self.deny('historical-pavia', lambda: _labels(label_pin,
                  SimpleNamespace(provenance={'kind': 'e4b_pavia_development'})), 'no scoring authorization')

        source_receipt = _read(self.source)
        for name, edits, error in (
            ('wrong-final-step', {'selected_step': 1}, 'fixed final step'),
            ('wrong-source-count', {'source_supervised_pixels': 56}, 'support counts'),
            ('wrong-weight-state', {'final_state_sha256': '0' * 64}, 'Actual source weight state'),
            ('source-as-label', {'checkpoint': label_pin}, 'must be distinct'),
            ('forged-producer', {'code': dict(source_receipt['code'], commit='f' * 40)}, 'code identity'),
        ):
            pin = _json_new(probes / f'{name}.json', dict(source_receipt, **edits))
            out = probes / f'output-{name}'
            self.deny(name, lambda: admit_source(self.authority, pin, out=out, expected_commit=self.sha), error)
            assert not out.exists()
        trusted = _read(self.authority)
        swap = _json_new(probes / 'synthetic-upgrade.json', dict(trusted, scope='registered_development'))
        self.deny('synthetic-upgrade', lambda: admit_source(swap, self.source, out=probes / 'real', expected_commit=self.sha),
                  'not pre-registered under this data authority')
        trial = _read(source_receipt['registration'])
        historical = _json_new(probes / 'historical-trial.json', dict(version=1, kind='known_source_fixed_budget_v1',
            config=trial['config'], seed=trial['seed'], updates=trial['updates'], prepared=trial['prepared']))
        historical_pin = _json_new(probes / 'historical-receipt.json', dict(source_receipt, registration=historical))
        self.deny('historical-source', lambda: admit_source(self.authority, historical_pin, out=probes / 'historical',
                  expected_commit=self.sha), 'authority')
        wrong_parent = _json_new(probes / 'arbitrary-parent.json', {'looks': 'valid'})
        self.deny('arbitrary-source-parent', lambda: score(self.plan, **dict(self.trust, source_receipt=wrong_parent)),
                  'parent source training receipt differs')
        raw = _read(self.context)
        raw['observations'][-1]['role'] = 'frozen_query'
        role = _json_new(probes / 'query-context.json', raw)
        query = _json_new(probes / 'query-authority.json', dict(trusted, context_registration=role))
        self.deny('query-role', lambda: score(self.plan, **dict(self.trust, authority=query)), 'original ordered spatial roles')
        frozen = _read(self.release)
        for name, edits in (
            ('release-scope', {'scope': 'registered_development'}),
            ('release-counts', {'training_class_counts': [25, 26, 6]}),
            ('release-samples', {'sample_count': 13}),
            ('release-label-parent', {'labels': raw['observations'][1]['input']}),
            ('release-role', {'role': 'source_validation'}),
            ('release-authority', {'authority': dict(self.authority, sha256='1' * 64)}),
        ):
            pin = _json_new(probes / f'{name}.json', dict(frozen, **edits))
            plan = _json_new(probes / f'{name}-plan.json', dict(_read(self.plan), label_release=pin))
            self.deny(name, lambda: score(plan, **self.trust), 'release differs')
        plan = _json_new(probes / 'hash-only-plan.json', dict(_read(self.plan),
                         label_release=dict(_pin(self.release), sha256='9' * 64)))
        self.deny('arbitrary-release-hash', lambda: score(plan, **self.trust), 'Input file identity changed')
        plan = _json_new(probes / 'early-label-plan.json', dict(_read(self.plan),
                         runs=[dict(self.cases[0], replay=label_pin), *self.cases[1:]]))
        self.deny('label-as-replay', lambda: score(plan, **self.trust), 'cannot be used as pre-scoring')
        last = Path(self.cases[-1]['directory'])
        self._deny_missing_or_corrupt(last / 'completion.json', None, 'incomplete-run', 'before scoring')
        self._deny_missing_or_corrupt(last / 'frame-000004' / 'prediction.npz', b'corrupt',
                                      'corrupted-prediction', 'bytes changed')

    def _deny_missing_or_corrupt(self, path, extra, name, error):
        raw = path.read_bytes()
        try:
            if extra is None:
                path.unlink()
            else:
                path.write_bytes(raw + extra)
            self.deny(name, lambda: score(self.plan, **self.trust), error)
        finally:
            path.write_bytes(raw)

    def independent_audit(self, scores):
        labels = np.zeros(24, dtype=np.int64)
        labels[self.indices] = self.targets
        full = labels.reshape(4, 6)
        truth = {'A': full[:, :3].ravel(), 'B': full[:, 3:].ravel()}
        results, online = {}, {}
        for item in self.cases:
            reg = _read(item['registration'])
            mode, order = reg['settings']['mode'], reg['order']
            sequence = list(FRAMES)
            if order == 'reverse':
                sequence.reverse()
            if order == 'seed64':
                sequence = [sequence[i] for i in np.random.default_rng(64).permutation(4)]
            name = f'{mode}/continual/{order}'
            packets = []
            for i, (frame_id, window) in enumerate(sequence, start=1):
                root = Path(item['directory']) / f'frame-{i:06d}'
                event = _read(root / 'event.json')
                assert event['frame']['frame_id'] == frame_id and event['frame']['observation'] == window
                assert event['counts'] == frame_budget(mode, 1)
                packets.append((truth[window], _arrays(root / 'prediction.npz')))
            online[name] = _independent_metrics(packets)
            _check_metrics(scores['online_pre_update'][name]['metrics'], online[name])
            replay = _read(item['replay'])
            assert replay['predictions'] == 4 and replay['optimizer_updates'] == replay['statistics_updates'] == 0
            anchor = {}
            for stage in ('initial', 'final'):
                views = []
                for row, expected in zip(replay['rows'], ('A', 'B'), strict=True):
                    assert row['frame']['observation'] == expected
                    path = Path(item['replay']['path']).parent / row[stage]['name']
                    views.append((truth[expected], _arrays(path)))
                anchor[stage] = _independent_metrics(views)
                _check_metrics(scores['zero_update_anchor_replay'][name][stage], anchor[stage])
            _check_recall(scores['zero_update_anchor_replay'][name]['final_minus_initial'],
                          anchor['final'], anchor['initial'])
            results[name] = dict(online=online[name], anchors=anchor, order=[frame_id for frame_id, _ in sequence])
        for name, values in online.items():
            mode, reset, order = name.split('/')
            row = scores['online_pre_update'][name]
            _check_recall(row['versus_frozen'], values, online[f'frozen/{reset}/{order}'])
            _check_recall(row['versus_original_order'], values, online[f'{mode}/{reset}/original'])
        return results


def _arrays(path):
    with np.load(path, allow_pickle=False) as packet:
        return {head: packet[head] for head in HEADS}


def _independent_metrics(packets):
    """Calculate each class explicitly using the fixture IDs, without production metric helpers."""
    results = {}
    for head in HEADS:
        matrix = [[0, 0, 0] for _ in CLASSES]
        losses = []
        for truth, all_heads in packets:
            for target, probabilities in zip(truth, all_heads[head], strict=True):
                if target:
                    i = CLASSES.index(target)
                    j = int(np.argmax(probabilities))
                    matrix[i][j] += 1
                    losses.append(-math.log(max(1e-8, min(float(probabilities[i]), 1.))))
        support = [sum(row) for row in matrix]
        predicted = [sum(row[j] for row in matrix) for j in range(len(CLASSES))]
        recall = [matrix[i][i] / count if count else None for i, count in enumerate(support)]
        f1 = [2 * matrix[i][i] / (count + predicted[i]) for i, count in enumerate(support) if count]
        n = sum(support)
        oa = sum(matrix[i][i] for i in range(len(CLASSES))) / n
        chance = sum(a * b for a, b in zip(support, predicted, strict=True)) / n**2
        results[head] = dict(matrix=matrix, support=support, recall=recall, count=n, oa=oa,
                             aa=sum(r for r in recall if r is not None) / len(f1),
                             macro_f1=sum(f1) / len(f1), kappa=(oa - chance) / (1 - chance),
                             nll=math.fsum(losses) / n)
    return results


def _check_metrics(actual, expected):
    for head, metrics in expected.items():
        got = actual[head]
        assert got['confusion_matrix'] == metrics['matrix'] and got['count'] == metrics['count']
        assert [item['support'] for item in got['per_class']] == metrics['support']
        assert [item['recall'] for item in got['per_class']] == metrics['recall']
        assert got['missing_class_ids'] == [19] and got['aa_full'] is None and got['macro_f1_full'] is None
        for key in ('oa', 'aa', 'kappa', 'macro_f1', 'nll'):
            assert math.isclose(got[key], metrics[key], rel_tol=0, abs_tol=1e-12), (head, key, got[key], metrics[key])


def _check_recall(actual, current, baseline):
    for head, now in current.items():
        previous = baseline[head]
        assert previous['support'] == now['support']
        deltas, gates = {}, {}
        for i, n in enumerate(now['support']):
            diff = now['matrix'][i][i] - previous['matrix'][i][i]
            deltas[str(CLASSES[i])] = diff / n if n else None
            gates[str(CLASSES[i])] = 100 * diff >= -2 * n if n else None
        assert actual[head]['recall'] == deltas
        assert actual[head]['recall_protected_at_2pp'] == gates
        assert actual[head]['small_class_protected_at_2pp'] == {k: gates[k] for k in ('7', '19')}
        assert actual[head]['worst_recall_drop'] == max(0., -min(v for v in deltas.values() if v is not None))


class KnownDevelopmentAdmissionTests(unittest.TestCase):
    def test_cli_pipeline_releases_only_frozen_development_and_matches_independent_metrics(self):
        configured = os.environ.get('WP09_TRUST_EVIDENCE_DIR')
        if configured:
            root = Path(configured)
            root.mkdir(parents=True, exist_ok=False)
            self._run(root)
        else:
            with TemporaryDirectory() as name:
                self._run(Path(name))

    def _run(self, root):
        pipeline = Pipeline(root)
        pipeline.train_and_admit()
        actual = pipeline.run_streams()
        report = pipeline.offline()
        checked = pipeline.independent_audit(report)
        pipeline.failures()
        _json_new(root / 'independent-metrics.json', checked)
        _json_new(root / 'acceptance.json', dict(commit=pipeline.sha, source=pipeline.source, admission=pipeline.admission,
            authority=pipeline.authority, release=_pin(pipeline.release), score=_pin(root / 'scores.json'),
            scope='synthetic', source_optimizer_updates=2, online=actual, preparation_known_pixels=120,
            fresh_admission_forwards=2, anchor_forwards=36, complete_reentries=9, independent_head_metrics=81,
            target_label_read_stages=['score'], target_label_payload_reads=1, real_source_updates=0, gt=0, query=0,
            failure_probes=pipeline.denials, commands=pipeline.commands))

    def test_paired_gate_uses_exact_count_and_retains_unsupported_classes(self):
        def metrics(counts):
            return {head: dict(class_ids=CLASSES, confusion_matrix=counts,
                per_class=[dict(class_id=c, support=sum(row)) for c, row in zip(CLASSES, counts, strict=True)],
                oa=0., aa=0., kappa=0., macro_f1=0., nll=0.) for head in HEADS}

        control = metrics([[100, 0, 0], [0, 100, 0], [0, 0, 0]])
        candidate = metrics([[98, 2, 0], [3, 97, 0], [0, 0, 0]])
        compared = _paired(candidate, control, [7, 19])
        for value in compared.values():
            self.assertEqual(value['recall_protected_at_2pp'], {'2': True, '7': False, '19': None})
            self.assertEqual(value['small_class_protected_at_2pp'], {'7': False, '19': None})
            self.assertEqual(value['worst_recall_drop'], .03)


if __name__ == '__main__':
    unittest.main()

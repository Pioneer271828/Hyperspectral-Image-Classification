"""Commit prediction, update and recovery state once per registered frame."""
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch

from evaluate_tta import HEADS, _Session, _fingerprint
from model.research_config import canonical_json, digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash
from transfer.online_tta_stream import StreamRegistration, fields_are, frame_budget, parse_json, read_pin


def runtime_identity():
    cpu = Path('/proc/cpuinfo').read_text().split('\n\n')[0].splitlines()
    return dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__),
                threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                mkldnn=torch.backends.mkldnn.enabled,
                cpu=digest([platform.machine(), *[row for row in cpu if row.startswith(('model name', 'flags'))]]))


def _sync_dir(path):
    descriptor = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path, writer):
    with path.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path, value):
    _write(path, lambda stream: stream.write((canonical_json(value) + '\n').encode()))


def _journal(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.journal-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((canonical_json(value) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    _sync_dir(path.parent)


def _load(path):
    if path.is_symlink():
        raise ValueError('Committed receipt may not be a symbolic link')
    return parse_json(path.read_bytes())


def _rng():
    name, keys, index, has_gauss, cached = np.random.get_state()
    return dict(torch=torch.random.get_rng_state(), python=random.getstate(),
                numpy=(name, keys.tolist(), index, has_gauss, cached))


def _restore_rng(state):
    torch.random.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name, keys, index, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), index, has_gauss, cached))


@contextmanager
def isolated_rng(seed):
    original = _rng()
    try:
        torch.random.default_generator.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        _restore_rng(original)


def new_session(reg, source):
    settings = dict(reg.settings)
    del settings['reset']
    session = _Session(source, **settings)
    if any(module.running_mean is None or module.running_var is None or module.num_batches_tracked is None
           for module in session.statistics):
        raise ValueError('Statistics mode requires real running buffers')
    return session


def predict(session, inputs, class_count):
    before = session.state()
    with torch.inference_mode():
        probabilities = dict(zip(HEADS, (v.detach().cpu().numpy().copy() for v in session.forward(inputs)),
                                 strict=True))
    validate_predictions(probabilities, inputs[0].shape[0] * inputs[0].shape[1], class_count)
    if session.state() != before:
        raise ValueError('Pre-update prediction modified state')
    return probabilities


def validate_predictions(probabilities, pixels, classes):
    fields_are(probabilities, 'fused graph cnn')
    for value in probabilities.values():
        if (value.shape != (pixels, classes) or value.dtype != np.float32 or not np.isfinite(value).all()
                or (value < 0).any() or (value > 1).any()
                or not np.allclose(value.sum(axis=-1), 1, atol=2e-6, rtol=0)):
            raise ValueError('Invalid three-head probability packet')


def read_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(HEADS) or set(archive.files) != set(HEADS):
            raise ValueError('Invalid prediction archive fields')
        return dict(archive)


def _parameters(session):
    if session.adapter is not None:
        return {'adapter.' + k: v for k, v in session.adapter.named_parameters()}
    return {k: v for k, v in session.model.named_parameters() if v.requires_grad}


def _checkpoint(session, cursor, frame, registration):
    return dict(registration=registration, cursor=cursor, frame=frame, model=session.model.state_dict(),
                adapter=None if session.adapter is None else session.adapter.state_dict(),
                optimizer=None if session.optimizer is None else session.optimizer.state_dict(), rng=_rng())


def _rng_hash(state):
    return digest(dict(torch=tensor_hash(state['torch']), python=state['python'], numpy=state['numpy']))


def load_checkpoint(path, checksum):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError('Committed checkpoint hash differs')
    return torch.load(BytesIO(raw), map_location='cpu', weights_only=True)


def restore(session, checkpoint):
    session.model.load_state_dict(checkpoint['model'], strict=True)
    if (session.adapter is None) != (checkpoint['adapter'] is None):
        raise ValueError('Committed adapter mode differs')
    if session.adapter is not None:
        session.adapter.load_state_dict(checkpoint['adapter'], strict=True)
    expected = None if session.optimizer is None else session.optimizer.state_dict()
    if expected != checkpoint['optimizer']:
        raise ValueError('Committed stateless SGD settings differ')
    if session.optimizer is not None:
        session.optimizer.load_state_dict(checkpoint['optimizer'])
    _restore_rng(checkpoint['rng'])


def _is_reset(reg, index):
    return index == 0 or (reg.settings['reset'] == 'episodic'
                          and reg.frames[index]['episode'] != reg.frames[index - 1]['episode'])


def _binding(reg):
    return dict(version=1, registration=reg.pin['sha256'], population=reg.population, budget=reg.value['budget'])


def _summary(reg, events, tip):
    return dict(status='complete', binding=_binding(reg), cursor=len(events), tip=tip,
                actual={key: sum(row['counts'][key] for row in events) for key in reg.value['budget']},
                target_labels_accessed=False)


def _progress(cursor, tip, in_flight=False):
    return dict(cursor=cursor, tip=tip, in_flight=in_flight)


def committed_history(reg, root, *, complete=False):
    """Validate the full chain and all immutable payload hashes before use."""
    if _load(root / 'binding.json') != _binding(reg):
        raise ValueError('Output root belongs to a different registration')
    if any(root.glob('.pending-*')):
        raise RuntimeError('Uncommitted frame attempt retained; this output cannot be resumed')
    entries = sorted(root.glob('frame-*'))
    if len(entries) > len(reg.frames):
        raise ValueError('Committed cursor exceeds registration')
    events, tip = [], None
    for index, folder in enumerate(entries):
        if folder.name != f'frame-{index + 1:06d}' or folder.is_symlink():
            raise ValueError('Committed history has a gap, alias, or extra frame')
        event_path = folder / 'event.json'
        event = _load(event_path)
        fields_are(event, 'registration cursor previous frame observation_sha256 reset counts '
                          'before after changed entropy predictions checkpoint optimizer rng gradients')
        frame = reg.frames[index]
        if (event['registration'] != reg.pin['sha256'] or event['cursor'] != index + 1
                or event['previous'] != tip or event['frame'] != frame
                or event['observation_sha256'] != reg.observation_identity(frame)
                or event['reset'] is not _is_reset(reg, index)
                or event['counts'] != frame_budget(reg.settings['mode'], 1)
                or (index and not event['reset'] and event['before'] != events[-1]['after'])):
            raise ValueError('Committed frame identity, transition, or budget differs')
        journal = _load(folder / 'attempt.json')
        if journal.get('phase') != 'ready' or journal.get('completed') != event['counts']:
            raise ValueError('Committed frame lacks a completed attempt journal')
        for name, key in (('prediction.npz', 'predictions'), ('checkpoint.pt', 'checkpoint')):
            path = folder / name
            expected = event[key]['file'] if key == 'predictions' else event[key]
            if path.is_symlink() or hash_file(path) != expected:
                raise ValueError('Committed prediction or checkpoint bytes changed')
        gradients = folder / 'gradients.pt'
        if (gradients.is_symlink() or event['gradients'] is None and gradients.exists()
                or event['gradients'] is not None and hash_file(gradients) != event['gradients']):
            raise ValueError('Committed gradient evidence changed')
        events.append(event)
        tip = hash_file(event_path)
    progress = _load(root / 'progress.json')
    current = _progress(len(events), tip)
    committed_after_interrupt = bool(events) and progress == _progress(
        len(events) - 1, events[-1]['previous'], in_flight=True)
    if progress != current and (complete or not committed_after_interrupt):
        raise ValueError('Progress marker disagrees with committed frames or records an interrupted attempt')
    completion = root / 'completion.json'
    if complete and not completion.is_file():
        raise ValueError('All predictions must be committed before scoring')
    if completion.exists() and (len(events) != len(reg.frames) or _load(completion) != _summary(reg, events, tip)):
        raise ValueError('Completion record differs from committed history')
    return events, tip


def _frame(reg, source, session, root, index, previous):
    frame = reg.frames[index]
    cursor = index + 1
    attempt = root / f'.pending-{cursor:06d}'
    attempt.mkdir()
    _sync_dir(root)
    completed = frame_budget(reg.settings['mode'], 0)
    started = dict(completed)
    begin = perf_counter()

    def journal(phase):
        _journal(attempt / 'attempt.json', dict(cursor=cursor, frame=frame, phase=phase, started=started,
                                               completed=completed, seconds=perf_counter() - begin))

    journal('observation_started')
    inputs = reg.observation(frame['observation'])
    before = session.state()
    started['predictions'] = 1
    journal('prediction_started')
    probabilities = predict(session, inputs, len(reg.value['class_ids']))
    completed['predictions'] = 1
    prediction_path = attempt / 'prediction.npz'
    _write(prediction_path, lambda stream: np.savez(stream, **probabilities))
    journal('prediction_frozen')
    fingerprints = reg.fingerprints(probabilities)
    baseline = reg.provenance['baseline_hashes'].get(frame['observation'])
    reset = _is_reset(reg, index)
    if baseline is not None and (reset or reg.settings['mode'] == 'frozen') and fingerprints != baseline:
        raise ValueError('Initial/frozen predictions differ from authenticated baseline')
    budget = frame_budget(reg.settings['mode'], 1)
    if budget['optimizer_updates'] or budget['statistics_updates']:
        started.update(budget)
        journal('adaptation_started')
        entropy = session.adapt(inputs)
        completed.update(budget)
    else:
        entropy = None
    journal('adaptation_finished')
    after = session.state()
    changed = sorted(key for key in before if before[key] != after[key])
    if (set(changed) - session.allowed or after.keys() != before.keys()
            or _fingerprint(source) != reg.initial_source):
        raise ValueError('TTA changed state outside its allowlist')
    gradients_hash = None
    if session.optimizer is not None:
        gradients = {key: param.grad for key, param in _parameters(session).items()}
        if not any(gradient is not None for gradient in gradients.values()):
            raise ValueError('Missing SGD gradient evidence')
        grad_path = attempt / 'gradients.pt'
        _write(grad_path, lambda stream: torch.save(gradients, stream))
        gradients_hash = hash_file(grad_path)
    state = _checkpoint(session, cursor, frame, reg.pin['sha256'])
    if any(not torch.isfinite(value).all() for mapping in (state['model'], state['adapter'] or {})
           for value in mapping.values()):
        raise ValueError('Nonfinite committed TTA state')
    state_path = attempt / 'checkpoint.pt'
    _write(state_path, lambda stream: torch.save(state, stream))
    event_path = attempt / 'event.json'
    event = dict(registration=reg.pin['sha256'], cursor=cursor, previous=previous, frame=frame,
                 observation_sha256=reg.observation_identity(frame), reset=reset, counts=completed,
                 before=digest(before), after=digest(after), changed=changed, entropy=entropy,
                 predictions=dict(file=hash_file(prediction_path), heads=fingerprints),
                 checkpoint=hash_file(state_path), optimizer=digest(state['optimizer']), rng=_rng_hash(state['rng']),
                 gradients=gradients_hash)
    _json(event_path, event)
    journal('ready')
    _sync_dir(attempt)
    attempt.rename(root / f'frame-{cursor:06d}')
    _sync_dir(root)
    return event, hash_file(root / f'frame-{cursor:06d}' / 'event.json')


def execute(registration, out, *, to_frame=None, admission=None):
    """Resume only from fully committed frames. A pending attempt remains terminal."""
    import fcntl

    snapshot = code_snapshot()
    require_frozen_code(snapshot)
    reg = StreamRegistration(registration, admission=admission)
    if (reg.value['commit'] != snapshot['commit'] or reg.value['source_hash'] != snapshot['source_hash']
            or reg.value['runtime'] != runtime_identity()):
        raise ValueError('Registered code or CPU runtime differs')
    stop = len(reg.frames) if to_frame is None else to_frame
    if type(stop) is not int or not 1 <= stop <= len(reg.frames):
        raise ValueError('Requested cursor is outside the registered stream')
    root = external_directory(out)
    root.mkdir(parents=True, exist_ok=True)
    _sync_dir(root.parent)
    with (root / '.writer.lock').open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / 'binding.json').exists():
            if {path.name for path in root.iterdir()} != {'.writer.lock'}:
                raise ValueError('New TTA output must be empty')
            _json(root / 'progress.json', _progress(0, None))
            _json(root / 'binding.json', _binding(reg))
            _sync_dir(root)
        events, tip = committed_history(reg, root)
        start = len(events)
        if _load(root / 'progress.json') != _progress(start, tip):
            _journal(root / 'progress.json', _progress(start, tip))
        if stop < start:
            raise ValueError('Requested cursor is behind committed progress')
        for key in {frame['observation'] for frame in reg.frames[:start]}:
            read_pin(reg.value['observations'][key])
        if start < stop:
            source = reg.source_model()
            reg.initial_source = _fingerprint(source)
            with isolated_rng(reg.settings['seed']):
                session = new_session(reg, source)
                if start and not _is_reset(reg, start):
                    previous = events[-1]
                    path = root / f'frame-{start:06d}' / 'checkpoint.pt'
                    state = load_checkpoint(path, previous['checkpoint'])
                    if (state['registration'] != reg.pin['sha256'] or state['cursor'] != start
                            or state['frame'] != reg.frames[start - 1] or _rng_hash(state['rng']) != previous['rng']):
                        raise ValueError('Checkpoint does not own the committed cursor')
                    restore(session, state)
                    if digest(session.state()) != previous['after'] or digest(state['optimizer']) != previous['optimizer']:
                        raise ValueError('Restored logical state differs')
                for index in range(start, stop):
                    _journal(root / 'progress.json', _progress(index, tip, in_flight=True))
                    if _is_reset(reg, index):
                        with isolated_rng(reg.settings['seed']):
                            session = new_session(reg, source)
                            event, tip = _frame(reg, source, session, root, index, tip)
                            next_rng = _rng()
                        _restore_rng(next_rng)
                    else:
                        event, tip = _frame(reg, source, session, root, index, tip)
                    events.append(event)
                    _journal(root / 'progress.json', _progress(index + 1, tip))
        StreamRegistration(registration, admission=admission)
        for key in {frame['observation'] for frame in reg.frames[start:stop]}:
            read_pin(reg.value['observations'][key])
        require_frozen_code(snapshot)
        if stop == len(reg.frames) and not (root / 'completion.json').exists():
            _journal(root / 'completion.json', _summary(reg, events, tip))
        counts = {key: sum(row['counts'][key] for row in events[start:]) for key in reg.value['budget']}
        return dict(cursor=stop, added=stop - start, actual=counts, complete=stop == len(reg.frames), tip=tip)

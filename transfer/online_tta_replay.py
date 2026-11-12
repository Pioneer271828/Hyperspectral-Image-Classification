"""Replay fixed early observation anchors from initial and final states, with zero updates."""
import argparse
from pathlib import Path

import numpy as np

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from transfer.online_tta_run import (committed_history, isolated_rng, load_checkpoint, new_session, predict,
                                     restore, runtime_identity, _json, _sync_dir, _write)
from transfer.online_tta_stream import StreamRegistration


def replay(registration, run, out):
    reg = StreamRegistration(registration)
    code = code_snapshot()
    require_frozen_code(code)
    if ((code['commit'], code['source_hash'], runtime_identity())
            != (reg.value['commit'], reg.value['source_hash'], reg.value['runtime'])):
        raise ValueError('Diagnostic code or runtime differs from the completed run')
    root = external_directory(run)
    events, tip = committed_history(reg, root, complete=True)
    out = external_directory(out)
    out.mkdir(parents=True, exist_ok=False)
    source = reg.source_model()
    # Choose the first frame for each registered observation in the original stream.
    anchors = {}
    for frame in reg.value['frames']:
        anchors.setdefault(frame['observation'], frame)
    cursor = len(events)
    final = load_checkpoint(root / f'frame-{cursor:06d}' / 'checkpoint.pt', events[-1]['checkpoint'])
    rows = []
    with isolated_rng(reg.settings['seed']):
        initial_session = new_session(reg, source)
        final_session = new_session(reg, source)
        restore(final_session, final)
        if digest(final_session.state()) != events[-1]['after']:
            raise ValueError('Replay final state differs from committed state')
        for index, (observation, frame) in enumerate(anchors.items(), start=1):
            packets = {}
            inputs = reg.observation(observation)
            for name, session in (('initial', initial_session), ('final', final_session)):
                probabilities = predict(session, inputs, len(reg.value['class_ids']))
                filename = f'{name}-{index:06d}.npz'
                path = out / filename
                _write(path, lambda stream: np.savez(stream, **probabilities))
                packets[name] = dict(name=filename, sha256=hash_file(path), heads=reg.fingerprints(probabilities))
            rows.append(dict(frame=frame, observation_sha256=reg.observation_identity(frame), **packets))
    require_frozen_code(code)
    _json(out / 'replay.json', dict(version=1, kind='initial_final_anchor_replay', registration=reg.pin['sha256'],
                                  tip=tip, checkpoint=events[-1]['checkpoint'], rows=rows,
                                  predictions=2 * len(rows), optimizer_updates=0, statistics_updates=0))
    _sync_dir(out)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    replay(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256), args.run, args.out)


if __name__ == '__main__':
    main()

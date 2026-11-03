"""BN-ALIGN2: replay all frozen BASE97C states; no optimization or test scoring."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from joint_baseline import NAMES
from joint_calibrated_selection import first_maximum, load_selected, observations
from joint_training import (_completed, _tree_fingerprint, import_preparation, load_plan,
                            schedule_for, validation_epochs, validation_snapshot)
from loadData.dataset_registry import file_hash
from model.bn_calibration import CONTEXT, POLICY, calibrate_denoise, state_digest
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json

TRAIN_COMMIT = '8daa65b116ba585b33a84e3af93f3741716b710a'
ALLOWED_TRAINER_CHANGES = {'joint_training.py', 'scripts/train_joint_baseline.py'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_metrics(actual, expected):
    if not np.isclose(actual['val_loss'], expected['val_loss'], rtol=1e-6, atol=1e-6):
        raise ValueError('Historical validation NLL changed')
    if actual['macro_oa'] != expected['macro_oa']:
        raise ValueError('Historical macro OA changed')
    for name in NAMES:
        for phase in ('train', 'val'):
            for head in ('fused', 'cnn', 'graph'):
                if actual['metrics'][name][phase][head]['confusion'] != expected['metrics'][name][phase][head]['confusion']:
                    raise ValueError('Historical supervised confusion changed')


def run(a):
    source = code_snapshot()
    if sys.platform != 'linux' or source['dirty'] or source['commit'] != a.expected_commit:
        raise RuntimeError('Remote clean checkout at published expected SHA required')
    out = external_directory(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        batch = Path(a.campaign_dir)
        launch = load(batch/'launch.json')
        if launch['code']['commit'] != TRAIN_COMMIT or load(batch/'status.json')['status'] != 'training_completed':
            raise ValueError('Accepted completed BASE97C campaign required')
        for name, value in launch['code']['files'].items():
            if name not in ALLOWED_TRAINER_CHANGES and source['files'].get(name) != value:
                raise ValueError('Frozen producer implementation changed: '+name)
        _, items, provenance = import_preparation(a.prepared_dir, a.data_root,
            load_plan('configs/experiments/base97_training.json'), source)
        inputs = {str(f): file_hash(f) for f in Path(a.prepared_dir).iterdir() if f.is_file()}
        for value in load(Path(a.prepared_dir)/'manifest.json')['datasets'].values():
            for role in ('image', 'label'):
                f = Path(a.data_root)/value['raw'][role+'_path']
                inputs[str(f)] = file_hash(f)
        originals = {}; results = {}
        for seed in (64, 512, 2048):
            directory = batch/('seed-'+str(seed)); result = load(directory/'result.json')
            settings = result['settings']
            if (result['start_code']['commit'] != TRAIN_COMMIT or result['status'] != 'completed'
                or result['optimizer_steps'] != 1503 or settings['model_seed'] != seed
                or settings['schedule_seed'] != seed or result['identity']['prepared'] != provenance
                or settings['profile'] != 'full' or settings['protocol'] != 'base97_honghu_joint_training_v1'):
                raise ValueError('Wrong frozen source run/settings')
            _completed(directory, result['identity'], settings, schedule_for(settings))
            originals[str(seed)] = {f.name: file_hash(f) for f in directory.iterdir() if f.is_file()}
            results[seed] = result
        write_json(out/'plan.json', {'code': source, 'context': CONTEXT, 'policy': POLICY,
            'optimizer_updates': 0, 'test_scoring': False, 'source_files': originals, 'input_files': inputs,
            'comparisons': ['raw_selected', 'calibrated_original_selected', 'calibrated_reselected'],
            'selection': '51 chronological events/seed; strict greater macro fused validation OA; first tie',
            'trajectory': 'fixed historical weights and raw-NLL scheduler; not a new calibrated training run'})
        items = {n: {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in i.items()} for n, i in items.items()}
        obs = observations(items); rows = []
        for seed in (64, 512, 2048):
            directory = batch/('seed-'+str(seed)); result = results[seed]
            epochs = validation_epochs(result['settings'])
            history = load(directory/'history.json')
            if [v['epoch'] for v in history] != epochs or len(history) != 51:
                raise ValueError('Expected all51 historical validation events')
            expected_paths = [directory/f'state-{(e+1)*3:06d}.pt' for e in epochs]
            if set(directory.glob('state-*.pt')) != set(expected_paths):
                raise ValueError('Unexpected missing/extra source snapshot')
            model = load_model_checkpoint(directory/'primary.pt', device=a.device)
            folder = out/('seed-'+str(seed)); folder.mkdir()
            raw_events = []; calibrated_events = []; saved_probabilities = []
            for epoch, path, expected in zip(epochs, expected_paths, history):
                state = torch.load(path, map_location='cpu', weights_only=True)
                cursor = (epoch+1)*3
                if (state['identity'] != result['identity'] or state['cursor'] != cursor
                    or [v['dataset'] for v in state['trace']] != schedule_for(result['settings'])[:cursor]
                    or state['history'] != history[:len(raw_events)+1]):
                    raise ValueError('Snapshot identity/cursor/history mismatch')
                for key in ('model', 'optimizer', 'scheduler'):
                    if _tree_fingerprint(state[key]) != state['state_fingerprints'][key]:
                        raise ValueError('Snapshot state fingerprint mismatch: '+key)
                # This is the raw model at THIS cursor, not its historical best.
                model.load_state_dict(state['model'], strict=True); model.eval()
                before = state_digest(model)
                raw, raw_prob = validation_snapshot(model, items)
                compare_metrics(raw, expected)
                candidate, receipt = calibrate_denoise(model, obs, mode=POLICY, context=CONTEXT,
                                                     source_checkpoint_sha256=file_hash(path))
                calibrated, cal_prob = validation_snapshot(candidate, items)
                if state_digest(model) != before or state_digest(candidate) != receipt['state_sha256']:
                    raise ValueError('Read-only replay modified model state')
                raw_events.append({'step': cursor, 'epoch': epoch, **raw})
                event = {'step': cursor, 'epoch': epoch, 'source_file': path.name,
                         'source_sha256': file_hash(path), **calibrated}
                calibrated_events.append(event)
                saved_probabilities.append({'step': cursor, 'raw': raw_prob, 'calibrated': cal_prob})
                if first_maximum(calibrated_events) is event:
                    candidate.training_config = copy.deepcopy(model.training_config)
                    candidate.training_config['bn_calibration'] = receipt
                    candidate.training_config['joint_runtime']['selected_step'] = cursor
                    candidate.training_config['checkpoint_optimizer_step'] = cursor
                    candidate.training_config['calibrated_selection_replay'] = {
                        'source_training_commit': TRAIN_COMMIT, 'replay_code': source,
                        'source_file': path.name, 'source_sha256': file_hash(path),
                        'selected_step': cursor, 'optimizer_updates': 0,
                        'trajectory': 'historical_raw_scheduler', 'test_scoring': False}
                    save_model_checkpoint(candidate, folder/'selected-calibrated.pt')
                print('REPLAY', seed, epoch, cursor, raw['macro_oa'], calibrated['macro_oa'], flush=True)
            original = first_maximum(raw_events); selected = first_maximum(calibrated_events)
            if original['step'] != result['selected_step']:
                raise ValueError('Historical selection replay did not reproduce primary')
            restored = load_selected(folder/'selected-calibrated.pt', items,
                                      directory/selected['source_file'], device=a.device)
            restored_metrics, restored_prob = validation_snapshot(restored, items)
            compare_metrics(restored_metrics, selected)
            selected_prob = next(v['calibrated'] for v in saved_probabilities if v['step'] == selected['step'])
            maximum = 0.
            for name in NAMES:
                for phase in ('train', 'val'):
                    for head in ('fused', 'cnn', 'graph'):
                        old, new = selected_prob[name][phase][head], restored_prob[name][phase][head]
                        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                        if not torch.equal(old.argmax(1), new.argmax(1)):
                            raise ValueError('Selected reloaded predictions changed')
                        maximum = max(maximum, float((old-new).abs().max()))
            calibrated_original = next(v for v in calibrated_events if v['step'] == original['step'])
            torch.save(saved_probabilities, folder/'event-probabilities.pt')
            write_json(folder/'history.json', {'raw': raw_events, 'calibrated': calibrated_events})
            row = {'seed': seed, 'raw_selected': original, 'calibrated_original_selected': calibrated_original,
                   'calibrated_reselected': selected, 'events': len(raw_events), 'reload_max_abs_error': maximum}
            write_json(folder/'review.json', row); rows.append(row)
        for seed, files in originals.items():
            directory = batch/('seed-'+seed)
            if files != {f.name: file_hash(f) for f in directory.iterdir() if f.is_file()}:
                raise ValueError('Historical training files changed')
        for path, value in inputs.items():
            if file_hash(path) != value: raise ValueError('Raw/prepared input changed')
        require_frozen_code(source)
        summary = {}
        for key in ('raw_selected', 'calibrated_original_selected', 'calibrated_reselected'):
            values = [r[key]['macro_oa']*100 for r in rows]
            summary[key] = {'values_percent': values, 'mean_percent': float(np.mean(values)),
                            'sample_sd_percent': float(np.std(values, ddof=1)),
                            'selected_steps': [r[key]['step'] for r in rows]}
        artifacts = {str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()}
        write_json(out/'result.json', {'status': 'passed', 'code': source, 'summary': summary,
            'replayed_states': sum(r['events'] for r in rows), 'optimizer_updates': 0, 'test_scoring': False,
            'historical_files_unchanged': sum(map(len, originals.values())), 'inputs_unchanged': len(inputs),
            'elapsed_seconds': time.monotonic()-started, 'artifacts': artifacts,
            'scope': 'reused validation; fixed historical raw-scheduler trajectory; no prospective accuracy claim'})
        print(json.dumps({'status': 'passed', 'summary': summary}), flush=True)
    except Exception as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'error': str(error), 'code': source})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit', 'campaign-dir', 'prepared-dir', 'data-root', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(p.parse_args())


if __name__ == '__main__': main()

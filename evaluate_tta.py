"""Run the E4b predict-then-adapt protocol on an ordered, label-free stream."""
import argparse
import copy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model.checkpoint_io import load_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.variable_channel_adapter import VariableChannelAdapter
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash, write_json


MODES = ('frozen', 'statistics', 'affine', 'adapter')
HEADS = ('fused', 'graph', 'cnn')
ARRAYS = {'spectral', 'coordinates', 'wavelengths_nm'}


def _fingerprint(module, prefix=''):
    return {prefix + key: tensor_hash(value) for key, value in module.state_dict().items()}


def _observation(frame, device):
    if not isinstance(frame, dict) or set(frame) != ARRAYS:
        raise ValueError('TTA accepts only spectral, coordinates and wavelengths_nm')
    if any(not isinstance(array, np.ndarray) or array.dtype != np.float32
           or not np.isfinite(array).all() for array in frame.values()):
        raise ValueError('TTA observation requires finite FP32 arrays')
    spectral, coords, waves = (frame[key] for key in ('spectral', 'coordinates', 'wavelengths_nm'))
    if (spectral.ndim != 3 or min(spectral.shape[:2]) < 2 or not spectral.shape[-1]
            or coords.shape != (*spectral.shape[:2], 2) or (coords < 0).any() or (coords > 1).any()
            or waves.shape != (spectral.shape[-1],) or (waves <= 0).any() or np.unique(waves).size != waves.size):
        raise ValueError('Invalid HWC observation geometry or physical band centers')
    return torch.tensor(np.concatenate((spectral, coords), axis=-1), device=device), torch.tensor(waves, device=device)


class _Session:
    def __init__(self, source, task, mode, lr, rank, seed, scale):
        self.model = copy.deepcopy(source).eval().requires_grad_(False)
        self.task, self.mode, self.scale = task, mode, scale
        self.adapter = None
        self.optimizer = None
        self.allowed = set()
        self.statistics = []
        for module in self.model.modules():
            if isinstance(module, VariableChannelAdapter):
                module.cache_attention = False
                module.clear_attention_cache()
        active = [(name, module) for name, module in self.model.named_modules()
                  if name.startswith('backbone.') and
                  ('.norms.' not in name or name.rsplit('.norms.', 1)[1] == task)]
        params = []
        if mode == 'statistics':
            for name, module in active:
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and module.track_running_stats:
                    self.statistics.append(module)
                    self.allowed.update(f'{name}.{key}' for key in ('running_mean', 'running_var', 'num_batches_tracked'))
        elif mode == 'affine':
            for name, module in active:
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    for key, parameter in module.named_parameters(recurse=False):
                        if key in ('weight', 'bias'):
                            self.allowed.add(f'{name}.{key}')
                            params.append(parameter.requires_grad_(True))
        elif mode == 'adapter':
            width = self.model.heads[task].in_features
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.adapter = nn.Sequential(nn.Linear(width, rank, bias=False), nn.ReLU(),
                                             nn.Linear(rank, width, bias=False))
                nn.init.zeros_(self.adapter[-1].weight)
            self.adapter.to(next(self.model.parameters()))
            self.allowed.update(_fingerprint(self.adapter, 'adapter.'))
            params.extend(self.adapter.parameters())
        if mode != 'frozen' and not self.allowed:
            raise ValueError('Selected TTA mode has no supported mutable state in this backbone')
        if params:
            self.optimizer = torch.optim.SGD(params, lr=lr)

    def state(self):
        result = _fingerprint(self.model)
        if self.adapter is not None:
            result.update(_fingerprint(self.adapter, 'adapter.'))
        return result

    def forward(self, inputs):
        x, waves = inputs
        kwargs = dict(wavelengths=waves, superpixel_scale=self.scale, wavelength_unit='nm')
        if self.adapter is None:
            return self.model(self.task, x, **kwargs)
        fused, graph, cnn = self.model.backbone.forward_features(x, dataset_name=self.task, **kwargs)
        return self.model.predict_features(self.task, fused + self.adapter(fused), graph, cnn)

    def adapt(self, inputs):
        if self.statistics:
            for module in self.statistics:
                module.train()
            try:
                with torch.no_grad():
                    self.forward(inputs)
            finally:
                self.model.eval()
        if self.optimizer is None:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        p = self.forward(inputs)[0]
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=-1).mean()
        if not torch.isfinite(entropy):
            raise ValueError('Nonfinite TTA entropy')
        entropy.backward()
        gradients = [p.grad for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError('Invalid TTA parameter gradients')
        self.optimizer.step()
        return float(entropy.detach())


def run_online(source, stream, *, task, mode, reset, scale, lr, rank, seed, on_prediction):
    """Invoke the prediction sink before adapting. Only observation tensors enter adaptation."""
    if (not isinstance(source, MultiDatasetVCASAHGCN) or any(m.training for m in source.modules())
            or task not in source.heads or source.backbone.graph_mode != 'learned'
            or source.backbone.coverage_modulation is not None):
        raise ValueError('TTA requires an eval-mode learned-graph checkpoint and registered head')
    if (mode not in MODES or reset not in ('episodic', 'continual') or type(scale) is not int or scale <= 0
            or type(rank) is not int or not 0 < rank < source.heads[task].in_features
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Invalid explicit TTA protocol settings')
    base_state = _fingerprint(source)
    device = next(source.parameters()).device
    previous = None
    seen, events = set(), []
    session = _Session(source, task, mode, lr, rank, seed, scale)
    for scene, frame_id, frame in stream:
        if (not isinstance(scene, str) or not scene or not isinstance(frame_id, str) or not frame_id
                or frame_id in seen):
            raise ValueError('TTA stream requires explicit scene and unique frame identity')
        seen.add(frame_id)
        reset_now = previous is None or (reset == 'episodic' and previous != scene)
        if reset_now and previous is not None:
            session = _Session(source, task, mode, lr, rank, seed, scale)
        previous = scene
        inputs = _observation(frame, device)
        before = session.state()
        with torch.inference_mode():
            probabilities = tuple(out.detach().cpu().numpy().copy() for out in session.forward(inputs))
        if not all(np.isfinite(p).all() for p in probabilities) or session.state() != before:
            raise ValueError('Pre-update prediction changed state or produced invalid probabilities')
        prediction = on_prediction(scene, frame_id, probabilities)
        entropy = session.adapt(inputs)
        after = session.state()
        changed = sorted(key for key in before if before[key] != after[key])
        if set(changed) - session.allowed:
            raise ValueError('TTA modified state outside the explicit allowlist')
        events.append(dict(scene=scene, frame_id=frame_id, reset=reset_now, prediction=prediction,
                           entropy=entropy, changed=changed, optimizer_updates=int(entropy is not None),
                           statistics_updates=int(mode == 'statistics')))
    if not events or _fingerprint(source) != base_state:
        raise ValueError('TTA requires a nonempty stream and unchanged source weights')
    return dict(mode=mode, reset=reset, task=task, scale=scale, lr=lr, rank=rank, seed=seed,
                allowed_state=sorted(session.allowed), events=events,
                optimizer_updates=sum(event['optimizer_updates'] for event in events))


def _pinned(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('TTA pinned input hash mismatch')
    return data


def run_registered(registration, registration_sha256, expected_commit, out):
    code = code_snapshot()
    if code['commit'] != expected_commit:
        raise ValueError('Unexpected TTA code identity')
    require_frozen_code(code)
    reg = json.loads(_pinned(registration, registration_sha256))
    if not isinstance(reg, dict) or set(reg) != set(
            'policy checkpoint checkpoint_sha256 sidecar_sha256 task mode reset scale lr rank seed stream'.split()
    ) or reg['policy'] != 'e4b_online_predict_then_adapt_cpu_v1':
        raise ValueError('Invalid TTA registration')
    entries = reg['stream']
    if not isinstance(entries, list) or not entries or any(
            not isinstance(entry, dict) or set(entry) != {'scene','frame_id','path','sha256'} for entry in entries):
        raise ValueError('TTA stream requires ordered, observation-only entries')
    checkpoint = Path(reg['checkpoint'])
    sidecar = Path(str(checkpoint) + '.json')
    if hash_file(checkpoint) != reg['checkpoint_sha256'] or hash_file(sidecar) != reg['sidecar_sha256']:
        raise ValueError('TTA source weights or sidecar changed')
    model = load_model_checkpoint(checkpoint, 'cpu')
    out = external_directory(out)
    out.mkdir(parents=True, exist_ok=False)

    def frames():
        for entry in entries:
            data = _pinned(entry['path'], entry['sha256'])
            with np.load(BytesIO(data), allow_pickle=False) as archive:
                if set(archive.files) != ARRAYS or len(archive.files) != len(ARRAYS):
                    raise ValueError('TTA observation archive contains unexpected fields')
                yield entry['scene'], entry['frame_id'], dict(archive)

    def freeze_prediction(scene, frame_id, probabilities):
        # Stream IDs are metadata, never filesystem path components.
        path = out / f'prediction-{freeze_prediction.count:06d}.npz'
        np.savez(path, **dict(zip(HEADS, probabilities)))
        freeze_prediction.count += 1
        return {'path': path.name, 'sha256': hash_file(path)}

    freeze_prediction.count = 1
    settings = {key: reg[key] for key in ('task','mode','reset','scale','lr','rank','seed')}
    result = run_online(model, frames(), on_prediction=freeze_prediction, **settings)
    if hash_file(checkpoint) != reg['checkpoint_sha256'] or hash_file(sidecar) != reg['sidecar_sha256']:
        raise ValueError('TTA source files changed during evaluation')
    require_frozen_code(code)
    result.update(status='completed_label_free_online_tta', commit=expected_commit,
                  registration_sha256=registration_sha256, checkpoint_sha256=reg['checkpoint_sha256'],
                  target_labels_accessed=False)
    write_json(out / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    run_registered(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

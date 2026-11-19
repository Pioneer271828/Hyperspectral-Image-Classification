"""Parameter-free center-spectral bypass for a registered native observation."""

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from model.research_config import digest
from transfer.native_scale_observation import forward

SCHEMA = "wp06_registered_center_spectral_v1"


@dataclass(frozen=True)
class CenterAxis:
    band_ids: tuple
    base_fingerprint: str
    bank_sha256: str
    width: int = 64

    @classmethod
    def from_bank(cls, bank):
        bank.verify_authority(bank.pin)
        bands = bank.release.bands
        return cls(bands.band_ids, bands.fingerprint, bank.pin["sha256"])

    @property
    def fingerprint(self):
        return digest([SCHEMA, self.band_ids, self.base_fingerprint, self.bank_sha256, self.width])

    def columns(self, retained):
        available = set(retained)
        selected = [position for position, key in enumerate(self.band_ids) if key in available]
        if not selected or tuple(self.band_ids[i] for i in selected) != tuple(retained):
            raise ValueError("Center bypass requires the exact order of registered retained IDs")
        return selected

    def projection(self, retained, device, dtype):
        # Fixed signed projection by original delivered positions. No missing measurement is invented.
        indices = self.columns(retained)
        rng = np.random.default_rng(6605)
        weights = rng.integers(0, 2, size=(len(self.band_ids), self.width), dtype=np.int8)
        weights = (2 * weights.astype(np.float32) - 1) / np.sqrt(len(self.band_ids))
        return torch.as_tensor(weights[indices], device=device, dtype=dtype)


@contextmanager
def fused_residual(head, residual):
    uses = []

    def add(_head, arguments):
        hidden, = arguments
        if hidden.shape != residual.shape or hidden.device != residual.device or hidden.dtype != residual.dtype:
            raise ValueError("Center residual and fused head features differ")
        uses.append(True)
        return (hidden + residual,)

    handle = head.register_forward_pre_hook(add)
    try:
        yield
        if len(uses) != 1:
            raise ValueError("Registered native forward must consume the bypass once")
    finally:
        handle.remove()


def predict(model, actual, producer, role, axis, *, execution=None):
    data = actual.verify(producer, role)
    if data["source"]["bank"] != axis.bank_sha256:
        raise ValueError("Center residual is bound to a different observation release")
    head = model.heads[actual.dataset]
    if head.in_features != axis.width:
        raise ValueError("Native fused width differs from the registered engineering contract")
    numeric = next(model.parameters())
    if numeric.is_cuda:
        torch.cuda.synchronize(numeric.device)
    started = perf_counter()
    timing = {}
    x = torch.tensor(actual.view.observation.spectral, device=numeric.device, dtype=numeric.dtype)
    weights = axis.projection(actual.view.bands.band_ids, numeric.device, numeric.dtype)
    residual = x.reshape(-1, x.shape[-1]) @ weights
    with fused_residual(head, residual):
        output, graph = forward(model, actual, producer, role, timed=timing, execution=execution)
    if numeric.is_cuda:
        torch.cuda.synchronize(numeric.device)
    timing["center_and_native_seconds"] = perf_counter() - started
    return output, graph, timing

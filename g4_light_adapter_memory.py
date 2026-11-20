"""Scoped value-projection recomputation for a frozen-mode G4 adapter."""

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from model.variable_channel_adapter import VariableChannelAdapter


RECOMPUTE = "eval_vca_value_projection_nonreentrant_v1"


@contextmanager
def recompute_value_projection(adapter):
    """Reuse the configured pixel slices without putting the adapter in train mode."""
    if (type(adapter) is not VariableChannelAdapter or adapter.training or not adapter.checkpoint_chunks
            or adapter.chunk_pixels <= 0 or "_project_to_latent" in vars(adapter)):
        raise ValueError("Eval VCA recomputation needs an unmodified, chunk-enabled adapter")
    project = adapter._project_to_latent
    trainable_values = any(param.requires_grad for param in adapter.value_proj.parameters())

    def project_chunk(flat_x, attention):
        if torch.is_grad_enabled() and (flat_x.requires_grad or attention.requires_grad or trainable_values):
            # Attention/dropout, graph and normalization remain outside the recomputed function.
            return checkpoint(project, flat_x, attention, use_reentrant=False)
        return project(flat_x, attention)

    adapter._project_to_latent = project_chunk
    try:
        yield
    finally:
        del adapter._project_to_latent

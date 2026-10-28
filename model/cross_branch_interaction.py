"""One simultaneous, optionally gated exchange between pixel and region features."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SelectiveMessage(nn.Module):
    def __init__(self, channels, hidden_dim, gated, strength, chunk_size):
        super().__init__()
        self.target_norm = nn.LayerNorm(channels) if gated else None
        self.source_norm = nn.LayerNorm(channels)
        self.projection = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        ) if gated else None
        self.strength = float(strength)
        self.chunk_size = int(chunk_size)

    def _chunk(self, target, source):
        source = self.source_norm(source)
        message = self.projection(source)
        if self.gate is None:
            gate = target.new_ones((target.shape[0], 1))
        else:
            target_normalized = self.target_norm(target)
            gate = self.gate(torch.cat(
                (target_normalized, source, (target_normalized - source).abs()), dim=-1,
            ))
        return target + self.strength * gate * message, gate

    def forward(self, target, source):
        if target.shape != source.shape:
            raise ValueError("Align source and target shapes before cross-branch interaction")
        size = self.chunk_size or target.shape[0]
        outputs, gates = [], []
        for start in range(0, target.shape[0], size):
            args = (target[start:start + size], source[start:start + size])
            if self.training and torch.is_grad_enabled() and self.chunk_size > 0:
                output, gate = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                output, gate = self._chunk(*args)
            outputs.append(output)
            gates.append(gate)
        return torch.cat(outputs, dim=0), torch.cat(gates, dim=0)


class BidirectionalSelectiveInteraction(nn.Module):
    """Alignment is external; both directions read pre-interaction features."""

    def __init__(self, channels=128, mode="gated", direction="both", hidden_dim=32,
                 strength=0.1, chunk_size=16384):
        super().__init__()
        if mode not in ("none", "add", "gated"):
            raise ValueError("interaction_mode must be none, add or gated")
        if direction not in ("both", "cnn_to_graph", "graph_to_cnn"):
            raise ValueError("Unknown interaction_direction")
        if channels <= 0 or hidden_dim <= 0 or strength < 0 or chunk_size < 0:
            raise ValueError("Invalid interaction dimensions, strength or chunk size")
        self.mode, self.direction = mode, direction
        kwargs = dict(channels=channels, hidden_dim=hidden_dim, gated=mode == "gated",
                      strength=strength, chunk_size=chunk_size)
        self.graph_to_cnn = SelectiveMessage(**kwargs) if mode != "none" and direction != "cnn_to_graph" else None
        self.cnn_to_graph = SelectiveMessage(**kwargs) if mode != "none" and direction != "graph_to_cnn" else None

    def forward(self, pixels, regions, region_at_pixels=None, pixels_at_regions=None,
                collect_stats=False, return_diagnostics=False):
        updated_pixels, updated_regions = pixels, regions
        info = {}
        for name, module, target, source in (
            ("graph_to_cnn", self.graph_to_cnn, pixels, region_at_pixels),
            ("cnn_to_graph", self.cnn_to_graph, regions, pixels_at_regions),
        ):
            if module is None:
                continue
            if source is None:
                raise ValueError("Missing aligned message for " + name)
            updated, gate = module(target, source)
            if name == "graph_to_cnn":
                updated_pixels = updated
            else:
                updated_regions = updated
            if collect_stats:
                info[name + "_gate_mean"] = gate.detach().mean()
                info[name + "_gate_std"] = gate.detach().std(unbiased=False)
            if return_diagnostics:
                info[name + "_gate"] = gate.detach()
        return updated_pixels, updated_regions, info

"""Train and restore opt-in WP07 variants through the spatial development protocol."""

from dataclasses import asdict, dataclass

import torch

from model.checkpoint_io import read_checkpoint_metadata
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN


@dataclass(frozen=True)
class MultiscaleTrial:
    cnn_multiscale: str = "fixed"
    cnn_dilation: int = 1
    graph_multiscale: str = "off"
    graph_radius: int = 1

    def initialize(self, baseline, seed):
        cfg = baseline.export_config()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            variant = ConditionalMultiscaleVCASAHGCN(
                cfg["class_counts"], **cfg["backbone_kwargs"], **asdict(self),
            ).to(next(baseline.parameters()))
        missing, unexpected = variant.load_state_dict(baseline.state_dict(), strict=False)
        new = set(dict(variant.named_parameters())) - set(dict(baseline.named_parameters()))
        if unexpected or set(missing) != new:
            raise ValueError("Multiscale trial must preserve the shared initialized backbone and heads")
        return variant


def load_multiscale_checkpoint(path, device="cpu"):
    metadata = read_checkpoint_metadata(path)
    if metadata is None or metadata.get("format_version") != 1:
        raise ValueError("Multiscale checkpoint requires its architecture sidecar")
    kwargs = metadata["backbone_kwargs"]
    if "cnn_multiscale" not in kwargs:
        raise ValueError("Checkpoint is missing the multiscale architecture")
    with torch.random.fork_rng(devices=[]):
        model = ConditionalMultiscaleVCASAHGCN(metadata["class_counts"], **kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.training_config = metadata.get("training_config", {})
    return model.eval()

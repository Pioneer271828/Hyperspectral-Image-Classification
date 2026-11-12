"""GroupNorm initialization for spatial zero-update consumers."""

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from train_spatial import SpatialConfig


TARGET_SEED = 64


def group_norm_baseline(config, scene, classes):
    """Use the established WP07 initialization with the accepted spatial geometry."""
    kwargs = SpatialConfig(config).research.model_kwargs()
    if (config["dataset"] != scene or kwargs["latent_channels"] != 128 or kwargs["adapter_embed_dim"] != 64
            or kwargs["norm_type"] != "domain_bn" or kwargs["graph_mode"] != "learned"):
        raise ValueError("Expected the accepted spatial model geometry")
    kwargs.update(norm_type="group_norm", graph_route=None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TARGET_SEED)
        model = MultiDatasetVCASAHGCN({scene: classes}, **kwargs).eval()
    model.backbone.adapter.cache_attention = False
    return model

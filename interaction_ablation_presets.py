"""Named, independent ablations; no experiment is executed on import."""

from model.interaction_config import interaction_kwargs


PRESETS = {
    "interaction_baseline": {},
    "interaction_add": {"interaction_mode": "add"},
    "interaction_gated": {"interaction_mode": "gated"},
    "interaction_cnn_to_graph": {"interaction_mode": "gated", "interaction_direction": "cnn_to_graph"},
    "interaction_graph_to_cnn": {"interaction_mode": "gated", "interaction_direction": "graph_to_cnn"},
    "interaction_edge_graph": {"graph_update_mode": "graph"},
    "interaction_edge_joint": {"graph_update_mode": "joint"},
    "interaction_add_edge": {"interaction_mode": "add", "graph_update_mode": "joint"},
    "interaction_full": {"interaction_mode": "gated", "graph_update_mode": "joint"},
    "interaction_global": {"alignment_mode": "global", "interaction_mode": "gated", "graph_update_mode": "joint"},
}


def build_interaction_experiments(args):
    base = dict(latent_channels=args.base_latent_channels, adapter_embed_dim=args.base_adapter_embed_dim,
                adapter_attention_mode="wavelength", norm_type=args.interaction_norm_type,
                graph_mode=args.interaction_graph_mode, train_drop_band_ratio=0.0)
    common = interaction_kwargs(vars(args))
    return [dict(base, **{"name": name, **common, **overrides}) for name, overrides in PRESETS.items()]

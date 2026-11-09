"""Resolve the existing independent interaction presets without running training."""

from interaction_ablation_presets import PRESETS


def build_research_experiments(config, names, seeds):
    if not names or not seeds or len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("Experiment names and seeds must be nonempty and unique")
    if "all" in names:
        if names != ["all"]:
            raise ValueError("all cannot be combined with named experiments")
        names = list(PRESETS)
    unknown = set(names) - (set(PRESETS) | {"configured"})
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    result = []
    for name in names:
        for seed in seeds:
            # Each named preset starts with all interaction switches at baseline.
            # Never inherit a gated switch accidentally from the template config.
            overrides = {"alignment_mode": "region", "interaction_mode": "none",
                         "interaction_direction": "both", "graph_update_mode": "none"}
            overrides.update(PRESETS[name]) if name != "configured" else overrides.clear()
            result.append(config.variant(config.name if name == "configured" else name, overrides, seed))
    return result

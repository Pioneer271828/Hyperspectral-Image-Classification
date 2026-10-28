"""Shared architecture switches for training, ablations and checkpoint readers."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class InteractionConfig:
    alignment_mode: str = "region"
    interaction_mode: str = "none"
    interaction_direction: str = "both"
    interaction_strength: float = 0.1
    interaction_hidden_dim: int = 32
    interaction_chunk_pixels: int = 16384
    graph_update_mode: str = "none"
    graph_update_dim: int = 32
    graph_update_mix: float = 0.5

    def __post_init__(self):
        choices = {
            "alignment_mode": ("region", "global"),
            "interaction_mode": ("none", "add", "gated"),
            "interaction_direction": ("both", "cnn_to_graph", "graph_to_cnn"),
            "graph_update_mode": ("none", "graph", "joint"),
        }
        for key, valid in choices.items():
            if getattr(self, key) not in valid:
                raise ValueError("{} must be one of {}".format(key, valid))
        if not math.isfinite(self.interaction_strength) or self.interaction_strength < 0:
            raise ValueError("interaction_strength must be finite and nonnegative")
        if self.interaction_hidden_dim <= 0 or self.graph_update_dim <= 0:
            raise ValueError("Interaction and graph update dimensions must be positive")
        if self.interaction_chunk_pixels < 0:
            raise ValueError("interaction_chunk_pixels must be nonnegative")
        if not 0 <= self.graph_update_mix <= 1:
            raise ValueError("graph_update_mix must lie in [0, 1]")

    @property
    def active(self):
        return self.interaction_mode != "none" or self.graph_update_mode != "none"


def interaction_kwargs(source, prefix=""):
    values = source if isinstance(source, dict) else vars(source)
    return asdict(InteractionConfig(**{
        key: values.get(prefix + key, default)
        for key, default in asdict(InteractionConfig()).items()
    }))


def add_interaction_arguments(parser, prefix=""):
    group = parser.add_argument_group("Modular pixel/region interaction and graph update ablations")
    options = {
        "alignment_mode": (str, ("region", "global")),
        "interaction_mode": (str, ("none", "add", "gated")),
        "interaction_direction": (str, ("both", "cnn_to_graph", "graph_to_cnn")),
        "graph_update_mode": (str, ("none", "graph", "joint")),
    }
    for key, value in asdict(InteractionConfig()).items():
        dtype, choices = options.get(key, (type(value), None))
        kwargs = dict(type=dtype, default=value)
        if choices is not None:
            kwargs["choices"] = choices
        group.add_argument("--" + prefix + key.replace("_", "-"), **kwargs)
    return group

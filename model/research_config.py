"""Versioned configuration for the existing trainer; no torch or data IO on import."""

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import math
import re
from types import SimpleNamespace

from model.interaction_config import InteractionConfig


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _group(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    defaults = asdict(cls())
    unknown = set(value) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    for key, item in value.items():
        default = defaults[key]
        if key == "locality_window" and item is None:
            continue
        if isinstance(default, bool):
            valid = type(item) is bool
        elif isinstance(default, int):
            valid = type(item) is int
        elif isinstance(default, float):
            valid = type(item) in (int, float) and math.isfinite(item)
        else:
            valid = type(item) is type(default)
        if not valid:
            raise ValueError(f"Invalid type or non-finite value for {cls.__name__}.{key}")
    return cls(**value)


@dataclass(frozen=True)
class ModelConfig:
    alph: float = 0.8
    latent_channels: int = 128
    adapter_embed_dim: int = 64
    wavelength_min: float = 400.0
    wavelength_max: float = 2500.0
    locality_sigma: float = 0.08
    locality_window: float = 0.16
    adapter_attention_mode: str = "wavelength"
    adapter_chunk_pixels: int = 0
    adapter_checkpoint_chunks: bool = True
    norm_type: str = "domain_bn"
    graph_mode: str = "static"
    learned_assignment_dim: int = 32
    learned_assignment_radius: int = 1
    learned_graph_radius: int = 1
    learned_assignment_temperature: float = 1.0
    learned_assignment_spatial_sigma: float = 1.0
    learned_graph_spatial_sigma: float = 1.5
    graph_assignment_mode: str = "learned"
    graph_adjacency_mode: str = "learned"


@dataclass(frozen=True)
class LossConfig:
    cnn_loss_weight: float = 1.0
    graph_branch_loss_weight: float = 1.0
    graph_compactness_weight: float = 0.01
    graph_entropy_weight: float = 0.001
    graph_balance_weight: float = 0.01
    balanced: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    steps_per_epoch: int = 0
    seed: int = 2026
    val_interval: int = 5
    lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 1e-3
    primary_checkpoint: str = "macro_oa"
    eval_checkpoint: str = "primary"


@dataclass(frozen=True)
class DataConfig:
    datasets: list = field(default_factory=lambda: ["paviaU"])
    protocol: str = "legacy_random_pixel"
    samples_type: str = "ratio"
    train_num: int = 30
    val_num: int = 10
    train_ratio: float = 0.01
    val_ratio: float = 0.005
    superpixel_scale: int = 100
    pavia_superpixel_scale: int = 50
    skip_missing: bool = False


@dataclass(frozen=True)
class AugmentationConfig:
    train_drop_band_ratio: float = 0.0
    train_band_mixup_ratio: float = 0.0
    train_band_mixup_prob: float = 1.0
    train_band_mixup_alpha: float = 0.4
    train_band_mixup_neighbors: int = 1


@dataclass(frozen=True)
class ResearchConfig:
    schema_version: int = 1
    name: str = "baseline"
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown fields or invalid research configuration")
        groups = {"model": ModelConfig, "interaction": InteractionConfig, "loss": LossConfig,
                  "training": TrainingConfig, "data": DataConfig, "augmentation": AugmentationConfig}
        kwargs = {k: v for k, v in value.items() if k not in groups}
        kwargs.update({k: _group(group, value.get(k, {})) for k, group in groups.items()})
        result = cls(**kwargs)
        result.validate()
        return result

    @classmethod
    def from_legacy(cls, args):
        values = vars(args)
        model = {f.name: values[f.name] for f in fields(ModelConfig) if f.name in values}
        model["adapter_checkpoint_chunks"] = not args.no_adapter_checkpoint
        loss = {f.name: values[f.name] for f in fields(LossConfig) if f.name in values}
        loss["balanced"] = not args.no_balanced_loss
        data = {f.name: values[f.name] for f in fields(DataConfig) if f.name in values}
        data["datasets"] = [s.strip() for s in args.datasets.split(",") if s.strip()]
        return cls.from_dict({
            "name": args.experiment_name or "legacy", "device": args.device,
            "model": model, "interaction": {f.name: values[f.name] for f in fields(InteractionConfig)},
            "loss": loss, "training": {f.name: values[f.name] for f in fields(TrainingConfig)},
            "data": data, "augmentation": {f.name: values[f.name] for f in fields(AugmentationConfig)},
        })

    def validate(self):
        # Re-check even dataclass-created values, so replace(...) cannot bypass validation.
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported research config schema_version")
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.name):
            raise ValueError("name must be a short safe ASCII identifier")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be cpu, cuda or cuda:<index>")
        for group in (self.model, self.interaction, self.loss, self.training, self.data, self.augmentation):
            _group(type(group), asdict(group))
        m, t, d, a = self.model, self.training, self.data, self.augmentation
        if not 0 <= m.alph <= 1:
            raise ValueError("model.alph must be in [0,1]")
        for key in ("latent_channels", "adapter_embed_dim", "learned_assignment_dim"):
            if getattr(m, key) <= 0:
                raise ValueError(f"model.{key} must be positive")
        for key in ("adapter_chunk_pixels", "learned_assignment_radius", "learned_graph_radius"):
            if getattr(m, key) < 0:
                raise ValueError(f"model.{key} must be nonnegative")
        for key in ("locality_sigma", "learned_assignment_temperature", "learned_assignment_spatial_sigma", "learned_graph_spatial_sigma"):
            if getattr(m, key) <= 0:
                raise ValueError(f"model.{key} must be positive")
        if m.locality_window is not None and m.locality_window <= 0:
            raise ValueError("locality_window must be positive or null")
        if m.wavelength_min >= m.wavelength_max:
            raise ValueError("wavelength_min must be smaller than wavelength_max")
        choices = [(m.graph_mode, ("static", "learned")), (m.norm_type, ("domain_bn", "batch_norm", "group_norm")),
                   (m.adapter_attention_mode, ("wavelength", "index", "uniform")),
                   (t.primary_checkpoint, ("macro_oa", "val_loss")), (d.samples_type, ("ratio", "number")),
                   (d.protocol, ("legacy_random_pixel",))]
        if any(v not in allowed for v, allowed in choices):
            raise ValueError("Unknown model, selection or data protocol mode")
        if m.graph_assignment_mode not in ("learned", "fixed_grid") or m.graph_adjacency_mode not in ("learned", "fixed_local"):
            raise ValueError("Unknown graph assignment/adjacency mode")
        if m.graph_mode == "static" and (m.graph_assignment_mode != "learned" or m.graph_adjacency_mode != "learned"):
            raise ValueError("Graph builder controls require graph_mode=learned")
        if t.eval_checkpoint not in ("primary", "last", "val_loss", "macro_oa", *d.datasets):
            raise ValueError("Invalid eval_checkpoint")
        if t.epochs < 0 or t.steps_per_epoch < 0 or not 0 <= t.seed < 2 ** 32 or t.val_interval <= 0:
            raise ValueError("Invalid training budget/seed; epochs=0 means one legacy epoch")
        if t.seed + len(d.datasets) - 1 >= 2 ** 32:
            raise ValueError("Dataset seed offsets exceed NumPy seed range")
        if t.lr <= 0 or t.head_lr <= 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if any(v < 0 for k, v in asdict(self.loss).items() if k != "balanced"):
            raise ValueError("Loss weights must be nonnegative")
        if not d.datasets or any(not isinstance(s, str) or not s.strip() or "." in s or "/" in s or "\\" in s for s in d.datasets):
            raise ValueError("Nonempty dataset identifiers without dots/path separators required")
        if len(set(d.datasets)) != len(d.datasets):
            raise ValueError("Duplicate datasets would silently overwrite preparation")
        if min(d.train_num, d.val_num, d.superpixel_scale, d.pavia_superpixel_scale) <= 0:
            raise ValueError("Sample counts and superpixel scales must be positive")
        if not 0 < d.train_ratio < 1 or not 0 < d.val_ratio < 1 or d.train_ratio + d.val_ratio >= 1:
            raise ValueError("Train/validation ratios must leave a nonempty test fraction")
        if not 0 <= a.train_drop_band_ratio < 1 or not 0 <= a.train_band_mixup_ratio <= 1 or not 0 <= a.train_band_mixup_prob <= 1:
            raise ValueError("Invalid augmentation ratios")
        if a.train_band_mixup_neighbors < 1 or a.train_band_mixup_alpha <= 0:
            raise ValueError("Mixup alpha and neighbors must be positive")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @property
    def config_hash(self):
        return digest(self.to_dict())

    @property
    def budget(self):
        steps = self.training.steps_per_epoch or len(self.data.datasets) * 3
        return {"epoch_iterations": self.training.epochs + 1, "steps_per_epoch": steps,
                "optimizer_steps": (self.training.epochs + 1) * steps, "legacy_inclusive_epochs": True}

    def model_kwargs(self):
        self.validate()
        return {**asdict(self.model), **asdict(self.interaction)}

    def build_model(self, class_counts):
        from model.VCASAHGCN import MultiDatasetVCASAHGCN

        return MultiDatasetVCASAHGCN(class_counts, **self.model_kwargs())

    def trainer_args(self, save_path, metrics_path):
        values = {**asdict(self.model), **asdict(self.interaction), **asdict(self.loss),
                  **asdict(self.training), **asdict(self.data), **asdict(self.augmentation)}
        values.update(device=self.device, datasets=",".join(self.data.datasets),
                      no_adapter_checkpoint=not self.model.adapter_checkpoint_chunks,
                      no_balanced_loss=not self.loss.balanced, no_wavelength_attention=False,
                      stable_preset="none", no_progress=True, step_progress=False,
                      save_path=str(save_path), metrics_json=str(metrics_path), experiment_name=self.name)
        return SimpleNamespace(**values)

    def variant(self, name, overrides, seed):
        result = replace(self, name=name, interaction=replace(self.interaction, **overrides),
                         training=replace(self.training, seed=seed))
        result.validate()
        return result

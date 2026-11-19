"""Registered production switches for retained-band native training and deployment."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.coverage_config import CONDITIONS
from model.research_config import digest
from transfer import native_positive_training as controls
from transfer.native_scale_observation import forward
from transfer.spectral_spacing_recovery import model_for
from transfer.wp06_center_spectral import fused_residual

ARMS = ("native", "center", "spatial_shift", "pixel")
SPECTRAL = ("none", *CONDITIONS)
RESIDUAL = ("none", "x_residual")
CONFIG = "wp05_wp06_production_switches_v1"


@dataclass(frozen=True)
class Switches:
    arm: str = "native"
    spectral: str = "none"
    residual: str = "none"
    graph_scale: int = 50

    def __post_init__(self):
        if (self.arm not in ARMS or self.spectral not in SPECTRAL or self.residual not in RESIDUAL
                or type(self.graph_scale) is not int or self.graph_scale not in (16, 50, 100)
                or (self.arm == "pixel" and self.residual != "none")):
            raise ValueError("Unknown or incompatible native candidate switches")

    @property
    def fingerprint(self):
        return digest([CONFIG, asdict(self)])


def pixel_dimensions(target, width, classes):
    middle = int(target**.5)
    choices = []
    for first in range(int(.85*middle), int(1.15*middle)):
        second = round((target-(width+1)*first-3*classes)/(first+1+3*classes))
        size = (width+1)*first+(first+1+3*classes)*second+3*classes
        if second > 0:
            choices.append((abs(size-target), first, second))
    error, first, second = min(choices)
    if error > .005*target:
        raise ValueError("Pure spectral reference does not meet the registered capacity tolerance")
    return first, second


def configured_model(base, config, width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_for(base, config.spectral, seed)
        if config.residual != "none":
            metadata = model.export_config()
            kwargs = {**metadata["backbone_kwargs"], "region_residual_mode": config.residual}
            new = type(model)(metadata["class_counts"], **kwargs)
            new.load_state_dict({**new.state_dict(), **model.state_dict()}, strict=True)
            model = new
        if config.arm == "pixel":
            classes, = (head.out_features for head in model.heads.values())
            target = sum(p.numel() for p in model.parameters())
            return controls.PixelModel(pixel_dimensions(target, width, classes), width=width, classes=classes)
    model.backbone.adapter.cache_attention = False
    return model


def axis_slots(registered, actual, producer, role):
    receipt = actual.verify(producer, role)
    retained = tuple(receipt["source"]["band_ids"])
    lookup = {key: i for i, key in enumerate(registered)}
    if (not retained or retained != actual.view.bands.band_ids or not set(retained).issubset(lookup)
            or len(set(lookup)) != len(registered)):
        raise ValueError("Only actual registered retained bands can enter the production model")
    indices = tuple(lookup[key] for key in retained)
    if indices != tuple(sorted(set(indices))):
        raise ValueError("Retained columns are out of the approved original order")
    return indices


def project_center(values, indices, registered, *, shifted):
    """Apply the same fixed spectral projection and pixel indexing to both arms."""
    # Draw weights without reference to any scene observations or labels.
    matrix = np.random.default_rng(6605).integers(0, 2, size=(len(registered), 64), dtype=np.int8)
    device, dtype = values.device, values.dtype
    weights = torch.as_tensor((2.*matrix[list(indices)].astype(np.float32)-1.)/np.sqrt(len(registered)),
                              device=device, dtype=dtype)
    height, width, bands = values.shape
    pixels = torch.arange(height*width, device=device).reshape(height,width)
    order = pixels.roll((height//2 if shifted else 0, width//3 if shifted else 0), (0, 1)).reshape(-1)
    return ((values.reshape(-1, bands)-.5) @ weights).index_select(0, order)


def run_forward(model, actual, producer, role, config, registered, *, execution=None):
    started = perf_counter()
    indices = axis_slots(registered, actual, producer, role)
    parameter = next(model.parameters())
    timing = {}
    if config.arm == "native":
        output, graph = forward(model, actual, producer, role, execution=execution, timed=timing,
                                graph_scale=config.graph_scale)
    else:
        values = torch.tensor(actual.view.observation.spectral, device=parameter.device, dtype=parameter.dtype)
        if config.arm == "pixel":
            index = torch.tensor(indices, dtype=torch.long, device=parameter.device)
            output = model(values, index)
            graph = dict(q_shape=[len(output[0]), 0], q_nnz=0, a0_nnz=0, scope="pixel_only")
            timing["graph_seconds"] = []
        else:
            residual = project_center(values, indices, registered, shifted=config.arm == "spatial_shift")
            with fused_residual(model.heads[actual.dataset], residual):
                output, graph = forward(model, actual, producer, role, execution=execution, timed=timing,
                                        graph_scale=config.graph_scale)
    timing["forward_seconds"] = perf_counter()-started
    return output, graph, timing


def export_production(model, path, config, registered):
    if config.arm == "pixel":
        raise ValueError("Spectral reference is assembled by the registered provider")
    identity = dict(schema=CONFIG, switches=asdict(config), axis=list(registered),
                    fingerprint=digest([asdict(config), tuple(registered)]))
    original = getattr(model, "training_config", {})
    model.training_config = {**original, CONFIG: identity}
    try:
        save_model_checkpoint(model, path)
    finally:
        model.training_config = original


def open_production(path, device, *, legacy=None, axis=None):
    """Load unchanged legacy checkpoints or native exports with explicit switches."""
    model = load_model_checkpoint(path, device)
    metadata = model.training_config.get(CONFIG)
    if metadata is None:
        if legacy is None or axis is None:
            raise ValueError("Legacy states require explicit native switches and admitted band IDs")
        config, registered = legacy, tuple(axis)
    else:
        config, registered = Switches(**metadata["switches"]), tuple(metadata["axis"])
        if (legacy is not None or axis is not None or metadata["schema"] != CONFIG
                or metadata["fingerprint"] != digest([asdict(config), registered])):
            raise ValueError("Exported switches or band identity differ")
    if (model.backbone.region_residual_mode != config.residual
            or model.backbone.coverage_mode != ("none" if config.spectral == "none" else "affine")
            or (config.spectral != "none" and model.backbone.coverage_modulation.condition != config.spectral)
            or config.arm == "pixel" or not registered or list(registered) != sorted(set(registered))):
        raise ValueError("Checkpoint topology or declared band axis differs")
    model.backbone.adapter.cache_attention = False
    return model, config, registered

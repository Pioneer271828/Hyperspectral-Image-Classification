"""Run matched spectral arms through the native released-point trainer."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata, _apply
from model.research_config import digest
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer import native_released_points as points, native_released_training as released, native_scale_training as training
from transfer.development_bank_training import _initial
from transfer.e1_ablation import MODES
from transfer.native_scale_fixture import Fixture, profiles
from transfer.native_scale_observation import _materialize
from transfer.spatial_multiscale import initialize
from transfer.spectral_spacing import spectral_spacing

ARMS = (*MODES, "spacing")
DENSITIES = ("dense", "sparse")
VIEWS = {
    "continuous": (3, 4, 5, 6, 7),
    "dispersed": (0, 1, 3, 6, 10),
    "shifted": (0, 1, 3, 6, 10),
    "missing": (0, 1, 3, 6, 10),
}
SEEDS = training.SEEDS
FIXTURE_PRODUCER = "9123c025b6bc60cd73a0f2eaf305b4f8261a0a91"
TRACKED = ("backbone.adapter.key_encoder.0.weight", "backbone.adapter.value_proj.0.weight")
NativeRun = training.Run


def model_for(source, arm, seed):
    model = initialize(source, "off", seed)
    if arm in ("none", "spacing"):
        return model
    config = model.export_config()
    kwargs = {**config["backbone_kwargs"], "coverage_mode": "affine", "coverage_condition": arm}
    with torch.random.fork_rng(devices=[]):
        e1 = type(model)(config["class_counts"], **kwargs)
    initial = e1.state_dict()
    common = model.state_dict()
    e1.load_state_dict({**initial, **common}, strict=True)
    e1.backbone.adapter.cache_attention = False
    return e1


def materialize(fixture, event, producer, density):
    actual, labels, _ = fixture.materialize(event["tile"], 1, "full", event["profile"], producer)
    keep = VIEWS[event["band"]]
    view = _apply(actual.view.observation, actual.view.bands, keep)
    bands = view.bands
    if event["band"] == "shifted":
        bands = replace(bands, wavelengths=tuple(wave + 10 for wave in bands.wavelengths))
    if event["band"] == "missing":
        bands = BandMetadata(bands.source_id, bands.band_ids, tuple(np.linspace(0, 1, len(keep))),
                             "normalized", "index_only", bands.valid)
    source = {**actual.receipt["source"], "positions": list(keep), "band_view": event["band"],
              "selection": digest({"selected": keep, "metadata": bands.fingerprint})}
    role = actual.receipt["role"]
    native = _materialize(actual.dataset, actual.classes, replace(view, bands=bands),
                          actual.lattice, source, role, producer, profile=next(p for p in profiles() if p.key == event["profile"]))
    y0, _, x0, _ = actual.lattice.box
    width = actual.lattice.scene_shape[1]
    locations = (np.argwhere(labels > 0) if density == "dense" else
                 [np.argwhere(labels == cls)[0] for cls in range(1, actual.classes + 1)])
    pairs = sorted([[(int(y)+y0)*width+int(x)+x0, int(labels[y, x])] for y, x in locations])
    row = dict(id=event["tile"], role=role, box=actual.lattice.box, samples=pairs)
    return native, labels, points.bind(native, row, producer, role, synthetic_labels=labels)


def contract(classes, seed, _arm, factor, producer, *, numeric=None, arm, density):
    check(classes == 9 and factor == 1 and numeric is None and _arm == "off"
          and seed in SEEDS and arm in ARMS and density in DENSITIES, "Unexpected synthetic design")
    train, val = (Fixture(classes, role) for role in ("devtrain", "devval"))
    order = np.random.default_rng(seed).permutation(train.tiles)
    modes = ("continuous", "dispersed", "shifted")
    schedule = [dict(step=index, tile=int(tile), factor=1, band=modes[(index-1) % len(modes)],
                     profile="a", condition="matched") for index, tile in enumerate(order, 1)]
    evaluate = [dict(tile=tile, factor=1, band=band, profile="b", condition="matched")
                for tile, band in product(val.tiles, VIEWS)]
    roster = {}
    for role, fixture, events in (("devtrain", train, schedule), ("devval", val, evaluate)):
        for event in events:
            roster[released._key(role, event)] = released._evidence(*materialize(fixture, event, producer, density))
    base = training.baseline(classes, seed)
    original = model_for(base, "none", seed + 7321)
    model = model_for(base, arm, seed + 7321)
    return dict(format="wp05_spacing_released_points_v1", producer=producer, classes=classes,
                seed=seed, arm="off", spectral_arm=arm, density=density, factor=factor,
                supervision=points.MODE, objective=points.LOSS, training=schedule, validation=evaluate,
                steps=6, prefix=2, events=[0, 3, 6], learning_rate=.0001, module_seed=seed+7321,
                rng_seed=seed+15000, selection="first_min_released_point_fused_nll",
                labels_root=digest({k: x["released"]["point_set"] for k,x in roster.items()}),
                inputs=roster, initialization=_initial(model, original), registered_graph_scale=16,
                parameters=sum(p.numel() for p in model.parameters()), numeric=None,
                raw_gt_loaded=False, query_accessed=False, real_updates=0)


class SpacingRun(NativeRun):
    def __init__(self, registered, producer, **kwargs):
        super().__init__(registered, producer, **kwargs)
        self._fresh = (*self._fresh, *TRACKED)

    def _forward(self, actual, role, intervention):
        with spectral_spacing(self.model, actual.view.bands, enabled=self._contract["spectral_arm"] == "spacing"):
            return super()._forward(actual, role, intervention)


@dataclass(frozen=True)
class FixtureCondition:
    source_key: str
    bands: BandMetadata
    gsd_m: float | None = None

    def validate(self, actual, receipt):
        check(receipt["observation"] == self.source_key and actual.view.bands == self.bands,
              "Synthetic condition does not match the released observed input")

    def values(self):
        return self.bands, None, None


def condition(actual, _registered):
    return FixtureCondition(actual.receipt["observation"], actual.view.bands)


@contextmanager
def design(arm, density):
    def freeze(*args, **kwargs):
        return contract(*args, **kwargs, arm=arm, density=density)

    def observation(fixture, event, producer):
        return materialize(fixture, event, producer, density)

    def model(source, _arm, seed):
        return model_for(source, arm, seed)

    with (patch.object(released, "contract", freeze), patch.object(released, "materialize", observation),
          patch.object(released, "model_factory", model, create=True), patch.object(training, "Run", SpacingRun),
          patch.object(released, "selection_condition", condition, create=True)):
        yield


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def one_case(directory, producer, seed, arm, density):
    start = perf_counter()
    with design(arm, density):
        spec = released.contract(9, seed, "off", 1, producer)
        phases = {}
        previous = {"resume": "prefix", "completed": "resume"}
        saved = {}
        for phase in ("full", "prefix", "resume", "completed"):
            path = directory / f"{phase}.pt"
            upstream = read_checkpoint(directory / f"{previous[phase]}.pt") if phase in previous else None
            phases[phase] = training.execute(spec, phase, path, producer, upstream, released=released)
            saved[phase] = pin(path)["sha256"]
        full, prefix, resumed, complete = (read_checkpoint(directory / f"{name}.pt") for name in phases)
        check(training._exact(full, resumed) and training._exact(resumed, complete)
              and training._exact(prefix["history"], full["history"][:2]), "Exact spectral recovery differs")
        selected = min(full["validation"], key=lambda row: row["fused_nll"])
        check(selected["step"] == full["selected"]["step"], "First-min selection changed")
        readout = training.Run(spec, producer, released=released)
        readout.model.load_state_dict(full["selected"]["state"], strict=True)
        reviewed = readout.evaluate(selected["step"], detailed=True)
        check(reviewed["fused_nll"] == selected["fused_nll"], "Selected evaluation differs")
    tracked = (*TRACKED, *spec["initialization"]["fresh"])
    updated = {name: sum(event["fresh"][name]["changed"] and
                        (event["fresh"][name]["gradient_l1"] or 0) > 0 for event in full["history"]) for name in tracked}
    check(all(value > 0 for key, value in updated.items() if key in TRACKED)
          and (arm in ("none", "spacing") or any(updated[key] for key in spec["initialization"]["fresh"])),
          "Registered VCA/E1 parameters did not receive an effective gradient")
    return dict(seed=seed, arm=arm, density=density, parameters=spec["parameters"],
                common_initial=spec["initialization"]["source"], labels_root=spec["labels_root"],
                inputs=digest(spec["inputs"]), selected_step=selected["step"], selected_nll=selected["fused_nll"],
                phases=phases, checkpoint_sha256=saved, updated=updated,
                validation=reviewed["rows"], seconds=perf_counter() - start)


def run_synthetic(directory, code):
    output = directory / "synthetic"
    output.mkdir(exist_ok=True)
    cases = {}
    for seed, density, arm in product(SEEDS, DENSITIES, ARMS):
        name = f"{seed}-{density}-{arm}"
        place = output / name
        if place.exists():
            link = pin(place / "receipt.json")
            case = _read(link)
            original_baseline = (name == "64-dense-none" and link["sha256"] ==
                                 "59b061007a0302a081e72b5f40e6c68d603006f2dc1d9e2fb1c03082e9ff1434")
            check((original_baseline or case.get("producer") == code["commit"])
                  and (case["seed"], case["density"], case["arm"]) == (seed, density, arm)
                  and all(pin(place / f"{phase}.pt")["sha256"] == sha
                          for phase, sha in case["checkpoint_sha256"].items()),
                  "Inconsistent completed synthetic receipt")
            print(f"synthetic {name} reused", flush=True)
        else:
            staging = Path(mkdtemp(prefix=f".{name}-", dir=output))
            case = one_case(staging, FIXTURE_PRODUCER, seed, arm, density)
            case["producer"] = code["commit"]
            _write(staging / "receipt.json", case)
            staging.rename(place)
            print(f"synthetic {name} passed", flush=True)
        cases[name] = case
    return _write(output / "receipt.json", dict(status="passed", producer=code["commit"],
                  registration=pin(directory / "registration.json")["sha256"], cases=cases, real_updates=0,
                  query_accessed=False, raw_gt_loaded=False, synthetic_updates=len(cases)*12))

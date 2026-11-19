"""Train only synthetic released points with the historical WP05 coverage arms."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import digest
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer import development_bank as custody, development_bank_training as fixture, native_scale_training as native
from transfer.development_bank_training import _initial
from transfer.g3_training import state_hash
from transfer.native_released_points import LOSS, MODE, scores
from transfer.p4_coverage import ARMS
from transfer.spectral_spacing_recovery import model_for
from transfer.spatial_coverage_protocol import VIEWS, TRAIN_VIEWS, bank_view, counts, model_from_source, release_statistics

SCHEMA = "wp05_spatial_released_coverage_contract_v1"
SOURCES = (81271, 99043)
SPLITS = ("a", "b")


def generated(classes, split):
    original, arrays = fixture.fixture(classes)
    changed = {"a": {}, "b": {3: "devval", 36: "devval", 81: "devtrain", 99: "devtrain"}}[split]
    rows = []
    for raw in original.rows:
        row = deepcopy(raw)
        key = row["id"]
        row["role"] = changed.get(key, row["role"])
        row["partition"] = "train" if row["role"] == "devtrain" else "val"
        arrays[key] = np.ascontiguousarray(arrays[key] + np.float32(key / 40.))
        labels = row["samples"]
        if classes == 22:
            row["samples"] = [[i, y] for i, y in labels if y != 21 and not (
                split == "a" and row["role"] == "devtrain" and y == 22)]
            # The fixture deliberately has exactly one class-21 point in each role.
            if key in ((3, 81) if split == "a" else (10, 3)):
                a, b, c, d = row["box"]
                ids = {i for i, _ in row["samples"]}
                point = next(y * original.shape[1] + x for y in range(a, b) for x in range(c, d)
                             if y * original.shape[1] + x not in ids)
                row["samples"].append([point, 21])
            row["samples"].sort()
        row["origin"] = {**row["origin"], "array_sha256": custody.array_hash(arrays[key])}
        rows.append(row)
    rows.sort(key=custody._row_key)
    plan = digest({"split": split, "fixture": original.plan_hash, "geometry": rows})
    statistics = release_statistics(arrays, rows, plan)
    poison = {key: value if any(r["id"] == key and r["role"] == "devtrain" for r in rows) else
              np.full_like(value, np.nan) for key, value in arrays.items()}
    poison[-1] = np.full((2, 2, original.shape[-1]), np.nan, dtype=np.float32)
    check(statistics == release_statistics(poison, rows, plan), "Held/buffer values altered training fit statistics")
    bands = replace(original.bands, band_ids=tuple(101+7*i for i in range(len(original.bands.band_ids))))
    return replace(original, plan_hash=plan, statistics=statistics, rows=tuple(rows), bands=bands), arrays


def shared_source(scene, seed):
    source = "paviaU" if scene != "paviaU" else "xuzhou"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MultiDatasetVCASAHGCN({source: 9}, graph_mode="static", norm_type="domain_bn",
                                      latent_channels=128, adapter_embed_dim=64).eval()


class SpatialReleasedFixture:
    """Explicit synthetic provider for the unchanged production optimizer and checkpoint loop."""

    scores = staticmethod(scores)

    def __init__(self, bank, source_seed, split, spectral_arm, producer, source_files):
        if (not isinstance(bank, custody.Bank) or bank.release.profile != "synthetic"
                or spectral_arm not in ARMS or source_seed not in SOURCES or split not in SPLITS):
            raise ValueError("Only the registered spatial coverage fixture can update")
        self.bank = bank
        self.source_seed, self.split, self.arm, self.producer = source_seed, split, spectral_arm, producer
        self.scene, self.classes = bank.release.scene, bank.release.classes
        self.source_files = deepcopy(source_files)
        self.roster = {}

    @property
    def is_synthetic(self):
        return self.bank.release.profile == "synthetic"

    def baseline(self, seed):
        return self.initialization(seed)[0]

    def initialization(self, seed):
        target = fixture._baseline(self.bank.release, seed)
        check(all(pin(item["path"])["sha256"] == item["sha256"] for item in self.source_files.values()),
              "Source checkpoint or sidecar identity changed")
        original = load_model_checkpoint(self.source_files["weights"]["path"], device="cpu")
        audit = model_from_source(original, target, self.scene)
        audit["source_files"] = {name:item["sha256"] for name,item in self.source_files.items()}
        return target, audit

    def model_factory(self, source, arm, seed):
        check(arm == "off", "Spatial coverage uses the original graph/feature configuration")
        return model_for(source, self.arm, seed)

    def materialize(self, role, event, producer):
        check(producer == self.producer and event["band"] in VIEWS and event["factor"] == 1,
              "Unexpected producer, band or lattice")
        row = self.bank.role_row(role, event["tile"])
        return bank_view(self.bank, row, producer, event["band"])

    @staticmethod
    def key(role, event):
        return f'{role}/{event["tile"]}/{event["band"]}'

    @staticmethod
    def evidence(actual, target):
        receipt = actual.receipt
        source = receipt["source"]
        return dict(observation=receipt["observation"], fit_stats=source["retained_statistics"],
                    mode=source["spectral_metadata_mode"], band_ids=source["band_ids"],
                    spectral=receipt["spectral"], authority=source["bank"], released=target.summary())

    def validate_observation(self, spec, actual, labels, target, event):
        role = actual.receipt["role"]
        if self.evidence(actual, target) != spec["inputs"][self.key(role, event)]:
            raise ValueError("Spatial role, statistics, spectral stream or label exposure changed")

    def selection_condition(self, actual, spec):
        return None

    def stage_evidence(self, actual):
        source = actual.receipt["source"]
        return dict(fit_stats=source["retained_statistics"], mode=source["spectral_metadata_mode"],
                    band_ids=source["band_ids"], physical_metadata_passed=source["spectral_metadata_mode"] != "missing")

    def validate_history(self, spec, packet):
        for role, rows in (("devtrain", packet["history"]),
                           ("devval", [r for event in packet["validation"] for r in event["rows"]])):
            for row in rows:
                ref = spec["inputs"][self.key(role, row)]
                if any(row[k] != ref[k] for k in ("observation", "fit_stats", "mode", "band_ids", "released")):
                    raise ValueError("Spatial coverage checkpoint role/band/statistics exposure differs")
                if role == "devval" and any(score["count"] != ref["released"]["points"] for score in row["scores"].values()):
                    raise ValueError("Spatial validation count changed")

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        check((classes, seed, arm, factor, producer, numeric) == (self.classes, 64, "off", 1, self.producer, None),
              "Unexpected fixed engineering request")
        self.bank.verify_authority(self.bank.pin)
        chosen = [r["id"] for r in self.bank.role_rows("devtrain") if r["samples"]]
        check(len(chosen) == len(TRAIN_VIEWS), "Engineering clock requires exactly six released training blocks")
        order = np.random.default_rng(seed).permutation(chosen)
        train = [dict(step=i, tile=int(tile), factor=1, band=band, profile=None, condition="missing")
                 for i,(tile, band) in enumerate(zip(order, TRAIN_VIEWS, strict=True), 1)]
        val = [dict(tile=row["id"], factor=1, band=band, profile=None, condition="missing")
               for row, band in product(self.bank.role_rows("devval"), VIEWS)]
        if not self.roster:
            for role, events in (("devtrain", train), ("devval", val)):
                for event in events:
                    actual, _, target = self.materialize(role, event, producer)
                    self.roster[self.key(role,event)] = self.evidence(actual, target)
        source, witness = self.initialization(seed)
        state = self.model_factory(source, arm, seed+7321)
        return dict(format=SCHEMA, producer=producer, classes=classes, seed=seed, arm=arm, factor=factor,
                    supervision=MODE, objective=LOSS, training=train, validation=val, inputs=deepcopy(self.roster),
                    steps=6, prefix=2, events=[0,3,6], learning_rate=.0001, module_seed=seed+7321,
                    rng_seed=seed+15000, registered_graph_scale=100,
                    initialization=_initial(state, model_for(source, "none", seed+7321)),
                    parameters=sum(p.numel() for p in state.parameters()), schema=SCHEMA,
                    split=self.split, spectral_arm=self.arm, bank=self.bank.pin, source=witness,
                    selection="first_min_released_point_fused_nll", numeric=None,
                    raw_gt_loaded=False, query_accessed=False, real_updates=0)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def run_case(place, provider):
    started = perf_counter()
    producer = provider.producer
    spec = provider.contract(provider.classes, 64, "off", 1, producer)
    phases = {}
    parents = {"resume":"prefix", "completed":"resume"}
    for phase in ("full", "prefix", "resume", "completed"):
        prior = load(place / f"{parents[phase]}.pt") if phase in parents else None
        phases[phase] = native.execute(spec, phase, place / f"{phase}.pt", producer, prior, released=provider)
    full, prefix, resume, completed = (load(place / f"{phase}.pt") for phase in phases)
    check(native._exact(full, resume) and native._exact(full, completed)
          and native._exact(full["history"][:2], prefix["history"]), "Spatial coverage checkpoint recovery drift")
    best = min(full["validation"], key=lambda row: row["fused_nll"])
    check(best["step"] == full["selected"]["step"], "Spatial first minimum selection differs")
    runner = native.Run(spec, producer, released=provider)
    runner.model.load_state_dict(full["selected"]["state"], strict=True)
    evaluated = runner.evaluate(best["step"], detailed=True)
    check(evaluated["fused_nll"] == best["fused_nll"], "Independent endpoint selection differs")
    changed = {name for history in full["history"] for name, step in history["fresh"].items()
               if step["changed"] and (step["gradient_l1"] or 0.) > 0}
    check(provider.arm == "none" or changed, "Coverage arm lacks an effective gradient update")
    reports = {}
    for band in VIEWS:
        rows = [r for r in evaluated["rows"] if r["band"] == band]
        reports[band] = {"mode": rows[0]["mode"], "band_ids": rows[0]["band_ids"],
                         "physical_metadata_passed": rows[0]["physical_metadata_passed"],
                         "graph_identity": digest([r["graph"] for r in rows]),
                         "heads": {h: dict(n=sum(r["scores"][h]["count"] for r in rows),
                             confusion=sum(np.asarray(r["scores"][h]["point_confusion"]) for r in rows).tolist(),
                             nll=sum(r["scores"][h]["nll_sum"] for r in rows)) for h in HEADS}}
    result = dict(scene=provider.scene, classes=provider.classes, split=provider.split, source_seed=provider.source_seed,
                  arm=provider.arm, source=spec["source"], parameters=spec["parameters"], costs=phases,
                  joint_inputs=digest(spec["inputs"]), step=best["step"], validation=reports,
                  optimizer=12, reentry=0, changed_coverage=sorted(changed),
                  roster_counts={r: counts(provider.bank.role_rows(r), provider.classes) for r in ("devtrain","devval")},
                  packet_sha={phase:pin(place/f"{phase}.pt")["sha256"] for phase in phases},
                  seconds=perf_counter()-started)
    return _write(place / "receipt.json", result), spec, prefix


def reject_recovery(candidate, packet, producer):
    runner = native.Run(candidate.contract(candidate.classes,64,"off",1,producer), producer, released=candidate)
    try:
        runner.restore(packet, 2)
    except ValueError:
        return True
    raise ValueError("Foreign source or split checkpoint was accepted")


def run_synthetic(folder, code):
    root = folder / "synthetic"
    root.mkdir(exist_ok=True)
    cases, checks, chronology = {}, {}, {}
    for classes in (9, 22):
        banks = {}
        for split in SPLITS:
            release, arrays = generated(classes, split)
            out = root / f"{classes}-{split}-bank"
            if (out/"bank.json").exists():
                bank_pin = dict(root=str(out),sha256=pin(out/"bank.json")["sha256"],
                                registration=pin(out/"registration.json")["sha256"])
            else:
                binding = custody.publish(out, root, release, lambda row: arrays[row["id"]], resume=out.exists())
                bank_pin = {key:binding[key] for key in ("root","sha256","registration")}
            banks[split] = custody.Bank(bank_pin, root, release)
        check(banks["a"].release.statistics["statistics_hash"] != banks["b"].release.statistics["statistics_hash"],
              "Geometry switch reused fit statistics")
        source_files = {}
        for seed in SOURCES:
            weights = root / f"{classes}-{seed}-old-source.pt"
            original = shared_source(banks["a"].release.scene, seed)
            if weights.exists():
                check(state_hash(load_model_checkpoint(weights)) == state_hash(original), "Historical source packet differs")
            else:
                save_model_checkpoint(original, weights)
            source_files[seed] = dict(weights=pin(weights), sidecar=pin(Path(str(weights)+".json")))
        for s, split, arm in product(SOURCES, SPLITS, ARMS):
            place = root / f"{classes}-{s}-{split}-{arm}"
            if (place/"receipt.json").exists():
                check(classes == 9, "Unexpected preexisting synthetic endpoint")
                original = "1147f4af81b28266dfaebbd4b8e3b42b8cfe333e"
                provider = SpatialReleasedFixture(banks[split],s,split,arm,original,source_files[s])
                spec = provider.contract(classes,64,"off",1,original)
                receipt = pin(place/"receipt.json")
                record = _read(receipt)
                prefix = load(place/"prefix.pt")
                check(prefix["registration"] == digest(spec) and all(
                    pin(place/f"{phase}.pt")["sha256"] == sha for phase,sha in record["packet_sha"].items()),
                    "Prior completed case is not bound to the original scope")
            else:
                place.mkdir()
                provider = SpatialReleasedFixture(banks[split],s,split,arm,code["commit"],source_files[s])
                receipt, _, prefix = run_case(place, provider)
            cases[place.name] = receipt
            if arm == "none":
                chronology[classes, s, split] = (provider, prefix)
            print(place.name, flush=True)
        provider, _ = chronology[classes, SOURCES[0], "a"]
        _, other_source = chronology[classes, SOURCES[1], "a"]
        _, other_split = chronology[classes, SOURCES[0], "b"]
        checks[str(classes)] = dict(wrong_source=reject_recovery(provider, other_source, provider.producer),
                                    wrong_split=reject_recovery(provider, other_split, provider.producer),
                                    train_statistics_poison_equal=True)
    return _write(root / "receipt.json", dict(status="passed", schema=SCHEMA, producer=code["commit"],
                   cases=cases, checks=checks, synthetic_updates=12*len(cases), real_updates=0,
                   query_accessed=False, raw_gt_loaded=False));

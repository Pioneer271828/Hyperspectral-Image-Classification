"""Registered synthetic coverage and resumable execution through the native trainer."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from itertools import product
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS
from loadData.observation_sampling import BandMetadata
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import check, pin, _read, _write
from transfer import development_bank as custody, native_scale_training as native
from transfer.development_bank_training import _initial
from transfer.native_released_points import LOSS, MODE
from transfer.p4_coverage import ARMS
from transfer.spatial_coverage_protocol import VIEWS, bank_view, counts, release_statistics
from transfer.spatial_coverage_training import SpatialReleasedFixture, load, shared_source

SCHEMA = "wp05_native_coverage_queue_v1"
SCENES = {"paviaU": 9, "xuzhou": 9, "whu-honghu": 22}
SUPPORT = (512, 2048)
SPLITS = ("a", "b")
TRAIN_BANDS = ("full", "central_gap")
VALIDATION_BANDS = (*VIEWS, "central_gap")
PHASES = ("full", "prefix", "resume", "completed")
PREDECESSOR = {"resume": "prefix", "completed": "resume"}


def design():
    rows = {}
    for scene, support, k, split, arm in product(SCENES, SUPPORT, (1, 5), SPLITS, ARMS):
        model = native.SEEDS[(list(SCENES).index(scene) + SUPPORT.index(support) + SPLITS.index(split)) % 3]
        name = f"{scene}-s{support}-k{k}-{split}-{arm}"
        rows[name] = dict(scene=scene, support_seed=support, k=k, split=split, spectral_arm=arm, model_seed=model)
    return rows


def generated(scene, support_seed, k, split):
    """Create two separated layouts, sampled support, and independently fixed validation."""
    classes = SCENES[scene]
    index = tuple(SCENES).index(scene)
    shape = (30, 60, 9 + index)
    spectral_seed = 20261119 + index
    bands = BandMetadata(f"native-coverage-{scene}", tuple(101 + 7*i for i in range(shape[-1])),
                         tuple(np.linspace(440, 890, shape[-1])), "nm", "approximate",
                         tuple(i != 2 for i in range(shape[-1])))
    shift = SPLITS.index(split) * 2
    rows, arrays = [], {}
    for offset in range(10):
        train = offset < 6
        key = 10 * (offset + 1)
        y, x = (2 + shift, 2 + offset*9) if train else (19 - shift, 3 + (offset-6)*13)
        box = [y, y+6, x, x+7]
        rng = np.random.default_rng([spectral_seed, y, x])
        values = rng.normal(size=(6, 7, shape[-1])).astype(np.float32)
        arrays[key] = values
        rows.append(dict(id=key, role="devtrain" if train else "devval", partition="train" if train else "val",
                         box=box, parent_tile=key, parent_box=box, samples=[],
                         origin={"synthetic_seed": spectral_seed, "array_sha256": custody.array_hash(values)},
                         representation="raw_fp32", blocked=None))
    for role, candidates, budget in (("devtrain", rows[:6], k), ("devval", rows[6:], 3)):
        rng = np.random.default_rng(support_seed if role == "devtrain" else 3031)
        pool = {row["id"]: rng.permutation(42).tolist() for row in candidates}
        absent = {21} if scene == "whu-honghu" else set()
        labels = [c for c in range(1, classes+1) if c not in absent]
        offset = int(rng.integers(len(candidates)))
        for number, label in enumerate(labels * budget):
            row = candidates[(offset+number) % len(candidates)]
            dy, dx = divmod(pool[row["id"]].pop(), 7)
            y, _, x, _ = row["box"]
            row["samples"].append([(y+dy)*shape[1]+x+dx, label])
        for row in candidates:
            row["samples"].sort()
        expected = [0 if c in absent else budget for c in range(1, classes+1)]
        check(counts(candidates, classes) == expected, "Released class budget differs")
    rows.sort(key=custody._row_key)
    geometry = digest(dict(scene=scene, split=split, boxes=[row["box"] for row in rows]))
    stats = release_statistics(arrays, rows, geometry)
    poisoned = {row["id"]: (arrays[row["id"]] if row["role"] == "devtrain"
                           else np.full_like(arrays[row["id"]], np.nan)) for row in rows}
    check(stats == release_statistics(poisoned, rows, geometry), "Validation poisoned the fitted statistics")
    authority = dict(minimum_buffer=3, construction=SCHEMA, geometry=geometry, validation_seed=3031,
                     support_seed=support_seed, k=k, partition=split)
    release = custody.BankRelease(scene, "synthetic", shape, classes, geometry, bands, stats,
                                  {"kind": "synthetic_static_source", "seed": support_seed},
                                  authority, tuple(rows))
    return release, arrays


def prepare(root, code):
    folder = root / "inputs"
    folder.mkdir(exist_ok=True)
    banks, sources = {}, {}
    for scene, support, k, split in product(SCENES, SUPPORT, (1, 5), SPLITS):
        name = f"{scene}-s{support}-k{k}-{split}"
        release, arrays = generated(scene, support, k, split)
        path = folder / name
        if not (path/"bank.json").exists():
            custody.publish(path, folder, release, lambda row: arrays[row["id"]], resume=path.exists())
        bank = dict(root=str(path), sha256=pin(path/"bank.json")["sha256"],
                    registration=pin(path/"registration.json")["sha256"])
        custody.Bank(bank, folder, release).verify_authority(bank)
        banks[name] = bank
        source = f"{scene}-s{support}"
        if source not in sources:
            weights = folder / f"{source}.pt"
            if not weights.exists():
                save_model_checkpoint(shared_source(scene, support), weights)
            sources[source] = dict(weights=pin(weights), sidecar=pin(Path(str(weights)+".json")))
    document = dict(schema=SCHEMA, producer=code["commit"], banks=banks, sources=sources, cases=design(),
                    scope="synthetic_only", query_accessed=False, raw_gt_loaded=False, real_updates=0)
    path = root/"queue.json"
    if path.exists():
        check(_read(pin(path)) == document, "Existing queue does not have this producer or roster")
        return pin(path)
    return _write(path, document)


class CoverageFixture(SpatialReleasedFixture):
    """Admit only the fixed, generated support/space roster to the optimizer."""

    def __init__(self, root, queue, name):
        cfg = queue["cases"][name]
        self.config = deepcopy(cfg)
        scene, support, k, split = (cfg[key] for key in ("scene", "support_seed", "k", "split"))
        key = f"{scene}-s{support}-k{k}-{split}"
        expected, _ = generated(scene, support, k, split)
        bank = custody.Bank(queue["banks"][key], root/"inputs", expected)
        source = queue["sources"][f"{scene}-s{support}"]
        self.bank, self.source_files, self.roster = bank, deepcopy(source), {}
        self.scene, self.classes = scene, SCENES[scene]
        self.source_seed, self.split, self.arm, self.producer = support, split, cfg["spectral_arm"], queue["producer"]

    def model_factory(self, source, arm, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return super().model_factory(source, arm, seed)

    def materialize(self, role, event, producer):
        check(producer == self.producer and event["band"] in VALIDATION_BANDS and event["factor"] == 1,
              "Unregistered producer, band or lattice")
        return bank_view(self.bank, self.bank.role_row(role, event["tile"]), producer, event["band"])

    def validate_observation(self, spec, actual, labels, target, event):
        role = actual.receipt["role"]
        if event not in spec[{"devtrain": "training", "devval": "validation"}[role]]:
            raise ValueError("Native event belongs to a different released role or budget")
        super().validate_observation(spec, actual, labels, target, event)

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        check((classes, seed, arm, factor, producer) ==
              (self.classes, self.config["model_seed"], "off", 1, self.producer),
              "Unexpected coverage case, seed, architecture or native scale")
        self.bank.verify_authority(self.bank.pin)
        chosen = [row["id"] for row in self.bank.role_rows("devtrain") if row["samples"]]
        check(len(chosen) == 6, "Registered budget requires six released training blocks")
        order = np.random.default_rng(seed).permutation(chosen).tolist()
        train = [dict(step=i, tile=tile, factor=1, band=band, profile=None, condition="missing")
                 for i, (band, tile) in enumerate(product(TRAIN_BANDS, order), 1)]
        val = [dict(tile=row["id"], factor=1, band=band, profile=None, condition="missing")
               for row, band in product(self.bank.role_rows("devval"), VALIDATION_BANDS)]
        if not self.roster:
            for role, events in (("devtrain", train), ("devval", val)):
                for event in events:
                    actual, _, targets = self.materialize(role, event, producer)
                    self.roster[self.key(role, event)] = self.evidence(actual, targets)
        source, witness = self.initialization(seed)
        model = self.model_factory(source, arm, seed+7321)
        baseline = model if self.arm == "none" else self._unconditioned(source, seed+7321)
        return dict(format=SCHEMA, producer=producer, classes=classes, seed=seed, arm=arm, factor=factor,
                    supervision=MODE, objective=LOSS, training=train, validation=val, inputs=deepcopy(self.roster),
                    steps=len(train), prefix=4, events=[0, 6, len(train)], learning_rate=.0001,
                    module_seed=seed+7321, rng_seed=seed+15000, registered_graph_scale=50 if self.scene=="paviaU" else 100,
                    initialization=_initial(model, baseline), parameters=sum(p.numel() for p in model.parameters()),
                    case=self.config, bank=self.bank.pin, source=witness, numeric=numeric,
                    selection="first_min_released_point_fused_nll", raw_gt_loaded=False, query_accessed=False, real_updates=0)

    @staticmethod
    def _unconditioned(source, seed):
        from transfer.spectral_spacing_recovery import model_for
        return model_for(source, "none", seed)


@contextmanager
def case_lock(place):
    place.mkdir(exist_ok=True)
    with (place/".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def load_queue(root, code):
    queue = _read(pin(root/"queue.json"))
    check(queue["producer"] == code["commit"] and queue["schema"] == SCHEMA and queue["cases"] == design(),
          "Queue has foreign registration or producer")
    return queue


def _stage(place, spec, phase, provider, stages, execution):
    packet, proof = place/f"{phase}.pt", place/f"{phase}.json"
    previous = stages.get(PREDECESSOR.get(phase))
    parent = None if previous is None else previous["checkpoint"]
    if proof.exists():
        report = _read(pin(proof))
        check(report["registration"] == digest(spec) and report["phase"] == phase
              and report["parent"] == parent and report["checkpoint"] == pin(packet),
              "Persisted phase receipt or checkpoint differs")
        return report
    if packet.exists():
        # A crash between checkpoint publication and its receipt leaves an unaccepted stage.
        leftover = pin(packet)
        orphan = place/f"{phase}.orphan-{leftover['sha256']}.pt"
        if orphan.exists():
            check(pin(orphan)["sha256"] == leftover["sha256"], "Conflicting interrupted checkpoint")
            packet.unlink()
        else:
            packet.rename(orphan)
    prior = None if parent is None else load(Path(parent["path"]))
    report = native.execute(spec, phase, packet, spec["producer"], prior, execution=execution, released=provider)
    report.update(registration=digest(spec), parent=parent, checkpoint=pin(packet))
    _write(proof, report)
    return report


def validation_rows(evaluated):
    result = {}
    for band in VALIDATION_BANDS:
        rows = [row for row in evaluated["rows"] if row["band"] == band]
        result[band] = dict(mode=rows[0]["mode"], band_ids=rows[0]["band_ids"],
                            graph=digest([row["graph"] for row in rows]),
                            heads={head: dict(n=sum(row["scores"][head]["count"] for row in rows),
                                confusion=sum(np.asarray(row["scores"][head]["point_confusion"]) for row in rows).tolist(),
                                nll=sum(row["scores"][head]["nll_sum"] for row in rows)) for head in HEADS})
    return result


def run_case(root, code, name, *, execution=None):
    started = perf_counter()
    queue = load_queue(root, code)
    provider = CoverageFixture(root, queue, name)
    numeric = None if execution is None else execution.numeric
    spec = provider.contract(provider.classes, provider.config["model_seed"], "off", 1, code["commit"], numeric=numeric)
    device = "cpu" if numeric is None else numeric.get("physical_uuid") or "cpu-device"
    group = root/"cases"/device
    group.mkdir(parents=True, exist_ok=True)
    place = group/name
    with case_lock(place):
        receipt = place/"receipt.json"
        phases = {}
        for phase in PHASES:
            phases[phase] = _stage(place, spec, phase, provider, phases, execution)
        if receipt.exists():
            result = _read(pin(receipt))
            check(result["registration"] == digest(spec) and result["stages"] == phases,
                  "Completed result does not match its registered execution")
            return dict(case=name, reused=True, new_updates=0, receipt=pin(receipt))
        packets = {phase: load(Path(row["checkpoint"]["path"])) for phase, row in phases.items()}
        full, prefix, resume, completed = (packets[key] for key in PHASES)
        check(native._exact(full, resume) and native._exact(full, completed)
              and native._exact(full["history"][:spec["prefix"]], prefix["history"]),
              "Native resumed training differs from the uninterrupted checkpoint")
        packet = full["selected"]
        runner = native.Run(spec, code["commit"], execution=execution, released=provider)
        runner.model.load_state_dict(packet["state"], strict=True)
        evaluated = runner.evaluate(packet["step"], detailed=True)
        check(evaluated["fused_nll"] == packet["fused_nll"], "Selected frozen validation changed")
        changed = sorted({key for row in full["history"] for key, value in row["fresh"].items()
                          if value["changed"] and (value["gradient_l1"] or 0) > 0})
        # Binary coverage deliberately has no physical signal when metadata is missing.
        check(provider.arm == "none" or changed, "Coverage arm made no effective update")
        require_frozen_code(code)
        result = dict(**provider.config, registration=digest(spec), bank=provider.bank.pin["sha256"],
                      fit_statistics=provider.bank.release.statistics["statistics_hash"],
                      inputs=digest(spec["inputs"]), source=spec["source"], parameters=spec["parameters"],
                      roster={role: counts(provider.bank.role_rows(role), provider.classes) for role in ("devtrain", "devval")},
                      selected_step=packet["step"], selection_nll=packet["fused_nll"],
                      validation=validation_rows(evaluated), stages=phases,
                      resume_bitwise_equal=True, completed_bitwise_equal=True, changed_coverage=changed,
                      optimizer_updates=sum(stage["updates"] for stage in phases.values()),
                      endpoint_validation_seconds=runner.timing["validation_seconds"],
                      elapsed_seconds=perf_counter()-started, real_updates=0, query_accessed=False)
        return dict(case=name, reused=False, receipt=_write(receipt, result))


def _cpu_job(args):
    torch.set_num_threads(1)
    return run_case(*args)


def run_queue(root, code, names, workers, *, execution=None):
    if execution is not None:
        check(workers == 1 and len(names) == 1, "CUDA qualification runs one case under one physical lease")
        return [run_case(root, code, names[0], execution=execution)]
    check(type(workers) is int and 1 <= workers <= 16, "CPU workers must be between 1 and 16")
    tasks = [(root, code, name) for name in names]
    if workers == 1:
        return [_cpu_job(args) for args in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = []
        for row in pool.map(_cpu_job, tasks, chunksize=1):
            print(row["case"], "reused" if row["reused"] else "completed", flush=True)
            result.append(row)
        return result

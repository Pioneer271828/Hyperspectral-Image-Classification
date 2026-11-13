"""A single registered CPU queue for the native support and WP07 matrices."""

from dataclasses import asdict
import io
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from research_runtime import external_directory, tensor_hash
from transfer import registered_development as dev, xuzhou_multiscale as stages
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import prepare_scene
from transfer.g3_training import state_hash
from transfer.multiscale_run import MultiscaleRun
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_support import bank_digest, make_bank, make_selector
from transfer.native_support_run import NativeSupportRun
from transfer.regional_support_evolution import Schedule


ISSUE = 447
FIXTURES = {"nine8": (9, 8), "nine11": (9, 11), "twentytwo13": (22, 13)}
SEEDS = (64, 512, 2048)
WP08 = ("off", "local", "remote", "tighten", "joint", "random", "fixed")
WP07 = {
    "graph2": MultiscaleTrial(graph_multiscale="fixed", graph_radius=2),
    "graph_metadata": MultiscaleTrial(graph_multiscale="metadata"),
    "graph_joint": MultiscaleTrial(graph_multiscale="joint"),
    "both_metadata": MultiscaleTrial(cnn_multiscale="metadata", graph_multiscale="metadata"),
    "both_joint": MultiscaleTrial(cnn_multiscale="joint", graph_multiscale="joint"),
    "both_fallback": MultiscaleTrial(cnn_multiscale="joint", graph_multiscale="joint"),
}
MAXIMUM_UPDATES = 6624
SCORER_SEED = 7321
RECOVERY_PREFIX = 7


class OriginalControl(CurriculumRun):
    """The off arm delegates the actual forward, optimizer and validation to CurriculumRun."""

    def __init__(self, *args, **kwargs):
        self.timings = {}
        super().__init__(*args, **kwargs)

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        output = super()._training_forward(tile, step, extra)
        self.timings.setdefault("forward", []).append(perf_counter() - started)
        return output

    def _history_extensions(self, outputs):
        return {"train_probabilities": [tensor_hash(head) for head in outputs[:3]]}

    def _history_extension_keys(self):
        return {"train_probabilities"}


def _fixture(root, classes, bands, model_seed, *, trusted):
    root.mkdir(parents=True)
    plan = make_tile_plan((121, 103), 38, 1, 2026)
    labels = np.zeros(plan["shape"], dtype=np.int64)
    tiles = [t for t in plan["tiles"] if t["partition"] == "train"][:3]
    if len(tiles) != 3:
        raise ValueError("Fixture requires two training and one independent validation block")
    for tile in tiles:
        view = labels[tile_slice(tile)]
        view[...] = np.resize(np.arange(1, classes + 1), view.shape)
    record = make_development(labels, plan, list(range(1, classes + 1)))
    wavelength = tuple(np.linspace(460., 850., bands))
    metadata = BandMetadata("synthetic", tuple(range(bands)), wavelength, "nm",
                            "verified" if trusted else "approximate", (True,) * bands,
                            (12.,) * bands if trusted else None, "nm" if trusted else None)
    rng = np.random.default_rng(904 + bands)
    cube = rng.normal(size=(*plan["shape"], bands)).astype(np.float32)
    # Inject a deterministic spectral class signal on synthetic released labels.
    for tile in tiles:
        sl = tile_slice(tile)
        cube[sl][..., 0] += labels[sl] / classes
    scene = prepare_scene(cube, metadata, plan, record)
    scene.dataset = "synthetic"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        baseline = MultiDatasetVCASAHGCN({scene.dataset: classes}, latent_channels=8, adapter_embed_dim=8,
            locality_window=None, graph_mode="learned", norm_type="group_norm").eval()
    baseline.backbone.adapter.cache_attention = False
    source = root / "source.pt"
    save_model_checkpoint(baseline, source)
    manifest = dev.export_development(scene, record, root / "development")
    return dict(source=dev._pin(source), sidecar=dev._pin(str(source) + ".json"),
                development=manifest, trusted=trusted), baseline, scene, record


def _trial(arm, support_seed):
    mode = {"local": "local", "remote": "joint", "tighten": "joint",
            "joint": "joint", "random": "random", "fixed": "joint"}[arm]
    end = 6 if arm in ("tighten", "joint", "random", "fixed") else 10
    return dict(mode=mode, schedule=asdict(Schedule(10, end, 3, 7,
                6 if arm in ("joint", "random", "fixed") else 0, 48, pool=5)),
                support_seed=support_seed, fixed=arm == "fixed")


def _stages(budget):
    return dict(full=(0, budget, None), prefix=(0, RECOVERY_PREFIX, None),
                resume=(RECOVERY_PREFIX, budget, "prefix"), completed=(budget, budget, "resume"))


def _load_pt(pin):
    return torch.load(io.BytesIO(dev._read(pin)), map_location="cpu", weights_only=True)


def _factory(case, data, arm, *, donor=None):
    baseline, scene, record, plan, bank = data
    settings = dict(plan=plan, policy=None, lr=1e-4, torch_seed=case["torch_seed"],
                    expected_plan_hash=plan.plan_hash, expected_policy_hash=None)
    if case["work_package"] == "WP07":
        trial = WP07[arm]
        conditions = {"coverage_fraction": .8, "gsd_m": 2.5} if arm != "both_fallback" else None
        extras = dict(synthetic_conditions=conditions)
        cls = MultiscaleRun
    elif arm == "off":
        trial = None
        extras = {}
        cls = OriginalControl
    else:
        trial = _trial(arm, case["support_seed"])
        extras = dict(bank=bank, donor=donor)
        cls = NativeSupportRun
    scorer_seed = case["scorer_seed"]

    def create(*_):
        if cls is OriginalControl:
            return cls(baseline, scene, record, expected_source_state=state_hash(baseline), **settings)
        return cls(baseline, scene, record, trial=trial, selector_seed=scorer_seed, **extras, **settings)

    def load(baseline, scene, record, path, *, selector_seed, **kwargs):
        if cls is OriginalControl:
            kwargs.pop("trial")
            return cls.load(baseline, scene, record, path, policy=None, **kwargs)
        return cls.load(baseline, scene, record, path, selector_seed=scorer_seed, **extras, **kwargs)

    return create, SimpleNamespace(load=load)


def _case_data(case, *, bank=True):
    scene, record, _ = dev.load_development(case["development"])
    if (scene.dataset != "synthetic" or record["kind"] != "oracle_spatial_development_v1"
            or scene.classes != case["classes"] or len(scene.bands.band_ids) != case["bands"]):
        raise ValueError("Campaign requires its registered synthetic fixture")
    baseline = dev._load_source(case["source"], case["sidecar"])
    plan = ExposurePlan(scene, record, case["exposure"], expected_plan_hash=case["exposure"]["plan_hash"])
    if bank and case.get("bank"):
        contexts = _load_pt(case["bank"])
        if bank_digest(contexts) != case["bank_hash"]:
            raise ValueError("Context bank identity differs")
    else:
        contexts = None
    return baseline, scene, record, plan, contexts


def _arms(case):
    return {arm: (WP07[arm] if case["work_package"] == "WP07" else
                  (None if arm == "off" else _trial(arm, case["support_seed"])))
            for arm in case["contracts"]}


def register(root, code):
    root = external_directory(root)
    root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for work_package, fixtures, seeds, budget in (
            ("WP08", FIXTURES, SEEDS, 48), ("WP07", ("nine8", "twentytwo13"), (64,), 24)):
        for name in fixtures:
            classes, bands = FIXTURES[name]
            for model_seed in seeds:
                for trusted in ((False,) if work_package == "WP08" else (True, False)):
                    key = f"{work_package.lower()}-{name}-{model_seed}" + ("-fallback" if work_package == "WP07" and not trusted else "")
                    pins, baseline, scene, record = _fixture(root / "inputs" / key, classes, bands, model_seed, trusted=trusted)
                    plan = ExposurePlan.freeze(scene, record, updates=budget, seed=model_seed,
                        minimum_class_blocks={str(c): 2 for c in record["classes"]}, minimum_support_visits=1)
                    plan.require_eligible()
                    if len(plan.to_dict()["sampler"]["tile_ids"]) != 2:
                        raise ValueError("Fixture training support must cover exactly two released blocks")
                    case = dict(**pins, work_package=work_package, classes=classes, bands=bands,
                        model_seed=model_seed, sampler_seed=model_seed, torch_seed=15000 + model_seed,
                        support_seed=19000 + model_seed, scorer_seed=SCORER_SEED, budget=budget, exposure=plan.to_dict())
                    contexts = None
                    if work_package == "WP08":
                        timing = {}
                        started = perf_counter()
                        contexts = make_bank(baseline, scene, record, plan, pins["development"]["sha256"], SCORER_SEED, timing)
                        path = root / "inputs" / key / "bank.pt"
                        CurriculumRun._write(contexts, path)
                        case.update(bank=dev._pin(path), bank_hash=bank_digest(contexts),
                                    bank_build_seconds=perf_counter() - started,
                                    bank_capture_timing={k: sum(v) for k, v in timing.items()})
                    arms = (WP08[:-1] if work_package == "WP08" else
                            tuple(WP07)[:-1] if trusted else ("both_fallback",))
                    data = baseline, scene, record, plan, contexts
                    case["contracts"] = {arm: _factory(case, data, arm)[0]().contract_hash for arm in arms}
                    cases[key] = case
    declared = sum(case["budget"] * 2 * (len(case["contracts"]) + (case["work_package"] == "WP08"))
                   for case in cases.values())
    if declared != MAXIMUM_UPDATES:
        raise ValueError("Synthetic optimizer envelope differs")
    return dev._json_new(root / "registration.json", dict(version=2, issue=ISSUE, scope="synthetic",
        code={k: code[k] for k in ("commit", "branch", "source_hash")}, cases=cases,
        cpu_workers=1, maximum_updates=declared, prefix=RECOVERY_PREFIX, query_accesses=0, real_updates=0))


def pinned_producer(pin, expected_commit):
    value = json.loads(dev._read(pin))["code"]
    if value["commit"] != expected_commit:
        raise ValueError("Evidence source differs from the frozen producer")
    return value


def read_registration(pin, code):
    value = json.loads(dev._read(pin))
    if (value["version"] != 2 or value["scope"] != "synthetic" or value["maximum_updates"] != MAXIMUM_UPDATES
            or value["code"] != {k: code[k] for k in ("commit", "branch", "source_hash")}):
        raise ValueError("Wrong frozen native campaign or source")
    return Path(pin["path"]).parent, value


def _read_stage(folder, pin, case, arm, phase, prior):
    return stages.read_stage(folder, pin, arm, phase, prior, arms=_arms(case), stages=_stages(case["budget"]))


def _donor(folder, pin, case, bank):
    donor_path = folder / "donor.json"
    status, receipt = _read_stage(folder, pin, case, "joint", "full", None)
    prefix = _read_stage(folder, pin, case, "joint", "prefix", None)[1]
    resume = _read_stage(folder, pin, case, "joint", "resume", prefix)[1]
    completed = _read_stage(folder, pin, case, "joint", "completed", resume)[1]
    packet = _load_pt(status["checkpoint"])
    trial = _trial("joint", case["support_seed"])
    snapshots = {}
    started = perf_counter()
    for key, entry in bank.items():
        if entry["context"]["role"] == "devtrain":
            selector = make_selector(entry, trial, trial["support_seed"])
            selector.advance(case["budget"])
            latest = next(row["support"]["snapshot"] for row in reversed(packet["history"]) if str(row["tile_id"]) == key)
            selector.authenticate(latest)
            snapshots[key] = selector.snapshot()
    payload = dict(donor=receipt, completed=completed, checkpoint=status["checkpoint"], snapshots=snapshots)
    if donor_path.exists():
        existing = json.loads(dev._read(dev._pin(donor_path)))
        if existing["contract"] != payload:
            raise ValueError("Fixed graph donor changed")
    else:
        dev._json_new(donor_path, dict(contract=payload, construction_seconds=perf_counter() - started))
    return payload


def open_case(root, pin, registration, key, arm):
    case = dict(registration["cases"][key])
    case["contracts"] = dict(case["contracts"])
    folder = root / "matrix" / key
    data = _case_data(case)
    donor = None
    if arm == "fixed":
        if case["work_package"] != "WP08":
            raise ValueError("Only the native matrix has a fixed donor")
        donor = _donor(folder, pin, case, data[-1])
        case["contracts"][arm] = _factory(case, data, arm, donor=donor)[0]().contract_hash
    if arm not in case["contracts"]:
        raise ValueError("Arm was never registered")
    return folder, case, data, donor


def execute(pin, code, key, arm, phase, prior):
    import fcntl

    root, registration = read_registration(pin, code)
    with (root / "cpu-queue.lock").open("ab") as queue:
        try:
            fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This registration already has a CPU stage running") from error
        return _execute_locked(root, registration, pin, key, arm, phase, prior)


def _execute_locked(root, registration, pin, key, arm, phase, prior):
    if not (root / "preflight.json").exists():
        raise ValueError("Synthetic preflight must succeed before the first optimizer step")
    evidence = json.loads(dev._read(dev._pin(root / "preflight.json")))
    if evidence["registration"] != pin or evidence["maximum_updates"] != MAXIMUM_UPDATES:
        raise ValueError("Synthetic preflight identity differs")
    folder, case, data, donor = open_case(root, pin, registration, key, arm)
    baseline, scene, record, plan, _ = data
    create, load = _factory(case, data, arm, donor=donor)
    return stages.execute_stage(folder, pin, arm, phase, prior, value=case, baseline=baseline,
        scene=scene, record=record, plan=plan, arms=_arms(case), stages=_stages(case["budget"]),
        run_factory=create, run_type=load)


def preflight(pin, code):
    root, registration = read_registration(pin, code)
    rows = {}
    for key, case in registration["cases"].items():
        data = _case_data(case)
        baseline, scene, record, plan, bank = data
        rng = torch.get_rng_state().clone()
        for arm, expected in case["contracts"].items():
            run = _factory(case, data, arm)[0]()
            if run.contract_hash != expected or not torch.equal(rng, torch.get_rng_state()):
                raise ValueError("Preflight initialization changed or consumed global RNG")
            if any(not torch.equal(value, run.source.state_dict()[name]) for name, value in baseline.state_dict().items()):
                raise ValueError("Shared initialization differs")
            run._evaluate(run.model)
        rows[key] = dict(exposure=plan.require_eligible(), bank=case.get("bank_hash"),
                         source=state_hash(baseline), bands=scene.bands.fingerprint)
    return dev._json_new(root / "preflight.json", dict(registration=pin, maximum_updates=MAXIMUM_UPDATES,
                                                       updates=0, rows=rows))

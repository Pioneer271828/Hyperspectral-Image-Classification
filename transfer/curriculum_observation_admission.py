"""Admit a fixed real-band curriculum without starting real optimization."""

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from compression.registered_precision_recovery import COMPATIBILITY_PATHS, stages
from loadData.observation_sampling import replay, subset_statistics
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest
from research_runtime import ROOT, external_directory, require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_real import FORMAT as SAMPLE_FORMAT, runtime, source_inputs
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import _forward, state_hash
from transfer.observation_curriculum import CurriculumPolicy, ORDERS
from transfer.observation_policy import ObservationPolicy, default_conditions
from transfer.xuzhou_multiscale import LR, SEED, execute_stage


FORMAT = "wp06_curriculum_observation_admission_v1"
CONDITIONS = ("full", "random_75", "random_50", "random_25")
TRAIN_SEED, VALIDATION_SEED, ORDER_SEED = 512, 3031, 2048


def bind_sample(pin, code):
    require_frozen_code(code)
    reg = admission.read(pin)
    if (reg["format"] != SAMPLE_FORMAT or reg["runtime"] != runtime()
            or pin["path"] != str(Path(reg["root"]) / "registration.json")
            or any(reg[k] for k in ("optimizer_updates", "query", "gt"))):
        raise ValueError("Original registered sample experiment required")
    commit = reg["commit"]
    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid registered sample producer commit")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, code["commit"]], check=True)
    changed = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", commit, code["commit"], "--",
                                      *COMPATIBILITY_PATHS, "transfer/curriculum_real.py", "transfer/curriculum_real_audit.py"], text=True)
    if changed:
        raise ValueError("Frozen sample producer paths differ")
    design, model, scene, record, plans = source_inputs(reg["request"])
    if any(reg[k] != v for k, v in design.items()):
        raise ValueError("Sample witness source, protocol or recovery schedule changed")
    return reg, model, scene, record, plans


def _tile_guard(scene, record, tile, phase):
    role = {"training": "devtrain", "validation": "devval"}.get(phase)
    idx = tile.get("id")
    if (role is None or type(idx) is not int or not 0 <= idx < len(record["roles"])
            or tile != scene.plan["tiles"][idx] or record["roles"][idx]["role"] != role):
        raise ValueError("Observation input is outside its registered development role")
    return tile["box"]


class RestrictedObservation(CurriculumPolicy):
    """Use original replay records with only selected raw columns and statistics."""

    def view(self, scene, record, tile, *, phase, step=None, condition="full"):
        y0, y1, x0, x1 = _tile_guard(scene, record, tile, phase)
        identity = self._value["identity"]
        if (scene.bands.fingerprint != identity["metadata_hash"] or scene.stats["statistics_hash"] != identity["statistics_hash"]
                or record["validation_hash"] != identity["development_hash"]):
            raise ValueError("Frozen metadata, development role or statistics differs")
        if phase == "training":
            self.training_record(step)
            frozen = self._value["training"][step-1]
        else:
            if step is not None or condition not in self.conditions:
                raise ValueError("Unregistered observation validation condition")
            frozen = self._value["validation"][condition]
        positions = frozen["selected_positions"]
        stub = replace(self._stub(scene.bands, scene.plan["plan_hash"]), partition=tile["partition"])
        metadata = replay(stub, scene.bands, frozen)
        stats = subset_statistics(scene.stats, scene.bands, frozen)
        raw = np.array(scene.cube[y0:y1, x0:x1, positions], dtype=np.float32, copy=True)
        low, high = (np.asarray(stats[k], dtype=np.float32) for k in ("minimum", "maximum"))
        if not np.isfinite(raw).all() or not np.isfinite([low, high]).all() or (high < low).any():
            raise ValueError("Invalid selected-band raw values or train statistics")
        values = (raw-low) / np.where(high > low, high-low, 1)
        h, w = scene.plan["shape"]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        coordinates = np.stack((yy/max(h-1, 1), xx/max(w-1, 1)), axis=-1).astype(np.float32)
        ids = yy*w+xx
        for array in (values, coordinates, ids):
            array.setflags(write=False)
        return replace(metadata, observation=PartitionObservation(tile["partition"], values, coordinates, ids, scene.plan["plan_hash"]))


def _verify_view(scene, record, tile, phase, view, frozen):
    y0, y1, x0, x1 = _tile_guard(scene, record, tile, phase)
    positions = frozen["selected_positions"]
    actual = view.observation
    bands = scene.bands.subset(positions)
    if (actual.plan_hash != scene.plan["plan_hash"] or actual.partition != tile["partition"]
            or bands != view.bands or bands.fingerprint != frozen["selected_metadata_hash"]
            or list(bands.band_ids) != frozen["selected_band_ids"]
            or actual.spectral.shape != (y1-y0, x1-x0, len(positions)) or actual.spectral.dtype != np.float32):
        raise ValueError("Forward view contains the wrong original band IDs or metadata")
    for selected, p in enumerate(positions):
        lower, upper = (np.float32(scene.stats[k][p]) for k in ("minimum", "maximum"))
        raw = scene.cube[y0:y1, x0:x1, p].astype(np.float32)
        scale = np.float32(1.) if upper == lower else upper-lower
        expected = (raw-lower)/scale
        if not np.array_equal(actual.spectral[..., selected], expected):
            raise ValueError("Forward view was normalized with the wrong original per-band statistics")
    h, w = scene.plan["shape"]
    coords = np.asarray([[(y/max(h-1, 1), x/max(w-1, 1)) for x in range(x0, x1)] for y in range(y0, y1)], dtype=np.float32)
    ids = np.asarray([list(range(y*w+x0, y*w+x1)) for y in range(y0, y1)])
    if not np.array_equal(actual.global_ids, ids) or not np.array_equal(actual.coordinates, coords):
        raise ValueError("Forward geometry differs from the registered block")
    return torch.tensor(actual.spectral), torch.tensor(coords)


def audited_forward(model, scene, record, tile, view, frozen, *, phase="training"):
    spectral, coords = _verify_view(scene, record, tile, phase, view, frozen)
    backbone = model.backbone
    if (backbone.graph_mode != "learned" or backbone.graph_assignment_mode != "learned"
            or backbone.coverage_modulation is not None or backbone.adapter.cache_attention):
        raise ValueError("Original uncached learned graph is required")
    expected_waves = torch.tensor(view.bands.wavelengths, dtype=torch.float32)
    expected_fwhm = None if view.bands.fwhm is None else torch.tensor(view.bands.fwhm, dtype=torch.float32)
    observed = {}
    hits = Counter()

    def input_adapter(_, args, kwargs):
        hits["adapter"] += 1
        fwhm = kwargs["fwhm"]
        if (not torch.equal(args[0], spectral) or not torch.equal(kwargs["wavelengths"], expected_waves)
                or kwargs["wavelength_unit"] != view.bands.wavelength_unit
                or (fwhm is None) != (expected_fwhm is None)
                or fwhm is not None and not torch.equal(fwhm, expected_fwhm)):
            raise ValueError("Adapter received out-of-policy bands or metadata")

    def latent(_, args, output):
        observed["latent"] = output.detach().clone()

    def enter_denoise(_, args):
        hits["denoise"] += 1
        target = torch.cat((observed["latent"], coords), dim=2).permute(2, 0, 1).unsqueeze(0)
        if not torch.equal(target, args[0]):
            raise ValueError("Non-admitted features entered the graph input path")

    def leave_denoise(_, args, output):
        observed["graph"] = output.detach().squeeze(0).permute(1, 2, 0).clone()

    graph = backbone.learnable_graph
    assignment = graph._build_assignment

    def enter_graph(features, *args, **kwargs):
        hits["graph"] += 1
        if "graph" not in observed or not torch.equal(features, observed["graph"]):
            raise ValueError("Graph assignment received features outside the selected-band denoising path")
        observed["graph_sha256"] = tensor_hash(features)
        return assignment(features, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(backbone.adapter.register_forward_pre_hook(input_adapter, with_kwargs=True))
        stack.enter_context(backbone.adapter.register_forward_hook(latent))
        stack.enter_context(backbone.CNN_denoise[0].register_forward_pre_hook(enter_denoise))
        stack.enter_context(backbone.CNN_denoise[-1].register_forward_hook(leave_denoise))
        stack.enter_context(patch.object(graph, "_build_assignment", enter_graph))
        outputs = _forward(model, scene, tile, observation_view=view)
    if hits != {"adapter": 1, "denoise": 1, "graph": 1} or any(not torch.isfinite(p).all() for p in outputs[:3]):
        raise ValueError("Selected bands did not pass all original forward boundaries")
    evidence = dict(band_count=len(view.bands.band_ids), band_ids=list(view.bands.band_ids),
                    spectral_sha256=tensor_hash(spectral), metadata_sha256=view.bands.fingerprint,
                    graph_features_sha256=observed["graph_sha256"], calls=dict(hits),
                    three_heads=[tensor_hash(p) for p in outputs[:3]])
    return outputs, evidence


def observation_policies(scene, record, budget):
    configs = default_conditions(scene.bands)
    ordered = [configs[k] for k in CONDITIONS]
    base = ObservationPolicy.freeze(scene, record, updates=budget, training_configs=ordered,
                                    train_seed=TRAIN_SEED, validation_seed=VALIDATION_SEED)
    severity = {digest(config): i for i, config in enumerate(ordered)}
    return {order: RestrictedObservation(CurriculumPolicy.freeze(base, order=order, severity=severity,
                                                                   seed=ORDER_SEED).to_dict()) for order in ORDERS}


def cross_exposure(plan, policy, scene, record):
    frozen = policy.to_dict()["training"]
    labels = dict(zip(record["train_support"]["indices"], record["train_support"]["labels"], strict=True))
    by_config = {}
    wavelengths = Counter()
    rows = []
    for step, observation in enumerate(frozen, 1):
        event = plan.event(step)
        key = digest(observation["config"])
        if key not in by_config:
            by_config[key] = dict(config=observation["config"], forwards=0, band_exposures=0, pixels_times_bands=0,
                                  tiles=Counter(), support=Counter(), class_blocks=Counter(), class_label_visits=Counter())
        bucket = by_config[key]
        bands = len(observation["selected_positions"])
        tile = scene.plan["tiles"][event["tile_id"]]
        y0, y1, x0, x1 = tile["box"]
        ids = event["support_ids"]
        visits = [str(labels[i]) for i in ids]
        bucket["forwards"] += 1
        bucket["band_exposures"] += bands
        bucket["pixels_times_bands"] += (y1-y0)*(x1-x0)*bands
        bucket["tiles"][str(tile["id"])] += 1
        bucket["support"].update(str(i) for i in ids)
        bucket["class_blocks"].update(set(visits))
        bucket["class_label_visits"].update(visits)
        wavelengths[str(bands)] += 1
        rows.append(dict(**event, config_hash=key, record_hash=observation["record_hash"], retained_bands=bands))
    return dict(by_config=by_config, retained_band_histogram=dict(wavelengths), events=rows,
                record_multiset=digest(sorted(v["record_hash"] for v in frozen)),
                train_forwards=len(rows), train_band_exposures=sum(r["retained_bands"] for r in rows),
                validation_forwards=len(rows)*sum(t["role"] == "devval" for t in record["roles"]),
                train_pixel_band_exposures=sum(r["pixels_times_bands"] for r in by_config.values()))


class ObservationRun(CurriculumRun):
    FORMAT = "vca-wp06-registered-observation-v1"
    FAILURE_FORMAT = "vca-wp06-observation-failure-v1"

    def __init__(self, source, scene, record, *, trial, **kwargs):
        self.trial = trial
        self.timings = defaultdict(list)
        super().__init__(source, scene, record, **kwargs)
        self.observation_policy = RestrictedObservation(self.observation_policy.to_dict())
        self.frozen_observations = self.observation_policy.to_dict()["training"]

    def _contract_extensions(self):
        return dict(observation_trial=self.trial, observation_boundary="subset_then_normalize_then_learned_graph_v1")

    def _training_forward(self, tile, step, extra):
        outputs, self.forward_evidence = audited_forward(self.model, self.scene, self.record, tile, extra["observation_view"],
                                                         self.frozen_observations[step-1])
        return outputs

    def _history_extensions(self, outputs):
        return {"forward_evidence": self.forward_evidence}

    def _history_extension_keys(self):
        return {"forward_evidence"}

    def _validate_history_extensions(self, row):
        v = row["forward_evidence"]
        r = self.frozen_observations[row["step"]-1]
        hashes = (v["spectral_sha256"], v["graph_features_sha256"], *v["three_heads"])
        if (v["band_ids"] != r["selected_band_ids"] or v["band_count"] != len(r["selected_positions"])
                or v["metadata_sha256"] != r["selected_metadata_hash"]
                or v["calls"] != {"adapter": 1, "denoise": 1, "graph": 1} or len(v["three_heads"]) != 3
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
            raise ValueError("Observation history does not authenticate the actual forward path")

    @classmethod
    def load(cls, source, scene, record, path, *, plan, trial, selector_seed, expected_sha256, expected_contract_hash):
        case, policy = trial
        if selector_seed != SEED:
            raise ValueError("Invalid generic stage selector")
        return super().load(source, scene, record, path, plan=plan, policy=None, observation_policy=policy,
                            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
                            constructor_kwargs={"trial": case})


def new_run(source, scene, record, plan, trial):
    case, policy = trial
    return ObservationRun(source, scene, record, trial=case, plan=plan, policy=None, lr=LR, torch_seed=case["seed"],
                          expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
                          expected_policy_hash=None, observation_policy=policy, expected_observation_hash=policy.policy_hash)


def _design(sample_pin, code):
    sample, source, scene, record, plans = bind_sample(sample_pin, code)
    policies = observation_policies(scene, record, sample["budget"])
    cases = [dict(case=f"s{seed}-{order}", seed=seed, order=order) for seed in sample["seeds"] for order in ORDERS]
    trails = {c["case"]: cross_exposure(plans[str(c["seed"])], policies[c["order"]], scene, record) for c in cases}
    for seed in sample["seeds"]:
        matched = [trails[c["case"]] for c in cases if c["seed"] == seed]
        signature = lambda r: (r["record_multiset"], r["retained_band_histogram"], r["train_band_exposures"],
                               [(e["tile_id"], e["anchor"], e["support_ids"]) for e in r["events"]])
        if any(signature(r) != signature(matched[0]) for r in matched):
            raise ValueError("Observation permutation changed the marginal multiset or sample sequence")
    return sample, source, scene, record, plans, policies, cases, trails


def register_observation(sample_pin, out, code):
    sample, source, scene, record, plans, policies, cases, trails = _design(sample_pin, code)
    out = external_directory(out)
    inputs = [sample_pin, sample["teacher"], *sample["protected"].values()]
    if out.exists() or any(out == Path(p["path"]) or out in Path(p["path"]).parents or out == Path(p["path"]).parent for p in inputs):
        raise ValueError("Fresh observation output outside the protected source required")
    trials = {case["case"]: (case, policies[case["order"]]) for case in cases}
    contracts = {name: new_run(source, scene, record, plans[str(trial[0]["seed"])], trial).contract_hash for name, trial in trials.items()}
    probes = {}
    for name, (case, policy) in trials.items():
        first = {}
        for event in trails[name]["events"]:
            first.setdefault(event["config_hash"], event)
        verified = []
        training = policy.to_dict()["training"]
        with torch.no_grad():
            for event in first.values():
                step = event["step"]
                tile = scene.plan["tiles"][event["tile_id"]]
                view = policy.view(scene, record, tile, phase="training", step=step)
                _, actual = audited_forward(source, scene, record, tile, view, training[step-1])
                verified.append(dict(step=step, tile=tile["id"], record_hash=event["record_hash"], **actual))
        probes[name] = verified
    if state_hash(source) != sample["teacher_state"]:
        raise ValueError("Zero-update observation probes changed the source")
    admission.check_pins(inputs)
    require_frozen_code(code)
    out.mkdir()
    policy_pins = {order: dev._json_new(out / f"{order}.json", policy.to_dict()) for order, policy in policies.items()}
    return dev._json_new(out / "registration.json", dict(format=FORMAT, root=str(out), scope=sample["scope"],
        sample_registration=sample_pin, source_commit=sample["commit"], state=sample["teacher_state"],
        commit=code["commit"], source_hash=code["source_hash"], runtime=runtime(), protocol=sample["protocol"],
        policies=policy_pins, policy_hashes={k: v.policy_hash for k, v in policies.items()},
        conditions=list(CONDITIONS), condition_rule="full_and_three_registered_keep_fractions",
        sampling_seeds=dict(train=TRAIN_SEED, validation=VALIDATION_SEED, order=ORDER_SEED),
        metadata_quality=scene.bands.wavelength_quality, original_band_ids=list(scene.bands.band_ids),
        statistics_sha256=scene.stats["statistics_hash"], cases=cases, budget=sample["budget"], prefix=sample["prefix"],
        coverage=trails, contracts=contracts, marginal_match=True, joint_match={
            str(seed): all(trails[f"s{seed}-{order}"]["by_config"] == trails[f"s{seed}-fixed"]["by_config"] for order in ORDERS)
            for seed in sample["seeds"]},
        preflight_probes=probes, preflight_forwards=sum(map(len, probes.values())), selected_observation_role="devtrain_image_only",
        original_gt_reads=0, query=0, optimizer_updates=0, real_training="NOT_RUN"))


def execute_synthetic(pin, code, case, stage, prior):
    require_frozen_code(code)
    reg = admission.read(pin)
    root = external_directory(reg["root"])
    if (reg["format"] != FORMAT or reg["scope"] != "synthetic" or reg["commit"] != code["commit"]
            or reg["source_hash"] != code["source_hash"] or reg["runtime"] != runtime()
            or pin["path"] != str(root/"registration.json")):
        raise ValueError("Only registered synthetic observation execution is authorized")
    sample, source, scene, record, plans, policies, cases, trails = _design(reg["sample_registration"], code)
    if (cases != reg["cases"] or trails != reg["coverage"] or sample["scope"] != "synthetic"
            or (reg["budget"], reg["prefix"]) != (sample["budget"], sample["prefix"])
            or set(reg["contracts"]) != {c["case"] for c in cases}):
        raise ValueError("Frozen observation case or cross exposure differs")
    for order, policy in policies.items():
        saved = reg["policies"][order]
        if (saved["path"] != str(root / f"{order}.json") or admission.read(saved) != policy.to_dict()
                or reg["policy_hashes"][order] != policy.policy_hash):
            raise ValueError("Frozen observation policy file or hash differs")
    trials = {c["case"]: (c, policies[c["order"]]) for c in cases}
    if case not in trials:
        raise ValueError("Unknown observation case")
    return execute_stage(root, pin, case, stage, prior, value=reg, baseline=source, scene=scene, record=record,
                         plan=plans[str(trials[case][0]["seed"])], arms=trials, stages=stages(reg["budget"], reg["prefix"]),
                         run_factory=new_run, run_type=ObservationRun)

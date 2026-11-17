"""Register and execute the bounded, block-local WP08 Xuzhou comparison."""

from pathlib import Path
from time import perf_counter

from model.checkpoint_io import save_model_checkpoint
from model.research_config import canonical_json
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.region_message_attention import ATTENTION_MODES
from transfer.region_message_run import ATTENTION, GATES, RegionMessageRun
from transfer.xuzhou_multiscale import (
    BUDGET, LR, SEED, SPLIT, STAGES, _accepted_inputs, baseline_from_config, execute_stage,
)


POLICY = "wp08_xuzhou_block_region_message_v1"
ARMS = {arm: arm for arm in ("off", "aligned", *ATTENTION_MODES)}
RECALL_DELTA = -.02
COMPARISONS = {
    "primary": [["combined", "local"]],
    "mechanism": [["combined", control] for control in ("dense", "compress", "route")],
    "vs_aligned": [[arm, "aligned"] for arm in ATTENTION_MODES],
    "vs_off": [[arm, "off"] for arm in ARMS if arm != "off"],
}


def new_run(baseline, scene, record, plan, arm):
    return RegionMessageRun(baseline, scene, record, plan=plan, trial=arm, selector_seed=SEED, lr=LR,
                            torch_seed=SEED, expected_plan_hash=plan.plan_hash)


def _rules():
    return dict(budget=BUDGET, resume_at=SPLIT, arm_order=list(ARMS), maximum_updates=len(ARMS)*2*BUDGET,
                completed_reload_updates=0, model_seed=SEED, exposure_seed=SEED, lr=LR,
                gate_settings=GATES.copy(), attention_settings={**ATTENTION, "landmark_grid": list(ATTENTION["landmark_grid"])},
                comparisons=COMPARISONS, recall_floor_delta=RECALL_DELTA, graph_support="original_A0_both_stages",
                disabled=["multiscale", "graph_router", "edge_updater", "coverage_modulation", "region_residual"])


def _arm_contracts(baseline, scene, record, plan):
    runs = {arm: new_run(baseline, scene, record, plan, arm) for arm in ARMS}
    initialization = {arm: run.initialization for arm, run in runs.items()}
    if (len({row["fingerprints"]["shared"] for row in initialization.values()}) != 1
            or len({initialization[arm]["fingerprints"]["outer_exchange"] for arm in ARMS if arm != "off"}) != 1
            or len({initialization[arm]["fingerprints"]["message_projection"] for arm in ATTENTION_MODES}) != 1):
        raise ValueError("Original, outer exchange or message projection initializations differ across arms")
    return {arm: run.contract_hash for arm, run in runs.items()}, initialization


def _metadata(scene, record):
    return dict(dataset=scene.dataset, bands=len(scene.bands.band_ids), band_sha256=scene.bands.fingerprint,
                wavelength_quality=scene.bands.wavelength_quality, wavelength_unit=scene.bands.wavelength_unit,
                fwhm_present=scene.bands.fwhm is not None, statistics=scene.stats["statistics_hash"],
                role_release=record["validation_hash"], train_support_seed=record["train_support"]["seed"],
                val_support_seed=record["validation_support"]["seed"], train_labels=record["training_labels"],
                val_labels=record["validation_labels"], gsd=None, coverage=None)


def register(root, prior, donor_audit, *, code, issue):
    require_frozen_code(code)
    started = perf_counter()
    scene, record, plan, config, protected = _accepted_inputs(prior, donor_audit)
    admission.check_pins(protected.values())
    root = external_directory(root)
    if type(issue) is not int or issue < 1 or any(root in Path(path).parents for path in protected):
        raise ValueError("Tracked issue and independent run directory required")
    root.mkdir(parents=True, exist_ok=False)
    baseline = baseline_from_config(config)
    source = root / "initial.pt"
    save_model_checkpoint(baseline, source)
    source_pin, sidecar = dev._pin(source), dev._pin(str(source) + ".json")
    contracts, initialization = _arm_contracts(baseline, scene, record, plan)
    admission.check_pins(protected.values())
    require_frozen_code(code)
    exposure = plan.require_eligible()
    return dev._json_new(root / "registration.json", dict(
        **_rules(), policy=POLICY, issue=issue, commit=code["commit"], source_hash=code["source_hash"], root=str(root),
        runtime=admission.runtime_identity(), prior=prior, donor_audit=donor_audit, protected=protected,
        source=source_pin, sidecar=sidecar, source_state=state_hash(baseline), contracts=contracts,
        initialization=initialization, metadata=_metadata(scene, record),
        exposure_hash=plan.plan_hash, exposure_prefix=plan.progress(BUDGET)["prefix_hash"],
        covered_support=len(exposure["support_visits"]), minimum_visits=min(exposure["support_visits"].values()),
        preparation_seconds=perf_counter()-started, registration_forwards=0, registration_updates=0))


def load_registration(pin, code):
    require_frozen_code(code)
    value = admission.read(pin)
    root = external_directory(value["root"])
    if (value["policy"] != POLICY or value["commit"] != code["commit"] or value["source_hash"] != code["source_hash"]
            or value["runtime"] != admission.runtime_identity() or pin["path"] != str(root / "registration.json")
            or any(value[key] != expected for key, expected in _rules().items())):
        raise ValueError("Wrong WP08 code, runtime or registered comparison")
    scene, record, plan, config, protected = _accepted_inputs(value["prior"], value["donor_audit"])
    if (protected != value["protected"] or plan.plan_hash != value["exposure_hash"]
            or plan.progress(BUDGET)["prefix_hash"] != value["exposure_prefix"] or value["covered_support"] != 45
            or value["minimum_visits"] != min(plan.require_eligible()["support_visits"].values())
            or _metadata(scene, record) != value["metadata"]
            or value["source"]["path"] != str(root / "initial.pt")
            or value["sidecar"]["path"] != str(root / "initial.pt.json")):
        raise ValueError("Original data, source or exposure identity changed")
    admission.check_pins(protected.values())
    baseline = dev._load_source(value["source"], value["sidecar"])
    recreated = baseline_from_config(config)
    if (state_hash(baseline) != value["source_state"] or state_hash(recreated) != value["source_state"]
            or baseline.export_config() != recreated.export_config()
            or baseline.training_config != recreated.training_config):
        raise ValueError("Frozen initialization differs from the original GroupNorm configuration")
    contracts, initialization = _arm_contracts(baseline, scene, record, plan)
    if contracts != value["contracts"] or canonical_json(initialization) != canonical_json(value["initialization"]):
        raise ValueError("Message initialization or registered arm contract differs")
    return root, value, baseline, scene, record, plan


def execute(pin, code, *, notice):
    root, value, baseline, scene, record, plan = load_registration(pin, code)
    notification = admission.read(notice)
    if (notice["path"] != str(root / "notice.json") or notification["registration"] != pin
            or notification["maximum_updates"] != value["maximum_updates"] or notification["cpu_workers"] != 1):
        raise ValueError("Announce the frozen registration and actual authorized budget before execution")
    start_pin = dev._json_new(root / "batch-start.json", dict(registration=pin, notice=notice))
    started = perf_counter()
    results = {}
    for arm in ARMS:
        receipts = {}
        for stage, (_, _, previous) in STAGES.items():
            receipts[stage] = execute_stage(root, pin, arm, stage, receipts.get(previous), value=value,
                                            baseline=baseline, scene=scene, record=record, plan=plan,
                                            arms=ARMS, stages=STAGES, run_factory=new_run, run_type=RegionMessageRun)
        results[arm] = receipts
    admission.check_pins([*value["protected"].values(), value["source"], value["sidecar"], pin])
    require_frozen_code(code)
    return dev._json_new(root / "batch-complete.json", dict(registration=pin, start=start_pin, attempts=results,
        updates=value["maximum_updates"], raw_gt=0, query=0, elapsed_seconds=perf_counter()-started))

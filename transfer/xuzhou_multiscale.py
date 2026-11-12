"""Freeze and run the bounded WP07 Xuzhou development experiment."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, tensor_hash
from train_spatial import SpatialConfig
from transfer import matched_route_admission as admission, registered_development as dev
from transfer import wp02_route_development as wp02
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import _forward, state_hash
from transfer.multiscale_run import MultiscaleRun, metadata_contract, selector_group
from transfer.multiscale_training import MultiscaleTrial
from transfer.sample_curriculum import curriculum_loss
from transfer.xuzhou_route_campaign import budget_prefixes


POLICY = "wp07_xuzhou_multiscale_v1"
PRIOR_CAMPAIGN = "2dd3eff9c32300e1df6c330daaa7c79539301a699030c37dc2a4ae6ab8cfb6a4"
PRIOR_DONOR_AUDIT = "1bfd78ef6da01f8fa9cb05b98dd5625c2d6e3be46af75b30a8475ebb8fc9688d"
BUDGET, SPLIT, SEED, LR = 256, 7, 64, 0.0001
ARMS = {
    "off": None,
    "cnn_fixed1": MultiscaleTrial(),
    "cnn_fixed2": MultiscaleTrial(cnn_dilation=2),
    "cnn_fixed3": MultiscaleTrial(cnn_dilation=3),
    "cnn_average": MultiscaleTrial(cnn_multiscale="mean"),
    "cnn_content": MultiscaleTrial(cnn_multiscale="content"),
    "graph_average": MultiscaleTrial(graph_multiscale="mean"),
    "both_content": MultiscaleTrial(cnn_multiscale="content", graph_multiscale="content"),
}
STAGES = {"full": (0, BUDGET, None), "prefix": (0, SPLIT, None),
          "resume": (SPLIT, BUDGET, "prefix"), "completed": (BUDGET, BUDGET, "resume")}


def baseline_from_config(config):
    kwargs = SpatialConfig(config).research.model_kwargs()
    if (kwargs["latent_channels"], kwargs["adapter_embed_dim"], kwargs["norm_type"], kwargs["graph_mode"]) != (
            128, 64, "domain_bn", "learned"):
        raise ValueError("Expected the accepted WP02 model geometry")
    kwargs.update(norm_type="group_norm", graph_route=None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = MultiDatasetVCASAHGCN({wp02.SCENE: len(wp02.CLASSES)}, **kwargs).eval()
    model.backbone.adapter.cache_attention = False
    model.training_config = {"initialization": "fresh_group_norm", "model_seed": SEED,
                             "shared_pretrained_tensors": 0, "architecture_ancestor": wp02.REGISTRATION}
    return model


def new_run(baseline, scene, record, plan, trial):
    return MultiscaleRun(baseline, scene, record, plan=plan, trial=trial, selector_seed=SEED,
                         lr=LR, torch_seed=SEED, expected_plan_hash=plan.plan_hash)


def fallback_check(baseline, scene, record, plan):
    """Real forward and gradient equivalence using the published, approximate metadata."""
    pairs = (("mean", "metadata"), ("content", "joint"))
    validation = [scene.plan["tiles"][r["id"]] for r in record["roles"] if r["role"] == "devval"]
    train = scene.plan["tiles"][plan.event(1)["tile_id"]]
    result = []
    for first, second in pairs:
        variants = [MultiscaleTrial(cnn_multiscale=mode, graph_multiscale=mode).initialize(baseline, SEED).eval()
                    for mode in (first, second)]
        for model in variants:
            model.backbone.adapter.cache_attention = False
        hashes = []
        with torch.no_grad():
            for tile in validation:
                left, right = [_forward(model, scene, tile, multiscale_condition_only=True) for model in variants]
                if any(not torch.equal(a, b) for a, b in zip(left[:3], right[:3], strict=True)):
                    raise ValueError("Missing-metadata V5 fallback differs")
                for key in ("multiscale_cnn", "multiscale_graph"):
                    if not torch.equal(left[3][key]["weights"], right[3][key]["weights"]):
                        raise ValueError("Missing-metadata selector weights differ")
                hashes.append([tensor_hash(probabilities) for probabilities in left[:3]])
        training = []
        for model in variants:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                model.train()
                outputs = _forward(model, scene, train, multiscale_condition_only=True)
                loss = curriculum_loss(outputs, scene, record, train, policy=None, step=None,
                                       class_balance=True)["total"]
                loss.backward()
                training.append(([tensor_hash(p) for p in outputs[:3]], float(loss.detach())))
            model.eval()
        if training[0] != training[1]:
            raise ValueError("Missing-metadata training probabilities or loss differ")
        a, b = [dict(model.named_parameters()) for model in variants]
        if any((p.grad is None) != (b[k].grad is None)
               or (p.grad is not None and not torch.equal(p.grad, b[k].grad)) for k, p in a.items()):
            raise ValueError("Missing-metadata fallback training gradients differ")
        inactive = {k for k in b if k not in a}
        if not inactive or any(selector_group(k) not in ("cnn_metadata", "graph_metadata") or b[k].grad is not None
                               for k in inactive):
            raise ValueError("Missing-metadata weights received a gradient")
        result.append(dict(first=first, second=second, matching_parameters=len(a),
                           inactive_parameters=len(inactive), v5_probability_hashes=hashes))
    return dict(pairs=result, forwards=4 * (len(validation) + 1), backward_calls=4, optimizer_updates=0)


def _accepted_inputs(prior_pin, donor_audit_pin):
    if (prior_pin["sha256"], donor_audit_pin["sha256"]) != (PRIOR_CAMPAIGN, PRIOR_DONOR_AUDIT):
        raise ValueError("Expected accepted Xuzhou exposure provenance")
    prior, audit = [admission.read(pin) for pin in (prior_pin, donor_audit_pin)]
    original = admission.read(prior["registrations"]["donor"])
    prefix = admission.read(prior["prefixes"])
    independent = admission.read(prefix["independent"])
    scene, record, files = dev.load_development(prior["development"])
    upstream = record["wp02"]
    config, approved, *_ = wp02.accepted_inputs(upstream["registration"], upstream["acceptance"])
    rows, plan = budget_prefixes(scene, record)
    if (audit.get("campaign") != prior_pin or audit["status"] != "passed"
            or rows != prefix["rows"] or rows != independent["rows"] or plan is None
            or plan.updates != BUDGET or plan.to_dict() != original["plan"]
            or plan.plan_hash != prior["exposure_hash"] or upstream != approved
            or scene.dataset != wp02.SCENE or len(scene.bands.band_ids) != 436
            or [sum(row["role"] == name for row in record["roles"]) for name in ("devtrain", "devval")] != [73, 5]
            or (record["training_labels"], record["validation_labels"]) != (45, 45)
            or scene.bands.wavelength_quality != "approximate" or scene.bands.fwhm is not None):
        raise ValueError("Original WP02 development or 256-update exposure differs")
    files = [prior_pin, donor_audit_pin, prior["registrations"]["donor"], prior["prefixes"],
             prefix["independent"], prior["source"], prior["source_metadata"], *approved.values(), *files]
    return scene, record, plan, config, {pin["path"]: pin for pin in files}


def register(root, prior, donor_audit, *, code, issue):
    require_frozen_code(code)
    started = perf_counter()
    scene, record, plan, config, pins = _accepted_inputs(prior, donor_audit)
    admission.check_pins(pins.values())
    root = external_directory(root)
    if type(issue) is not int or issue < 1 or any(root in Path(path).parents for path in pins):
        raise ValueError("Issue and independent run root required")
    root.mkdir(parents=True, exist_ok=False)
    baseline = baseline_from_config(config)
    source = root / "initial.pt"
    save_model_checkpoint(baseline, source)
    source_pin, sidecar = dev._pin(source), dev._pin(str(source) + ".json")
    contracts = {name: new_run(baseline, scene, record, plan, trial).contract_hash for name, trial in ARMS.items()}
    fallback = fallback_check(baseline, scene, record, plan)
    admission.check_pins(pins.values())
    require_frozen_code(code)
    exposure = plan.exposure()
    return dev._json_new(root / "registration.json", dict(
        policy=POLICY, issue=issue, commit=code["commit"], source_hash=code["source_hash"], root=str(root),
        runtime=admission.runtime_identity(), prior=prior, donor_audit=donor_audit, protected=pins,
        source=source_pin, sidecar=sidecar, source_state=state_hash(baseline), contracts=contracts,
        exposure_hash=plan.plan_hash, exposure_prefix=plan.progress(BUDGET)["prefix_hash"],
        covered_support=len(exposure["support_visits"]), minimum_visits=min(exposure["support_visits"].values()),
        budget=BUDGET, resume_at=SPLIT, arm_order=list(ARMS), maximum_updates=len(ARMS) * 2 * BUDGET,
        completed_reload_updates=0, recall_floor_delta=-.02, metadata=metadata_contract(scene),
        fallback=fallback, preparation_seconds=perf_counter() - started))


def load_registration(pin, code):
    require_frozen_code(code)
    value = admission.read(pin)
    root = external_directory(value["root"])
    if (value["policy"] != POLICY or value["commit"] != code["commit"] or value["source_hash"] != code["source_hash"]
            or value["runtime"] != admission.runtime_identity() or pin["path"] != str(root / "registration.json")
            or (value["budget"], value["resume_at"], value["maximum_updates"], value["completed_reload_updates"])
            != (BUDGET, SPLIT, len(ARMS) * 2 * BUDGET, 0)
            or value["arm_order"] != list(ARMS) or value["recall_floor_delta"] != -.02):
        raise ValueError("Wrong WP07 registration, code, runtime or budget")
    scene, record, plan, config, pins = _accepted_inputs(value["prior"], value["donor_audit"])
    if (pins != value["protected"] or plan.plan_hash != value["exposure_hash"]
            or plan.progress(BUDGET)["prefix_hash"] != value["exposure_prefix"] or value["covered_support"] != 45
            or value["minimum_visits"] != min(plan.exposure()["support_visits"].values())
            or metadata_contract(scene) != value["metadata"]
            or value["source"]["path"] != str(root / "initial.pt")
            or value["sidecar"]["path"] != str(root / "initial.pt.json")):
        raise ValueError("WP07 frozen inputs changed")
    admission.check_pins(pins.values())
    baseline = dev._load_source(value["source"], value["sidecar"])
    recreated = baseline_from_config(config)
    if (state_hash(baseline) != value["source_state"] or state_hash(recreated) != value["source_state"]
            or baseline.export_config() != recreated.export_config()
            or baseline.training_config != recreated.training_config):
        raise ValueError("Fresh baseline initialization or architecture changed")
    return root, value, baseline, scene, record, plan


def _stage_paths(root, arm, name, arms, stages):
    if arm not in arms or name not in stages:
        raise ValueError("Unknown registered arm or interval")
    folder = root / "attempts" / arm / name
    claim = root / "claims" / arm / (name + ".json")
    if folder.resolve() != folder or claim.resolve() != claim:
        raise ValueError("Stage path escapes the registration")
    return folder, claim


def read_stage(root, registration, arm, name, prior, *, arms=None, stages=None):
    arms = ARMS if arms is None else arms
    stages = STAGES if stages is None else stages
    folder, claim = _stage_paths(root, arm, name, arms, stages)
    start, end, _ = stages[name]
    expected = dict(registration=registration, arm=arm, stage=name, start=start, stop=end, prior=prior)
    claim_pin = dev._pin(claim)
    if admission.read(claim_pin) != expected:
        raise ValueError("Stage claim differs from registration")
    receipt = dev._pin(folder / "status.json")
    status = admission.read(receipt)
    if (status["status"] != "passed" or status["claim"] != claim_pin or status["new_updates"] != end - start
            or status["checkpoint"]["path"] != str(folder / f"step-{end}.pt")):
        raise ValueError("Stage has not completed its own frozen interval")
    return status, receipt


def execute_stage(root, registration, arm, name, prior, *, value, baseline, scene, record, plan,
                  arms=None, stages=None, run_factory=new_run, run_type=MultiscaleRun):
    arms = ARMS if arms is None else arms
    stages = STAGES if stages is None else stages
    folder, claim = _stage_paths(root, arm, name, arms, stages)
    start, end, previous = stages[name]
    budget = stages["full"][1] if stages is not STAGES else BUDGET
    if plan.updates != value["budget"] or value["budget"] != budget:
        raise ValueError("Plan cannot spend another registered budget")
    if folder.exists() or claim.exists():
        raise FileExistsError("Registered arm interval already consumed")
    packet = None
    if previous is None:
        if prior is not None:
            raise ValueError("Fresh interval cannot consume a prefix")
    else:
        prev_folder, prev_claim = _stage_paths(root, arm, previous, arms, stages)
        if prior is None or prior["path"] != str(prev_folder / "status.json"):
            raise ValueError("Cross-arm or cross-interval resume rejected")
        old = admission.read(prior)
        previous_claim = admission.read(dev._pin(prev_claim))
        verified, checked = read_stage(root, registration, arm, previous, previous_claim["prior"], arms=arms, stages=stages)
        if checked != prior or verified != old:
            raise ValueError("Wrong preceding checkpoint receipt")
        packet = old["checkpoint"]
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim_pin = dev._json_new(claim, dict(registration=registration, arm=arm, stage=name,
                                        start=start, stop=end, prior=prior))
    folder.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    run = None
    try:
        if packet is None:
            run = run_factory(baseline, scene, record, plan, arms[arm])
        else:
            run = run_type.load(baseline, scene, record, packet["path"], plan=plan, trial=arms[arm],
                                selector_seed=SEED, expected_sha256=packet["sha256"],
                                expected_contract_hash=value["contracts"][arm])
        if run.contract_hash != value["contracts"][arm] or run.completed != start:
            raise ValueError("Arm checkpoint contract or registered update start differs")
        ready = perf_counter()
        for index in range(start + 1, end + 1):
            run.step()
            if index % 16 == 0 or index == end or (arm == "off" and name == "full" and index in (1, 2)):
                print(f"{arm} {name} {index}/{end} {perf_counter() - ready:.3f}s", flush=True)
            if arm == "off" and name == "full" and index == 2:
                dev._json_new(folder / "first-two.json", dict(new_updates=2, seconds=perf_counter() - ready))
        trained = perf_counter()
        checkpoint = folder / f"step-{end}.pt"
        checksum = run.save(checkpoint)
        finish = run.finish() if end == budget else None
        return dev._json_new(folder / "status.json", dict(status="passed", claim=claim_pin, new_updates=end - start,
            checkpoint=dict(path=str(checkpoint), sha256=checksum), finish=finish, contract=run.contract_hash,
            restore_or_prepare_seconds=ready - started, update_loop_seconds=trained - ready,
            checkpoint_finish_seconds=perf_counter() - trained, timing={
                key: dict(seconds=sum(times), calls=len(times)) for key, times in run.timings.items()}))
    except BaseException as error:
        failure = None
        if run is not None:
            if not run.failed:
                run.failed = True
                run.failure = dict(type=type(error).__name__, message=str(error), attempted_step=run.completed + 1)
            failure_path = folder / "failure.pt"
            failure = dict(path=str(failure_path), sha256=run.save_failure(failure_path))
        dev._json_new(folder / "status.json", dict(status="failed", claim=claim_pin, error=repr(error), failure=failure,
                                                   completed=None if run is None else run.completed))
        raise


def execute(pin, code, *, notice):
    root, value, baseline, scene, record, plan = load_registration(pin, code)
    notification = admission.read(notice)
    if (notice["path"] != str(root / "notice.json") or notification["registration"] != pin
            or notification["maximum_updates"] != value["maximum_updates"] or notification["cpu_workers"] != 1):
        raise ValueError("Frozen budget announcement required")
    start_pin = dev._json_new(root / "batch-start.json", dict(registration=pin, notice=notice))
    started = perf_counter()
    results = {}
    for arm in ARMS:
        receipts = {}
        for stage, (_, _, previous) in STAGES.items():
            receipts[stage] = execute_stage(root, pin, arm, stage, receipts.get(previous), value=value,
                                            baseline=baseline, scene=scene, record=record, plan=plan)
        results[arm] = receipts
    admission.check_pins([*value["protected"].values(), value["source"], value["sidecar"], pin])
    require_frozen_code(code)
    return dev._json_new(root / "batch-complete.json", dict(registration=pin, start=start_pin, attempts=results,
        updates=value["maximum_updates"], raw_gt=0, query=0, elapsed_seconds=perf_counter() - started))

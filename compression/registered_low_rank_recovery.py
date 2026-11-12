"""Freeze and execute the original Xuzhou low-rank rank ladder on matched budgets."""

from pathlib import Path

from compression.low_rank_checkpoint import load_low_rank, save_low_rank
from compression.low_rank_recovery import (RECOVERY_SEEDS, TARGET, LowRankRecoveryRun, exposure, initial_evidence,
                                           ladder, new_run, transform)
from compression.precision_fallback import LIMITS, SMALL_CLASS_MAX_SUPPORT, _validation_inputs
from compression.registered_precision_recovery import _historical, runtime, stages
from inference.light_cnn import validate_source
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_exposure import ExposurePlan
from transfer.xuzhou_multiscale import execute_stage, read_stage


POLICY = "wp10_low_rank_recovery_v1"
REAL = "wp08_xuzhou_off"
ORIGIN_SHA256 = "7dd6479583fff240e3f3c7b4e56d1f496e1d0c60eab8159b36f1cb58f0cb6ed6"


def _source(request):
    spec = admission.read(request)
    if spec.get("scope") == REAL and set(spec) == {"scope", "origin"}:
        if spec["origin"]["sha256"] != ORIGIN_SHA256:
            raise ValueError("Exact accepted WP08 off release required")
        model, scene, record, info, protected, prior = _historical(spec["origin"])
        if (len(record["train_support"]["indices"]), len(record["validation_support"]["indices"]),
                record["train_support"]["seed"], record["validation_support"]["seed"]) != (45, 45, 512, 3031):
            raise ValueError("Original Xuzhou K5/V5 support identity required")
        seeds, limit, prefix = RECOVERY_SEEDS, 512, 7
    elif spec.get("scope") == "synthetic" and set(spec) == {"scope", "development", "weights", "sidecar", "budget"}:
        model = dev._load_source(spec["weights"], spec["sidecar"])
        scene, record, pins = dev.load_development(spec["development"])
        info = {k: spec[k] for k in ("weights", "sidecar", "development")}
        protected = {p["path"]: p for p in (*pins, spec["weights"], spec["sidecar"])}
        limit = spec["budget"]
        if type(limit) is not int or not 2 <= limit <= 16 or max(scene.plan["shape"]) > 160:
            raise ValueError("Only bounded synthetic fixtures can set a budget")
        seeds, prefix, prior = (64,), limit//2, None
    else:
        raise ValueError("Unsupported low-rank source")
    validate_source(model)
    if model.backbone.norm_type != "group_norm" or info["sidecar"]["path"] != info["weights"]["path"] + ".json":
        raise ValueError("Original GroupNorm source and adjacent sidecar required")
    return spec["scope"], model, scene, record, info, {**protected, request["path"]: request}, seeds, limit, prefix, prior


def _design(request):
    scope, model, scene, record, info, protected, seeds, limit, prefix, prior = _source(request)
    shape = ladder(model)
    coverage, wide = exposure(scene, record, seeds, limit)
    eligible = [seed for seed in seeds if not coverage[str(seed)]["missing"]]
    budget = (max(coverage[str(seed)]["shortest"] for seed in eligible) if eligible and scope == REAL
              else limit if eligible else 0)
    if eligible and not prefix < budget <= limit:
        raise ValueError("Frozen split must have a positive suffix")
    plans = {}
    for seed in eligible:
        plan = ExposurePlan.freeze(scene, record, updates=budget, seed=seed,
                                   minimum_class_blocks={str(c): 1 for c in record["classes"]}, minimum_support_visits=1)
        plan.require_eligible()
        if plan.to_dict()["events"] != wide[seed].to_dict()["events"][:budget]:
            raise ValueError("Shared budget changed the prescanned original stream")
        if prior is not None and seed == 64 and plan.to_dict()["events"] != prior.to_dict()["events"][:budget]:
            raise ValueError("Original seed-64 stream differs from the WP08 producer")
        plans[str(seed)] = plan
    cases = [dict(case=f"s{seed}-{arm}", seed=seed, arm=arm, assignment=setting)
             for seed in eligible for arm, setting in shape["arms"].items()]
    spec = dict(scope=scope, source=info, protected=protected, layer=TARGET, ladder=shape, recovery_seeds=list(seeds),
                source_model_seed=64, limit=limit, prefix=prefix, budget=budget, coverage=coverage,
                rejected_seeds=[seed for seed in seeds if seed not in eligible],
                exposures={seed: plan.to_dict() for seed, plan in plans.items()}, cases=cases,
                maximum_updates=2*budget*len(cases), completed_reload_updates=0,
                validation=_validation_inputs(scene, record)[0], selection="first_min_deployment_fused_nll",
                optimizer="original_adam_1e-4_class_balanced_three_heads", channel_pruning=False,
                protection=dict(limits={k: float(v) for k, v in LIMITS.items()}, small_support_max=SMALL_CLASS_MAX_SUPPORT, small_new_errors_max=0))
    return spec, model, scene, record, plans


def register(request, out, code):
    require_frozen_code(code)
    env = runtime()
    spec, dense, scene, record, plans = _design(request)
    root = external_directory(out)
    if any(root == Path(path) or root in Path(path).parents or Path(path).parent == root
           for path in spec["protected"]):
        raise ValueError("New output must be isolated from protected inputs")
    root.mkdir()
    initial_dir = root / "initial"
    initial_dir.mkdir()
    initial, contracts = {}, {}
    for case in spec["cases"]:
        name, assignment, seed = case["case"], case["assignment"], case["seed"]
        candidate = transform(dense, assignment, seed)
        evidence = initial_evidence(dense, candidate, assignment)
        if assignment["rank"] is None:
            pin = spec["source"]["weights"]
        else:
            path = initial_dir / f"{name}.pt"
            pin = dict(path=str(path), sha256=save_low_rank(candidate, path))
        initial[name] = dict(checkpoint=pin, evidence=evidence)
        contracts[name] = new_run(candidate, scene, record, plans[str(seed)], name).contract_hash
    admission.check_pins([*spec["protected"].values(), *[v["checkpoint"] for v in initial.values()]])
    require_frozen_code(code)
    return dev._json_new(root / "registration.json", dict(policy=POLICY, root=str(root), commit=code["commit"],
                         source_hash=code["source_hash"], runtime=env, request=request, **spec,
                         initial=initial, contracts=contracts, optimizer_updates=0, gt=0, query=0))


def load_registration(pin, code):
    require_frozen_code(code)
    value = admission.read(pin)
    root = external_directory(value["root"])
    if (value["policy"] != POLICY or pin["path"] != str(root / "registration.json")
            or value["commit"] != code["commit"] or value["source_hash"] != code["source_hash"]
            or value["runtime"] != runtime() or any(value[k] != 0 for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Expected exact registered code, runtime and original path")
    spec, dense, scene, record, plans = _design(value["request"])
    expected = {c["case"] for c in spec["cases"]}
    if (any(value[k] != v for k, v in spec.items()) or set(value["initial"]) != expected
            or set(value["contracts"]) != expected):
        raise ValueError("Frozen source, assignment, coverage or budget changed")
    return root, value, dense, scene, record, plans


def initial_model(root, reg, dense, case):
    name, setting = case["case"], case["assignment"]
    row = reg["initial"][name]
    pin = row["checkpoint"]
    if setting["rank"] is None:
        if pin != reg["source"]["weights"]:
            raise ValueError("The matched dense arm must use the same source")
        model = dense
    else:
        if pin["path"] != str(root / "initial" / f"{name}.pt"):
            raise ValueError("Wrong case initialization path")
        model = load_low_rank(pin["path"], pin["sha256"])
    if initial_evidence(dense, model, setting) != row["evidence"]:
        raise ValueError("Initial factors or shared dense weights differ")
    return model


def verify_notice(root, reg, pin, notice):
    value = admission.read(notice)
    if (set(value) != {"registration", "cases", "budget", "maximum_updates", "cpu_workers", "estimate"}
            or notice["path"] != str(root / "notice.json") or value["registration"] != pin
            or value["cases"] != [case["case"] for case in reg["cases"]]
            or value["budget"] != reg["budget"] or value["maximum_updates"] != reg["maximum_updates"]
            or value["cpu_workers"] != 1 or not isinstance(value["estimate"], str) or not value["estimate"].strip()):
        raise ValueError("Frozen exact CPU case and budget notice required")


def execute(pin, code, notice, case, stage, prior):
    root, reg, dense, scene, record, plans = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    cases = {c["case"]: c for c in reg["cases"]}
    if case not in cases:
        raise ValueError("Unknown or exposure-rejected case")
    chosen = cases[case]
    baseline = initial_model(root, reg, dense, chosen)
    return execute_stage(root, pin, case, stage, prior, value=reg, baseline=baseline, scene=scene, record=record,
                         plan=plans[str(chosen["seed"])], arms={k: k for k in cases},
                         stages=stages(reg["budget"], reg["prefix"]), run_factory=new_run, run_type=LowRankRecoveryRun)


def complete(pin, code, notice):
    root, reg, _, _, _, _ = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    intervals = stages(reg["budget"], reg["prefix"])
    names = {c["case"]: c["case"] for c in reg["cases"]}
    attempts, count = {}, 0
    for case in names:
        rows = {}
        for stage, (_, _, previous) in intervals.items():
            status, rows[stage] = read_stage(root, pin, case, stage, rows.get(previous), arms=names, stages=intervals)
            count += status["new_updates"]
        attempts[case] = rows
    if count != reg["maximum_updates"]:
        raise ValueError("Observed updates differ from the registered ceiling")
    admission.check_pins([pin, notice, *reg["protected"].values(), *[x["checkpoint"] for x in reg["initial"].values()]])
    require_frozen_code(code)
    return dev._json_new(root / "batch-complete.json", dict(registration=pin, notice=notice, attempts=attempts,
                         updates=count, rejected_seeds=reg["rejected_seeds"], optimizer_updates=0, gt=0, query=0))

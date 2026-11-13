"""Train only the new real combination, with authenticated historical controls."""

from pathlib import Path
from time import perf_counter

from compression.registered_precision_recovery import stages
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_factorial import ARMS, FactorialRun, REAL_PROPOSAL
from transfer.curriculum_observation_admission import CONDITIONS, RestrictedObservation, bind_sample
from transfer.curriculum_observation_real import REAL_ADMISSION
from transfer.curriculum_real import REAL, SEEDS, runtime, verify_notice
from transfer.g3_training import state_hash
from transfer.sample_curriculum import SampleCurriculum
from transfer.xuzhou_multiscale import LR, execute_stage, read_stage


FORMAT = "wp01_xuzhou_mixed_producer_factorial_v1"
BUDGET, PREFIX = 178, 7
NEW_UPDATES, REFERENCED_UPDATES = 1068, 3204
ACCEPTED = {
    "sample": {
        "registration": "3faa32ceb295008133e8dcd2fa9fa2d5e75da4b0aa5997b6f9f4755350dd6da0",
        "completion": "cc44f9b9a378f31fc764b64d8538334ed8198f381e2b29539b94db1077e1f905",
        "audit": "ebb4eb24d7ef24542fd08e8b3ca8fa87ecba55f80fd855d8e7d7ecd90746590b"
    },
    "observation": {
        "registration": "45f5badb8579413cd1b28cfc74949420d8d2150b251a64e8e86a0fbd0747cd11",
        "completion": "a70b6a310cde6a76dc79177f3644b6e395b1450108701095f47d3fa3785adfae",
        "audit": "1ccf81149b4fcc17d9e95cd5515368e4cfff8565e65871e480a2158f28a43eaf",
        "frozen_selected": "9e43e1a5715c28709544dda858b3b61e8b11b62d78994e9b31feca2e2732df10"
    },
    "synthetic": {
        "3": "760e02c8e5d9b1ed24fecd0ccc3fad771bc2d3e379d0e0a990821b8e0fbc024b",
        "9": "39afbc73d403c712b109c1e608ab1f2163d3de91663968c4ddcce83f1e05e282",
        "22": "59e82ea60425dd8ade19194740042306f7a58b9e362b2a218a58126fa12fcd78"
    }
}
ORDER = "easy_to_hard"
CASES = tuple(dict(case=f"s{seed}-{arm}", seed=seed, arm=arm) for seed in SEEDS for arm in ARMS)
NEW_CASES = tuple(c for c in CASES if c["arm"] == "s_plus_o")


class RealFactorialRun(FactorialRun):
    FORMAT = "vca-wp01-real-factorial-recovery-v1"
    FAILURE_FORMAT = "vca-wp01-real-factorial-failure-v1"
    CONTRACT_SCOPE = FORMAT

    def _scene_allowed(self, scene, trial):
        return (trial == "s_plus_o" and scene.dataset == "xuzhou" and scene.classes == 9
                and len(scene.bands.band_ids) == 436)

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        result = super()._training_forward(tile, step, extra)
        self.forward_evidence["graph_info"] = {k: result[3][k] for k in ("graph_nnz", "grid_shape") if k in result[3]}
        self.timings["training_forward"].append(perf_counter() - started)
        return result


def new_run(source, scene, record, plan, trial):
    arm, sample, observation, teacher = trial
    return RealFactorialRun(source, scene, record, trial=arm, teacher_hash=teacher, plan=plan, policy=sample, lr=LR,
        torch_seed=plan.to_dict()["seed"], expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
        expected_policy_hash=sample.policy_hash, observation_policy=observation, expected_observation_hash=observation.policy_hash)


def accepted_request(pin):
    value = admission.read(pin)
    if set(value) != set(ACCEPTED) or any(group.keys() != ACCEPTED[key].keys() for key, group in value.items()):
        raise ValueError("Exact independently accepted sample, observation and synthetic receipts required")
    if any(p["sha256"] != ACCEPTED[key][name] for key, group in value.items() for name, p in group.items()):
        raise ValueError("Historical mixed-producer receipt digest differs")
    return value


def inputs(request, code):
    req = accepted_request(request)
    s, model, scene, record, plans = bind_sample(req["sample"]["registration"], code)
    o = admission.read(req["observation"]["registration"])
    old = admission.read(o["prior"])
    teacher = admission.read(s["teacher"])
    pacing = s["pacing"]
    if (s["scope"] != REAL or o["scope"] != REAL or o["sample_registration"] != req["sample"]["registration"]
            or o["prior"]["sha256"] != REAL_ADMISSION or s["source"] != o["source"] or s["teacher_state"] != o["state"]
            or s["model_seed"] != o["model_seed"] or s["model_seed"] != 64 or s["seeds"] != list(SEEDS)
            or (s["budget"], o["budget"], s["prefix"], o["prefix"]) != (BUDGET, BUDGET, PREFIX, PREFIX)
            or s["validation"] != o["validation"] or s["exposures"] != o["exposures"] or s["support"] != o["train_support"]
            or s["selection"] != o["selection"] or s["protection"] != o["protection"] or s["runtime"] != o["runtime"]
            or pacing != dict(seed=512, start_fraction=.2, minimum_weight=.1, full_at=BUDGET)
            or o["validation_conditions"] != list(CONDITIONS) or s["protection"]["small_new_errors_max"] != 0
            or teacher["score_hash"] != digest({k:v for k,v in teacher.items() if k != "score_hash"})
            or teacher["teacher_state"] != state_hash(model) or teacher["support"] != record["train_support"]):
        raise ValueError("Historical source, teacher, role, event, budget or metric contract differs")
    sample = SampleCurriculum.freeze(teacher, updates=BUDGET, order=ORDER, **pacing)
    sample.validate(scene, record)
    observation = RestrictedObservation(admission.read(old["policies"][ORDER]))
    observation.validate(scene, record, BUDGET)
    if sample.policy_hash != s["policies"][ORDER] or observation.policy_hash != o["policy_hashes"][ORDER]:
        raise ValueError("Real factorial must use the two original easy-to-hard policies")
    trials = {c["case"]: ("s_plus_o", sample, observation, teacher["score_hash"]) for c in NEW_CASES}
    return req, s, o, model, scene, record, plans, trials


def _design(request, controls, code):
    req, s, o, model, scene, record, plans, trials = inputs(request, code)
    witness = admission.read(controls)
    if (witness["format"] != "wp01_factorial_authenticated_controls_v1" or witness["status"] != "passed"
            or witness["request"] != request or witness["commit"] != code["commit"]
            or witness["source_hash"] != code["source_hash"] or witness["historical_updates"] != REFERENCED_UPDATES
            or set(witness["cases"]) != {c["case"] for c in CASES if c["arm"] != "s_plus_o"}
            or any(witness[k] for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Exact zero-update control admission required")
    _, sample, observation, _ = next(iter(trials.values()))
    design = dict(request=request, controls=controls, scope=REAL, protocol=s["protocol"], source=s["source"],
                  state=s["teacher_state"], model_seed=s["model_seed"], teacher=s["teacher"],
                  support=s["support"], validation=s["validation"], preprocessing=s["preprocessing"],
                  plan_hashes={k: plan.plan_hash for k, plan in plans.items()},
                  sample_policy=sample.policy_hash, observation_policy=observation.policy_hash,
                  validation_identity=o["validation_identity"], selection=s["selection"], protection=s["protection"],
                  all_cases=list(CASES), cases=list(NEW_CASES), budget=BUDGET, prefix=PREFIX,
                  comparison_updates=NEW_UPDATES+REFERENCED_UPDATES, maximum_updates=NEW_UPDATES,
                  historical_control_updates=REFERENCED_UPDATES, historical_batches=dict(sample=s["maximum_updates"], observation=o["maximum_updates"]),
                  historical_full_rerun_proposal=REAL_PROPOSAL, protected=witness["protected"])
    return design, model, scene, record, plans, trials


def register(request, controls, out, code):
    started = perf_counter()
    design, model, scene, record, plans, trials = _design(request, controls, code)
    out = external_directory(out)
    pins = [request, controls, *design["protected"]]
    if out.exists() or any(out == Path(p["path"]).parent or out in Path(p["path"]).parents for p in pins):
        raise ValueError("Fresh factorial output separate from prior evidence required")
    contracts = {c["case"]: new_run(model, scene, record, plans[str(c["seed"])], trials[c["case"]]).contract_hash
                 for c in NEW_CASES}
    admission.check_pins(pins)
    require_frozen_code(code)
    out.mkdir()
    return dev._json_new(out/"registration.json", dict(format=FORMAT, root=str(out), commit=code["commit"],
        source_hash=code["source_hash"], runtime=runtime(), contracts=contracts, **design, optimizer_updates=0,
        gt=0, query=0, preparation_seconds=perf_counter()-started))


def load_registration(pin, code):
    require_frozen_code(code)
    r = admission.read(pin)
    root = external_directory(r["root"])
    if (r["format"] != FORMAT or pin["path"] != str(root/"registration.json") or r["runtime"] != runtime()
            or (r["commit"], r["source_hash"]) != (code["commit"], code["source_hash"])
            or any(r[k] for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Wrong exact factorial execution registration")
    design, model, scene, record, plans, trials = _design(r["request"], r["controls"], code)
    if any(r[k] != v for k, v in design.items()) or set(r["contracts"]) != set(trials):
        raise ValueError("Real factorial comparison or update budget differs")
    return root, r, model, scene, record, plans, trials


def execute(pin, code, notice, case, stage, prior):
    root, reg, model, scene, record, plans, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    candidates = {c["case"]: c["seed"] for c in NEW_CASES}
    if case not in candidates:
        raise ValueError("Only the new S+O cases may execute")
    return execute_stage(root, pin, case, stage, prior, value=reg, baseline=model, scene=scene, record=record,
        plan=plans[str(candidates[case])], arms=trials, stages=stages(BUDGET, PREFIX), run_factory=new_run, run_type=RealFactorialRun)


def complete(pin, code, notice):
    root, reg, _, _, _, _, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    intervals = stages(BUDGET, PREFIX)
    attempts, updates = {}, 0
    for name in trials:
        receipts = {}
        for stage, (_, _, previous) in intervals.items():
            result, receipts[stage] = read_stage(root, pin, name, stage, receipts.get(previous), arms=trials, stages=intervals)
            updates += result["new_updates"]
        attempts[name] = receipts
    if updates != NEW_UPDATES:
        raise ValueError("The completed new factorial budget differs")
    admission.check_pins([pin, notice, reg["request"], reg["controls"], *reg["protected"]])
    require_frozen_code(code)
    return dev._json_new(root/"batch-complete.json", dict(registration=pin, notice=notice, attempts=attempts,
        updates=updates, optimizer_updates=0, gt=0, query=0))

"""Explicit execution registration for the previously admitted CUR-O observation plan."""

from pathlib import Path
from time import perf_counter

from compression.registered_precision_recovery import stages
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_observation_admission import CONDITIONS, FORMAT as ADMISSION, ObservationRun, _design
from transfer.curriculum_real import REAL, runtime, verify_notice
from transfer.g3_training import state_hash
from transfer.observation_curriculum import ORDERS
from transfer.xuzhou_multiscale import LR, execute_stage, read_stage


FORMAT = "wp01_registered_real_observation_execution_v1"
REAL_ADMISSION = "109acaf99286f763096ae03069fc302564aaa8a1410046e040decb22588172c0"
SAMPLE_WITNESS = "3faa32ceb295008133e8dcd2fa9fa2d5e75da4b0aa5997b6f9f4755350dd6da0"


class MeasuredObservationRun(ObservationRun):
    FORMAT = "vca-wp01-measured-observation-v1"
    FAILURE_FORMAT = "vca-wp01-measured-observation-failure-v1"

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        result = super()._training_forward(tile, step, extra)
        graph = result[3]
        self.forward_evidence["graph_info"] = {k: graph[k] for k in ("graph_nnz", "grid_shape") if k in graph}
        self.timings["training_forward"].append(perf_counter()-started)
        return result

    def _evaluate(self, model):
        started = perf_counter()
        result = super()._evaluate(model)
        self.timings["full_v5_event"].append(perf_counter()-started)
        return result


def new_run(source, scene, record, plan, trial):
    case, policy = trial
    return MeasuredObservationRun(source, scene, record, trial=case, plan=plan, policy=None, lr=LR,
                                  torch_seed=case["seed"], expected_source_state=state_hash(source),
                                  expected_plan_hash=plan.plan_hash, expected_policy_hash=None,
                                  observation_policy=policy, expected_observation_hash=policy.policy_hash)


def inputs(prior, code):
    old = admission.read(prior)
    if (old["format"] != ADMISSION or old["runtime"] != runtime() or old["real_training"] != "NOT_RUN"
            or old["scope"] not in (REAL, "synthetic")
            or prior["path"] != str(Path(old["root"])/"registration.json")
            or any(old[k] for k in ("optimizer_updates", "query", "original_gt_reads"))
            or old["conditions"] != list(CONDITIONS)):
        raise ValueError("Exact immutable zero-update observation admission required")
    if old["scope"] == REAL and (prior["sha256"], old["sample_registration"]["sha256"]) != (REAL_ADMISSION, SAMPLE_WITNESS):
        raise ValueError("Reviewed real CUR-O admission and sample registration required")
    sample, source, scene, record, plans, policies, cases, exposure = _design(old["sample_registration"], code)
    if (old["scope"] != sample["scope"] or old["source_commit"] != sample["commit"]
            or old["state"] != sample["teacher_state"] or old["statistics_sha256"] != scene.stats["statistics_hash"]
            or old["original_band_ids"] != list(scene.bands.band_ids) or old["metadata_quality"] != scene.bands.wavelength_quality
            or (old["budget"], old["prefix"]) != (sample["budget"], sample["prefix"])
            or old["cases"] != cases or old["coverage"] != exposure or old["marginal_match"] is not True
            or old["policy_hashes"] != {k: v.policy_hash for k, v in policies.items()}):
        raise ValueError("Admitted observation source, metadata, exposure or policy differs")
    for name, policy in policies.items():
        pin = old["policies"][name]
        if pin["path"] != str(Path(old["root"])/f"{name}.json") or admission.read(pin) != policy.to_dict():
            raise ValueError("Wrong admitted observation policy")
    if old["scope"] == REAL and (len(cases), old["budget"], old["prefix"], old["preflight_forwards"]) != (12, 178, 7, 48):
        raise ValueError("Only the 12 reviewed real cases at the frozen 178-step budget are authorized")
    trials = {c["case"]: (c, policies[c["order"]]) for c in cases}
    protected = [prior, old["sample_registration"], *old["policies"].values(), *sample["protected"].values()]
    design = dict(prior=prior, sample_registration=old["sample_registration"], scope=old["scope"], cases=cases,
                  budget=old["budget"], prefix=old["prefix"], maximum_updates=2*old["budget"]*len(cases),
                  preflight_forwards=old["preflight_forwards"], validation_conditions=list(CONDITIONS),
                  validation_identity={name: policies[ORDERS[0]].to_dict()["validation"][name]["record_hash"] for name in CONDITIONS},
                  source=sample["source"], state=old["state"], protocol=sample["protocol"], model_seed=sample["model_seed"],
                  class_ids=sample["class_ids"], train_support=sample["support"], validation=sample["validation"],
                  exposures=sample["exposures"], policy_hashes=old["policy_hashes"], source_replay=sample["source_replay"],
                  protection=sample["protection"], selection=sample["selection"], protected=protected)
    return design, source, scene, record, plans, trials


def register(prior, output, code):
    require_frozen_code(code)
    started = perf_counter()
    design, source, scene, record, plans, trials = inputs(prior, code)
    root = external_directory(output)
    if root.exists() or any(root == Path(p["path"]) or root in Path(p["path"]).parents for p in design["protected"]):
        raise ValueError("Fresh observation execution output required")
    contracts = {name: new_run(source, scene, record, plans[str(trial[0]["seed"])], trial).contract_hash
                 for name, trial in trials.items()}
    admission.check_pins(design["protected"])
    require_frozen_code(code)
    root.mkdir()
    return dev._json_new(root/"registration.json", dict(format=FORMAT, root=str(root), commit=code["commit"],
        source_hash=code["source_hash"], runtime=runtime(), **design, contracts=contracts, optimizer_updates=0, query=0, gt=0,
        preparation_seconds=perf_counter()-started))


def load_registration(pin, code):
    require_frozen_code(code)
    reg = admission.read(pin)
    root = external_directory(reg["root"])
    if (reg["format"] != FORMAT or pin["path"] != str(root/"registration.json")
            or (reg["commit"], reg["source_hash"]) != (code["commit"], code["source_hash"])
            or reg["runtime"] != runtime() or any(reg[k] for k in ("optimizer_updates","query","gt"))):
        raise ValueError("Observation execution registration does not match this clean producer")
    design, source, scene, record, plans, trials = inputs(reg["prior"], code)
    if any(reg[k] != v for k, v in design.items()) or set(reg["contracts"]) != set(trials):
        raise ValueError("Observation source or registered budget differs")
    return root, reg, source, scene, record, plans, trials


def execute(pin, code, notice, case, stage, prior):
    root, reg, source, scene, record, plans, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    if case not in trials:
        raise ValueError("Unknown observation execution case")
    trial = trials[case]
    return execute_stage(root, pin, case, stage, prior, value=reg, baseline=source, scene=scene, record=record,
                         plan=plans[str(trial[0]["seed"])], arms=trials, stages=stages(reg["budget"], reg["prefix"]),
                         run_factory=new_run, run_type=MeasuredObservationRun)


def complete(pin, code, notice):
    root, reg, _, _, _, _, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    intervals = stages(reg["budget"], reg["prefix"])
    attempts, updates = {}, 0
    for name in trials:
        receipts = {}
        for stage, (_, _, previous) in intervals.items():
            status, receipts[stage] = read_stage(root, pin, name, stage, receipts.get(previous), arms=trials, stages=intervals)
            updates += status["new_updates"]
        attempts[name] = receipts
    if updates != reg["maximum_updates"]:
        raise ValueError("Actual observation updates differ from the registration")
    admission.check_pins([pin, notice, *reg["protected"]])
    require_frozen_code(code)
    return dev._json_new(root/"batch-complete.json", dict(registration=pin, notice=notice, attempts=attempts,
        updates=updates, optimizer_updates=0, query=0, gt=0))

"""Register the accepted Xuzhou source and run the six unchanged CUR-S arms."""

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import os

import torch

from compression.precision_fallback import LIMITS, SMALL_CLASS_MAX_SUPPORT
from compression.precision_recovery_audit import independent_predictions
from compression.registered_low_rank_recovery import ORIGIN_SHA256, verify_notice
from compression.registered_precision_recovery import _historical, stages
from inference.light_cnn import validate_source
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev, sample_curriculum as sample
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import LR, SEED, execute_stage, read_stage


FORMAT = "wp01_xuzhou_k5_v5_sample_curriculum_v1"
REAL = "wp08_xuzhou_off"
EXPOSURE_WITNESS = "1c13c895555d410e359d46cf0dc76d2a48873f65f1253478175ea5671d8152f0"
ARMS = ("off", "balanced", "fixed", "random", "easy_to_hard", "hard_to_easy")
SEEDS = (64, 512, 2048)
PACE = dict(seed=512, start_fraction=.2, minimum_weight=.1)
REVIEW = dict(version=1, difficulty="fused_teacher_nll_ascending_ties_global_id",
              bins="floor(3*zero_based_class_rank/class_support_count)",
              outcomes="train_fused_nll_and_hard_error_for_selected_and_last",
              effective_supervision="sum_applied_fp32_weights_per_sample",
              devval="separate_per_support_paired_error_changes_all_heads",
              association="descriptive_within_class_spearman_average_ties",
              history="exact_replay_and_hashes_with_forward_replay_at_endpoints")


def runtime():
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    threads = {k: os.environ.get(k) for k in keys}
    if torch.get_num_threads() != 1 or set(threads.values()) != {"1"} or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Registered single-thread CPU runtime required")
    return dict(admission.runtime_identity(), threads=threads, interop=torch.get_num_interop_threads())


def source_inputs(pin):
    req = admission.read(pin)
    if req.get("scope") == REAL and set(req) == {"scope", "origin", "exposure_witness"}:
        if req["origin"]["sha256"] != ORIGIN_SHA256 or req["exposure_witness"]["sha256"] != EXPOSURE_WITNESS:
            raise ValueError("Original WP08 off source and reviewed exposure witness required")
        source, scene, record, info, protected, _ = _historical(req["origin"])
        witness = admission.read(req["exposure_witness"])
        counts = Counter(row["role"] for row in record["roles"])
        if (scene.dataset != "xuzhou" or scene.classes != 9 or len(scene.bands.band_ids) != 436
                or (counts["devtrain"], counts["devval"]) != (73, 5)
                or [(len(record[k]["indices"]), record[k]["seed"]) for k in ("train_support", "validation_support")]
                != [(45, 512), (45, 3031)]
                or witness["source"]["origin"] != req["origin"] or witness["source"]["weights"] != info["weights"]
                or witness["recovery_seeds"] != list(SEEDS) or witness["rejected_seeds"]
                or (witness["budget"], witness["prefix"]) != (178, 7)):
            raise ValueError("Original 436-band Xuzhou K5/V5 roles or recovery exposure differs")
        plans = {str(seed): ExposurePlan(scene, record, witness["exposures"][str(seed)],
                                       expected_plan_hash=witness["exposures"][str(seed)]["plan_hash"]) for seed in SEEDS}
        protected = {**protected, req["exposure_witness"]["path"]: req["exposure_witness"]}
        design = dict(seeds=list(SEEDS), budget=178, prefix=7, coverage=witness["coverage"],
                      protocol="wp01_xuzhou_k5_v5", model_seed=64)
    elif req.get("scope") == "synthetic" and set(req) == {"scope", "weights", "sidecar", "development", "budget", "prefix"}:
        scene, record, files = dev.load_development(req["development"])
        if (type(req["budget"]) is not int or type(req["prefix"]) is not int
                or not 1 <= req["prefix"] < req["budget"] <= 16 or max(scene.plan["shape"]) > 160):
            raise ValueError("Bounded synthetic curriculum fixture required")
        info = {k: req[k] for k in ("weights", "sidecar", "development")}
        source = dev._load_source(info["weights"], info["sidecar"])
        protected = {p["path"]: p for p in (*files, info["weights"], info["sidecar"], info["development"])}
        plans = {"64": ExposurePlan.freeze(scene, record, updates=req["budget"], seed=64,
                  minimum_class_blocks={str(c): 1 for c in record["classes"]}, minimum_support_visits=1)}
        design = dict(seeds=[64], budget=req["budget"], prefix=req["prefix"], coverage={}, protocol="synthetic", model_seed=64)
    else:
        raise ValueError("Explicit authorized sample curriculum source required")
    validate_source(source)
    if source.backbone.norm_type != "group_norm":
        raise ValueError("Original GroupNorm source required")
    for seed, plan in plans.items():
        plan.require_eligible()
        if plan.to_dict()["seed"] != int(seed) or plan.updates != design["budget"]:
            raise ValueError("Exposure witness seed or budget differs")
    support = record["train_support"]
    ownership = {}
    width = scene.plan["shape"][1]
    for row in record["roles"]:
        if row["role"] != "devtrain":
            continue
        y0, y1, x0, x1 = row["box"]
        for index in support["indices"]:
            y, x = divmod(index, width)
            if y0 <= y < y1 and x0 <= x < x1:
                if index in ownership:
                    raise ValueError("Ambiguous teacher score on overlapping train cores")
                ownership[index] = row["id"]
    if set(ownership) != set(support["indices"]):
        raise ValueError("Teacher support omitted from train cores")
    cases = [dict(case=f"s{seed}-{arm}", seed=seed, arm=arm) for seed in design["seeds"] for arm in ARMS]
    design.update(scope=req["scope"], source=info, protected={**protected, pin["path"]: pin}, cases=cases,
                  maximum_updates=2 * design["budget"] * len(cases), selection="first_min_full_devval_fused_NLL_each_update",
                  exposures={seed: plan.to_dict() for seed, plan in plans.items()}, teacher_state=state_hash(source),
                  support=[dict(index=i, label=c, tile=ownership[i]) for i, c in zip(support["indices"], support["labels"], strict=True)],
                  scoring="one_fused_clipped_nll_per_support_unique_devtrain_core",
                  validation=record["validation_support"], class_ids=record["classes"],
                  preprocessing=dict(plan=scene.plan["plan_hash"], stats=scene.stats["statistics_hash"],
                                     bands=scene.bands.fingerprint),
                  review=REVIEW, pacing=dict(PACE, full_at=design["budget"]),
                  protection=dict(limits={k: float(v) for k, v in LIMITS.items()},
                                  small_support_max=SMALL_CLASS_MAX_SUPPORT, small_new_errors_max=0))
    return design, source, scene, record, plans


def policies(teacher, scene, record, design):
    if teacher["teacher_state"] != design["teacher_state"] or teacher["forward_tiles"] != len({r["tile"] for r in design["support"]}):
        raise ValueError("Teacher must be the original state and score only occupied train cores")
    result = {"off": None}
    for arm in ARMS[1:]:
        policy = sample.SampleCurriculum.freeze(teacher, updates=design["budget"], order=arm, **design["pacing"])
        policy.validate(scene, record)
        result[arm] = policy
    return result


class RegisteredSampleRun(CurriculumRun):
    FORMAT = "vca-wp01-sample-curriculum-recovery-v1"
    FAILURE_FORMAT = "vca-wp01-sample-curriculum-failure-v1"

    def __init__(self, source, scene, record, *, trial, teacher_hash, **kwargs):
        self.trial = trial
        self.teacher_hash = teacher_hash
        self.timings = defaultdict(list)
        super().__init__(source, scene, record, **kwargs)

    def _contract_extensions(self):
        return dict(trial=self.trial, teacher_hash=self.teacher_hash, applied_weight_history="fp32_loss_mask")

    def _evaluate(self, model):
        before = perf_counter()
        result = super()._evaluate(model)
        self.timings["full_v5"].append(perf_counter() - before)
        return result

    def _training_forward(self, tile, step, extra):
        before = perf_counter()
        result = super()._training_forward(tile, step, extra)
        self.timings["training_forward"].append(perf_counter() - before)
        return result

    def step(self):
        def observe(outputs, labels, mask, **kwargs):
            positive = mask[:, 0] > 0
            if (not torch.equal(mask, mask[:, :1].expand_as(mask)) or int(positive.sum()) != len(event["support_ids"])
                    or labels.shape != mask.shape):
                raise ValueError("Original loss mask differs from the registered support event")
            self.applied_weights = mask[positive, 0].detach().tolist()
            return loss(outputs, labels, mask, **kwargs)

        if self.completed == self.plan.updates:
            return super().step()
        event = self.plan.event(self.completed + 1)
        loss = sample.probability_loss_components
        with patch.object(sample, "probability_loss_components", observe):
            return super().step()

    def _history_extensions(self, outputs):
        return {"applied_weights": self.applied_weights}

    def _history_extension_keys(self):
        return {"applied_weights"}

    def _validate_history_extensions(self, row):
        if row["applied_weights"] != torch.tensor(row["weights"], dtype=torch.float32).tolist():
            raise ValueError("Actual loss weights differ from the recorded schedule")

    @classmethod
    def load(cls, source, scene, record, path, *, plan, trial, selector_seed, expected_sha256, expected_contract_hash):
        case, policy, teacher_hash = trial
        if selector_seed != SEED:
            raise ValueError("Unexpected stage selector identity")
        return super().load(source, scene, record, path, plan=plan, policy=policy, expected_sha256=expected_sha256,
                            expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(trial=case, teacher_hash=teacher_hash))


def new_run(source, scene, record, plan, trial):
    case, policy, teacher_hash = trial
    return RegisteredSampleRun(source, scene, record, plan=plan, policy=policy, lr=LR, torch_seed=case["seed"],
                               expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
                               expected_policy_hash=None if policy is None else policy.policy_hash,
                               trial=case, teacher_hash=teacher_hash)


def register(request, out, code):
    require_frozen_code(code)
    env = runtime()
    design, source, scene, record, plans = source_inputs(request)
    out = external_directory(out)
    if any(out == Path(p) or out in Path(p).parents or Path(p).parent == out for p in design["protected"]):
        raise ValueError("Fresh registration outside protected inputs required")
    original = independent_predictions(source, scene, record)
    old = design["source"].get("historical_baseline")
    if old is not None and (original["metrics"] != old["metrics"] or original["scores"] != old["scores"]):
        raise ValueError("Original three-head outputs differ from the accepted source")
    teacher = sample.score_training_support(source, scene, record, expected_teacher_state=design["teacher_state"])
    arms = policies(teacher, scene, record, design)
    contracts = {case["case"]: new_run(source, scene, record, plans[str(case["seed"])],
                 (case, arms[case["arm"]], teacher["score_hash"])).contract_hash for case in design["cases"]}
    admission.check_pins(design["protected"].values())
    require_frozen_code(code)
    out.mkdir()
    teacher_pin = dev._json_new(out / "teacher.json", teacher)
    return dev._json_new(out / "registration.json", dict(format=FORMAT, root=str(out), request=request, commit=code["commit"],
                         source_hash=code["source_hash"], runtime=env, **design, teacher=teacher_pin,
                         source_replay={k: original[k] for k in ("metrics", "scores", "forwards")},
                         policies={arm: None if p is None else p.policy_hash for arm, p in arms.items()},
                         contracts=contracts, optimizer_updates=0, query=0, gt=0))


def load_registration(pin, code):
    require_frozen_code(code)
    reg = admission.read(pin)
    root = external_directory(reg["root"])
    if (reg["format"] != FORMAT or pin["path"] != str(root / "registration.json") or reg["runtime"] != runtime()
            or reg["teacher"]["path"] != str(root / "teacher.json")
            or (reg["commit"], reg["source_hash"]) != (code["commit"], code["source_hash"])
            or any(reg[k] != 0 for k in ("optimizer_updates", "query", "gt"))):
        raise ValueError("Wrong curriculum registration, runtime or producer")
    design, source, scene, record, plans = source_inputs(reg["request"])
    teacher = admission.read(reg["teacher"])
    arms = policies(teacher, scene, record, design)
    if (any(reg[k] != v for k, v in design.items())
            or reg["policies"] != {arm: None if p is None else p.policy_hash for arm, p in arms.items()}
            or set(reg["contracts"]) != {case["case"] for case in reg["cases"]}):
        raise ValueError("Curriculum source, exposure, teacher, or policy differs")
    trials = {case["case"]: (case, arms[case["arm"]], teacher["score_hash"]) for case in reg["cases"]}
    return root, reg, source, scene, record, plans, trials


def execute(pin, code, notice, case, stage, prior):
    root, reg, source, scene, record, plans, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    if case not in trials:
        raise ValueError("Unknown registered curriculum case")
    plan = plans[str(trials[case][0]["seed"])]
    return execute_stage(root, pin, case, stage, prior, value=reg, baseline=source, scene=scene, record=record, plan=plan,
                         arms=trials, stages=stages(reg["budget"], reg["prefix"]),
                         run_factory=new_run, run_type=RegisteredSampleRun)


def complete(pin, code, notice):
    root, reg, _, _, _, _, trials = load_registration(pin, code)
    verify_notice(root, reg, pin, notice)
    intervals = stages(reg["budget"], reg["prefix"])
    attempts, updates = {}, 0
    for case in trials:
        receipts = {}
        for stage, (_, _, preceding) in intervals.items():
            status, receipts[stage] = read_stage(root, pin, case, stage, receipts.get(preceding), arms=trials, stages=intervals)
            updates += status["new_updates"]
        attempts[case] = receipts
    if updates != reg["maximum_updates"]:
        raise ValueError("Executed updates differ from the frozen count")
    admission.check_pins([pin, notice, reg["teacher"], *reg["protected"].values()])
    require_frozen_code(code)
    return dev._json_new(root / "batch-complete.json", dict(registration=pin, notice=notice, attempts=attempts,
                         updates=updates, query=0, gt=0, optimizer_updates=0))

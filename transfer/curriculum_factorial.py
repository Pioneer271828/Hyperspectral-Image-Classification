"""Explicit synthetic 2x2 curriculum execution. The default runner remains unchanged."""

from pathlib import Path

from compression.registered_precision_recovery import stages
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_observation_admission import ObservationRun, RestrictedObservation, audited_forward, observation_policies
from transfer.curriculum_real import PACE, RegisteredSampleRun, runtime
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.sample_curriculum import SampleCurriculum, score_training_support
from transfer.xuzhou_multiscale import LR, SEED, execute_stage, read_stage


FORMAT = "wp01_curriculum_2x2_synthetic_v1"
ARMS = {"none": (False,False), "s_only": (True,False), "o_only": (False,True), "s_plus_o": (True,True)}
SYNTHETIC_BUDGET = {3:48, 9:4, 22:4}
REAL_PROPOSAL = dict(status="NOT_RUN", source="accepted_wp08_off_246", protocol="xuzhou_K5_V5",
                     seeds=[64,512,2048], updates_per_trajectory=178, resume_prefix=7, proposed_maximum_updates=4272,
                     prerequisites=["independent_s_and_o_evidence_review","separately_authorized_frozen_real_registration",
                                    "same_exposure_and_teacher_identity","predeclared_paired_small_class_guard"],
                     optimizer_updates=0)


class FactorialRun(RegisteredSampleRun):
    FORMAT = "vca-wp01-explicit-synthetic-factorial-v1"
    FAILURE_FORMAT = "vca-wp01-synthetic-factorial-failure-v1"

    def __init__(self, source, scene, record, *, trial, teacher_hash, policy, expected_policy_hash,
                 observation_policy=None, expected_observation_hash=None, **kwargs):
        both = policy is not None and observation_policy is not None
        if (trial not in ARMS or ARMS[trial] != (policy is not None, observation_policy is not None)
                or len(scene.bands.band_ids) > 16 or max(scene.plan["shape"]) > 160):
            raise ValueError("Explicit bounded synthetic factorial arm required")
        super().__init__(source,scene,record,trial=trial,teacher_hash=teacher_hash,
            policy=None if both else policy,expected_policy_hash=None if both else expected_policy_hash,
            observation_policy=observation_policy,expected_observation_hash=expected_observation_hash,**kwargs)
        if both:
            if (not isinstance(policy,SampleCurriculum) or policy.policy_hash != expected_policy_hash
                    or policy.to_dict()["updates"] != self.plan.updates):
                raise ValueError("Frozen factorial sample policy or budget differs")
            policy.validate(scene,record)
            self.policy = SampleCurriculum(policy.to_dict())
            self.spec["policy_hash"] = expected_policy_hash
            self.contract_hash = digest(self.spec)
            self._seal()
        if self.observation_policy is not None:
            self.observation_policy = RestrictedObservation(self.observation_policy.to_dict())
            self.frozen_observations = self.observation_policy.to_dict()["training"]

    def _contract_extensions(self):
        return dict(super()._contract_extensions(), factorial_scope=FORMAT, factor_switches=list(ARMS[self.trial]))

    def _training_forward(self,tile,step,extra):
        if self.observation_policy is None:
            return super()._training_forward(tile,step,extra)
        outputs,self.forward_evidence = audited_forward(self.model,self.scene,self.record,tile,extra["observation_view"],
                                                        self.frozen_observations[step-1])
        return outputs

    def _history_extensions(self,outputs):
        actual = super()._history_extensions(outputs)
        if self.observation_policy is not None:
            actual["forward_evidence"] = self.forward_evidence
        return actual

    def _history_extension_keys(self):
        keys = super()._history_extension_keys()
        return keys if self.observation_policy is None else keys | {"forward_evidence"}

    def _validate_history_extensions(self,row):
        super()._validate_history_extensions(row)
        if self.observation_policy is not None:
            ObservationRun._validate_history_extensions(self,row)

    @classmethod
    def load(cls,source,scene,record,path,*,plan,trial,selector_seed,expected_sha256,expected_contract_hash):
        arm,sample,obs,teacher = trial
        if selector_seed != SEED:
            raise ValueError("Unexpected stage selector")
        return CurriculumRun.load.__func__(cls,source,scene,record,path,plan=plan,policy=sample,
            observation_policy=obs,expected_sha256=expected_sha256,expected_contract_hash=expected_contract_hash,
            constructor_kwargs=dict(trial=arm,teacher_hash=teacher))


def new_run(source,scene,record,plan,trial):
    arm,sample,obs,teacher = trial
    return FactorialRun(source,scene,record,trial=arm,teacher_hash=teacher,plan=plan,policy=sample,lr=LR,
        torch_seed=plan.to_dict()["seed"],expected_source_state=state_hash(source),expected_plan_hash=plan.plan_hash,
        expected_policy_hash=None if sample is None else sample.policy_hash,observation_policy=obs,
        expected_observation_hash=None if obs is None else obs.policy_hash)


def _inputs(pin):
    req = admission.read(pin)
    if set(req) != {"scope","weights","sidecar","development","exposure"} or req["scope"] != "synthetic":
        raise ValueError("Only bounded synthetic factorial input is authorized")
    scene,record,protected = dev.load_development(req["development"])
    if (scene.classes not in SYNTHETIC_BUDGET or len(scene.bands.band_ids) > 16
            or max(scene.plan["shape"]) > 160):
        raise ValueError("Only bounded synthetic geometry and class counts are authorized")
    value = admission.read(req["exposure"])
    plan = ExposurePlan(scene,record,value,expected_plan_hash=value["plan_hash"])
    plan.require_eligible()
    if plan.updates != SYNTHETIC_BUDGET[scene.classes] or value["seed"] != 64:
        raise ValueError("Only the fixed per-fixture budget and seed are authorized")
    source = dev._load_source(req["weights"],req["sidecar"])
    protected = [pin,*protected,*(req[k] for k in ("weights","sidecar","development","exposure"))]
    return source,scene,record,plan,protected


def _policies(source,scene,record,plan,teacher):
    if teacher["teacher_state"] != state_hash(source):
        raise ValueError("Factorial teacher must be the original full-band source")
    sample = SampleCurriculum.freeze(teacher,updates=plan.updates,order="easy_to_hard",full_at=plan.updates,**PACE)
    sample.validate(scene,record)
    obs = observation_policies(scene,record,plan.updates)["easy_to_hard"]
    trials = {name:(name,sample if s else None,obs if o else None,teacher["score_hash"]) for name,(s,o) in ARMS.items()}
    return trials,sample,obs


def register(request,output,code):
    require_frozen_code(code)
    source,scene,record,plan,protected = _inputs(request)
    env = runtime()
    root = external_directory(output)
    if root.exists() or any(root == Path(p["path"]) or root in Path(p["path"]).parents for p in protected):
        raise ValueError("Factorial output must be separate and fresh")
    teacher = score_training_support(source,scene,record,expected_teacher_state=state_hash(source))
    trials,sample,obs = _policies(source,scene,record,plan,teacher)
    contracts = {name:new_run(source,scene,record,plan,v).contract_hash for name,v in trials.items()}
    admission.check_pins(protected)
    require_frozen_code(code)
    root.mkdir()
    pin = dev._json_new(root/"teacher.json",teacher)
    return dev._json_new(root/"registration.json",dict(format=FORMAT,scope="synthetic",root=str(root),request=request,
        commit=code["commit"],source_hash=code["source_hash"],runtime=env,source_state=state_hash(source),
        teacher=pin,protected=protected,exposure_hash=plan.plan_hash,policy_hashes=dict(sample=sample.policy_hash,observation=obs.policy_hash),
        cases=[dict(case=k,sample=s,observation=o) for k,(s,o) in ARMS.items()],contracts=contracts,
        budget=plan.updates,prefix=7 if plan.updates > 7 else 2,maximum_updates=8*plan.updates,
        optimizer_updates=0,query=0,gt=0,real_training="NOT_RUN"))


def load_registration(pin,code):
    require_frozen_code(code)
    r = admission.read(pin)
    root = external_directory(r["root"])
    if (r["format"] != FORMAT or r["scope"] != "synthetic" or pin["path"] != str(root/"registration.json")
            or (r["commit"],r["source_hash"]) != (code["commit"],code["source_hash"]) or r["runtime"] != runtime()
            or r["teacher"]["path"] != str(root/"teacher.json") or any(r[k] for k in ("optimizer_updates","query","gt"))
            or r["real_training"] != "NOT_RUN"):
        raise ValueError("Wrong frozen synthetic factorial registration")
    source,scene,record,plan,protected = _inputs(r["request"])
    teacher = admission.read(r["teacher"])
    trials,sample,obs = _policies(source,scene,record,plan,teacher)
    if (r["protected"] != protected or r["source_state"] != state_hash(source) or r["exposure_hash"] != plan.plan_hash
            or (r["budget"],r["prefix"],r["maximum_updates"]) != (plan.updates,7 if plan.updates > 7 else 2,8*plan.updates)
            or r["policy_hashes"] != dict(sample=sample.policy_hash,observation=obs.policy_hash)
            or r["cases"] != [dict(case=k,sample=s,observation=o) for k,(s,o) in ARMS.items()]
            or set(r["contracts"]) != set(trials)):
        raise ValueError("Factorial source, switches, schedule or budget differs")
    return root,r,source,scene,record,plan,trials


def execute(pin,code,case,stage,prior):
    root,reg,source,scene,record,plan,trials = load_registration(pin,code)
    if case not in trials:
        raise ValueError("Unknown factorial arm")
    return execute_stage(root,pin,case,stage,prior,value=reg,baseline=source,scene=scene,record=record,
        plan=plan,arms=trials,stages=stages(reg["budget"],reg["prefix"]),run_factory=new_run,run_type=FactorialRun)


def complete(pin,code):
    root,reg,_,_,_,_,trials = load_registration(pin,code)
    attempts = {}
    updates = 0
    intervals = stages(reg["budget"],reg["prefix"])
    for case in trials:
        receipts = {}
        for stage,(_,_,previous) in intervals.items():
            row,receipts[stage] = read_stage(root,pin,case,stage,receipts.get(previous),arms=trials,stages=intervals)
            updates += row["new_updates"]
        attempts[case] = receipts
    if updates != reg["maximum_updates"]:
        raise ValueError("Synthetic 2x2 execution differs from the registered update budget")
    admission.check_pins([pin,reg["teacher"],*reg["protected"]])
    return dev._json_new(root/"batch-complete.json",dict(registration=pin,attempts=attempts,updates=updates,
         optimizer_updates=0,query=0,gt=0,real_training="NOT_RUN"))

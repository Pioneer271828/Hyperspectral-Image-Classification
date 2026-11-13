"""Fixed-support recovery on the original three-head curriculum runner."""

from copy import deepcopy

from transfer.curriculum_training import CurriculumRun
from transfer.native_support_intervention import BoundIntervention, forward_intervention
from transfer.native_support_run import NativeSupportRun


class ControlledSupportRun(NativeSupportRun):
    FORMAT = "vca-matched-regional-recovery-v1"
    FAILURE_FORMAT = "vca-matched-regional-recovery-failure-v1"

    def __init__(self, baseline, scene, record, *, intervention, arm, random_trial, **kwargs):
        self.intervention = deepcopy(intervention)
        self.arm = arm
        self.fixed_intervention = {}
        super().__init__(baseline, scene, record, **kwargs)
        if self.intervention["blocks"].keys() != self.train.keys():
            raise ValueError("Incomplete fixed intervention coverage")
        for key, receipt in self.intervention["blocks"].items():
            entry = kwargs["bank"][key]
            bound = BoundIntervention(entry, self.trial, random_trial, receipt, arm,
                                      self.intervention["owner"], expected_sha256=receipt["sha256"])
            if bound.context["role"] != "devtrain" or receipt["reference"]["step"] != self.plan.updates:
                raise ValueError("Recovery requires completed training block supports")
            self.fixed_intervention[key] = bound

    def _contract_extensions(self):
        value = super()._contract_extensions()
        value["fixed_intervention"] = dict(owner=self.intervention["owner"], arm=self.arm,
            receipts={key: row["sha256"] for key, row in self.intervention["blocks"].items()},
            validation="same_original_fixed_step_zero_on_all_arms")
        return value

    def _check_extensions(self):
        super()._check_extensions()
        if self.spec["fixed_intervention"] != self._contract_extensions()["fixed_intervention"]:
            raise ValueError("Fixed intervention changed during the registered trajectory")

    def selector_at(self, key, step):
        return self.fixed_intervention[key]._origin

    def _training_forward(self, tile, step, extra):
        bound = self.fixed_intervention[str(tile["id"])]
        before = bound.cost.copy()
        output, frame = forward_intervention(self.model, self.scene, self.record, tile, bound, timing=self.timings)
        output[-1]["regional_support"] = frame["trace"]
        for key, seconds in bound.cost.items():
            self.timings.setdefault(key, []).append(seconds - before[key])
        return output

    def _validate_history_extensions(self, row):
        bound = self.fixed_intervention[str(row["tile_id"])]
        trace = row["support"]
        if trace.get("intervention") != bound.trace() or trace.get("origin_snapshot") != bound.snapshot():
            raise ValueError("Fixed intervention history or origin differs")
        # Reuse the inherited scorer and native graph shape checks. Supply the
        # genuine controller origin only for the original support-count check.
        if trace["stage2_nnz"] != bound.trace()["nnz"]:
            raise ValueError("Fixed intervention support size differs")
        original = dict(trace, snapshot=trace["origin_snapshot"], stage2_nnz=len(bound.snapshot()["selected"]))
        super()._validate_history_extensions(dict(row, support=original))

    @classmethod
    def load(cls, baseline, scene, record, path, *, plan, trial, selector_seed, expected_sha256,
             expected_contract_hash, bank, intervention, arm, random_trial):
        return CurriculumRun.load.__func__(cls, baseline, scene, record, path, plan=plan, policy=None,
            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
            constructor_kwargs=dict(trial=trial, selector_seed=selector_seed, bank=bank, intervention=intervention,
                                    arm=arm, random_trial=random_trial))

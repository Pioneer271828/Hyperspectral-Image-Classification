"""Opt-in synthetic QAT factor recovery using the original curriculum state machine."""

from compression.low_rank_factor_precision import deploy, validate
from compression.precision_fallback import _validation_inputs
from compression.precision_recovery import validate_deployment
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import LR


class FactorPrecisionRun(CurriculumRun):
    FORMAT = "vca-wp10-factor-qat-recovery-v1"
    FAILURE_FORMAT = "vca-wp10-factor-qat-recovery-failure-v1"

    def __init__(self, source, scene, record, **kwargs):
        self.precision_plan = validate(source)
        if self.precision_plan["phase"] != "qat":
            raise ValueError("Explicit frozen QAT factor source required")
        self.validation_spec, self.validation_blocks = _validation_inputs(scene, record)
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        if validate(model) != self.precision_plan:
            raise ValueError("Factor QAT dependency or observer identity changed")

    def _contract_extensions(self):
        return dict(factor_precision=self.precision_plan, validation_scope=self.validation_spec,
                    selection_representation="deployed_integer_factors")

    def _evaluate(self, model):
        model.eval()
        deployed = deploy(model)
        rows, _ = validate_deployment(deployed, self.scene, self.validation_blocks, arm="fp32")
        return rows["fused"]


def new_run(source, scene, record, plan):
    return FactorPrecisionRun(source, scene, record, plan=plan, policy=None, lr=LR,
                              torch_seed=plan.to_dict()["seed"], expected_source_state=state_hash(source),
                              expected_plan_hash=plan.plan_hash, expected_policy_hash=None)

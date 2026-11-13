"""Opt-in same-capacity selector control for the native joint scale model."""
from dataclasses import dataclass
from itertools import product

from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from transfer.native_scale_fixture import condition_for
from transfer.spatial_multiscale import initialize


SCHEMA = "wp07_native_released_condition_capacity_v1"
CONSTANT = "constant_joint"
CONDITIONS = ("matched", "permuted", "perturbed", "missing")
# Abstract fixed encoder stimulus. It never reaches the spectral VCA or
# represents a sensor, its wavelengths or a physical estimate.
_REFERENCE = BandMetadata("joint_selector_constant_control_v1", (0, 1, 2),
                          (475., 675., 875.), "nm", "verified", (True, True, True),
                          (15., 15., 15.), "nm")


@dataclass(frozen=True)
class ConstantSelector:
    observation: str

    @property
    def source_key(self):
        return (CONSTANT, self.observation)

    @property
    def gsd_m(self):
        return 1.0

    def values(self):
        return _REFERENCE, .5, self.gsd_m

    def validate(self, actual, receipt):
        if self.observation != receipt["observation"] or receipt["role"] not in ("devtrain", "devval"):
            raise ValueError("Constant scale control belongs to another observation")


def condition(actual, kind):
    if kind == CONSTANT:
        return ConstantSelector(actual.receipt["observation"])
    return condition_for(actual, kind)


def model(baseline, arm, seed):
    if arm not in ("joint", CONSTANT):
        raise ValueError("Same-capacity control requires joint or constant_joint")
    return initialize(baseline, "joint", seed)


def draft(original):
    factor_arms = [(2, "joint"), (4, "joint")]
    candidates = [case for case in original["cases"]
                  if any(case["id"].endswith(f"/{factor}/{arm}") for factor, arm in factor_arms)]
    cases = []
    for case, arm in product(candidates, ("joint", CONSTANT)):
        # Real conditions are missing, so no donor or perturbation is
        # manufactured for the released bank. Keep its future release separate.
        new = dict(case, id=case["id"].rsplit("/", 1)[0] + f"/{arm}",
                   arm=arm, condition="missing" if arm == "joint" else CONSTANT)
        cases.append(new)
    return {"schema": SCHEMA, "source_matrix_sha256": digest(original), "cases": cases,
            "planned_optimizer_updates": sum(case["updates"] for case in cases),
            "optimizer_updates": 0, "query_authorized": False, "status": "review_only",
            "synthetic_controls": {"options": list(CONDITIONS),
                "permuted": "Fixed cyclic donor within the same band view, factor and registered synthetic role",
                "seed": "Pair same model seed and source checkpoint; donor cycle has no random draw",
                "missing": "Native joint falls back to its content selector"},
            "constant_descriptor_substitution": "Only the selector input changes, all joint parameters are shared"}

"""Control descriptive accuracy, protection evaluation and formal qualification."""

from evaluation.frozen_training_stages import require
from transfer.native_paired_head_campaign import GATES


def classification_status(result, *, accuracy_allowed, cost_allowed, real):
    if not accuracy_allowed:
        require(result is None,"A cost-only release cannot contain classification scores")
        return dict(raw_metrics=False,protection_evaluable=False,qualified=False,
                    protection_passed=None,cost_eligible=cost_allowed,
                    reason="classification_not_authorized")
    require(result is not None,"An authorized classification record needs released point scores")
    threshold = GATES["minimum_real_confirmation_per_class"]
    minimum = min(record["n"] for record in result["transitions"].values())
    comparable = minimum>=threshold
    if comparable:
        require(result["applied_population_threshold"]==threshold if real else
                result["applied_population_threshold"]<=threshold,
                "Classification gate used a different class floor")
    protection = result["protected"] if comparable else None
    qualified = real and comparable and result["accepted"]
    reason = None
    if not comparable:
        reason = "fewer_than_20_released_points_in_a_class"
    elif not result["accepted"]:
        reason = "preregistered_protection_or_target_gain_not_met"
    elif not real:
        reason = "test_publication"
    return dict(raw_metrics=True,protection_evaluable=comparable,qualified=qualified,
                protection_passed=protection,cost_eligible=cost_allowed,
                minimum_class_n=minimum,minimum_formal_n=threshold,reason=reason)

"""Separate paired measurements, protection, directed effects and release decisions."""

from fractions import Fraction

from evaluation.frozen_training_stages import HEADS, require
from transfer.native_paired_head_campaign import GATES

SEEDS = (64, 512, 2048)
PHYSICAL = "physical_axis_v1"
COLUMNS = "delivered_columns_v1"
PAVIA = "pavia_fused_aa_central_gap_v1"
FOCUS = "registered_focus_recovery_v1"
FLOOR = GATES["minimum_real_confirmation_per_class"]


def registered_policy(scene, observation_protocol, declared):
    if declared is not None:
        require(scene == "paviaU" and declared == PAVIA, "Unregistered prospective scientific claim")
        return declared
    if observation_protocol == PHYSICAL and scene in GATES["targets"]:
        return FOCUS
    return None


def head_counts(head, classes):
    """Use the consumer's per-point transition tally, never its decision flags."""
    table = head["transitions"]
    require(set(table) == {str(i) for i in range(1,classes+1)}, "Missing paired class transition")
    rows = []
    for y in range(1,classes+1):
        row = table[str(y)]
        n,old,new,plus,minus = (row[k] for k in ("n","before","after","gains","losses"))
        require(all(type(x) is int and x>=0 for x in (n,old,new,plus,minus))
                and old<=n and new<=n and plus<=min(n-old,new) and minus<=min(old,n-new)
                and new-old==plus-minus, "Paired class counts disagree")
        rows.append((n,old,new,plus,minus))
    return rows


def view_result(scores, scene, policy):
    if scores is None:
        return dict(comparable=False,protection_passed=None,directed_effect=None,
                    minimum_class_n=None,fused_aa_difference=None,fused_aa_fraction=None,
                    reason="empty_or_unavailable_classification")
    require(set(scores)==set(HEADS), "Each scientific contrast needs all three heads")
    classes = {"xuzhou":9,"paviaU":9,"whu-honghu":22}[scene]
    counts = {h:head_counts(scores[h],classes) for h in HEADS}
    reference = [row[0] for row in counts["fused"]]
    require(all([row[0] for row in column]==reference for column in counts.values()),
            "Head populations differ")
    minimum = min(reference)
    if not minimum:
        return dict(comparable=False,protection_passed=None,directed_effect=None,
                    minimum_class_n=minimum,fused_aa_difference=None,fused_aa_fraction=None,
                    reason="unrepresented_class")
    delta = {h:sum((Fraction(new-old,n) for n,old,new,_,_ in rows),Fraction())/classes
             for h,rows in counts.items()}
    adequate = minimum>=FLOOR
    targets = GATES["targets"].get(scene,[])
    passed = {}
    for h,rows in counts.items():
        checks = []
        for y,(n,old,new,_,lost) in enumerate(rows,1):
            if scene=="whu-honghu" and y in targets:
                checks.append(lost<=GATES["honghu_rare_added_errors"])
            elif policy==PAVIA and y in (2,6):
                checks.append(lost==0)
            else:
                checks.append(Fraction(new-old,n)>=Fraction(str(GATES["ordinary_delta_recall"])))
        passed[h] = all(checks) and (scene not in GATES["targets"]
                    or delta[h]>=Fraction(str(GATES["aggregate_delta"])))
    effect = None
    if policy==FOCUS:
        focus = [counts["fused"][y-1] for y in targets]
        changes = [new-old for _,old,new,_,_ in focus]
        effect = all(v>=0 for v in changes) and sum(changes)>=GATES["target_correct_gain_min"]
    if policy==PAVIA:
        effect = delta["fused"]>0
    return dict(comparable=True,protection_evaluable=adequate,
                protection_passed=all(passed.values()) if adequate else None,
                directional_evaluable=policy is not None and adequate,
                directed_effect=effect if adequate else None,minimum_class_n=minimum,
                fused_aa_difference=float(delta["fused"]),
                fused_aa_fraction=[delta["fused"].numerator,delta["fused"].denominator],
                reason=None if adequate else "fewer_than_20_per_class")


def paired_result(pair, cohort, scene, protocol, declared, expected_views):
    label = f'{pair["baseline"]}/{pair["candidate"]}'
    policy = registered_policy(scene,protocol,declared)
    views = cohort["pairs"].get(label,{})
    complete = set(views)==set(expected_views) and bool(expected_views)
    scores = {band:view_result(views.get(band,{}).get("accuracy"),scene,policy) for band in expected_views}
    return label,dict(capacity=pair["comparison"],policy=policy,views=scores,
                      comparable=complete and all(row["comparable"] for row in scores.values()),
                      protection_passed=complete and all(row["protection_passed"] is True for row in scores.values()),
                      complete_views=complete)


def conclusion(seeds, required_views, *, authentic_real):
    """Only a complete preregistered cohort can support an adoption."""
    actual = [entry["seed"] for entry in seeds]
    require(len(set(actual))==len(actual),"Repeated initialization inside a scientific cohort")
    policies = {entry["policy"] for entry in seeds}
    require(len(policies)<=1,"Scientific policy changed between model seeds")
    policy = next(iter(policies),None)
    complete = (set(actual)==set(SEEDS) and set(required_views)>={"full","central_gap","matched_dispersed"}
                and all(row["complete_views"] for row in seeds))
    comparable = bool(seeds) and all(row["comparable"] for row in seeds)
    protected = bool(seeds) and all(row["protection_passed"] for row in seeds)
    direction = None
    if policy is not None:
        direction = False
    if policy==FOCUS and complete:
        direction = all(view["directed_effect"] is True for row in seeds for view in row["views"].values())
    if policy==PAVIA and complete and comparable:
        primary = [Fraction(*s["views"]["central_gap"]["fused_aa_fraction"]) for s in seeds]
        direction = (sum(v>0 for v in primary)>=2 and sum(primary)>0
                     and all(Fraction(*row["fused_aa_fraction"])>=0 for s in seeds
                             for row in s["views"].values()))
    eligible = bool(complete and comparable and protected and direction)
    return dict(policy=policy,expected_model_seeds=list(SEEDS),observed_model_seeds=sorted(actual),
                preregistered_claim=policy is not None,complete=complete,comparable=comparable,
                protection_passed=protected,directed_effect=direction,adoption_requirements_met=eligible,
                adopted=authentic_real and eligible,criteria=seeds)

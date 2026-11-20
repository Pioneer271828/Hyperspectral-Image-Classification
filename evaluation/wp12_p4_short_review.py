"""Point-matched, development-only reporting for the original P4 short roster."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from evaluation import wp12_p4_migration as migration
from evaluation.frozen_training_stages import json_pin, pinned
from evaluation.wp12_cross_backend import FORMAL_FLOOR
from evaluation.wp12_p4_progress import case_stages, observe_lock
from evaluation.wp12_real_readiness import transition
from scripts.review_wp05_p4_real import case_labels
from transfer import p4_coverage_review as coverage_review
from transfer import p4_frozen_diagnostic_metrics as metrics
from transfer.p4_resource_migration import release_for
from transfer.p4_three_arm_contract import PROTECTION

require = migration.require
HEADS = migration.HEADS
CLASSES = tuple(range(1, 23))
FOCUS = (15, 21)
PAIR_ARMS = ("anchor_only", "binary", "continuous", "shuffled")
SCHEMA = "wp12_honghu_p4_short_development_review_v1"

CONTEXT = {
    "paviaU": ("wp07-paviau-real-results-20261118.TC3eXtC5/aggregate/results/comparison-audit.json",
               "728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb"),
    "xuzhou": ("wp07-xuzhou-multiscale-real01/audit.json",
               "29ed4f67bf08f13070c20fec21ecb77f834a55778c5a090121737dcba0705a32"),
    "honghu": ("wp05-coverage-real-summary01/overview.json",
               "50d51d93d8b42b45276d903be51032db578dbe988b92034ca432f83a333a22f9"),
}


def scene_context(root):
    pins = {tag: dict(path=str(root / name), sha256=sha) for tag, (name, sha) in CONTEXT.items()}
    pavia, xuzhou, honghu = (json_pin(root,pins[key]) for key in ("paviaU","xuzhou","honghu"))
    require(pavia["status"] == xuzhou["status"] == "passed" and pavia["synthetic"] is False
            and pavia["query_accessed"] is pavia["raw_gt_loaded"] is False
            and xuzhou["raw_gt"] == xuzhou["query"] == xuzhou["independent_updates"] == 0,
            "Historical scene scope is not approved or still requires a sealed read")
    require(pavia["scope"] == "released_development_only" and pavia["denominators"] == [5]*9
            and pavia["case_count"] == 39 and honghu["optimizer_updates"] == 1920,
            "Prior PaviaU or HongHu case scope changed")
    xuzhou_baseline = xuzhou["results"]["off"]["endpoints"]["selected"]["metrics"]["fused"]["confusion_matrix"]
    xuzhou_counts = np.asarray(xuzhou_baseline,dtype=np.int64).sum(1).tolist()
    require(len(xuzhou_counts)==9 and all(n>0 for n in xuzhou_counts),
            "Historical Xuzhou archive is missing supported development classes")
    return {
        "paviaU": dict(receipt=pins["paviaU"], cases=pavia["case_count"], steps=pavia["optimizer_updates"],
                       devval_by_class=pavia["denominators"], observations=pavia["bands"],
                       seeds=sorted(map(int,pavia["seed_gpus"])), endpoints=pavia["endpoints"],
                       role=pavia["scope"], scope="native_released_point_multiscale"),
        "xuzhou": dict(receipt=pins["xuzhou"], registration=xuzhou["registration"],
                       cases=len(xuzhou["results"]), steps=sum(r["updates"] for r in xuzhou["results"].values()),
                       arms=sorted(xuzhou["results"]), devval_by_class=xuzhou_counts,
                       role="calibrated_spatial_development",
                       scope="matched_multiscale_512_step_arms"),
        "whu-honghu": dict(receipt=pins["honghu"], steps=honghu["optimizer_updates"],
                           observation_columns=[honghu["source_bands"],honghu["retained_bands"]],
                           population=honghu["validation_population_sha256"],
                           scope="separate_historical_five_mode_short_runs"),
    }


def quantiles(values):
    data = np.asarray(values,dtype=np.float64)
    require(len(data) and np.isfinite(data).all() and (data >= 0).all(), "Malformed P4 timing sample")
    median, p95 = np.quantile(data,[.50,.95])
    return dict(samples=len(data),p50_seconds=float(median),p95_seconds=float(p95))


def cost(audit, source):
    timings = audit["synchronized_timing"]
    result = {}
    for path in ("full","split"):
        stage = timings[path]
        steps = [entry for call in stage["advance"] for entry in call["steps"]]
        completed = stage["completed"]
        require(len(steps) == source[path]["optimizer_updates"] == 60
                and source[path]["completed_resume_updates"] == 0,
                "A cost statement lost an acknowledged step or completed-reentry guard")
        memory = [row["train_peak_memory"] for row in steps]
        require(all(0 < m["allocated_bytes"] <= m["reserved_bytes"] for m in memory),
                "Invalid CUDA peak memory samples")
        result[path] = dict(**source[path], worker_synchronized_step=quantiles(
            [r["end_to_end_step_seconds"] for r in steps]),
            peak_allocated_bytes=max(m["allocated_bytes"] for m in memory),
            peak_reserved_bytes=max(m["reserved_bytes"] for m in memory))
        require(result[path]["completed_resume_seconds"] == completed["call_seconds"],
                "Completed-state timing differs from the independent review")
    return result


def archived_mechanism(case, registration, state, audit):
    trace = state["trace"]
    require([event["step"] for event in trace] == list(range(1,61)),
            "Archived P4 coverage evidence must include every optimizer step")
    coverage = registration["coverage"]
    initialization = audit["initialization"]["coverage"]
    active = case["arm"] != "none"
    require(coverage["metadata"] == "nominal" and initialization["arm"] == case["arm"],
            "P4 coverage configuration differs from the archived arm")
    records = coverage["records"]
    participation = coverage_review.participation(trace,audit["initialization"],records["train"])
    require(participation == audit["coverage_participation"],
            "Archived optimizer participation disagrees with the independent audit")
    for name,evidence in participation["parameters"].items():
        observed = [(event["step"],event["coverage"]["parameters"][name]) for event in trace]
        evidence["first_gradient_step"] = next(
            (step for step,value in observed if (value["gradient_l1"] or 0) > 0),None)
        evidence["first_change_step"] = next(
            (step for step,value in observed if value["changed_elements"] > 0),None)
    graphs = {}
    for stage in ("initial","selected","last"):
        graphs[stage] = {}
        for role in ("train","val"):
            rows = audit["fit"][stage][role]["graphs"]
            require(rows and len({r["tile"] for r in rows}) == len(rows),
                    "Duplicate or absent audited graph tiles")
            record = records[role]
            require(all((r["bands"],r["metadata_sha256"],r["coverage_active"]) ==
                        (len(record["selected_positions"]),record["selected_metadata_hash"],active) for r in rows),
                    "Archived graph view differs from the registered retained columns or coverage arm")
            graphs[stage][role] = rows
    return dict(metadata_mode=coverage["metadata"],
                observations={role:{field:record[field] for field in (
                    "source_band_count","source_metadata_hash","selected_positions","selected_band_ids",
                    "selected_metadata_hash","record_hash")} for role,record in records.items()},
                initialized_operator=initialization.get("operator"), optimizer_participation=participation,
                first_all_train_classes_step=audit["first_all_classes_step"], graphs=graphs)


def graph_comparison(base, candidate):
    result = {}
    for stage,roles in base.items():
        result[stage] = {}
        for role,before in roles.items():
            after = candidate[stage][role]
            require([row["tile"] for row in before] == [row["tile"] for row in after],
                    "P4 graph comparison requires the same ordered tiles")
            result[stage][role] = dict(tiles=len(before),
                q_changed=sum(a["q_sha256"] != b["q_sha256"] for a,b in zip(before,after,strict=True)),
                a_changed=sum(a["a_sha256"] != b["a_sha256"] for a,b in zip(before,after,strict=True)))
    return result


def accepted_scores(root, row, labels, audit, registration):
    folder = Path(row["audit"]["path"]).parent
    artifacts = audit["artifact_pins"]
    best = None
    paths = ("split","full") if row["source"] == "migrated" else ("split",)
    for lane in paths:
        path = folder / lane / "state.pt"
        pin = dict(path=str(path),sha256=artifacts[str(path)])
        state = torch.load(pinned(root,pin),map_location="cpu",weights_only=True)
        history = state["history"]
        require(state["cursor"] == 60 and [e["step"] for e in history] == [30,60]
                and state["runtime"]["scope_sha256"] == row["scope"]["sha256"]
                and state["fingerprints"] == audit["exact_path_fingerprints"][lane]
                and audit["exact_path_fingerprints"]["split"] == audit["exact_path_fingerprints"]["full"],
                "Persisted P4 state does not match the accepted recovery fingerprints and calendar")
        selected = min(history, key=lambda event: event["metrics"]["fused"]["nll"])
        require(selected["step"] == state["best"]["step"] == audit["selected_step"] == row["selected_step"],
                "P4 output differs from the first minimum on the registered clock")
        probabilities = {h: v.numpy() for h, v in state["best"]["scores"].items()}
        require(set(probabilities) == set(HEADS), "Incomplete selected three-head archive")
        if best is None:
            best = probabilities
            nll = [e["metrics"]["fused"]["nll"] for e in history]
            mechanism = archived_mechanism(row["case"],registration,state,audit)
        else:
            require(all(np.array_equal(best[head],probabilities[head]) for head in HEADS),
                    "Migrated full and split selected scores disagree")
    for head,values in best.items():
        actual = metrics.recompute(values,labels)
        expected = {k:row["heads"][head][k] for k in actual}
        require(metrics.equivalent(actual,expected), "Selected point decisions disagree with the accepted three-head metrics")
    return best,nll,mechanism


def nearest_other_labels(layout, truth):
    """Only spatial neighbours among differently labelled points in the same released block."""
    result = []
    offset = 0
    for tile,cells in layout:
        n = len(cells)
        labels = truth[offset:offset+n]
        width = tile["box"][3] - tile["box"][2]
        xy = np.stack(np.divmod(cells,width),axis=1)
        delta = xy[:,None,:]-xy[None,:,:]
        squared = (delta*delta).sum(2)
        for i,label in enumerate(labels):
            allowed = labels != label
            if allowed.any():
                nearest = int(squared[i][allowed].min())
                tied = allowed & (squared[i] == nearest)
                result.append((sorted(set(map(int,labels[tied]))),nearest))
            else:
                result.append(([],None))
        offset += n
    require(offset == len(truth), "Neighbour geometry and validation roles differ")
    return result


def focus_readout(labels, probabilities, label, neighbours=None):
    block = probabilities[labels == label].astype(np.float64)
    rivals = block.copy()
    rivals[:,label-1] = -np.inf
    best_other = rivals.argmax(1)
    predicted = block.argmax(1) + 1
    wrong = Counter(int(i) for i in predicted if i!=label)
    result = dict(n=len(block),correct=int((predicted==label).sum()),
                  misclassified_into=dict(sorted(wrong.items())),
                  largest_competing_class=dict(sorted(Counter(int(i)+1 for i in best_other).items())),
                  mean_signed_probability_margin=float((block[:,label-1]-rivals[np.arange(len(block)),best_other]).mean()))
    if neighbours is not None:
        comparable = [(classes,distance,int(pred)) for j,pred in zip(np.flatnonzero(labels==label),predicted,strict=True)
                      for classes,distance in (neighbours[j],) if classes]
        errors = [pred in classes for classes,_,pred in comparable if pred != label]
        result["nearest_released_other_class"] = dict(points_with_neighbour=len(comparable),
            wrong_with_neighbour=len(errors),wrong_predicted_one_of_nearest=int(sum(errors)),
            ties=sum(len(classes)>1 for classes,_,_ in comparable),
            median_distance_pixels=None if not comparable else float(np.median(
                [np.sqrt(distance) for _,distance,_ in comparable])))
    return result


def decision_margins(baseline, candidate):
    before,after = (scores.astype(np.float64,copy=True) for scores in (baseline,candidate))
    winner = before.argmax(1)
    index = np.arange(len(winner))
    shift = np.abs(after-before).max(1)
    prior_confidence,later_confidence = before.max(1),after.max(1)
    prior_winner,later_winner = before[index,winner],after[index,winner]
    before[index,winner] = -np.inf
    after[index,winner] = -np.inf
    gap = prior_winner-before.max(1)
    subsequent_gap = later_winner-after.max(1)
    return dict(points_with_score_change=int(np.count_nonzero(shift)),
                points_with_sufficient_margin=int(np.count_nonzero(gap > 2*shift)),
                points_with_baseline_tie=int(np.count_nonzero(gap == 0)),
                points_with_baseline_winner_still_strictly_first=int(np.count_nonzero(subsequent_gap > 0)),
                baseline_gap_min=float(gap.min()),baseline_gap_median=float(np.median(gap)),
                candidate_gap_to_baseline_winner_min=float(subsequent_gap.min()),
                candidate_gap_to_baseline_winner_median=float(np.median(subsequent_gap)),
                max_absolute_point_score_change=float(shift.max()),
                mean_max_probability=[float(prior_confidence.mean()),float(later_confidence.mean())])


def threshold_comparison(labels, baseline, candidate, base_metrics, candidate_metrics):
    """Compare actual saved point decisions and report the registered descriptive limits."""
    a,b = (value.argmax(1)+1 for value in (baseline,candidate))
    left,right = ({i:(int(label),int(predicted)) for i,(label,predicted) in
                   enumerate(zip(labels,predictions,strict=True))} for predictions in (a,b))
    delta = metrics.difference(candidate,baseline,candidate_metrics,base_metrics)
    low = [i for i,r in zip(CLASSES,delta["recall"],strict=True)
           if 100*r < -PROTECTION["max_small_recall_drop_pp"]]
    return dict(delta=delta,decision_margins=decision_margins(baseline,candidate), recall_loss_classes_gt_2pp=low,
                oa_aa_limits_passed=all(100*delta[m]>=-PROTECTION["max_oa_aa_drop_pp"] for m in ("oa","aa")),
                focus={str(label):transition(left,right,label) for label in FOCUS})


def cohort_key(case):
    return (case["support_seed"],case["k"],case["method"],case["model_seed"],case["sampler_seed"])


def roster_pair(name, lines, probability):
    identity = lambda row: (row["pair_sha256"], row["validation"]["sha256"],row["support"]["sha256"],
                            row["observation"],row["physical_uuid"],row["scope"])
    require(set(lines) == migration.ARMS and len({identity(r) for r in lines.values()}) == 1,
            f"Case arms do not share frozen initialization, support, device and validation: {name}")
    comparisons = {}
    for base,arms in (("none",PAIR_ARMS),("anchor_only",("binary","continuous","shuffled"))):
        first = lines[base]
        labels,before = probability[first["case"]]
        for arm in arms:
            other = lines[arm]
            truth,after = probability[other["case"]]
            require(np.array_equal(labels,truth), "Different released point sequence in a claimed P4 pair")
            comparisons[f"{base}/{arm}"] = {h:threshold_comparison(labels,before[h],after[h],
                first["heads"][h],other["heads"][h]) for h in HEADS}
    return comparisons


def descriptive_cross_cohorts(groups):
    result = []
    for (a, left), (b, right) in combinations(sorted(groups.items()),2):
        if (a[0],a[3],a[4]) != (b[0],b[3],b[4]) or (a[1]!=b[1]) == (a[2]!=b[2]):
            continue
        one,two = left["none"],right["none"]
        require((one["validation"]["sha256"],one["observation"]) ==
                (two["validation"]["sha256"],two["observation"]), "Cross-cohort axes differ")
        factors = ["support_count_and_points"] if a[1]!=b[1] else ["initialization_method"]
        if one["physical_uuid"] != two["physical_uuid"]:
            factors.append("physical_device")
        result.append(dict(
            from_group="/".join(map(str,a)), to_group="/".join(map(str,b)),
            changed_factors=factors, validation=one["validation"]["sha256"],
            by_arm={arm:dict(selected_steps=[left[arm]["selected_step"],right[arm]["selected_step"]],
                head_delta={head:{m:right[arm]["heads"][head][m]-left[arm]["heads"][head][m]
                                  for m in ("oa","aa","kappa","macro_f1","nll")} for head in HEADS})
                for arm in sorted(migration.ARMS)},
            causal_device_or_method_claim=False))
    return result


def retained_rows(root, plan, parent, parent_pin, retained, review):
    old = review["cases"]
    require(review["schema"] == "wp05_p4_real_selected_recompute_v1" and review["status"] == "partial"
            and review["acceptance"] is None and review["observed_cases"] == len(old) == 10
            and review["planned_cases"] == 20 and review["committed_optimizer_updates"] == 1200
            and review["resource"] == plan["parent_resource"] and review["scientific_scope"] == parent_pin
            and review["release"] == plan["parent_release"] and review["input_pins"] == parent["protected_files"]
            and review["audit_optimizer_updates"] == 0 and review["raw_gt_loaded"] is review["query_accessed"] is False
            and retained["case_audits"] == {r["case"]["id"]:r["audit"] for r in old},
            "Ten previously independently reviewed cases required")
    cases = {row["id"]:row for row in parent["cases"]}
    result = []
    for row in old:
        case = row["case"]
        expected_path = Path(plan["parent_execution"]) / "cases" / case["id"] / "audit.json"
        require(case == cases[case["id"]] and row["audit"]["path"] == str(expected_path),
                "Retained audit belongs to another original case")
        audited=migration.published_case(row["audit"],case,parent_pin,plan["parent_resource"],
                                        case["physical_uuid"],retained["protected_files"],root)
        result.append(migration.case_row(audited,parent_pin,row["audit"],"original",row["paths"]))
    return result


def run(root, plan_pin, authority, acceptance=None):
    plan = migration.read(plan_pin,root)
    require(plan["synthetic"] is False and len(plan["cases"]) == len(set(plan["cases"])) == 10
            and migration.read(authority,root) == release_for(plan_pin,plan), "Exact approved real P4 migration required")
    source = migration.read(plan["parent_resource"],root)
    parent_pin,new_pin = plan["parent_science"],plan["migrated_science"]
    parent,new = (migration.read(p,root) for p in (parent_pin,new_pin))
    migration.unchanged_science(parent,parent_pin,new,plan,root)
    proposal = migration.read(plan["proposal"],root)
    retained = migration.read(proposal["retained_evidence"],root)
    review_pin = retained["review"]
    review = migration.read(review_pin,root)
    require(parent["synthetic"] is new["synthetic"] is False and source["science"] == parent_pin
            and set(retained["cases"]).isdisjoint(plan["cases"])
            and set(retained["cases"])|set(plan["cases"]) == {r["id"] for r in parent["cases"]},
            "The two physical lanes do not partition the frozen P4 roster")
    if acceptance is not None:
        joined = migration.intake(acceptance,plan["parent_resource"],source,parent,review_pin,plan["parent_release"],root)
        require(joined["plan"] == plan_pin, "Acceptance covers another migration")
        rows = [*joined["retained"],*joined["migrated"]]
        stage = dict(optimizer_updates=1200,completed_cases=10)
    else:
        origin = migration.directory(plan["parent_execution"],root)
        with observe_lock(origin/".coordinator.lock") as busy:
            require(busy is False, "Original P4 queue is active")
            migration.check_files(retained["protected_files"],root.parent)
            rows = retained_rows(root,plan,parent,parent_pin,retained,review)
        stage = dict(optimizer_updates=0,completed_cases=0)
        target = Path(plan["execution"])
        require(target.resolve() == target and target.parent == root, "Unexpected migrated destination")
        if target.exists():
            with observe_lock(target/".coordinator.lock") as busy:
                require(busy is False, "Finish the active migrated worker before review")
                for lane, slot in plan["slots"].items():
                    for name in slot["cases"]:
                        updates, complete,_ = case_stages(target,name,plan_pin["sha256"],new_pin["sha256"],lane)
                        stage["optimizer_updates"] += updates
                        stage["completed_cases"] += complete

    scopes = {parent_pin["sha256"]:parent,new_pin["sha256"]:new}
    roster, probability, result, by_cohort = {},{}, {}, defaultdict(dict)
    for row in rows:
        case = row["case"]
        scope = scopes[row["scope"]["sha256"]]
        registration = scope["registration"]
        reg = migration.read(dict(path=registration,sha256=scope["protected_files"][registration]),root)
        key = (row["scope"]["sha256"],case["index"])
        budget = scope["exposure"][f's{case["support_seed"]}-k{case["k"]}']
        if key not in roster:
            layout, truth = case_labels(reg,case,budget,with_layout=True)
            roster[key] = truth, nearest_other_labels(layout,truth)
        truth,neighbours = roster[key]
        audit = migration.read(row["audit"],root)
        require(audit["populations"] == budget["populations"] and len(truth) == 110
                and np.bincount(truth,minlength=23).tolist() == [0]+[5]*22,
                "Accepted P4 validation population differs from the fixed 110-point role")
        scores,nll,mechanism = accepted_scores(root,row,truth,audit,reg)
        probability[case["id"]] = (truth,scores)
        summary = dict(case=case["id"],source=row["source"],scope=row["scope"]["sha256"],audit=row["audit"],
            selected_step=row["selected_step"],validation_nll=nll,physical_uuid=case["physical_uuid"],
            pair_sha256=case["pair_sha256"],observation=case["observation_sha256"],
            support=audit["populations"]["train"],validation=audit["populations"]["val"],
            heads=row["heads"],exposure=audit["exposure"],mechanism=mechanism,audit_seconds=row["audit_seconds"],
            paths=cost(audit,row["paths"]),focus={h:{str(label):focus_readout(truth,scores[h],label,neighbours)
                                                for label in FOCUS} for h in HEADS})
        result[case["id"]] = summary
        by_cohort[cohort_key(case)][case["arm"]] = summary
    paired = {"/".join(map(str,key)):roster_pair(key,lines,probability) for key,lines in by_cohort.items()}
    graph_pairs = {"/".join(map(str,key)):{arm:graph_comparison(lines["none"]["mechanism"]["graphs"],
                    lines[arm]["mechanism"]["graphs"]) for arm in PAIR_ARMS} for key,lines in by_cohort.items()}
    roster_cells = [dict(support_seed=key[0],k=key[1],method=key[2],model_seed=key[3],sampler_seed=key[4],
                         accepted_arms=sorted(by_cohort.get(key,{})),
                         pending_arms=sorted(migration.ARMS-by_cohort.get(key,{}).keys()))
                    for key in sorted({cohort_key(case) for case in parent["cases"]})]
    pending = sorted(set(plan["cases"])-result.keys())
    require(len(result) in (10,20) and len(result)+len(pending)==20,
            "A case was counted twice or a nonaccepted case entered the publication")
    return dict(schema=SCHEMA,status="accepted_two_executions" if acceptance else "partial",
                plan=plan_pin,approval=authority,combined_acceptance=acceptance,original_review=review_pin,
                cases=result,matched_cohorts=paired,pending=pending,migrated_progress=stage,
                cross_cohort_descriptive=descriptive_cross_cohorts(by_cohort),
                roster_cells=roster_cells, graph_comparisons_to_none=graph_pairs,
                recovered_paths_are_replicates=False,original_timing=review["timing"],
                protection=dict(registered_fit_limits=PROTECTION,minimum_independent_devval_per_class=FORMAL_FLOOR,
                                independent_confirmation=None,adoption_result=None),
                other_real_scopes=scene_context(root), optimizer_updates_by_this_reader=0,
                model_forwards_by_this_reader=0,raw_gt_loaded=False,query_accessed=False)

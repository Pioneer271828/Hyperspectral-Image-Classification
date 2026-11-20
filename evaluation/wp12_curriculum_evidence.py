"""Integrate authenticated CUR-S quality and supervision without deployment timing."""

import math
from pathlib import Path

from evaluation.wp12_low_rank_evidence import verify_guard
from evaluation.wp12_production_evidence import FULL_HEADS, metrics, missing, provenance
from model.research_config import digest
from research_runtime import hash_file
from transfer.curriculum_observation_admission import FORMAT as OBSERVATION_FORMAT
from transfer.curriculum_real import ARMS, FORMAT, REAL, REVIEW
from transfer.curriculum_support_review import SUPERVISION_SCOPE, VERSION
from transfer.registered_wp11_learning import read


def exposure_totals(exposure, history, support):
    labels = {str(row["index"]): str(row["label"]) for row in support}
    per_support, per_class = exposure["by_support"], exposure["by_class"]
    if set(per_support) != set(labels) or set(per_class) != set(labels.values()):
        raise ValueError("Curriculum exposure support or classes differ")
    raw, effective = {}, {}
    for row in history:
        for i, weight in zip(row["support_ids"], row["applied_weights"], strict=True):
            key = str(i)
            if key not in labels or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Invalid applied curriculum support weight")
            raw[key] = raw.get(key, 0)+1
            effective.setdefault(key, []).append(weight)
    for i, row in per_support.items():
        if row != dict(raw=raw.get(i, 0), effective=math.fsum(effective.get(i, []))) or row["raw"] < 1:
            raise ValueError("Recorded curriculum support mass differs")
    for cls, row in per_class.items():
        members = [per_support[i] for i, c in labels.items() if c == cls]
        if (row["raw"] != sum(v["raw"] for v in members)
                or row["effective"] != math.fsum(v["effective"] for v in members)):
            raise ValueError("Recorded curriculum class mass differs")
    if sum(raw.values()) != exposure["total_label_visits"]:
        raise ValueError("Recorded curriculum total native support exposure differs")
    return dict(raw_support_visits=sum(raw.values()), effective_support_mass=math.fsum(v["effective"] for v in per_support.values()),
                minimum_applied_weight=min(weight for values in effective.values() for weight in values))


def endpoint_supervision(diagnostic, item, support, budget):
    history = item["history"]
    endpoints = diagnostic.get("endpoints")
    full = diagnostic.get("full_run")
    if (len(history) != budget or not isinstance(endpoints, dict) or set(endpoints) != {"initial","selected","last"}
            or not isinstance(full, dict) or set(full) != {"step","history_sha256","exposure"}):
        raise ValueError("Versioned endpoint and full-run supervision required")
    labels = {str(s["index"]):s["label"] for s in support}

    def verify(payload, step):
        if type(step) is not int or not 0 <= step <= budget or type(payload.get("step")) is not int or payload["step"] != step:
            raise ValueError("Endpoint supervision step differs from its frozen checkpoint")
        prefix = history[:step]
        weights = {i: [] for i in labels}
        for row in prefix:
            for i,w in zip(row["support_ids"],row["applied_weights"],strict=True):
                weights[str(i)].append(w)
        expected = {i:dict(raw=len(values),effective=math.fsum(values)) for i,values in weights.items()}
        if payload.get("history_sha256") != digest(prefix) or payload.get("exposure") != expected:
            raise ValueError("Endpoint supervision prefix hash or applied exposure differs")
        return dict(step=step, history_sha256=payload["history_sha256"], exposure=expected)

    full_result = verify(full,budget)
    if full_result["exposure"] != item["exposure"]["by_support"]:
        raise ValueError("Full-run exposure differs from the sample audit")
    results = {}
    for point in ("initial","selected","last"):
        payload = endpoints[point]
        if not isinstance(payload, dict) or set(payload) != {"step","history_sha256","exposure","train_support"}:
            raise ValueError("Incomplete endpoint supervision record")
        step = item[point]["step"]
        if point == "initial" and step != 0 or point == "last" and step != budget or point == "selected" and step == 0:
            raise ValueError("Invalid frozen endpoint cursor")
        verified = verify(payload,step)
        seen = set()
        for cls, group in payload["train_support"].items():
            for row in group["samples"]:
                key = str(row["index"])
                if (key in seen or key not in labels or row["class_id"] != labels[key] or str(labels[key]) != cls
                        or {k:row[k] for k in ("raw","effective")} != verified["exposure"][key]):
                    raise ValueError("Endpoint strata use a different exposure prefix")
                seen.add(key)
        if seen != labels.keys():
            raise ValueError("Endpoint strata omit released support")
        results[point] = verified
    return results


def _evidence(spec):
    if set(spec) != {"registration", "audit", "mechanism", "observation"}:
        raise ValueError("Explicit CUR-S registration, audit and both WP06 artifacts required")
    reg, audit, mechanism, observation = (read(spec[k]) for k in ("registration", "audit", "mechanism", "observation"))
    root = Path(reg["root"])
    if (reg["format"] != FORMAT or reg["scope"] != REAL or reg["protocol"] != "wp01_xuzhou_k5_v5"
            or spec["registration"]["path"] != str(root/"registration.json")
            or spec["audit"]["path"] != str(root/"audit.json") or reg["teacher"]["path"] != str(root/"teacher.json")
            or audit["status"] != "passed_independent_curriculum" or audit["registration"] != spec["registration"]
            or audit["teacher"] != reg["teacher"] or audit["off_balanced_exact"] is not True
            or audit["actual_updates"] != reg["maximum_updates"]
            or any(audit[k] != 0 for k in ("original_gt_reads", "query", "optimizer_updates"))
            or mechanism["status"] != "passed_frozen_support_review" or mechanism["sample_registration"] != spec["registration"]
            or mechanism["sample_audit"] != spec["audit"] or mechanism["teacher"] != reg["teacher"]
            or mechanism.get("version") != VERSION or mechanism.get("supervision_scope") != SUPERVISION_SCOPE
            or mechanism["protocol"] != reg["protocol"] or mechanism["rule"] != reg["review"] or reg["review"] != REVIEW
            or mechanism["source_v5_hash"] != digest(audit["source"]["scores"])
            or any(mechanism[k] != 0 for k in ("gt", "query", "optimizer_updates"))
            or observation["format"] != OBSERVATION_FORMAT or observation["sample_registration"] != spec["registration"]
            or observation["source_commit"] != reg["commit"]
            or observation["protocol"] != reg["protocol"] or observation["state"] != reg["teacher_state"]
            or observation["real_training"] != "NOT_RUN" or observation["marginal_match"] is not True
            or any(observation[k] != 0 for k in ("original_gt_reads", "query", "optimizer_updates"))):
        raise ValueError("Original real curriculum, frozen mechanism and zero-update observation evidence required")
    complete = read(audit["completion"])
    notice = read(complete["notice"])
    if (audit["completion"]["path"] != str(root/"batch-complete.json") or complete["registration"] != spec["registration"]
            or complete["updates"] != audit["actual_updates"] or notice["registration"] != spec["registration"]
            or notice["maximum_updates"] != complete["updates"] or notice["budget"] != reg["budget"]
            or notice["cases"] != [c["case"] for c in reg["cases"]]):
        raise ValueError("Curriculum completion does not match the frozen budget or notice")
    return reg, audit, mechanism, observation


def adapt(spec):
    reg, audit, mechanism, observation = _evidence(spec)
    origin = read(reg["source"]["origin"])
    historical, old = read(origin["registration"]), read(origin["off_full"])
    source = audit["source"]
    if (origin["source"] != reg["source"]["weights"] or origin["sidecar"] != reg["source"]["sidecar"]
            or old["new_updates"] != historical["budget"] or old["finish"]["selected_state_hash"] != origin["selected_state"]
            or source["state"] != reg["teacher_state"] or source["state"] != origin["selected_state"]
            or any(source[k] != reg["source_replay"][k] for k in ("metrics", "scores", "forwards"))):
        raise ValueError("Historical producer, original model and curriculum source differ")
    proof = dict(source_commit=origin["producer_commit"], witness_commit=historical["commit"],
                 source_evidence=origin["registration"], source_steps=historical["budget"])
    cases = {c["case"]: c for c in reg["cases"]}
    if (len(cases) != len(reg["cases"]) or set(cases) != set(audit["cases"]) or set(cases) != set(mechanism["cases"])
            or {(c["seed"], c["arm"]) for c in cases.values()} != {(s, a) for s in reg["seeds"] for a in ARMS}
            or reg["maximum_updates"] != 2 * len(cases) * reg["budget"]):
        raise ValueError("Incomplete real curriculum case grid or update budget")
    teacher = read(reg["teacher"])
    support = reg["support"]
    released = dict(indices=[v["index"] for v in support], labels=[v["label"] for v in support],
                    seed=next(iter(reg["exposures"].values()))["support"]["seed"])
    classes = reg["class_ids"]
    val = reg["validation"]
    if (teacher["teacher_state"] != source["state"] or teacher["support"] != released
            or teacher["score_hash"] != digest({k:v for k,v in teacher.items() if k != "score_hash"})
            or teacher["classes"] != classes or len(teacher["losses"]) != len(support)
            or teacher["forward_tiles"] != len({s["tile"] for s in support})
            or audit["teacher_independent_forwards"] != teacher["forward_tiles"]
            or (len(support), len(val["indices"]), released["seed"], val["seed"]) != (45, 45, 512, 3031)):
        raise ValueError("Frozen teacher scoring or K5/V5 release identity differs")
    teacher_cost = dict(scope="one_shared_frozen_teacher_devtrain_support", producer=origin["producer_commit"],
                        state=teacher["teacher_state"], evidence=reg["teacher"], semantic_hash=teacher["score_hash"],
                        scoring_commit=reg["commit"], scoring_kind=teacher["score_kind"],
                        forwards=teacher["forward_tiles"], label_accesses=len(support), seconds=teacher["seconds"],
                        independent_forwards=audit["teacher_independent_forwards"])
    common = dict(scene="xuzhou", data_protocol=reg["protocol"], model_seed=reg["model_seed"], train_labels=len(support),
                  validation_labels=len(val["indices"]), class_ids=classes, split=reg["preprocessing"],
                  support_sha256=digest(released), evaluation_sha256=digest(val), validation_role="repeated_development_v5")
    no_cost = missing("NOT_RUN", "No course deployment benchmark was registered")
    rows = [dict(module="WP01-CUR-S", arm="original_fp32", case_id="original_fp32", identity=dict(common, sampler_seed=None),
                 **provenance(**proof), state=source["state"], package=origin["source"]["sha256"], precision="fp32",
                 package_bytes=None, budget=0, actual_optimizer_calls=0, selected_step=origin["selected_step"],
                 selection="original_wp08_off", quality=metrics(source["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                 protection=None, cost=no_cost)]
    for name, case in cases.items():
        item, diagnostic = audit["cases"][name], mechanism["cases"][name]
        chosen = item["selected"]
        package = chosen["package"]
        if (item["case"] != case or diagnostic["case"] != case or item["policy_hash"] != reg["policies"][case["arm"]]
                or item["steps"] != reg["budget"] or item["updates"] != 2 * reg["budget"]
                or not 1 <= chosen["step"] <= reg["budget"] or item["last"]["step"] != reg["budget"]
                or chosen["state"] != package["state"] or package["format"] != "fp32"
                or any(hash_file(package[k]["path"]) != package[k]["sha256"] for k in ("weights", "sidecar"))):
            raise ValueError("Curriculum endpoint, recovery trajectory or WP06 join differs")
        reference = audit["cases"][f"s{case['seed']}-off"]
        guards = {point: verify_guard(reference[point], item[point], classes, audit["small_classes"], reg["protection"]["limits"])
                  for point in ("initial", "selected", "last")}
        if guards != audit["guards"][name]:
            raise ValueError("Recorded curriculum per-head protection differs")
        endpoint_mass = endpoint_supervision(diagnostic,item,support,reg["budget"])
        e = exposure_totals(item["exposure"], item["history"], support)
        if e["minimum_applied_weight"] != item["exposure"]["minimum_applied_weight"]:
            raise ValueError("Recorded minimum applied curriculum weight differs")
        training = dict(scope="sum_registered_stage_update_loop", updates=item["updates"],
                        update_loop_seconds=sum(v["update_loop_seconds"] for v in item["stages"].values()),
                        preparation_seconds=sum(v["restore_or_prepare_seconds"] for v in item["stages"].values()),
                        checkpoint_seconds=sum(v["checkpoint_finish_seconds"] for v in item["stages"].values()),
                        stage_v5_checks=item["validation_checks"], label_accesses=2 * e["raw_support_visits"],
                        stage_v5_label_accesses=item["validation_checks"] * len(val["indices"]))
        lineage = provenance(**proof, recovery_commit=reg["commit"], recovery_steps=reg["budget"],
                             release_commit=reg["commit"], release_evidence=spec["audit"])
        rows.append(dict(module="WP01-CUR-S", arm=case["arm"], case_id=name, identity=dict(common, sampler_seed=case["seed"]),
                         **lineage, state=package["state"], package=package["weights"]["sha256"], precision=package["format"],
                         package_bytes=package["bytes"], budget=reg["budget"], actual_optimizer_calls=item["updates"],
                         selected_step=chosen["step"], selection=reg["selection"],
                         quality=metrics(chosen["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                         other_checkpoints={p: dict(step=item[p]["step"], quality=metrics(
                             item[p]["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix")) for p in ("initial", "last")},
                         protection=guards["selected"], initial_protection=guards["initial"], last_protection=guards["last"],
                         teacher=teacher_cost, curriculum=dict(policy_hash=item["policy_hash"], pacing=reg["pacing"]),
                         exposure=dict(**e, by_support=item["exposure"]["by_support"], by_class=item["exposure"]["by_class"],
                                       maximum_class_mass_roundoff=item["exposure"]["maximum_class_mass_roundoff"]),
                         training_cost=training, support_review=spec["mechanism"], cost=no_cost,
                         exposure_scope="full_run", endpoint_supervision=endpoint_mass))
    return rows, dict(producer=reg["commit"], registration=spec["registration"], independent_audit=spec["audit"],
                      cases=len(cases), recovery_seeds=reg["seeds"], source_model_seed=reg["model_seed"], budget=reg["budget"],
                      actual_optimizer_calls=reg["maximum_updates"], off_balanced_exact=True, default="off",
                      teacher=teacher_cost, support_review=spec["mechanism"], review_scope=mechanism["history_scope"],
                      review_version=mechanism["version"],
                      observation_registration=spec["observation"], observation_probes=observation["preflight_forwards"],
                      observation_joint_match=observation["joint_match"], observation_training=missing("NOT_RUN", "Zero-update admission only"),
                      deployment=missing("NOT_RUN", "No registered deployment timing"),
                      independent_spatial=missing("NOT_RUN", "Quality measured on the fixed development V5"))

"""Frozen teacher probes on train support and endpoint-specific joint supervision."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from compression.precision_recovery_audit import HEADS
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_factorial_real import CASES, inputs
from transfer.curriculum_observation_admission import CONDITIONS, audited_forward
from transfer.curriculum_real_audit import fp32
from transfer.curriculum_support_review import _ranks, _spearman
from transfer.g3_training import state_hash


def _condition_key(i):
    return digest({"mode": "none"} if i == 0 else {"mode": "random_subset", "keep_fraction": (100-i*25)/100})


def freeze(request, output, code):
    require_frozen_code(code)
    _, s, o, _, scene, _, _, trials = inputs(request, code)
    policy = next(iter(trials.values()))[2].to_dict()
    entries = {}
    for i, condition in enumerate(CONDITIONS):
        step, original = next((n, r) for n, r in enumerate(policy["training"], 1) if digest(r["config"]) == _condition_key(i))
        entries[condition] = dict(step=step, record_hash=original["record_hash"], selected_band_ids=original["selected_band_ids"])
    return dev._json_new(external_directory(output), dict(format="wp06_teacher_train_conditions_v1", request=request,
        teacher=s["teacher"], state=s["teacher_state"], policy=policy["policy_hash"], entries=entries,
        preprocessing=s["preprocessing"], train_support=s["support"], role="devtrain_support_only",
        occupied_tiles=sorted({v["tile"] for v in s["support"]}), original_band_count=len(scene.bands.band_ids),
        observation=o["prior"], rule="first_original_training_record_per_configuration", optimizer_updates=0, gt=0, query=0))


def diagnostic_train_view(manifest, policy, scene, record, tile, condition):
    """Only the first frozen training record may probe an occupied train core."""
    idx = tile.get("id")
    if (manifest["role"] != "devtrain_support_only" or condition not in CONDITIONS or set(manifest["entries"]) != set(CONDITIONS)
            or idx not in manifest["occupied_tiles"] or record["roles"][idx]["role"] != "devtrain"
            or policy.policy_hash != manifest["policy"]
            or manifest["train_support"] != _ownership(scene, record)):
        raise ValueError("Diagnostic role, condition or released training support differs")
    first = manifest["entries"][condition]
    records = policy.to_dict()["training"]
    key = _condition_key(CONDITIONS.index(condition))
    step, frozen = next((n, r) for n, r in enumerate(records, 1) if digest(r["config"]) == key)
    if first != dict(step=step, record_hash=frozen["record_hash"], selected_band_ids=frozen["selected_band_ids"]):
        raise ValueError("Diagnostic must use the first admitted record and original band IDs")
    return policy.view(scene, record, tile, phase="training", step=step), frozen


def _ownership(scene, record):
    w = scene.plan["shape"][1]
    supports = record["train_support"]
    rows = []
    for i, label in zip(supports["indices"], supports["labels"], strict=True):
        y, x = divmod(i,w)
        ids = [t["id"] for t in record["roles"] if t["role"] == "devtrain"
               and t["box"][0] <= y < t["box"][1] and t["box"][2] <= x < t["box"][3]]
        if len(ids) != 1:
            raise ValueError("Support ownership must be a single devtrain block")
        rows.append(dict(index=i,label=label,tile=ids[0]))
    return rows


def difficulty(rows, reference):
    if set(rows) != set(reference) or any(rows[i]["label"] != reference[i]["label"] for i in rows):
        raise ValueError("Paired teacher support IDs or labels differ")
    by_class = {}
    for cls in sorted({row["label"] for row in rows.values()}):
        ids = sorted(i for i, row in rows.items() if row["label"] == cls)
        base = [reference[i]["nll"] for i in ids]
        actual = [rows[i]["nll"] for i in ids]
        a, b = _ranks(base), _ranks(actual)
        base_order = sorted(range(len(ids)), key=lambda j: (base[j], ids[j]))
        actual_order = sorted(range(len(ids)), key=lambda j: (actual[j], ids[j]))
        bin_a = {j: 3*n//len(ids) for n,j in enumerate(base_order)}
        bin_b = {j: 3*n//len(ids) for n,j in enumerate(actual_order)}
        by_class[str(cls)] = dict(spearman=_spearman(base, actual), n=len(ids),
            migrations={str(i):dict(full_rank=a[j], rank=b[j], delta=b[j]-a[j],
                                     full_bin=bin_a[j], bin=bin_b[j]) for j,i in enumerate(ids)},
            changed_bin=sum(bin_a[j] != bin_b[j] for j in range(len(ids))))
    return by_class


@torch.no_grad()
def teacher_diagnosis(pin, output, code):
    started = perf_counter()
    manifest = admission.read(pin)
    _, s, _, model, scene, record, _, trials = inputs(manifest["request"], code)
    out = external_directory(output)
    if out.exists() or Path(pin["path"]).parent == out:
        raise ValueError("Diagnostic output must be separate and fresh")
    teacher = admission.read(manifest["teacher"])
    if (manifest["format"] != "wp06_teacher_train_conditions_v1" or manifest["state"] != s["teacher_state"]
            or manifest["teacher"] != s["teacher"] or manifest["preprocessing"] != s["preprocessing"]
            or manifest["train_support"] != _ownership(scene,record)
            or any(manifest[k] for k in ("optimizer_updates","gt","query"))):
        raise ValueError("Wrong frozen train-only diagnostic receipt")
    policy = next(iter(trials.values()))[2]
    groups = defaultdict(list)
    for member in manifest["train_support"]:
        groups[member["tile"]].append(member)
    if set(groups) != set(manifest["occupied_tiles"]):
        raise ValueError("Train diagnostic may only visit occupied released blocks")
    width = scene.plan["shape"][1]
    result, checks = {}, {}
    for condition in CONDITIONS:
        rows, witnesses = {}, {}
        for tid, members in groups.items():
            tile = scene.plan["tiles"][tid]
            y0, _, x0, x1 = tile["box"]
            view, original = diagnostic_train_view(manifest, policy, scene, record, tile, condition)
            outputs, observed = audited_forward(model, scene, record, tile, view, original, phase="training")
            witnesses[str(tid)] = observed
            for member in members:
                i, label = member["index"], member["label"]
                y, x = divmod(i,width)
                local = (y-y0)*(x1-x0)+x-x0
                heads = {}
                for head, values in zip(HEADS, outputs[:3], strict=True):
                    p = values[local].detach().cpu()
                    guess = int(p.argmax())+1
                    heads[head] = dict(nll=float(-p[label-1].clamp(min=1e-8,max=1.).log()),
                                       prediction=guess, correct=guess == label, probabilities=p.tolist())
                rows[i] = dict(**member, heads=heads)
        result[condition] = rows
        checks[condition] = witnesses
    baseline = result["full"]
    maximum = 0.
    for i, expected in zip(teacher["support"]["indices"],teacher["losses"],strict=True):
        actual = baseline[i]["heads"]["fused"]["nll"]
        if not math.isclose(actual,expected,rel_tol=2e-7,abs_tol=1e-7):
            raise ValueError("Diagnostic full fused score differs from the frozen original teacher")
        maximum = max(maximum,abs(actual-expected))
    ranks = {}
    for condition, rows in result.items():
        ranks[condition] = {h:difficulty({i:dict(label=v["label"],nll=v["heads"][h]["nll"]) for i,v in rows.items()},
                                        {i:dict(label=v["label"],nll=v["heads"][h]["nll"]) for i,v in baseline.items()}) for h in HEADS}
    if state_hash(model) != teacher["teacher_state"]:
        raise ValueError("Frozen teacher changed in diagnostic")
    admission.check_pins([pin, manifest["teacher"], s["source"]["weights"], s["source"]["sidecar"]])
    require_frozen_code(code)
    out.mkdir()
    with (out/"teacher-support.csv").open("x", encoding="utf-8", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("condition","head","class","support","tile","nll","prediction","correct","full_rank","rank","full_bin","bin"))
        for condition, rows in result.items():
            for i,row in sorted(rows.items()):
                for head,v in row["heads"].items():
                    r=ranks[condition][head][str(row["label"])]["migrations"][str(i)]
                    writer.writerow((condition,head,row["label"],i,row["tile"],v["nll"],v["prediction"],v["correct"],
                                     r["full_rank"],r["rank"],r["full_bin"],r["bin"]))
    return dev._json_new(out/"teacher-diagnosis.json",dict(status="passed_frozen_train_teacher", manifest=pin, commit=code["commit"],
        teacher=manifest["teacher"], teacher_source=s["source"]["origin"], state=manifest["state"], rows=result,
        ranks=ranks, audited_forwards=checks, forwards=len(groups)*len(CONDITIONS), labels=len(baseline)*len(CONDITIONS),
        maximum_original_full_nll_delta=maximum, seconds=perf_counter()-started, optimizer_updates=0, gt=0, query=0,
        interpretation="Previously seen training support; five per class; descriptive sensitivity only"))


def joint_prefix(history, plan, support, step, observations, tiles, full_bands, strata, condition_strata=None):
    if type(step) is not int or not 0 <= step <= plan.updates or len(history) != plan.updates:
        raise ValueError("Complete history and valid endpoint cutoff required")
    if observations is not None and len(observations) != plan.updates:
        raise ValueError("Observation schedule budget differs")
    labels = {v["index"]:v["label"] for v in support}
    mass = {c:defaultdict(list) for c in CONDITIONS}
    band_count, pixel_bands = Counter(), Counter()
    for n, row in enumerate(history[:step],1):
        event = plan.event(n)
        if any(row[k] != v for k,v in event.items()):
            raise ValueError("Wrong ordered support, tile or anchor in joint prefix")
        if observations is None:
            condition, bands = "full", full_bands
            if "observation" in row:
                raise ValueError("Unexpected restricted training observation")
        else:
            obs = observations[n-1]
            condition = CONDITIONS[tuple(_condition_key(i) for i in range(4)).index(digest(obs["config"]))]
            bands = len(obs["selected_band_ids"])
            if row["observation"]["record_hash"] != obs["record_hash"] or row["observation"]["selected_band_count"] != bands:
                raise ValueError("Actual selected observation differs from the frozen event")
        band_count[condition] += bands
        y0,y1,x0,x1 = tiles[event["tile_id"]]["box"]
        pixel_bands[condition] += (y1-y0)*(x1-x0)*bands
        per_class = defaultdict(list)
        ids, weights = event["support_ids"], row["weights"]
        actual = row.get("applied_weights", weights)
        for i, w, applied in zip(ids,weights,actual,strict=True):
            if not math.isfinite(w) or w <= 0 or applied != fp32(w) or ("applied_weights" not in row and applied != 1.):
                raise ValueError("Actual FP32 loss mask differs from the declared weights")
            mass[condition][i].append(applied)
            per_class[labels[i]].append(applied)
        if any(abs(math.fsum(v)-len(v)) > len(v)*2e-7 for v in per_class.values()):
            raise ValueError("Actual class loss mass is not preserved")
    def by_bin(labels, rank, field, aggregate):
        return {str(k):{str(b):aggregate([i for i,v in labels.items()
                                         if v == k and rank[str(k)]["migrations"][str(i)][field] == b])
                        for b in range(3)} for k in sorted(set(labels.values()))}

    conditions = {}
    for c in CONDITIONS:
        rows = {str(i):dict(raw=len(mass[c][i]),effective=math.fsum(mass[c][i])) for i in labels}
        def total(ids):
            return dict(raw=sum(rows[str(i)]["raw"] for i in ids),effective=math.fsum(rows[str(i)]["effective"] for i in ids))
        by_class = {str(k):total([i for i,v in labels.items() if v==k]) for k in sorted(set(labels.values()))}
        conditions[c] = dict(support=rows, classes=by_class, bins=by_bin(labels, strata, "full_bin", total),
                             bands=band_count[c], pixel_bands=pixel_bands[c], unseen=sum(v["raw"]==0 for v in rows.values()))
        if condition_strata is not None:
            conditions[c]["condition_teacher_bins"] = by_bin(labels, condition_strata[c]["fused"], "bin", total)
    return dict(step=step, history_sha256=digest(history[:step]), conditions=conditions)


def _class_errors(source, case, classes):
    answer = {}
    for point, entry in case["endpoints"].items():
        answer[point] = {}
        for condition, end in entry["conditions"].items():
            start = source[condition]
            if start["targets"] != end["targets"]:
                raise ValueError("Class error transition requires the same released V5 targets")
            answer[point][condition] = {}
            for head in HEADS:
                counts = {str(c):dict(repaired=0,new_error=0,still_correct=0,still_wrong=0) for c in classes}
                for y,a,b in zip(end["targets"],start["predictions"][head],end["predictions"][head],strict=True):
                    tag = ("still_correct" if b == y else "new_error") if a == y else ("repaired" if b == y else "still_wrong")
                    counts[str(y)][tag] += 1
                answer[point][condition][head] = counts
    return answer


def _write_joint_csv(out, cases, ranks, support):
    labels = {str(v["index"]):v["label"] for v in support}
    with (out/"joint-supervision.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case","producer","endpoint","step","history_sha256","condition","class","support",
                         "raw","effective","full_teacher_bin","condition_teacher_bin","teacher_rank_delta","unseen"))
        for name, row in cases.items():
            for point, endpoint in row["supervision"].items():
                for c, config in endpoint["conditions"].items():
                    for index, mass in config["support"].items():
                        cls=labels[index]
                        rank=ranks[c]["fused"][str(cls)]["migrations"][index]
                        writer.writerow((name,row["producer"],point,endpoint["step"],endpoint["history_sha256"],c,cls,index,
                            mass["raw"],mass["effective"],rank["full_bin"],rank["bin"],rank["delta"],mass["raw"]==0))
    with (out/"teacher-class-and-errors.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case","endpoint","condition","head","class","teacher_spearman","teacher_bin_changes","class_raw",
                         "class_effective","class_unseen","new_errors","repaired","still_correct","still_wrong"))
        for name,row in cases.items():
            for point, conditions in row["errors_from_source"].items():
                for c, heads in conditions.items():
                    conf=row["supervision"][point]["conditions"][c]
                    for head,classes in heads.items():
                        for cls, errors in classes.items():
                            teacher=ranks[c][head][cls]
                            mass=conf["classes"][cls]
                            unseen=sum(v["raw"]==0 for i,v in conf["support"].items() if labels[i]==int(cls))
                            writer.writerow((name,point,c,head,cls,teacher["spearman"],teacher["changed_bin"],mass["raw"],
                                             mass["effective"],unseen,*[errors[k] for k in ("new_error","repaired","still_correct","still_wrong")]))


def review_joint(request, teacher_pin, audit_pin, output, code):
    started = perf_counter()
    _, s, _, _, scene, record, plans, trials = inputs(request, code)
    teacher, audit = [admission.read(pin) for pin in (teacher_pin,audit_pin)]
    reg = admission.read(audit["registration"])
    if (teacher["status"] != "passed_frozen_train_teacher" or teacher["teacher"] != s["teacher"]
            or audit["status"] != "passed_real_mixed_factorial" or reg["request"] != request
            or admission.read(teacher["manifest"])["request"] != request
            or teacher["state"] != reg["state"] or audit["adapter"] != code["commit"]
            or any(v[k] for v in (teacher,audit) for k in ("optimizer_updates","gt","query"))):
        raise ValueError("Exactly the original teacher and independently audited real factorial are required")
    ranks = teacher["ranks"]
    strata = ranks["full"]["fused"]
    policy = next(iter(trials.values()))[2].to_dict()
    cases = {}
    for c in CASES:
        name, seed, arm = c["case"], c["seed"], c["arm"]
        item = audit["cases"][name]
        if item["case"] != c or digest(item["history"]) != item["history_sha256"]:
            raise ValueError("Joint diagnostic trajectory identity differs")
        history = item["history"]
        obs = policy["training"] if arm in ("o_only","s_plus_o") else None
        cutoffs = {"initial":0,"selected":item["endpoints"]["selected"]["step"],
                   "last":item["endpoints"]["last"]["step"],"full_run":plans[str(seed)].updates}
        points = {point:joint_prefix(history,plans[str(seed)],reg["support"],step,obs,scene.plan["tiles"],
                                     len(scene.bands.band_ids),strata,ranks) for point,step in cutoffs.items()}
        cases[name] = dict(case=c,producer=item["producer"],supervision=points,
                           errors_from_source=_class_errors(audit["source"],item,record["classes"]))
    equality = {}
    for seed in plans:
        o,so = (cases[f"s{seed}-{arm}"]["supervision"]["full_run"] for arm in ("o_only","s_plus_o"))
        raw = lambda p:{c:{i:v["raw"] for i,v in x["support"].items()} for c,x in p["conditions"].items()}
        if raw(o) != raw(so):
            raise ValueError("S weighting changed the O-only actual raw joint exposure")
        equality[seed] = dict(raw_matched=True, effective_changed_cells=sum(
            so["conditions"][c]["support"][i]["effective"] != v["effective"]
            for c,x in o["conditions"].items() for i,v in x["support"].items()))
    paths = [request,teacher_pin,teacher["manifest"],audit_pin,audit["frozen"],reg["teacher"],*reg["protected"]]
    paths += [value["package"][k] for row in audit["cases"].values() for value in row["endpoints"].values()
              for k in ("weights","sidecar")]
    out = external_directory(output)
    if out.exists() or any(out in Path(pin["path"]).parents for pin in paths):
        raise ValueError("Fresh external diagnostic directory required")
    admission.check_pins(paths)
    require_frozen_code(code)
    out.mkdir()
    _write_joint_csv(out,cases,ranks,reg["support"])
    return dev._json_new(out/"joint-review.json",dict(status="passed_joint_supervision", commit=code["commit"],
        request=request, teacher=teacher_pin, audit=audit_pin, cases=cases, restricted_pair=equality,
        analyzed_classes=record["classes"], small_classes=audit["small_classes"], scope="actual_fp32_through_each_frozen_endpoint",
        optimizer_updates=0, gt=0, query=0, forwards=0, seconds=perf_counter()-started))

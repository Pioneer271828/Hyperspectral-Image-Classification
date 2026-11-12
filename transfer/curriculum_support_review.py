"""Read-only curriculum mechanism diagnostics on the two originally released roles."""

from collections import Counter, defaultdict
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from compression.precision_recovery_audit import HEADS, independent_predictions
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_observation_admission import bind_sample
from transfer.curriculum_real import REVIEW
from transfer.curriculum_real_audit import train_support_predictions
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _packet
from transfer.curriculum_real import RegisteredSampleRun


def _ranks(values):
    result = [0.] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    by_value = defaultdict(list)
    for rank, index in enumerate(ordered):
        by_value[values[index]].append((rank, index))
    for tied in by_value.values():
        average = mean(rank for rank, _ in tied)
        for _, index in tied:
            result[index] = average
    return result


def _spearman(left, right):
    x, y = _ranks(left), _ranks(right)
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    x, y = [v-mx for v in x], [v-my for v in y]
    denom = math.sqrt(math.fsum(v*v for v in x)*math.fsum(v*v for v in y))
    return math.fsum(a*b for a, b in zip(x, y, strict=True))/denom if denom else None


def _training_exposure(history, plan, support):
    counts = Counter()
    mass = defaultdict(list)
    for step, row in enumerate(history, 1):
        event = plan.event(step)
        if any(row[k] != value for k, value in event.items()):
            raise ValueError("Support review history uses a different class-first draw")
        ids = event["support_ids"]
        counts.update(ids)
        for index, value in zip(ids, row["applied_weights"], strict=True):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid applied weight in the frozen support history")
            mass[index].append(value)
    return {str(i): dict(raw=counts[i], effective=math.fsum(mass[i])) for i in support}


def _stratify(teacher, actual, exposure):
    result = {}
    for cls in sorted({v["label"] for v in teacher.values()}):
        ids = sorted((i for i in teacher if teacher[i]["label"] == cls), key=lambda i: (teacher[i]["nll"], i))
        rows = []
        for rank, index in enumerate(ids):
            baseline, final = teacher[index], actual[index]
            if (baseline["tile"], baseline["label"]) != (final["tile"], final["label"]):
                raise ValueError("Training support endpoint moved to a different class or tile")
            rows.append(dict(index=index, class_id=cls, tile=baseline["tile"], rank=rank, bin=3*rank//len(ids),
                             teacher_nll=baseline["nll"], **exposure[str(index)],
                             teacher_prediction=baseline["prediction"], final_prediction=final["prediction"],
                             final_nll=final["nll"], loss_delta=final["nll"]-baseline["nll"],
                             teacher_wrong=baseline["prediction"] != cls, final_wrong=final["prediction"] != cls))
        bins = {}
        for b in range(3):
            members = [r for r in rows if r["bin"] == b]
            bins[str(b)] = dict(n=len(members), ids=[r["index"] for r in members],
                mean_teacher_nll=mean(r["teacher_nll"] for r in members) if members else None,
                mean_effective=mean(r["effective"] for r in members) if members else None,
                mean_final_nll=mean(r["final_nll"] for r in members) if members else None,
                final_errors=sum(r["final_wrong"] for r in members))
        result[str(cls)] = dict(n=len(rows), samples=rows, bins=bins, association={
            "teacher_nll_vs_effective": _spearman([r["teacher_nll"] for r in rows], [r["effective"] for r in rows]),
            "effective_vs_final_nll": _spearman([r["effective"] for r in rows], [r["final_nll"] for r in rows]),
            "teacher_nll_vs_final_nll": _spearman([r["teacher_nll"] for r in rows], [r["final_nll"] for r in rows])})
    return result


def _validation_members(scene, record):
    released = dict(zip(record["validation_support"]["indices"], record["validation_support"]["labels"], strict=True))
    width = scene.plan["shape"][1]
    result = []
    for tile in record["roles"]:
        if tile["role"] == "devval":
            y0, y1, x0, x1 = tile["box"]
            result.extend((i, released[i], tile["id"]) for i in sorted(released)
                          if y0 <= i//width < y1 and x0 <= i % width < x1)
    if len(result) != len(released) or {i: label for i, label, _ in result} != released:
        raise ValueError("Devval support identities differ")
    return result


def _paired(members, old, new, guard, small):
    if old["targets"] != new["targets"] or new["targets"] != [label for _, label, _ in members]:
        raise ValueError("Paired V5 targets do not match released support")
    output = {}
    for head in HEADS:
        rows = []
        by_class = defaultdict(Counter)
        for (index, label, tile), prev, nxt in zip(members, old["predictions"][head], new["predictions"][head], strict=True):
            if prev == label:
                change = "still_correct" if nxt == label else "new_error"
            else:
                change = "repaired" if nxt == label else "still_wrong"
            by_class[str(label)][change] += 1
            rows.append(dict(index=index, tile=tile, label=label, off_prediction=prev, candidate_prediction=nxt, change=change))
        if {str(c): by_class[str(c)]["new_error"] for c in small} != guard[head]["new_small_class_errors"]:
            raise ValueError("Independent paired V5 small-class error counts differ")
        output[head] = dict(samples=rows, per_class={k: dict(v) for k, v in by_class.items()},
                            totals=dict(Counter(r["change"] for r in rows)))
    return output


def review(sample_pin, audit_pin, output, code):
    started = perf_counter()
    reg, source, scene, record, plans = bind_sample(sample_pin, code)
    a = admission.read(audit_pin)
    root = Path(reg["root"])
    completion = admission.read(a["completion"])
    names = {c["case"] for c in reg["cases"]}
    if (audit_pin["path"] != str(root/"audit.json") or a["registration"] != sample_pin or a["teacher"] != reg["teacher"]
            or a["status"] != "passed_independent_curriculum" or a["actual_updates"] != reg["maximum_updates"]
            or a["completion"]["path"] != str(root/"batch-complete.json") or completion["registration"] != sample_pin
            or completion["updates"] != a["actual_updates"] or set(a["cases"]) != names
            or any(a[k] for k in ("original_gt_reads", "query", "optimizer_updates")) or reg["review"] != REVIEW):
        raise ValueError("Original completed curriculum audit and preregistered diagnostic required")
    output = external_directory(output)
    if output.exists() or root == output or output in root.parents or root in output.parents:
        raise ValueError("Separate output required for the read-only support review")
    teacher, teacher_forwards = train_support_predictions(source, scene, record, reg["support"])
    scores = admission.read(reg["teacher"])
    for index, score in zip(scores["support"]["indices"], scores["losses"], strict=True):
        if not math.isclose(teacher[index]["nll"], score, rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError("Original teacher replay differs")
        teacher[index]["nll"] = score
    initial_v5 = independent_predictions(source, scene, record)
    if any(initial_v5[key] != a["source"][key] for key in ("scores", "metrics", "predictions", "targets")):
        raise ValueError("Original full V5 source output differs")
    members = _validation_members(scene, record)
    rows, pins = {}, [sample_pin, audit_pin, a["completion"], reg["teacher"], *reg["protected"].values()]
    train_calls, val_calls = 0, initial_v5["forwards"]
    for name, row in a["cases"].items():
        case = row["case"]
        if case not in reg["cases"] or case["case"] != name:
            raise ValueError("Unexpected curriculum case identity")
        full_pin = completion["attempts"][name]["full"]
        receipt = admission.read(full_pin)
        packet = _packet(receipt, reg["contracts"][name], reg["budget"], packet_format=RegisteredSampleRun.FORMAT)
        history_hash = digest(packet["history"])
        if history_hash != digest(row["history"]):
            raise ValueError("Historical record differs from the authenticated full trajectory")
        pins.extend((full_pin, receipt["checkpoint"]))
        exposed = _training_exposure(packet["history"], plans[str(case["seed"])], teacher)
        if exposed != row["exposure"]["by_support"]:
            raise ValueError("Independent cumulative applied supervision differs")
        result = dict(case=case, history_sha256=history_hash, steps=len(packet["history"]),
                      exposure=exposed, train_support={}, devval={})
        for point in ("selected", "last"):
            endpoint = row[point]
            package = endpoint["package"]
            if package["weights"]["path"] != str(root/"endpoints"/name/f"{point}.pt"):
                raise ValueError("Endpoint has the wrong arm or checkpoint role")
            model = dev._load_source(package["weights"], package["sidecar"])
            if state_hash(model) != endpoint["state"] or endpoint["state"] != row["history"][endpoint["step"]-1]["state_hash"]:
                raise ValueError("Frozen endpoint state differs from the audited trajectory")
            pins.extend(package[k] for k in ("weights", "sidecar"))
            final, calls = train_support_predictions(model, scene, record, reg["support"])
            train_calls += calls
            v5 = independent_predictions(model, scene, record)
            val_calls += v5["forwards"]
            if any(v5[k] != endpoint[k] for k in ("scores", "metrics", "predictions", "targets")):
                raise ValueError("Development endpoint predictions changed during review")
            result["train_support"][point] = _stratify(teacher, final, exposed)
            result["devval"][point] = _paired(members, a["cases"][f"s{case['seed']}-off"][point], v5,
                                             a["guards"][name][point], a["small_classes"])
        rows[name] = result
        print(f"{name}: isolated devtrain strata and devval paired errors", flush=True)
    admission.check_pins(pins)
    require_frozen_code(code)
    output.mkdir()
    return dev._json_new(output/"review.json", dict(status="passed_frozen_support_review", commit=code["commit"],
        source_hash=code["source_hash"], sample_registration=sample_pin, sample_audit=audit_pin, teacher=reg["teacher"],
        protocol=reg["protocol"], rule=REVIEW, cases=rows, train_forwards=train_calls, teacher_forwards=teacher_forwards,
        devval_forwards=val_calls, train_support_per_endpoint=len(teacher), devval_support_per_endpoint=len(members),
        source_v5_hash=digest(initial_v5["scores"]), history_scope="frozen_full_packet_hash_and_event_weight_reconciliation",
        probability_replay_scope="original_source_selected_and_last_only", gt=0, query=0, optimizer_updates=0,
        elapsed_seconds=perf_counter()-started))

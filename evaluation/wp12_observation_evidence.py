"""Adapt the frozen CUR-O development review and admit synthetic factorial receipts."""

from collections import Counter
import csv
from itertools import product
import json
import math
from pathlib import Path

from evaluation.classification_metrics import classification_metrics
from evaluation.wp12_cost_primitives import pin
from evaluation.wp12_low_rank_evidence import verify_guard
from evaluation.wp12_production_evidence import FULL_HEADS, metrics, missing, provenance
from model.research_config import digest
from research_runtime import hash_file
from transfer.curriculum_factorial import ARMS, FORMAT as FACTORIAL, SYNTHETIC_BUDGET
from transfer.curriculum_factorial_audit import effects
from transfer.curriculum_observation_admission import CONDITIONS
from transfer.curriculum_observation_real import FORMAT, REAL, REAL_ADMISSION, SAMPLE_WITNESS
from transfer.observation_curriculum import ORDERS
from transfer.registered_wp11_learning import read


STAGES = {"full": 178, "prefix": 7, "resume": 171, "completed": 0}
SEEDS = (64, 512, 2048)
POINTS = ("selected", "last")
SCORE_FIELDS = ("oa", "aa", "kappa", "macro_f1", "nll")


def _campaign(spec):
    if set(spec) != {"registration", "audit", "campaign"}:
        raise ValueError("Exact observation registration, audit and execution campaign required")
    reg, audit, campaign = (read(spec[k]) for k in ("registration", "audit", "campaign"))
    root = Path(reg["root"])
    old, sample = read(reg["prior"]), read(reg["sample_registration"])
    if (reg["format"] != FORMAT or reg["scope"] != REAL or reg["protocol"] != "wp01_xuzhou_k5_v5"
            or reg["prior"]["sha256"] != REAL_ADMISSION or reg["sample_registration"]["sha256"] != SAMPLE_WITNESS
            or old["sample_registration"] != reg["sample_registration"] or old["real_training"] != "NOT_RUN"
            or old["coverage"].keys() != audit["cases"].keys()
            or (reg["budget"], reg["prefix"], reg["maximum_updates"], reg["preflight_forwards"]) != (178, 7, 4272, 48)
            or reg["validation_conditions"] != list(CONDITIONS) or set(reg["validation_identity"]) != set(CONDITIONS)
            or any(reg[key] != sample[key] for key in ("source", "model_seed", "class_ids", "validation", "exposures",
                                                       "source_replay", "protection", "selection"))
            or reg["train_support"] != sample["support"] or reg["state"] != sample["teacher_state"]
            or reg["cases"] != old["cases"] or reg["policy_hashes"] != old["policy_hashes"]
            or spec["registration"]["path"] != str(root/"registration.json")
            or spec["audit"]["path"] != str(root/"audit.json")
            or audit["status"] != "passed_independent_real_observation" or audit["registration"] != spec["registration"]
            or any(v[k] for v in (reg, audit) for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Reviewed observation producer or evidence scope differs")
    completion, frozen = read(audit["completion"]), read(audit["frozen_selected"])
    notice = read(completion["notice"])
    cases = {c["case"]: c for c in reg["cases"]}
    if (cases.keys() != audit["cases"].keys() or cases.keys() != audit["guards"].keys()
            or {(c["seed"], c["order"]) for c in cases.values()} != set(product(SEEDS, ORDERS))
            or len(reg["cases"]) != len(cases) or any(name != f"s{c['seed']}-{c['order']}" for name, c in cases.items())
            or audit["completion"]["path"] != str(root/"batch-complete.json")
            or audit["frozen_selected"]["path"] != str(root/"frozen-selected.json")
            or completion["notice"]["path"] != str(root/"notice.json") or completion["registration"] != spec["registration"]
            or completion["attempts"].keys() != cases.keys()
            or notice["registration"] != spec["registration"] or notice["cases"] != list(cases)
            or (notice["budget"], notice["maximum_updates"], notice["cpu_workers"]) != (reg["budget"], reg["maximum_updates"], 1)
            or frozen["registration"] != spec["registration"] or frozen["completion"] != audit["completion"]
            or frozen["packages"] != {name: {k: audit["cases"][name]["selected"][k] for k in ("step", "package")} for name in cases}
            or frozen["optimizer_updates"] != 0 or any(completion[k] for k in ("optimizer_updates", "gt", "query"))
            or audit["actual_updates"] != completion["updates"] or completion["updates"] != reg["maximum_updates"]
            or campaign["status"] != "completed" or campaign["commit"] != reg["commit"]
            or campaign["updates"] != completion["updates"] or campaign["gt"] or campaign["query"]):
        raise ValueError("Incomplete observation design, frozen endpoints or execution receipts")
    pinned = dict(registration=spec["registration"], audit=spec["audit"], completion=audit["completion"], notice=completion["notice"])
    if any(campaign[k] != v["sha256"] for k, v in pinned.items()):
        raise ValueError("Campaign input and output digests differ")
    commands = {f"{case}-{stage}" for case in cases for stage in STAGES} | {"complete", "audit"}
    stages = campaign["stages"]
    if set(stages) != commands or not math.isfinite(campaign["wall_seconds"]):
        raise ValueError("Campaign must contain all stage, completion and audit processes")
    directory = Path(spec["campaign"]["path"]).parent
    for key, receipt in stages.items():
        if (receipt["exit"] != 0 or not math.isfinite(receipt["wall_seconds"]) or receipt["wall_seconds"] <= 0
                or read({"path": str(directory/f"{key}.exit.json"),
                         "sha256": hash_file(directory/f"{key}.exit.json")}) != receipt):
            raise ValueError("Observation subprocess failed or its timing differs")
    if campaign["wall_seconds"] < math.fsum(v["wall_seconds"] for v in stages.values()):
        raise ValueError("Campaign wall time is smaller than sequential process timings")
    return reg, audit, campaign, old, sample, cases, completion


def _quality(snapshot, classes, targets, condition, blocks):
    if (snapshot["condition_record"] != condition or snapshot["targets"] != targets
            or snapshot["labels"] != len(targets) or snapshot["forwards"] != blocks
            or len(snapshot["scores"]) != blocks or any(len(v) != len(FULL_HEADS) for v in snapshot["scores"])
            or any(not pin(digest_value) for v in snapshot["scores"] for digest_value in v)
            or set(snapshot["predictions"]) != set(FULL_HEADS)
            or not math.isfinite(snapshot["seconds"]) or snapshot["seconds"] <= 0):
        raise ValueError("Observation condition, development targets or replay cost differs")
    values = metrics(snapshot["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix")
    for head, value in values.items():
        original = snapshot["metrics"][head]
        reconstructed = classification_metrics(targets, snapshot["predictions"][head], classes)
        if (original["confusion_matrix"] != reconstructed["confusion_matrix"]
                or any(original[key] != value[key] for key in ("kappa", "macro_f1"))
                or not math.isfinite(value["nll"]) or value["nll"] < 0
                or original["zero_recall"] != [r["class_id"] for r in value["per_class"] if r["recall"] == 0.]):
            raise ValueError("Observation ordered predictions, confusion or reported metrics differ")
    return dict(condition_record=condition, forwards=blocks, labels=len(targets), seconds=snapshot["seconds"], quality=values)


def _exposure(item, reg, old, policy):
    case, history, value = item["case"], item["history"], item["exposure"]
    events = reg["exposures"][str(case["seed"])]["events"]
    seen, counts = [], Counter()
    labels = {v["index"]: v["label"] for v in reg["train_support"]}
    per_config = {}
    for row, expected, frozen in zip(history, events, policy["training"], strict=True):
        key = digest(frozen["config"])
        bands = len(frozen["selected_positions"])
        graph = row["forward_evidence"]
        if (any(row[k] != v for k, v in expected.items())
                or row["observation"] != dict(policy_hash=reg["policy_hashes"][case["order"]],
                                              record_hash=frozen["record_hash"], selected_band_count=bands)
                or row["weights"] != [1.] * len(row["support_ids"])
                or graph["band_count"] != bands or graph["band_ids"] != frozen["selected_band_ids"]
                or graph["metadata_sha256"] != frozen["selected_metadata_hash"]
                or graph["calls"] != {"adapter": 1, "denoise": 1, "graph": 1}):
            raise ValueError("Executed observation record or frozen support event differs")
        config = per_config.setdefault(key, dict(config=frozen["config"], forwards=0, band_exposures=0,
                                                tiles=Counter(), support=Counter(), class_blocks=Counter(),
                                                class_label_visits=Counter()))
        config["forwards"] += 1
        config["band_exposures"] += bands
        config["tiles"][str(row["tile_id"])] += 1
        config["support"].update(map(str, row["support_ids"]))
        config["class_blocks"].update({str(labels[i]) for i in row["support_ids"]})
        config["class_label_visits"].update(str(labels[i]) for i in row["support_ids"])
        seen.append(frozen["record_hash"])
        counts.update(row["support_ids"])
    expected = old["coverage"][case["case"]]
    if (set(value) != {"by_config", "train_band_exposures", "train_pixel_band_exposures", "record_multiset",
                       "train_forwards", "train_label_visits"}
            or value["train_label_visits"] != counts.total() or value["record_multiset"] != digest(sorted(seen))
            or any(v != expected[k] for k, v in value.items() if k != "train_label_visits")
            or value["train_band_exposures"] != sum(v["band_exposures"] for v in per_config.values())
            or value["by_config"].keys() != per_config.keys()
            or any(any(value["by_config"][key][k] != v for k, v in counts.items()) for key, counts in per_config.items())
            or value["train_pixel_band_exposures"] != sum(v["pixels_times_bands"] for v in value["by_config"].values())
            or value["train_forwards"] != reg["budget"] or set(counts) != set(labels)):
        raise ValueError("Actual joint observation exposure differs from registered marginals or events")
    return value


def _training(item, campaign, blocks, labels):
    stage_costs = item["stage_costs"]
    if set(stage_costs) != set(STAGES):
        raise ValueError("Observation recovery and completed-state reentry costs are required")
    events = 0
    seconds = Counter()
    for name, updates in STAGES.items():
        entry = stage_costs[name]
        time = entry["timing"]
        if (entry["new_updates"] != updates or time.get("training_forward", {}).get("calls", 0) != updates
                or time["full_v5_event"]["calls"] != updates + (2 if name in ("resume", "completed") else 0)):
            raise ValueError("Stage timing includes an unexpected observation or V5 call count")
        events += time["full_v5_event"]["calls"]
        for key in ("restore_or_prepare_seconds", "update_loop_seconds", "checkpoint_finish_seconds"):
            if not math.isfinite(entry[key]) or entry[key] < 0:
                raise ValueError("Finite recorded stage timing required")
            seconds[key] += entry[key]
        seconds["subprocess_wall_seconds"] += campaign["stages"][f"{item['case']['case']}-{name}"]["wall_seconds"]
    updates = sum(STAGES.values())
    if events != item["v5_stage_events"] or item["updates"] != updates:
        raise ValueError("Observation case stage totals differ")
    extra = events - updates
    return dict(scope="registered_cpu_stage_cost", updates=updates, training_forwards=updates,
                label_accesses=2 * item["exposure"]["train_label_visits"], training_v5_events=updates,
                restore_v5_events=extra, training_v5_forwards=blocks*updates, restore_v5_forwards=blocks*extra,
                training_v5_labels=labels*updates, restore_v5_labels=labels*extra, stage_v5_label_accesses=labels*events,
                **seconds)


def _endpoint(entry, item, point, root, reg):
    history = item["history"]
    step = min(range(len(history)), key=lambda i: history[i]["validation"]["nll"]) + 1 if point == "selected" else reg["budget"]
    package = entry["package"]
    expected = root/"endpoints"/item["case"]["case"]/f"{point}.pt"
    if (entry["step"] != step or package["state"] != history[step-1]["state_hash"] or package["format"] != "fp32"
            or package["weights"]["path"] != str(expected) or package["sidecar"]["path"] != str(expected)+".json"
            or any(hash_file(package[k]["path"]) != package[k]["sha256"] for k in ("weights", "sidecar"))
            or package["bytes"] != expected.stat().st_size + Path(package["sidecar"]["path"]).stat().st_size
            or set(entry["conditions"]) != set(CONDITIONS)):
        raise ValueError("Observation endpoint package or first-min selection differs")
    original, actual = history[step-1]["validation"], entry["conditions"]["full"]["metrics"]["fused"]
    if (actual["confusion_matrix"] != original["confusion"] or actual["oa"] != original["oa"]
            or not math.isclose(actual["nll"], original["nll"], abs_tol=1e-7)):
        raise ValueError("Full-condition replay differs from its selected historical event")


def adapt(spec):
    reg, audit, campaign, old, sample, cases, completion = _campaign(spec)
    root = Path(reg["root"])
    origin = read(reg["source"]["origin"])
    historic, stage = read(origin["registration"]), read(origin["off_full"])
    if (origin["source"] != reg["source"]["weights"] or origin["sidecar"] != reg["source"]["sidecar"]
            or (origin["selected_step"], historic["budget"]) != (246, 256) or stage["new_updates"] != historic["budget"]
            or stage["finish"]["selected_state_hash"] != origin["selected_state"] or reg["state"] != origin["selected_state"]
            or set(audit["source"]) != set(CONDITIONS)
            or any(audit["source"]["full"][k] != reg["source_replay"][k] for k in ("scores", "metrics", "forwards"))
            or any(hash_file(origin[k]["path"]) != origin[k]["sha256"] for k in ("source", "sidecar"))):
        raise ValueError("Observation source differs from the original WP08 off selected endpoint")
    proof = dict(source_commit=origin["producer_commit"], witness_commit=historic["commit"],
                 source_evidence=origin["registration"], source_steps=historic["budget"])
    classes, support, val = reg["class_ids"], reg["train_support"], reg["validation"]
    released = reg["exposures"][str(SEEDS[0])]["support"]
    targets = audit["source"]["full"]["targets"]
    small = sorted(c for c, n in Counter(released["labels"]).items() if n <= reg["protection"]["small_support_max"])
    blocks = audit["v5_blocks_per_event"]
    if (len(classes) != 9 or (len(support), len(val["indices"]), released["seed"], val["seed"]) != (45, 45, 512, 3031)
            or (len(targets), blocks) != (45, 5) or Counter(targets) != Counter(val["labels"])
            or released["indices"] != [v["index"] for v in support] or released["labels"] != [v["label"] for v in support]
            or any(v["support"] != released for v in reg["exposures"].values()) or small != audit["small_classes"]
            or audit["labels_per_v5_event"] != len(targets)):
        raise ValueError("Observation support and role identity differs from released K5/V5")
    common = dict(scene="xuzhou", data_protocol=reg["protocol"], model_seed=reg["model_seed"], train_labels=len(support),
                  validation_labels=len(targets), class_ids=classes, split=sample["preprocessing"], support_sha256=digest(released),
                  evaluation_sha256=digest(val), validation_role="repeated_development_v5")
    condition_identity = reg["validation_identity"]
    initial = {c: _quality(audit["source"][c], classes, targets, condition_identity[c], blocks) for c in CONDITIONS}
    no_cost = missing("NOT_RUN", "No observation curriculum deployment benchmark was registered")
    no_teacher = missing("NOT_RUN", "Observation recovery does not require teacher scoring")
    condition_records = read(old["policies"]["fixed"])["validation"]
    condition_bands = {c: condition_records[c]["selected_band_ids"] for c in CONDITIONS}
    if any(condition_records[c]["record_hash"] != condition_identity[c] for c in CONDITIONS):
        raise ValueError("Validation conditions differ from the admitted fixed reference")
    protocol = dict(scope="curriculum_observation_full_v5_selection", primary_condition="full",
                    conditions=condition_identity, condition_bands=condition_bands)
    source = dict(module="WP01-CUR-O", arm="original_fp32", case_id="original_fp32", identity=dict(common, sampler_seed=None),
                  **provenance(**proof), state=reg["state"], package=origin["source"]["sha256"], package_bytes=None,
                  precision="fp32", budget=0, actual_optimizer_calls=0, selected_step=origin["selected_step"],
                  selection="original_wp08_off", quality=initial["full"]["quality"], protection=None, cost=no_cost,
                  observation_protocol=protocol, condition_quality={"initial": dict(step=0, conditions=initial)})
    rows = [source]
    for name, case in cases.items():
        item = audit["cases"][name]
        policy = read(old["policies"][case["order"]])
        if (item["case"] != case or len(item["history"]) != reg["budget"]
                or policy["policy_hash"] != reg["policy_hashes"][case["order"]]
                or any(policy["validation"][c] != condition_records[c] for c in CONDITIONS)):

            raise ValueError("Recorded case or frozen observation policy differs")
        attempts = completion["attempts"][name]
        if set(attempts) != set(STAGES):
            raise ValueError("All observation interval receipts are required")
        for stage_name, witness in attempts.items():
            raw_stage = read(witness)
            if (witness["path"] != str(root/"attempts"/name/stage_name/"status.json") or raw_stage["status"] != "passed"
                    or any(raw_stage[k] != v for k, v in item["stage_costs"][stage_name].items())):
                raise ValueError("Observation stage cost differs from its executed receipt")
        exposure = _exposure(item, reg, old, policy)
        training = _training(item, campaign, blocks, len(targets))
        reference = audit["cases"][f"s{case['seed']}-fixed"]
        reviewed = {}
        for point in POINTS:
            entry = item[point]
            _endpoint(entry, item, point, root, reg)
            conditions = {}
            for c in CONDITIONS:
                snapshot = entry["conditions"][c]
                guard = verify_guard(reference[point]["conditions"][c], snapshot, classes, small, reg["protection"]["limits"])
                if guard != audit["guards"][name][point][c]:
                    raise ValueError("Recorded observation protection differs from same-seed fixed")
                conditions[c] = dict(**_quality(snapshot, classes, targets, condition_identity[c], blocks), protection=guard)
            reviewed[point] = dict(step=entry["step"], conditions=conditions)
        losses = {}
        for point, event in zip(("first", "last"), (item["history"][0], item["history"][-1]), strict=True):
            losses[point] = dict(step=event["step"], train_loss=event["loss"], full_v5_fused_nll=event["validation"]["nll"])
            if not all(math.isfinite(v) for v in losses[point].values()):
                raise ValueError("Nonfinite first or last observation loss")
        chosen, package = reviewed["selected"], item["selected"]["package"]
        primary = chosen["conditions"]["full"]
        lineage = provenance(**proof, recovery_commit=reg["commit"], recovery_steps=reg["budget"],
                             release_commit=reg["commit"], release_evidence=spec["audit"])
        rows.append(dict(module="WP01-CUR-O", arm=case["order"], case_id=name, identity=dict(common, sampler_seed=case["seed"]),
                         **lineage, state=package["state"], package=package["weights"]["sha256"], package_bytes=package["bytes"],
                         precision="fp32", budget=reg["budget"], actual_optimizer_calls=item["updates"],
                         selected_step=chosen["step"], selection=reg["selection"], quality=primary["quality"],
                         protection=primary["protection"], cost=no_cost, observation_protocol=dict(protocol, policy_hash=policy["policy_hash"]),
                         condition_quality=reviewed, initial_condition_reference="original_fp32",
                         training_cost=training, observation_exposure=exposure, loss_trace_summary=losses))
    training = {key: sum(r["training_cost"][key] for r in rows[1:]) for key in
                ("training_forwards", "label_accesses", "training_v5_events", "restore_v5_events",
                 "training_v5_forwards", "restore_v5_forwards", "training_v5_labels", "restore_v5_labels")}
    replays = [condition for row in rows[1:] for point in row["condition_quality"].values()
               for condition in point["conditions"].values()]
    source_forwards, source_labels = sum(v["forwards"] for v in initial.values()), sum(v["labels"] for v in initial.values())
    endpoint_forwards, endpoint_labels = sum(v["forwards"] for v in replays), sum(v["labels"] for v in replays)
    totals = dict(actual_training_forwards=training["training_forwards"], actual_train_labels=training["label_accesses"],
                  training_v5_events=training["training_v5_events"], restore_v5_events=training["restore_v5_events"],
                  source_condition_forwards=source_forwards, source_condition_labels=source_labels,
                  endpoint_condition_forwards=endpoint_forwards, endpoint_condition_labels=endpoint_labels)
    if any(audit[k] != v for k, v in totals.items()):
        raise ValueError("Observation campaign validation and training cost totals differ")
    return rows, dict(producer=reg["commit"], **spec, completion=audit["completion"], frozen_selected=audit["frozen_selected"],
                      protocol=reg["protocol"], cases=len(cases), seeds=list(SEEDS), budget=reg["budget"], prefix=reg["prefix"],
                      actual_optimizer_calls=audit["actual_updates"], previous_preflight_forwards=reg["preflight_forwards"],
                      preparation_seconds=reg["preparation_seconds"], campaign_wall_seconds=campaign["wall_seconds"],
                      audit_seconds=audit["audit_seconds"], condition_review_seconds=audit["condition_review_seconds"],
                      independent_input_replays=audit["independent_input_replays"], training=training, **totals,
                      teacher=no_teacher, deployment=no_cost, initial_reference="original_fp32",
                      matched_reference="same_recovery_seed_fixed_observation_order",
                      independent_spatial=missing("NOT_RUN", "Only the fixed released development V5 was evaluated"))


def synthetic(spec):
    if set(spec) != set(map(str, SYNTHETIC_BUDGET)):
        raise ValueError("Exactly the three synthetic factorial class counts are required")
    receipts = {}
    for count, budget in SYNTHETIC_BUDGET.items():
        pin = spec[str(count)]
        audit = read(pin)
        reg, complete = read(audit["registration"]), read(audit["completion"])
        if (audit["status"] != "passed_synthetic_factorial" or audit["scope"] != "synthetic"
                or (audit["fixture_classes"], audit["budget"]) != (count, budget)
                or reg["format"] != FACTORIAL or reg["scope"] != "synthetic" or reg["budget"] != budget
                or audit["producer"] != reg["commit"] or complete["registration"] != audit["registration"]
                or audit["results"].keys() != set(ARMS) or audit["joint_exposure"].keys() != set(ARMS)
                or set(audit["effects"]) != set(FULL_HEADS)
                or reg["maximum_updates"] != 8*budget or complete["updates"] != 8*budget
                or audit["synthetic_updates"] != 8*budget or complete["real_training"] != "NOT_RUN"
                or any(v[k] for v in (audit, reg, complete) for k in ("optimizer_updates", "query", "gt"))
                or audit["real_proposal"]["status"] != "NOT_RUN" or audit["real_proposal"]["optimizer_updates"] != 0):
            raise ValueError("Synthetic factorial scope or execution budget differs")
        request = read(reg["request"])
        plan = read(request["exposure"])
        record = read(request["development"])["record"]
        if (request["scope"] != "synthetic" or plan["plan_hash"] != reg["exposure_hash"]
                or len(record["classes"]) != count or (plan["updates"], plan["seed"]) != (budget, 64)):
            raise ValueError("Synthetic factorial role or exposure identity differs")
        for head, contrasts in audit["effects"].items():
            if set(contrasts) != set(SCORE_FIELDS):
                raise ValueError("Synthetic factorial effect metrics differ")
            for metric, value in contrasts.items():
                actual = effects([dict(arm=arm, source=reg["source_state"], labels=digest(record["validation_support"]),
                                       exposure=reg["exposure_hash"], budget=budget, metric=metric, seed=plan["seed"],
                                       condition="full_development_v5", value=row["metrics"][head][metric])
                                  for arm, row in audit["results"].items()])
                if actual != value:
                    raise ValueError("Synthetic factorial effect arithmetic differs")
        receipts[str(count)] = dict(audit=pin, producer=audit["producer"], synthetic_updates=8*budget,
                                    budget=budget, effects=audit["effects"], real_proposal=audit["real_proposal"])
    return dict(scope="synthetic_engineering_only", status="passed", fixtures=receipts, optimizer_updates=0,
                real_training="NOT_RUN", production_rows_added=0)

def write_conditions(out, rows):
    fields = ("module", "case_id", "arm", "scene", "data_protocol", "recovery_seed", "model_seed", "endpoint", "step",
              "selection_condition", "validation_condition", "condition_sha256", "band_count", "head",
              *SCORE_FIELDS, "guard_passed", "failed_recall_classes", "new_small_class_errors", "per_class_recall",
              "forwards", "validation_labels", "replay_seconds")
    with (out/"observation-condition-table.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["module"] != "WP01-CUR-O":
                continue
            identity = row["identity"]
            protocol = row["observation_protocol"]
            for point, value in row["condition_quality"].items():
                for condition, snapshot in value["conditions"].items():
                    for head, score in snapshot["quality"].items():
                        guard = snapshot.get("protection", {}).get(head)
                        writer.writerow(dict(module=row["module"], case_id=row["case_id"], arm=row["arm"],
                            scene=identity["scene"], data_protocol=identity["data_protocol"], recovery_seed=identity["sampler_seed"],
                            model_seed=identity["model_seed"], endpoint=point, step=value["step"],
                            selection_condition=protocol["primary_condition"], validation_condition=condition,
                            condition_sha256=snapshot["condition_record"], band_count=len(protocol["condition_bands"][condition]),
                            head=head, **{key: score[key] for key in SCORE_FIELDS},
                            guard_passed=guard["passes_metric_guards"] if guard else None,
                            failed_recall_classes=json.dumps(guard["failed_recall_classes"]) if guard else None,
                            new_small_class_errors=json.dumps(guard["new_small_class_errors"], sort_keys=True) if guard else None,
                            per_class_recall=json.dumps({v["class_id"]: v["recall"] for v in score["per_class"]}),
                            forwards=snapshot["forwards"], validation_labels=snapshot["labels"], replay_seconds=snapshot["seconds"]))

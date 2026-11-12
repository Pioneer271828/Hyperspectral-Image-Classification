"""Adapt reviewed WP10 and WP11 artifacts into a single production evidence table."""

import csv
from pathlib import Path

from evaluation.unified_cohort import _counts
from evaluation.wp12_cost_primitives import pin, timing_summary
from model.research_config import digest
from research_runtime import external_directory, hash_file, write_json
from transfer.registered_wp11_learning import read


POLICY = "wp12-production-evidence-v1"
WP10_COMMIT = "9dea199653079c9c430264b72ac968f1fefd7d7e"
WP10_MANIFEST = "025b4175bd7fd725dd9fdcde57632f8188eee950d89b31f99e608b2d3c172cfe"
WP11_TRAINED = "ab0a169d6e8121484f892e443dca7f23de056508"
WP11_STATIC = "5d2265c73e7c38450341a3b9a5d7150660593553"
FULL_HEADS = ("fused", "graph", "cnn")


def missing(status, reason):
    return dict(status=status, reason=reason)


def archive(spec):
    root = external_directory(spec["root"])
    manifest = root / "SHA256SUMS"
    if spec["manifest_sha256"] != WP10_MANIFEST or hash_file(manifest) != WP10_MANIFEST:
        raise ValueError("Only the independently reviewed WP10 archive is accepted")
    entries = {}
    for line in manifest.read_text().splitlines():
        sha, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not pin(sha) or root not in path.parents or name in entries or hash_file(path) != sha:
            raise ValueError("WP10 archive asset identity differs")
        entries[name] = sha

    def load(name):
        return read(dict(path=str(root / name), sha256=entries[name]))

    return load, entries


def metrics(heads, classes, expected, *, matrix_key):
    if heads.keys() != set(expected):
        raise ValueError("Classifier exit set differs from its recorded model")
    result = {}
    for name, head in heads.items():
        matrix = head[matrix_key]
        value = _counts(matrix, classes)
        if (head["count"] != value["count"] or head["oa"] != value["oa"]
                or [p["support"] for p in value["per_class"]] != [5] * len(classes)):
            raise ValueError("Recorded development quality differs from its confusion")
        if "aa" in head and head["aa"] != value["aa"]:
            raise ValueError("Recorded macro recall differs from the confusion")
        value["nll"] = head["nll"]
        result[name] = value
    return result


def timing(values, blocks):
    stats = timing_summary(values, repeats=len(values), blocks=blocks)
    return dict(p50=stats["p50_ms"], p95=stats["p95_ms"], n=len(values), samples=values)


def wp10_cost(resident, receipt, arm, entry, samples):
    measured = receipt["cases"][arm]
    if measured["package"] != entry["package"] or measured["arm"] != {"fp32": "fp32", "ptq": "qat-both", "ternary": "ternary-first"}[arm]:
        raise ValueError("WP10 accuracy endpoint differs from the timed package")
    model = resident["models"][arm]
    workers = receipt["workers"][arm]
    if (len(workers) != 3 or len({v["pid"] for v in workers}) != 3
            or model["state_sha256"] != entry["package"]["state"]
            or any(v["tiles"] != resident["tiles"] or v["output_hashes"] != resident["output_hashes"][arm]
                   or v["package_bytes"] != entry["package"]["bytes"] or v["affinity"] != resident["affinity"]
                   for v in workers)):
        raise ValueError("WP10 startup and resident identity differ")
    blocks = len(resident["tiles"])
    resident_t = timing([v["timings"][arm]["wall_ms"] for v in samples], blocks)
    load = timing([v["load_ms"] for v in workers], blocks)
    first = timing([v["first_sweep_ms"] for v in workers], blocks)
    peak = timing([v["after_first_sweep"]["peak_rss_kib"] for v in workers], blocks)
    reference = receipt["startup_summary"][arm]
    if (resident_t["p50"] != resident["summary"][arm]["wall_p50_ms"]
            or resident_t["p95"] != resident["summary"][arm]["wall_p95_ms"]
            or any((actual["p50"], actual["p95"]) != (reference[key]["p50"], reference[key]["p95"])
                   for key, actual in (("load_ms", load), ("first_sweep_ms", first), ("peak_rss_kib", peak)))):
        raise ValueError("WP10 reported quantiles differ from original samples")
    return dict(status="MEASURED", device="cpu", resident_scope=resident["scope"],
                startup_scope="separate_model_load_and_first_sweep",
                workload=digest(resident["tiles"]), pixels=resident["pixels_per_sweep"], blocks=blocks,
                parameters=model["parameter_count"], package_bytes=entry["package"]["bytes"],
                load_ms=load, first_ms=first, resident_ms=resident_t, peak_rss_kib=peak,
                end_to_end_ms=missing("NOT_RUN", "No continuous process-to-output timing was registered"),
                threads=resident["threads"], affinity=resident["affinity"], cache="uncontrolled")


def adapt_wp10(spec):
    load, pins = archive(spec)
    reg, audit, end, receipt, resident = (load(name) for name in
        ("real/registration.json", "real/audit.json", "real/batch-complete.json",
         "real/cost/receipt.json", "real/cost/resident.json"))
    if (reg["commit"] != WP10_COMMIT or reg["scope"] != "wp08_xuzhou_off" or reg["maximum_updates"] != 2170
            or audit["scope"] != reg["scope"] or audit["actual_updates"] != end["updates"]
            or end["updates"] != reg["maximum_updates"] or audit["independent_updates"] != 0
            or audit["registration"]["sha256"] != pins["real/registration.json"]
            or audit["completion"]["sha256"] != pins["real/batch-complete.json"]
            or receipt["audit"]["sha256"] != pins["real/audit.json"]
            or receipt["resident"]["sha256"] != pins["real/cost/resident.json"]
            or receipt["registration"]["sha256"] != audit["registration"]["sha256"]
            or any(result["query"] != 0 or result["gt"] != 0 for result in (audit, end, receipt))
            or resident["scope"] != "cpu_resident_full_model_forward" or resident["threads"] != 1
            or resident["query_labels_accessed"] is not False or resident["dataset"] != "xuzhou"
            or resident["repeats"] != len(resident["samples"])):
        raise ValueError("Reviewed WP10 model, evaluation or CPU timing identity differs")
    classes = reg["validation"]["class_ids"]
    support = reg["exposure"]["support"]
    family = dict(scene="xuzhou", sampler_seed=reg["seed"], model_seed=reg["seed"],
                  support_sha256=digest(support), train_labels=len(support["indices"]),
                  validation_labels=45, class_ids=classes,
                  split=reg["validation"]["identity"], validation_role="repeated_development_v5",
                  evaluation_sha256=digest(reg["validation"]))
    measured = {row["arm"]: key for key, row in receipt["cases"].items()}
    guards = {row["arm"]: row for chosen in audit["decisions"]["selected"].values() for row in chosen["trials"]}
    results = []
    for arm in reg["arm_order"]:
        selected = audit["report"][arm]["selected"]
        cost = (wp10_cost(resident, receipt, measured[arm], selected, resident["samples"])
                if arm in measured else missing("NOT_RUN", "This candidate had no registered CPU deployment measurement"))
        results.append(dict(module="WP10", arm=arm, identity=family, producer=reg["commit"],
                            state=selected["package"]["state"], package=selected["package"]["weights"]["sha256"],
                            precision=selected["package"]["format"], package_bytes=selected["package"]["bytes"],
                            budget=reg["budget"], selected_step=selected["step"], selection=reg["selection"],
                            quality=metrics(selected["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                            protection=guards.get(arm), cost=cost))
    results.append(dict(module="WP10", arm="original_fp32", identity=family, producer=reg["source"]["historical_baseline"],
                        state=reg["initial"]["fp32"]["evidence"]["dense_state"],
                        package=reg["source"]["weights"]["sha256"], precision="fp32", package_bytes=None,
                        budget=0, selected_step=reg["source"]["historical_selected_step"],
                        selection="original_off_selected",
                        quality=metrics(audit["source"]["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                        protection=None, cost=missing("NOT_RUN", "Original unadapted reference had no WP10 CPU timing")))
    by_arm = {r["arm"]: r for r in results}
    q, f = by_arm["qat-both"], by_arm["fp32"]
    q_cnn = q["protection"]["heads"]["cnn"]
    comparison = dict(arm=q["arm"], baseline=f["arm"], device="cpu", scope=q["cost"]["resident_scope"],
                      same_development_evaluation=True, same_cpu_workload=q["cost"]["workload"] == f["cost"]["workload"],
                      fused_guard=q["protection"]["heads"]["fused"]["passes_metric_guards"],
                      cnn_guard=q_cnn["passes_metric_guards"], cnn_failed_classes=q_cnn["failed_recall_classes"],
                      cnn_new_errors=q_cnn["new_small_class_errors"], cnn_oa_drop=q_cnn["oa_drop"],
                      resident_p50_delta_ms=q["cost"]["resident_ms"]["p50"] - f["cost"]["resident_ms"]["p50"],
                      first_p50_delta_ms=q["cost"]["first_ms"]["p50"] - f["cost"]["first_ms"]["p50"],
                      ternary_fallback=audit["decisions"]["selected"]["ternary"]["arm"],
                      evidence_role="one_development_split_one_seed", overall_deployment_benefit=False)
    return results, comparison


def wp11_cost(receipt, mode, state):
    actual = receipt["summary"][mode]
    workers = receipt["startup"][mode]
    resident = receipt["resident"]
    if (actual["weights"] != state or len(workers) != 3 or len(resident) != 12
            or any(v["path"] != mode or v["release"] != receipt["release"]["sha256"]
                   or v["input_hash"] != receipt["input_hash"] or v["environment"] != receipt["environment"]
                   or v["first"]["output_hashes"] != actual["output_hashes"] or v["parameters"] != actual["parameters"]
                   for v in workers)
            or any(row["paths"][mode]["output_hashes"] != actual["output_hashes"] for row in resident)):
        raise ValueError("WP11 model or input differs from its timed outputs")
    blocks = receipt["blocks"]
    series = {
        "resident_ms": [r["paths"][mode]["wall_ms"] for r in resident],
        "prepare_ms": [w["prepare_ms"] for w in workers],
        "load_ms": [w["load_ms"] for w in workers],
        "first_ms": [w["first"]["wall_ms"] for w in workers],
        "end_to_end_ms": [w["ready_to_first_output_ms"] for w in workers],
        "peak_rss_kib": [w["peak_rss_kib"] for w in workers],
    }
    values = {key: timing(samples, blocks) for key, samples in series.items()}
    for key, value in values.items():
        source = actual["ready_to_first_output_ms" if key == "end_to_end_ms" else key]
        if (value["p50"], value["p95"]) != (source["p50"], source["p95"]):
            raise ValueError("WP11 reported quantiles differ from original samples")
    return dict(status="MEASURED", device="cpu", resident_scope=receipt["environment"]["scope"],
                startup_scope="ready_process_through_verified_first_output",
                workload=receipt["input_hash"], blocks=blocks, pixels=receipt["pixels"], threads=1,
                affinity=receipt["environment"]["affinity"], cache=receipt["environment"]["fs_cache"],
                parameters=actual["parameters"], package_bytes=actual["package_bytes"],
                graph_builds_per_scan=workers[0]["first"]["graph_builds"], **values)


def adapt_wp11(spec):
    release, cost, unified = (read(spec[key]) for key in ("release", "cost", "unified"))
    audit, reg = read(release["history_evidence"]), read(release["registration"])
    origin = read(release["source_evidence"])
    source_config = read(release["source_metadata"])["training_config"]
    request = read(reg["request"])
    use = audit["label_use"]
    if (release["commit"] != WP11_TRAINED or reg["commit"] != release["commit"] or audit["commit"] != release["commit"]
            or audit["registration"] != release["registration"] or audit["status"] != "audited"
            or audit["actual_updates"] != reg["total_updates"] or reg["total_updates"] != 2048
            or reg["updates_per_arm"] != 512 or request["scope"] != "xuzhou_90_45"
            or audit["query"] != 0 or use["query"] != 0 or use["read_confirmation"] is not False
            or use["read_calibration"] is not False or audit["exact_continuous_resume_completed"] is not True
            or use["development"] != release["development"] or use["scene"] != "xuzhou"
            or len(use["training_indices"]) != 90 or len(use["selection_indices"]) != 45
            or cost["commit"] != WP11_STATIC or cost["status"] != "passed" or cost["query"] != 0
            or cost["release"] != spec["release"] or cost["scope"] != "existing_devval_image_only"
            or cost["environment"]["device"] != "cpu" or cost["environment"]["threads"] != 1
            or cost["environment"]["scope"] != "static_cpu_resident_requests"
            or unified["release"] != spec["release"] or unified["scope"] != request["scope"]
            or unified["static_input"] != cost["input_hash"] or unified["archived_cost"]["file"] != spec["cost"]
            or unified["routing"] is not None or unified["observation_role"] != "devval_image_only"):
        raise ValueError("WP11 model, development role, independent status or CPU timing scope differs")
    classes = unified["classes"]
    family = dict(scene=use["scene"], sampler_seed=request["inputs"]["training"]["seed"],
                  source_model_seed=source_config["model_seed"], support_sha256=use["training_support_hash"], train_labels=90,
                  validation_labels=45, class_ids=classes, split=unified["preprocessing"],
                  validation_role="repeated_development_v5", evaluation_sha256=use["selection_support_hash"])
    endpoints = audit["endpoints"]
    available = {"light": (endpoints["light"]["selected"], ("light",), reg["updates_per_arm"]),
                 "full_source": (endpoints["full"]["initial"], FULL_HEADS, 0),
                 "full_matched": (endpoints["full"]["selected"], FULL_HEADS, reg["updates_per_arm"])}
    rows = []
    for mode, (values, names, updates) in available.items():
        model = unified["models"][mode]
        if (model["heads"] != list(names) or model["precision"] != "fp32"
                or (mode != "light" and model["state"] != release["source_state" if mode == "full_source" else "full_state"])):
            raise ValueError("WP11 execution manifest uses another classifier")
        cost_row = wp11_cost(cost, mode, model["state"])
        rows.append(dict(module="WP11", arm=mode, identity=family, producer=release["commit"],
                         state=model["state"], package=model["package"]["sha256"],
                         package_bytes=cost_row["package_bytes"], precision=model["precision"],
                         budget=updates, historical_source_updates=origin["steps"], selection=model["selection"],
                         selected_step=model["selected_step"], quality=metrics(values, classes, names, matrix_key="confusion"),
                         protection=missing("BLOCKED", "Independent role-scoped label authorization is absent"),
                         cost=cost_row))
    return rows


def build(spec):
    if set(spec) != {"wp10", "wp11"}:
        raise ValueError("Exact WP10 and WP11 evidence references required")
    rows, comparison = adapt_wp10(spec["wp10"])
    if "blocked_source" in spec["wp11"]:
        blocked = read(spec["wp11"]["blocked_source"])
        if blocked["exposure"]["meets_minimum"] is not False or blocked["optimizer_updates"] != 0:
            raise ValueError("WP11 blocked source needs a zero-update exposure rejection")
        wp11 = missing("BLOCKED", "The registered 512-step stream misses training support")
    else:
        rows.extend(adapt_wp11(spec["wp11"]))
        wp11 = missing("BLOCKED", "Independent calibration and confirmation require both authorized custody packages")
    return dict(policy=POLICY, status="production_development_evidence", rows=rows, wp10_comparison=comparison,
                wp11_independent=wp11, gpu=missing("NOT_RUN", "The supplied evidence measures CPU only"),
                cross_protocol_benefit=missing("NOT_RUN", "WP10 K5/V5 and WP11 K10/V5 use distinct training evidence"),
                synthetic="Engineering receipts are outside the real production table",
                raw_label_reads=0, query_reads=0, optimizer_updates=0)


def write_table(out, report):
    write_json(out / "production-evidence.json", report)
    fields = ("module", "arm", "scene", "sampler_seed", "train_labels", "validation_labels", "role", "updates",
              "selected_step", "precision", "head", "oa", "aa", "kappa", "macro_f1", "nll",
              "parameters", "package_bytes", "cost_status", "device", "resident_scope",
              "resident_p50_ms", "load_p50_ms", "first_p50_ms", "end_to_end_p50_ms", "peak_rss_p50_kib")
    with (out / "production-table.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            cost, identity = row["cost"], row["identity"]

            def p50(key):
                return cost.get(key, {}).get("p50")

            for head, metrics in row["quality"].items():
                writer.writerow(dict(module=row["module"], arm=row["arm"], scene=identity["scene"],
                     sampler_seed=identity["sampler_seed"], train_labels=identity["train_labels"],
                     validation_labels=identity["validation_labels"], role=identity["validation_role"], updates=row["budget"],
                     selected_step=row["selected_step"], precision=row["precision"], head=head,
                     **{key: metrics[key] for key in ("oa", "aa", "kappa", "macro_f1", "nll")},
                     parameters=cost.get("parameters"), package_bytes=row["package_bytes"], cost_status=cost["status"],
                     device=cost.get("device"), resident_scope=cost.get("resident_scope"),
                     resident_p50_ms=p50("resident_ms"), load_p50_ms=p50("load_ms"), first_p50_ms=p50("first_ms"),
                     end_to_end_p50_ms=p50("end_to_end_ms"), peak_rss_p50_kib=p50("peak_rss_kib")))

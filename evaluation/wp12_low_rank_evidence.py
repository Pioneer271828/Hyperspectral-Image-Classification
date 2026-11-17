"""Verify and adapt a real dynamic low-rank case registry without pooling seeds."""

from collections import Counter
from fractions import Fraction
from itertools import permutations
from pathlib import Path

from evaluation.wp12_production_evidence import FULL_HEADS, metrics, missing, provenance, timing
from inference.wp10_low_rank_execution import KIND, POLICY as DEPLOYMENT_POLICY, WORKER_SCOPE
from model.research_config import digest
from research_runtime import hash_file
from transfer.registered_wp11_learning import read


def verify_guard(reference, candidate, classes, small, limits):
    """Recompute guards from recorded matrices and ordered released predictions."""
    if candidate["targets"] != reference["targets"]:
        raise ValueError("Low-rank V5 score identities differ")
    result = {}
    for head in FULL_HEADS:
        left = reference["metrics"][head]["confusion_matrix"]
        right = candidate["metrics"][head]["confusion_matrix"]
        support = [sum(row) for row in left]
        drops = [Fraction(left[i][i] - right[i][i], n) for i, n in enumerate(support)]
        oa = sum(d*n for d, n in zip(drops, support, strict=True))/sum(support)
        aa = sum(drops)/len(drops)
        failed = [c for c, drop in zip(classes, drops, strict=True) if drop > Fraction(str(limits["recall"]))]
        errors = {str(c): sum(y == c and old == c and new != c for y, old, new in zip(
                  reference["targets"], reference["predictions"][head], candidate["predictions"][head], strict=True))
                  for c in small}
        ok = oa <= Fraction(str(limits["oa"])) and aa <= Fraction(str(limits["aa"])) and not failed and not any(errors.values())
        result[head] = dict(oa_drop=float(oa), aa_drop=float(aa), per_class_recall_drop=dict(zip(
            map(str, classes), map(float, drops), strict=True)), failed_recall_classes=failed,
            new_small_class_errors=errors, passes_metric_guards=ok)
    return result


def measured_rows(manifest, receipt, groups):
    entries, measured = manifest["cases"], {}
    hashes = {}
    summaries = {}
    pids = set()
    for key, group in groups.items():
        names = group["cases"]
        run = group["timing"]
        calls = tuple(permutations(sorted(names)))
        kind = "expansion_control" if "full-rank" in names else "reduction_and_random"
        if (group["kind"] != kind or kind == "expansion_control" and set(names) != {"dense", "full-rank"}
                or "dense" not in names or set(run["models"]) != set(names) or run["repeats"] != 12
                or run["warmups_per_arm"] != 3 or run["scope"] != "cpu_resident_full_model_forward"
                or run["threads"] != 1 or run["tiles"] != manifest["tiles"] or run["dataset"] != manifest["dataset"]
                or run["pixels_per_sweep"] != sum(t["pixels"] for t in manifest["tiles"])
                or Counter(tuple(s["order"]) for s in run["samples"]) != Counter({v: 12//len(calls) for v in calls})):
            raise ValueError("Original low-rank timing scope, full-rank control or balanced order differs")
        summaries[key] = dict(seed=group["seed"], kind=kind, cases=names, scope=run["scope"], samples=run["samples"])
        for arm, name in names.items():
            entry = entries[name]
            if entry["case"]["seed"] != group["seed"] or entry["case"]["arm"] != arm:
                raise ValueError("Cross-seed or wrong-arm cost comparison")
            outputs = run["output_hashes"][arm]
            if hashes.setdefault(name, outputs) != outputs:
                raise ValueError("Dense pair score hashes differ")
            factor_calls = {} if arm == "dense" else dict(right=len(run["tiles"]), left=len(run["tiles"]))
            if group["factors"][arm] != factor_calls:
                raise ValueError("Factor operator execution differs")
            if (run["models"][arm]["parameter_count"] != entry["package"]["parameters"]
                    or run["models"][arm]["state_sha256"] != entry["package"]["state"]):
                raise ValueError("Resident package state or parameters differ")
            values = timing([row["timings"][arm]["wall_ms"] for row in run["samples"]], len(run["tiles"]))
            if (values["p50"], values["p95"]) != (run["summary"][arm]["wall_p50_ms"], run["summary"][arm]["wall_p95_ms"]):
                raise ValueError("Resident samples and reported quantiles differ")
            measured.setdefault(name, []).append((key, values))
    if set(measured) != set(entries) or set(receipt["startup"]) != set(entries):
        raise ValueError("Every admitted case requires its own measurements")
    costs = {}
    for name, entry in entries.items():
        startup = receipt["startup"][name]
        rows = receipt["startup_summary"][name]
        pkg = entry["package"]
        if (len(startup) != 3 or any(w["state"] != pkg["state"] or w["case"] != name
                or w["scope"] != WORKER_SCOPE or w["manifest"] != receipt["manifest"] or w["tiles"] != manifest["tiles"]
                or w["output_hashes"] != hashes[name] or w["package_bytes"] != pkg["bytes"]
                or w["parameters"] != pkg["parameters"] for w in startup)):
            raise ValueError("Low-rank startup workers differ from the timed package")
        pids.update(w["pid"] for w in startup)
        data = dict(load_ms=[w["load_ms"] for w in startup], first_ms=[w["first_sweep_ms"] for w in startup],
                    peak_rss_kib=[w["after_first_sweep"]["peak_rss_kib"] for w in startup])
        values = {k: timing(v, len(manifest["tiles"])) for k, v in data.items()}
        for key, value in values.items():
            declared = rows["first_sweep_ms" if key == "first_ms" else key]
            if (value["p50"], value["p95"]) != (declared["p50_ms"], declared["p95_ms"]):
                raise ValueError("Low-rank startup quantiles differ from the measurements")
        resident_groups = dict(measured[name])
        primary = next((k for k in resident_groups if summaries[k]["kind"] != "expansion_control"), next(iter(resident_groups)))
        costs[name] = dict(status="MEASURED", device="cpu", resident_scope=summaries[primary]["scope"],
                           startup_scope=WORKER_SCOPE, resident_ms=resident_groups[primary], resident_groups=resident_groups,
                           primary_group=primary, blocks=len(manifest["tiles"]),
                           pixels=sum(t["pixels"] for t in manifest["tiles"]), parameters=pkg["parameters"],
                           package_bytes=pkg["bytes"], threads=1, cache=receipt["filesystem_cache"],
                           workload=digest(manifest["tiles"]), **values,
                           end_to_end_ms=missing("NOT_RUN", "Only resident inputs and isolated package load/first sweep were timed"))
    if len(pids) != 3*len(entries):
        raise ValueError("Low-rank startup workers are not independent")
    return costs, summaries


def adapt(spec):
    if set(spec) != {"registration", "audit", "manifest", "cost"}:
        raise ValueError("Explicit low-rank registration, independent audit, manifest and cost required")
    reg, audit, manifest, receipt = (read(spec[key]) for key in ("registration", "audit", "manifest", "cost"))
    if (reg["scope"] != "wp08_xuzhou_off" or audit["scope"] != reg["scope"]
            or reg["policy"] != "wp10_low_rank_recovery_v1" or audit["status"] != "passed_low_rank_recovery_independent"
            or audit["registration"] != spec["registration"] or audit["actual_updates"] != reg["maximum_updates"]
            or any(audit[key] != 0 for key in ("query", "gt", "independent_updates"))
            or manifest["kind"] != KIND or manifest["policy"] != DEPLOYMENT_POLICY
            or manifest["producer_commit"] != reg["commit"] or manifest["registration"] != spec["registration"]
            or manifest["audit"] != spec["audit"] or manifest["observation_role"] != "devval_image_only"
            or manifest["data_protocol"] != "wp10_xuzhou_k5_v5" or manifest["dataset"] != "xuzhou"
            or receipt["status"] != "passed_low_rank_cpu_cost" or receipt["commit"] != manifest["commit"]
            or receipt["manifest"] != spec["manifest"] or receipt["registration"] != spec["registration"]
            or receipt["audit"] != spec["audit"] or receipt["reduction_and_expansion_measured_separately"] is not True
            or any(receipt[key] != 0 for key in ("query", "gt", "optimizer_updates"))
            or any(manifest[key] != 0 for key in ("query", "gt", "optimizer_updates"))
            or len(manifest["tiles"]) != 5 or sum(t["pixels"] for t in manifest["tiles"]) != 3920):
        raise ValueError("Original registered low-rank production, evaluation and CPU evidence required")
    origin = read(reg["source"]["origin"])
    witness, original_stage = read(origin["registration"]), read(origin["off_full"])
    if (origin["source"] != reg["source"]["weights"] or origin["sidecar"] != reg["source"]["sidecar"]
            or origin["selected_step"] != reg["source"]["historical_selected_step"]
            or original_stage["new_updates"] != witness["budget"]
            or original_stage["finish"]["selected_state_hash"] != origin["selected_state"]):
        raise ValueError("Low-rank original source and WP08 witness differ")
    provenance_info = dict(source_commit=origin["producer_commit"], witness_commit=witness["commit"],
                           source_evidence=origin["registration"], source_steps=witness["budget"])
    cases = {c["case"]: c for c in reg["cases"]}
    seeds = [s for s in reg["recovery_seeds"] if s not in reg["rejected_seeds"]]
    if (len(cases) != len(reg["cases"]) or set(cases) != set(audit["report"]) or set(cases) != set(manifest["cases"])
            or list(cases) != manifest["case_order"]
            or {(c["seed"], c["arm"]) for c in cases.values()} != {(s, a) for s in seeds for a in reg["ladder"]["arms"]}
            or reg["maximum_updates"] != 2*reg["budget"]*len(cases)):
        raise ValueError("Low-rank case grid or registered total differs")
    classes = reg["validation"]["class_ids"]
    small = [int(k) for k, n in reg["validation"]["train_support"].items() if n <= reg["protection"]["small_support_max"]]
    if (classes != manifest["class_ids"] or sorted(small) != sorted(audit["small_classes"])
            or reg["protection"]["small_new_errors_max"] != 0):
        raise ValueError("Class or small-class protection differs")
    costs, group_summaries = measured_rows(manifest, receipt, read(receipt["resident"]))
    support = next(iter(reg["exposures"].values()))["support"]
    if any(plan["support"] != support for plan in reg["exposures"].values()):
        raise ValueError("Recovered low-rank seeds must share the original training support")
    family = dict(scene="xuzhou", data_protocol="wp10_xuzhou_k5_v5", model_seed=reg["source_model_seed"],
                  support_sha256=digest(support), train_labels=len(support["indices"]), validation_labels=5*len(classes),
                  class_ids=classes, split=reg["validation"]["identity"], validation_role="repeated_development_v5",
                  evaluation_sha256=digest(reg["validation"]))
    if family["train_labels"] != 45 or family["validation_labels"] != 45:
        raise ValueError("Low-rank production must preserve the WP10 45/45 contract")
    rows = [dict(module="WP10-LR", arm="original_fp32", case_id="original_fp32", identity=dict(family, sampler_seed=None),
                 **provenance(**provenance_info), state=origin["selected_state"], package=origin["source"]["sha256"],
                 precision="fp32", package_bytes=None, budget=0, actual_optimizer_calls=0,
                 selected_step=origin["selected_step"], selection="original_wp08_off",
                 quality=metrics(audit["source"]["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                 protection=None, cost=missing("NOT_RUN", "Original unadapted reference was replayed for quality"))]
    for name, case in cases.items():
        report = audit["report"][name]
        chosen = report["selected"]
        entry = manifest["cases"][name]
        package = chosen["package"]
        if (case["assignment"] != reg["ladder"]["arms"][case["arm"]] or entry["case"] != case or report["case"] != case
                or report["updates"] != 2*reg["budget"] or report["steps"] != reg["budget"]
                or entry["package"] != package or entry["selected_step"] != chosen["step"]
                or not 1 <= chosen["step"] <= reg["budget"]
                or [dict(zip(FULL_HEADS, block, strict=True)) for block in chosen["probabilities"]]
                != receipt["startup"][name][0]["output_hashes"]
                or any(hash_file(package[k]["path"]) != package[k]["sha256"] for k in ("weights", "sidecar"))):
            raise ValueError("Selected low-rank package, case or recovery cursor differs")
        dense = audit["report"][f"s{case['seed']}-dense"]
        guards = {point: verify_guard(dense[point], report[point], classes, small, reg["protection"]["limits"])
                  for point in ("initial", "selected", "last")}
        if guards != audit["guards"][name]:
            raise ValueError("Low-rank registered head or small-class guard differs")
        lineage = provenance(**provenance_info, recovery_commit=reg["commit"], recovery_steps=reg["budget"],
                             release_commit=reg["commit"], release_evidence=spec["audit"])
        rows.append(dict(module="WP10-LR", arm=case["arm"], case_id=name, identity=dict(family, sampler_seed=case["seed"]),
                         **lineage, state=package["state"], package=package["weights"]["sha256"],
                         precision=package["format"], package_bytes=package["bytes"], assignment=case["assignment"],
                         budget=reg["budget"],
                         actual_optimizer_calls=report["updates"], selected_step=chosen["step"], selection=reg["selection"],
                         quality=metrics(chosen["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"),
                         other_checkpoints={point: dict(step=report[point]["step"], quality=metrics(
                             report[point]["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix"))
                             for point in ("initial", "last")},
                         protection=guards["selected"], initial_protection=guards["initial"], last_protection=guards["last"],
                         cost=costs[name]))
    qualified = {str(seed): [r["arm"] for r in rows if r["identity"]["sampler_seed"] == seed and r["arm"].startswith("svd-")
                             and all(g["passes_metric_guards"] for g in r["protection"].values())] for seed in seeds}
    if qualified != audit["reduced_all_heads_per_seed"]:
        raise ValueError("Reduced-rank choices differ from their individual head guards")
    return rows, dict(producer=reg["commit"], total_actual_updates=reg["maximum_updates"], coverage=reg["coverage"],
                      cases=len(cases), recovery_seeds=seeds, rejected_seeds=reg["rejected_seeds"], budget=reg["budget"],
                      rank_mapping=reg["ladder"]["requested_mapping"], selected_reduced_per_seed=qualified,
                      default="full_fp32", cost_groups=group_summaries,
                      combination=missing("NOT_RUN", "This batch registered no real factor INT8 combination"),
                      independent=missing("NOT_RUN", "Fixed development V5 supplied the quality evidence"))

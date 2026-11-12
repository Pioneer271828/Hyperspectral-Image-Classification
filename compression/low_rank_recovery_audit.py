"""Independently reconcile all low-rank trajectories and authenticate deployment packages."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

from compression.low_rank_checkpoint import load_low_rank, save_low_rank
from compression.low_rank_recovery import HEADS, TARGET, LowRankRecoveryRun, factor_calls, factor_counts
from compression.precision_fallback import SMALL_CLASS_MAX_SUPPORT
from compression.precision_recovery_audit import _comparison, _guard, independent_predictions, save_package
from compression.registered_low_rank_recovery import initial_model, load_registration
from compression.registered_precision_recovery import stages
from research_runtime import require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


def package_model(model, case, initial, path):
    if case["assignment"]["rank"] is None:
        package = save_package(model, "fp32", path)
        package.update(state=state_hash(model), parameters=sum(p.numel() for p in model.parameters()))
        return package
    weight = dict(path=str(path), sha256=save_low_rank(model, path))
    metadata = dict(format="wp10_low_rank_deployment_v1", weights=weight, state=state_hash(model),
                    layer=TARGET, assignment=case["assignment"], seed=case["seed"], plan=model._low_rank_plan,
                    parameters=sum(p.numel() for p in model.parameters()), architecture=model.export_config(),
                    dense_source_state=initial["dense_state"], initial_factor_state=initial["state"],
                    shared_initial_tensors=initial["shared_tensors"])
    sidecar = dev._json_new(Path(str(path) + ".json"), metadata)
    return dict(format="low_rank_fp32", weights=weight, sidecar=sidecar, state=metadata["state"],
                parameters=metadata["parameters"], bytes=path.stat().st_size + Path(sidecar["path"]).stat().st_size)


def load_package(package, *, initial=None, case=None):
    if package["format"] == "fp32":
        model = dev._load_source(package["weights"], package["sidecar"])
    elif package["format"] == "low_rank_fp32":
        sidecar = admission.read(package["sidecar"])
        model = load_low_rank(package["weights"]["path"], package["weights"]["sha256"])
        if (sidecar["format"] != "wp10_low_rank_deployment_v1" or sidecar["weights"] != package["weights"]
                or sidecar["state"] != package["state"] or sidecar["architecture"] != model.export_config()
                or sidecar["plan"] != model._low_rank_plan or sidecar["layer"] != TARGET
                or sidecar["parameters"] != package["parameters"]
                or initial is None or case is None or sidecar["assignment"] != case["assignment"] or sidecar["seed"] != case["seed"]
                or sidecar["dense_source_state"] != initial["dense_state"] or sidecar["initial_factor_state"] != initial["state"]
                or sidecar["shared_initial_tensors"] != initial["shared_tensors"]):
            raise ValueError("Low-rank deployment sidecar, factors or original shared tensors differ")
    else:
        raise ValueError("Unsupported low-rank deployment format")
    if (sum(p.numel() for p in model.parameters()) != package["parameters"] or state_hash(model) != package["state"]
            or sum(Path(package[k]["path"]).stat().st_size for k in ("weights", "sidecar")) != package["bytes"]):
        raise ValueError("Loaded model state, parameters or byte count differs")
    return model.eval()


def _endpoint(model, scene, record):
    with factor_calls(model) as counts:
        result = independent_predictions(model, scene, record)
    if counts != factor_counts(model, result["forwards"]):
        raise ValueError("Both real factor operators must execute per block")
    result["factor_calls"] = counts
    return result


def _public(result, step, package=None):
    return dict(step=step, metrics=result["metrics"], probabilities=result["scores"],
                predictions=result["predictions"], targets=result["targets"],
                factor_calls=result["factor_calls"], forwards=result["forwards"], package=package)


def _agree(head, validated):
    return (head["confusion_matrix"] == validated["confusion"] and head["oa"] == validated["oa"]
            and abs(head["nll"] - validated["nll"]) <= 1e-7)


def audit(pin, completion, code):
    started = perf_counter()
    root, reg, dense, scene, record, plans = load_registration(pin, code)
    batch = admission.read(completion)
    if (completion["path"] != str(root / "batch-complete.json") or batch["registration"] != pin
            or batch["updates"] != reg["maximum_updates"] or batch["rejected_seeds"] != reg["rejected_seeds"]
            or any(batch[k] != 0 for k in ("gt", "query", "optimizer_updates"))):
        raise ValueError("Completed registration evidence differs")
    intervals = stages(reg["budget"], reg["prefix"])
    cases = {case["case"]: case["case"] for case in reg["cases"]}
    out = root / "deployment"
    out.mkdir()
    original = _endpoint(deepcopy(dense), scene, record)
    historical = reg["source"].get("historical_baseline")
    if historical is not None and (original["metrics"] != historical["metrics"] or original["scores"] != historical["scores"]):
        raise ValueError("Original WP08 source metrics or scores differ")
    reports, endpoints = {}, {}
    actual_updates, forwards = 0, original["forwards"]
    for case in reg["cases"]:
        name, seed = case["case"], str(case["seed"])
        statuses, receipts, packets = {}, {}, {}
        for stage, (_, stop, previous) in intervals.items():
            status, receipt = read_stage(root, pin, name, stage, receipts.get(previous), arms=cases, stages=intervals)
            if receipt != batch["attempts"][name][stage]:
                raise ValueError("Stage receipt differs from batch completion")
            statuses[stage], receipts[stage] = status, receipt
            packets[stage] = _packet(status, reg["contracts"][name], stop, packet_format=LowRankRecoveryRun.FORMAT)
            actual_updates += status["new_updates"]
        full = packets["full"]
        if (any(not _exact(full[key], packets[other][key]) for other in ("resume", "completed") for key in SCIENCE)
                or not _exact(full["history"][:reg["prefix"]], packets["prefix"]["history"])):
            raise ValueError("Matched continuous and resumed low-rank states differ")
        selected = _verify_history(full, plans[seed])
        initial = initial_model(root, reg, dense, case)
        endpoints[name] = {}
        path = out / name
        path.mkdir()
        activity = full["history"]
        report = dict(case=case, steps=reg["budget"], updates=sum(v["new_updates"] for v in statuses.values()),
                      identical_fields=list(SCIENCE), history=activity,
                      target_activity={p: {field: sum(row["target_activity"][p][field] > 0 for row in activity)
                                           for field in ("gradient_nonzero", "changed")} for p in activity[0]["target_activity"]},
                      stage_seconds={s: {k: v for k, v in row.items() if k.endswith("_seconds")}
                                     for s, row in statuses.items()})
        for key, step, state in (("initial", 0, initial.state_dict()), ("selected", selected, full["selected_state"]),
                                 ("last", reg["budget"], full["state"])):
            model = deepcopy(initial).eval()
            model.load_state_dict(state, strict=True)
            result = _endpoint(model, scene, record)
            forwards += result["forwards"]
            package = None
            if step:
                if (state_hash(model) != activity[step-1]["state_hash"]
                        or not all(_agree(result["metrics"][h], activity[step-1]["deployment"]["heads"][h]) for h in HEADS)):
                    raise ValueError("Frozen endpoint differs from recorded V5 event")
                package = package_model(model, case, reg["initial"][name]["evidence"], path / f"{key}.pt")
                restored = load_package(package, initial=reg["initial"][name]["evidence"], case=case)
                replay = _endpoint(restored, scene, record)
                forwards += replay["forwards"]
                if replay["scores"] != result["scores"] or replay["metrics"] != result["metrics"]:
                    raise ValueError("Published endpoint differs from live FP32 factors")
            report[key] = _public(result, step, package)
            endpoints[name][key] = result
        if case["arm"] == "full-rank":
            report["initial_full_rank_probability_max_abs"] = _comparison(original, endpoints[name]["initial"])
        reports[name] = report
        print(f"{name}: three frozen endpoints, actual factor operators and complete history verified", flush=True)
    if actual_updates != reg["maximum_updates"]:
        raise ValueError("Independent executed budget differs")
    small = [int(k) for k, n in reg["validation"]["train_support"].items() if n <= SMALL_CLASS_MAX_SUPPORT]
    decisions = {}
    for case in reg["cases"]:
        name, seed = case["case"], case["seed"]
        reference = endpoints[f"s{seed}-dense"]
        decisions[name] = {point: _guard(reference[point], result, record["classes"], small)
                           for point, result in endpoints[name].items()}
    qualified = {str(seed): [case["arm"] for case in reg["cases"] if case["seed"] == seed
                 and case["arm"].startswith("svd-") and
                 all(row["passes_metric_guards"] for row in decisions[case["case"]]["selected"].values())]
                 for seed in reg["recovery_seeds"] if seed not in reg["rejected_seeds"]}
    admission.check_pins([pin, completion, *reg["protected"].values(), *[r["checkpoint"] for r in reg["initial"].values()]])
    require_frozen_code(code)
    return dev._json_new(root / "audit.json", dict(status="passed_low_rank_recovery_independent",
        registration=pin, completion=completion, scope=reg["scope"], source=_public(original, 0),
        report=reports, guards=decisions, reduced_all_heads_per_seed=qualified, small_classes=small,
        actual_updates=actual_updates, independent_forwards=forwards, independent_updates=0,
        gt=0, query=0, elapsed_seconds=perf_counter()-started))

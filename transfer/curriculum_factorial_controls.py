"""Authenticate nine existing controls without taking a training step."""

import math
from pathlib import Path
import subprocess
from time import perf_counter

from compression.precision_recovery_audit import independent_predictions
from compression.registered_precision_recovery import COMPATIBILITY_PATHS, stages
from model.research_config import digest
from research_runtime import ROOT, external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_factorial_real import BUDGET, CASES, ORDER, PREFIX, REFERENCED_UPDATES, inputs
from transfer.curriculum_observation_real import MeasuredObservationRun
from transfer.curriculum_real import RegisteredSampleRun, verify_notice
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import LR, SEED, read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


FORMAT = "wp01_factorial_authenticated_controls_v1"
CONTROLS = {"none": ("sample", "off"), "s_only": ("sample", ORDER), "o_only": ("observation", ORDER)}
CORE = (*COMPATIBILITY_PATHS, "transfer/curriculum_real.py", "transfer/curriculum_real_audit.py",
        "transfer/observation_curriculum.py", "compression/precision_recovery_audit.py",
        "model/task_losses.py", "evaluation/classification_metrics.py", "research_runtime.py")
OBSERVATION = ("transfer/curriculum_observation_real.py", "transfer/curriculum_observation_admission.py",
               "transfer/curriculum_observation_audit.py")
EXTENSIONS = {
    "sample": {"trial", "teacher_hash", "applied_weight_history"},
    "observation": {"observation_trial", "observation_boundary"},
}


def unchanged(producer, code, paths):
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", producer, code["commit"]], check=True)
    changed = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", producer, code["commit"],
                                       "--", *paths], text=True).splitlines()
    if changed:
        raise ValueError("Historical numeric dependency changed: " + ", ".join(changed))
    return dict(producer=producer, candidate=code["commit"], unchanged_paths=list(paths))


def _legacy(req, reg, kind, code):
    group = req[kind]
    root = Path(reg["root"])
    audit, complete = (admission.read(group[key]) for key in ("audit", "completion"))
    allowed = {"sample": ("passed_independent_curriculum", 6408, "original_gt_reads"),
               "observation": ("passed_independent_real_observation", 4272, "gt")}
    status, count, raw = allowed[kind]
    if (group["audit"]["path"] != str(root/"audit.json")
            or group["completion"]["path"] != str(root/"batch-complete.json")
            or audit["status"] != status or audit["registration"] != group["registration"]
            or audit["completion"] != group["completion"] or audit["actual_updates"] != count
            or complete["registration"] != group["registration"] or complete["updates"] != count
            or any(audit[k] for k in (raw, "query", "optimizer_updates"))
            or any(complete[k] for k in ("gt", "query", "optimizer_updates"))):
        raise ValueError("Wrong accepted historical producer completion or audit")
    verify_notice(root, reg, group["registration"], complete["notice"])
    proof = unchanged(reg["commit"], code, CORE + (OBSERVATION if kind == "observation" else ()))
    paths = [*group.values(), complete["notice"]]
    if kind == "observation":
        if audit["frozen_selected"] != group["frozen_selected"]:
            raise ValueError("Wrong original observation endpoint freeze")
        selected = admission.read(group["frozen_selected"])
        if (selected["registration"] != group["registration"] or selected["completion"] != group["completion"]
                or selected["optimizer_updates"] != 0):
            raise ValueError("Original observation endpoint boundary changed")
    return audit, complete, paths, proof


def _same_arithmetic(spec, common, *, kind, sample, observation):
    # Explicitly account for runtime wrappers. All other scientific contract fields must match.
    observed = {k:v for k,v in spec.items() if k not in EXTENSIONS[kind]}
    core = {k:v for k,v in common.items() if k not in EXTENSIONS["sample"] | {"factorial_scope", "factor_switches"}}
    core["policy_hash"] = sample
    if observation is None:
        core.pop("observation_policy_hash")
    else:
        core["observation_policy_hash"] = observation
    if observed != core:
        changed = [k for k in sorted(observed.keys() | core.keys()) if observed.get(k) != core.get(k)]
        raise ValueError("Control arithmetic or identity differs: " + ", ".join(changed))
    return digest(observed)


def _point(item, point, step, packet_state, history, scene, record):
    value = item[point]
    package = value["package"]
    restored = dev._load_source(package["weights"], package["sidecar"])
    if (value["step"] != step or package["state"] != history[step-1]["state_hash"]
            or state_hash(restored) != package["state"] or not _exact(restored.state_dict(), packet_state)):
        raise ValueError("Control endpoint package differs from the authenticated trajectory")
    view = value.get("conditions", {}).get("full", value)
    fresh = independent_predictions(restored, scene, record)
    if any(fresh[k] != view[k] for k in ("metrics", "scores", "predictions", "targets", "forwards")):
        raise ValueError("Reused control full condition failed exact replay")
    return dict(step=step, package=package, full_sha256=digest({k:fresh[k] for k in ("metrics", "scores", "predictions", "targets", "forwards")}))


def _control(reg, family, source, scene, record, plan, joint, kind, old_case, s_arm, o_arm, witness):
    audit, complete, protected, compatibility = family
    root = Path(reg["root"])
    intervals = stages(BUDGET, PREFIX)
    group = witness[kind]
    receipts, packets, restore_cost = {}, {}, []
    klass = RegisteredSampleRun if kind == "sample" else MeasuredObservationRun
    original = next(c for c in reg["cases"] if c["case"] == old_case)
    policy, obs, teacher = joint[1:]
    trial = (original, policy if s_arm else None, teacher) if kind == "sample" else (original, obs)
    math_hash = None
    for stage, (_, stop, prior) in intervals.items():
        status, pin = read_stage(root, group["registration"], old_case, stage, receipts.get(prior),
                                 arms=reg["contracts"], stages=intervals)
        if pin != complete["attempts"][old_case][stage]:
            raise ValueError("Historical stage receipt differs from the original completion")
        packet = _packet(status, reg["contracts"][old_case], stop, packet_format=klass.FORMAT)
        math_hash = _same_arithmetic(packet["spec"], joint[0], kind=kind,
                                     sample=policy.policy_hash if s_arm else None,
                                     observation=obs.policy_hash if o_arm else None)
        checkpoint = status["checkpoint"]
        start = perf_counter()
        restored = klass.load(source, scene, record, checkpoint["path"], plan=plan, trial=trial,
            selector_seed=SEED, expected_sha256=checkpoint["sha256"], expected_contract_hash=reg["contracts"][old_case])
        timing = restored.timings["full_v5" if kind == "sample" else "full_v5_event"]
        if (restored.completed != stop or len(timing) != 2 or restored.contract_hash != packet["contract_hash"]
                or not _exact(restored.history, packet["history"])
                or not _exact(restored.optimizer.state_dict(), packet["optimizer"])
                or not _exact(restored.torch_rng, packet["torch_rng"])
                or not _exact(restored.rng.bit_generator.state, packet["rng"])):
            raise ValueError("Historical authenticated zero-update recovery differs")
        restore_cost.append(dict(stage=stage, v5_events=len(timing), seconds=perf_counter()-start))
        receipts[stage], packets[stage] = pin, packet
        protected.extend([pin, checkpoint, status["claim"]])
    full = packets["full"]
    item = audit["cases"][old_case]
    if (item["case"] != original or item["updates"] != 2*BUDGET or digest(item["history"]) != digest(full["history"])
            or not _exact(full["history"][:PREFIX], packets["prefix"]["history"])
            or not all(_exact(full[key], packets[stage][key]) for stage in ("resume", "completed") for key in SCIENCE)):
        raise ValueError("Reused continuous, prefix, resume, completed, or audited history differs")
    selection = _verify_history(full, plan)
    points = {p:_point(item, p, step, full[key], full["history"], scene, record)
              for p, step, key in (("selected", selection, "selected_state"), ("last", BUDGET, "state"))}
    for value in points.values():
        protected.extend(value["package"][k] for k in ("weights", "sidecar"))
    return dict(producer=reg["commit"], original_case=old_case, registration=group["registration"],
        audit=group["audit"], completion=group["completion"], attempts=receipts, endpoints=points,
        arithmetic_sha256=math_hash, history_sha256=digest(full["history"]), historical_updates=2*BUDGET,
        restoration=restore_cost, full_endpoint_replays=2)


def authenticate(request, out, code):
    from transfer.curriculum_factorial_real import new_run

    started = perf_counter()
    req, s, o, model, scene, record, plans, trials = inputs(request, code)
    root = external_directory(out)
    if root.exists() or any(root == Path(pin["path"]).parent or root in Path(pin["path"]).parents
                           for value in req.values() for pin in value.values()):
        raise ValueError("Control authentication needs a fresh separate output")
    root.mkdir()
    families = {}
    try:
        families = {kind:_legacy(req, reg, kind, code) for kind, reg in (("sample", s), ("observation", o))}
        synthetic = {}
        for count, pin in req["synthetic"].items():
            a = admission.read(pin)
            if (a["status"] != "passed_synthetic_factorial" or a["fixture_classes"] != int(count)
                    or a["scope"] != "synthetic" or a["optimizer_updates"] != 0):
                raise ValueError("Missing accepted synthetic arithmetic compatibility")
            synthetic[count] = dict(audit=pin, producer=a["producer"], synthetic_updates=a["synthetic_updates"])
        cases = {}
        for c in CASES:
            name, seed, arm = c["case"], c["seed"], c["arm"]
            if arm not in CONTROLS:
                continue
            kind, old_arm = CONTROLS[arm]
            old_case = f"s{seed}-{old_arm}"
            plan = plans[str(seed)]
            trial = trials[f"s{seed}-s_plus_o"]
            core = new_run(model, scene, record, plan, trial).spec
            cases[name] = _control(s if kind == "sample" else o, families[kind], model, scene, record, plan,
                                   (core, *trial[1:]), kind, old_case, arm == "s_only", arm == "o_only", req)
            print(f"{name}: exact original recovery, arithmetic and full endpoints", flush=True)
        if sum(c["historical_updates"] for c in cases.values()) != REFERENCED_UPDATES:
            raise ValueError("Historical control budget differs")
        all_pins = [request, *s["protected"].values(), *o["protected"], s["teacher"], *req["synthetic"].values(),
                    *(p for family in families.values() for p in family[2])]
        protected = list({p["path"]:p for p in all_pins}.values())
        admission.check_pins(protected)
        require_frozen_code(code)
        return dev._json_new(root/"controls.json", dict(format=FORMAT, status="passed", request=request,
            commit=code["commit"], source_hash=code["source_hash"], historical_updates=REFERENCED_UPDATES,
            cases=cases, protected=protected, dependencies={k:v[3] for k,v in families.items()},
            synthetic_arithmetic=synthetic, source=s["source"], state=state_hash(model),
            optimizer_updates=0, query=0, gt=0, seconds=perf_counter()-started))
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, RuntimeError) as error:
        dev._json_new(root/"blocked.json", dict(status="BLOCKED", request=request, reason=str(error),
                                               optimizer_updates=0, query=0, gt=0, seconds=perf_counter()-started))
        raise

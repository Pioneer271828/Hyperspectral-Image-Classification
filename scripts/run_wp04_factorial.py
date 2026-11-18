"""Register and verify the fixed-Q/A WP04 scratch factorial through the bounded target runtime."""
import argparse
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import csv
import ctypes
import json
from pathlib import Path
import os
import resource
import subprocess
import sys
from time import perf_counter, sleep
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from g4_position_fusion_target_runtime import prepare
from g4_support_context import prepare_contexts, read, validate_manifest
from joint_training import _tree_fingerprint, run_training
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash, write_json
from target_initialization import METHODS
from target_training import observation
from g4_position_fusion_target import calibrate
from target_initialization import state_id
from transfer import p4_runtime_device as p4
from transfer.shared_gpu_lease import physical_lease


SEEDS = (64, 512, 2048)
ARMS = {
    "B0": ("none", "fixed"),
    "R": ("x_residual", "fixed"),
    "F": ("none", "position_gate"),
    "R+F": ("x_residual", "position_gate"),
    "Bcap": ("capacity_control", "fixed"),
    "Bcap+F": ("capacity_control", "position_gate"),
}
HEADS = {"fused": "heads", "graph": "hgnn_heads", "cnn": "cnn_heads"}
EXTRA = ("backbone.region_residual.", "backbone.position_fusion.")
STEPS = {"whole": 12, "prefix": 2, "resume": 10, "completed": 0}
CONTRASTS = {
    "R-B0": {"R": 1, "B0": -1},
    "F-B0": {"F": 1, "B0": -1},
    "R+F-B0": {"R+F": 1, "B0": -1},
    "R+F-R-F+B0": {"R+F": 1, "R": -1, "F": -1, "B0": 1},
    "R-Bcap": {"R": 1, "Bcap": -1},
    "R+F-(Bcap+F)": {"R+F": 1, "Bcap+F": -1},
    "Bcap+F-Bcap-F+B0": {"Bcap+F": 1, "Bcap": -1, "F": -1, "B0": 1},
}
SCHEMA = "wp04_fixed_graph_scratch_factorial_v1"
REAL_CATALOG_SHA = "a2c23f9b60ccf094008e6ac4c7bcd4aa289dece6e8b3993e91431ee0a75c502e"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(code):
    return {k: code[k] for k in ("commit", "source_hash")}


def pinned(path):
    return {"path": str(path), "sha256": hash_file(path)}


def verify(pin):
    path = Path(pin["path"])
    require(path.resolve(strict=True) == path and hash_file(path) == pin["sha256"], "Pinned input changed")
    return path


def roster():
    return [{"seed": seed, "arm": arm, "mode": mode, "fusion": fusion}
            for seed in SEEDS for arm, (mode, fusion) in ARMS.items()]


def plan(catalog, code, scope, *, arithmetic=None):
    require(arithmetic in (None, p4.CUDA_FP64_WEIGHT), "Unknown WP04 arithmetic")
    value = {"schema": SCHEMA, "scope": scope, "code": identity(code), "catalog": pinned(catalog),
            "target": "xuzhou", "support_seed": 512, "per_class": 1, "cases": roster(),
            "phase_updates": STEPS, "method": METHODS[0], "graph": "input_Q_and_A_fixed",
            "selection": "first_min_fused_validation_nll", "real_updates_authorized": 0,
            "query_access_authorized": False,
            "full_source_ft": "separate_nonfactorial_scope_different_pretrained_source_by_mode"}
    if arithmetic is not None:
        value["arithmetic"] = arithmetic
    return value


def registration(args, code):
    path = verify({"path": str(Path(args.registration[0]).resolve()), "sha256": args.registration[1]})
    value = read(path)
    catalog = verify(value["catalog"])
    require(value == plan(catalog, code, "synthetic_only", arithmetic=value.get("arithmetic"))
            and catalog == path.parent / "contexts" / "manifest.json",
            "Only the generated synthetic fixture registration can run optimization")
    return value


def bound(reg, case, out, device):
    return prepare(reg["catalog"]["path"], reg["catalog"]["sha256"], out,
                   target=reg["target"], support_seed=reg["support_seed"], per_class=reg["per_class"],
                   method=reg["method"], model_seed=case["seed"], mode=case["mode"],
                   condition=case["fusion"], device=device)


def attest(gpu):
    lib = ctypes.CDLL("libcuda.so.1")
    ordinal, value = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    require(lib.cuInit(0) == 0 and lib.cuDeviceGet(ctypes.byref(ordinal), 0) == 0
            and lib.cuDeviceGetUuid(ctypes.byref(value), ordinal) == 0
            and "GPU-" + str(UUID(bytes=bytes(value))) == gpu, "Actual CUDA device differs")
    props = torch.cuda.get_device_properties(0)
    return {"uuid": gpu, "name": props.name, "memory_total_bytes": props.total_memory, "cuda": torch.version.cuda}


def hardware(gpu, started):
    if gpu:
        torch.cuda.synchronize()
        return {"wall_seconds": perf_counter()-started, "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
    return {"wall_seconds": perf_counter()-started, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def model_snapshot(model):
    tensors = {k: tensor_hash(v) for k, v in model.state_dict().items()}
    return {"shared": digest({k: v for k, v in tensors.items() if not k.startswith(EXTRA)}),
            "residual": {k: v for k, v in tensors.items() if k.startswith(EXTRA[0])},
            "fusion": {k: v for k, v in tensors.items() if k.startswith(EXTRA[1])},
            "parameters": sum(p.numel() for p in model.parameters()),
            "residual_parameters": sum(p.numel() for k, p in model.named_parameters() if k.startswith(EXTRA[0])),
            "fusion_parameters": sum(p.numel() for k, p in model.named_parameters() if k.startswith(EXTRA[1]))}


def move_items(items, device):
    return {name: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            for name, item in items.items()}


def validation_evidence(probabilities, item, recorded):
    data = {}
    truth = item["val_labels"].cpu().numpy()
    classes = item["class_num"]
    for head in HEADS:
        scores = probabilities[head].cpu().numpy()
        matrix = np.bincount(truth * classes + scores.argmax(1), minlength=classes**2).reshape(classes, classes)
        stats = metrics_from_confusion(matrix, range(1, classes+1))
        trace = recorded[head]
        expected_nll = -np.log(np.maximum(scores[np.arange(len(truth)), truth], 1e-8)).mean(dtype=np.float64)
        require(stats["confusion_matrix"] == trace["confusion"]
                and abs(stats["oa"]-trace["oa"]) < 1e-12 and abs(stats["aa"]-trace["aa"]) < 1e-12
                and abs(float(expected_nll)-trace["nll"]) < 1e-6
                and [c["support"] for c in stats["per_class"]] == [5] * classes,
                "Independent validation confusion, count, or NLL differs")
        data[head] = {"oa": stats["oa"], "aa": stats["aa"], "nll": trace["nll"],
                      "classes": [{**c, "correct": int(matrix[i, i])} for i, c in enumerate(stats["per_class"])],
                      "probability_sha256": tensor_hash(probabilities[head])}
    return data


def terminal(path):
    progress = read(path / "progress.json")
    terminal_path = path / "state-000012.pt"
    require(progress["cursor"] == 12 and progress["state_file"] == terminal_path.name
            and progress["state_sha256"] == hash_file(terminal_path), "Terminal checkpoint differs")
    return torch.load(terminal_path, map_location="cpu", weights_only=True)


def audit_case(reg, case, out, gpu, *, real=False):
    device = "cuda:0" if gpu else "cpu"
    split = out / "split"
    opts, cpu_items, _, _, runtime, guard = bound(reg, case, split, device)
    target = reg["target"]
    item = cpu_items[target]
    inputs = {k: tensor_hash(item[k]) for k in ("net_input", "Q", "A", "train_indices", "train_labels", "val_indices", "val_labels")}
    model = runtime.build_model(ResearchConfig.from_dict(opts["config"]), {target: item["class_num"]}).to(device)
    init = model_snapshot(model)
    require("test_labels" not in item and opts["optimizer_steps"] == 12, "Foreign input role or budget")
    items = move_items(cpu_items, device)

    def calibrated(model):
        return calibrate(model, observation(model, items[target]), expected_binding=runtime.position_binding,
                         source_state_sha256=state_id(model))[0]

    if real:
        before = state_id(model)
        checked, probs = runtime.validation_snapshot(calibrated(model), items)
        require(state_id(model) == before, "Zero-update audit changed initial model")
        guard()
        return {"case": case, "inputs": inputs, "init": init, "optimizer_updates": 0,
                "step_zero": validation_evidence(probs[target]["val"], item, checked["metrics"][target]["val"])}

    first, last = terminal(out / "whole"), terminal(split)
    fields = ("model", "optimizer", "scheduler", "history", "trace", "rng", "best_model", "best_probabilities")
    if reg.get("arithmetic"):
        one_a, one_b = (torch.load(out / route / "state-000001.pt", map_location="cpu", weights_only=True)
                        for route in ("whole", "split"))
        require(all(_tree_fingerprint(one_a[k]) == _tree_fingerprint(one_b[k]) for k in fields),
                "Independent first-step training states differed")
    require(all(_tree_fingerprint(first[k]) == _tree_fingerprint(last[k]) for k in fields),
            "Whole and split training states differed")
    minimum = min(last["history"], key=lambda row: row["val_loss"])
    selected_step = minimum["step"]
    require(selected_step == last["best_step"] == read(split / "result.json")["selected_step"],
            "First-min selection changed")
    changes = {}
    for head, prefix in HEADS.items():
        names = [k for k in last["model"] if k.startswith(prefix + ".")]
        norms = [row["gradient_norms"][head + "_head"] for row in last["trace"]]
        changed = sum(int(torch.count_nonzero(last["model"][k] != model.state_dict()[k].detach().cpu())) for k in names)
        require(len(norms) == 12 and all(np.isfinite(norms)) and max(norms) > 0 and changed > 0,
                "One of the three heads did not receive gradient and update")
        changes[head] = {"nonzero_gradient_steps": sum(v > 0 for v in norms), "changed_elements": changed}
    raw = split / "raw-validation-000012.pt"
    require(_tree_fingerprint(last["model"]) == _tree_fingerprint(torch.load(raw, weights_only=True, map_location="cpu")),
            "Final calibration used the wrong raw state")
    model.load_state_dict(last["model"], strict=True)
    _, probs = runtime.validation_snapshot(calibrated(model), items)
    final = validation_evidence(probs[target]["val"], item, last["history"][-1]["metrics"][target]["val"])
    primary = runtime.load_selected(split / "primary.pt", items, split / f"raw-validation-{selected_step:06d}.pt", device=device)
    _, probs = runtime.validation_snapshot(primary, items)
    selected = validation_evidence(probs[target]["val"], item, minimum["metrics"][target]["val"])
    guard()
    costs = {phase: read(out / f"{phase}.json") for phase in STEPS}
    require([costs[phase]["updates"] for phase in STEPS] == list(STEPS.values()), "Wrong phase updates")
    return {"case": case, "inputs": inputs, "init": init, "steps": {"selected": selected_step, "last": 12},
            "endpoints": {"selected": selected, "last": final}, "head_updates": changes, "phases": costs,
            "exact_whole_split": True, "resume_completed_updates": 0, "audit_updates": 0,
            "trace_proof": digest(last["trace"]),
            **({"exact_independent_step_one": True} if reg.get("arithmetic") else {})}


def paired_initialization(rows):
    for seed in SEEDS:
        cases = [row for row in rows if row["case"]["seed"] == seed]
        require({row["case"]["arm"] for row in cases} == set(ARMS)
                and len({row["init"]["shared"] for row in cases}) == 1
                and len({digest(row["inputs"]) for row in cases}) == 1, "Paired input or shared initialization differs")
        by_arm = {r["case"]["arm"]: r["init"] for r in cases}
        for base, treatment in (("B0", "F"), ("R", "R+F"), ("Bcap", "Bcap+F")):
            a, b = (by_arm[k] for k in (base, treatment))
            require(a["residual"] == b["residual"] and b["parameters"]-a["parameters"] == 129
                    and b["fusion"] == by_arm["F"]["fusion"], "Gate/state or extra parameter pairing differs")
        require(by_arm["R"]["parameters"] - by_arm["B0"]["parameters"] == 8256
                and by_arm["Bcap"]["parameters"] - by_arm["B0"]["parameters"] == 8320, "Capacity control differs")
    require(len({digest(row["inputs"]) for row in rows}) == 1, "Comparison changed support across seeds")


def make_tables(rows):
    endpoints, per_class, paired, paired_class, costs = [], [], [], [], []
    selected = {}
    for row in rows:
        case = row["case"]
        for endpoint, heads in row["endpoints"].items():
            for name, score in heads.items():
                context = dict(seed=case["seed"], arm=case["arm"], endpoint=endpoint, head=name)
                selected[case["seed"], endpoint, name, case["arm"]] = score
                endpoints.append({**context, "step": row["steps"][endpoint], "oa": score["oa"], "aa": score["aa"], "nll": score["nll"],
                                  "parameters": row["init"]["parameters"]})
                per_class.extend({**context, **{k: label[k] for k in ("class_id", "correct", "support", "recall")}}
                                 for label in score["classes"])
        for phase, cost in row["phases"].items():
            costs.append(dict(seed=case["seed"], arm=case["arm"], phase=phase, updates=cost["updates"],
                              wall_seconds=cost["wall_seconds"], peak_allocated_bytes=cost.get("max_allocated_bytes"),
                              peak_reserved_bytes=cost.get("max_reserved_bytes"), peak_rss_bytes=cost.get("max_rss_bytes"),
                              hardware=cost["gpu"]["uuid"] if cost["gpu"] else "cpu"))
    for seed in SEEDS:
        for endpoint in ("selected", "last"):
            for head in HEADS:
                for comparison, design in CONTRASTS.items():
                    context = dict(seed=seed, endpoint=endpoint, head=head, comparison=comparison)
                    arms = [(selected[seed, endpoint, head, arm], v) for arm, v in design.items()]
                    deltas = []
                    for group in zip(*(a["classes"] for a, _ in arms), strict=True):
                        require(len({(c["class_id"], c["support"]) for c in group}) == 1, "Different class denominators")
                        delta = sum(c["recall"]*sign for (_, sign), c in zip(arms, group, strict=True))
                        deltas.append(delta)
                        paired_class.append(dict(**context, class_id=group[0]["class_id"], denominator=group[0]["support"],
                            correct_by_arm=json.dumps({arm: c["correct"] for arm, c in zip(design, group, strict=True)}),
                            delta_recall=delta))
                    paired.append({**context, **{f"delta_{k}": sum(a[k]*w for a, w in arms) for k in ("oa", "aa", "nll")},
                                   "minimum_class_recall_difference": min(deltas), "negative_class_differences": sum(d < -1e-12 for d in deltas)})
    return {"endpoints": endpoints, "per-class": per_class, "paired": paired, "paired-per-class": paired_class, "costs": costs}


def stage(args, code, reg, out):
    case = next(c for c in reg["cases"] if (c["seed"], c["arm"]) == (args.seed, args.arm))
    gpu, phase = args.gpu, args.phase
    p4.check_inherited_lease(args.matrix_fd, out.parent / ".matrix.lock")
    require(out == out.parent / f"{case['seed']}-{case['arm']}" and
            not (out / f"{phase}.json").exists(), "Wrong or already committed operation")
    numeric = reg.get("arithmetic")
    require(numeric is None or gpu is not None, "Strict arithmetic requires a registered physical GPU")
    calls = Counter()
    compute = p4.arithmetic(numeric, gpu, kernel_calls=calls) if numeric else nullcontext(None)
    lease = physical_lease(gpu, inherited_fd=args.lease_fd) if gpu else nullcontext()
    with lease, compute as details:
        inventory = (p4.snapshot(gpu) if numeric else p4.gate(gpu)) if gpu else None
        device = "cuda:0" if gpu else "cpu"
        card = attest(gpu) if gpu else None
        if phase == "audit":
            audit = audit_case(reg, case, out, gpu)
            if numeric:
                audit["arithmetic"] = details
                audit["kernel_calls"] = dict(calls)
            write_json(out / "audit.json", audit)
            return
        if gpu:
            torch.cuda.reset_peak_memory_stats(0)
        started = perf_counter()
        target = out / ("whole" if phase == "whole" else "split")
        opts, items, provenance, _, runtime, guard = bound(reg, case, target, device)
        if numeric:
            provenance["registered_wp04_cuda_arithmetic"] = details
        with redirect_stdout(sys.stderr):
            result = run_training(opts, items, target, provenance, code, runtime_adapter=runtime, finish_guard=guard,
                                  device=device, resume=phase in ("resume", "completed"),
                                  stop_after=2 if phase == "prefix" else None)
        require(result["updates_this_call"] == STEPS[phase], "Different optimizer cursor")
        cost = hardware(gpu, started)
        if numeric:
            cost["arithmetic"] = details
            cost["kernel_calls"] = dict(calls)
            require(phase == "completed" or calls["sparse_mm"] > 0,
                    "Strict path did not intercept the sparse training operator")
        write_json(out / f"{phase}.json", {**cost, "updates": STEPS[phase], "gpu": card, "resources": inventory,
                   "torch": str(torch.__version__), "scope": "phase_wall_including_prepare_checkpoint_and_verification"})


def run(args, code, reg, out):
    out.mkdir(exist_ok=True)
    gpu = args.gpu
    require(reg.get("arithmetic") is None or gpu, "Strict arithmetic requires a registered physical GPU")
    chosen = [c for c in reg["cases"] if args.arm is None or (c["seed"], c["arm"]) == (args.seed, args.arm)]
    require(chosen, "No selected registered case")
    lock = physical_lease(gpu) if gpu else nullcontext()
    with p4.lease(out / ".matrix.lock") as matrix_lock, lock as lease:
        owner = {"registration": args.registration, "gpu": gpu}
        path = out / "owner.json"
        if path.exists():
            require(read(path) == owner, "Output belongs to another matrix or device")
        else:
            write_json(path, owner)
        for case in chosen:
            directory = out / f"{case['seed']}-{case['arm']}"
            directory.mkdir(exist_ok=True)
            for phase in (*STEPS, "audit"):
                if (directory / f"{phase}.json").exists():
                    continue
                cmd = [sys.executable, "-B", __file__, "--expected-commit", code["commit"], "stage",
                       "--registration", *args.registration, "--out", str(directory), "--seed", str(case["seed"]),
                       "--arm", case["arm"], "--phase", phase, "--matrix-fd", str(matrix_lock.fileno())]
                fd = (matrix_lock.fileno(),)
                if gpu:
                    fd += (lease.fileno(),)
                    cmd += ["--gpu", gpu, "--lease-fd", str(lease.fileno())]
                started = perf_counter()
                with (directory / f"{phase}.log").open("ab") as log:
                    while True:
                        result = subprocess.run(cmd, check=False, pass_fds=fd, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        if result.returncode != 75 or perf_counter()-started >= args.resource_wait_seconds:
                            result.check_returncode()
                        sleep(5)
        if args.arm is not None:
            return
        rows = [read(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]]
        paired_initialization(rows)
        products = make_tables(rows)
        if (out / "evidence.json").exists():
            previous = read(out / "evidence.json")
            require(previous["case_pins"] == [pinned(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]],
                    "Existing audited rows differ")
            for pin in previous["tables"]:
                verify(pin)
            print(json.dumps({"status": "reused", "evidence": pinned(out / "evidence.json")}))
            return
        for name, data in products.items():
            with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        write_json(out / "evidence.json", {"schema": SCHEMA, "scope": reg["scope"], "device": gpu or "cpu",
                   "code": identity(code), "registration": dict(zip(("path", "sha256"), args.registration, strict=True)),
                   "tables": [pinned(out / f"{name}.csv") for name in products], "cases": len(rows),
                   "case_pins": [pinned(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]],
                   "counts": {name: len(data) for name, data in products.items()}, "synthetic_updates": 432,
                   "reporting_scope": "engineering_fixture_only", "real_updates": 0, "query_access": False,
                   **({"arithmetic": reg["arithmetic"], "strict_qualification": True} if reg.get("arithmetic") else {})})
        print(json.dumps({"status": "passed", "evidence": pinned(out / "evidence.json")}))


def probe(args, code, out):
    published = verify({"path": str(Path(args.manifest[0]).resolve()), "sha256": args.manifest[1]})
    require(args.manifest[1] == REAL_CATALOG_SHA, "Expected the independently admitted Xuzhou context")
    out.mkdir(exist_ok=False)
    original = read(published)
    validate_manifest(original, published.parent)
    checksum = original["input_spec_sha256"]
    sources = [path for path, sha in original["protected_files"].items() if sha == checksum]
    require(len(sources) == 1, "Missing original support-context input registration")
    rebound = out / "rebound-contexts"
    updated = prepare_contexts(sources[0], checksum, rebound, code)
    require(updated["observations"] == original["observations"]
            and updated["protected_files"] == original["protected_files"]
            and [{k: v for k, v in row.items() if k != "sha256"} for row in updated["contexts"]]
            == [{k: v for k, v in row.items() if k != "sha256"} for row in original["contexts"]],
            "Rebound support roles, graph or observations changed")
    static_proof = pinned(published)
    catalog = rebound / "manifest.json"
    arithmetic = p4.CUDA_FP64_WEIGHT if args.strict_cuda else None
    require(arithmetic is None or args.gpu, "Strict real probe requires a registered physical GPU")
    reg = plan(catalog, code, "real_zero_update_draft", arithmetic=arithmetic)
    rows = []
    leases = physical_lease(args.gpu) if args.gpu else nullcontext()
    calls = Counter()
    compute = p4.arithmetic(arithmetic, args.gpu, kernel_calls=calls) if arithmetic else nullcontext(None)
    with leases, compute as numeric:
        if args.gpu:
            if not arithmetic:
                p4.gate(args.gpu)
            attest(args.gpu)
        for case in reg["cases"]:
            start = perf_counter()
            row = audit_case(reg, case, out / f"{case['seed']}-{case['arm']}", args.gpu, real=True)
            rows.append({**row, **hardware(args.gpu, start)})
    paired_initialization(rows)
    write_json(out / "registration-draft.json", {**reg, "candidate_updates_after_separate_release": 216,
               "phase_updates": {key: STEPS[key] for key in ("prefix", "resume", "completed")},
               "published_catalog": static_proof,
               "execution_enabled": False, "approval": None})
    write_json(out / "evidence.json", {"schema": SCHEMA, "scope": reg["scope"], "code": identity(code),
               "cases": rows, "registration_draft": pinned(out / "registration-draft.json"),
               "published_catalog": static_proof, "rebound_contexts_verified": 27,
               "real_updates": 0, "query_access": False, "fixed_graph": True, "device": args.gpu or "cpu",
               **({"arithmetic": numeric, "kernel_calls": dict(calls)} if arithmetic else {})})
    verify(static_proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    for kind in ("prepare", "run", "stage", "probe"):
        selector = sub.add_parser(kind)
        selector.add_argument("--out", required=True, type=Path)
        if kind in ("run", "stage"):
            selector.add_argument("--registration", required=True, nargs=2)
            selector.add_argument("--seed", type=int, choices=SEEDS, required=kind == "stage")
            selector.add_argument("--arm", choices=ARMS, required=kind == "stage")
        if kind in ("run", "stage", "probe"):
            selector.add_argument("--gpu")
        if kind in ("prepare", "probe"):
            selector.add_argument("--strict-cuda", action="store_true")
        if kind == "run":
            selector.add_argument("--resource-wait-seconds", type=float, default=0)
        if kind == "stage":
            selector.add_argument("--lease-fd", type=int)
            selector.add_argument("--matrix-fd", required=True, type=int)
            selector.add_argument("--phase", required=True, choices=(*STEPS, "audit"))
        if kind == "probe":
            selector.add_argument("--manifest", nargs=2, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require(sys.platform == "linux" and __debug__ and code["commit"] == args.expected_commit, "Expected Linux frozen revision")
    require_frozen_code(code)
    torch.set_num_threads(1)
    out = external_directory(args.out)
    if args.action == "prepare":
        from test_g4_support_context import prepared

        out.mkdir()
        _, _, directory, _ = prepared(out)
        path = out / "registration.json"
        write_json(path, plan(directory / "manifest.json", code, "synthetic_only",
                              arithmetic=p4.CUDA_FP64_WEIGHT if args.strict_cuda else None))
        print(json.dumps(pinned(path)))
    elif args.action == "probe":
        probe(args, code, out)
    else:
        require((args.seed is None) == (args.arm is None), "Specify a seed and arm together")
        value = registration(args, code)
        try:
            (stage if args.action == "stage" else run)(args, code, value, out)
        except p4.ResourceUnavailable as exc:
            print(json.dumps({"status": "waiting_for_resources", "reason": str(exc)}), file=sys.stderr, flush=True)
            raise SystemExit(75) from exc


if __name__ == "__main__":
    main()

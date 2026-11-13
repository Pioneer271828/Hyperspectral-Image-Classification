"""Replay the pinned WP10 development workload and compare SVD initializations."""

from argparse import ArgumentParser
import io
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns, process_time_ns

import numpy as np
import torch

from compression.low_rank import matrix
from compression.low_rank_recovery import HEADS, TARGET, initial_evidence, transform
from compression.low_rank_recovery_audit import package_model
from compression.low_rank_tensorly import RANKS, TENSORLY_COMMIT, TensorLySVD
from compression.precision_cost import ResidentPrecisionTile, _tile_identity, prepare_resident_precision_tiles
from compression.registered_low_rank_recovery import _design
from evaluation.classification_metrics import metrics_from_confusion
from evaluation.wp12_low_rank_evidence import adapt, verify_guard
from inference.wp10_low_rank_execution import RUNTIME_PATHS, _outputs, _scan, package
from research_runtime import ROOT, code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash


PRODUCER = "01b99515156694b09c250723b89277f475700897"
DEPLOYMENT = "2e5807e46d1c37163fbe8aed4fe54cd62eed139c"
PINS = {
    "registration": ("wp10-low-rank-xuzhou01/registration.json", "1c13c895555d410e359d46cf0dc76d2a48873f65f1253478175ea5671d8152f0"),
    "audit": ("wp10-low-rank-xuzhou01/audit.json", "0eceb1e01657acc960860969d2360d2399f9d25987e03973e802d73c99ad400b"),
    "batch": ("wp10-low-rank-xuzhou01/batch-complete.json", "2ca3c9e9705ae53da7a764a4bfd42c39f14faa688efd839910f4fd75539d197e"),
    "manifest": ("wp10-low-rank-frozen-execution01/manifest.json", "a0cae3f8723902555820964e88184f476b11da5a4a01e19e92d67f0a1a7dae6b"),
    "cost": ("wp10-low-rank-cpu-cost01/receipt.json", "2d4b0618e6c5548f68ce1b3a9c3b8076ba4b2688512a8211482264f1882c2e9d"),
}


def measure(fn):
    cpu, wall = process_time_ns(), perf_counter_ns()
    value = fn()
    return value, dict(wall_ms=(perf_counter_ns()-wall)/1e6, process_cpu_ms=(process_time_ns()-cpu)/1e6)


def paired(arms, warmups, repeats):
    names = list(arms)
    if len(names) != 2 or repeats % 2:
        raise ValueError("Even, balanced paired comparisons required")
    for _ in range(warmups):
        for fn in arms.values():
            fn()
    rows = []
    for index in range(repeats):
        order = names if index % 2 == 0 else names[::-1]
        rows.append(dict(order=order, arms={name: measure(arms[name])[1] for name in order}))
    return dict(warmups_per_arm=warmups, samples=rows,
                median_ms={name: median(row["arms"][name]["wall_ms"] for row in rows) for name in names},
                paired_median_first_over_second=median(
                    row["arms"][names[0]]["wall_ms"]/row["arms"][names[1]]["wall_ms"] for row in rows))


def load_scope(runs, legacy, code):
    pins = {key: dict(path=str(runs / path), sha256=sha) for key, (path, sha) in PINS.items()}
    values = {key: admission.read(pin) for key, pin in pins.items()}
    for revision, paths in ((PRODUCER, RUNTIME_PATHS),
                            (DEPLOYMENT, (*RUNTIME_PATHS, "inference/wp10_low_rank_execution.py",
                                          "compression/precision_cost.py", "evaluation/classification_metrics.py"))):
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", revision, code["commit"]], check=True)
        subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", revision, code["commit"],
                        "--", *paths], check=True)
    # The original manifest binds its producer's deployment code, which verifies itself in its own checkout.
    check = ("import sys; from research_runtime import code_snapshot; "
             "from inference.wp10_low_rank_execution import load_manifest; "
             "load_manifest(dict(path=sys.argv[1],sha256=sys.argv[2]),code_snapshot())")
    subprocess.run([sys.executable, "-c", check, pins["manifest"]["path"], pins["manifest"]["sha256"]],
                   cwd=legacy, env={**os.environ, "PYTHONPATH": str(legacy)}, check=True)
    reg, manifest = values["registration"], values["manifest"]
    if (reg["commit"] != PRODUCER or manifest["commit"] != DEPLOYMENT or len(manifest["cases"]) != 18
            or values["batch"]["registration"] != pins["registration"]
            or values["batch"]["updates"] != reg["maximum_updates"]):
        raise ValueError("Original frozen WP10 completion and manifest are required")
    rows, overview = adapt({name: pins[name] for name in ("registration", "audit", "manifest", "cost")})
    spec, dense, scene, record, _ = _design(reg["request"])
    if any(reg[key] != value for key, value in spec.items()):
        raise ValueError("Original source or labels differ")
    extra = [*reg["protected"].values(), manifest["resident_inputs"], values["cost"]["resident"]]
    for entry in manifest["cases"].values():
        extra.extend(entry["package"][name] for name in ("weights", "sidecar"))
    admission.check_pins([*pins.values(), *extra])
    return pins, values, extra, rows, overview, dense, scene, record


def resident_tiles(manifest):
    rows = torch.load(io.BytesIO(dev._read(manifest["resident_inputs"])), map_location="cpu", weights_only=True)
    tiles = [ResidentPrecisionTile(**row) for row in rows]
    if _tile_identity(tiles) != manifest["tiles"]:
        raise ValueError("Original resident tiles differ")
    return tiles


def locations(reg, scene, record):
    released = dict(zip(record["validation_support"]["indices"], record["validation_support"]["labels"], strict=True))
    width = scene.plan["shape"][1]
    roles = {row["id"]: row for row in record["roles"] if row["role"] == "devval"}
    targets, per_tile, used = [], [], set()
    for tile in reg["tiles"]:
        y0, y1, x0, x1 = roles[tile["tile_id"]]["box"]
        indices = sorted(i for i in released if y0 <= i//width < y1 and x0 <= i%width < x1)
        if used.intersection(indices):
            raise ValueError("Released coordinate occurs more than once")
        used.update(indices)
        targets.extend(released[i] for i in indices)
        per_tile.append([(i//width-y0)*(x1-x0)+i%width-x0 for i in indices])
    if used != released.keys() or len(targets) != 45 or set(targets) != set(reg["class_ids"]):
        raise ValueError("Only the original V5 support is admitted")
    return targets, per_tile


def score(model, reg, tiles, selection):
    before = state_hash(model)
    values, calls = _scan(model, reg, tiles, evidence=True)
    targets, offsets = selection
    classes = np.asarray(reg["class_ids"])
    index = {value: i for i, value in enumerate(classes)}
    metrics, predictions = {}, {}
    for n, head in enumerate(HEADS):
        labels = [label for block, rows in zip(values, offsets, strict=True)
                  for label in classes[block[n].cpu().numpy()[rows].argmax(axis=1)].tolist()]
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for actual, predicted in zip(targets, labels, strict=True):
            confusion[index[actual], index[predicted]] += 1
        truth = [index[label] for label in targets]
        predicted = [index[label] for label in labels]
        vectorized = np.bincount(np.asarray(truth)*len(classes)+predicted, minlength=confusion.size)
        np.testing.assert_array_equal(confusion, vectorized.reshape(confusion.shape))
        metrics[head] = metrics_from_confusion(confusion, classes)
        predictions[head] = labels
    if state_hash(model) != before:
        raise ValueError("Inference changed package state")
    return dict(metrics=metrics, predictions=predictions, targets=targets, factor_calls=calls,
                output_hashes=_outputs(values, reg)), values


def agreement(actual, expected):
    if (actual["targets"] != expected["targets"] or actual["predictions"] != expected["predictions"]
            or actual["output_hashes"] != [dict(zip(HEADS, row, strict=True)) for row in expected["probabilities"]]
            or any(any(value != expected["metrics"][head][key] for key, value in row.items())
                   for head, row in actual["metrics"].items())):
        raise ValueError("Frozen WP10 outputs or independent metrics differ")


def save_candidate(model, dense, name, assignment, out):
    model.backbone.adapter.cache_attention = False
    case = dict(case=name, seed=64, assignment=assignment)
    initial = initial_evidence(dense, model, assignment)
    return dict(case=case, initial=initial, package=package_model(model, case, initial, out / f"{name}.pt"))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--legacy-checkout", type=Path, required=True)
    parser.add_argument("--tensorly-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if args.expected_commit != code["commit"] or torch.get_num_threads() != 1:
        raise ValueError("Exact checked-out commit and single-thread CPU required")
    require_frozen_code(code)
    if torch.cuda.is_available():
        raise ValueError("The frozen replay requires CPU forward and no visible GPUs")
    out = external_directory(args.out)
    out.mkdir()
    (out / "packages").mkdir()
    scope, setup_time = measure(lambda: load_scope(args.runs.resolve(), args.legacy_checkout.resolve(), code))
    pins, values, protected, historical_rows, historical, dense, scene, record = scope
    manifest, reg, audit = (values[key] for key in ("manifest", "registration", "audit"))
    tiles, input_load = measure(lambda: resident_tiles(manifest))
    prepared, prepare_time = measure(lambda: prepare_resident_precision_tiles(scene, reg["validation"]["tiles"]))
    if _tile_identity(prepared) != manifest["tiles"]:
        raise ValueError("Original normalized role inputs differ")
    dense.backbone.adapter.cache_attention = False
    selection = locations(manifest, scene, record)
    write_json(out / "scope.json", dict(commit=code["commit"], source_hash=code["source_hash"],
               pins=pins, protected_files=len(protected)+len(pins), preparation=dict(
                   authenticate_and_load_source=setup_time, read_decode_resident=input_load,
                   canonical_role_observation_and_tensor_preparation=prepare_time),
               tiles=manifest["tiles"], threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
               torch=str(torch.__version__), numpy=np.__version__, query=0, gt=0, optimizer_updates=0))
    original, _ = score(dense, manifest, tiles, selection)
    old_source = dict(audit["source"])
    agreement(original, old_source)
    frozen = {}
    for name in manifest["case_order"]:
        entry = manifest["cases"][name]
        model, load_time = measure(lambda: package(entry))
        result, sweep = measure(lambda: score(model, manifest, tiles, selection))
        row, _ = result
        agreement(row, audit["report"][name]["selected"])
        if row["output_hashes"] != values["cost"]["startup"][name][0]["output_hashes"]:
            raise ValueError("Replayed selected package differs from original startup receipt")
        frozen[name] = dict(row, package=entry["package"], selected_step=entry["selected_step"],
                            single_authenticated_load=load_time, instrumented_score_and_sweep=sweep)
        print("frozen", name, flush=True)
    write_json(out / "replay.json", dict(original=original, cases=frozen, historical=historical,
               reviewed_cost_rows={row["case_id"]: row["cost"] for row in historical_rows},
               probability_hash_matches=18*5*3, query=0, gt=0, optimizer_updates=0))

    oracle, provider_open = measure(lambda: TensorLySVD(args.tensorly_source))
    assignments = reg["ladder"]["arms"]
    candidates = {}
    retained = {}
    for arm, assignment in assignments.items():
        model = transform(dense, assignment, 64)
        name = f"native-{arm}"
        retained[name] = save_candidate(model, dense, name, assignment, out / "packages")
        if assignment["init"] == "svd":
            name = f"tensorly-{arm}"
            model = oracle.initialize(dense, assignment["rank"]).eval()
            retained[name] = save_candidate(model, dense, name, assignment, out / "packages")
    small = audit["small_classes"]
    limits = reg["protection"]["limits"]
    probabilities, models = {}, {}
    for name, entry in retained.items():
        models[name] = package(entry)
        row, probabilities[name] = score(models[name], manifest, tiles, selection)
        candidates[name] = dict(row, **entry, guard_against_original=verify_guard(
            original, row, manifest["class_ids"], small, limits))
    weight = matrix(dense.get_submodule(TARGET))[1].cpu().double()
    singular = np.linalg.svd(weight.numpy(), compute_uv=False)
    comparisons = {}
    for arm, assignment in assignments.items():
        if assignment["init"] != "svd":
            continue
        rank = assignment["rank"]
        native, tensorly = f"native-{arm}", f"tensorly-{arm}"
        deltas = {head: max(float((a[i]-b[i]).abs().max()) for a, b in
                  zip(probabilities[native], probabilities[tensorly], strict=True)) for i, head in enumerate(HEADS)}
        provider_guard = verify_guard(candidates[native], candidates[tensorly], manifest["class_ids"], small, limits)
        if max(deltas.values()) > 1e-5 or any(not v["passes_metric_guards"] for v in provider_guard.values()):
            raise ValueError("TensorLy and native initial candidates differ beyond the comparison gate")
        reconstructions = {}
        for name in (native, tensorly):
            target = models[name].get_submodule(TARGET)
            actual = (target.left.double() @ target.right.double()).detach()
            error = float(torch.linalg.norm(actual-weight))
            np.testing.assert_allclose(error, np.linalg.norm(singular[rank:]), rtol=1e-5, atol=1e-6)
            reconstructions[name] = error
        layer = dense.get_submodule(TARGET)
        comparisons[arm] = dict(rank=rank, max_probability_delta=deltas, provider_guard=provider_guard,
            optimal_tail=float(np.linalg.norm(singular[rank:])), reconstruction_residual=reconstructions,
            decomposition=paired({native: lambda: torch.linalg.svd(weight, full_matrices=False),
                tensorly: lambda: oracle.svd_module.truncated_svd(weight.numpy(), n_eigenvecs=rank)}, 3, 24),
            initializer=paired({native: lambda: transform(dense, assignment, 64),
                tensorly: lambda: oracle.initialize(dense, rank)}, 3, 12),
            reload_and_first_sweep=paired({name: lambda name=name: _scan(package(retained[name]), manifest, tiles)
                                          for name in (native, tensorly)}, 2, 12),
            authenticated_reload=paired({name: lambda name=name: package(retained[name])
                                         for name in (native, tensorly)}, 2, 12),
            resident=paired({name: lambda name=name: _scan(models[name], manifest, tiles)
                             for name in (native, tensorly)}, 3, 12))
        print("initial", arm, flush=True)
    if set(RANKS) != {row["rank"] for row in comparisons.values()}:
        raise ValueError("Expected all four original SVD ranks")
    native_dense = probabilities["native-dense"]
    dense_delta = {head: max(float((a[i]-b[i]).abs().max()) for a,b in
                   zip(native_dense, probabilities["native-full-rank"], strict=True)) for i,head in enumerate(HEADS)}
    if max(dense_delta.values()) > 1e-5 or _tile_identity(tiles) != manifest["tiles"]:
        raise ValueError("Full-rank native control or resident inputs differ")
    for name, model in models.items():
        row, _ = score(model, manifest, tiles, selection)
        if row["output_hashes"] != candidates[name]["output_hashes"] or state_hash(model) != retained[name]["package"]["state"]:
            raise ValueError("Resident timings changed a frozen candidate")
    admission.check_pins([*pins.values(), *protected])
    require_frozen_code(code)
    write_json(out / "initial.json", dict(status="passed", tensorly_commit=TENSORLY_COMMIT,
               provider_attestation_and_import=provider_open, numpy_backend=True, original_dense=original,
               candidates=candidates, comparisons=comparisons, native_full_rank_probability_delta=dense_delta,
               decomposition_scope="CPU FP64 contiguous weight already resident, before factor placement",
               resident_scope="CPU full model on fixed resident inputs, includes partition graph generation",
               startup_scope="warm process with authenticated load and first full sweep on loaded instance",
               dependencies={"tensorly": oracle.tensorly.__version__, "module_sha256": hash_file(oracle.svd_module.__file__)},
               query=0, gt=0, optimizer_updates=0))


if __name__ == "__main__":
    main()

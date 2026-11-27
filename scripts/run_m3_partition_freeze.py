"""Lock and run a three-arm frozen M3 comparison on the released development role."""
from time import perf_counter

START = perf_counter()

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from m3_partition_graph import grid_labels, slic_graph

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "395caef57fba8ab6a09223538f748179dab552f2"
READER = "6bb51b7d2b1e01e579fb07aad521f23d1c2ccc17"
PILOT = "bddd9f59daf2a755c719de647a429d08d0932337547e65eb2e91a8070fea58b9"
SLIC_WHEEL = "a4c464b90e978d137330be433df4e76d92ad3c5f46a22f159520ce0fdbea8a09"
SEEDS = (64, 512, 2048)
ARMS = ("native", "grid", "slic")
HEADS = ("fused", "graph", "cnn")
SCENES = {"paviaU": (9, 17, 50), "xuzhou": (9, 5, 100), "whu-honghu": (22, 10, 100)}
COHORTS = {"c3-pavia": ("c3", "paviaU"), "c3-xuzhou": ("c3", "xuzhou"),
           "c3-honghu": ("c3", "whu-honghu"), "pavia-40": ("p40", "paviaU")}
SOURCE = {
    "c3": ("paper-c-recovery-20261126", "engineering3-execution01",
           "8263398636b284a27ee5f14cec3494fe0f8dc943195cde7042315d658d8cad17",
           "engineering3-independent-audit01.json",
           "4b80cb2734b0c1e66c0c229acd3068fd6813fb83398731e642994ea56e02e116"),
    "p40": ("paper-a-external-pavia-20261127", "execution",
            "b37d23eab46c331b389a0a9472b277360cbc99a20c4259b8a7ef8a6f671b6e01",
            "full-review.json", "cf23ce22f97f6c7bc789e3bac7cc4c62f7c7fd6e7cfd3c702ac244342c953eab"),
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pin(path, digest):
    check(sha(path) == digest, f"Frozen file differs: {path.name}")
    return path


def read(path, digest=None):
    if digest:
        check_pin(path, digest)
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def clean(root, expected):
    git = lambda *args: subprocess.check_output(["git", "-C", root, *args], text=True).strip()
    check(git("rev-parse", "HEAD") == expected and not git("status", "--porcelain"),
          f"Expected a clean checkout at {expected}")


def candidate():
    path = ROOT / "transfer" / "slic_assignment.py"
    spec = importlib.util.spec_from_file_location("m3_local_slic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.slic_assignment


def check_slic(runs):
    import skimage
    from skimage.segmentation import _slic, slic_superpixels

    from zipfile import ZipFile
    previous = runs / "oss-slic-portability-20261127"
    wheel, = (previous / "wheels").glob("scikit_image-0.25.2-cp310-*.whl")
    check_pin(wheel, SLIC_WHEEL)
    site = previous / "site-0.25.2"
    check(skimage.__version__ == "0.25.2", "Run the single previously qualified SLIC release")
    sourced = {}
    with ZipFile(wheel) as archive:
        for module in (skimage, _slic, slic_superpixels):
            active = Path(module.__file__).resolve()
            check(active.is_relative_to(site), "SLIC must come from the isolated prior wheel")
            name = active.relative_to(site).as_posix()
            check(archive.read(name) == active.read_bytes(), "Loaded SLIC file differs from its pinned wheel")
            sourced[name] = sha(active)
    return {"version": "0.25.2", "wheel_sha256": SLIC_WHEEL, "loaded": sourced}


def accepted_cases(runs, cohort):
    from tasks import MODEL
    from transactions import digest

    history, scene = COHORTS[cohort]
    folder, execution, reg_sha, audit_name, audit_sha = SOURCE[history]
    root = runs / folder
    register = read(root / execution / "registration.json", reg_sha)
    audit = read(root / audit_name, audit_sha)
    pilot = Path(register["pilot"])
    check(pilot.is_relative_to(runs), "Prior bank receipt escapes runs")
    check_pin(pilot, PILOT)
    audit_cases = {row["seed"]: row for row in audit["cases"] if row["scene"] == scene and row["method"] == "native"}
    check(set(audit_cases) == set(SEEDS), "Expected three accepted native model seeds")
    cases = []
    sweep_count = 3 if history == "c3" else 40
    for seed in SEEDS:
        case_name = f"{scene}-s{seed}-native"
        case, = [row["contract"] for row in register["cases"] if row["id"] == case_name]
        checked = audit_cases[seed]
        check(checked["status"] == "PASSED" and checked["query_reads"] == 0
              and case["method"] == "native" and case["scene"] == scene and case["seed"] == seed
              and case["native_model"] == MODEL and case["dependency"] == NATIVE
              and case["sweeps"] == checked["sweeps"] == sweep_count,
              f"Original case role, method, or checkpoint contract differs: {case_name}")
        directory = root / execution / case_name
        done = read(directory / "done.json")
        step = checked["selected_step"]
        saved = read(directory / "contract.json", checked["contract_sha256"])
        check(saved == case and done["selected_step"] == step and done["contract"] == digest(case),
              "Original selected cursor or completed contract differs")
        record_path = directory / "stages" / f"{step:06d}" / "record.json"
        record = read(record_path)
        checkpoint = record_path.with_name("state.pt")
        check(record["step"] == step and record["contract"] == done["contract"]
              and checkpoint.stat().st_size == record["checkpoint_bytes"], "Wrong selected checkpoint envelope")
        check_pin(checkpoint, record["state_sha256"])
        cases.append({"seed": seed, "checkpoint": str(checkpoint.relative_to(runs)),
                      "checkpoint_sha256": record["state_sha256"], "checked_step": step,
                      "expected_fused": checked["selected_metrics"], "record_sha256": sha(record_path),
                      "contract_identity": done["contract"], "contract_file_sha256": checked["contract_sha256"]})
    return {"scene": scene, "sweeps": sweep_count, "cases": cases,
            "producer": register["producer"], "registration_sha256": reg_sha, "review_sha256": audit_sha}


def material(runs, reader, scene, compute_slic, locked=None):
    from model.learnable_superpixel_graph import LearnableSuperpixelGraph
    from research_runtime import tensor_hash
    from tasks import prior_module

    pilot_path = runs / "paper1-c-baselines-20261126" / "registration02.json"
    declared = read(pilot_path, PILOT)["scenes"][scene]
    original = prior_module(reader, "pilot")
    bank, published = original.load_bank(SimpleNamespace(runs_root=runs), scene)
    check(declared["receipt"] == published and declared["pin"] == bank.pin
          and declared["geometry"] == original.geometry(bank), "Admitted bank or role geometry changed")
    classes, blocks, scale = SCENES[scene]
    release = bank.release
    check(release.classes == classes and release.statistics["fit_partition"] == "train",
          "Class mapping or training statistics changed")
    metadata, inputs = [], {}
    for row in bank.role_rows("devval"):
        if not row["samples"]:
            continue
        _, _, view, context = bank.materialize(row, {"mode": "none"})
        obs = view.observation
        y0, _, x0, _ = row["box"]
        h, w, _ = obs.spectral.shape
        waves = np.array(view.bands.wavelengths, dtype=np.float32)
        unit = f"t{len(metadata)+1:02d}"
        points = [{"point": f"{unit}-p{index:02d}", "class_id": label,
                   "offset": (pixel // release.shape[1] - y0) * w + (pixel % release.shape[1] - x0)}
                  for index, (pixel, label) in enumerate(row["samples"], 1)]
        check(context["role"] == "devval" and obs.partition in ("train", "val")
              and len({p["offset"] for p in points}) == len(points) and
              all(0 <= p["offset"] < h*w for p in points), "Point escapes its released validation tile")
        grid = LearnableSuperpixelGraph._grid_shape(h, w, scale)
        grid_map = grid_labels((h, w), grid)
        if locked is None:
            slic_map, _ = compute_slic(obs.spectral, regions=grid[0] * grid[1])
            slic_members = slic_map.ravel().tolist()
        else:
            slic_members = locked["blocks"][len(metadata)]["slic_members"]
        metadata.append({"unit": unit, "shape": list(obs.spectral.shape), "scale": scale,
                         "grid": list(grid), "native_budget": grid[0] * grid[1],
                         "spectral_sha256": tensor_hash(np.array(obs.spectral)),
                         "coordinates_sha256": tensor_hash(np.array(obs.coordinates)),
                         "wavelength_sha256": tensor_hash(waves), "published_context_sha256": context["normalized"],
                         "partition": obs.partition, "points": points, "slic_members": slic_members,
                         "grid_sha256": tensor_hash(grid_map)})
        inputs[unit] = (obs, waves)
    check(len(metadata) == bank.reads == blocks
          and Counter(p["class_id"] for row in metadata for p in row["points"]) == dict.fromkeys(range(1, classes+1), 5),
          "Incomplete or changed released devval points")
    bank.verify_authority(bank.pin)
    value = dict(classes=classes, bank_sha256=bank.pin["sha256"], publication_sha256=published["sha256"],
                 statistics_sha256=release.statistics["statistics_hash"], blocks=metadata)
    if locked is not None:
        check(value == locked, "Released bank or locked view changed")
    return value, inputs


def load_model(runs, case, scene, classes):
    from model.VCASAHGCN import MultiDatasetVCASAHGCN
    from tasks import MODEL

    path = check_pin(runs / case["checkpoint"], case["checkpoint_sha256"])
    data = torch.load(path, weights_only=True, map_location="cpu")
    with torch.random.fork_rng(devices=[]):
        model = MultiDatasetVCASAHGCN({scene: classes}, **MODEL).eval().requires_grad_(False)
    model.load_state_dict(data["model"], strict=True)
    backbone = model.backbone
    check(backbone.learnable_graph.graph_offsets.shape == (9, 2)
          and all(getattr(backbone, name) is None for name in ("interaction", "edge_updater", "graph_router"))
          and backbone.region_residual_mode == "none", "Unsupported frozen graph or CNN route")
    return model


def sparse(tensor):
    return {"shape": list(tensor.shape), "indices": tensor.indices().tolist(), "values": tensor.values().tolist()}


def infer(model, scene, obs, waves, block, arm, compute_slic, *, witness=False):
    from loadData.spatial_protocol import forward_partition
    from model.region_pixel_alignment import pool_regions
    from research_runtime import tensor_hash

    back = model.backbone
    builder = back.learnable_graph
    build, stages = builder.build_graph, back.forward_stages
    graph1, graph2 = back.graph_stage1, back.graph_stage2
    saved = {}

    def build_state(features, scale, *, assignment_mode, adjacency_mode):
        check(scale == block["scale"] and assignment_mode == adjacency_mode == "learned",
              "Original graph configuration or scale was changed")
        start = perf_counter()
        if arm == "native":
            graph = build(features, scale, assignment_mode=assignment_mode, adjacency_mode=adjacency_mode)
        elif arm == "grid":
            graph = build(features, scale, assignment_mode="fixed_grid", adjacency_mode="learned")
        else:
            labels, _ = compute_slic(obs.spectral, regions=block["native_budget"])
            check(labels.ravel().tolist() == block["slic_members"], "SLIC differs from locked pre-prediction assignment")
            graph = slic_graph(builder, features, labels, tuple(block["grid"]))
        saved.update(graph=graph, build_seconds=perf_counter() - start)
        if witness:
            saved["clean_features"] = features
        return graph

    def observed_stages(cnn_input, region_input, q, a, *args, **kwargs):
        graph = saved["graph"]
        flat = saved["clean_features"].reshape(q.shape[0], -1)
        check(q is graph.Q and a is graph.A0 and torch.equal(region_input, pool_regions(q, flat)),
              "Backbone stages did not consume the candidate Q and regional pooling")
        saved.update(input_columns=flat[:, :2], pooled=region_input[:, :2])
        result = stages(cnn_input, region_input, q, a, *args, **kwargs)
        saved["broadcast"] = result[1][:, :2]
        return result

    def checked1(features, adjacency, *args, **kwargs):
        check(adjacency is saved["graph"].A0 and features.shape[0] == adjacency.shape[0],
              "First graph stage bypassed the candidate adjacency")
        saved["graph1"] = True
        return graph1(features, adjacency, *args, **kwargs)

    def checked2(features, adjacency, *args, **kwargs):
        check(adjacency is saved["graph"].A0 and features.shape[0] == adjacency.shape[0],
              "Second graph stage bypassed the candidate adjacency")
        result = graph2(features, adjacency, *args, **kwargs)
        saved["region_output"] = result[:, :2]
        return result

    start = perf_counter()
    with torch.inference_mode(), patch.object(builder, "build_graph", side_effect=build_state):
        if witness:
            with (patch.object(back, "forward_stages", side_effect=observed_stages),
                  patch.object(back, "graph_stage1", side_effect=checked1),
                  patch.object(back, "graph_stage2", side_effect=checked2)):
                heads = forward_partition(model, scene, obs, waves, block["scale"], wavelength_unit="nm")[:3]
        else:
            heads = forward_partition(model, scene, obs, waves, block["scale"], wavelength_unit="nm")[:3]
    elapsed = perf_counter() - start
    state = saved["graph"]
    binding = {"q_sha256": tensor_hash(state.Q), "a_sha256": tensor_hash(state.A0)}
    if witness:
        check(saved["graph1"] and "region_output" in saved, "Both graph stages must run")
        detail = {name: saved[name].tolist() for name in ("input_columns", "pooled", "region_output", "broadcast")}
        detail.update(Q=sparse(state.Q), A0=sparse(state.A0), positions=state.region_positions.tolist())
        return heads, binding, detail
    return heads, binding, {"forward_ms": elapsed * 1000, "build_ms": saved["build_seconds"] * 1000}


def execute(args, plan, chosen, compute_slic):
    from transactions import model_hash
    from research_runtime import tensor_hash

    config = accepted_cases(args.runs, chosen)
    check(plan["cohorts"][chosen] == config, "Frozen cohort authority changed")
    scene = config["scene"]
    bank, inputs = material(args.runs, args.reader, scene, compute_slic, plan["banks"][scene])
    intake_seconds = perf_counter() - START
    rows, loaded = [], []
    first_result = None
    timed_calls = 0
    for case in config["cases"]:
        load_start = perf_counter()
        model = load_model(args.runs, case, scene, bank["classes"])
        initial, rng = model_hash(model), torch.get_rng_state().clone()
        loaded.append({"seed": case["seed"], "checkpoint_load_seconds": perf_counter() - load_start,
                       "state_sha256": initial, "parameters": sum(p.numel() for p in model.parameters())})
        for block in bank["blocks"]:
            started = perf_counter()
            obs, waves = inputs[block["unit"]]
            heads, binding, cost = infer(model, scene, obs, waves, block, args.arm, compute_slic)
            if first_result is None:
                first_result = perf_counter() - START
            costs = [cost]
            full_hashes = [tensor_hash(v) for v in heads]
            timed_calls += 1
            for _ in range(2):
                repeat, actual, sample = infer(model, scene, obs, waves, block, args.arm, compute_slic)
                check(actual == binding and [tensor_hash(v) for v in repeat] == full_hashes, "Timed repeats differ")
                costs.append(sample)
                timed_calls += 1
            proof, observed, witness = infer(model, scene, obs, waves, block, args.arm, compute_slic, witness=True)
            check(binding == observed and [tensor_hash(v) for v in proof] == full_hashes,
                  "The graph-consumption witness changed the measured output")
            points = [dict(point=p["point"], probabilities=[v[p["offset"]].tolist() for v in heads])
                      for p in block["points"]]
            rows.append(dict(seed=case["seed"], unit=block["unit"], graph=binding, witness=witness,
                             heads=dict(zip(HEADS, full_hashes, strict=True)), points=points, times=costs,
                             total_with_evidence_ms=(perf_counter() - started) * 1000))
        check(model_hash(model) == initial and torch.equal(rng, torch.get_rng_state()), "Frozen state or RNG changed")
    return dict(schema="m3_partition_capture_v1", arm=args.arm, cohort=chosen, producer=args.expected_commit,
                registration_sha256=args.registration_sha256, inputs_and_identity_seconds=intake_seconds,
                process_to_first_prediction_seconds=first_result, process_before_write_seconds=perf_counter()-START,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                timed_forwards=timed_calls, proof_forwards=len(rows), models=loaded, rows=rows,
                optimizer_updates=0, query_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "execute"))
    for name in ("runs", "native", "reader", "registration", "out"):
        parser.add_argument(f"--{name}", type=Path, required=name != "out")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    for name in ("runs", "native", "reader", "registration"):
        setattr(args, name, getattr(args, name).resolve())
    clean(ROOT, args.expected_commit)
    clean(args.native, NATIVE)
    clean(args.reader, READER)
    check(args.registration.is_relative_to(args.runs) and "model.VCASAHGCN" not in sys.modules
          and not torch.cuda.is_initialized(), "Read-only experiment requires an isolated CPU import and authorized runs")
    sys.path.insert(0, str(args.native))
    sys.path.insert(1, str(args.reader / "research" / "paper_c_recovery"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    compute_slic = candidate()
    slic = check_slic(args.runs)
    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Frozen comparison only")):
        if args.action == "register":
            check(not (args.registration_sha256 or args.cohort or args.arm or args.out), "Register before predictions")
            banks = {scene: material(args.runs, args.reader, scene, compute_slic)[0] for scene in SCENES}
            cohorts = {key: accepted_cases(args.runs, key) for key in COHORTS}
            write(args.registration, dict(schema="m3_partition_registration_v1", producer=args.expected_commit,
                                           cohorts=cohorts, banks=banks, slic=slic, arms=ARMS, heads=HEADS,
                                           native=NATIVE, reader=READER, query_accessed=False, optimizer_updates=0))
            print(sha(args.registration))
        else:
            check(args.out and args.out.resolve().is_relative_to(args.runs)
                  and args.cohort and args.arm and args.registration_sha256,
                  "Execution requires a registered cohort, exact input hash and fresh output")
            plan = read(args.registration, args.registration_sha256)
            check(plan["schema"] == "m3_partition_registration_v1" and plan["producer"] == args.expected_commit
                  and plan["slic"] == slic and tuple(plan["arms"]) == ARMS and plan["optimizer_updates"] == 0
                  and plan["query_accessed"] is False, "Wrong frozen plan")
            write(args.out, execute(args, plan, args.cohort, compute_slic))
    clean(ROOT, args.expected_commit)
    clean(args.native, NATIVE)


if __name__ == "__main__":
    main()

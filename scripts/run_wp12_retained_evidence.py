"""Compare frozen learned native endpoints on released development points."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation import frozen_training_stages as stages, retained_spectral_evidence as probe
from evaluation.wp12_release_results import BAND, BAND_PINS
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts import run_frozen_band_region as frozen
from transfer.band_scale import evidence
from transfer.band_scale_real import FOCUS
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward, state_hash
from transfer.native_coverage_replay import frozen_model
from transfer.native_scale_observation import forward
from transfer.registered_development import load_development
from transfer.spatial_coverage_protocol import bank_view

SCHEMA = "wp12_retained_spectral_devval_v1"
RELEASE = ("wp11-matched-xuzhou-frozen01/release.json",
           "9a4493d2c25c6cff64d6b8ee8958e467546985865b72df41540327262d24aacb")
PREVIOUS = ("wp05-wp06-frozen-stage-20261119-qa/candidate/cpu.json",
            "99e8599c3ca3a5f5b06737af5a8eba57ea539be2237d71d81d70bb3ab5a950c1")
STATIC = ("wp05-wp06-frozen-stage-20261119-qa/verified/wp12-stages.json",
          "e4b1d9732cd89cbac4276f3ba51bdcc3889dd2dbb6c57e768b4e7c5724bd1f70")
CASES = ("initial", "prefix", "selected", "last", "paviaU-64", "whu-honghu")
RIVALS = {"xuzhou": {5: 8, 6: 3}, "whu-honghu": {15: 1, 21: 18}, "paviaU": {}}
REQUIRE = stages.require


def pin(root, source):
    return dict(path=str(root/source[0]), sha256=source[1])


def compare_probabilities(old, new, exact):
    a, b = np.asarray(old, dtype=np.float32), np.asarray(new, dtype=np.float32)
    REQUIRE(a.shape == b.shape and np.array_equal(a.argmax(1), b.argmax(1))
            and (np.array_equal(a, b) if exact else np.allclose(a, b, atol=5e-5, rtol=1e-5)),
            "Archived native prediction differs")
    return float(np.max(np.abs(a-b)))


def collect_xuzhou(model, scene, record, band, bands):
    views, native = {}, []
    for role, support in (("devtrain", "train_support"), ("devval", "validation_support")):
        released = views[role] = probe.ReleasedBlocks(role, scene.classes)
        labels = sparse_labels(record, scene.plan, support)
        for tile, binding in zip(scene.plan["tiles"], record["roles"], strict=True):
            if binding["role"] != role:
                continue
            target = labels[tile_slice(tile)].ravel()
            cells = np.flatnonzero(target)
            if not len(cells):
                continue
            view = stages.checked_view(scene, tile, band, bands)
            actual = view.observation
            identifiers = actual.global_ids.ravel()[cells].tolist()
            data = probe.capture(model, actual.spectral, lambda: (_forward(model, scene, tile, observation_view=view), None))
            heads = released.add(len(released.q), cells, identifiers, target[cells], data)
            if role == "devval":
                native.extend(dict(id=[tile["id"], identifier], truth=int(label),
                                   probabilities={head: value[j].tolist() for head, value in heads.items()})
                              for j, (identifier, label) in enumerate(zip(identifiers, target[cells], strict=True)))
    return views, native


def collect_bank(model, bank, producer, band, archive, exact):
    views, native, checks, seen = {}, [], [], set()
    classes = model.heads[archive["scene"]].out_features
    previous = {row["tile"]: row for row in archive["rows"]
                if row["event"] == dict(band=band, factor=1, policy="native")}
    for role in ("devtrain", "devval"):
        released = views[role] = probe.ReleasedBlocks(role, classes)
        for ordinal, row in enumerate(bank.role_rows(role)):
            actual, labels, targets = bank_view(bank, row, producer, band, factor=1)
            ids, truth = labels[:, 0].tolist(), labels[:, 1].tolist()
            data = probe.capture(model, actual.view.observation.spectral, lambda: forward(model, actual, producer, role))
            heads = released.add(ordinal, targets.cells, ids, truth, data)
            if role == "devtrain":
                continue
            native.extend(dict(id=[ordinal, key], truth=label,
                               probabilities={head: value[i].tolist() for head, value in heads.items()})
                          for i, (key, label) in enumerate(zip(ids, truth, strict=True)))
            if ordinal not in previous:
                continue
            old = previous[ordinal]
            now = evidence(actual, targets)
            REQUIRE(all(value == now[key] for key, value in old["evidence"].items() if key != "view"),
                    "Archived admitted observation members differ")
            selected = [i for i, label in enumerate(truth) if label in FOCUS[archive["scene"]]]
            REQUIRE(old["points"] == selected and old["labels"] == [truth[i] for i in selected],
                    "Archived focus label order differs")
            graph = data[3]
            original = old["computation"]["graph"]
            REQUIRE(graph == original if exact else graph["q_shape"] == original["q_shape"],
                    "Archived graph differs")
            checks.append({head: compare_probabilities(old["probabilities"][head], value[selected], exact)
                           for head, value in heads.items()})
            seen.add(ordinal)
    REQUIRE(seen == previous.keys(), "Incomplete archived released factor-1 replay")
    return views, native, checks


def report_case(root, case, producer, device):
    exact = device.type == "cpu"
    if case in CASES[:4]:
        plan = stages.stage_plan(root, pin(root, RELEASE))
        archive = stages.json_pin(root, pin(root, PREVIOUS))
        REQUIRE(archive["plan_sha256"] == digest(plan) and archive["status"] == "passed"
                and archive["optimizer_updates"] == 0 and archive["query_accessed"] is archive["raw_gt_loaded"] is False,
                "Historical frozen trajectory receipt differs")
        release = stages.json_pin(root, plan["release"])
        spec = plan["stages"][case]
        model = stages.load_stage(root, release, spec, case)
        scene, record, dependencies = load_development(plan["development"])
        name, classes = scene.dataset, scene.classes
        REQUIRE(stages.roster(record, name, classes) == plan["identities"]
                and scene.stats["statistics_hash"] == plan["statistics"], "Released trajectory support or statistics differ")
        identity, step, checkpoint = state_hash(model), spec["step"], spec["state"]["sha256"]

        def collect(band):
            roles, rows = collect_xuzhou(model, scene, record, band, plan["bands"][band])
            old = archive["cases"][case]["views"][band]["points"]
            REQUIRE([(r["id"], r["truth"]) for r in rows] == [(r["id"], r["truth"]) for r in old],
                    "Historical trajectory label roster differs")
            checks = [{head: compare_probabilities([r["probabilities"][head] for r in old],
                        [r["probabilities"][head] for r in rows], exact) for head in stages.HEADS}]
            return roles, rows, checks, plan["bands"][band]

        archive_digest = PREVIOUS[1]
        authority = plan["development"]["sha256"]
    else:
        archive_digest = BAND_PINS[case]
        archive = stages.json_pin(root, pin(root, (f"{BAND}/{case}.json", archive_digest)))
        name = archive["scene"]
        REQUIRE(archive["case"] == case and archive["role"] == "devval" and archive["real_updates"] == 0
                and archive["query_accessed"] is archive["raw_gt_loaded"] is False, "Historical archive scope differs")
        parents, documents = frozen.source_documents(root)
        bank, intake = frozen._freeze_bank(root, name, {"sources": parents})
        model, seed, identity, step, checkpoint = frozen_model(root, case, bank, intake, documents)
        authority = bank.pin["sha256"]
        REQUIRE((identity, seed, step, checkpoint, authority) == (archive["model"], archive["model_seed"],
                archive["selected_step"], archive["checkpoint"], archive["preflight_population"]),
                "Frozen checkpoint or development authority differs")
        classes = model.heads[name].out_features
        dependencies = ()

        def collect(band):
            roles, rows, checks = collect_bank(model, bank, producer, band, archive, exact)
            row = bank.role_rows("devval")[0]
            actual, _, _ = bank_view(bank, row, producer, band, factor=1)
            return roles, rows, checks, list(actual.view.bands.band_ids)

    model.to(device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention = False
    model.backbone.adapter.clear_attention_cache()
    rivals = RIVALS[name]
    by_view, reference = {}, None
    for band in stages.VIEWS:
        roles, native, checked, retained = collect(band)
        for i, row in enumerate(native):
            row["id"] = [i + 1]
        train, val = roles["devtrain"], roles["devval"]
        REQUIRE([r["truth"] for r in native] == val.truth, "Native and readout label order differ")
        if reference is None:
            reference = native
            full_ids = retained
            roster = tuple(val.identities)
        REQUIRE(roster == tuple(val.identities) and len(set(retained)) == len(retained)
                and retained == [i for i in full_ids if i in set(retained)], "Same-point roster or retained order differs")
        metrics = {}
        for head in stages.HEADS:
            p = np.asarray([row["probabilities"][head] for row in native])
            metrics[head] = probe.point_metrics(p, list(range(1, classes+1)), val.truth, rivals)
        by_view[band] = dict(band_ids=retained, roles={k: v.summary() for k, v in roles.items()},
                archived_comparison=checked, native_scores=stages.scores(native, classes),
                native_transitions=stages.paired(reference, native, classes, rivals),
                truth=val.truth, native=metrics, native_probabilities=[row["probabilities"] for row in native],
                readout=probe.readout(train, val, rivals))
    for dependency in dependencies:
        stages.pinned(root, dependency)
    REQUIRE(state_hash(model) == identity
            and len(by_view["central_gap"]["band_ids"]) == len(by_view["matched_dispersed"]["band_ids"]),
            "Frozen state or equal-size band comparison differs")
    return dict(case=case, scene=name, classes=classes, stage=step, state=identity, checkpoint=checkpoint,
                authority=authority, archive_sha256=archive_digest, focus=FOCUS[name], views=by_view)


def validate_static(root):
    archived = stages.json_pin(root, pin(root, STATIC))
    original = archived["original_archive"]
    REQUIRE(archived["policy"] == stages.POLICY and archived["query_accessed"] is archived["raw_gt_loaded"] is False
            and archived["optimizer_updates"] == 0 and original["checked_cases"] == len(original["cases"]) == 54,
            "Historical static full-band audit differs")
    identities = [value["identity"]["scene"] for value in original["cases"].values()]
    return dict(sha256=STATIC[1], checked_cases=54,
                scenes={scene: identities.count(scene) for scene in sorted(set(identities))},
                use="historical_full_band_only", altered_static_q_replay=False)


def run(root, directory, expected, device, selected):
    from evaluation.wp11_deployment_cost import gpu_scope

    code = code_snapshot()
    REQUIRE(code["commit"] == expected, "Expected complete source commit differs")
    require_frozen_code(code)
    directory = external_directory(directory)
    REQUIRE(root in directory.parents, "Outputs must stay within authorized runs")
    directory.mkdir(parents=True, exist_ok=True)
    static = validate_static(root)
    missing = []
    for case in selected:
        path = directory/f"{case}.json"
        if path.exists():
            data = json.loads(path.read_bytes())
            REQUIRE(data["schema"] == SCHEMA and data["producer"] == expected
                    and data["device"] == str(device) and data["result"]["case"] == case
                    and set(data["result"]["views"]) == set(stages.VIEWS),
                    "Existing receipt requires a separate run directory")
        else:
            missing.append(case)
    if not missing:
        return
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    REQUIRE(device.type == "cpu" or uuid is not None and uuid.startswith("GPU-"), "Pin a single physical GPU UUID")
    with (nullcontext((None, None, None)) if device.type == "cpu" else gpu_scope(uuid)) as (admission, numeric, _):
        for case in missing:
            result = report_case(root, case, expected, device)
            require_frozen_code(code)
            write_json(directory/f"{case}.json", dict(schema=SCHEMA, producer=expected, device=str(device),
                       case=case, static=static, result=result, admission=admission, numeric=numeric,
                       optimizer_updates=0, query_accessed=False, raw_gt_loaded=False))
            print(f"accepted {case} {device}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    run(args.runs.resolve(strict=True), args.out, args.expected_commit, torch.device(args.device), args.case or CASES)


if __name__ == "__main__":
    main()

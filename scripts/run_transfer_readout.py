"""Publish matched all-class transfer readout evidence on released development roles."""

import argparse
import io
import os
from collections import defaultdict
from pathlib import Path
import resource
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS
from evaluation.wp12_coverage_readout import softmax
from loadData.observation_sampling import _apply
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_coverage_readout import WINDOWS, assert_probabilities, parent_case, registration as prior_registration
from scripts.run_frozen_band_region import (_freeze_bank, _honghu_context, _new_honghu,
                                            _read, _write, check, pin, source_documents)
from scripts.run_frozen_mechanisms import case_lock, read_bundle, write_bundle
from transfer import frozen_mechanism_honghu as hh, frozen_mechanism_plan as plan, frozen_mechanism_runner as native
from transfer.development_bank import _atomic_new
from transfer.frozen_branch_readout import BRANCH_TREATMENTS, centroid_reference, fusion_interventions
from transfer.g3_training import state_hash
from transfer.support_readout import SupportReadout, output_scores, support_scale, support_weights, tensor

SCHEMA = "wp05_wp06_transfer_readout_development_v1"
OLD = "wp05-wp06-coverage-readout-20261119"
RECEIPT = "2831b779bf72cadfbe6036715102630a2928d95fc9c2acddf973168b1e00aff0"
PAVIA = ("full", "w051-at000", "w051-at052", "w051-r7719")
SUPPORT_SEED = 12031
GROUPS = ("all", "k2a", "k2b")


def freeze(root, code):
    _, scope, baseline, batch = prior_registration(root, code)
    link = pin(root / OLD / "wp12/receipt.json", RECEIPT)
    receipt = _read(link)
    check(receipt["status"] == "accepted_development_and_synthetic_only" and receipt["case_count"] == 13
          and receipt["real_updates"] == 0 and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False,
          "Only accepted frozen development evidence may enter")
    cases = {f"{scene}__{name}": dict(checkpoint=scope["selected"][scene][name], scene=scene)
             for scene in plan.ARCHITECTURES for name in plan.ARCHITECTURES[scene]}
    for lane in ("honghu24", "honghu60"):
        options = scope["registered_honghu"][lane]
        if lane == "honghu24":
            options = [row for row in options if row["support_seed"] in (512, 2048) and row["arm"] == "none"]
            cells = {(row["support_seed"], row["k"], row["method"]) for row in options}
            check(len(options) == len(cells) == 8, "The declared orthogonal short support grid is missing")
        cases.update({f'{lane}__{row["id"]}': row for row in options})
    reg = dict(schema=SCHEMA, producer=code["commit"], source_hash=code["source_hash"], parent=link,
               focus=scope["previous"]["focus"], guards=scope["previous"]["guards"], cases=cases,
               windows={"xuzhou": list(WINDOWS), "paviaU": list(PAVIA)}, seed=SUPPORT_SEED,
               groups=list(GROUPS), real_updates=0, query_accessed=False, raw_gt_loaded=False,
               validation_selection=False)
    return reg, scope, baseline, batch, receipt


def archive(root, receipt, case):
    name = f"{case}.npz"
    return read_bundle(pin(root / OLD / name, receipt["input_cases"][case]))


def stratified(aliases, labels):
    check(len(set(aliases)) == len(labels) == len(aliases), "Support identities repeat")
    labels = np.asarray(labels)
    a, b = [], []
    for cls in sorted(set(labels.tolist())):
        order = sorted(np.flatnonzero(labels == cls).tolist(),
                       key=lambda i: digest([SUPPORT_SEED, int(cls), aliases[i]]))
        check(len(order) == 5, "Disjoint support repetition requires five released points per class")
        a.extend(order[:2])
        b.extend(order[2:4])
    check(not set(a).intersection(b), "Support repetitions overlap")
    return {"all": np.arange(len(labels)), "k2a": np.asarray(sorted(a)), "k2b": np.asarray(sorted(b))}


def outputs(model, scene, bank, scope, view_names, role, model_identity):
    holders = {win: (defaultdict(list), defaultdict(list)) for win in view_names}
    catalog = {win["id"]: win for win in scope["windows"][scene]}
    population = dict(aliases=[], labels=[], blocks=[])
    for row in plan.population_rows(bank, role):
        view, _, offsets, labels, aliases, tile = native._reader(bank, row, scope["alias_salt"], scene)
        population["aliases"].extend(aliases)
        population["labels"].extend(labels)
        population["blocks"].append(dict(id=tile, aliases=aliases))
        for win in view_names:
            current = view if win == "full" else _apply(
                view.observation, view.bands, tuple(catalog[win]["retained"]))
            result = native._forward(model, scene, current, offsets, bank.release.classes,
                                     {"model": model_identity}, role, make_embeddings=True)
            scores, features = holders[win]
            for head in HEADS:
                scores[head].append(result.probabilities[head])
                features[head].append(result.features[head])
    return population, {win: tuple({k: np.concatenate(v) for k,v in part.items()} for part in parts)
                        for win, parts in holders.items()}


def save_packet(folder, filename, layer, native_scores, values, arrays, key):
    path = folder / filename
    with io.BytesIO() as data:
        torch.save(layer.state(), data)
        _atomic_new(path, data.getvalue())
    restored = SupportReadout.from_state(torch.load(path, weights_only=True, map_location="cpu"), layer.binding)
    start = perf_counter()
    result = restored.predict(native_scores, values, layer.binding)
    elapsed = perf_counter() - start
    deterministic = layer.predict(native_scores, values, layer.binding)
    check(all(torch.equal(result[h], deterministic[h]) for h in HEADS), "Portable readout replay differs")
    for head in HEADS:
        arrays[f"{key}__candidate__{head}"] = result[head].numpy()
    return dict(sha256=pin(path)["sha256"], bytes=path.stat().st_size, binding=layer.binding,
                latency_seconds=elapsed, class_support=layer.counts.int().tolist(),
                scale={h: layer.scales[h].item() for h in HEADS}, weight=layer.weights.tolist())


def add_trial(arrays, folder, scene, window, group, model, observation, train, validation, t, selected, classes):
    native_scores, features = validation
    train_scores, train_features = train
    selected_aliases = [t["aliases"][i] for i in selected]
    labels = np.asarray(t["labels"])[selected]
    check(not set(selected_aliases).intersection(t["forbidden_aliases"]), "Validation alias admitted into support")
    binding = dict(scene=scene, checkpoint=model, observation=observation,
                   support=digest(selected_aliases))
    reduced = {h: train_features[h][selected] for h in HEADS}
    layer = SupportReadout.fit(binding, reduced, {h: train_scores[h][selected] for h in HEADS}, labels, classes)
    key = f"{window}__{group}"
    arrays[f"{key}__fit__labels"] = labels
    for h in HEADS:
        arrays[f"{key}__fit__features__{h}"] = reduced[h]
        arrays[f"{key}__fit__probabilities__{h}"] = train_scores[h][selected]
        arrays[f"{window}__validation_features__{h}"] = features[h]
    control = centroid_reference(reduced, labels, features, classes)
    for h in HEADS:
        arrays[f"{key}__centroid__{h}"] = softmax(-control[f"{h}__distance"])
    return save_packet(folder, f"{key}.pt", layer, native_scores, features, arrays, key)


def native_case(root, folder, reg, scope, batch, receipt, scene, name):
    started = perf_counter()
    before, old = parent_case(root, batch, f"{scene}__{name}")
    _, docs = source_documents(root)
    bank, model, selected = native._configuration(root, scope, docs, scene, name, "cpu")
    windows = reg["windows"][scene]
    train, support = outputs(model, scene, bank, scope, windows, "devtrain", selected["model"])
    val, features = outputs(model, scene, bank, scope, windows, "devval", selected["model"])
    check(val["aliases"] == before["aliases"] and val["labels"] == before["truth"]
          and train["aliases"] == before["train"]["aliases"] and train["labels"] == before["train"]["labels"],
          "Point populations differ from the parent archive")
    train["forbidden_aliases"] = val["aliases"]
    sampled = stratified(train["aliases"], train["labels"])
    arrays, records = {}, {}
    existing = archive(root, receipt, f"{scene}__{name}")[1] if scene == "xuzhou" else None
    for win in windows:
        val_scores, embeddings = features[win]
        parent_key = win if win == "full" else f"{win}__clean"
        for head in HEADS:
            assert_probabilities(val_scores[head], old[f"{parent_key}__{head}"])
            arrays[f"{win}__native__{head}"] = old[f"{parent_key}__{head}"]
        if existing is None:
            logits, _, _ = fusion_interventions(model, scene, embeddings)
        else:
            logits = {arm: existing[f"{win}__logit__{arm}"] for arm in BRANCH_TREATMENTS}
        assert_probabilities(softmax(logits["native"]), old[f"{parent_key}__fused"])
        check(np.array_equal(logits["restored"], logits["native"]), "Fixed fusion restore differs")
        for arm in ("mask_cnn", "mask_graph", "exchange"):
            arrays[f"{win}__{arm}__fused"] = softmax(logits[arm])
        for group, indexes in sampled.items():
            observation = digest([scene, win, bank.pin["sha256"]])
            records[f"{win}__{group}"] = add_trial(
                arrays, folder, scene, win, group, selected["model"], observation,
                support[win], features[win], train, indexes, bank.release.classes)
    check(state_hash(model) == selected["model"], "Frozen model changed")
    return dict(scene=scene, case=name, model=selected, witnesses=records, support_counts={k: len(v) for k,v in sampled.items()},
                population=val, support_repetitions=dict(seed=SUPPORT_SEED, disjoint=True, same_val=True),
                forwards=len(windows) * (len(train["blocks"]) + len(val["blocks"])),
                seconds=perf_counter()-started, windows=windows, groups=list(GROUPS)), arrays


def honghu_archive(root, scope, batch, receipt, case, entry):
    started = perf_counter()
    before, old = parent_case(root, batch, case)
    _, values = archive(root, receipt, case)
    train, val = (before["populations"][role] for role in ("train", "val"))
    truth = train["labels"]
    native_scores = {h: values[f"{h}__native"] for h in HEADS}
    train_scores = {h: old[f"nominal__selected__train__{h}"] for h in HEADS}
    distances = {h: values[f"{h}__distance"] for h in HEADS}
    counts = values["fused__counts"]
    weights = support_weights(train_scores, truth, 22)
    scales = {h: support_scale(tensor(values[f"{h}__between"]), tensor(counts)) for h in HEADS}
    check(np.all(counts == entry["k"]) and sum(counts) == len(truth)
          and all(np.array_equal(counts, values[f"{h}__counts"]) for h in HEADS), "Historical support changed")
    for head in HEADS:
        assert_probabilities(native_scores[head], old[f"nominal__selected__val__{head}"])
    start = perf_counter()
    corrected = output_scores(native_scores, distances, counts, scales, weights)
    latency = perf_counter() - start
    result = {f"nominal__native__{h}": x for h,x in native_scores.items()}
    for h in HEADS:
        result[f"nominal__all__centroid__{h}"] = softmax(-distances[h])
        result[f"nominal__all__candidate__{h}"] = corrected[h].numpy()
        result[f"nominal__distance__{h}"] = distances[h]
        result[f"nominal__between__{h}"] = values[f"{h}__between"]
        result[f"nominal__projection__{h}"] = values[f"{h}__projection"]
        result[f"nominal__bias__{h}"] = values[f"{h}__bias"]
    report = dict(scene="whu-honghu", case=case, model=entry["endpoint_hashes"]["selected"],
                  population=val, seed=entry["seed"], support_seed=entry["support_seed"], method=entry["method"],
                  arm=entry["arm"], k=entry["k"], budget=60, windows=["nominal"], groups=["all"],
                  witnesses={"nominal__all": dict(scale={h: scales[h].item() for h in HEADS}, weight=weights.tolist(),
                                                class_support=counts.tolist(), latency_seconds=latency,
                                                source="archived_frozen_centroid_distance")},
                  support_counts=dict(all=len(truth)), forwards=0, seconds=perf_counter()-started)
    return report, result


def honghu_short(root, folder, scope, baseline, batch, name, entry):
    started = perf_counter()
    before, old = parent_case(root, batch, f"honghu24__{name}")
    short = _read(scope["sources"]["short_scope"])
    ctx = _honghu_context(short)
    _, docs = source_documents(root)
    bank, _ = _freeze_bank(root, "whu-honghu", baseline)
    model, seed, identity, step, _, packet = _new_honghu(root, docs, name, ctx)
    check((seed, identity, step) == (entry["seed"], entry["endpoint_hashes"]["selected"], entry["selected"]),
          "Short holdout checkpoint differs")
    model.eval()
    model.backbone.adapter.cache_attention = False
    chosen = next(row["case"] for row in short["cases"] if row["case"]["id"] == name)
    populations = hh.populations(ctx, chosen, scope["alias_salt"])
    check(hh.serialize_populations(populations) == before["populations"], "Short archive roles differ")
    records = ctx[1]["coverage"]["records"]
    roles = {role: hh._population_pass(model, bank, item, records[role], identity, embed=True)
             for role,item in populations.items()}
    for role in roles:
        for h in HEADS:
            assert_probabilities(roles[role][0][h], old[f"nominal__selected__{role}__{h}"])
    check(state_hash(model) == identity, "Short model mutated")
    train, val = (before["populations"][role] for role in ("train", "val"))
    train["forbidden_aliases"] = val["aliases"]
    arrays = {f"nominal__native__{h}": old[f"nominal__selected__val__{h}"] for h in HEADS}
    witness = add_trial(arrays, folder, "whu-honghu", "nominal", "all", identity,
                        records["val"]["record_hash"], roles["train"], roles["val"], train,
                        np.arange(len(train["labels"])), 22)
    return dict(scene="whu-honghu", case=name, model=identity, checkpoint=packet, seed=seed, method=entry["method"],
                support_seed=entry["support_seed"], k=entry["k"], budget=24, arm=entry["arm"],
                population=val, windows=["nominal"], groups=["all"], witnesses={"nominal__all": witness},
                support_counts=dict(all=len(train["labels"])),
                forwards=sum(len(row["layout"]) for row in populations.values()), seconds=perf_counter()-started), arrays


def registered(root, folder, code):
    inputs = freeze(root, code)
    link = pin(folder / "registration.json")
    manifest = _read(link)
    current = inputs[0]
    if current != manifest:
        check(link["sha256"] == "70dcf954b9e9a71000d35b95f703c32ba1d94cabf420b579970bb2642e1d3133"
              and manifest["producer"] == "c0ae2e407a6dcf4a42819b0bbabaf70ec1a74636"
              and {**current, "producer": manifest["producer"], "source_hash": manifest["source_hash"]} == manifest,
              "Registration, roles or parent changed")
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", manifest["producer"], code["commit"]], text=True
        ).splitlines()
        check(set(paths) <= {"scripts/run_transfer_readout.py", "evaluation/wp12_transfer_readout.py"}, "Completed producer dependencies changed")
    inputs = (manifest, *inputs[1:])
    return link, inputs


def execute(root, folder, code, lane):
    link, (reg, scope, baseline, batch, previous) = registered(root, folder, code)
    for case, entry in reg["cases"].items():
        kind, name = case.split("__", 1)
        if lane not in ("all", kind, case):
            continue
        final = folder / case
        with case_lock(folder / f".{case}.lock"):
            if final.exists():
                info = _read(pin(final / "receipt.json"))
                check(info["registration"] == link["sha256"] and info["case"] == case, "Foreign case resume")
                read_bundle(pin(final / "arrays.npz", info["arrays"]))
                print(f"reuse {case}", flush=True)
                continue
            with TemporaryDirectory(prefix=f".{case}-", dir=folder) as temp:
                stage = Path(temp) / "case"
                stage.mkdir()
                if kind == "honghu60":
                    result, arrays = honghu_archive(root, scope, batch, previous, case, entry)
                elif kind == "honghu24":
                    result, arrays = honghu_short(root, stage, scope, baseline, batch, name, entry)
                else:
                    result, arrays = native_case(root, stage, reg, scope, batch, previous, kind, name)
                require_frozen_code(code)
                write_bundle(stage / "arrays.npz", dict(case=case, scope=link["sha256"]), arrays)
                _write(stage / "receipt.json", dict(case=case, registration=link["sha256"], result=result,
                                                    arrays=pin(stage / "arrays.npz")["sha256"], real_updates=0,
                                                    raw_gt_loaded=False, query_accessed=False,
                                                    rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                os.rename(stage, final)
                print(f'completed {case}: {result["forwards"]} forwards', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "synthetic", "wp12"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected transfer readout checkout")
    root, folder = external_directory(args.root), external_directory(args.directory)
    check(folder != root and folder.is_relative_to(root), "Transfer artifacts must remain in the authorized runs")
    if args.phase == "prepare":
        reg, *_ = freeze(root, code)
        folder.mkdir()
        print(_write(folder / "registration.json", reg))
    elif args.phase == "run":
        execute(root, folder, code, args.case)
    elif args.phase == "synthetic":
        from transfer.support_readout_fixture import exercise
        print(_write(folder / "synthetic.json", exercise(registered(root, folder, code)[0]["sha256"])))
    else:
        from evaluation.wp12_transfer_readout import publish
        print(publish(root, folder, code))


if __name__ == "__main__":
    main()

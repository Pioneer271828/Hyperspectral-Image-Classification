"""Register, run and independently collect WP05/WP06 development evidence."""

import argparse
from collections import defaultdict
from pathlib import Path
import resource
import subprocess
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS
from loadData.observation_sampling import _apply
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import _freeze_bank, _honghu_context, _read, _write, check, pin, source_documents
from scripts.run_frozen_mechanisms import case_lock, read_bundle, scope_check, write_bundle
from transfer import frozen_mechanism_honghu as honghu, frozen_mechanism_plan as plan, frozen_mechanism_runner as native
from transfer.frozen_branch_readout import BRANCH_TREATMENTS, centroid_reference, fusion_interventions
from transfer.g3_training import state_hash
from transfer.spectral_spacing import spectral_spacing

PARENT = "wp05-wp06-mechanisms-20261119.LKplDgPi"
SCOPE = "7a2f73fb913d0f0d6212dcc0bae9b99681d1d4c098ff56909710f166a1342ff4"
BATCH = "286f1b5c006eec80c3bd50f264837fe05cf587200f0dd0dc2c73d0b62a4a3856"
REVIEW = "20b5420fe5ea93c129a248ca05ef385f09058be5d73e29baabe8b2350dad75d3"
POLICY = "wp05_wp06_spacing_and_frozen_readout_v1"
WINDOWS = ("full", "w218-at000", "w218-at218", "w218-r7719")
ABS = 2e-6


def registration(root, code):
    old = root / PARENT
    links = dict(scope=pin(old / "registration-v3/scope.json", SCOPE),
                 batch=pin(old / "batch-v3.json", BATCH), review=pin(old / "wp12-final/receipt.json", REVIEW))
    scope, baseline, _ = scope_check(root, links["scope"], code, producer=False)
    batch, review = _read(links["batch"]), _read(links["review"])
    check(batch["scope"] == review["scope"] == SCOPE and review["batch"] == BATCH
          and batch["cases"] == review["case_count"] == 56 and batch["status"] == "complete"
          and review["status"] == "accepted_development_only", "Expected accepted mechanism batch")
    selected = [item for item in scope["windows"]["xuzhou"] if item["id"] in WINDOWS]
    check({item["id"] for item in selected} == set(WINDOWS[1:]), "Missing preregistered observation windows")
    cases = {f"xuzhou__{name}": scope["selected"]["xuzhou"][name] for name in scope["cases"]["xuzhou"]}
    cases.update({f'honghu60__{row["id"]}': row for row in scope["registered_honghu"]["honghu60"]})
    result = dict(schema=POLICY, producer=code["commit"], source_hash=code["source_hash"], parents=links,
                  cases=cases, windows=selected, guards=scope["previous"]["guards"],
                  focus={"xuzhou": [5, 6], "whu-honghu": [15, 21]},
                  foils={scene: {str(k): list(v) for k, v in plan.EXPECTED_CONFUSIONS[scene].items()} for scene in
                         ("xuzhou", "whu-honghu")}, real_updates=0, query_accessed=False, raw_gt_loaded=False)
    return result, scope, baseline, batch


def load_registered(root, directory, code):
    current, scope, baseline, batch = registration(root, code)
    link = pin(directory / "registration.json")
    old = _read(link)
    if current != old:
        check(link["sha256"] == "73f02889bc7caaef481c5c99ece7e9025a73eeb0561d69dd69a0a7c847d39652"
              and old["producer"] == "a1c3cafb41aa6d655bc1d552914bd721ef3c45e7"
              and {**current, "producer": old["producer"], "source_hash": old["source_hash"]} == old,
              "Frozen population or source registration changed")
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", old["producer"], code["commit"]], text=True
        ).splitlines()
        check(set(paths) <= {"scripts/run_coverage_readout.py", "transfer/spectral_spacing_recovery.py",
                             "evaluation/wp12_coverage_readout.py", "tests/test_coverage_readout.py",
                             "transfer/native_scale_observation.py",
                             "docs/architecture/current-status.json"},
              "New code changed the producer of archived frozen evidence")
    return old, scope, baseline, batch


def parent_case(root, batch, case):
    filename = f"{case}.npz"
    metadata, arrays = read_bundle(pin(root / PARENT / "cases-v3" / filename,
                                       batch["artifacts"][filename]["sha256"]))
    check(metadata["scope"] == SCOPE and metadata["optimizer_updates"] == 0
          and metadata["query_accessed"] is metadata["raw_gt_loaded"] is False,
          "Frozen prediction provenance differs")
    return metadata["result"], arrays


def assert_probabilities(values, expected):
    check(values.shape == expected.shape and np.max(np.abs(values - expected)) <= ABS
          and np.array_equal(values.argmax(1), expected.argmax(1)), "Replay differs from frozen probabilities")


def xuzhou(root, scope, reg, case, archived):
    before, saved = archived
    _, source = source_documents(root)
    bank, model, selected = native._configuration(root, scope, source, "xuzhou", case, "cpu")
    arrays, aliases, truth, widths, forwards = defaultdict(list), [], [], {}, 0
    branches, enabled = {}, defaultdict(int)
    stage_times = defaultdict(float)
    start = perf_counter()
    for row in plan.population_rows(bank, "devval"):
        full, _, offsets, labels, tagged, _ = native._reader(bank, row, scope["alias_salt"], "xuzhou")
        aliases.extend(tagged)
        truth.extend(labels)
        views = {"full": full, **{win["id"]: _apply(full.observation, full.bands, tuple(win["retained"]))
                                 for win in reg["windows"]}}
        for win, view in views.items():
            key = win if win == "full" else f"{win}__clean"
            widths[win] = len(view.bands.band_ids)
            opts = (model, "xuzhou", view, offsets, bank.release.classes, selected, "devval")
            tick = perf_counter()
            observed = native._forward(*opts, make_embeddings=True)
            stage_times["native_forward"] += perf_counter() - tick
            tick = perf_counter()
            with spectral_spacing(model, view.bands):
                spaced = native._forward(*opts)
            stage_times["spacing_forward"] += perf_counter() - tick
            tick = perf_counter()
            logits, sides, coefficients = fusion_interventions(model, "xuzhou", observed.features)
            branches[win] = coefficients
            forwards += 2
            enabled[win] += int(view.bands.wavelength_unit == "nm" and view.bands.wavelength_quality in
                                ("verified", "approximate"))
            for head in HEADS:
                arrays[f"{win}__original__{head}"].append(observed.probabilities[head])
                arrays[f"{win}__spacing__{head}"].append(spaced.probabilities[head])
            for name, scores in logits.items():
                arrays[f"{win}__logit__{name}"].append(scores)
            for side in ("cnn", "graph"):
                arrays[f"{win}__contribution__{side}"].append(sides[side])
            arrays[f"{win}__bias"].append(np.broadcast_to(sides["bias"], logits["native"].shape))
            stage_times["branch_readout"] += perf_counter() - tick
    check(aliases == before["aliases"] and truth == before["truth"] and
          state_hash(model) == selected["model"], "Fixed native roster or model changed")
    values = {key: np.concatenate(parts) for key, parts in arrays.items()}
    for win in WINDOWS:
        prefix = win if win == "full" else f"{win}__clean"
        for head in HEADS:
            assert_probabilities(values[f"{win}__original__{head}"], saved[f"{prefix}__{head}"])
            values[f"{win}__archived__{head}"] = saved[f"{prefix}__{head}"]
    return dict(aliases=aliases, truth=truth, sample_n=len(truth), bands=widths, weights=branches,
                eligible_spacing_blocks=dict(enabled), forwards=forwards, model=selected,
                parameters=sum(p.numel() for p in model.parameters()),
                fresh_parameters=0, seconds=perf_counter()-start, stages_seconds=dict(stage_times),
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), values


def honghu_case(root, scope, baseline, entry, archived):
    start = perf_counter()
    old, saved = archived
    ctx = _honghu_context(_read(scope["sources"]["short_scope"]))
    bank, _ = _freeze_bank(root, "whu-honghu", baseline)
    model, states, _, terminal, case = honghu.load_long_model(ctx, entry)
    authority = honghu.populations(ctx, case, scope["alias_salt"])
    check(honghu.serialize_populations(authority) == old["populations"], "Historical HongHu point roles differ")
    model.load_state_dict(states["selected"], strict=True)
    identity = state_hash(model)
    check(identity == entry["endpoint_hashes"]["selected"], "Wrong selected HongHu checkpoint")
    records = terminal["coverage"]["records"]
    scores, features, stage = {}, {}, {}
    for role in ("train", "val"):
        tick = perf_counter()
        scores[role], features[role] = honghu._population_pass(
            model, bank, authority[role], records[role], identity, embed=True)
        stage[f"{role}_forward_seconds"] = perf_counter() - tick
        for head in HEADS:
            assert_probabilities(scores[role][head], saved[f"nominal__selected__{role}__{head}"])
    tick = perf_counter()
    arrays = centroid_reference(features["train"], authority["train"]["labels"], features["val"], 22)
    stage["fit_and_readout_seconds"] = perf_counter() - tick
    for head in HEADS:
        arrays[f"{head}__native"] = scores["val"][head]
        arrays[f"{head}__archived"] = saved[f"nominal__selected__val__{head}"]
        layer = {"fused": model.heads, "cnn": model.cnn_heads, "graph": model.hgnn_heads}[head]["whu-honghu"]
        x = features["val"][head].astype(np.float64)
        weight, bias = layer.weight.detach().numpy().astype(np.float64), layer.bias.detach().numpy().astype(np.float64)
        check(weight.shape == (22, x.shape[1]), "Reference and native readout widths differ")
        arrays[f"{head}__projection"] = x @ weight.T
        arrays[f"{head}__bias"] = bias
    assignment = {}
    for key, mask in old["geometry_groups"].items():
        aliases = [a for block in old["spatial"] for a in block["aliases"]]
        check(len(aliases) == len(mask), "Archived spatial membership is not aligned")
        assignment[key] = [alias for alias, member in zip(aliases, mask, strict=True) if member]
    check(state_hash(model) == identity, "Frozen support readout altered the model")
    return dict(aliases=authority["val"]["aliases"], truth=authority["val"]["labels"].tolist(),
                support_hash=entry["train_population"], support_n=authority["train"]["n"],
                sample_n=authority["val"]["n"], spatial=assignment,
                blocks=[b["aliases"] for b in authority["val"]["block_groups"]],
                forwards=sum(len(role["layout"]) for role in authority.values()), model=identity,
                applied_step=entry["selected"], k=entry["k"], seed=entry["seed"],
                support_seed=entry["support_seed"], seconds=perf_counter()-start, stages_seconds=stage,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                width=features["val"]["fused"].shape[1]), arrays


def run(root, directory, code, selection):
    reg, scope, baseline, batch = load_registered(root, directory, code)
    manifest = pin(directory / "registration.json")
    for name, entry in reg["cases"].items():
        if selection not in ("all", name.split("__", 1)[0], name):
            continue
        path = directory / f"{name}.npz"
        with case_lock(directory / f".{name}.lock"):
            if path.exists():
                meta, _ = read_bundle(pin(path))
                check(meta["registration"] == manifest["sha256"] and meta["case"] == name,
                      "Existing replay receipt has another identity")
                print(f"reused {name}", flush=True)
                continue
            archived = parent_case(root, batch, name)
            lane, case = name.split("__", 1)
            result, arrays = (xuzhou(root, scope, reg, case, archived) if lane == "xuzhou"
                              else honghu_case(root, scope, baseline, entry, archived))
            require_frozen_code(code)
            write_bundle(path, dict(case=name, registration=manifest["sha256"], result=result,
                                   keys=sorted(arrays), real_updates=0, raw_gt_loaded=False,
                                   query_accessed=False), arrays)
            print(f'completed {name} {result["forwards"]} block forwards', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "synthetic", "wp12"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected readout checkout")
    root, directory = external_directory(args.root), external_directory(args.directory)
    check(directory != root and directory.is_relative_to(root), "Replay directory escapes the authorized runs")
    if args.phase == "prepare":
        reg, *_ = registration(root, code)
        directory.mkdir()
        print(_write(directory / "registration.json", reg))
    else:
        reg, *_ = load_registered(root, directory, code)
        if args.phase == "run":
            check(args.case in ("all", "xuzhou", "honghu60", *reg["cases"]), "Unknown frozen readout case")
            run(root, directory, code, args.case)
        elif args.phase == "synthetic":
            from transfer.spectral_spacing_recovery import run_synthetic
            print(run_synthetic(directory, code))
        else:
            from evaluation.wp12_coverage_readout import ingest
            print(ingest(root, directory, code))


if __name__ == "__main__":
    main()

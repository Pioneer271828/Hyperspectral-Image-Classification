"""Independent-space admission, support-only diagnostics, and released WP05 engineering."""

import argparse
from collections import defaultdict
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_coverage_readout import registration as parent_registration, parent_case
from scripts.run_frozen_band_region import (_freeze_bank, _honghu_context, _read, _write, check, pin,
                                           source_documents)
from scripts.run_frozen_mechanisms import case_lock, read_bundle, write_bundle
from transfer import frozen_band_region as region, frozen_mechanism_honghu as hh
from transfer import frozen_mechanism_plan as plan, frozen_mechanism_runner as native
from transfer.development_bank import _gap
from transfer.g3_training import state_hash
from transfer.p4_coverage import ARMS
from transfer.spatial_coverage_protocol import SEED, VIEWS, apply_view, counts, parent_folds, retained_positions

SCHEMA = "wp02_wp05_wp06_guarded_space_readout_v1"
BASES = ("full", "continuous", "dispersed")
SPATIAL = ("clean", "target", "matched")
PINS = {
    "xuzhou_independent": ("p4-spatial-dev-curator01/independent01.json",
                          "cbac31a0c55b934dbfd6716841e1f99e8a96066534fe3991aa9bfcb039fdf97e"),
    "honghu_old_infeasible": ("p4-honghu-dev-feasibility01/receipt.json",
                             "8bfca29627eac24018fd15d784df972fc6efa9c565602e1c0c9b743fbbe6907a"),
    "honghu_cut": ("wp02-honghu-cut-independent01/receipt.json",
                  "c8d37eb318cab74ccd8377a310b46ba56460c453886a2a005f7f5c81c868353a"),
    "honghu_release": ("wp02-honghu-guarded-roles-independent01/receipt.json",
                      "ac5d0e5cf6ea0c43f392cbc1d2b76dfe726499d576d3d2158f16af8035e25827"),
    "paviau_fixed": ("wp02-paviau-final-nov18/preflight/paviaU.json",
                    "61a5264dc021979803cda0d7bd279d9644f8a50505f256aeded876ab39a7b77d"),
    "previous_wp12": ("wp05-wp06-transfer-readout-20261119/wp12/receipt.json",
                      "1e03ae04a95d6c8470f5a3bc31f72dad5894634c753df15d1b539d395134f169"),
}


def rosters(bank):
    return {role: [row for row in bank.role_rows(role) if row["blocked"] is None]
            for role in ("devtrain", "devval")}


def sample_rows(rows, salt, scene):
    return [plan.point_alias(salt, scene, row["role"], index) for row in rows for index, _ in row["samples"]]


def draft(bank, rows):
    old = bank.release
    labels = [r for r in rows["devtrain"] if r["samples"]]
    roster = {str(r["id"]): counts([r], old.classes) for r in labels}
    train_conditions = ("full", "continuous", "dispersed", "shuffled")
    entries = []
    for seed in (64, 512, 2048):
        order = sorted(roster, key=lambda key: digest([seed, key]))
        clock = [[int(key), condition] for condition in train_conditions for key in order]
        exposures = [x * len(train_conditions) for x in counts(labels, old.classes)]
        for arm in ARMS:
            entries.append(dict(model_seed=seed, arm=arm, updates=len(clock), events=[0,len(clock)//2,len(clock)],
                                schedule_sha256=digest(clock), released_exposure=exposures))
    return dict(authorization="future_separate_release_required", status="NO_REAL_UPDATES", bank=bank.pin["sha256"],
                statistics=old.statistics["statistics_hash"], model_seeds=[64,512,2048], arms=list(ARMS),
                budget=sum(x["updates"] for x in entries), source=old.source.get("initialization"),
                shared_source=old.source["shared_state"], validation_views=list(VIEWS), train_views=list(train_conditions),
                role_counts={role: counts(group,old.classes) for role,group in rows.items()}, case_count=len(entries),
                cases=entries, actual_updates=0, selection="first_min_released_point_fused_nll",
                required_for_launch=["exact_fresh_space_release", "actual_source_and_head_manifest",
                                     "same_arithmetic_recovery", "separate_real_update_authority"])


def registration(root, code):
    _, scope, old, batch = parent_registration(root, code)
    pins = {k:pin(root/name, sha) for k,(name,sha) in PINS.items()}
    admitted = {k:_read(p) for k,p in pins.items()}
    check(admitted["previous_wp12"]["case_count"] == 24 and admitted["previous_wp12"]["real_updates"] == 0,
          "Expected the previous WP12 frozen evidence")
    cut = admitted["honghu_cut"]["chosen_geometry_for_followup"]
    check(admitted["honghu_cut"]["status"] == "passed" and admitted["honghu_cut"]["cut_count"] == 38
          and admitted["honghu_cut"]["witness_count"] == 13 and len(cut["train_counts"]) == len(cut["validation_counts"]) == 22
          and cut["cut"] == 180 and cut["blocker_counts"] == [22,21], "HongHu cut ledger differs")
    models = {f"{scene}__{name}": {"scene": scene, "model": scope["selected"][scene][name]}
              for scene in plan.ARCHITECTURES for name in scope["cases"][scene]}
    options = [row for row in scope["registered_honghu"]["honghu60"] if row["arm"] == "none"]
    check(len(options) == 2, "Expected the two frozen HongHu methods")
    models.update({f'whu-honghu__{entry["id"]}': dict(scene="whu-honghu", model=entry) for entry in options})
    spaces, budgets, banks = {}, {}, {}
    for scene in ("xuzhou","paviaU","whu-honghu"):
        bank, _ = _freeze_bank(root, scene, old)
        rows = rosters(bank)
        devval = rows["devval"]
        classes = bank.release.classes
        aliases = sample_rows(devval, scope["alias_salt"], scene)
        if scene == "whu-honghu":
            before, _ = parent_case(root, batch, "honghu60__"+options[0]["id"])
            historical = before["populations"]["val"]["aliases"]
        else:
            name = scope["cases"][scene][0]
            before, _ = parent_case(root, batch, f"{scene}__{name}")
            historical = before["aliases"]
        check(set(aliases) == set(historical) and len(aliases) == classes*5,
              "Unreviewed change to previously released validation points")
        train_parents, val_parents = ({r["parent_tile"] for r in rows[role]} for role in ("devtrain","devval"))
        shared_ids = {i for r in rows["devtrain"] for i,_ in r["samples"]} & {i for r in devval for i,_ in r["samples"]}
        gap = bank.release.custody["minimum_buffer"]
        minimum = min(_gap(a["box"],b["box"]) for a in rows["devtrain"] for b in devval)
        check(not shared_ids and minimum >= gap, "Published train/val point or rectangle overlap")
        spaces[scene] = dict(classes=classes, authority=bank.pin["sha256"], plan=bank.release.plan_hash,
            role_blocks={k:len(v) for k,v in rows.items()}, counts={k:counts(v,classes) for k,v in rows.items()},
            points_seen_validation=len(aliases), previously_seen_intersection=len(set(aliases)&set(historical)),
            previously_unseen_validation=0, independent_release="BLOCKED_NO_ADDITIONAL_RELEASED_LABELS",
            stats=bank.release.statistics["statistics_hash"], stats_role="historical_authorized_training_blocks",
            common_parent_count=len(train_parents & val_parents), buffer=gap, actual_gap=minimum,
            bands={mode: [bank.release.bands.band_ids[i] for i in retained_positions(bank.release.bands,mode)]
                   for mode in VIEWS}, folds=parent_folds(rows["devtrain"],classes,gap))
        budgets[scene] = draft(bank, rows)
        banks[scene] = bank
    check(spaces["whu-honghu"]["common_parent_count"] == 1
          and admitted["honghu_release"]["status"] == "passed", "Shared HongHu class-21 parent ledger differs")
    manifest = dict(schema=SCHEMA, code=code["commit"], source_hash=code["source_hash"], previous=pins,
        cases=models, spaces=spaces, next_budget=budgets, focus=scope["previous"]["focus"],
        guards=scope["previous"]["guards"], foils=scope["competitors"], folds_seed=SEED,
        honghu_feasibility=dict(original_parent_21=47, original_whole_parent_feasible=False,
            guard_cut=180, checked_cuts=38, feasible_cuts=13, train_candidate_counts=cut["train_counts"],
            validation_candidate_counts=cut["validation_counts"], dropped_in_guard=4,
            split_parent_components_in_prior_release=1, scope="previous_audited_cut_not_a_new_label_release"),
        real_updates=0, raw_gt_loaded=False, query_accessed=False)
    return manifest, scope, banks


def registered(root, folder, code):
    entries = registration(root, code)
    source = pin(folder / "registration.json")
    check(_read(source) == entries[0], "Release, model scope, source or geometry changed")
    return source, entries


def training_rows(bank, scope, ctx=None):
    scene = bank.release.scene
    if ctx is None:
        for row in plan.population_rows(bank, "devtrain"):
            full, _, offsets, labels, tagged, tile = native._reader(bank, row, scope["alias_salt"], scene)
            yield row, full, offsets, labels, tagged, tile
        return
    _, authority = ctx
    used = 0
    train = authority["train"]
    for tile, raw in train["layout"]:
        row = bank.role_row("devtrain", tile["id"])
        _, _, full, _ = bank.materialize(row, {"mode":"none"})
        n = len(raw)
        labels = train["labels"][used:used+n].tolist()
        tagged = train["aliases"][used:used+n]
        token = plan.point_alias(scope["alias_salt"],scene,"devtrain/tile",row["id"])
        yield row, full, raw.tolist(), labels, tagged, token
        used += n
    check(used == train["n"], "Partial HongHu support identity")


def run_case(root, reg, scope, bank, scene, name):
    tick = perf_counter()
    if scene == "whu-honghu":
        ctx = _honghu_context(_read(scope["sources"]["short_scope"]))
        entry = reg["cases"][f"{scene}__{name}"]["model"]
        model, states, _, _, case = hh.load_long_model(ctx,entry)
        model.load_state_dict(states["selected"], strict=True)
        auth = hh.populations(ctx,case,scope["alias_salt"])
        check(auth["train"]["authority"] == entry["train_population"], "HongHu support differs")
        selected = dict(model=entry["endpoint_hashes"]["selected"], step=entry["selected"], seed=entry["seed"],
                        support_seed=entry["support_seed"], k=entry["k"], method=entry["method"])
        extra = (ctx, auth)
    else:
        _, docs = source_documents(root)
        admitted, model, selected = native._configuration(root,scope,docs,scene,name,"cpu")
        check(admitted.pin == bank.pin, "Wrong scene release")
        extra = None
    data = defaultdict(list)
    truth, aliases, tiles, actual_conditions, traces = [], [], [], {}, []
    for row, full, offsets, labels, tagged, alias_block in training_rows(bank, scope, extra):
        truth.extend(labels)
        aliases.extend(tagged)
        tiles.extend([row["id"]]*len(labels))
        boxes = region.region_boxes(full.observation.spectral.shape[:2],offsets,labels,reg["focus"][scene])
        for band in BASES:
            for mode in SPATIAL:
                view, positions, _ = apply_view(full,bank.release.bands,band,boxes.get(mode))
                actual_conditions[band] = list(view.bands.band_ids)
                scores = native._forward(model,scene,view,offsets,bank.release.classes,selected,"devtrain",
                                         make_embeddings=scene == "whu-honghu" and mode == "clean")
                for head, array in scores.probabilities.items():
                    data[f"{band}__{mode}__{head}"].append(array)
                if scores.features:
                    for head,array in scores.features.items():
                        data[f"{band}__embedding__{head}"].append(array)
                traces.append(dict(tile=alias_block, band=band, spatial=mode, pixels=int(np.prod(
                    view.observation.spectral.shape[:2])), retained=len(positions), graph=scores.trace["graph"],
                    input=scores.trace["observation"], normalized_source="historical_training_statistics"))
    check(state_hash(model) == selected["model"] and len(set(aliases)) == len(truth) == len(aliases),
          "Frozen case model or point roster changed")
    arrays = {k:np.concatenate(v) for k,v in data.items()}
    return dict(scene=scene, model=selected, aliases=aliases, labels=truth, tiles=tiles, band_ids=actual_conditions,
                traces=traces, forwards=len(traces), seconds=perf_counter()-tick, rss_peak_kib=resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss, role="historical_devtrain_support", independent_accuracy=False), arrays


def run(root, folder, code, select):
    link,(reg,scope,banks) = registered(root,folder,code)
    for label in reg["cases"]:
        scene,name = label.split("__",1)
        if select not in ("all",scene,label):
            continue
        path = folder / f"{label}.npz"
        with case_lock(folder / f".{label}.lock"):
            if path.exists():
                record,_ = read_bundle(pin(path))
                check(record["registration"] == link["sha256"] and record["case"] == label, "Foreign saved case")
                continue
            value, arrays = run_case(root,reg,scope,banks[scene],scene,name)
            require_frozen_code(code)
            write_bundle(path,dict(case=label,registration=link["sha256"],result=value,raw_gt_loaded=False,
                                   query_accessed=False,real_updates=0),arrays)
            print(label,value["forwards"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare","inspect","run","synthetic","wp12","start-real"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--scene",default="all")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected spatial checkout")
    root,out = external_directory(args.root),external_directory(args.directory)
    check(out.is_relative_to(root) and out != root, "Spatial output must stay inside authorized runs")
    if args.phase == "start-real":
        raise ValueError("Real optimization requires a separately authorized execution release")
    if args.phase == "synthetic":
        from transfer.spatial_coverage_training import run_synthetic
        print(run_synthetic(out,code))
    elif args.phase == "inspect":
        manifest, *_ = registration(root,code)
        print({name:value["independent_release"] for name,value in manifest["spaces"].items()})
    elif args.phase == "prepare":
        manifest, *_ = registration(root,code)
        out.mkdir()
        print(_write(out/"registration.json",manifest))
    elif args.phase == "run":
        run(root,out,code,args.scene)
    else:
        from evaluation.wp12_spatial_coverage import publish
        print(publish(root,out,code))


if __name__ == "__main__":
    main()

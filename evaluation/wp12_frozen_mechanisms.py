"""Strict WP12 replay of the released band-window, graph and HongHu evidence."""

from collections import defaultdict
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from statistics import fmean

import numpy as np

from evaluation.frozen_band_region import HEADS, paired
from evaluation.frozen_mechanism import (
    _margin, apply_temperature, calibration, contrasts,
    fit_algebraic_temperature, interaction, leave_one, point_scores, validate_probabilities,
)
from model.research_config import digest
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import LANES, _case_form, read_bundle
from transfer import frozen_mechanism_plan as plan
from transfer.development_blocks import array_hash


GROUPS = ("in_target_box", "outside_sharing_native_q", "outside_other_native_q")
CASE_FIELDS = {"scope", "lane", "name", "producer", "optimizer_updates", "query_accessed", "raw_gt_loaded",
               "result", "finished", "elapsed", "array_keys", "rss_peak_kib"}
POINT_FIELDS = ("scene", "case", "protocol", "role", "stage", "condition", "cell", "head", "point", "truth",
                "group", "evidence", "prediction", "true_probability", "rank", "best_nontrue", "top_probability",
                "truth_minus_best", "foils", "calibrated_probability", "calibrated_truth_minus_best",
                "calibrated_foils")
METRIC_FIELDS = ("scene", "case", "protocol", "role", "window", "contrast", "evidence", "head", "group", "n",
                 "source_oa", "target_oa", "delta_oa", "source_aa", "target_aa", "delta_aa", "flip",
                 "correct_to_error", "error_to_correct", "source_nll", "target_nll", "cal_source_nll", "cal_target_nll",
                 "source_ece", "target_ece", "cal_source_ece", "cal_target_ece",
                 "source_brier", "target_brier", "cal_source_brier", "cal_target_brier")
CLASS_FIELDS = ("scene", "case", "protocol", "role", "window", "contrast", "evidence", "head", "group",
                "class_id", "denominator", "source_correct", "target_correct", "source_recall", "target_recall",
                "correct_to_error", "error_to_correct", "source_predictions", "target_predictions",
                "lost_to", "gained_from", "transitions", "delta_margin", "delta_calibrated_margin")
DECISION_FIELDS = ("scene", "case", "window", "head", "class_id", "denominator",
                   "graph_only_delta_recall", "feature_only_delta_recall", "joint_delta_recall",
                   "graph_only_delta_margin", "feature_only_delta_margin", "interaction_delta_margin",
                   "native_band_protection", "supported_effect")


def _csv_value(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True) if isinstance(
        value, (dict, list, tuple)) else value


@dataclass
class Tables:
    point: object
    metric: object
    classes: object
    decision: object

    def metrics(self, identity, row):
        values = row["metrics"]
        keys = {key: row[key] for key in ("window", "contrast", "evidence", "head", "group")}
        base = {**identity, **keys}
        cal_rows = {}
        if row.get("before") is not None:
            for field in ("nll", "ece10", "brier"):
                key = "ece" if field == "ece10" else field
                for prefix, pair in (("", row["before"]), ("cal_", row["after"] or [{}, {}])):
                    cal_rows.update({f"{prefix}source_{key}": pair[0].get(field),
                                     f"{prefix}target_{key}": pair[1].get(field)})
        names = ("source_oa", "target_oa", "delta_oa", "source_aa", "target_aa", "delta_aa",
                 "correct_to_error", "error_to_correct")
        self.metric.writerow({**base, **{key: values[key] for key in names}, **cal_rows,
                              "n": values["denominator"], "flip": values["flips"]})
        for cls in values["classes"]:
            self.classes.writerow({**base, **{key: _csv_value(value) for key, value in cls.items()}})
        return len(values["classes"])

    def points(self, identity, aliases, labels, values, *, focus, foils, group, temperature=None):
        before = point_scores(values, labels, foils)
        after = point_scores(apply_temperature(values, temperature), labels, foils) if temperature else before
        count = 0
        for index, (alias, label, old, new) in enumerate(zip(aliases, labels, before, after, strict=True)):
            if label not in focus:
                continue
            check(old["predicted"] == new["predicted"] and old["rank"] == new["rank"],
                  "Diagnostic temperature changed probability rank or hard prediction")
            self.point.writerow(dict(
                **identity, point=alias, truth=label, group=group[index], prediction=old["predicted"],
                true_probability=old["true_probability"], rank=old["rank"], best_nontrue=old["best_nontrue"],
                top_probability=old["top_probability"], truth_minus_best=old["true_minus_best"],
                foils=_csv_value(old["foils"]), calibrated_probability=new["true_probability"] if temperature else None,
                calibrated_truth_minus_best=new["true_minus_best"] if temperature else None,
                calibrated_foils=_csv_value(new["foils"]) if temperature else None))
            count += 1
        return count


def check_aliases(aliases, truth, count, classes):
    check(len(aliases) == len(set(aliases)) == len(truth) == count and
          all(re.fullmatch(r"[a-f0-9]{32}", value) for value in aliases) and
          all(type(x) is int and 1 <= x <= classes for x in truth),
          "Private coordinates, incompatible true class or non-unique point identity")


def _digest_trace(data, trace, expected_source, expected_features=None):
    check(set(trace) == {"observation", "metadata", "geometry", "source_observation", "graph",
                         "source_graph_features", "feature_input", "projections", "probabilities"}
          and set(trace["graph"]) == {"q", "a", "positions", "q_nnz", "a_nnz", "wide_a", "wide_nnz", "range_weights"}
          and trace["graph"] == expected_source["graph"]
          and trace["geometry"] == expected_source["geometry"]
          and trace["source_observation"] == expected_source["observation"]
          and trace["source_graph_features"] == expected_source["feature_input"]
          and trace["projections"]["clean"] == trace["feature_input"],
          "Stored graph provenance, actual support or projection identity differs")
    if expected_features is not None:
        check(trace["feature_input"] == expected_features["feature_input"] and
              trace["observation"] == expected_features["observation"] and
              trace["metadata"] == expected_features["metadata"], "Graph swap changed retained feature identity")
    for head in HEADS:
        check(array_hash(data[head]) == trace["probabilities"][head], "Point score bytes differ from graph trace")


def _window_inputs(scope, lane, case, meta, arrays):
    old = meta["result"]
    n = 45
    check_aliases(old["aliases"], old["truth"], n, 9)
    check(old["selected"] == scope["selected"][lane][case] and old["device"] == "cpu"
          and old["observation_scope"] == "devval" and old["hashes"] == {
              "windows": digest(scope["windows"][lane]), "roles": digest(old["aliases"])}
          and set(old["negative"]) == {"foreign_graph_rejected", "deleted_poison_equal",
                                      "reordered_metadata_distinct", "full_and_retained_cache_separate",
                                      "feature_input_changes_with_target"}
          and all(old["negative"].values()), "Window runtime identity, control or role differs")
    check_aliases(old["train"]["aliases"], old["train"]["labels"], n, 9)
    check(not set(old["train"]["aliases"]).intersection(old["aliases"]), "Calibration support leaks validation")
    windows = scope["windows"][lane]
    expected = {f"full__{h}" for h in HEADS} | {f"calibration_train__{h}" for h in HEADS} | {
        f"{w['id']}__{cell}__{head}" for w in windows for cell in plan.ALL_CELLS for head in HEADS}
    check(set(arrays) == expected and set(old["group_masks"]) == set(GROUPS) and
          all(len(mask) == n for mask in old["group_masks"].values()) and
          len(old["random_available"]) == n and len(old["trace"]) == (5 if lane == "xuzhou" else 17),
          "Incomplete window, group or probability inventory")
    masks = {key: np.asarray(value, dtype=bool) for key, value in old["group_masks"].items()}
    check((sum(mask.astype(int) for mask in masks.values()) == 1).all() and
          {k: int(mask.sum()) for k, mask in masks.items()} == old["group_count"],
          "Missing or overlapping spatial group")
    for head in HEADS:
        expected_temp = fit_algebraic_temperature(arrays[f"calibration_train__{head}"], old["train"]["labels"])
        check(expected_temp == old["temperatures"][head], "Calibration was not the fixed train-only calculation")
    structured = {"full": {head: validate_probabilities(arrays[f"full__{head}"], n, 9) for head in HEADS}}
    for window in windows:
        identifier = window["id"]
        structured[identifier] = {cell: {h: validate_probabilities(arrays[f"{identifier}__{cell}__{h}"], n, 9)
                                        for h in HEADS} for cell in plan.ALL_CELLS}
    cursor = 0
    for block in old["trace"]:
        length = len(block["aliases"])
        section = slice(cursor, cursor + length)
        check(old["aliases"][section] == block["aliases"] and set(block["windows"]) ==
              {window["id"] for window in windows}, "Block sweep reordered the fixed points or omitted a window")
        full = block["full"]
        _digest_trace({head: arrays[f"full__{head}"][section] for head in HEADS}, full, full, full)
        for ident, cells in block["windows"].items():
            check(set(cells) == set(plan.ALL_CELLS), "Unregistered or absent graph/feature experiment cell")
            sources = {
                "clean_target_graph": ("target", "clean"),
                "target_clean_graph": ("clean", "target"),
                "clean_full_graph": ("full", "clean"),
                "target_full_graph": ("full", "target"),
            }
            for cell, trace in cells.items():
                source_name, feature_name = sources.get(cell, (cell, cell))
                source = full if source_name == "full" else cells[source_name]
                data = {head: arrays[f"{ident}__{cell}__{head}"][section] for head in HEADS}
                _digest_trace(data, trace, source, cells[feature_name])
            check([block["random_available"]] * length == old["random_available"][section],
                  "Spatial random control availability differs")
        cursor += length
    check(cursor == n, "Traces did not consume the released population")
    return structured, masks


def _delta(row):
    return row["target_recall"] - row["source_recall"] if row["denominator"] else None


def _decision(scene, case, rows, matrix, truth, scope, foils, tables):
    focus = plan.FOCUS[scene]
    rules = scope["previous"]["guards"]
    indexed = {(r["window"], r["contrast"], r["head"]): r["metrics"] for r in rows if r["group"] == "all"}
    interesting = []
    for window in scope["windows"][scene]:
        identifier = window["id"]
        inter = interaction(truth, matrix[identifier], foils)
        for head in HEADS:
            native = indexed[identifier, "missing_vs_full", head]
            graph, feat, joint = [indexed[identifier, name, head]["classes"] for name in (
                "graph_only_clean_features", "feature_only_clean_graph", "joint_vs_clean")]
            for cls in focus:
                g, f, j = (items[cls-1] for items in (graph, feat, joint))
                dg, df, dj = (_delta(item) for item in (g, f, j))
                cutoff = rules["focus_min_delta_recall"]
                if not j["denominator"]:
                    verdict = "empty"
                elif dg < cutoff and df < cutoff:
                    verdict = "both_components"
                elif dg < cutoff:
                    verdict = "graph_component"
                elif df < cutoff:
                    verdict = "feature_component"
                elif dj < cutoff:
                    verdict = "joint_only_at_cutoff"
                else:
                    verdict = "no_adverse_effect_at_cutoff"
                guard = (native["delta_oa"] >= rules["aggregate_min_delta"]
                         and native["delta_aa"] >= rules["aggregate_min_delta"]
                         and all(_delta(native["classes"][k-1]) >= cutoff
                                 for k in scope["previous"]["focus"][scene]))
                item = dict(scene=scene, case=case, window=identifier, head=head, class_id=cls, denominator=j["denominator"],
                            graph_only_delta_recall=dg, feature_only_delta_recall=df, joint_delta_recall=dj,
                            graph_only_delta_margin=g["delta_margin"],feature_only_delta_margin=f["delta_margin"],
                            interaction_delta_margin=inter[head][str(cls)]["mean"],
                            native_band_protection=guard, supported_effect=verdict)
                tables.decision.writerow(item)
                interesting.append(item)
    return interesting


def _native(root, scope, lane, case, meta, arrays, tables):
    values, masks = _window_inputs(scope, lane, case, meta, arrays)
    result = meta["result"]
    truth = np.asarray(result["truth"], dtype=np.int64)
    temperatures = result["temperatures"]
    foils = plan.EXPECTED_CONFUSIONS[lane]
    focus = plan.FOCUS[lane]
    by_group = [""] * len(truth)
    for group, member in masks.items():
        for index in np.flatnonzero(member):
            by_group[index] = group
    windows = scope["windows"][lane]
    ident = dict(scene=lane, case=case, protocol="paired_graph_window_cpu", role="devval")
    all_metrics = contrasts(windows, truth, values, temperatures, foils, masks)
    random_mask = np.asarray(result["random_available"], dtype=bool)
    if not random_mask.all():
        possible = [row for row in all_metrics if row["contrast"] != "target_vs_random"]
        masked_values = {key: {head: array[random_mask] for head, array in matrix.items()} if key == "full"
                         else {cell: {head: array[random_mask] for head, array in observed.items()}
                               for cell, observed in matrix.items()} for key, matrix in values.items()}
        possible.extend(row for row in contrasts(windows, truth[random_mask], masked_values, temperatures, foils,
                                                {k: v[random_mask] for k, v in masks.items()})
                        if row["contrast"] == "target_vs_random")
        all_metrics = possible
    class_rows = point_rows = 0
    for row in all_metrics:
        class_rows += tables.metrics(ident, row)
    controls = defaultdict(list)
    for window in windows:
        if window["seed"] is not None:
            controls[window["width"]].append(window)
    random_comparisons = 0
    for window in windows:
        if window["seed"] is not None:
            continue
        for control in controls[window["width"]]:
            series = {"full": values[control["id"]]["clean"],
                      window["id"]: {"clean": values[window["id"]]["clean"]}}
            for row in contrasts([window], truth, series, temperatures, foils, {}):
                row.update(contrast=f"window_vs_fixed_random_{control['seed']}", evidence="native_same_count_control")
                class_rows += tables.metrics(ident, row)
                random_comparisons += 1
    point_groups = {"full": {"full": values["full"]},
                    **{key: value for key, value in values.items() if key != "full"}}
    for condition, cells in point_groups.items():
        for cell, heads in cells.items():
            mask = random_mask if cell == "random" else np.ones(len(truth), dtype=bool)
            selected_aliases = np.asarray(result["aliases"])[mask].tolist()
            chosen_groups = np.asarray(by_group)[mask].tolist()
            for head, probability in heads.items():
                evidence = "counterfactual_full_information" if cell in plan.FULL_INFORMATION else "observed_or_mechanism"
                point_rows += tables.points(
                    dict(**ident, stage="selected", condition=condition, cell=cell, head=head, evidence=evidence),
                    selected_aliases, truth[mask].tolist(), probability[mask], foils=foils, focus=focus,
                    group=chosen_groups, temperature=temperatures[head]["temperature"])
    decisions = _decision(lane, case, all_metrics, values, truth, scope, foils, tables)
    projections = [dict(window=r["window"], head=r["head"], delta_oa=r["metrics"]["delta_oa"],
                        delta_aa=r["metrics"]["delta_aa"],
                        n=r["metrics"]["denominator"], from_cal_ece=r["before"][1]["ece10"],
                        to_cal_ece=r["after"][1]["ece10"])
                   for r in all_metrics if r["contrast"] == "missing_vs_full" and r["group"] == "all"]
    return dict(scene=lane, case=case, n=len(truth), points=point_rows, class_rows=class_rows,
                comparisons=len(all_metrics)+random_comparisons, windows=len(windows), group_count=result["group_count"],
                plot_values=projections, decisions=decisions, temperatures=temperatures,
                eligible_random_n=int(random_mask.sum()), model_seed=result["model_seed"],
                time=result["runtime_seconds"], rss_peak_kib=result["rss_peak_kib"]), result["aliases"]


def _honghu_population(result, registry):
    roles = result["populations"]
    check(set(roles) == {"train", "val"}, "HongHu requires both authorized role populations")
    for role, budget, bound in (("train", registry["k"], "train_population"), ("val", 5, "population")):
        data = roles[role]
        check_aliases(data["aliases"], data["labels"], 22*budget, 22)
        check(data["authority"] == registry[bound] and data["role"] == f"dev{role}"
              and [a for block in data["block_groups"] for a in block["aliases"]] == data["aliases"],
              "HongHu population authority, grouping or support denominator changed")
    check(not set(roles["train"]["aliases"]).intersection(roles["val"]["aliases"]),
          "HongHu train and validation identities overlap")
    return roles


def _honghu(scope, lane, case, meta, arrays, tables):
    registry = next(row for row in scope["registered_honghu"][lane] if row["id"] == case)
    result = meta["result"]
    roles = _honghu_population(result, registry)
    is_short = (lane == "honghu24")
    stages = ("initial", "selected", "last")
    keys = {f"nominal__{stage}__{role}__{head}" for stage in stages for role in ("train", "val") for head in HEADS}
    keys.update(f"full__{stage}__val__{head}" for stage in (("selected",) if is_short else stages) for head in HEADS)
    if registry["spatial"]:
        keys.update(f"spatial__{kind}__{head}" for kind in ("clean", "target", "matched") for head in HEADS)
    check(keys == set(arrays) and (result["forwards"] == 0) == is_short
          and (result["feature_readout"] is not None) == registry["spatial"],
          "HongHu archive or replay stages mixed incompatible training protocols")
    shape_values, count_points, count_classes, count_comparisons, readouts = {}, 0, 0, 0, []
    focus = plan.FOCUS["whu-honghu"]
    foils = plan.EXPECTED_CONFUSIONS["whu-honghu"]
    exposed = dict(scene="whu-honghu", case=case, protocol=registry["protocol"])
    for key in sorted(k for k in keys if not k.startswith("spatial")):
        observation, stage, role, head = key.split("__")
        subjects = roles[role]
        labels, aliases = subjects["labels"], subjects["aliases"]
        probabilities = validate_probabilities(arrays[key], len(labels), 22)
        shape_values[key] = probabilities
        block_map = {item: block["block"] for block in subjects["block_groups"] for item in block["aliases"]}
        count_points += tables.points(dict(**exposed, role=subjects["role"], condition=observation,
                                           stage=stage, cell="natural", head=head, evidence="frozen_point_scores"),
                                      aliases, labels, probabilities, focus=focus, foils=foils,
                                      group=[block_map[key] for key in aliases])
        pool = point_scores(probabilities, labels, foils)
        for cls in focus:
            rows = [row for row, truth in zip(pool, labels, strict=True) if cls == truth]
            readouts.append(dict(condition=observation, stage=stage, role=role, head=head, class_id=cls,
                n=len(rows), correct=sum(x["predicted"] == cls for x in rows),
                rank_mean=fmean(x["rank"] for x in rows),
                true_probability_mean=fmean(x["true_probability"] for x in rows),
                margin_mean=fmean(x["true_minus_best"] for x in rows),
                confusions=dict(sorted({pred: sum(x["predicted"] == pred for x in rows)
                                        for pred in {x["predicted"] for x in rows}}.items()))))

    def compare(condition, name, a, b, labels, role, group="all"):
        nonlocal count_classes, count_comparisons
        for head in HEADS:
            source, target = a[head], b[head]
            metrics = paired(target.argmax(1)+1, source.argmax(1)+1, labels, 22)
            for cls in metrics["classes"]:
                if cls["denominator"]:
                    subset = np.asarray(labels) == cls["class_id"]
                    truth = np.asarray(labels)[subset]
                    cls["delta_margin"] = float((_margin(target[subset], truth, foils)
                                                   - _margin(source[subset], truth, foils)).mean())
            count_classes += tables.metrics(dict(**exposed, role=role), dict(
                window=condition, contrast=name, head=head, group=group, evidence="honghu_frozen_descriptive",
                metrics=metrics, before=[calibration(source, labels), calibration(target, labels)], after=None))
            count_comparisons += 1

    for condition in ("nominal", "full"):
        for role in roles:
            for a, b in (("initial", "selected"), ("selected", "last"), ("initial", "last")):
                if f"{condition}__{a}__{role}__fused" not in shape_values or (
                    f"{condition}__{b}__{role}__fused" not in shape_values
                ):
                    continue
                lookup = [{h: shape_values[f"{condition}__{stage}__{role}__{h}"] for h in HEADS} for stage in (a, b)]
                compare(condition, f"{b}_vs_{a}", *lookup, roles[role]["labels"], roles[role]["role"])
    if registry["spatial"]:
        aliases = [a for block in result["spatial"] for a in block["aliases"]]
        labels = [label for block in result["spatial"] for label in block["labels"]]
        lookup = dict(zip(roles["val"]["aliases"], roles["val"]["labels"], strict=True))
        check(all(lookup[a] == c for a, c in zip(aliases, labels, strict=True)) and set(result["geometry_groups"]) == set(GROUPS),
              "HongHu spatial controls escaped the released population")
        masks = {key: np.asarray(mask, dtype=bool) for key, mask in result["geometry_groups"].items()}
        check(all(len(mask) == len(aliases) for mask in masks.values())
              and (sum(m.astype(int) for m in masks.values()) == 1).all(), "HongHu spatial groups are incomplete")
        spatial = {kind: {h: validate_probabilities(arrays[f"spatial__{kind}__{h}"], len(labels), 22) for h in HEADS}
                   for kind in ("clean", "matched", "target")}
        for key, mask in {"all": np.ones(len(labels), dtype=bool), **masks}.items():
            for a, b in (("clean", "target"), ("clean", "matched"), ("matched", "target")):
                compare("full_spatial", f"{b}_vs_{a}", {h: spatial[a][h][mask] for h in HEADS},
                        {h: spatial[b][h][mask] for h in HEADS}, np.asarray(labels)[mask].tolist(), "devval", key)
        chosen_groups = [next(key for key in masks if masks[key][i]) for i in range(len(labels))]
        for cell, heads in spatial.items():
            for head, probability in heads.items():
                count_points += tables.points(
                    dict(**exposed, role="devval", condition="full_spatial", stage="selected", cell=cell,
                         head=head, evidence="honghu_matched_region"), aliases, labels, probability,
                    focus=focus, foils=foils, group=chosen_groups)
    return dict(scene="whu-honghu", case=case, protocol=registry["protocol"], seed=registry["seed"],
                selected=registry["selected"], k=registry["k"], method=registry["method"], budget=registry["budget"],
                points=count_points, class_rows=count_classes, comparisons=count_comparisons, readouts=readouts,
                feature_readout=result["feature_readout"], groups={k: sum(v) for k, v in result["geometry_groups"].items()},
                existing_cuda_max=result.get("nominal_cpu_cuda_probability_max_abs"),
                existing_cuda_flips=result.get("nominal_cpu_cuda_hard_flips")), roles["val"]["aliases"]


def _seed_boundary(scope, details):
    native = [row for row in details if row["scene"] in plan.ARCHITECTURES]
    within, by_end = defaultdict(dict), defaultdict(dict)
    for row in native:
        scene, case = row["scene"], row["case"]
        if scene == "paviaU":
            for metric in row["plot_values"]:
                key = (metric["window"], metric["head"])
                within[key][case] = metric["delta_oa"]
        if scene != "paviaU" and case != "off":
            continue
        views = {x["id"]: x for x in scope["windows"][scene]}
        for metric in row["plot_values"]:
            window = views[metric["window"]]
            if "end" not in window["families"]:
                continue
            key = ("quarter" if window["width"] < (103 if scene == "paviaU" else 436)//2 else "half",
                   "low" if window["first"] == 0 else "high", metric["head"])
            by_end[key].setdefault(scene, []).append(metric["delta_oa"])
    return dict(pavia=[
        dict(window=key[0], head=key[1], **leave_one(values)) for key, values in sorted(within.items())],
        matched_end_descriptive=[
            dict(width=key[0], end=key[1], head=key[2], scene_means={k: fmean(v) for k, v in values.items()},
                 **leave_one({k: fmean(v) for k, v in values.items()})) for key, values in sorted(by_end.items())],
        limits=["Xuzhou architectures reuse one seed", "HongHu uses a different baseline protocol and is held out",
                "Remaining-scene estimates are descriptive and never model selection"])


def ingest(root, scope_link, batch_link, case_dir, output, commit, device_link=None):
    for path in (scope_link["path"], batch_link["path"], case_dir, output):
        check(Path(path).resolve().is_relative_to(root), "WP12 path escapes authorized runs")
    scope, batch = _read(scope_link), _read(batch_link)
    check(scope["schema"] == batch["schema"] == plan.POLICY and batch["status"] == "complete" and
          batch["scope"] == scope_link["sha256"] and batch["commit"] == scope["code"]["commit"] and
          batch["query_accessed"] is batch["raw_gt_loaded"] is scope["model_selection"] is False and
          scope["optimizer_updates"] == batch["optimizer_updates"] == 0, "WP12 refuses altered or training evidence")
    forms = {_case_form(lane, name): (lane, name) for lane in LANES for name in scope["cases"][lane]}
    check(len(forms) == batch["cases"] == 56 and set(forms) == set(batch["artifacts"]),
          "WP12 requires the complete registered case matrix")
    output.mkdir(exist_ok=False)
    specs = {"point": ("point-scores.csv", POINT_FIELDS), "metric": ("paired-metrics.csv", METRIC_FIELDS),
             "classes": ("paired-classes.csv", CLASS_FIELDS), "decision": ("mechanism-decisions.csv", DECISION_FIELDS)}
    details, populations = [], {}
    with ExitStack() as stack:
        writers = {}
        for name, (relative, fields) in specs.items():
            handle = stack.enter_context((output / relative).open("x", encoding="utf-8", newline=""))
            writers[name] = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writers[name].writeheader()
        tables = Tables(**writers)
        for filename, (lane, case) in forms.items():
            meta, arrays = read_bundle(pin(case_dir / filename, batch["artifacts"][filename]["sha256"]))
            check(set(meta) == CASE_FIELDS and (meta["scope"], meta["lane"], meta["name"], meta["producer"]) ==
                  (scope_link["sha256"], lane, case, batch["commit"]) and
                  meta["query_accessed"] is meta["raw_gt_loaded"] is False and meta["optimizer_updates"] == 0 and
                  meta["array_keys"] == sorted(arrays), "Missing field, data access or mixed protocol receipt")
            row, identities = (_native(root, scope, lane, case, meta, arrays, tables) if lane in plan.ARCHITECTURES
                               else _honghu(scope, lane, case, meta, arrays, tables))
            check(populations.setdefault(row["scene"], identities) == identities,
                  "Case is no longer on the same released point ordering")
            details.append(row)
    artifacts = {name: pin(output / name)["sha256"] for name, _ in specs.values()}
    results = _write(output / "scientific-summary.json", dict(details=details, boundary=_seed_boundary(scope, details),
        populations={scene: dict(n=len(ids), alias_sha256=digest(ids)) for scene, ids in populations.items()},
        registered_honghu=scope["eligible_honghu"], holds=dict(optimization=0, query=0, raw_gt=0)))
    artifacts["scientific-summary.json"] = results["sha256"]
    device = _read(device_link) if device_link else None
    if device:
        check(device["scope"] == scope_link["sha256"] and device["optimizer_updates"] == 0 and
              device["query_accessed"] is device["raw_gt_loaded"] is False, "Device experiment scope differs")
    return _write(output / "receipt.json", dict(
        schema="wp12_frozen_band_graph_mechanism_v1", status="accepted_development_only",
        scope=scope_link["sha256"], batch=batch_link["sha256"], producer=batch["commit"], intake=commit,
        case_count=len(details), point_score_rows=sum(row["points"] for row in details),
        comparative_rows=sum(row["comparisons"] for row in details),
        class_rows=sum(row["class_rows"] for row in details),
        physical_forwards=sum(row["forwards"] for row in batch["artifacts"].values()),
        query_accessed=False, raw_gt_loaded=False, optimizer_updates=0, model_selection=False,
        full_band_graph_effects="counterfactual_excluded_from_native_performance",
        device_comparison=None if device is None else dict(status=device["status"], cases=len(device["rows"])),
        artifacts=artifacts))

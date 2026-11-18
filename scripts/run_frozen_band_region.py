"""Scope, run/resume, and aggregate the WP05/WP06 frozen development supplement."""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS, comparisons, hard, summarize
from loadData.observation_sampling import sample
from model.research_config import canonical_json, digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from train_spatial import SpatialConfig
from transfer import frozen_band_region as runtime
from transfer import p4_frozen_diagnostics as honghu
from transfer import registered_development as custody
from transfer.development_bank import _atomic_new
from transfer.g3_training import state_hash
from transfer.native_released_preflight import open_bank, target_head_baseline
from transfer.p4_spatial import initialize as init_honghu, SEED as HONGHU_SEED
from transfer.spatial_multiscale import initialize as init_native
from transfer.xuzhou_multiscale import ARMS as XUZHOU_ARMS, SEED as XUZHOU_SEED, read_stage
from transfer.xuzhou_multiscale_audit import _packet


POLICY = "wp05_wp06_frozen_band_region_v1"
SCENES = ("whu-honghu", "xuzhou", "paviaU")
FOCUS = {"whu-honghu": [15, 21], "xuzhou": [5, 6], "paviaU": [2, 6]}
PAVIA_PRIOR = "wp07-paviau-real-results-20261118.TC3eXtC5/verified-replays"
PINS = {
    "hh_scope": ("wp05-frozen-observation-registration01/scope.json",
                 "bdde8bc27c37af0ed57cdf1486f2168cd3974f80060a0ed49565a533a2e067ea"),
    "hh_predictions": ("wp05-frozen-observation-replay01/receipt.json",
                       "32c69dbf6cedd982e7b62b34eb1b7777473a8e7599cabe7820215d0454ad0bc7"),
    "hh_audit": ("wp05-frozen-observation-audit01/receipt.json",
                 "6f519a6d57a4a2f6a8319ce30455a9de257b387679afaa4c8f691aa715b7f1c2"),
    "xz_audit": ("wp07-xuzhou-multiscale-real01/audit.json",
                 "29ed4f67bf08f13070c20fec21ecb77f834a55778c5a090121737dcba0705a32"),
    "pavia_bank": ("wp02-paviau-bank-nov18.json",
                   "54c87a5f5d4a84bc4d6e4055dbf8874f633874dce7a01d82d79a66488073c8a4"),
    "pavia_64": (f"{PAVIA_PRIOR}/paviaU-64-1-off.json",
                 "7e3d2ea6deb421d0870305bb27048a8a1f53cc0734b902e31ff8fc2b91567e41c"),
    "pavia_512": (f"{PAVIA_PRIOR}/paviaU-512-1-off.json",
                  "f59090454ae87b14692c4610f4e402c6064551be718af6ed1ff31871c260fe18"),
    "pavia_2048": (f"{PAVIA_PRIOR}/paviaU-2048-1-off.json",
                   "88bccbe84e27d7f583f1ad216764b2c58fc8fadfd527f03415254fea5ff8fca0"),
}
ARCHIVE_NAMES = {"full": "full", "contiguous": "low_half", "scattered": "random50",
                 "contiguous_opposite_end": "high_half"}
BASE_PAIRS = {"contiguous_vs_full": ("contiguous", "full"),
              "scattered_vs_full": ("scattered", "full"),
              "contiguous_vs_scattered": ("contiguous", "scattered")}
REGION_PAIRS = {
    "target_vs_clean_full": ("full_target", "full"),
    "matched_vs_clean_full": ("full_matched", "full"),
    "target_vs_matched_full": ("full_target", "full_matched"),
    "target_vs_clean_contiguous": ("contiguous_target", "contiguous"),
    "matched_vs_clean_contiguous": ("contiguous_matched", "contiguous"),
    "target_vs_matched_contiguous": ("contiguous_target", "contiguous_matched"),
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, sha=None):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hash_file(path) if sha is None else sha)


def source_documents(root):
    links = {key: pin(root / name, sha) for key, (name, sha) in PINS.items()}
    return links, {key: _read(link) for key, link in links.items()}


def _read(link):
    return json.loads(custody._read(link))


def _state(link):
    return torch.load(io.BytesIO(custody._read(link)), map_location="cpu", weights_only=True)


def _write(path, value):
    _atomic_new(path, (canonical_json(value) + "\n").encode())
    return pin(path)


def _time():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, out, code):
    links, docs = source_documents(root)
    hh, scores, audited = (docs[k] for k in ("hh_scope", "hh_predictions", "hh_audit"))
    check(audited["status"] == scores["status"] == "passed" and audited["cases"] == len(hh["cases"])
          and scores["scope_sha256"] == audited["scope_sha256"] == links["hh_scope"]["sha256"]
          and audited["diagnostic_sha256"] == links["hh_predictions"]["sha256"], "HongHu replay scope differs")
    for item in (hh, scores, audited):
        check(item["raw_gt_loaded"] is item["query_accessed"] is False and not item.get("updates"),
              "HongHu parent escapes released zero-update scope")
    check(docs["xz_audit"]["status"] == "passed"
          and not any(docs["xz_audit"][key] for key in ("raw_gt", "query", "independent_updates")),
          "Xuzhou audit lacks released endpoint certification")
    check(all(docs[f"pavia_{seed}"]["case"] == f"paviaU/{seed}/1/off"
              and docs[f"pavia_{seed}"]["audit_optimizer_updates"] == 0 for seed in (64, 512, 2048)),
          "Pavia seed registry differs")
    ids = [row["case"]["id"] for row in hh["cases"]]
    check(len(ids) == len(set(ids)) == 40, "HongHu archived design differs")
    cases = {
        "archive": ids,
        "whu-honghu": ["s512-k5-shared_transfer-none", "s512-k5-shared_transfer-continuous"],
        "xuzhou": ["off", "cnn_content", "graph_average"],
        "paviaU": ["64", "512", "2048"],
    }
    scope = dict(policy=POLICY, commit=code["commit"], source_hash=code["source_hash"], sources=links,
                 cases=cases, focus=FOCUS,
                 rare_policy={"scene": "whu-honghu", "historical_training_support_le": 25,
                              "training_counts": {"15": 11, "21": 14}, "v5_design": "5 per class"},
                 guards=dict(aggregate_min_delta=-.005, focus_min_delta_recall=-.02,
                             honghu_rare_no_added_error=True),
                 query_accessed=False, raw_gt_loaded=False, optimizer_updates=0)
    out.mkdir(exist_ok=False)
    return _write(out / "scope.json", scope)


def _honghu_context(scope):
    src = scope["parents"]["scope"]
    source = _read(src)
    pins = scope["protected_files"]
    reg = _read(pin(source["registration"], source["registration_sha256"]))
    plan, geometry = (_read(pin(reg[key], pins[reg[key]])) for key in ("plan", "development"))
    scene, allowed = honghu.guarded.bind_inputs(reg, plan, geometry)
    return source, reg, plan, geometry, scene, allowed


def _archive(root, docs, case_id, hh_ctx):
    scope, receipt = docs["hh_scope"], docs["hh_predictions"]
    _, reg, plan, geometry, scene, allowed = hh_ctx
    item = next(row for row in scope["cases"] if row["case"]["id"] == case_id)
    case = item["case"]
    released = honghu.prior.released_roles(reg, plan, geometry, allowed, case)["val"]
    _, labels, population = honghu.fixed_population(released, allowed, "val", scene["shape"][1], 5)
    check(population == item["populations"]["val"], "HongHu archived row order/labels differ")
    positions = {name: scope["conditions"][old]["sampling"]["selected_positions"]
                 for name, old in ARCHIVE_NAMES.items()}
    check(len(positions["full"]) == 270 and {len(v) for k, v in positions.items() if k != "full"} == {135},
          "Archived HongHu deleted-band count differs")
    path = root / PINS["hh_predictions"][0]
    arrays = pin(path.parent / f"{case_id}.npz", receipt["artifacts"][f"{case_id}.npz"])
    historical = _read(pin(path.parent / f"{case_id}.json", receipt["artifacts"][f"{case_id}.json"]))
    with np.load(io.BytesIO(custody._read(arrays)), allow_pickle=False) as stored:
        predictions = {condition: {head: hard(stored[f"obs__{old}__{head}"], len(labels), 22)
                                   for head in HEADS} for condition, old in ARCHIVE_NAMES.items()}
    pairs = {**BASE_PAIRS, "opposite_contiguous_vs_full": ("contiguous_opposite_end", "full"),
             "opposite_contiguous_vs_scattered": ("contiguous_opposite_end", "scattered")}
    paired = comparisons(predictions, labels, 22, pairs)
    for comparison, (target, source) in pairs.items():
        for head in HEADS:
            result = paired[comparison][head]
            for key, condition in (("target_confusion", target), ("source_confusion", source)):
                old = historical["readouts"][f"obs__{ARCHIVE_NAMES[condition]}"]["heads"][head]
                check(result[key] == old["confusion"], "HongHu paired classes differ from independent replay")
    return dict(scene="whu-honghu", cohort=f"archive/{case_id}", model_seed=HONGHU_SEED,
                population=population["sha256"], model=item["state"]["selected"], selected=item["state"]["selected_step"],
                support_seed=case["support_seed"], support_k=case["k"], method=case["method"],
                arm=case["arm"], source=arrays["sha256"], positions=positions, paired=paired,
                source_band_quality="approximate", physical_forwards=0)


def _xuzhou(docs, arm):
    audit = docs["xz_audit"]
    registration = _read(audit["registration"])
    completed = _read(audit["complete"])
    check(completed["registration"] == audit["registration"] and not completed["raw_gt"] and not completed["query"],
          "Xuzhou completed release differs")
    status, stage = read_stage(Path(registration["root"]), audit["registration"], arm, "full", None)
    check(completed["attempts"][arm]["full"] == stage, "Xuzhou checkpoint receipt differs")
    packet = _packet(status, registration["contracts"][arm], registration["budget"])
    model = custody._load_source(registration["source"], registration["sidecar"])
    if XUZHOU_ARMS[arm] is not None:
        model = XUZHOU_ARMS[arm].initialize(model, XUZHOU_SEED)
    model.load_state_dict(packet["selected_state"], strict=True)
    accepted = audit["results"][arm]["endpoints"]["selected"]
    identity = state_hash(model)
    check(accepted["step"] == packet["selected_step"] and
          identity == packet["history"][accepted["step"] - 1]["state_hash"], "Xuzhou selected state differs")
    historical = {"full": {head: accepted["metrics"][head]["confusion_matrix"] for head in HEADS}}
    return model, XUZHOU_SEED, identity, accepted["step"], historical, status["checkpoint"]["sha256"]


def _pavia(docs, cohort, intake, bank):
    seed = int(cohort)
    audit = docs[f"pavia_{seed}"]
    registered = _read(audit["registration"])
    stage = _read(audit["stages"][-1])
    packet = _state(stage["checkpoint"])
    population = digest(sorted(point for row in bank.role_rows("devval") for point in row["samples"]))
    check(registered["seed"] == seed and registered["arm"] == "off" and registered["factor"] == 1 and
          registered["bank"] == bank.pin and audit["point_set"]["devval"] == population and
          packet["registration"] == digest(registered) and packet["cursor"] == audit["last_step"],
          "Pavia checkpoint or development role differs")
    model = init_native(target_head_baseline(intake.baseline, "paviaU", seed), "off", registered["module_seed"])
    model.load_state_dict(packet["selected"]["state"], strict=True)
    identity, step = state_hash(model), packet["selected"]["step"]
    check(step == audit["selected_step"] and identity == audit["selected_model_sha256"],
          "Pavia selected model identity differs")
    historical = {name: {head: audit["head_metrics"][head][old]["confusion_matrix"] for head in HEADS}
                  for name, old in (("full", "full"), ("scattered", "even_retained_positions"))}
    return model, seed, identity, step, historical, stage["checkpoint"]["sha256"]


def _new_honghu(root, docs, case_id, hh_ctx):
    _, reg, *_ = hh_ctx
    scope, receipt = docs["hh_scope"], docs["hh_predictions"]
    item = next(row for row in scope["cases"] if row["case"]["id"] == case_id)
    case, manifest = item["case"], item["state"]
    args = (reg["source"]["weight"], reg["source"]["result"]) if case["method"] == "shared_transfer" else ()
    model, _ = init_honghu(SpatialConfig(reg["config"]), *args, coverage_arm=case["arm"])
    saved = _state(pin(manifest["path"], manifest["state_sha256"]))
    model.load_state_dict(saved["best"]["model"], strict=True)
    identity = state_hash(model)
    check(identity == manifest["selected"], "HongHu selected model identity differs")
    previous = _read(pin(root / Path(PINS["hh_predictions"][0]).parent / f"{case_id}.json",
                         receipt["artifacts"][f"{case_id}.json"]))
    historical = {"full": {head: previous["readouts"]["obs__full"]["heads"][head]["confusion"]
                           for head in HEADS}}
    return model, HONGHU_SEED, identity, manifest["selected_step"], historical, manifest["state_sha256"]


def _freeze_bank(root, scene, scope):
    link = scope["sources"]["pavia_bank"] if scene == "paviaU" else None
    bank, prior, intake = open_bank(root, scene, publication=link)
    return bank, intake


def _shapes(row, width):
    a, b, c, d = row["box"]
    return (b-a, d-c), [(i // width - a) * (d-c) + i % width-c for i, _ in row["samples"]]


def _guard(paired, focus, policy, rare):
    results = {}
    for name in BASE_PAIRS:
        results[name] = {}
        for head, result in paired[name].items():
            classes = [row for row in result["classes"] if row["class_id"] in focus and row["denominator"]]
            oa = result["delta_oa"] >= policy["aggregate_min_delta"]
            aa = result["delta_aa"] >= policy["aggregate_min_delta"]
            recall = all(row["target_recall"] - row["source_recall"] >= policy["focus_min_delta_recall"]
                         for row in classes)
            no_loss = all(row["correct_to_error"] == 0 for row in classes) if rare else None
            results[name][head] = dict(oa=oa, aa=aa, focus_recall=recall, rare_no_added_error=no_loss,
                                       passed=oa and aa and recall and no_loss is not False)
    return results


def _cross(paired):
    output = {}
    for head in HEADS:
        full, missing = (paired[f"target_vs_matched_{band}"][head] for band in ("full", "contiguous"))
        output[head] = {key: missing[key] - full[key] for key in ("delta_oa", "delta_aa")}
        output[head]["classes"] = [
            dict(class_id=a["class_id"], denominator=a["denominator"],
                 difference_of_recall_differences=None if not a["denominator"] else
                 (b["target_recall"] - b["source_recall"]) - (a["target_recall"] - a["source_recall"]))
            for a, b in zip(full["classes"], missing["classes"], strict=True)]
    return output


def _infer_case(root, scope, docs, scene, case, hh_ctx, bank, intake):
    begin = perf_counter()
    context = _xuzhou(docs, case) if scene == "xuzhou" else (
        _pavia(docs, case, intake, bank) if scene == "paviaU" else _new_honghu(root, docs, case, hh_ctx))
    model, seed, identity, selected, historical, checkpoint = context
    model.eval()
    model.backbone.adapter.cache_attention = False
    model.backbone.adapter.clear_attention_cache()
    spec = runtime.band_specs(bank.release.bands)
    scale = 50 if scene == "paviaU" else 100
    options = dict(wavelength_unit=scene != "xuzhou")
    classes, width = bank.release.classes, bank.release.shape[1]
    predictions, labels, positions, traces, groups, points = {}, [], {}, [], {}, []
    negative = None
    for row in bank.role_rows("devval"):
        if not row["samples"]:
            continue
        shape, offsets = _shapes(row, width)
        tile_labels = [label for _, label in row["samples"]]
        points.extend(row["samples"])
        labels.extend(tile_labels)
        boxes = runtime.region_boxes(shape, offsets, tile_labels, FOCUS[scene])
        _, _, original, _ = bank.materialize(row, spec["full"])
        trace = dict(tile=row["id"], region_area=(boxes["target"][1]-boxes["target"][0])**2,
                     labels=len(offsets), views={})
        for band, recipe in spec.items():
            clean, record = sample(original.observation, original.bands, recipe)
            retained = record["selected_positions"]
            check(list(clean.bands.band_ids) == retained and
                  (band == "full" or len(retained) == (len(original.bands.band_ids) + 1) // 2),
                  "Actual retained positions and equal-count design disagree")
            check(positions.setdefault(band, retained) == retained, "Band order differs between tiles")
            alterations = ("clean", "target", "matched") if band != "scattered" else ("clean",)
            for alteration in alterations:
                chosen = band if alteration == "clean" else f"{band}_{alteration}"
                box = boxes.get(alteration)
                actual = runtime.replace(clean, observation=runtime._spatial_view(clean.observation, box))
                output, graph, membership = runtime.infer(model, scene, actual, offsets, classes, scale, **options)
                trace["views"][chosen] = graph
                heads = predictions.setdefault(chosen, {head: [] for head in HEADS})
                for head in HEADS:
                    heads[head].extend(hard(output[head], len(offsets), classes))
                if chosen == "full":
                    grouping = runtime.same_region_groups(offsets, boxes["target"], shape, membership)
                    for key, mask in grouping.items():
                        groups.setdefault(key, []).extend(mask.tolist())
            if band == "contiguous" and negative is None:
                _, _, direct, _ = bank.materialize(row, recipe)
                check(np.array_equal(clean.observation.spectral, direct.observation.spectral)
                      and clean.bands == direct.bands, "Direct retained bank reader differs")
                negative = runtime.validate_observation_path(
                    model, scene, original.observation, original.bands, bank.release.statistics,
                    recipe, offsets, classes, scale, **options)
        traces.append(trace)
    check(len(points) == len({idx for idx, _ in points}) == 5 * classes and state_hash(model) == identity,
          "Frozen weights or fixed validation population changed")
    paired = comparisons(predictions, labels, classes, {**BASE_PAIRS, **REGION_PAIRS},
                         group_masks={key: np.asarray(value, dtype=bool) for key, value in groups.items()})
    replay = {}
    for condition, heads in historical.items():
        replay[condition] = {}
        for head, expected in heads.items():
            actual = next(result[head][field] for name, (target, source) in BASE_PAIRS.items()
                          for result in [paired[name]]
                          for direct, field in ((target, "target_confusion"), (source, "source_confusion"))
                          if direct == condition)
            replay[condition][head] = dict(equal=actual == expected,
                                          count_l1=int(np.abs(np.asarray(actual)-np.asarray(expected)).sum()))
    return dict(scene=scene, cohort="factor1/off" if scene == "paviaU" else f"frozen/{case}",
                model_seed=seed, model=identity, checkpoint=checkpoint, selected=selected,
                population=digest(points), bank=bank.pin["sha256"], positions=positions,
                paired=paired, source_band_quality=bank.release.bands.wavelength_quality,
                group_counts={key: sum(mask) for key, mask in groups.items()}, trace=traces,
                deleted_negative=negative, historical_confusion_check=replay,
                protection=_guard(paired, FOCUS[scene], scope["guards"], scene == "whu-honghu"),
                occlusion_interaction=_cross(paired), physical_forwards=7*len(traces) + 3,
                wall_seconds=perf_counter()-begin)


def run(root, scope_path, output, code, partition):
    started = _time()
    tick = perf_counter()
    scope_link = pin(scope_path)
    scope = _read(scope_link)
    links, docs = source_documents(root)
    check(scope["policy"] == POLICY and scope["sources"] == links and
          (scope["commit"], scope["source_hash"]) == (code["commit"], code["source_hash"]),
          "Scope and exact checkout differ")
    names = list(scope["cases"]) if partition == "all" else [partition]
    output.mkdir(exist_ok=True)
    results = {}
    hh_ctx, new_cases = None, 0
    for name in names:
        pending = [case for case in scope["cases"][name] if not (output / f"{name}__{case}.json").exists()]
        if pending and name in ("archive", "whu-honghu") and hh_ctx is None:
            hh_ctx = _honghu_context(docs["hh_scope"])
        bank, intake = _freeze_bank(root, name, scope) if pending and name != "archive" else (None, None)
        for case in scope["cases"][name]:
            path = output / f"{name}__{case}.json"
            if path.exists():
                row = _read(pin(path))
                check(row["scope"] == scope_link["sha256"] and row["case"] == case and row["lane"] == name
                      and row["result_hash"] == digest(row["result"]), "Completed case differs on resume")
                result = row["result"]
            else:
                result = (_archive(root, docs, case, hh_ctx) if name == "archive"
                          else _infer_case(root, scope, docs, name, case, hh_ctx, bank, intake))
                if name == "archive":
                    result["protection"] = _guard(result["paired"], FOCUS["whu-honghu"], scope["guards"], True)
                _write(path, dict(scope=scope_link["sha256"], case=case, lane=name,
                                  result_hash=digest(result), result=result))
                new_cases += 1
                print(f"completed {name}/{case}", flush=True)
            results[path.name] = pin(path)["sha256"]
    require_frozen_code(code)
    print(json.dumps(dict(scope=scope_link["sha256"], artifacts=results, cases=len(results),
                          new_cases=new_cases,
                          started=started, finished=_time(), seconds=perf_counter()-tick), sort_keys=True))


def aggregate(scope_path, output, path, code):
    link = pin(scope_path)
    scope = _read(link)
    check((scope["commit"], scope["source_hash"]) == (code["commit"], code["source_hash"]),
          "Aggregator checkout differs from scope")
    cases, artifacts = [], {}
    for name, entries in scope["cases"].items():
        for case in entries:
            row_path = output / f"{name}__{case}.json"
            row_pin = pin(row_path)
            row = _read(row_pin)
            check(row["scope"] == link["sha256"] and row["lane"] == name and row["case"] == case
                  and row["result_hash"] == digest(row["result"]), "Aggregate case provenance differs")
            cases.append(row["result"])
            artifacts[row_path.name] = row_pin["sha256"]
    value = dict(schema=POLICY, status="passed", scope=link["sha256"], commit=code["commit"],
                 optimizer_updates=0, raw_gt_loaded=False, query_accessed=False,
                 cases=len(cases), physical_forwards=sum(row["physical_forwards"] for row in cases),
                 archived_cases=sum(not row["physical_forwards"] for row in cases),
                 focus=scope["focus"], rare_policy=scope["rare_policy"], artifacts=artifacts,
                 groups=summarize(cases), finished=_time())
    require_frozen_code(code)
    return _write(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--root", type=Path, required=True, help="Authorized parent runs directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--partition", choices=("all", "archive", *SCENES), default="all")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    torch.set_num_threads(1)
    root, out = external_directory(args.root), external_directory(args.output)
    check(out.is_relative_to(root) and out != root, "Output must be inside the authorized runs directory")
    if args.phase == "prepare":
        print(prepare(root, out, code))
    else:
        check(args.scope is not None and args.scope.resolve().is_relative_to(root), "Frozen run scope required")
        if args.phase == "run":
            run(root, args.scope, out, code, args.partition)
        else:
            print(aggregate(args.scope, out, out.parent / "summary.json", code))


if __name__ == "__main__":
    main()

"""Audit the admitted three-scene history and verify equal-capacity readouts."""

from collections import Counter, defaultdict
from csv import DictReader
import json
import os
from pathlib import Path
import re

from evaluation import wp06_capacity_released as capacity
from evaluation.frozen_training_stages import HEADS, VIEWS, json_pin, pinned, require
from model.research_config import digest

RETAINED = "wp05-wp06-real-evidence-20261119-qa/delivery-copy"
PINS = {
    "admission": (f"{RETAINED}/history-admission.json",
                  "654d35e7c23f3731b6d4647905bc4ce065286f66fffe22e2561a7900bbe48119"),
    "acceptance": (f"{RETAINED}/wp12-retained-acceptance.json",
                   "8b70844acfc506e09244235069c478b9420e3b5d0af482650036fc896af51c75"),
    "same_points": (f"{RETAINED}/same-points.csv",
                    "980089e47c89bfc4444310cfbcd27c45e06ac97428f519c5ddd7a7bca722a758"),
    "paired": ("wp05-wp06-external-paired-20261119-qa/delivery/wp12-external-paired-index.json",
               "642c38f8262dc7bce187f717f5aab9ee218af5034b9250a95caae2f711013979"),
    "spaces": ("wp02-wp05-wp06-spatial-20261119/registration.json",
               "354562ee401501d35804a6b309fc230f1354926b0e9ab5fae9625dd67b5df9d1"),
}
PROVENANCE = "docs/references/wp05-scene-provenance-20261119.json"
CANDIDATE_NAME = re.compile(r"approv|authorit|holder|optical.*release|semantic.*map", re.I)


def approval_inventory(root, incoming):
    """Count release-shaped names and inspect only small approval documents in runs."""
    result = {}
    for area, base in (("incoming", incoming), ("runs", root)):
        counts, schemas, candidates = Counter(), Counter(), []
        for folder, directories, filenames in os.walk(base, followlinks=False):
            parent = Path(folder)
            directories.sort()
            if len(parent.relative_to(base).parts) >= 7:
                directories.clear()
            for name in sorted(filenames):
                if not CANDIDATE_NAME.search(name):
                    continue
                path = parent / name
                if path.is_symlink() or not path.is_file():
                    continue
                counts["filename_matches"] += 1
                if area == "incoming" or path.suffix != ".json" or path.stat().st_size > 65536:
                    continue
                counts["small_json_examined"] += 1
                try:
                    record = json.loads(path.read_bytes())
                except (UnicodeError, ValueError, OSError):
                    counts["unreadable"] += 1
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("schema"), str):
                    continue
                schema = record["schema"]
                mode = record.get("source_mode")
                mode = mode if isinstance(mode, str) else "unspecified"
                if schema.startswith(("wp02_", "wp05_")):
                    schemas[f"{schema}/{mode}"] += 1
                if schema in ("wp02_independent_development_approval_v1", "wp02_optical_authority_v1"):
                    if mode != "synthetic":
                        candidates.append(dict(schema=schema, source_mode=mode,
                            metadata_sha256=digest(record), location=path.relative_to(base).as_posix()))
        result[area] = dict(counts=dict(counts), protocol_schemas=dict(schemas),
                            candidates_requiring_owner_binding=candidates)
    return dict(areas=result, directory_depth=7, maximum_metadata_bytes=65536,
                incoming_contents_opened=False, authority_inferred_from_filename=False)


def transition(before, after, label):
    require(before.keys() == after.keys() and all(before[i][0] == after[i][0] for i in before),
            "The retained views omit points or change their class identity")
    cells = Counter((old, after[i][1]) for i, (truth, old) in before.items() if truth == label)
    n = sum(cells.values())
    old = sum(v for (i, _), v in cells.items() if i == label)
    new = sum(v for (_, j), v in cells.items() if j == label)
    return dict(n=n, before=old, after=new, baseline_wrong=n-old,
                gains=sum(v for (i, j), v in cells.items() if i != label and j == label),
                losses=sum(v for (i, j), v in cells.items() if i == label and j != label),
                destinations=[[i, j, v] for (i, j), v in sorted(cells.items())])


def archived_points(path, acceptance, archived):
    by_scope = defaultdict(dict)
    native = {f"native_{head}" for head in HEADS}
    with path.open(newline="", encoding="utf-8") as file:
        for row in DictReader(file):
            case, view, readout = row["case"], row["view"], row["readout"]
            if case not in archived or readout not in native:
                continue
            source = archived[case]
            require(view in VIEWS and row["scene"] == source["scene"] and int(row["stage"]) == source["step"],
                    "Retained point has a different scene, view or checkpoint cursor")
            labels = by_scope[case, view, readout]
            point = int(row["point_ordinal"])
            require(point not in labels, "Duplicate native declaration for a released point")
            labels[point] = int(row["truth"]), int(row["predicted"])
    outcomes = {}
    for case, source in archived.items():
        attested = acceptance["cases"][case]
        fields = ("scene", "step", "checkpoint", "authority")
        require(all(source[k] == attested[k] for k in fields) and source["role"] == "archived_devval",
                "Historical role, checkpoint or development approval differs")
        views, common = {}, None
        for view in VIEWS:
            views[view] = {}
            for head in HEADS:
                key = f"native_{head}"
                a, b = by_scope[case, "full", key], by_scope[case, view, key]
                roster = sorted((i, label) for i, (label, _) in a.items())
                if common is None:
                    common = roster
                    require([i for i, _ in roster] == list(range(1, len(roster)+1)),
                            "Expected complete deidentified development ordinals")
                require(roster == common, "Heads disagree about the actual point or label roster")
                focus = {}
                for label in attested["focus"]:
                    cell = transition(a, b, label)
                    expected = source["views"][view][head]["focus"][str(label)]
                    require(all(cell[k] == v for k, v in expected.items()),
                            "Same-point joined integer totals differ from the independently admitted archive")
                    focus[str(label)] = cell
                views[view][head] = focus
        outcomes[case] = dict(**{k: source[k] for k in fields}, role=source["role"],
                              archived_population=source["population_order"], n=len(common),
                              ordinal_truth_sha256=digest(common), views=views)
    return outcomes


def short_support_pairs(history):
    cohorts = history["cohorts"]
    pairs = {}
    for k in (1, 5):
        for method in ("scratch", "shared_transfer"):
            a, b = (cohorts[f"s{seed}-k{k}-{method}"] for seed in (512, 2048))
            require(a.keys() == b.keys() and len(a) == 5, "Incomplete five-arm HongHu support comparison")
            for arm in sorted(a):
                left, right = a[arm], b[arm]
                require(left["support_seed"] == 512 and right["support_seed"] == 2048
                        and left["support"] != right["support"] and left["views"].keys() == right["views"].keys()
                        and all(left[f] == right[f] for f in ("k", "method", "model_seed", "role", "validation")),
                        "History changes the metric population or the model/support protocol")
                focus = {}
                for view in sorted(left["views"]):
                    focus[view] = {}
                    for head in HEADS:
                        x, y = left["views"][view][head]["focus"], right["views"][view][head]["focus"]
                        require(x.keys() == y.keys() == {"15", "21"} and all(
                            x[c]["n"] == y[c]["n"] == 5 for c in x), "Focus denominators differ across support draws")
                        focus[view][head] = {c: dict(n=x[c]["n"], seed_512=[x[c]["before"], x[c]["after"]],
                                                  seed_2048=[y[c]["before"], y[c]["after"]]) for c in x}
                pairs[f"k{k}-{method}-{arm}"] = dict(model_seed=left["model_seed"], role=left["role"],
                    validation=left["validation"], supports=[left["support"],right["support"]],
                    archived_arm_predictions=[left["archived_predictions"],right["archived_predictions"]],
                    checkpoints=[left["selected_state"],right["selected_state"]],
                    steps=[left["selected_step"],right["selected_step"]],
                    matched_checkpoint_cursor=left["selected_step"] == right["selected_step"], views=focus)
    return pairs


def index(root, incoming, capacity_pins, producer):
    pins = {key: dict(path=str(root / path), sha256=sha) for key, (path, sha) in PINS.items()}
    acceptance, history, spatial, paired = (json_pin(root, pins[key])
                                            for key in ("acceptance", "admission", "spaces", "paired"))
    require(acceptance["point_csv_sha256"] == pins["same_points"]["sha256"]
            and acceptance["optimizer_updates"] == history["optimizer_updates"]
            == paired["real_optimizer_updates"] == spatial["real_updates"] == 0
            and all(record["query_accessed"] is False for record in (acceptance,history,spatial,paired))
            and all(record["raw_gt_loaded"] is False for record in (acceptance,spatial,paired)),
            "The historical chain includes an unapproved role or update")
    native = archived_points(pinned(root, pins["same_points"]), acceptance, paired["frozen_actual_retained_views"])
    require(set(capacity_pins) == capacity.CASES.keys(), "Both preregistered real native endpoints are required")
    checked = capacity.review(root, capacity_pins, {"commit":producer})
    for case, probe in checked.items():
        archived = native[case]
        require((probe["checkpoint"],probe["authority"],probe["stage"]) ==
                (archived["checkpoint"],archived["authority"],archived["step"]),
                "Archived transitions and equal-width probe use different frozen cases")
        for view in VIEWS:
            for head in HEADS:
                for label, row in archived["views"][view][head].items():
                    result = probe["results"]["views"][view]["native"][head]["focus"][label]
                    counts = Counter()
                    for _, dst, n in row["destinations"]:
                        if dst != int(label):
                            counts[str(dst)] += n
                    require((result["n"],result["correct"],result["errors"]) == (row["n"],row["after"],dict(counts)),
                            "Current native focus destinations differ from the archived point joins")
    provenance = json.loads((Path(__file__).resolve().parents[1] / PROVENANCE).read_bytes())
    require(provenance["approved_real_sidecars"] == [] and set(history["scenes"]) == set(spatial["spaces"])
            == set(provenance["scenes"]), "Scene coverage or optical authority changed")
    spaces = {}
    for scene, row in spatial["spaces"].items():
        scope = history["scenes"][scene]
        spaces[scene] = dict(
            provenance=provenance["scenes"][scene], band_axis=scope["band_axis"],
            bank=scope["bank"], independent_checks=scope["independent_checks"],
            role_buffer_conflicts=scope["role_buffer_conflicts"], historical_pinned_roles=scope["role_reports"],
            historical_space={key:row[key] for key in ("authority","actual_gap","common_parent_count","counts",
                "role_blocks","independent_release","previously_seen_intersection","previously_unseen_validation")},
            newly_approved_independent_labels=0)
    reviewed = paired["honghu_reviewed_120"]
    return dict(schema="wp12_real_development_readiness_v1", pins=pins, cases=native, scenes=spaces,
                approval_inventory=approval_inventory(root,incoming),
                existing_real_release_accepted_for_independent_accuracy=paired["independent_real_accuracy"],
                honghu=dict(short_support_pairs=short_support_pairs(paired["honghu_historical_short"]),
                            reviewed_120={key:reviewed[key] for key in ("archive","review","coverage",
                                "scope","point_transition_join_available")}),
                equal_capacity=checked, optimizer_updates=0, query_accessed=False, raw_gt_loaded=False)

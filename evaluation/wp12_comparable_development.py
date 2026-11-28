"""Reconcile the admitted WP04 Xuzhou development results without opening checkpoints."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from evaluation.wp12_residual_interaction_evidence import EXPECTED_PARAMS, FACTORIALS, PAIR_COMPARISONS
from scripts.run_wp04_factorial import ASSIGNMENT_GROUP, FACTORIAL_EFFECTS, I7_PARAMS, MATCHED_PARAMS, REMAINING

SEEDS = (64, 512, 2048)
ENDPOINTS = ("selected", "last")
HEADS = ("fused", "cnn", "graph")
CLASSES = range(1, 10)
METRICS = ("oa", "aa", "nll")
FINAL_PHASES = ("whole", "prefix", "resume", "completed")
ARMS = {
    "residual-fusion": ("B0", "R", "F", "R+F", "Bcap", "Bcap+F"),
    "direction-capacity": ("off", "add", "both", "C2G", "G2C", "local-add", "local-both",
                           "local-C2G", "local-G2C"),
    "matched-edge-message": ("M00", "M10", "M01", "M11"),
    "fixed-edge-weights": ("off", "norm", "Wg", "Wj"),
    "message-correspondence": ("I8", "I9-message", "Wj", "Wg65"),
    "residual-assignment": ("B0", "R", "R3", "XRegion", "Grid0", "GridR", "Learn0", "LearnR"),
    "legacy-direct-joint": ("I0", "I1", "I2", "I6", "I7", "I8"),
}
PINS = {
    "protocol-map.csv": "20a85ea61f69eda0f14316952dab1f9e1d0e3953ccbc2490df07ea95b7938099",
    "factorial/evidence.json": "e283e28275feaaa0c47c230e31d44dc641d08f7d8b18b366d078ea5745339154",
    "factorial/review.json": "d48a3b7c07ea131ca5214b5cafdaf187d49433b239911abc43c9ad454306599e",
    "direction/evidence.json": "599ba1ae59489ce93bd1f25c989cbc54c289825ce0dd7ea402b34a3b68df138b",
    "direction/review.json": "9a1fcc4692b93c7e398d1ce2842568dd69069e7544be33173ee8480825622ca8",
    "matched/supplement.json": "20b9f9fb4e23b78d758d4a71e3d88d1d232a5e2c0e565ec115f1e85fa2fb2737",
    "edge/evidence.json": "c4c166aa0c2b7c9ea41eaa1dde888d8e8da131feb87e117ca55b2bbbbb007159",
    "edge/supplement.json": "a8c5aa236bb3e6fe4bb5f65bec72e1950e41587838dc3b53e395f2aa76220b00",
    "message/evidence.json": "e98c2d65dc6028932a3f51e1ef1d99ecdf1f673877ccff12666524cbaa39418c",
    "message/supplement.json": "1a6de8dcc53aebd8b4689f8b78c4cfd26ab34f0dcff5be97d518d4c36254ec9d",
    "residual/supplement.json": "bbfe778ed69b11824376cbda8a8fbc70a1d6908f5f798072dd90acd31cede10d",
    "residual/history.json": "561af2b2a1d9f81a205c926c89631971fca0dbcc0b788babd6d2076ca8dd9d9e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(root, name, digest):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), f"Evidence escapes the input directory: {name}")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, f"Evidence digest mismatch: {name}")
    return payload


def close(left, right, context, tolerance=1e-11):
    require(math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance),
            f"Numerical disagreement for {context}: {left} vs {right}")


def read_csv(root, name, digest):
    return list(csv.DictReader(checked(root, name, digest).decode("utf-8-sig").splitlines()))


def metric_case(values, tallies, context):
    counts = [(int(row["class_id"]), int(row["correct"]), int(row["support"])) for row in tallies]
    require([item[0] for item in counts] == list(CLASSES), f"Incomplete class roster: {context}")
    require(all(0 <= correct <= support == 5 for _, correct, support in counts),
            f"Invalid development denominator: {context}")
    metrics = {"oa": sum(correct for _, correct, _ in counts) / 45,
               "aa": math.fsum(correct / support for _, correct, support in counts) / 9,
               "nll": float(values["nll"])}
    require(math.isfinite(metrics["nll"]) and metrics["nll"] >= 0, f"Invalid NLL: {context}")
    for name in ("oa", "aa"):
        close(metrics[name], values[name], (context, name))
    return {**metrics, "counts": counts}


def point_case(values, context):
    points = values["points"]
    require(sorted((point["class_id"], point["point"]) for point in points)
            == [(c, i) for c in CLASSES for i in range(1, 6)], f"Invalid released point ordinals: {context}")
    counts = {c: 0 for c in CLASSES}
    for point in points:
        require(point["predicted"] in CLASSES and 0 < point["true_probability"] <= 1,
                f"Invalid released prediction: {context}")
        counts[point["class_id"]] += point["class_id"] == point["predicted"]
    tallies = [{"class_id": c, "correct": counts[c], "support": 5} for c in CLASSES]
    expected = values.get("correct_by_class")
    if expected is None:
        expected = [row["correct"] for row in values["classes"]]
    require(list(counts.values()) == expected, f"Point and class totals disagree: {context}")
    nll = -math.fsum(math.log(p["true_probability"]) for p in points) / len(points)
    close(nll, values["nll"], context, tolerance=5e-7)
    return metric_case(values, tallies, context)


def audit_identity(pin):
    case = Path(pin["path"]).parent.name
    seed, arm = case.split("-", maxsplit=1)
    return (int(seed), arm), case


def table_cases(root, folder, evidence, review=None):
    require(evidence["query_access"] is False, f"Unexpected query access: {folder}")
    table_hashes = {Path(pin["path"]).name: pin["sha256"] for pin in evidence["tables"]}
    tables = {name: read_csv(root, f"{folder}/{name}", digest) for name, digest in table_hashes.items()}
    reviewed = {(row["seed"], row["arm"]): row for row in review["cases"]} if review else {}
    pins = evidence["case_pins"] + evidence.get("reused_case_pins", [])
    cases = {}
    for pin in pins:
        key, name = audit_identity(pin)
        require(key not in cases, f"Duplicate producer case: {folder}, {key}")
        local = f"{folder}/audits/{name}/audit.json"
        if pin in evidence["case_pins"]:
            audit = json.loads(checked(root, local, pin["sha256"]))
        else:
            audit = reviewed[key]
            require(audit["case_sha256"] == pin["sha256"], f"Wrong historical direction reference: {key}")
            local = f"{folder}/review.json"
        if review:
            require(reviewed[key].get("audit_sha256", reviewed[key].get("case_sha256")) == pin["sha256"],
                    f"Independent case receipt disagrees: {folder}, {key}")
        cases[key] = {"pin": pin, "locator": local, "source": f"{folder}/evidence.json", "audit": audit,
                      "endpoints": {}, "costs": {}}

    tallies = defaultdict(list)
    for row in tables["per-class.csv"]:
        key = (int(row["seed"]), row["arm"], row["endpoint"], row["head"])
        close(int(row["correct"]) / int(row["support"]), row["recall"], (folder, key, row["class_id"]))
        tallies[key].append(row)
    for row in tables["endpoints.csv"]:
        key = (int(row["seed"]), row["arm"])
        ep = (row["endpoint"], row["head"])
        case = cases[key]
        require(ep not in case["endpoints"], f"Duplicate endpoint: {folder}, {key}, {ep}")
        source = case["audit"]["endpoints"][ep[0]][ep[1]]
        context = (folder, key, ep)
        metric = metric_case(row, tallies.pop((*key, *ep)), context)
        source_metric = metric_case(source, source["classes"], context)
        require(metric["counts"] == source_metric["counts"], f"Case and table classes disagree: {context}")
        close(metric["nll"], source_metric["nll"], context)
        if review:
            independent = reviewed[key]["endpoints"][ep[0]][ep[1]]
            close(metric["nll"], independent["nll"], (context, "review"))
        case["endpoints"][ep] = {**metric, "step": int(row["step"]), "parameters": int(row["parameters"])}
    require(not tallies, f"Unbound class rows: {folder}")
    for row in tables["costs.csv"]:
        case = cases[int(row["seed"]), row["arm"]]
        phase = row["phase"]
        require(phase not in case["costs"], f"Duplicate cost row: {folder}, {row}")
        case["costs"][phase] = {name: row[name] for name in (
            "updates", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "peak_rss_bytes", "hardware")}
        case["costs"][phase]["cost_scope"] = case["audit"].get("phases", {}).get(phase, {}).get("scope")
    for case in cases.values():
        case["validation"] = case["audit"].get("inputs", {}).get("val_indices")
        del case["audit"]
    return cases, tables


def supplement_case(record, source):
    key = (record["seed"], record["arm"])
    endpoints = {}
    phases = record["phases"]
    for endpoint in ENDPOINTS:
        for head in HEADS:
            context = (source, key, endpoint, head)
            metric = point_case(record["endpoints"][endpoint][head], context)
            endpoints[endpoint, head] = {
                **metric,
                "step": record["steps"][endpoint] if "steps" in record else phases["whole"]["updates"],
                "parameters": None,
            }
    costs = {}
    for name, phase in phases.items():
        costs[name] = {
            "updates": phase["updates"], "wall_seconds": phase["wall_seconds"],
            "peak_allocated_bytes": phase.get("max_allocated_bytes"),
            "peak_reserved_bytes": phase.get("max_reserved_bytes"),
            "peak_rss_bytes": phase.get("peak_rss_bytes"), "hardware": record["device"],
            "cost_scope": phase.get("scope"),
        }
    return key, {"pin": record["origin"], "locator": f"{source}#case={key[0]}/{key[1]}", "source": source,
                 "endpoints": endpoints, "costs": costs, "validation": record["input"]["val_indices"]}


def read_cases(root, data):
    cohorts = {}
    producer_tables = {}
    for protocol, folder in (("residual-fusion", "factorial"), ("direction-capacity", "direction"),
                             ("fixed-edge-weights", "edge"), ("message-correspondence", "message")):
        cohorts[protocol], producer_tables[protocol] = table_cases(
            root, folder, data[f"{folder}/evidence.json"], data.get(f"{folder}/review.json"))
    for seed in SEEDS:
        cohorts["message-correspondence"][seed, "Wj"] = cohorts["fixed-edge-weights"][seed, "Wj"]
    source = "matched/supplement.json"
    cohorts["matched-edge-message"] = dict(supplement_case(record, source) for record in data[source]["cases"])
    source = "residual/supplement.json"
    residual = dict(supplement_case(record, source) for record in data[source]["cases"])
    for protocol in ("residual-assignment", "legacy-direct-joint"):
        cohorts[protocol] = {key: case for key, case in residual.items() if key[1] in ARMS[protocol]}
    require(sum(len(cohorts[p]) for p in ("residual-assignment", "legacy-direct-joint")) == len(residual),
            "Unbound residual supplement case")
    aliases = defaultdict(list)
    reports = {}
    for protocol, arms in ARMS.items():
        cases = cohorts[protocol]
        require(set(cases) == {(seed, arm) for seed in SEEDS for arm in arms}, f"Wrong cohort roster: {protocol}")
        for (seed, arm), case in cases.items():
            require(set(case["endpoints"]) == {(ep, head) for ep in ENDPOINTS for head in HEADS}
                    and set(case["costs"]) == set(FINAL_PHASES), f"Incomplete case: {protocol}, {seed}, {arm}")
            close(case["costs"]["whole"]["updates"], 12, (protocol, seed, arm, "whole"))
            require(all(row["step"] == 12 for row in case["endpoints"].values()),
                    f"Unexpected endpoint selection: {protocol}, {seed}, {arm}")
            require(all(case["endpoints"]["selected", head]["counts"] == case["endpoints"]["last", head]["counts"]
                        for head in HEADS), f"Step-12 endpoints disagree: {protocol}, {seed}, {arm}")
            parameters = {
                "matched-edge-message": MATCHED_PARAMS,
                "residual-assignment": {**EXPECTED_PARAMS, **REMAINING[ASSIGNMENT_GROUP][1]},
                "legacy-direct-joint": {**EXPECTED_PARAMS, **I7_PARAMS},
            }.get(protocol)
            if parameters:
                for row in case["endpoints"].values():
                    row["parameters"] = parameters[arm]
            digest = case["pin"]["sha256"]
            aliases[digest].append((protocol, seed, arm))
            if digest in reports:
                other = reports[digest]
                require(case["pin"] == other["pin"], f"Conflicting identity: {digest}")
                for ep, row in case["endpoints"].items():
                    previous = other["endpoints"][ep]
                    require(row["counts"] == previous["counts"], f"Conflicting alias counts: {digest}, {ep}")
                    close(row["nll"], previous["nll"], (digest, ep), tolerance=5e-7)
                    require(row["parameters"] == previous["parameters"], f"Conflicting alias parameters: {digest}")
            else:
                reports[digest] = case
        for seed in SEEDS:
            roles = {cases[seed, arm].get("validation") for arm in arms} - {None}
            require(len(roles) == 1, f"Conflicting development indices: {protocol}, {seed}")
    return cohorts, producer_tables, aliases


def pair_terms(names):
    return {name: {a: 1, b: -1} for name, (a, b) in names.items()}


def contrast_catalog(tables, data):
    catalog = {"residual-fusion": FACTORIAL_EFFECTS}
    for protocol in ("direction-capacity", "fixed-edge-weights", "message-correspondence"):
        names = sorted({row["comparison"] for row in tables[protocol]["paired.csv"]})
        pairs = {}
        for name in names:
            a, b = next((a, b) for a in ARMS[protocol] for b in ARMS[protocol] if f"{a}-{b}" == name)
            pairs[name] = (a, b)
        catalog[protocol] = pair_terms(pairs)
    catalog["message-correspondence"]["Wj-Wg65"] = {"Wj": 1, "Wg65": -1}
    matched = data["matched/supplement.json"]
    catalog["matched-edge-message"] = {
        **{name: record["coefficients"] for name, record in matched["effects"].items()},
        **pair_terms({name: name.split("-", maxsplit=1) for name in matched["conditional"]}),
    }
    for protocol in ("residual-assignment", "legacy-direct-joint"):
        allowed = set(ARMS[protocol])
        catalog[protocol] = {name: terms for name, terms in {**pair_terms(PAIR_COMPARISONS), **FACTORIALS}.items()
                             if terms.keys() <= allowed}
    return catalog


def compare(cases, seed, endpoint, head, terms):
    rows = {arm: cases[seed, arm]["endpoints"][endpoint, head] for arm in terms}
    metric = {f"delta_{name}": math.fsum(weight * rows[arm][name] for arm, weight in terms.items())
              for name in METRICS}
    classes = []
    for i, class_id in enumerate(CLASSES):
        correct = {arm: rows[arm]["counts"][i][1] for arm in terms}
        change = math.fsum(terms[arm] * count for arm, count in correct.items())
        classes.append(dict(class_id=class_id, denominator=5, correct_by_arm=correct, delta_recall=change / 5))
    return metric, classes


def verify_pair_rows(effects, tables, data):
    """Cross-check recomputation against producer tables and the separately admitted WP12 values."""
    map_key = lambda p, row: (p, row["comparison"], int(row["seed"]), row["endpoint"], row["head"])
    checked_rows = defaultdict(int)
    for protocol, native in tables.items():
        original = set()
        for row in native["paired.csv"]:
            key = map_key(protocol, row)
            require(key not in original, f"Duplicated native pair: {key}")
            original.add(key)
            result, _ = effects[key]
            for metric in METRICS:
                close(result[f"delta_{metric}"], row[f"delta_{metric}"], key)
            checked_rows[protocol] += 1
        original_classes = set()
        for row in native["paired-per-class.csv"]:
            key = map_key(protocol, row)
            class_id = int(row["class_id"])
            require((*key, class_id) not in original_classes, f"Duplicated native paired class: {key}")
            original_classes.add((*key, class_id))
            value = effects[key][1][class_id - 1]
            require(value["denominator"] == int(row["denominator"])
                    and value["correct_by_arm"] == json.loads(row["correct_by_arm"]), f"Paired tally mismatch: {key}")
            close(value["delta_recall"], row["delta_recall"], key)
        require(original_classes == {(*key, c) for key in original for c in CLASSES},
                f"Incomplete native paired classes: {protocol}")

    def compare_pinned(protocol, name, endpoints):
        for endpoint in ENDPOINTS:
            for head in HEADS:
                for seed in SEEDS:
                    key = (protocol, name, seed, endpoint, head)
                    stated = endpoints[endpoint][head][str(seed)]
                    metric, classes = effects[key]
                    for field in METRICS:
                        close(metric[f"delta_{field}"], stated[f"delta_{field}"], key, tolerance=5e-7)
                    if "correct_delta_by_class" in stated:
                        for row, correct in zip(classes, stated["correct_delta_by_class"], strict=True):
                            close(row["delta_recall"], correct / 5, key)
                    checked_rows[protocol] += 1

    for protocol, folder in (("fixed-edge-weights", "edge"), ("message-correspondence", "message")):
        for name, record in data[f"{folder}/supplement.json"]["comparisons"].items():
            compare_pinned(protocol, name, record["endpoints"])
    matched = data["matched/supplement.json"]
    for name, record in {**matched["effects"], **matched["conditional"]}.items():
        compare_pinned("matched-edge-message", name, record["endpoints"])
    residual = data["residual/supplement.json"]
    for name, record in residual["pairs"].items():
        protocol = "legacy-direct-joint" if name in ("I7-I6", "I8-I7") else "residual-assignment"
        compare_pinned(protocol, name, record["endpoints"])
    for name, record in residual["factorials"].items():
        protocol = "legacy-direct-joint" if name.startswith("(I") else "residual-assignment"
        compare_pinned(protocol, name, record["effects"])
    return dict(checked_rows)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(out, cohorts, contrasts, effects, aliases):
    tables = defaultdict(list)
    scenario = "Xuzhou"
    for protocol, arms in ARMS.items():
        cases = cohorts[protocol]
        identity = dict(scenario=scenario, protocol=protocol)
        for arm in arms:
            for seed in SEEDS:
                case = cases[seed, arm]
                digest = case["pin"]["sha256"]
                reference = dict(**identity, seed=seed, arm=arm, audit_sha256=digest,
                                 source=case["source"], locator=case["locator"], original_audit=case["pin"]["path"],
                                 distinct_protocol_uses=len(aliases[digest]), hardware=case["costs"]["whole"]["hardware"])
                tables["case-index.csv"].append(reference)
                for (endpoint, head), values in case["endpoints"].items():
                    key = dict(**identity, seed=seed, arm=arm, endpoint=endpoint, head=head, audit_sha256=digest)
                    tables["endpoints.csv"].append(dict(**key, step=values["step"], parameters=values["parameters"],
                                                        support=45, **{name: values[name] for name in METRICS}))
                    for class_id, correct, support in values["counts"]:
                        tables["per-class.csv"].append(dict(**key, class_id=class_id, correct=correct,
                                                            support=support, recall=correct / support))
                for phase in FINAL_PHASES:
                    tables["costs.csv"].append(dict(**identity, seed=seed, arm=arm, audit_sha256=digest,
                                                     phase=phase, **case["costs"][phase]))
            for endpoint in ENDPOINTS:
                for head in HEADS:
                    group = [cases[seed, arm]["endpoints"][endpoint, head] for seed in SEEDS]
                    means = {f"mean_{m}": math.fsum(row[m] for row in group) / len(SEEDS) for m in METRICS}
                    tables["arm-summary.csv"].append(dict(**identity, arm=arm, endpoint=endpoint, head=head,
                                                           seeds="64;512;2048", **means))
        for name, terms in contrasts[protocol].items():
            for endpoint in ENDPOINTS:
                for head in HEADS:
                    by_seed = []
                    for seed in SEEDS:
                        values, classes = effects[protocol, name, seed, endpoint, head]
                        by_seed.append(values)
                        paired = [cases[seed, arm] for arm in terms]
                        devices = {case["costs"]["whole"]["hardware"] for case in paired}
                        role = "released-development"
                        if protocol == "legacy-direct-joint":
                            role = "historical-descriptive-nonmatched-initialization"
                        key = dict(**identity, comparison=name, endpoint=endpoint, head=head, seed=seed)
                        tables["paired.csv"].append(dict(**key, **values, terms=dumps(terms),
                                                          same_device=len(devices) == 1, interpretation=role,
                                                          audit_by_arm=dumps({a: case["pin"]["sha256"]
                                                                              for a, case in zip(terms, paired, strict=True)})))
                        for row in classes:
                            tables["paired-per-class.csv"].append(
                                dict(**key, **{**row, "correct_by_arm": dumps(row["correct_by_arm"])}))
                    means = {f"mean_delta_{metric}": math.fsum(row[f"delta_{metric}"] for row in by_seed) / len(SEEDS)
                             for metric in METRICS}
                    tables["contrast-summary.csv"].append(dict(**identity, comparison=name, endpoint=endpoint,
                                                               head=head, seeds="64;512;2048", **means))
    for name, rows in tables.items():
        write_csv(out / name, rows)
    return {name: len(rows) for name, rows in sorted(tables.items())}


def intake(runs, directory, out, adapter):
    root = directory.resolve(strict=True)
    require(root.is_relative_to(runs) and not out.is_relative_to(root),
            "Use a pinned evidence directory under runs and a separate output directory")
    data = {name: json.loads(checked(root, name, digest)) for name, digest in PINS.items() if name.endswith(".json")}
    mapping = read_csv(root, "protocol-map.csv", PINS["protocol-map.csv"])
    require(len(mapping) == 17, "Incomplete registered R/I protocol map")
    require(all(value.get("query_access", value.get("query", False)) is False
                and value.get("raw_labels", False) is False for value in data.values()),
            "Historical evidence records query or raw label access")
    cohorts, tables, aliases = read_cases(root, data)
    contrasts = contrast_catalog(tables, data)
    effects = {(protocol, name, seed, endpoint, head): compare(cohorts[protocol], seed, endpoint, head, terms)
               for protocol, records in contrasts.items() for name, terms in records.items()
               for seed in SEEDS for endpoint in ENDPOINTS for head in HEADS}
    checked_pairs = verify_pair_rows(effects, tables, data)
    out.mkdir()
    counts = export(out, cohorts, contrasts, effects, aliases)
    receipt = dict(schema="wp12_wp04_comparable_development_v1", adapter=adapter, sources=PINS,
                   scenario="Xuzhou", seeds=SEEDS, validation_points_per_class=5, selected_and_last_step=12,
                   historical_query_access=False, production_accuracy_qualified=False, research_updates=0,
                   cohort_counts={protocol: len(cases) for protocol, cases in cohorts.items()},
                   checked_producer_and_review_pairs=checked_pairs, row_counts=counts,
                   distinct_audits=len(aliases),
                   duplicate_uses={digest: uses for digest, uses in sorted(aliases.items()) if len(uses) > 1},
                   tables={name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in counts})
    (out / "supplement.json").write_text(dumps(receipt) + "\n", encoding="utf-8")

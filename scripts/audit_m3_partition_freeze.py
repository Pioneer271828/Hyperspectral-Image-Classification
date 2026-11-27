"""Recompute graph, score and cost evidence for the released frozen M3 comparison."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess

import numpy as np


ARMS = ("native", "grid", "slic")
HEADS = ("fused", "graph", "cnn")
SEEDS = (64, 512, 2048)
COHORTS = ("c3-pavia", "c3-xuzhou", "c3-honghu", "pavia-40")
PAIRS = (("grid", "native"), ("slic", "native"), ("slic", "grid"))
FOCUS = {"paviaU": (3, 5, 7, 9), "xuzhou": (5, 6), "whu-honghu": (15, 21)}
PAPER = "616d1c849bd3794a8f9c07244b91e18d70899fb9a017b02b4cf6a6b3a5ec476b"
WP12 = "98199bc17929799dfc5f81ef3757e693814413bcf7a360935227d6e7e2dbfbb9"


def check(valid, message):
    if not valid:
        raise ValueError(message)


def json_bytes(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def checked(path, expected):
    check(file_digest(path) == expected, f"Evidence digest differs: {path.name}")
    return path


def capture_index(root, index, expected):
    root = root.resolve(strict=True)
    rows = {}
    for line in checked(index, expected).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (root / name).resolve(strict=True)
        check(name not in rows and len(digest) == 64 and path.is_relative_to(root), "Invalid capture index")
        checked(path, digest)
        rows[name] = digest
    names = {"registration.json"} | {f"captures/{c}-{a}.{e}" for c in COHORTS for a in ARMS for e in ("json", "time")}
    check(rows.keys() == names, "Incomplete capture and timing roster")
    return rows


def array_fingerprint(value, dtype):
    header = json_bytes({"dtype": dtype, "shape": list(value.shape)})
    return hashlib.sha256(header + np.ascontiguousarray(value).tobytes()).hexdigest()


def matrix(record, expected):
    shape = tuple(record["shape"])
    indices = np.asarray(record["indices"], dtype=np.int64)
    values = np.asarray(record["values"], dtype=np.float32)
    check(len(shape) == 2 and min(shape) > 0 and indices.shape == (2, len(values))
          and np.all(np.isfinite(values) & (values > 0)), "Invalid COO shape or nonpositive graph weights")
    r, c = indices
    encoded = r * shape[1] + c
    check(np.all((r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1]))
          and np.all(np.diff(encoded) > 0), "COO indices must be ordered and duplicate free")
    check(np.allclose(np.bincount(r, weights=values, minlength=shape[0]), 1, atol=1e-5, rtol=0),
          "COO rows are not normalized")
    identity = dict(dtype="torch.float32", shape=list(shape),
                    indices=array_fingerprint(indices, "torch.int64"), values=array_fingerprint(values, "torch.float32"))
    check(hashlib.sha256(json_bytes(identity)).hexdigest() == expected, "COO fingerprint differs")
    return shape, r, c, values


def independent_support(labels, shape, *, diagonal):
    """Enumerate undirected touching pixels without using the producer's array slices."""
    h, w = shape
    shifts = [(0, 1), (1, 0), (1, -1), (1, 1)] if diagonal else [(0, 1), (1, 0)]
    for r in range(h):
        for c in range(w):
            i = r*w+c
            for dy, dx in shifts:
                y, x = r+dy, c+dx
                if 0 <= y < h and 0 <= x < w:
                    j = y*w+x
                    if labels[i] != labels[j]:
                        yield i, j


def disconnected(labels, shape):
    h, w = shape
    areas = Counter()
    visited = set()
    for i, label in enumerate(labels):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        areas[int(label)] += 1
        while stack:
            pos = stack.pop()
            y, x = divmod(pos, w)
            for j in (pos-w if y else -1, pos+w if y+1<h else -1,
                      pos-1 if x else -1, pos+1 if x+1<w else -1):
                if j >= 0 and j not in visited and labels[j] == label:
                    visited.add(j)
                    stack.append(j)
    return sum(count > 1 for count in areas.values())


def geometry(row, block, arm):
    h, w, _ = block["shape"]
    witness = row["witness"]
    (pixels, regions), q_i, q_j, q_value = matrix(witness["Q"], row["graph"]["q_sha256"])
    a_shape, a_i, a_j, _ = matrix(witness["A0"], row["graph"]["a_sha256"])
    check(pixels == h*w and a_shape == (regions, regions), "Actual Q/A0 shape differs")
    positions = np.asarray(witness["positions"], dtype=np.float64)
    check(positions.shape == (regions, 2) and np.all(np.isfinite(positions) & (positions >= 0) & (positions <= 1)),
          "Invalid normalized region coordinates")
    check(set(a_i[a_i == a_j].tolist()) == set(range(regions)), "Graph lost mandatory self-loops")
    if arm in ("native", "grid"):
        check(regions == block["native_budget"], "Native and Grid region budgets must match exactly")
        gh, gw = block["grid"]
        expected = {(a, b) for a in range(regions) for b in range(regions)
                    if abs(a//gw - b//gw) <= 1 and abs(a%gw - b%gw) <= 1}
        check(expected == set(zip(a_i, a_j)), "Original local grid support differs")
    mass = np.bincount(q_j, weights=q_value, minlength=regions)
    hard = np.empty(pixels, dtype=np.int64)
    maximum = np.full(pixels, -np.inf)
    for i, j, value in zip(q_i, q_j, q_value, strict=True):
        if value > maximum[i]:
            maximum[i], hard[i] = value, j
    area = np.bincount(hard, minlength=regions)
    centroid_error = None
    if arm != "native":
        check(np.array_equal(q_i, np.arange(pixels)) and np.all(q_value == 1)
              and np.all(area > 0), "Fixed Q must assign one nonempty region per pixel")
        if arm == "grid":
            gh, gw = block["grid"]
            canonical = [(i//w*gh//h)*gw + (i%w*gw//w) for i in range(pixels)]
            check(np.array_equal(canonical, hard), "The registered Grid assignment was bypassed")
        else:
            check(np.array_equal(hard, block["slic_members"]), "The registered SLIC assignment was bypassed")
            idx = np.arange(pixels)
            points = np.column_stack(((idx//w + .5)/h, (idx%w + .5)/w))
            exact = np.stack([points[hard == j].mean(0) for j in range(regions)])
            fp32 = np.zeros((regions, 2), dtype=np.float32)
            np.add.at(fp32, hard, points.astype(np.float32))
            fp32 /= area[:, None].astype(np.float32)
            check(np.array_equal(positions.astype(np.float32), fp32),
                  "Recorded SLIC centers differ from independent FP32 accumulation")
            centroid_error = float(np.max(np.abs(positions - exact)))
            expected = {(j, j) for j in range(regions)}
            for i, j in independent_support(hard, (h, w), diagonal=True):
                expected |= {(int(hard[i]), int(hard[j])), (int(hard[j]), int(hard[i]))}
            check(expected == set(zip(a_i, a_j)), "SLIC adjacency differs from actual touching regions")

    columns = np.asarray(witness["input_columns"], dtype=np.float64)
    region_output = np.asarray(witness["region_output"], dtype=np.float64)
    check(columns.shape == (pixels, 2) and region_output.shape == (regions, 2),
          "Invalid observed projection witnesses")
    pool = np.zeros((regions, 2), dtype=np.float64)
    np.add.at(pool, q_j, columns[q_i] * (q_value / np.maximum(mass[q_j], 1e-6))[:, None])
    broadcast = np.zeros((pixels, 2), dtype=np.float64)
    np.add.at(broadcast, q_i, region_output[q_j] * q_value[:, None])
    pool_error = float(np.max(np.abs(pool - np.asarray(witness["pooled"]))))
    broadcast_error = float(np.max(np.abs(broadcast - np.asarray(witness["broadcast"]))))
    check(pool_error <= 2e-5 and broadcast_error <= 2e-5, "Independent Q pool or backprojection differs")
    borders = set(independent_support(hard, (h, w), diagonal=False))
    known = {point["offset"]: point["class_id"] for point in block["points"]}
    neighbor_pairs = [(i, j) for i in known for j in (i+1, i+w)
                      if j in known and (j != i+1 or i//w == j//w)]
    semantic = [(i, j) for i, j in neighbor_pairs if known[i] != known[j]]
    by_region = defaultdict(Counter)
    for pixel, class_id in known.items():
        by_region[hard[pixel]][class_id] += 1
    purity_numerator = sum(max(hist.values()) for hist in by_region.values())
    return dict(actual_regions=regions, requested_native_budget=block["native_budget"], q_nnz=len(q_value),
                a_nnz=len(a_i), hard_nonempty_regions=int(np.count_nonzero(area)),
                hard_areas=dumps(area.tolist()), soft_masses=dumps(mass.tolist()),
                hard_area_min=int(area.min()), hard_area_max=int(area.max()),
                hard_area_median=float(np.median(area)), split_hard_regions=disconnected(hard, (h, w)),
                four_neighbor_boundaries=len(borders), known_neighbor_pairs=len(neighbor_pairs),
                known_semantic_boundaries=len(semantic), captured_semantic_boundaries=sum(p in borders for p in semantic),
                labeled_region_purity_numerator=purity_numerator, labeled_points=len(known),
                pool_max_abs_error=pool_error, broadcast_max_abs_error=broadcast_error,
                centroid_max_abs_error_fp64=centroid_error), hard


def dumps(value):
    return json_bytes(value).decode()


def scores(labels, probabilities, classes):
    estimate = np.asarray(probabilities, dtype=np.float64)
    check(estimate.shape == (len(labels), classes) and np.isfinite(estimate).all()
          and np.all(estimate >= 0) and np.allclose(estimate.sum(1), 1.0, atol=1e-6, rtol=0), "Bad three-head probabilities")
    truth = np.asarray(labels, dtype=np.int64) - 1
    prediction = estimate.argmax(1)
    cm = np.bincount(truth*classes + prediction, minlength=classes**2).reshape(classes, classes)
    support, correct = cm.sum(1), np.diag(cm)
    present = support > 0
    loss = -np.log(np.maximum(estimate[np.arange(len(truth)), truth], 1e-8))
    result = dict(support=len(truth), correct=int(correct.sum()), oa=float(correct.sum()/len(truth)),
                  aa=float(np.mean(correct[present]/support[present])), nll=float(loss.mean()))
    return result, cm, prediction + 1, loss


def write_table(out, name, values):
    with (out / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return file_digest(out / name)


def benchmark(capture, wall, identity):
    samples = [t for record in capture["rows"] for t in record["times"]]
    warm = [t["forward_ms"] for record in capture["rows"] for t in record["times"][1:]]
    check(len(samples) == capture["timed_forwards"] == len(capture["rows"])*3
          and capture["proof_forwards"] == len(capture["rows"])
          and all(0 < t["build_ms"] <= t["forward_ms"] for t in samples), "Bad timing population")
    elapsed, rss_mib = wall.read_text().strip().split(",")
    return dict(**identity, blocks=len(capture["rows"]), timed_forwards=len(samples), proof_forwards=len(capture["rows"]),
                model_load_seconds=sum(row["checkpoint_load_seconds"] for row in capture["models"]),
                input_pins_and_blocks_seconds=capture["inputs_and_identity_seconds"],
                process_to_first_prediction_seconds=capture["process_to_first_prediction_seconds"],
                process_wall_seconds=float(elapsed), process_peak_rss_mib=int(rss_mib)/1024,
                rss_before_serializing_mib=capture["process_peak_rss_bytes"]/1024**2,
                all_forward_seconds=sum(t["forward_ms"] for t in samples)/1000,
                graph_build_p50_ms=median(t["build_ms"] for t in samples),
                forward_p50_ms=median(t["forward_ms"] for t in samples),
                warm_forward_p50_ms=median(warm), warm_forward_p95_ms=float(np.percentile(warm, 95)))


def import_cohort(root, index, plan, cohort, output):
    spec = plan["cohorts"][cohort]
    scene = spec["scene"]
    identity = dict(cohort=cohort, scene=scene, training_sweeps=spec["sweeps"])
    bank = plan["banks"][scene]
    classes = bank["classes"]
    blocks = {row["unit"]: row for row in bank["blocks"]}
    weights = {row["seed"]: row for row in spec["cases"]}
    captures, distinct_states = {}, {}
    for arm in ARMS:
        stem = f"captures/{cohort}-{arm}"
        data = json.loads((root / f"{stem}.json").read_text())
        check(data["registration_sha256"] == index["registration.json"] and data["producer"] == plan["producer"]
              and data["arm"] == arm and data["cohort"] == cohort and data["optimizer_updates"] == 0
              and data["query_accessed"] is False, "Capture role or registration disagrees")
        observed_states = [(r["seed"], r["state_sha256"], r["parameters"]) for r in data["models"]]
        if arm == "native":
            distinct_states = observed_states
        else:
            check(distinct_states == observed_states, "Candidate weights differ from native")
        captures[arm] = {(r["seed"], r["unit"]): r for r in data["rows"]}
        check(len(data["rows"]) == len(captures[arm]) and captures[arm].keys() ==
              {(seed, unit) for seed in SEEDS for unit in blocks}, "One or more sample blocks were excluded")
        output["process-cost.csv"].append(benchmark(data, root / f"{stem}.time", dict(**identity, arm=arm)))

    for seed in SEEDS:
        metrics, correct_by_arm, predictions, losses, supports = {}, {}, {}, {}, {}
        labels = [p["class_id"] for block in blocks.values() for p in block["points"]]
        point_ids = [p["point"] for block in blocks.values() for p in block["points"]]
        units = [unit for unit, block in blocks.items() for p in block["points"]]
        geometry_by_arm = {}
        for arm in ARMS:
            graph_rows, prob_rows = {}, []
            for unit, block in blocks.items():
                source = captures[arm][seed, unit]
                check([p["point"] for p in source["points"]] == [p["point"] for p in block["points"]],
                      "Released point identity or order changed")
                check(source["heads"]["cnn"] == captures["native"][seed, unit]["heads"]["cnn"],
                      "CNN changed when only the graph was replaced")
                measured, hard = geometry(source, block, arm)
                same_budget = measured["actual_regions"] == block["native_budget"]
                key = dict(**identity, seed=seed, arm=arm, unit=unit, height=block["shape"][0], width=block["shape"][1])
                output["block-geometry.csv"].append(dict(**key, **measured, same_region_budget=same_budget))
                for n, t in enumerate(source["times"], 1):
                    output["block-cost.csv"].append(dict(**key, sample=n, **t))
                graph_rows[unit] = (measured["actual_regions"], hard)
                prob_rows.extend(p["probabilities"] for p in source["points"])
            geometry_by_arm[arm] = graph_rows
            for i, head in enumerate(HEADS):
                key = dict(**identity, seed=seed, arm=arm, head=head, selected_step=weights[seed]["checked_step"])
                row, cm, pred, loss = scores(labels, [prob[i] for prob in prob_rows], classes)
                check(cm.sum(1).tolist() == [5]*classes, "Unexpected class denominator")
                if arm == "native" and head == "fused":
                    expected = weights[seed]["expected_fused"]
                    check(cm.tolist() == expected["confusion"] and math.isclose(row["nll"], expected["nll"], abs_tol=2e-6, rel_tol=0),
                          "Native frozen reference did not reproduce the accepted classification")
                metrics[arm, head] = row
                predictions[arm, head], losses[arm, head] = pred, loss
                correct_by_arm[arm, head] = pred == np.asarray(labels)
                supports[arm, head] = cm
                output["three-heads.csv"].append(dict(**key, **row, confusion=dumps(cm.tolist())))
                for c, (count, correct) in enumerate(zip(cm.sum(1), cm.diagonal(), strict=True), 1):
                    output["per-class.csv"].append(dict(**key, class_id=c, support=int(count), correct=int(correct),
                                                       recall=float(correct/count), focal_class=c in FOCUS[scene]))
        for candidate, reference in PAIRS:
            equal = {unit for unit in blocks if geometry_by_arm[candidate][unit][0] == geometry_by_arm[reference][unit][0]}
            subset = np.array([unit in equal for unit in units])
            for unit in blocks:
                _, a = geometry_by_arm[reference][unit]
                _, b = geometry_by_arm[candidate][unit]
                agree = lambda counts: sum(v*(v-1)//2 for v in counts.values())
                diff = agree(Counter(a)) + agree(Counter(b)) - 2*agree(Counter(zip(a, b)))
                output["partition-pairs.csv"].append(dict(**identity, seed=seed, unit=unit, arm=candidate,
                                                          reference=reference, same_region_budget=unit in equal,
                                                          changed_coassignment_pairs=diff))
            for head in HEADS:
                key = dict(**identity, seed=seed, arm=candidate, reference=reference, head=head)
                a, b = correct_by_arm[reference, head], correct_by_arm[candidate, head]
                p, q = predictions[reference, head], predictions[candidate, head]
                loss_a, loss_b = losses[reference, head], losses[candidate, head]
                lost, gained = a & ~b, ~a & b
                check(np.count_nonzero(gained) - np.count_nonzero(lost) == int(b.sum()) - int(a.sum()), "Invalid flip accounting")
                output["paired-heads.csv"].append(dict(**key, support=len(labels), correct_to_wrong=int(lost.sum()),
                    wrong_to_correct=int(gained.sum()), changed_prediction=int((p != q).sum()),
                    delta_oa_pp=100*float(np.mean(b.astype(int)-a.astype(int))),
                    delta_aa_pp=100*(metrics[candidate, head]["aa"]-metrics[reference, head]["aa"]),
                    delta_nll=float(np.mean(loss_b-loss_a)), same_budget_blocks=len(equal), same_budget_support=int(subset.sum()),
                    same_budget_delta_oa_pp=100*float(np.mean(b[subset].astype(int)-a[subset].astype(int))) if subset.any() else None,
                    same_budget_delta_nll=float(np.mean((loss_b-loss_a)[subset])) if subset.any() else None))
                for i, pid in enumerate(point_ids):
                    output["paired-flips.csv"].append(dict(**key, point=pid, class_id=labels[i],
                        reference_prediction=int(p[i]), candidate_prediction=int(q[i]), lost=bool(lost[i]),
                        gained=bool(gained[i]), delta_nll=float(loss_b[i]-loss_a[i]), same_region_budget=bool(subset[i])))
                ref_cm, new_cm = supports[reference, head], supports[candidate, head]
                for i in range(classes):
                    output["paired-classes.csv"].append(dict(**key, class_id=i+1, support=5, focal_class=i+1 in FOCUS[scene],
                        reference_correct=int(ref_cm[i, i]), candidate_correct=int(new_cm[i, i]),
                        delta_recall_pp=20*int(new_cm[i, i]-ref_cm[i, i])))


def supplement(plan, index, output, out, prior_paper, prior_wp12, adapter):
    paper = checked(prior_paper, PAPER).read_bytes()
    checked(prior_wp12, WP12)
    summary, append = [], []
    for cohort in COHORTS:
        scene = plan["cohorts"][cohort]["scene"]
        for arm, reference in PAIRS:
            for head in HEADS:
                rows = [r for r in output["paired-heads.csv"] if
                        (r["cohort"], r["arm"], r["reference"], r["head"]) == (cohort, arm, reference, head)]
                check([r["seed"] for r in rows] == list(SEEDS), "Incomplete seeded contrast")
                summary.append(dict(cohort=cohort, scene=scene, arm=arm, reference=reference, head=head,
                    support_per_seed=rows[0]["support"], mean_delta_oa_pp=math.fsum(r["delta_oa_pp"] for r in rows)/3,
                    mean_delta_aa_pp=math.fsum(r["delta_aa_pp"] for r in rows)/3,
                    mean_delta_nll=math.fsum(r["delta_nll"] for r in rows)/3,
                    total_correct_to_wrong=sum(r["correct_to_wrong"] for r in rows),
                    total_wrong_to_correct=sum(r["wrong_to_correct"] for r in rows)))
                if head == "fused":
                    item = summary[-1]
                    append.append(dict(id=f"M3-{cohort}-{arm}-{reference}", manuscript_location="Regional mechanism supplement",
                        claim_or_table=(f"Frozen {arm} versus {reference}: mean fused OA difference "
                                        f"{item['mean_delta_oa_pp']:+.2f} percentage points and NLL difference "
                                        f"{item['mean_delta_nll']:+.6f}."),
                        source_revision=plan["producer"], source_path="deidentified/paired-heads.csv",
                        locator=f"{cohort}, {arm} vs {reference}, fused, seeds 64/512/2048",
                        units_and_protocol=f"{rows[0]['support']} shared labeled development points per seed",
                        qualification="Frozen development sensitivity; no independent confirmation or training adaptation; actual region budget is in the block table"))
    out.mkdir()
    output["cohort-summary.csv"] = summary
    output["paper-evidence-append.csv"] = append
    digests = {name: write_table(out, name, rows) for name, rows in output.items()}
    addition = (out / "paper-evidence-append.csv").read_bytes().split(b"\n", maxsplit=1)[1]
    combined = out / "paper-evidence-unified.csv"
    combined.write_bytes(paper + (b"" if paper.endswith(b"\n") else b"\n") + addition)
    digests[combined.name] = file_digest(combined)
    return dict(status="passed", schema="wp12_m3_frozen_partition_development_v1", adapter=adapter,
                producer=plan["producer"], registration_sha256=index["registration.json"], captures=index,
                sources={key: {"historical_review_sha256": spec["review_sha256"],
                               "trained_sweeps": spec["sweeps"], "scene": spec["scene"]}
                         for key, spec in plan["cohorts"].items()},
                acceptance=dict(role="devval", scientific_scope="frozen_q_and_a_sensitivity", optimizer_updates=0,
                                query=False, independent_confirmation=False, production_accuracy_qualified=False),
                table_sha256=digests, table_rows={name: len(rows) for name, rows in output.items()},
                links=dict(prior_wp12_sha256=WP12, prior_paper_sha256=PAPER,
                           prior_paper_root="../paper-a-region-factorial-20261127",
                           method="keep all existing paper rows; append separately scoped M3 entries"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "index", "prior-paper", "prior-wp12", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("index-sha256", "expected-commit"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(["git", "-C", checkout, *a], text=True).strip()
    check(git("rev-parse", "HEAD") == args.expected_commit and not git("status", "--porcelain")
          and not args.out.exists() and not args.out.resolve().is_relative_to(checkout), "Use a clean consumer and fresh output")
    index = capture_index(args.input, args.index, args.index_sha256)
    registration = json.loads((args.input / "registration.json").read_text())
    check(registration["schema"] == "m3_partition_registration_v1" and registration["optimizer_updates"] == 0
          and not registration["query_accessed"] and tuple(registration["heads"]) == HEADS
          and tuple(registration["arms"]) == ARMS, "Wrong registration")
    output = defaultdict(list)
    for cohort in COHORTS:
        import_cohort(args.input, index, registration, cohort, output)
    receipt = supplement(registration, index, output, args.out, args.prior_paper, args.prior_wp12, args.expected_commit)
    (args.out / "wp12-m3-partition.json").write_text(dumps(receipt) + "\n", encoding="utf-8")
    print(dumps({"status": "passed", "rows": receipt["table_rows"], "tables": receipt["table_sha256"]}))


if __name__ == "__main__":
    main()

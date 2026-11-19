"""Summarize already published M1/E1 training without opening images, labels or weights."""

from collections import defaultdict
from itertools import product
from math import isclose
from statistics import mean, stdev

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from model.research_config import digest

SCHEMA = "wp12_m1_e1_frozen_training_history_v1"
SEEDS = (64, 512, 2048)
FOCUS = {"paviaU": (2, 6), "xuzhou": (5, 6), "whu-honghu": (15, 21)}
JOINT_SCENES = {"paviaU": 9, "xuzhou": 9, "whu-honghu": 22}
M1_C = "m1c-paired-screen-20261102-01"
E1_C2 = "e1c-issue89-attempt02/registered-full-budget"
M1_J2 = "m1j2-issue127-analysis01"
PINS = {
    f"{M1_C}/plan.json": "bbbaeb4f2fa3e5cff5cf998ccce4521bdf7f034574eedd7de3ba53de0af3082c",
    f"{M1_C}/paired-results.json": "c4504bb353bbb3a2d00e3ae7711542be8dd7385ab350b4bb48004e21989f7b93",
    f"{E1_C2}/plan.json": "09c51550b4d070ceb4d7b2a0f0e976492862a65e314873913265cdf179c135a4",
    f"{E1_C2}/paired-results.json": "646a8c1e6cf0bc48744fe4f9f61a79e3fc0aba5d2ead2fc9c7e467e1416a4178",
    f"{E1_C2}/launches/issue91-20261102-01/final-acceptance.json":
        "561acd5bafd7c7efdface0331fc8acb3ffead03279c5e145893773269bdb2663",
    f"{M1_J2}/results.json": "c393f6491a0156c2bc19118805fbde7fd9ed81f78373c225c3a6287125f2e642",
    f"{M1_J2}/independent-receipt.json": "89397b842d9f5b30ec3993aa1aa52b5b334bfe494a4569e402ff010cb7e5bb6e",
    f"{M1_J2}/frozen.json": "b11bd838be6bb1650bb24a6d5b001720fa36e54b8920a266f2650a11dc607dea",
    "m1j2-issue127-full01/status.json": "2a86708558fba3a9b641fd734020ad6d2e1cddba0b5bbeaaa9bb40335f373b91",
    "wp05-wp06-external-paired-20261119-qa/delivery/wp12-external-paired-index.json":
        "642c38f8262dc7bce187f717f5aab9ee218af5034b9250a95caae2f711013979",
}


class Archive:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def read(self, relative, expected=None):
        path = self.root / relative
        require(path.suffix == ".json", "Only frozen summary and case JSON are admissible")
        pin = (dict(path=str(path), sha256=expected or PINS[relative])
               if expected or relative in PINS else file_pin(path))
        row = json_pin(self.root, pin)
        self.files[relative] = pin["sha256"]
        return row


def scores(record, scene):
    width = JOINT_SCENES[scene]
    matrix = record.get("confusion_matrix", record.get("confusion"))
    if matrix is None:
        return dict(oa=record["oa"], aa=record["aa"], denominator=None,
                    focus={str(c): dict(correct=None, n=None) for c in FOCUS[scene]},
                    counts_source="summary_only")
    require(len(matrix) == width and all(len(row) == width and all(
            type(count) is int and count >= 0 for count in row) for row in matrix),
            "Historical confusion matrix has the wrong class system")
    recalls = []
    focus = {}
    for label, row in enumerate(matrix, 1):
        n, correct = sum(row), row[label-1]
        if n:
            recalls.append(correct / n)
        if label in FOCUS[scene]:
            focus[str(label)] = dict(correct=correct, n=n)
    total = sum(map(sum, matrix))
    oa = sum(matrix[i][i] for i in range(width))/total if total else None
    aa = mean(recalls) if recalls else None
    require(total and isclose(oa, record["oa"], abs_tol=1e-9)
            and isclose(aa, record["aa"], abs_tol=1e-9), "Archived OA/AA disagree with integer counts")
    require("count" not in record or total == record["count"], "Historical class denominator changed")
    return dict(oa=oa, aa=aa, denominator=total, focus=focus, counts_source="frozen_confusion_diagonal")


def stat(values):
    return dict(values=values, mean=mean(values), sample_sd=stdev(values) if len(values)>1 else None)


def read_screen(archive, directory, modes, commit, expected_decision):
    plan = archive.read(f"{directory}/plan.json")
    paired = archive.read(f"{directory}/paired-results.json")
    require(plan["code"]["commit"] == commit and plan["seeds"] == list(SEEDS)
            and set(plan["modes"]) == set(modes) and plan["updates"] == 5000
            and plan["val_every"] == 100 and paired["plan_hash"] == plan["plan_hash"]
            and paired["status"] == "completed" and paired["decision"] == expected_decision
            and paired["test_evaluated"] is False and paired["target_updates"] == 5000*len(modes)*len(SEEDS),
            "Registered development archive identity or budget differs")
    indexed = {(row["seed"], row["mode"]):row for row in paired["rows"]}
    require(len(indexed) == len(paired["rows"]) == len(SEEDS)*len(modes), "Frozen arm roster differs")
    results = []
    for seed, arm in product(SEEDS, modes):
        place = f"{directory}/cases/seed-{seed}/{arm}"
        case = archive.read(f"{place}/case.json")
        review = archive.read(f"{place}/review.json", case["artifacts"]["review.json"])
        aggregate = indexed[seed, arm]
        require(case["receipt_hash"] == digest({key: value for key, value in case.items() if key != "receipt_hash"})
                and case["status"] == "completed" and case["seed"] == seed and case["mode"] == arm
                and case["plan_hash"] == plan["plan_hash"] and case["optimizer_steps"] == 5000
                and case["evaluation_bank_hash"] == plan["evaluation_bank_hash"]
                and review["test_evaluated"] is False and review["selection_condition"] == "full"
                and review["selected_step"] == aggregate["selected_step"]
                and review["selected_step"] == min(review["validation_curve"], key=lambda row: row["nll"])["step"]
                and all(isclose(measures["fused"]["oa"], aggregate["condition_oa"][view], abs_tol=1e-9)
                        for view, measures in review["conditions"].items()),
                "Development case or its published selection/readout is foreign")
        results.append(dict(scene="xuzhou", arm=arm, seed=seed, updates=5000,
            role="fixed_development_validation", selected_step=review["selected_step"], checkpoint_sha256=case["artifacts"]["training/model.pt"],
            views={name: {h: scores(value[h], "xuzhou") for h in HEADS} for name, value in review["conditions"].items()},
            cost=dict(wall_seconds=aggregate["seconds"], peak_cuda_bytes=review["participation"]["peak_training_allocated_bytes"]),
            parameter_count=review["participation"]["parameters"], train_widths=review["participation"]["train_width_counts"],
            observations=case["evaluation_bank_hash"], source_case=f"{place}/case.json", source_metrics=f"{place}/review.json"))
    return dict(source_commit=commit, archived_gate=paired["decision"], total_updates=paired["target_updates"],
                registered_aggregate=paired["summary"], registered_paired_differences=paired["paired_differences"],
                rows=results, k=dict(support=10, development=5), populations=dict(support=90, development=45),
                graph_context=dict(geometry="same_registered_tiles", input="actual_observation_per_tile",
                                   graph="learned_from_each_supplied_observation", old_zero="full_axis_and_non_support_values_retained"
                                   if directory==M1_C else None))


def read_joint(archive):
    results, qa, freeze = (archive.read(f"{M1_J2}/{name}.json")
                           for name in ("results","independent-receipt","frozen"))
    status = archive.read("m1j2-issue127-full01/status.json")
    commit = "9942c0f31949bed746f3038e26b53b51d0dddc26"
    require(results["source"] == freeze["source"] and results["source"]["commit"] == qa["training_commit"] == commit
            and results["test_scoring"] is True and results["optimizer_updates"] == qa["new_training_updates"] == 0
            and qa["status"] == "passed" and qa["head_metrics_recomputed"] == 216
            and qa["batch_status"] == status and status["launch_sha256"] == freeze["launch_sha256"]
            and status["status"] == "training_completed" and status["test_scoring"] is False,
            "Historical joint receipt lost its frozen verified provenance")
    views = ("full","random75","random50","stride2")
    arms = ("full","alternate75")
    archived = {(row["dataset"],row["preset"],row["seed"],row["condition"]):row for row in results["rows"]}
    require(set(archived) == set(product(JOINT_SCENES,arms,SEEDS,views)) and len(archived) == len(results["rows"]),
            "Incomplete or duplicate frozen joint measurements")
    jobs = {(row["preset"],row["seed"]):row for row in status["jobs"]}
    require(len(jobs) == 6 and len(status["jobs"]) == 6, "Joint job receipts differ")
    rows = []
    for scene, arm, seed in product(JOINT_SCENES,arms,SEEDS):
        training = qa["training_reviews"][f"{arm}-{seed}"]
        job = jobs[arm, seed]
        require(training["updates"] == 1503 and training["test_scoring"] is False
                and job["status"] == "completed" and job["exit_code"] == 0,
                "A reported full-budget joint case was unfinished")
        points = {}
        for view in views:
            entry = archived[scene,arm,seed,view]
            frozen = freeze["evaluation_bank"][view][scene]
            require(entry["bands"] == len(frozen["selected_band_ids"]), "Frozen retained-column width differs")
            points[view] = {h: scores(entry["scores"][h], scene) for h in HEADS}
        rows.append(dict(scene=scene,arm=arm,seed=seed,updates=1503,selected_step=training["selected_step"],
            role="historical_fixed_transductive_joint_test",views=points,
            train_widths=training["observed_band_counts"],
            cost=dict(wall_seconds=job["training_seconds"], peak_cuda_bytes=None,
                      sharing="one_three_scene_training_job_count_once_per_arm_seed"),
            sources=dict(metrics=f"{M1_J2}/results.json", training=f"{M1_J2}/independent-receipt.json",
                         freeze=f"{M1_J2}/frozen.json")))
    return dict(source_commit=commit,archived_gate=qa["gates"],total_updates=6*1503,rows=rows,
        k=None,support_per_class=results["train_counts"], calibration="historical_joint_1pct_train_0_5pct_validation",
        graph_context=dict(graph="fixed_legacy_full_scene_q_and_a", normalization="historical_unlabelled_domain_bn_calibration",
                           independent_spatial_bank=False), registered_paired_differences=qa["paired"],
        evaluation_seconds=results["elapsed_seconds"],batch_wall_seconds=status["elapsed_seconds"],
        review_input_opened=False,scope="published_aggregate_of_historical_scoring")


def compare(rows, baseline):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["scene"],row["arm"]][row["seed"]] = row
    output = {}
    for (scene, arm), cases in grouped.items():
        reference = grouped[scene, baseline]
        require(tuple(sorted(cases)) == SEEDS and tuple(sorted(reference)) == SEEDS, "Incomplete matched seed series")
        by_view = {}
        for view in cases[64]["views"]:
            heads = {}
            for head in HEADS:
                before = [reference[s]["views"][view][head] for s in SEEDS]
                after = [cases[s]["views"][view][head] for s in SEEDS]
                require([a["denominator"] for a in before] == [a["denominator"] for a in after],
                        "Matched protocol uses a different evaluation denominator")
                metrics = {m: stat([a[m] for a in after]) for m in ("oa","aa")}
                metrics["delta_from_training_baseline"] = {m: stat([b[m]-a[m] for a,b in zip(before,after,strict=True)])
                                                            for m in ("oa","aa")}
                full = [cases[s]["views"]["full"][head] for s in SEEDS]
                metrics["delta_from_same_arm_full"] = {m: stat([b[m]-a[m] for a,b in zip(full,after,strict=True)])
                                                         for m in ("oa","aa")}
                metrics["focus"] = {}
                for c in map(str,FOCUS[scene]):
                    a,b = ([row["focus"][c] for row in source] for source in (before,after))
                    require([r["n"] for r in a] == [r["n"] for r in b], "Matched class denominators differ")
                    metrics["focus"][c] = dict(baseline=a, candidate=b, delta_correct=[
                        last["correct"]-first["correct"] if first["n"] is not None else None
                        for first,last in zip(a,b,strict=True)])
                heads[head] = metrics
            by_view[view] = heads
        output.setdefault(scene, {})[arm] = by_view
    return output


def index(root):
    archive = Archive(root)
    m1c = read_screen(archive,M1_C,("full","legacy_support_zero","real_subset"),
                      "2e2200ebc262e20070e5a346945b1cd2e2cfefb2","no_joint_accuracy_robustness_gain_keep_full")
    e1 = read_screen(archive,E1_C2,("none","anchor_only","binary","continuous","shuffled"),
                     "21f72ad073737144896c4c3d8937b1ba43b8fe00","no_joint_coverage_gain_keep_e1_off")
    e1_receipt = f"{E1_C2}/launches/issue91-20261102-01/final-acceptance.json"
    archive.read(e1_receipt)
    e1["source_batch_acceptance"] = e1_receipt
    joint = read_joint(archive)
    experiments = {"m1_c":m1c, "m1_j2":joint, "e1_c2":e1}
    for name, baseline in (("m1_c","full"),("m1_j2","full"),("e1_c2","none")):
        row = experiments[name]
        row["baseline"] = baseline
        row["same_protocol"] = compare(row["rows"],baseline)
    attribution = next(name for name in PINS if name.endswith("wp12-external-paired-index.json"))
    archive.read(attribution)
    return dict(schema=SCHEMA, inputs=archive.files, experiments=experiments,
                prior_attribution=dict(attachment=attribution, relationship="separate_released_development_protocols"),
                missing_conditions=dict(m1_j2_central_gap="unobserved", paviaU_honghu_e1c2="unobserved",
                                        strictly_independent_physical_confirmation="unreleased",
                                        legacy_case_inference_p95="unrecorded_in_these_training_archives"),
                cross_protocol_pooled=False, raw_gt_loaded=False, query_accessed=False, optimizer_updates=0)

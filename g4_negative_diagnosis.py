"""Read-only, post hoc Issue184 diagnostics from accepted outputs and imagery."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from scipy.io import loadmat
import torch

from g4_matched_target_analysis import FT, HEADS, SCRATCH, _value, validate_rows
from g4_matched_target_study import MODES, SEEDS
from g4_support_context import LABEL_KEYS, overlay_identity, validate_overlay
from joint_baseline import NAMES, _item_fingerprint
from loadData.dataset_registry import DATASETS, file_hash
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_bn_calibration import OBS_KEYS


INPUTS = {
    "analysis": ("g4-matched-target-issue184-analysis01/analysis.json",
                 "580b0b5c358d1818af2a97e9f0c7c69cc393d1ac9d54473a535318ef2a80752d"),
    "review": ("g4-matched-target-issue184-review01/summary.json",
               "547f7a748a6ed7540366a0f8afe851ae7ad33da37d08891213cc64dc156af7d0"),
    "prepared": ("base97a-honghu-issue97-20261103-prepared01/manifest.json",
                 "414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b"),
    "contexts": ("g4-support-context-issue188-preflight01/manifest.json",
                 "57d13699414b1c041c5332d834a392e7d3af499684a56886d75e76c3c87d66c6"),
}
QUANTILES = (.01, .05, .5, .95, .99)


def distribution(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def spectrum(covariance):
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    total = eigenvalues.sum()
    if total == 0:
        return {"participation_dimension": 0.0, "pca95": 0, "pca99": 0,
                "cumulative_variance": [0.0] * len(eigenvalues)}
    cumulative = np.cumsum(eigenvalues) / total
    return {"participation_dimension": float(total ** 2 / np.square(eigenvalues).sum()),
            "pca95": int(np.searchsorted(cumulative, .95) + 1),
            "pca99": int(np.searchsorted(cumulative, .99) + 1),
            "cumulative_variance": cumulative.tolist()}


def spectral_statistics(sample):
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 2 or min(sample.shape) < 2 or not np.isfinite(sample).all():
        raise ValueError("Finite pixel-by-band sample with at least two rows/bands required")
    covariance = np.cov(sample, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    valid = std > 0
    scaled = covariance[np.ix_(valid, valid)] / np.outer(std[valid], std[valid])
    adjacent_valid = valid[:-1] & valid[1:]
    adjacent = np.diag(covariance, 1)[adjacent_valid] / (std[:-1] * std[1:])[adjacent_valid]
    return {"sample_pixels": len(sample), "bands": sample.shape[1],
            "constant_bands": int((~valid).sum()), "covariance": spectrum(covariance),
            "correlation": spectrum(scaled),
            "adjacent_band_correlation_p05_median":
                np.quantile(adjacent, [.05, .5]).tolist() if len(adjacent) else None,
            "global_minmax_values_p01_p05_p50_p95_p99": np.quantile(sample, QUANTILES).tolist(),
            "band_std_min_median_max": [float(std.min()), float(np.median(std)), float(std.max())]}


def graph_statistics(observation):
    q = observation["Q"].coalesce()
    row, region = q.indices().numpy()
    pixels, regions = q.shape
    if (len(row) != pixels or not np.array_equal(row, np.arange(pixels))
            or not torch.equal(q.values(), torch.ones_like(q.values()))):
        raise ValueError("This diagnostic requires the registered hard one-region-per-pixel Q")
    sizes = np.bincount(region, minlength=regions)
    adjacency = observation["A"].numpy() > 0
    np.fill_diagonal(adjacency, False)
    degree = adjacency.sum(axis=1)
    return region, {"regions": regions, "pixels": pixels,
        "region_size_min_p05_median_p95_max": np.quantile(sizes, [0, .05, .5, .95, 1]).tolist(),
        "mean_region_pixels": float(sizes.mean()),
        "off_diagonal_degree_min_median_max": [int(degree.min()), float(np.median(degree)), int(degree.max())],
        "isolated_regions": int((degree == 0).sum()), "graph_fit_label_count": 0}


def support_statistics(flat, overlay, region, width):
    train_i, train_y, val_i, val_y = [overlay[k].numpy() for k in
        ("train_indices", "train_labels", "val_indices", "val_labels")]
    classes = int(train_y.max()) + 1
    means = np.stack([flat[train_i[train_y == c]].mean(axis=0) for c in range(classes)])
    within = np.array([np.square(flat[train_i[train_y == c]] - means[c]).mean()
                       for c in range(classes)])
    distances = np.sqrt(np.square(means[:, None] - means[None, :]).mean(axis=2))
    np.fill_diagonal(distances, np.inf)
    per_class = []
    for c in range(classes):
        ti, vi = train_i[train_y == c], val_i[val_y == c]
        nearest = int(distances[c].argmin())
        pooled = float(np.sqrt((within[c] + within[nearest]) / 2))
        train_xy = np.column_stack(np.divmod(ti, width))
        val_xy = np.column_stack(np.divmod(vi, width))
        nearest_pixels = np.sqrt(np.square(val_xy[:, None] - train_xy[None, :]).sum(axis=2)).min(axis=1)
        per_class.append({"class_id": c + 1, "train_pixels": len(ti), "validation_pixels": len(vi),
            "training_regions": len(np.unique(region[ti])),
            "validation_regions": len(np.unique(region[vi])),
            "nearest_training_centroid_class_id": nearest + 1,
            "nearest_centroid_rms_distance": float(distances[c, nearest]),
            "within_training_rms": float(np.sqrt(within[c])),
            "centroid_distance_over_pooled_within_rms": float(distances[c, nearest] / pooled) if pooled else None,
            "validation_to_training_mean_rms": float(np.sqrt(np.square(flat[vi].mean(axis=0) - means[c]).mean())),
            "same_class_val_to_train_pixel_distance_median_max":
                [float(np.median(nearest_pixels)), float(nearest_pixels.max())]})
    observed = {}
    for index, label in zip(train_i, train_y):
        observed.setdefault(int(region[index]), set()).add(int(label))
    return {"support_seed": 64, "k": 10, "validation_k": 5,
            "training_regions": len(observed),
            "training_region_fraction": len(observed) / (int(region.max()) + 1),
            "observed_multi_class_training_regions": sum(len(v) > 1 for v in observed.values()),
            "per_class": per_class,
            "scope": "Released support/validation only, raw spectral distances, no classifier fitting/scoring"}


def frozen_diagnostics(analysis, review):
    lookup = validate_rows(analysis["models"])
    cases = {r["name"]: r for r in review["cases"]}
    if len(cases) != 54 or set(cases) != {r["name"] for r in lookup.values()}:
        raise ValueError("Exact54 query/review name pairing required")
    groups, per_model, contrasts = [], [], []
    for target in NAMES:
        for mode in MODES:
            for method in (SCRATCH, FT):
                models = [lookup[target, seed, method, mode] for seed in SEEDS]
                paired = [cases[r["name"]] for r in models]
                query_support = np.array([c["support"] for c in models[0]["metrics"]["fused"]["per_class"]])
                group = {"target": target, "mode": mode, "method": method,
                    "selected_steps": [r["selected_step"] for r in paired],
                    "budget_end_count": sum(r["selected_at_budget_end"] for r in paired),
                    "three_heads": {}}
                for head in HEADS:
                    group["three_heads"][head] = {
                        "train_oa_pct": distribution(100 * r["heads"]["train"][head]["oa"] for r in paired),
                        "validation_oa_pct": distribution(100 * r["heads"]["val"][head]["oa"] for r in paired),
                        "query_oa_pct": distribution(100 * float(_value(r, head, "oa")) for r in models),
                        "query_aa_pct": distribution(100 * float(_value(r, head, "aa")) for r in models),
                        "validation_query_prior_weighted_oa_pct": distribution(100 * float(np.dot(
                            r["heads"]["val"][head]["recall"], query_support) / query_support.sum()) for r in paired)}
                branches = [r["branches"] for r in models]
                group["branches"] = {"fused_minus_cnn_pp": distribution(100 * (r["fused_accuracy"] - r["cnn_accuracy"]) for r in branches),
                    "graph_minus_cnn_pp": distribution(100 * (r["graph_accuracy"] - r["cnn_accuracy"]) for r in branches),
                    "either_branch_correct_minus_fused_pp": distribution(100 * (r["branch_oracle_accuracy"] - r["fused_accuracy"]) for r in branches),
                    "graph_only_correct_used_pct": distribution(100 * r["strata"]["graph_only_correct"]["fused_correct"] / r["strata"]["graph_only_correct"]["count"] for r in branches),
                    "cnn_only_correct_used_pct": distribution(100 * r["strata"]["cnn_only_correct"]["fused_correct"] / r["strata"]["cnn_only_correct"]["count"] for r in branches)}
                groups.append(group)
                for model, case in zip(models, paired):
                    per_model.append({"name": model["name"], "selected_step": case["selected_step"],
                        "late_nll_delta": case["final_minus_previous_five_events_validation_nll"],
                        "train_fused_oa": case["heads"]["train"]["fused"]["oa"],
                        "val_fused_oa": case["heads"]["val"]["fused"]["oa"],
                        "query_fused_oa": float(_value(model, "fused", "oa")),
                        "query_fused_aa": float(_value(model, "fused", "aa"))})
            scratch = [lookup[target, s, SCRATCH, mode] for s in SEEDS]
            ft = [lookup[target, s, FT, mode] for s in SEEDS]
            contrasts.append({"target": target, "mode": mode,
                "ft_minus_scratch_three_heads_pp": {h: distribution(100 * float(_value(a, h, "oa") - _value(b, h, "oa"))
                    for a, b in zip(ft, scratch)) for h in HEADS}})
    failures = []
    for target in NAMES:
        for class_i in range(DATASETS[target][2]):
            seed_rows = []
            for seed in SEEDS:
                pair = [lookup[target, seed, FT, mode]["metrics"]["fused"] for mode in ("capacity_control", "x_residual")]
                support = pair[0]["per_class"][class_i]["support"]
                seed_rows.append({"seed": seed, "query_class_pixels": support,
                    "capacity_recall_pct": pair[0]["per_class"][class_i]["recall"] * 100,
                    "residual_recall_pct": pair[1]["per_class"][class_i]["recall"] * 100,
                    "delta_pp": 100 * (pair[1]["per_class"][class_i]["recall"] - pair[0]["per_class"][class_i]["recall"]),
                    "capacity_prediction_counts": pair[0]["confusion_matrix"][class_i],
                    "residual_prediction_counts": pair[1]["confusion_matrix"][class_i]})
            failures.append({"target": target, "class_id": class_i + 1, "seeds": seed_rows,
                             "mean_recall_delta_pp": statistics.mean(r["delta_pp"] for r in seed_rows)})
    return {"primary": analysis["primary"], "groups": groups, "ft_minus_scratch": contrasts,
            "primary_per_class": failures, "models": per_model,
            "limitations": ["One support seed, three model seeds, exploratory post hoc interpretation",
                "Either-branch-correct is diagnostic, fused can also correct both-wrong pixels",
                "Prior-weighted V5 recalls reuse five samples per class and remain noisy"]}


def plot_results(out, scenes, frozen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, target in zip(axes, NAMES):
        rows = [r for r in frozen["ft_minus_scratch"] if r["target"] == target]
        for i, row in enumerate(rows):
            values = row["ft_minus_scratch_three_heads_pp"]["fused"]["values"]
            ax.scatter(np.full(3, i), values, color=["#336699", "#cc8833", "#228866"][i])
            ax.plot([i - .18, i + .18], [statistics.mean(values)] * 2, color="black")
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set(xticks=range(3), xticklabels=["none", "capacity", "residual"], title=target,
               ylabel="Full fine-tune minus scratch (pp)")
    fig.savefig(out / "adaptation_deltas.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for scene in scenes:
        values = scene["spectral"]["covariance"]["cumulative_variance"]
        axes[0].plot(np.arange(1, len(values) + 1), values, label=scene["target"])
        counts = scene["query_class_counts"]
        axes[1].plot(range(1, len(counts) + 1), counts, marker=".", label=scene["target"])
    axes[0].set(xlabel="Principal components", ylabel="Unlabeled spectral variance", xlim=(1, 35), ylim=(.7, 1.005))
    axes[1].set(xlabel="Within-dataset class ID", ylabel="Frozen query pixels", yscale="log")
    for ax in axes:
        ax.legend()
    fig.savefig(out / "dataset_characteristics.png", dpi=180)
    plt.close(fig)


def run(runs, out, expected_commit):
    started = time.monotonic()
    code = code_snapshot()
    if code["commit"] != expected_commit or code["dirty"]:
        raise ValueError("Expected clean full tested commit required")
    out = external_directory(out)
    out.mkdir(parents=True, exist_ok=False)
    pins = {str(runs / path): sha for path, sha in INPUTS.values()}

    def verify():
        require_frozen_code(code)
        for path, sha in pins.items():
            if file_hash(path) != sha:
                raise ValueError("Frozen diagnostic input changed: " + path)

    verify()
    records = {key: json.loads((runs / path).read_text()) for key, (path, _) in INPUTS.items()}
    frozen = frozen_diagnostics(records["analysis"], records["review"])
    prepared, contexts = records["prepared"], records["contexts"]
    scenes = []
    for target in NAMES:
        metadata = prepared["datasets"][target]
        raw = metadata["raw"]
        image_path = Path(prepared["data_root"]) / raw["image_path"]
        observation_files = contexts["observations"][target]["input"]
        row = next(r for r in contexts["contexts"] if r["target"] == target
                   and r["support_seed"] == 64 and r["support_per_class"] == 10)
        overlay_path = runs / INPUTS["contexts"][0]
        overlay_path = overlay_path.parent / row["file"]
        pins.update({str(image_path): raw["image_sha256"], str(overlay_path): row["sha256"],
            observation_files["observation"]: observation_files["observation_sha256"],
            observation_files["record"]: observation_files["record_sha256"]})
        verify()
        # Only the registered image variable is decoded. No GT path is opened.
        cube = loadmat(image_path, variable_names=[DATASETS[target][1]])[DATASETS[target][1]]
        if list(cube.shape) != raw["shape"] or not np.isfinite(cube).all():
            raise ValueError("Finite registered image shape required")
        observation = torch.load(observation_files["observation"], map_location="cpu", weights_only=True)
        observation_record = json.loads(Path(observation_files["record"]).read_text())
        if (set(observation) != OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
                or observation_record["graph_fit_labels"] != 0
                or observation_record["target"] != target
                or _item_fingerprint(observation) != observation_record["prepared_fields"]):
            raise ValueError("Label-free observation fingerprint differs")
        overlay = torch.load(overlay_path, map_location="cpu", weights_only=True)
        validate_overlay(overlay, target, 10, row["shape"])
        if set(overlay) != LABEL_KEYS or overlay_identity(overlay) != row["overlay_identity"]:
            raise ValueError("Released K10/V5 overlay identity differs")
        region, graph = graph_statistics(observation)
        flat = cube.reshape(-1, cube.shape[-1])
        rng = np.random.default_rng(64)
        chosen = rng.choice(len(flat), min(20000, len(flat)), replace=False)
        minimum, maximum = metadata["normalization"]["raw_min"], metadata["normalization"]["raw_max"]
        sample = (flat[chosen].astype(np.float64) - minimum) / (maximum - minimum)
        support_flat = flat  # Scalar affine normalization preserves centroid distance ratios.
        counts = [int(raw["label_counts"][str(c)]) - 15 for c in range(1, raw["class_count"] + 1)]
        actual_counts = next(r for r in records["analysis"]["models"] if r["target"] == target)["metrics"]["fused"]["per_class"]
        if counts != [r["support"] for r in actual_counts]:
            raise ValueError("Frozen query cohort differs from accepted metadata minus K10/V5")
        scene = {"target": target, "shape": raw["shape"], "sensor_family": raw["sensor_family"],
            "wavelength_quality": raw["wavelength_quality"], "fwhm_available": raw["fwhm_available"],
            "supplied_wavelength_endpoints_nm": observation["wavelengths"][[0, -1]].tolist(),
            "normalization": metadata["normalization"], "query_class_counts": counts,
            "query_majority_fraction": max(counts) / sum(counts), "query_max_min_class_ratio": max(counts) / min(counts),
            "spectral": spectral_statistics(sample), "graph": graph,
            "support": support_statistics(support_flat, overlay, region, cube.shape[1])}
        scenes.append(scene)
        print("SCENE_COMPLETE", target, flush=True)
        del cube, observation, flat, support_flat, sample, region
    verify()
    result = {"policy": "issue184_posthoc_readonly_diagnosis_v1", "scenes": scenes, "frozen": frozen,
        "sampling": {"pixels_per_scene": 20000, "seed": 64, "replacement": False},
        "limitations": ["Raw spectral rank does not establish discriminative feature rank or safe pruning",
            "Only released K10/V5 labels, no global region purity or boundary measurements",
            "No causal isolation of normalization, source pretraining, optimization, or fusion",
            "Candidate improvements need independently registered comparisons"]}
    write_json(out / "diagnosis.json", result)
    plot_results(out, scenes, frozen)
    verify()
    receipt = {"status": "passed", "code": code, "input_files_verified_unchanged": len(pins),
        "protected_files": pins, "analysis_sha256": file_hash(out / "diagnosis.json"),
        "exported_files": {p.name: file_hash(p) for p in out.iterdir()},
        "models_reviewed": 54, "query_heads_reviewed": 162, "raw_gt_files_decoded": 0,
        "new_query_releases": 0, "new_model_forwards": 0, "new_optimizer_updates": 0,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write_json(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    receipt = run(args.runs.resolve(strict=True), args.out, args.expected_commit)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"code", "protected_files"}}))


if __name__ == "__main__":
    main()

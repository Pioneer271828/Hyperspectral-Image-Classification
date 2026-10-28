from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import (  # noqa: E402
    DISPLAY_DATASETS,
    HONGHU_CLASS_NAMES,
    RESULTS_DIR,
    ROOT,
    SEEDS,
    ensure_dirs,
    is_exact_honghu_group,
    load_metric_jsons,
    mean_std,
    metric_datasets,
    metric_test_rows,
    normalize_dataset_name,
    write_csv,
)


PYTHON_EXE = "python"
RUN_ROOT = ROOT / "experiments" / "honghu_replacement_runs"


def cfg(metric: dict[str, Any], key: str, default: Any = None) -> Any:
    return (metric.get("config") or {}).get(key, default)


def float_cfg(metric: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = cfg(metric, key, default)
    if value in ("", None):
        return default
    return float(value)


def int_cfg(metric: dict[str, Any], key: str, default: int = 0) -> int:
    value = cfg(metric, key, default)
    if value in ("", None):
        return default
    return int(value)


def str_cfg(metric: dict[str, Any], key: str, default: str = "") -> str:
    value = cfg(metric, key, default)
    return str(value)


def same_config(metric: dict[str, Any], expected: dict[str, Any]) -> bool:
    if "latent_channels" in expected and int_cfg(metric, "latent_channels") != int(expected["latent_channels"]):
        return False
    if "adapter_embed_dim" in expected and int_cfg(metric, "adapter_embed_dim") != int(expected["adapter_embed_dim"]):
        return False
    if "adapter_attention_mode" in expected and str_cfg(metric, "adapter_attention_mode") != expected["adapter_attention_mode"]:
        return False
    if "norm_type" in expected and str_cfg(metric, "norm_type") != expected["norm_type"]:
        return False
    if "class_balanced_loss" in expected and bool(cfg(metric, "class_balanced_loss", False)) != bool(expected["class_balanced_loss"]):
        return False
    if "train_drop_band_ratio" in expected and abs(float_cfg(metric, "train_drop_band_ratio") - float(expected["train_drop_band_ratio"])) > 1e-12:
        return False
    if "train_band_mixup_ratio" in expected and abs(float_cfg(metric, "train_band_mixup_ratio") - float(expected["train_band_mixup_ratio"])) > 1e-12:
        return False
    if "primary_checkpoint" in expected and str_cfg(metric, "primary_checkpoint") != expected["primary_checkpoint"]:
        return False
    if "eval_checkpoint" in expected and str_cfg(metric, "eval_checkpoint") != expected["eval_checkpoint"]:
        return False
    return True


def metric_lookup() -> list[tuple[Path, dict[str, Any]]]:
    return load_metric_jsons()


def find_multi(metrics: list[tuple[Path, dict[str, Any]]], seed: int, expected: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    for path, metric in metrics:
        if int(metric.get("seed", -1)) != int(seed):
            continue
        if not is_exact_honghu_group(metric):
            continue
        if same_config(metric, expected):
            return path, metric
    return None


def find_single(metrics: list[tuple[Path, dict[str, Any]]], dataset: str, seed: int, expected: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    target = normalize_dataset_name(dataset)
    for path, metric in metrics:
        if int(metric.get("seed", -1)) != int(seed):
            continue
        if metric_datasets(metric) != [target]:
            continue
        if same_config(metric, expected):
            return path, metric
    return None


MAIN_K128 = {
    "name": "K=128 wavelength attention + DomainBN main model",
    "kind": "multi",
    "latent_channels": 128,
    "adapter_embed_dim": 64,
    "adapter_attention_mode": "wavelength",
    "norm_type": "domain_bn",
    "class_balanced_loss": True,
    "train_drop_band_ratio": 0.0,
    "train_band_mixup_ratio": 0.0,
    "primary_checkpoint": "macro_oa",
    "eval_checkpoint": "primary",
}


EXPECTED_ABLATIONS = [
    MAIN_K128,
    {
        "name": "K=32 wavelength attention lightweight configuration",
        "kind": "multi",
        "latent_channels": 32,
        "adapter_embed_dim": 16,
        "adapter_attention_mode": "wavelength",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "K=64 index attention",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "index",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "K=64 uniform attention",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "uniform",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "K=64 wavelength attention + DomainBN",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "wavelength",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "K=64 wavelength attention + standard BN",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "wavelength",
        "norm_type": "batch_norm",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "BandDrop 0.1",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "wavelength",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.1,
        "train_band_mixup_ratio": 0.0,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
    {
        "name": "Band-level Mixup 0.1",
        "kind": "multi",
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "wavelength",
        "norm_type": "domain_bn",
        "class_balanced_loss": True,
        "train_drop_band_ratio": 0.0,
        "train_band_mixup_ratio": 0.1,
        "primary_checkpoint": "macro_oa",
        "eval_checkpoint": "primary",
    },
]


SINGLE_DATASET_EXPECTED = {
    "name": "single-dataset VCA-SAHGCN K=128",
    "kind": "single",
    "latent_channels": 128,
    "adapter_embed_dim": 64,
    "adapter_attention_mode": "wavelength",
    "norm_type": "domain_bn",
    "class_balanced_loss": True,
    "train_drop_band_ratio": 0.0,
    "train_band_mixup_ratio": 0.0,
    "primary_checkpoint": "macro_oa",
    "eval_checkpoint": "primary",
}


def command_for_multi(expected: dict[str, Any], seed: int) -> str:
    tag = safe_tag(expected["name"])
    save_path = RUN_ROOT / tag / "checkpoints" / f"{tag}_seed{seed}.pt"
    metrics_path = RUN_ROOT / tag / "metrics" / f"{tag}_seed{seed}.json"
    parts = [
        PYTHON_EXE,
        "train_vca_multidataset.py",
        "--device cuda",
        "--datasets paviaU,xuzhou,whu-honghu",
        "--epochs 500",
        "--steps-per-epoch 3",
        "--val-interval 10",
        f"--seed {seed}",
        "--samples-type ratio",
        "--train-ratio 0.01",
        "--val-ratio 0.005",
        "--train-num 30",
        "--val-num 10",
        f"--latent-channels {expected['latent_channels']}",
        f"--adapter-embed-dim {expected['adapter_embed_dim']}",
        f"--adapter-attention-mode {expected['adapter_attention_mode']}",
        f"--train-drop-band-ratio {expected['train_drop_band_ratio']}",
        f"--train-band-mixup-ratio {expected.get('train_band_mixup_ratio', 0.0)}",
        f"--norm-type {expected['norm_type']}",
        "--primary-checkpoint macro_oa",
        "--eval-checkpoint primary",
        f"--save-path {save_path}",
        f"--metrics-json {metrics_path}",
        f"--experiment-name {tag}",
    ]
    return " ".join(str(part) for part in parts)


def command_for_single(dataset: str, seed: int) -> str:
    dataset_arg = {"PaviaU": "paviaU", "Xuzhou": "xuzhou", "HongHu": "whu-honghu"}[dataset]
    tag = f"single_dataset_vca_k128_{dataset.lower()}"
    save_path = RUN_ROOT / tag / "checkpoints" / f"{tag}_seed{seed}.pt"
    metrics_path = RUN_ROOT / tag / "metrics" / f"{tag}_seed{seed}.json"
    parts = [
        PYTHON_EXE,
        "train_vca_multidataset.py",
        "--device cuda",
        f"--datasets {dataset_arg}",
        "--epochs 500",
        "--steps-per-epoch 3",
        "--val-interval 10",
        f"--seed {seed}",
        "--samples-type ratio",
        "--train-ratio 0.01",
        "--val-ratio 0.005",
        "--train-num 30",
        "--val-num 10",
        "--latent-channels 128",
        "--adapter-embed-dim 64",
        "--adapter-attention-mode wavelength",
        "--train-drop-band-ratio 0.0",
        "--train-band-mixup-ratio 0.0",
        "--norm-type domain_bn",
        "--primary-checkpoint macro_oa",
        "--eval-checkpoint primary",
        f"--save-path {save_path}",
        f"--metrics-json {metrics_path}",
        f"--experiment-name {tag}",
    ]
    return " ".join(str(part) for part in parts)


def safe_tag(text: str) -> str:
    tag = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")


def collect_seed_rows(metrics: list[tuple[Path, dict[str, Any]]], expected: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    source_rows = []
    for seed in SEEDS:
        found = find_multi(metrics, seed, expected)
        row = {
            "seed": seed,
            "PaviaU_OA": "",
            "PaviaU_AA": "",
            "PaviaU_Kappa": "",
            "Xuzhou_OA": "",
            "Xuzhou_AA": "",
            "Xuzhou_Kappa": "",
            "HongHu_OA": "",
            "HongHu_AA": "",
            "HongHu_Kappa": "",
            "Macro_OA": "",
        }
        if found:
            path, metric = found
            oas = []
            for dataset_key, prefix in [("PaviaU", "PaviaU"), ("Xuzhou", "Xuzhou"), ("HongHu", "HongHu")]:
                test = None
                for original_name, values in (metric.get("test") or {}).items():
                    if normalize_dataset_name(original_name) == dataset_key:
                        test = values
                        break
                if test:
                    row[f"{prefix}_OA"] = test.get("oa", "")
                    row[f"{prefix}_AA"] = test.get("aa", "")
                    row[f"{prefix}_Kappa"] = test.get("kappa", "")
                    oas.append(test.get("oa"))
            if len(oas) == 3:
                row["Macro_OA"] = sum(float(v) for v in oas) / 3.0
            source_rows.append({"Experiment_Name": expected["name"], "Seed": seed, "Source_File": str(path.relative_to(ROOT))})
        rows.append(row)
    return rows, source_rows


def summarize_main(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for dataset, prefix in [("PaviaU", "PaviaU"), ("Xuzhou", "Xuzhou"), ("WHU-Hi-HongHu", "HongHu")]:
        oas = [row.get(f"{prefix}_OA") for row in seed_rows if row.get(f"{prefix}_OA") not in ("", None)]
        aas = [row.get(f"{prefix}_AA") for row in seed_rows if row.get(f"{prefix}_AA") not in ("", None)]
        kappas = [row.get(f"{prefix}_Kappa") for row in seed_rows if row.get(f"{prefix}_Kappa") not in ("", None)]
        oa_mean, oa_std = mean_std(oas)
        aa_mean, aa_std = mean_std(aas)
        k_mean, k_std = mean_std(kappas)
        out.append(
            {
                "Dataset": dataset,
                "n": len(oas),
                "OA_mean": "" if oa_mean is None else oa_mean,
                "OA_std": "" if oa_std is None else oa_std,
                "AA_mean": "" if aa_mean is None else aa_mean,
                "AA_std": "" if aa_std is None else aa_std,
                "Kappa_mean": "" if k_mean is None else k_mean,
                "Kappa_std": "" if k_std is None else k_std,
            }
        )
    macros = [row.get("Macro_OA") for row in seed_rows if row.get("Macro_OA") not in ("", None)]
    macro_mean, macro_std = mean_std(macros)
    out.append(
        {
            "Dataset": "Macro",
            "n": len(macros),
            "OA_mean": "" if macro_mean is None else macro_mean,
            "OA_std": "" if macro_std is None else macro_std,
            "AA_mean": "",
            "AA_std": "",
            "Kappa_mean": "",
            "Kappa_std": "",
        }
    )
    return out


def append_auxiliary_honghu_k32(ablation_seed_rows: list[dict[str, Any]], ablation_summary_rows: list[dict[str, Any]], metrics: list[tuple[Path, dict[str, Any]]]) -> None:
    aux_values = []
    for seed in SEEDS:
        found = find_single(
            metrics,
            "HongHu",
            seed,
            {
                "latent_channels": 32,
                "adapter_embed_dim": 16,
                "adapter_attention_mode": "wavelength",
                "norm_type": "domain_bn",
                "class_balanced_loss": True,
                "train_drop_band_ratio": 0.0,
                "train_band_mixup_ratio": 0.0,
                "primary_checkpoint": "macro_oa",
                "eval_checkpoint": "primary",
            },
        )
        row = {
            "Experiment_Name": "auxiliary: HongHu K=32 single-dataset VCA-SAHGCN",
            "Seed": seed,
            "PaviaU_OA": "",
            "Xuzhou_OA": "",
            "HongHu_OA": "",
            "Macro_OA": "",
        }
        if found:
            _, metric = found
            values = next(iter((metric.get("test") or {}).values()))
            row["HongHu_OA"] = values.get("oa", "")
            aux_values.append(values)
        ablation_seed_rows.append(row)
    oa_mean, oa_std = mean_std([v.get("oa") for v in aux_values])
    aa_mean, aa_std = mean_std([v.get("aa") for v in aux_values])
    k_mean, k_std = mean_std([v.get("kappa") for v in aux_values])
    ablation_summary_rows.append(
        {
            "Experiment_Name": "auxiliary: HongHu K=32 single-dataset VCA-SAHGCN",
            "Dataset": "WHU-Hi-HongHu",
            "n": len(aux_values),
            "OA_mean": "" if oa_mean is None else oa_mean,
            "OA_std": "" if oa_std is None else oa_std,
            "AA_mean": "" if aa_mean is None else aa_mean,
            "AA_std": "" if aa_std is None else aa_std,
            "Kappa_mean": "" if k_mean is None else k_mean,
            "Kappa_std": "" if k_std is None else k_std,
            "Notes": "Available local run; not a substitute for the requested K=128 three-dataset main experiment.",
        }
    )


def collect_ablation(metrics: list[tuple[Path, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    seed_rows = []
    summary_rows = []
    missing = []
    for expected in EXPECTED_ABLATIONS:
        rows, _ = collect_seed_rows(metrics, expected)
        for row in rows:
            seed_rows.append(
                {
                    "Experiment_Name": expected["name"],
                    "Seed": row["seed"],
                    "PaviaU_OA": row["PaviaU_OA"],
                    "Xuzhou_OA": row["Xuzhou_OA"],
                    "HongHu_OA": row["HongHu_OA"],
                    "Macro_OA": row["Macro_OA"],
                }
            )
        if not any(row["Macro_OA"] not in ("", None) for row in rows):
            missing.append(expected["name"])
        for dataset, prefix in [("PaviaU", "PaviaU"), ("Xuzhou", "Xuzhou"), ("WHU-Hi-HongHu", "HongHu")]:
            oas = [row.get(f"{prefix}_OA") for row in rows if row.get(f"{prefix}_OA") not in ("", None)]
            aas = [row.get(f"{prefix}_AA") for row in rows if row.get(f"{prefix}_AA") not in ("", None)]
            kappas = [row.get(f"{prefix}_Kappa") for row in rows if row.get(f"{prefix}_Kappa") not in ("", None)]
            oa_mean, oa_std = mean_std(oas)
            aa_mean, aa_std = mean_std(aas)
            k_mean, k_std = mean_std(kappas)
            summary_rows.append(
                {
                    "Experiment_Name": expected["name"],
                    "Dataset": dataset,
                    "n": len(oas),
                    "OA_mean": "" if oa_mean is None else oa_mean,
                    "OA_std": "" if oa_std is None else oa_std,
                    "AA_mean": "" if aa_mean is None else aa_mean,
                    "AA_std": "" if aa_std is None else aa_std,
                    "Kappa_mean": "" if k_mean is None else k_mean,
                    "Kappa_std": "" if k_std is None else k_std,
                    "Notes": "missing exact PaviaU/Xuzhou/HongHu run" if not oas else "exact HongHu replacement run",
                }
            )
    for dataset in ["PaviaU", "Xuzhou", "HongHu"]:
        expected = SINGLE_DATASET_EXPECTED
        vals = []
        for seed in SEEDS:
            found = find_single(metrics, dataset, seed, expected)
            row = {
                "Experiment_Name": f"{expected['name']}: {DISPLAY_DATASETS.get(dataset, dataset)}",
                "Seed": seed,
                "PaviaU_OA": "",
                "Xuzhou_OA": "",
                "HongHu_OA": "",
                "Macro_OA": "",
            }
            if found:
                _, metric = found
                values = next(iter((metric.get("test") or {}).values()))
                row[f"{dataset}_OA" if dataset != "HongHu" else "HongHu_OA"] = values.get("oa", "")
                vals.append(values)
            seed_rows.append(row)
        if not vals:
            missing.append(f"{expected['name']}: {DISPLAY_DATASETS.get(dataset, dataset)}")
        oa_mean, oa_std = mean_std([v.get("oa") for v in vals])
        aa_mean, aa_std = mean_std([v.get("aa") for v in vals])
        k_mean, k_std = mean_std([v.get("kappa") for v in vals])
        summary_rows.append(
            {
                "Experiment_Name": f"{expected['name']}: {DISPLAY_DATASETS.get(dataset, dataset)}",
                "Dataset": DISPLAY_DATASETS.get(dataset, dataset),
                "n": len(vals),
                "OA_mean": "" if oa_mean is None else oa_mean,
                "OA_std": "" if oa_std is None else oa_std,
                "AA_mean": "" if aa_mean is None else aa_mean,
                "AA_std": "" if aa_std is None else aa_std,
                "Kappa_mean": "" if k_mean is None else k_mean,
                "Kappa_std": "" if k_std is None else k_std,
                "Notes": "missing exact single-dataset K=128 run" if not vals else "exact run",
            }
        )
    append_auxiliary_honghu_k32(seed_rows, summary_rows, metrics)
    return seed_rows, summary_rows, missing


def classwise_rows() -> list[dict[str, Any]]:
    rows = []
    for class_id, name in HONGHU_CLASS_NAMES.items():
        rows.append(
            {
                "Class_ID": class_id,
                "Class_Name": name,
                "Seed_0_Accuracy": "",
                "Seed_2026_Accuracy": "",
                "Seed_3407_Accuracy": "",
                "Mean_Accuracy": "",
                "Std_Accuracy": "",
            }
        )
    for metric, label in [("OA", "Overall Accuracy"), ("AA", "Average Accuracy"), ("Kappa", "Kappa")]:
        rows.append(
            {
                "Class_ID": metric,
                "Class_Name": label,
                "Seed_0_Accuracy": "",
                "Seed_2026_Accuracy": "",
                "Seed_3407_Accuracy": "",
                "Mean_Accuracy": "",
                "Std_Accuracy": "",
            }
        )
    return rows


def write_missing_report(missing: list[str]) -> None:
    lines = [
        "# Missing HongHu Replacement Experiments",
        "",
        "The project currently has no exact K=128 or requested ablation runs trained/evaluated on the replacement dataset group `paviaU,xuzhou,whu-honghu`.",
        "Existing `paviaU,xuzhou,WHUL` results are not reused because they were trained with LongKou and would invalidate the new Macro-OA.",
        "",
        "## Required Multi-Dataset Runs",
        "",
    ]
    for expected in EXPECTED_ABLATIONS:
        lines.append(f"### {expected['name']}")
        if expected["name"] in missing:
            lines.append("Status: missing")
        else:
            lines.append("Status: found")
        lines.append("")
        for seed in SEEDS:
            lines.append("```powershell")
            lines.append(command_for_multi(expected, seed))
            lines.append("```")
        lines.append("")
    lines.extend(["## Required Single-Dataset Runs", ""])
    for dataset in ["PaviaU", "Xuzhou", "HongHu"]:
        lines.append(f"### single-dataset VCA-SAHGCN K=128: {DISPLAY_DATASETS.get(dataset, dataset)}")
        lines.append("Status: missing" if f"single-dataset VCA-SAHGCN K=128: {DISPLAY_DATASETS.get(dataset, dataset)}" in missing else "Status: found or partially found")
        lines.append("")
        for seed in SEEDS:
            lines.append("```powershell")
            lines.append(command_for_single(dataset, seed))
            lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Expected Outputs",
            "",
            "- One metrics JSON per seed under `experiments/honghu_replacement_runs/**/metrics/`.",
            "- Checkpoints under `experiments/honghu_replacement_runs/**/checkpoints/`.",
            "- After runs finish, rerun `python scripts/honghu_replacement/collect_honghu_results.py` to recompute all CSVs and Macro-OA values.",
            "",
            "## Available Auxiliary Result",
            "",
            "- `experiments/honghu_vca_k32/metrics/vca_honghu_k32_seed{0,2026,3407}.json` is a real HongHu K=32 single-dataset VCA-SAHGCN run. It is useful for qualitative visualization, but it is not the requested K=128 three-dataset main experiment.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "missing_experiments_honghu.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "need_to_run_experiments.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    metrics = metric_lookup()
    main_rows, _ = collect_seed_rows(metrics, MAIN_K128)
    write_csv(
        RESULTS_DIR / "main_k128_seed_results_honghu.csv",
        main_rows,
        [
            "seed",
            "PaviaU_OA",
            "PaviaU_AA",
            "PaviaU_Kappa",
            "Xuzhou_OA",
            "Xuzhou_AA",
            "Xuzhou_Kappa",
            "HongHu_OA",
            "HongHu_AA",
            "HongHu_Kappa",
            "Macro_OA",
        ],
    )
    write_csv(
        RESULTS_DIR / "main_k128_summary_honghu.csv",
        summarize_main(main_rows),
        ["Dataset", "n", "OA_mean", "OA_std", "AA_mean", "AA_std", "Kappa_mean", "Kappa_std"],
    )
    write_csv(
        RESULTS_DIR / "classwise_accuracy_honghu.csv",
        classwise_rows(),
        ["Class_ID", "Class_Name", "Seed_0_Accuracy", "Seed_2026_Accuracy", "Seed_3407_Accuracy", "Mean_Accuracy", "Std_Accuracy"],
    )
    ablation_seed, ablation_summary, missing = collect_ablation(metrics)
    write_csv(
        RESULTS_DIR / "ablation_seed_results_honghu.csv",
        ablation_seed,
        ["Experiment_Name", "Seed", "PaviaU_OA", "Xuzhou_OA", "HongHu_OA", "Macro_OA"],
    )
    write_csv(
        RESULTS_DIR / "ablation_summary_honghu.csv",
        ablation_summary,
        ["Experiment_Name", "Dataset", "n", "OA_mean", "OA_std", "AA_mean", "AA_std", "Kappa_mean", "Kappa_std", "Notes"],
    )
    write_missing_report(missing)
    print(f"Wrote result CSVs and missing report to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

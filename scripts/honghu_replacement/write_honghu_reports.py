from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import FIGURES_DIR, RESULTS_DIR, ensure_dirs, read_json  # noqa: E402


CHECKLIST_ITEMS = [
    ("Abstract", "replace; recalculate", "Replace WHU-Hi LongKou/WHUL claims with WHU-Hi-HongHu; update performance claims after missing runs finish."),
    ("Keywords near dataset description", "replace", "Use WHU-Hi-HongHu and short name HongHu."),
    ("Introduction", "replace", "Remove LongKou/WHUL wording; revise cross-scene motivation if it refers to LongKou agriculture classes."),
    ("Contributions", "replace; recalculate", "Update dataset count wording and any Macro-OA improvement statements."),
    ("Dataset description", "replace", "Use local/official HongHu metadata: 940 x 475 x 270, 400-1000 nm, 0.043 m, 22 classes."),
    ("Table 1 dataset spectral configurations", "replace; recalculate", "Use dataset_honghu_metadata.json/csv and class_distribution_honghu.csv."),
    ("Fig. 1 dataset overview", "redraw", "Use fig01_dataset_overview_honghu.*."),
    ("Table 2 experimental settings", "replace", "Use dataset short name HongHu; keep train_ratio=0.01 and val_ratio=0.005."),
    ("Table 3 main K=128 results", "missing data; recalculate", "Exact paviaU/xuzhou/whu-honghu K=128 runs not found."),
    ("Table 4 seed results", "missing data; recalculate", "Seed 0/2026/3407 replacement results are missing."),
    ("Section 4.3 main experimental analysis", "missing data; recalculate", "Do not reuse old WHUL Macro-OA."),
    ("Section 4.3.1 visualization and cross-method comparison", "redraw; missing data", "Ground truth is available; competing HongHu prediction maps are missing."),
    ("Table 4d HongHu comparison", "missing data", "No verified cross-method HongHu numeric table was found locally."),
    ("Fig. 3 HongHu classification maps", "redraw; missing data", "Use fig_honghu_classification_maps.*; only auxiliary VCA K=32 map is available."),
    ("Attention heatmap figure", "redraw; missing data", "fig_attention_heatmap_honghu.* uses K=32 auxiliary checkpoint, not K=128."),
    ("t-SNE supplementary figure", "redraw; missing data", "fig_tsne_honghu.* uses K=32 auxiliary checkpoint if generated."),
    ("Ablation tables", "missing data; recalculate", "Requested HongHu replacement ablations are missing."),
    ("BandDrop / Mixup tables", "missing data; recalculate", "Old WHUL augmentation runs are invalid for HongHu."),
    ("Limitations", "replace", "Mention missing or pending HongHu reruns if the paper is updated before completion."),
    ("Conclusion", "replace; recalculate", "Do not retain LongKou-specific conclusion or old Macro-OA."),
]


def write_checklist() -> None:
    lines = [
        "# Replace LongKou/WHUL With WHU-Hi-HongHu Checklist",
        "",
        "| Location | Status | New data source / action |",
        "|---|---|---|",
    ]
    for item, status, source in CHECKLIST_ITEMS:
        lines.append(f"| {item} | {status} | {source} |")
    lines.append("")
    lines.append("Naming convention: full name `WHU-Hi-HongHu`, short name `HongHu`. Do not use `WHUL` as a HongHu abbreviation.")
    (RESULTS_DIR / "replace_longkou_with_honghu_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def list_paths(directory: Path, patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))
    return sorted(set(paths))


def summarize_main() -> list[str]:
    path = RESULTS_DIR / "main_k128_summary_honghu.csv"
    if not path.exists():
        return ["Main K=128 summary file is missing."]
    df = pd.read_csv(path)
    lines = []
    for _, row in df.iterrows():
        dataset = row["Dataset"]
        n = int(row["n"]) if str(row["n"]).strip() else 0
        if n == 0:
            lines.append(f"- {dataset}: missing exact replacement result.")
        else:
            lines.append(f"- {dataset}: OA mean {float(row['OA_mean']) * 100:.2f}% over n={n}.")
    return lines


def summarize_ablation() -> list[str]:
    path = RESULTS_DIR / "ablation_summary_honghu.csv"
    if not path.exists():
        return ["Ablation summary file is missing."]
    df = pd.read_csv(path)
    exact = df[pd.to_numeric(df["n"], errors="coerce").fillna(0) > 0]
    lines = []
    if exact.empty:
        lines.append("- No requested exact HongHu replacement ablation result was found.")
    else:
        for _, row in exact.iterrows():
            oa = pd.to_numeric(pd.Series([row["OA_mean"]]), errors="coerce").iloc[0]
            if pd.isna(oa):
                continue
            lines.append(f"- {row['Experiment_Name']} / {row['Dataset']}: OA {oa * 100:.2f}% (n={int(row['n'])}).")
    return lines


def residual_summary(limit: int = 30) -> list[str]:
    rows = csv_rows(RESULTS_DIR / "residual_longkou_terms.csv")
    if not rows:
        return ["- No residual WHUL/LongKou strings were found in scanned files."]
    lines = [f"- Residual scanner found {len(rows)} matches. First {min(limit, len(rows))}:"]
    for row in rows[:limit]:
        loc = row["File"]
        if row.get("Line"):
            loc += f":{row['Line']}"
        if row.get("Container"):
            loc += f" ({row['Container']})"
        lines.append(f"  - {loc}: `{row['Term']}`")
    if len(rows) > limit:
        lines.append(f"  - ... {len(rows) - limit} more rows in residual_longkou_terms.csv")
    return lines


def write_final_report() -> None:
    metadata_path = RESULTS_DIR / "dataset_honghu_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    csv_files = list_paths(
        RESULTS_DIR,
        [
            "*.csv",
        ],
    )
    other_result_files = list_paths(RESULTS_DIR, ["*.json", "*.md", "*.npy"])
    fig_files = list_paths(FIGURES_DIR, ["*.png", "*.pdf", "*.tiff"])
    missing_text = (RESULTS_DIR / "missing_experiments_honghu.md").read_text(encoding="utf-8") if (RESULTS_DIR / "missing_experiments_honghu.md").exists() else ""

    lines = [
        "# Final HongHu Data and Figures Report",
        "",
        "## 1. Confirmed WHU-Hi-HongHu Dataset Information",
        "",
    ]
    if metadata:
        lines.extend(
            [
                f"- Dataset: {metadata.get('dataset_name')} (short name: {metadata.get('short_name')}).",
                f"- Local cube shape: {metadata.get('height')} x {metadata.get('width')} x {metadata.get('bands')}.",
                f"- Spectral range: {metadata.get('spectral_range_nm', ['',''])[0]}-{metadata.get('spectral_range_nm', ['',''])[1]} nm.",
                f"- Spatial resolution: {metadata.get('spatial_resolution')}.",
                f"- Sensor: {metadata.get('sensor')}.",
                f"- Classes: {metadata.get('class_count')} foreground classes plus background label 0.",
                "- True center wavelengths: not found locally; approximate linear 400-1000 nm positions are used by the project loader.",
                "- Important correction: the local/official HongHu ground truth has 22 classes, not 19. The draft 19-class table is incomplete.",
                f"- Official reference checked: {metadata.get('official_source_url')}.",
            ]
        )
    else:
        lines.append("- Metadata file is missing.")

    lines.extend(["", "## 2. Generated CSV Files", ""])
    lines.extend([f"- `{path}`" for path in csv_files] or ["- None found."])

    lines.extend(["", "Other generated result/report files:", ""])
    lines.extend([f"- `{path}`" for path in other_result_files] or ["- None found."])

    lines.extend(["", "## 3. Generated Figure Files", ""])
    lines.extend([f"- `{path}`" for path in fig_files] or ["- None found."])

    lines.extend(["", "## 4. Main Experiment Result Summary", ""])
    lines.extend(summarize_main())
    lines.append("- Macro-OA was not recalculated because the exact three-dataset K=128 HongHu run is missing.")
    lines.append("- `classwise_accuracy_honghu.csv` contains the verified 22 HongHu class names, but K=128 seed-level class-wise accuracies are blank because no valid source was found.")

    lines.extend(["", "## 5. Ablation Result Summary", ""])
    lines.extend(summarize_ablation())

    lines.extend(
        [
            "",
            "## 6. Conclusions That Need Rewriting Versus the Old WHUL/LongKou Version",
            "",
            "- Any statement using WHUL or WHU-Hi LongKou performance must be removed or rewritten.",
            "- Old Macro-OA values are invalid because the third dataset changed.",
            "- Dataset-description text must change from 9-class LongKou to 22-class HongHu.",
            "- Visual claims based on LongKou classification maps, attention, or t-SNE panels are invalid for HongHu.",
            "- The available HongHu K=32 single-dataset checkpoint can support qualitative auxiliary figures only; it cannot support the K=128 main-result claims.",
            "",
            "## 7. Missing Results and Rerun Need",
            "",
        ]
    )
    if missing_text:
        lines.append("See `missing_experiments_honghu.md` and `need_to_run_experiments.md` for runnable commands.")
    else:
        lines.append("Missing-experiment report is not available.")
    lines.append("The generated performance/seed/ablation figures that show `missing` are placeholders documenting the absence of valid data, not synthetic results.")

    lines.extend(["", "## 8. Figure and Table Paths", ""])
    lines.append("- Results directory: `results_honghu_replacement/`")
    lines.append("- Figures directory: `figures_honghu_replacement/`")
    lines.append("- Scripts directory: `scripts/honghu_replacement/`")

    lines.extend(["", "## 9. Residual WHUL / LongKou Strings", ""])
    lines.append("The scan includes diagnostic outputs and reports; rows in `missing_experiments_honghu.md`, `need_to_run_experiments.md`, and this report may intentionally mention old terms when explaining why old results are invalid.")
    lines.extend(residual_summary())
    lines.append("")
    (RESULTS_DIR / "final_honghu_data_and_figures_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_checklist()
    write_final_report()
    print(f"Wrote {RESULTS_DIR / 'replace_longkou_with_honghu_checklist.md'}")
    print(f"Wrote {RESULTS_DIR / 'final_honghu_data_and_figures_report.md'}")


if __name__ == "__main__":
    main()

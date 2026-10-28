from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import (  # noqa: E402
    CLASS_NAMES,
    HONGHU_CLASS_NAMES,
    MAT_DATASETS,
    RESULTS_DIR,
    class_distribution,
    ensure_dirs,
    load_dataset,
    write_csv,
    write_json,
)


OFFICIAL_URL = "https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm"


def build_metadata() -> dict:
    image, gt = load_dataset("HongHu")
    height, width, bands = image.shape
    distribution = class_distribution(gt)
    non_background = [row for row in distribution if row["Class_ID"] != 0]
    classes = []
    for row in non_background:
        class_id = int(row["Class_ID"])
        classes.append(
            {
                "class_id": class_id,
                "class_name": HONGHU_CLASS_NAMES.get(class_id, ""),
                "pixel_count": int(row["Pixel_Count"]),
            }
        )

    spec = MAT_DATASETS["HongHu"]
    return {
        "dataset_name": "WHU-Hi-HongHu",
        "short_name": "HongHu",
        "local_image_file": str(spec["image"]),
        "local_ground_truth_file": str(spec["gt"]),
        "height": int(height),
        "width": int(width),
        "bands": int(bands),
        "spectral_range_nm": spec["spectral_range_nm"],
        "spatial_resolution": spec["spatial_resolution"],
        "sensor": spec["sensor"],
        "official_source_url": OFFICIAL_URL,
        "has_true_center_wavelengths": False,
        "center_wavelengths_note": (
            "No local true center-wavelength file was found. The project loader uses "
            "linear interpolation over 400-1000 nm as approximate spectral-position metadata."
        ),
        "approximate_center_wavelengths_nm": np.linspace(
            spec["spectral_range_nm"][0],
            spec["spectral_range_nm"][1],
            bands,
            dtype=np.float64,
        ).round(6).tolist(),
        "class_count": int(len(non_background)),
        "classes": classes,
        "ground_truth_has_background": bool(any(row["Class_ID"] == 0 for row in distribution)),
        "background_label": 0,
        "background_pixel_count": int(next((row["Pixel_Count"] for row in distribution if row["Class_ID"] == 0), 0)),
        "train_ratio": 0.01,
        "val_ratio": 0.005,
        "split_note": "The requested replacement keeps train_ratio=0.01 and val_ratio=0.005.",
        "local_vs_requested_class_note": (
            "Local ground truth contains 22 foreground classes. The 19-class draft table is incomplete "
            "and omits/reindexes classes including Cotton firewood, Garlic sprout, Broad bean, and Tree."
        ),
    }


def main() -> None:
    ensure_dirs()
    metadata = build_metadata()
    json_path = RESULTS_DIR / "dataset_honghu_metadata.json"
    csv_path = RESULTS_DIR / "dataset_honghu_metadata.csv"
    dist_path = RESULTS_DIR / "class_distribution_honghu.csv"

    write_json(json_path, metadata)

    write_csv(
        csv_path,
        [
            {
                "Field": "dataset_name",
                "Value": metadata["dataset_name"],
                "Source": "local .mat + official WHU-Hi page",
            },
            {"Field": "short_name", "Value": metadata["short_name"], "Source": "project convention"},
            {"Field": "height", "Value": metadata["height"], "Source": "local WHU_Hi_HongHu.mat"},
            {"Field": "width", "Value": metadata["width"], "Source": "local WHU_Hi_HongHu.mat"},
            {"Field": "bands", "Value": metadata["bands"], "Source": "local WHU_Hi_HongHu.mat"},
            {"Field": "spectral_range_nm", "Value": "400-1000", "Source": "official WHU-Hi page + project loader"},
            {"Field": "spatial_resolution", "Value": metadata["spatial_resolution"], "Source": "official WHU-Hi page"},
            {"Field": "sensor", "Value": metadata["sensor"], "Source": "official WHU-Hi page"},
            {
                "Field": "has_true_center_wavelengths",
                "Value": metadata["has_true_center_wavelengths"],
                "Source": "local file inspection",
            },
            {
                "Field": "center_wavelengths_note",
                "Value": metadata["center_wavelengths_note"],
                "Source": "loadData/data_read.py",
            },
            {"Field": "class_count", "Value": metadata["class_count"], "Source": "local WHU_Hi_HongHu_gt.mat"},
            {
                "Field": "ground_truth_has_background",
                "Value": metadata["ground_truth_has_background"],
                "Source": "local WHU_Hi_HongHu_gt.mat",
            },
            {"Field": "background_label", "Value": metadata["background_label"], "Source": "local WHU_Hi_HongHu_gt.mat"},
            {"Field": "train_ratio", "Value": metadata["train_ratio"], "Source": "requested setting"},
            {"Field": "val_ratio", "Value": metadata["val_ratio"], "Source": "requested setting"},
        ],
        ["Field", "Value", "Source"],
    )

    dist_rows = []
    _, gt = load_dataset("HongHu")
    for row in class_distribution(gt):
        class_id = int(row["Class_ID"])
        dist_rows.append(
            {
                "Class_ID": class_id,
                "Class_Name": "Background" if class_id == 0 else CLASS_NAMES["HongHu"].get(class_id, ""),
                "Pixel_Count": int(row["Pixel_Count"]),
            }
        )
    write_csv(dist_path, dist_rows, ["Class_ID", "Class_Name", "Pixel_Count"])
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {dist_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_honghu_replacement"
FIGURES_DIR = ROOT / "figures_honghu_replacement"
SEEDS = [0, 2026, 3407]


DATASET_ALIASES = {
    "paviau": "PaviaU",
    "paviau": "PaviaU",
    "paviau": "PaviaU",
    "paviau": "PaviaU",
    "paviau": "PaviaU",
    "paviau": "PaviaU",
    "paviaU": "PaviaU",
    "xuzhou": "Xuzhou",
    "WHUL": "WHUL",
    "whul": "WHUL",
    "whu-longkou": "WHUL",
    "whu_longkou": "WHUL",
    "whu-honghu": "HongHu",
    "whu_honghu": "HongHu",
    "honghu": "HongHu",
    "WHU-Hi-HongHu": "HongHu",
    "HongHu": "HongHu",
    "PaviaU": "PaviaU",
    "Xuzhou": "Xuzhou",
}


DISPLAY_DATASETS = {
    "PaviaU": "PaviaU",
    "Xuzhou": "Xuzhou",
    "HongHu": "WHU-Hi-HongHu",
}


MAT_DATASETS = {
    "PaviaU": {
        "image": ROOT / "dataset" / "PaviaU" / "PaviaU.mat",
        "image_key": "paviaU",
        "gt": ROOT / "dataset" / "PaviaU" / "PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
        "spectral_range_nm": [430.0, 860.0],
        "spatial_resolution": "1.3 m",
        "sensor": "ROSIS-03",
    },
    "Xuzhou": {
        "image": ROOT / "dataset" / "Xuzhou" / "xuzhou.mat",
        "image_key": "xuzhou",
        "gt": ROOT / "dataset" / "Xuzhou" / "xuzhou_gt.mat",
        "gt_key": "xuzhou_gt",
        "spectral_range_nm": [415.0, 2508.0],
        "spatial_resolution": "0.73 m",
        "sensor": "Airborne HYSPEX",
    },
    "HongHu": {
        "image": ROOT / "dataset" / "WHU-Hi" / "WHU-Hi-HongHu" / "WHU_Hi_HongHu.mat",
        "image_key": "WHU_Hi_HongHu",
        "gt": ROOT / "dataset" / "WHU-Hi" / "WHU-Hi-HongHu" / "WHU_Hi_HongHu_gt.mat",
        "gt_key": "WHU_Hi_HongHu_gt",
        "spectral_range_nm": [400.0, 1000.0],
        "spatial_resolution": "0.043 m",
        "sensor": "Headwall Nano-Hyperspec sensor on a DJI Matrice 600 Pro UAV",
    },
}


PAVIAU_CLASS_NAMES = {
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare Soil",
    7: "Bitumen",
    8: "Self-Blocking Bricks",
    9: "Shadows",
}


XUZHOU_CLASS_NAMES = {
    1: "Crops",
    2: "Water",
    3: "Vegetation",
    4: "Bare soil",
    5: "Coal",
    6: "Cement",
    7: "Road",
    8: "Buildings",
    9: "Other",
}


HONGHU_CLASS_NAMES = {
    1: "Red roof",
    2: "Road",
    3: "Bare soil",
    4: "Cotton",
    5: "Cotton firewood",
    6: "Rape",
    7: "Chinese cabbage",
    8: "Pakchoi",
    9: "Cabbage",
    10: "Tuber mustard",
    11: "Brassica parachinensis",
    12: "Brassica chinensis",
    13: "Small brassica chinensis",
    14: "Lactuca sativa",
    15: "Celtuce",
    16: "Film covered lettuce",
    17: "Romaine lettuce",
    18: "Carrot",
    19: "White radish",
    20: "Garlic sprout",
    21: "Broad bean",
    22: "Tree",
}


CLASS_NAMES = {
    "PaviaU": PAVIAU_CLASS_NAMES,
    "Xuzhou": XUZHOU_CLASS_NAMES,
    "HongHu": HONGHU_CLASS_NAMES,
}


PALETTE = np.array(
    [
        [0, 0, 0],
        [222, 37, 37],
        [44, 124, 191],
        [241, 196, 15],
        [39, 174, 96],
        [155, 89, 182],
        [230, 126, 34],
        [26, 188, 156],
        [127, 140, 141],
        [192, 57, 43],
        [52, 73, 94],
        [142, 68, 173],
        [22, 160, 133],
        [243, 156, 18],
        [41, 128, 185],
        [211, 84, 0],
        [46, 204, 113],
        [149, 165, 166],
        [231, 76, 60],
        [52, 152, 219],
        [241, 90, 36],
        [124, 179, 66],
        [96, 125, 139],
        [233, 30, 99],
        [0, 150, 136],
        [255, 193, 7],
    ],
    dtype=np.uint8,
)


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def normalize_dataset_name(name: str | None) -> str:
    if name is None:
        return ""
    text = str(name).strip()
    return DATASET_ALIASES.get(text, DATASET_ALIASES.get(text.lower(), text))


def load_mat(path: Path, key: str) -> np.ndarray:
    try:
        import scipy.io as sio

        return np.asarray(sio.loadmat(path)[key])
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            arr = np.asarray(handle[key])
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        return arr


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    key = normalize_dataset_name(dataset)
    if key not in MAT_DATASETS:
        raise KeyError(f"Unsupported dataset {dataset!r}")
    spec = MAT_DATASETS[key]
    image = load_mat(spec["image"], spec["image_key"])
    gt = load_mat(spec["gt"], spec["gt_key"])
    return image, gt


def class_distribution(gt: np.ndarray) -> list[dict[str, Any]]:
    values, counts = np.unique(gt, return_counts=True)
    return [
        {"Class_ID": int(v), "Pixel_Count": int(c)}
        for v, c in zip(values, counts)
    ]


def dataset_class_distribution(dataset: str) -> list[dict[str, Any]]:
    key = normalize_dataset_name(dataset)
    _, gt = load_dataset(key)
    names = CLASS_NAMES.get(key, {})
    rows = []
    for row in class_distribution(gt):
        class_id = row["Class_ID"]
        rows.append(
            {
                "Dataset": DISPLAY_DATASETS.get(key, key),
                "Class_ID": class_id,
                "Class_Name": "Background" if class_id == 0 else names.get(class_id, ""),
                "Pixel_Count": row["Pixel_Count"],
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def pct(value: float | None) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value) * 100.0:.2f}"


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int64)
    if label.max(initial=0) >= len(PALETTE):
        rng = np.random.default_rng(20260609)
        extra = rng.integers(20, 235, size=(int(label.max()) + 1 - len(PALETTE), 3), dtype=np.uint8)
        palette = np.vstack([PALETTE, extra])
    else:
        palette = PALETTE
    safe = np.where((label >= 0) & (label < len(palette)), label, 0)
    return palette[safe]


def normalize_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def false_color_rgb(cube: np.ndarray, dataset: str) -> np.ndarray:
    key = normalize_dataset_name(dataset)
    bands = cube.shape[2]
    start, end = MAT_DATASETS[key]["spectral_range_nm"]
    wavelengths = np.linspace(start, end, bands)
    target = np.array([650.0, 550.0, 470.0])
    indices = [int(np.argmin(np.abs(wavelengths - t))) for t in target]
    rgb = np.stack([normalize_01(cube[:, :, idx]) for idx in indices], axis=2)
    return rgb


def save_figure(fig, stem: str, formats: Iterable[str] = ("png", "pdf", "tiff"), dpi: int = 600) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        suffix = "tiff" if fmt.lower() in {"tif", "tiff"} else fmt.lower()
        path = FIGURES_DIR / f"{stem}.{suffix}"
        kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=suffix, **kwargs)
        outputs.append(str(path))
    return outputs


def load_metric_jsons() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((ROOT / "experiments").rglob("*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict) and "test" in data and ("seed" in data or "config" in data):
            rows.append((path, data))
    return rows


def metric_datasets(metric: dict[str, Any]) -> list[str]:
    datasets = metric.get("datasets")
    if datasets is None and "dataset" in metric:
        datasets = [metric["dataset"]]
    return [normalize_dataset_name(name) for name in datasets or []]


def metric_test_rows(path: Path, metric: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, values in (metric.get("test") or {}).items():
        dataset_key = normalize_dataset_name(dataset)
        rows.append(
            {
                "source_file": str(path.relative_to(ROOT)),
                "experiment": metric.get("experiment", path.stem),
                "script": metric.get("script", ""),
                "dataset": dataset_key,
                "seed": metric.get("seed", ""),
                "oa": values.get("oa"),
                "aa": values.get("aa"),
                "kappa": values.get("kappa"),
                "loss": values.get("loss"),
                "config": metric.get("config", {}),
                "datasets": metric_datasets(metric),
            }
        )
    return rows


def is_exact_honghu_group(metric: dict[str, Any]) -> bool:
    return set(metric_datasets(metric)) == {"PaviaU", "Xuzhou", "HongHu"}


def is_old_longkou_group(metric: dict[str, Any]) -> bool:
    return "WHUL" in set(metric_datasets(metric))


def format_missing(value: Any) -> str:
    if value is None:
        return ""
    return str(value)

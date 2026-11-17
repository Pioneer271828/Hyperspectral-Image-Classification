"""Load a reviewed, byte-pinned calibration table for a delivered cube."""

from hashlib import sha256
import json
import re

from loadData.observation_sampling import BandMetadata, _digest
from loadData.dataset_registry import file_hash


SHA256 = re.compile(r"[a-f0-9]{64}")
UNITS = {"nm": 1, "um": 1000, "µm": 1000, "μm": 1000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_calibration(path, *, dataset, bands, image_path, approval):
    """Authenticate bytes against an external scientific and license review.

    A digest binds the reviewed artifacts. The external reviewer establishes
    whether the table comes from measurement and whether its use is permitted.
    """
    require(isinstance(approval, dict) and
            set(approval) == {"dataset", "image_sha256", "table_sha256", "authority", "version",
                              "license", "evidence_url", "evidence_sha256", "retained_band_ids"}
            and approval["dataset"] == dataset
            and all(isinstance(approval[name], str) and SHA256.fullmatch(approval[name])
                    for name in ("table_sha256", "image_sha256", "evidence_sha256"))
            and all(isinstance(approval[name], str) and approval[name].strip()
                    for name in ("authority", "version", "license", "evidence_url")),
            "Explicit reviewed source and delivered cube identity required")
    require(image_path.is_file() and file_hash(image_path) == approval["image_sha256"],
            "Delivered image bytes differ from the external approval")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == approval["table_sha256"], "Spectral table bytes differ")
    table = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(table, dict) and
            set(table) == {"schema", "dataset", "image_sha256", "authority", "version",
                           "center_unit", "fwhm_unit", "bands"}
            and table["schema"] == "delivered-spectral-calibration-v1"
            and all(table[key] == approval[key] for key in ("dataset", "image_sha256", "authority", "version")),
            "Spectral table release identity differs")
    rows = table["bands"]
    require(type(bands) is int and bands > 0 and isinstance(rows, list) and len(rows) == bands
            and all(isinstance(row, dict) and set(row) == {"original_band_id", "center", "fwhm", "valid"}
                    for row in rows), "One calibration row is required for every delivered column")
    ids = tuple(row["original_band_id"] for row in rows)
    mapping = approval["retained_band_ids"]
    require(isinstance(mapping, (list, tuple)) and all(type(value) is int for value in mapping)
            and ids == tuple(mapping), "Delivered-to-original band mapping differs")
    centers = convert((row["center"] for row in rows), table["center_unit"])
    widths = tuple(row["fwhm"] for row in rows)
    if table["fwhm_unit"] is None:
        require(all(value is None for value in widths), "Unknown FWHM must be absent for every band")
        widths = None
    else:
        widths = convert(widths, table["fwhm_unit"])
    return BandMetadata("calibration:" + _digest(approval), ids, centers, "nm", "verified",
                        tuple(row["valid"] for row in rows), widths, None if widths is None else "nm")


def unique_object(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate calibration field")
    return result


def convert(values, unit):
    values = tuple(values)
    require(isinstance(unit, str) and unit in UNITS and all(type(value) in (int, float) for value in values),
            "Explicit physical unit and complete numeric vector required")
    return tuple(value * UNITS[unit] for value in values)

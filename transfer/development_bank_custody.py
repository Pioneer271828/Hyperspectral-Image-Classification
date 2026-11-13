"""Enumerate only previously released spatial roles and their bounded origins."""

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
import torch

from research_runtime import hash_file
from transfer import development_block_custody as prior, p4_guarded, p4_spatial, registered_development as transport
from transfer.development_bank import BankRelease, _row_key
from transfer.development_blocks import SCENES, inside, read_pin
from transfer.development_mat import open_guarded
from transfer.g3_training import state_hash


@dataclass
class Intake:
    release: BankRelease
    baseline: torch.nn.Module
    read: object
    protected: tuple


def _samples(shape, tiles, groups):
    result = {row["id"]: [] for row in tiles}
    width = shape[1]
    for role, group in groups.items():
        lookup = {}
        for row in tiles:
            if row["role"] != role:
                continue
            a, b, c, d = row["box"]
            for y in range(a, b):
                for x in range(c, d):
                    lookup[y * width + x] = row["id"]
        for pixel, label in zip(group["indices"], group["labels"], strict=True):
            if pixel not in lookup:
                raise ValueError("Released sample escapes authorized development geometry")
            result[lookup[pixel]].append([pixel, label])
    return result


def _rows(plan, tiles, groups, origins, representation="raw_fp32"):
    samples = _samples(plan["shape"], tiles, groups)
    rows = []
    for tile in sorted(tiles, key=_row_key):
        key = tile["id"]
        parent = tile.get("parent_tile", key)
        origin, blocked = origins(tile)
        rows.append(dict(id=key, role=tile["role"], partition=tile["partition"], box=tile["box"],
                         parent_tile=parent, parent_box=plan["tiles"][parent]["box"], samples=samples[key],
                         origin=origin, representation=representation, blocked=blocked))
    return tuple(rows)


def _groups(support, validation):
    return {"devtrain": support, "devval": validation}


def xuzhou(root):
    admitted = prior.xuzhou(root)
    data = prior._json(admitted.release.custody["development"], root)
    plan, record = data["plan"], data["record"]
    tiles = [row for row in record["roles"] if row["role"] in ("devtrain", "devval")]
    by_id = {row["id"]: row for row in data["blocks"]}
    if len(by_id) != len(data["blocks"]) or set(by_id) != {row["id"] for row in tiles}:
        raise ValueError("Xuzhou published block roster differs from its authorized geometry")
    folder = Path(admitted.release.custody["development"]["path"]).parent

    def origin(row):
        entry = by_id[row["id"]]
        expected = f"tile-{row['id']}.npy"
        if entry["role"] != row["role"] or entry["name"] != expected:
            raise ValueError("Xuzhou publication role or relative filename differs")
        path = folder / expected
        if not path.exists():
            return None, "PINNED_DEVELOPMENT_ARRAY_MISSING"
        return {"path": str(inside(path, folder)), "sha256": entry["sha256"]}, None

    rows = _rows(plan, tiles, _groups(record["train_support"], record["validation_support"]), origin)
    release = admitted.release
    custody = {**release.custody, "selection": "all_original_manifest_devtrain_and_devval",
               "release_seed": 512, "validation_seed": 3031, "per_class": 5, "source_has_fresh_head": True}
    bank = BankRelease(release.scene, "nested", release.shape, release.classes, release.plan_hash, release.bands,
                       release.statistics, release.source, custody, rows)

    def read(row):
        return np.load(io.BytesIO(read_pin(row["origin"], folder)), allow_pickle=False)

    return Intake(bank, admitted.baseline, read, admitted.protected)


def honghu(root):
    admitted = prior.honghu(root)
    original = admitted.release
    custody = original.custody
    registration = prior._json(custody["registration"], root)
    plan = prior._json(custody["role_plan"], root)
    geometry = prior._json(custody["geometry"], root)
    labels = prior._json(custody["label_release"], root)
    cut = prior._json(prior._pin(registration, registration["guarded"]["cut_audit"]), root)
    shape = original.shape
    image = original.roles[0]["origin"]
    tiles = [dict(row, partition="train" if row["role"] == "devtrain" else "val")
             for row in geometry["roles"] if row["role"] in ("devtrain", "devval")]
    rows = _rows(plan, tiles, _groups(labels["support"], labels["validation"]), lambda _: (image, None))
    cut_counts = cut["chosen_geometry_for_followup"]
    authority = {**custody, "selection": "all_original_guarded_devtrain_and_devval",
        "release_seed": 512, "validation_seed": 3031, "per_class": 5, "source_has_fresh_head": True,
        "curator_available_per_class": {"devtrain": cut_counts["train_counts"], "devval": cut_counts["validation_counts"]},
        "class_21_shared_parent": 65, "guarded_sibling_boxes": [cut_counts["first_box"], cut_counts["second_box"]]}
    bank = BankRelease(original.scene, "guarded", shape, original.classes, original.plan_hash, original.bands,
                       original.statistics, original.source, authority, rows)
    bounded = None

    def read(row):
        nonlocal bounded
        if bounded is None:
            bounded = open_guarded(image, root, "WHU_Hi_HongHu", shape, rows)
        return bounded(row)

    return Intake(bank, admitted.baseline, read, admitted.protected)


PAVIA_OBSERVATION = ("wp09-online-tta-preparation01/legacy/provenance.json",
                     "ee6d050a1ac186399ce8dc994a6124f30f166c768b374c33729fbb14ce143f71")


def pavia(root):
    status = prior.pavia_status(root)
    shape, classes = SCENES["paviaU"]
    plan, labels, stats = [prior._json(status[name], root) for name in (
        "role_plan", "label_release", "training_statistics")]
    name, checksum = PAVIA_OBSERVATION
    archive_pin = {"path": str(root / name), "sha256": checksum}
    archive = prior._json(archive_pin, root)
    bands, physical = prior._band_metadata(root, "paviaU")
    if (archive["origin_hash"] != status["training_only_observation_receipt"]["sha256"]
            or archive["bands_hash"] != bands.fingerprint or archive["selected_step"] != 7):
        raise ValueError("PaviaU normalized observation provenance differs")
    old = dict(zip(status["training_only_tiles"], archive["observations"].values(), strict=True))
    for tile, pin in old.items():
        if Path(pin["path"]).name != f"tile-{tile}.npz" or plan["tiles"][tile]["partition"] != "train":
            raise ValueError("PaviaU persisted block is outside fixed training geometry")
        inside(pin["path"], root)
    tiles = [dict(row, role="devtrain" if row["partition"] == "train" else "devval")
             for row in plan["tiles"] if row["partition"] in ("train", "val")]
    rows = _rows(plan, tiles, _groups(labels["support"], labels["validation"]),
                 lambda row: (old[row["id"]], None) if row["id"] in old else (
                     None, "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED" if row["role"] == "devval" else
                     "DEVTRAIN_BOUNDED_ARRAY_NOT_PUBLISHED"), "normalized_fp32")
    model_files = (archive["source"]["weights"], archive["source"]["sidecar"])
    for pin in model_files:
        inside(pin["path"], root)
    baseline = transport._load_source(*model_files)
    source_state = state_hash(baseline)
    source = dict(weight=model_files[0], sidecar=model_files[1], state=source_state, shared_state=source_state,
                  initialization="accepted_pavia_p4_selected_domain_bn_step_7")
    custody = dict(selection="fixed_spatial_with_partial_normalized_e4b_arrays", original_role_plan=status["role_plan"],
        label_release=status["label_release"], training_statistics=status["training_statistics"],
        staging=status["nonpersistent_staging"], source_provenance=archive_pin, release_seed=512, validation_seed=3031,
        per_class=5, minimum_buffer=2 * plan["gap"], source_has_fresh_head=False, physical=physical)
    bank = BankRelease("paviaU", "fixed", shape, classes, plan["plan_hash"], bands, stats, source, custody, rows)

    def read(row):
        y0, y1, x0, x1 = row["box"]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        coords = np.stack((yy / (shape[0] - 1), xx / (shape[1] - 1)), -1).astype(np.float32)
        with np.load(io.BytesIO(read_pin(row["origin"], root)), allow_pickle=False) as data:
            if (sorted(data.files) != ["coordinates", "spectral", "wavelengths_nm"]
                    or not np.array_equal(data["coordinates"], coords)
                    or not np.array_equal(data["wavelengths_nm"], np.asarray(bands.wavelengths, dtype=np.float32))):
                raise ValueError("PaviaU published coordinates or wave vector differs")
            return data["spectral"]

    return Intake(bank, baseline, read, (archive_pin, *model_files, *old.values(),
                  status["role_plan"], status["training_statistics"], status["label_release"]))


PROVIDERS = {"xuzhou": xuzhou, "whu-honghu": honghu, "paviaU": pavia}


def protect(intake):
    pins = {item["path"]: item["sha256"] for item in intake.protected}
    for row in intake.release.rows:
        if row["origin"] is not None:
            pin = row["origin"]
            if pin["path"] in pins and pins[pin["path"]] != pin["sha256"]:
                raise ValueError("Bank origin references conflicting hashes")
            pins[pin["path"]] = pin["sha256"]
    if any(hash_file(path) != sha for path, sha in pins.items()):
        raise ValueError("Original input changed")
    return len(pins)

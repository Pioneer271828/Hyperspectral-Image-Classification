"""Publish and replay exact blocks from an existing authenticated spatial release."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from loadData.observation_sampling import BandMetadata, sample, subset_statistics
from loadData.spatial_protocol import validate_statistics
from loadData.spatial_tiles import tile_slice
from model.research_config import canonical_json, digest
from research_runtime import external_directory, tensor_hash
from transfer import p4_guarded, registered_development as transport
from transfer.g3_training import Scene


SCENES = {
    "paviaU": ((610, 340, 103), 9),
    "xuzhou": ((500, 260, 436), 9),
    "whu-honghu": ((940, 475, 270), 22),
}
ALIASES = {"paviau": "paviaU", "xuzhou": "xuzhou", "honghu": "whu-honghu",
           "whuh": "whu-honghu", "whu-honghu": "whu-honghu",
           "whu-hi-honghu": "whu-honghu"}
ROLES = ("devtrain", "devval")
KIND = "wp02_spatial_development_blocks_v1"


def scene_name(value):
    if not isinstance(value, str):
        raise ValueError("Expected a registered scene identity")
    try:
        return ALIASES[value.lower().replace("_", "-")]
    except KeyError as error:
        raise ValueError("Unknown development scene alias") from error


def scene_roster(names):
    canonical = [scene_name(name) for name in names]
    if len(canonical) != len(set(canonical)):
        raise ValueError("Duplicate scene aliases in development request")
    return canonical


def inside(path, root):
    root = external_directory(Path(root).resolve(strict=True))
    path = Path(path)
    resolved = path.resolve(strict=True)
    if path != resolved or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Input path escapes the authorized root or is aliased")
    return resolved


def read_pin(pin, root):
    inside(pin["path"], root)
    return transport._read(pin)


@dataclass(frozen=True)
class Release:
    scene: str
    profile: str
    shape: tuple
    classes: int
    plan_hash: str
    bands: BandMetadata
    statistics: dict
    roles: tuple
    custody: dict
    physical: dict
    source: dict

    def identity(self):
        if self.scene != scene_name(self.scene) or self.profile not in (
                "fixed_spatial", "nested_development", "guarded_development", "engineering_fixture"):
            raise ValueError("Unsupported scene or spatial custody profile")
        if (self.profile != "engineering_fixture" and (self.shape, self.classes) != SCENES[self.scene]) or (
                len(self.shape) != 3 or any(type(size) is not int or size < 1 for size in self.shape)
                or len(self.bands.band_ids) != self.shape[2] or self.classes != SCENES[self.scene][1]):
            raise ValueError("Scene shape, class count or band metadata differs")
        validate_statistics(self.statistics, {"plan_hash": self.plan_hash})
        if len(self.statistics["minimum"]) != self.shape[2]:
            raise ValueError("Training statistics and delivered bands differ")
        if (not self.roles or len(self.roles) > 2
                or len({row["id"] for row in self.roles}) != len(self.roles)
                or [row["role"] for row in self.roles] != [role for role in ROLES
                                                           if any(row["role"] == role for row in self.roles)]):
            raise ValueError("Select at most one authenticated block for each development role")
        for row in self.roles:
            if set(row) != {"id", "role", "partition", "box", "parent_tile", "samples", "origin", "origin_content_sha256"}:
                raise ValueError("Development block metadata fields differ")
            y0, y1, x0, x1 = row["box"]
            if (type(row["id"]) is not int or row["id"] < 0 or type(row["parent_tile"]) is not int
                    or row["parent_tile"] < 0 or any(type(i) is not int for i in row["box"])
                    or not (0 <= y0 < y1 <= self.shape[0] and 0 <= x0 < x1 <= self.shape[1])
                    or row["partition"] not in (("train",) if row["role"] == "devtrain" else ("val", "train"))
                    or (self.profile in ("fixed_spatial", "guarded_development")
                        and row["partition"] != ("train" if row["role"] == "devtrain" else "val"))):
                raise ValueError("Block outside its spatial role or scene bounds")
            checksum = row["origin_content_sha256"]
            if (not isinstance(checksum, str) or len(checksum) != 64 or set(checksum) - set("0123456789abcdef")):
                raise ValueError("Development origin requires a bound array digest")
            sample_identity(row, self.shape, self.classes)
        if len(self.roles) == 2:
            first, second = self.roles
            a, b, c, d = first["box"]
            e, f, g, h = second["box"]
            gap = max(e - b, a - f, g - d, c - h)
            minimum = self.custody["minimum_buffer"]
            if type(minimum) is not int or minimum < 0 or gap < minimum:
                raise ValueError("Development blocks cross their spatial buffer")
        return json.loads(canonical_json(dict(kind=KIND, scene=self.scene, profile=self.profile,
            shape=self.shape, classes=self.classes, plan_hash=self.plan_hash, bands=asdict(self.bands),
            band_order=self.bands.fingerprint, statistics_hash=self.statistics["statistics_hash"],
            custody=self.custody, physical=self.physical, source=self.source,
            roles=[{**{k: v for k, v in row.items() if k != "samples"},
                    "released_samples": sample_identity(row, self.shape, self.classes)} for row in self.roles])))


def sample_identity(row, shape, classes):
    y0, y1, x0, x1 = row["box"]
    points = row["samples"]
    ids = []
    for sample_id, label in points:
        if (type(sample_id) is not int or type(label) is not int
                or not y0 <= sample_id // shape[1] < y1 or not x0 <= sample_id % shape[1] < x1
                or not 1 <= label <= classes):
            raise ValueError("Released sample identity escapes its role, box or class range")
        ids.append(sample_id)
    if ids != sorted(set(ids)):
        raise ValueError("Released samples must keep unique ordered pixel identities")
    return {"count": len(ids), "sha256": digest(points)}


def block_values(array, row, bands):
    y0, y1, x0, x1 = row["box"]
    if (not isinstance(array, np.ndarray) or array.dtype != np.float32
            or array.shape != (y1 - y0, x1 - x0, bands) or not np.isfinite(array).all()):
        raise ValueError("Development block must contain finite FP32 values matching its geometry and bands")
    return np.ascontiguousarray(array)


def array_hash(array):
    return tensor_hash(np.array(array, copy=True, order="C"))


def _publication(release, read_array):
    members, payloads = [], []
    for row in release.roles:
        array = block_values(read_array(row), row, release.shape[-1])
        with io.BytesIO() as buffer:
            np.save(buffer, array, allow_pickle=False)
            raw = buffer.getvalue()
        member = {"id": row["id"], "role": row["role"], "name": f"tile-{row['id']}.npy",
                  "array_sha256": array_hash(array), "sha256": transport._sha(raw),
                  "bytes": len(raw), "dtype": array.dtype.str}
        if member["array_sha256"] != row["origin_content_sha256"]:
            raise ValueError("Published content differs from the authenticated origin array")
        if any(item["array_sha256"] == member["array_sha256"] for item in members):
            raise ValueError("Identical block content cannot occupy different roles")
        members.append(member)
        payloads.append((member["name"], raw))
    metadata = (canonical_json(dict(identity=release.identity(), members=members)) + "\n").encode()
    return metadata, payloads


def recover(path, root, release, read_array):
    """Re-derive the manifest from custody and verify the existing publication in place."""
    metadata, _ = _publication(release, read_array)
    pin = {"path": str(path), "manifest_sha256": transport._sha(metadata)}
    opened = Blocks(pin, root, release)
    for row in release.roles:
        opened.read(row["id"], row["role"], row["box"])
    return pin


def publish(path, release, read_array):
    """Commit a single immutable ZIP by hard link, with pinned metadata and NPY members."""
    release.identity()
    path = external_directory(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            metadata, payloads = _publication(release, read_array)
            with ZipFile(stream, "w", compression=ZIP_STORED) as archive:
                for name, raw in payloads:
                    archive.writestr(name, raw)
                archive.writestr("development.json", metadata)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {"path": str(path), "manifest_sha256": transport._sha(metadata)}


class Blocks:
    """Verify release identity and requested role before opening a single member."""

    def __init__(self, pin, root, release):
        self.path = inside(pin["path"], root)
        self.release = release = deepcopy(release)
        with ZipFile(self.path) as archive:
            info = archive.getinfo("development.json")
            if info.file_size > 1024 * 1024:
                raise ValueError("Development manifest exceeds metadata bound")
            raw = archive.read(info)
            if transport._sha(raw) != pin["manifest_sha256"]:
                raise ValueError("Development manifest hash differs")
            metadata = json.loads(raw)
            if set(metadata) != {"identity", "members"} or metadata["identity"] != release.identity():
                raise ValueError("Development manifest differs from authenticated release")
            members = metadata["members"]
            expected = [(row["id"], row["role"], f"tile-{row['id']}.npy") for row in release.roles]
            if ([(m["id"], m["role"], m["name"]) for m in members] != expected
                    or len({m["array_sha256"] for m in members}) != len(members)
                    or sorted(archive.namelist()) != sorted(["development.json", *(x[2] for x in expected)])):
                raise ValueError("Development member roster or role content differs")
            for member, row in zip(members, release.roles, strict=True):
                y0, y1, x0, x1 = row["box"]
                size = (y1 - y0) * (x1 - x0) * release.shape[2] * 4
                item = archive.getinfo(member["name"])
                if (set(member) != {"id", "role", "name", "array_sha256", "sha256", "bytes", "dtype"}
                        or member["array_sha256"] != row["origin_content_sha256"]
                        or member["dtype"] != "<f4" or not size < member["bytes"] <= size + 4096
                        or item.file_size != member["bytes"] or item.compress_type != ZIP_STORED):
                    raise ValueError("Invalid bounded FP32 development member")
        self.members = {m["id"]: m for m in members}

    def read(self, tile_id, role, box):
        row = next((row for row in self.release.roles if row["id"] == tile_id), None)
        if (type(tile_id) is not int or row is None or (row["role"], row["box"]) != (role, box)):
            raise ValueError("Request differs from admitted development block role or context")
        member = self.members[tile_id]
        with ZipFile(self.path) as archive:
            info = archive.getinfo(member["name"])
            if info.file_size != member["bytes"] or info.compress_type != ZIP_STORED:
                raise ValueError("Development block size changed")
            raw = archive.read(info)
        if transport._sha(raw) != member["sha256"]:
            raise ValueError("Development block file hash differs")
        array = block_values(np.load(io.BytesIO(raw), allow_pickle=False), row, self.release.shape[-1])
        if array_hash(array) != member["array_sha256"]:
            raise ValueError("Development array content differs")
        array.setflags(write=False)
        return array


class BlockCube:
    """A scene-shaped view that can materialize only the single admitted rectangle."""

    ndim = 3

    def __init__(self, shape, tile, block):
        self.shape = (*shape[:2], block.shape[2])
        self.window = tile_slice(tile)
        self.block = block

    def __getitem__(self, key):
        if key != self.window:
            raise ValueError("Input access outside the bound block")
        return self.block


def view(blocks, row, condition):
    release = blocks.release
    block = blocks.read(row["id"], row["role"], row["box"])
    tile = {key: row[key] for key in ("id", "partition", "box")}
    plan = {"shape": list(release.shape[:2]), "plan_hash": release.plan_hash, "tiles": {tile["id"]: tile}}
    scene = Scene(release.scene, BlockCube(release.shape, tile, block), plan,
                  release.bands, release.classes, release.statistics)
    allowed = {tile["partition"]: [tile]}
    original = p4_guarded.observation_provider(scene, allowed)(tile)
    observed, selection = sample(original.observation, release.bands, condition)
    statistics = subset_statistics(release.statistics, release.bands, selection)
    positions = selection["selected_positions"]
    retained = np.ascontiguousarray(block[:, :, positions])
    retained.setflags(write=False)
    scene = Scene(scene.dataset, BlockCube(release.shape, tile, retained), plan, observed.bands,
                  scene.classes, statistics)
    identity = dict(tile=tile["id"], role=row["role"], box=row["box"], partition=tile["partition"],
        original=blocks.members[tile["id"]]["array_sha256"], raw=array_hash(retained),
        observed=array_hash(observed.observation.spectral),
        coordinates=array_hash(observed.observation.coordinates), ids=array_hash(observed.observation.global_ids),
        band_ids=list(observed.bands.band_ids), band_order=observed.bands.fingerprint, selection=selection,
        statistics=statistics, preprocessing=release.statistics["statistics_hash"])
    return scene, tile, observed, identity

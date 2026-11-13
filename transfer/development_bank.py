"""Versioned development observations with immutable, resumable block publication."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from loadData.observation_sampling import BandMetadata, sample, subset_statistics
from loadData.spatial_protocol import PartitionObservation, validate_statistics
from model.research_config import canonical_json, digest
from research_runtime import external_directory
from transfer import registered_development as transport
from transfer.development_blocks import BlockCube, ROLES, SCENES, array_hash, block_values, inside, sample_identity
from transfer.g3_training import Scene


KIND = "wp02_complete_development_bank_v1"


def _plain(value):
    return json.loads(canonical_json(value))


def _box(box, shape):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and all(type(i) is int for i in box)
            and 0 <= box[0] < box[1] <= shape[0] and 0 <= box[2] < box[3] <= shape[1])


def _gap(a, b):
    return max(b[0] - a[1], a[0] - b[1], b[2] - a[3], a[2] - b[3])


def _row_key(row):
    return ROLES.index(row["role"]), row["id"]


def _json_bytes(value):
    return (canonical_json(value) + "\n").encode()


def _atomic_new(path, data):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _relative(root, name):
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError("Bank member must be a registered relative filename")
    return inside(root / path, root)


def _read_json(root, name):
    path = _relative(root, name)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Bank metadata exceeds bounded size")
    return path.read_bytes()


def _sha(value):
    return transport._sha(value)


@dataclass(frozen=True)
class BankRelease:
    scene: str
    profile: str
    shape: tuple
    classes: int
    plan_hash: str
    bands: BandMetadata
    statistics: dict
    source: dict
    custody: dict
    rows: tuple

    def identity(self):
        if self.scene not in SCENES or self.profile not in ("nested", "guarded", "fixed", "synthetic"):
            raise ValueError("Unknown bank scene or release profile")
        if (self.profile != "synthetic" and (self.shape, self.classes) != SCENES[self.scene]) or (
                len(self.shape) != 3 or any(type(n) is not int or n <= 0 for n in self.shape)
                or len(self.bands.band_ids) != self.shape[-1] or self.classes != SCENES[self.scene][1]):
            raise ValueError("Bank dimensions, bands or class count differ")
        validate_statistics(self.statistics, {"plan_hash": self.plan_hash})
        if len(self.statistics["minimum"]) != self.shape[-1]:
            raise ValueError("Bank statistics band count differs")
        if not self.rows or tuple(sorted(self.rows, key=_row_key)) != self.rows:
            raise ValueError("Bank roster must be sorted by role and tile")
        seen, samples = set(), set()
        for row in self.rows:
            if set(row) != {"id", "role", "partition", "box", "parent_tile", "parent_box", "samples", "origin",
                           "representation", "blocked"}:
                raise ValueError("Unexpected bank row fields")
            key, role, box = row["id"], row["role"], row["box"]
            parent = row["parent_tile"]
            expected_partition = "train" if role == "devtrain" or self.profile == "nested" else "val"
            if (type(key) is not int or key < 0 or key in seen or type(parent) is not int or parent < 0
                    or row["partition"] != expected_partition or not _box(box, self.shape)
                    or not _box(row["parent_box"], self.shape)):
                raise ValueError("Bank tile identity, role, partition or bounds differ")
            a, b, c, d = row["parent_box"]
            e, f, g, h = box
            if not (a <= e < f <= b and c <= g < h <= d) or (parent == key and box != row["parent_box"]):
                raise ValueError("Bank tile crosses the registered parent geometry")
            if (row["representation"] not in ("raw_fp32", "normalized_fp32")
                    or (row["representation"] == "normalized_fp32" and self.profile != "fixed")
                    or (row["blocked"] is None) != (row["origin"] is not None)):
                raise ValueError("Bank row lacks an authorized source or exact blocked status")
            sample_identity(row, self.shape, self.classes)
            ids = {i for i, _ in row["samples"]}
            if samples & ids:
                raise ValueError("Bank sample IDs repeat between blocks")
            seen.add(key)
            samples.update(ids)
        train = [row for row in self.rows if row["role"] == "devtrain"]
        val = [row for row in self.rows if row["role"] == "devval"]
        gap = self.custody["minimum_buffer"]
        if type(gap) is not int or gap < 0 or not train or not val:
            raise ValueError("Bank needs an explicit guarded train/validation roster")
        for i, row in enumerate(self.rows):
            for other in self.rows[i + 1:]:
                if _gap(row["box"], other["box"]) < (gap if row["role"] != other["role"] else 0):
                    raise ValueError("Bank blocks overlap or cross the role buffer")
        return _plain(dict(kind=KIND, scene=self.scene, profile=self.profile, shape=self.shape,
            classes=self.classes, plan_hash=self.plan_hash, bands=asdict(self.bands),
            band_order=self.bands.fingerprint, statistics=self.statistics, source=self.source, custody=self.custody,
            rows=[{**{key: value for key, value in row.items() if key != "samples"},
                   "released_samples": sample_identity(row, self.shape, self.classes)} for row in self.rows]))

    def coverage(self):
        counts = {}
        for role in ROLES:
            rows = [row for row in self.rows if row["role"] == role]
            published = [row for row in rows if row["blocked"] is None]
            totals = lambda group: np.bincount([label for row in group for _, label in row["samples"]],
                                              minlength=self.classes + 1)[1:].tolist()
            counts[role] = dict(authorized_blocks=len(rows), publishable_blocks=len(published),
                published_labeled_blocks=sum(bool(row["samples"]) for row in published),
                released_per_class=totals(rows), publishable_per_class=totals(published))
        return counts


def _files(row):
    return f"tile-{row['id']}.json", f"tile-{row['id']}.npy"


def _manifest(root, release, registration, row):
    name, member = _files(row)
    raw = _read_json(root, name)
    value = json.loads(raw)
    metadata = {key: v for key, v in value.items() if key != "content"}
    expected = dict(registration=registration, row=next(r for r in release["rows"] if r["id"] == row["id"]),
                    member=member, dtype="<f4")
    if metadata != expected or not isinstance(value.get("content"), dict):
        raise ValueError("Existing bank block manifest differs from its authority")
    content = value["content"]
    if set(content) != {"npy_sha256", "array_sha256", "bytes"} or any(
            not isinstance(content[k], str) or len(content[k]) != 64 or set(content[k]) - set("0123456789abcdef")
            for k in ("npy_sha256", "array_sha256")):
        raise ValueError("Invalid bank block content digest")
    y0, y1, x0, x1 = row["box"]
    size = (y1 - y0) * (x1 - x0) * release["shape"][-1] * 4
    if type(content["bytes"]) is not int or not size < content["bytes"] <= size + 4096:
        raise ValueError("Bank NPY exceeds the registered block extent")
    return value, _sha(raw)


def _decode(root, member, content, row, bands):
    path = _relative(root, member)
    if path.stat().st_size != content["bytes"]:
        raise ValueError("Bank block byte count differs")
    with path.open("rb") as stream:
        raw = stream.read(content["bytes"] + 1)
    if _sha(raw) != content["npy_sha256"]:
        raise ValueError("Bank NPY digest differs")
    # The producer emits ordinary NPY v1. Check the header before NumPy can
    # allocate from a hostile shape in a payload discovered during recovery.
    stream = io.BytesIO(raw)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("Unexpected bank NPY version")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    a, b, c, d = row["box"]
    if (shape != (b-a, d-c, bands) or fortran or dtype != np.dtype("<f4")
            or len(raw) - stream.tell() != (b-a)*(d-c)*bands*4):
        raise ValueError("Bank NPY header differs from the authorized block")
    stream.seek(0)
    value = block_values(np.load(stream, allow_pickle=False), row, bands)
    if array_hash(value) != content["array_sha256"]:
        raise ValueError("Bank decoded content differs")
    value.setflags(write=False)
    return value


def _bank_root(path, authorized, *, new=False):
    path = Path(path)
    authorized = external_directory(Path(authorized).resolve(strict=True))
    if path != path.resolve() or path.parent != authorized or path == authorized:
        raise ValueError("Bank path escapes the authorized root or is aliased")
    if not new and not path.is_dir():
        raise ValueError("Bank directory is missing")
    return path


def publish(path, authorized, release, reader, *, resume=False, stop=None):
    """Seal one member at a time. Pending metadata can resume without overwriting it."""
    if stop is not None and (type(stop) is not int or stop < 0):
        raise ValueError("Publication stop must be a nonnegative block count")
    release = deepcopy(release)
    identity = release.identity()
    registration = _sha(_json_bytes(identity))
    root = _bank_root(path, authorized, new=not resume)
    if resume:
        if _read_json(root, "registration.json") != _json_bytes(identity):
            raise ValueError("Bank registration differs before source access")
        if (root / "bank.json").exists():
            raise FileExistsError("Completed bank cannot be republished")
    else:
        root.mkdir()
        _atomic_new(root / "registration.json", _json_bytes(identity))
    rows = [row for row in release.rows if row["blocked"] is None]
    # Authenticate every already published record before opening any source array.
    expected_names = {"registration.json", *(name for row in rows for name in _files(row))}
    if any(path.name not in expected_names for path in root.iterdir() if not path.name.startswith("tmp")):
        raise ValueError("Unregistered bank member path")
    existing = {}
    for row in rows:
        name, member = _files(row)
        if (root / name).exists() or (root / name).is_symlink():
            existing[row["id"]] = _manifest(root, identity, registration, row)
            if (root / member).exists() or (root / member).is_symlink():
                _relative(root, member)
        elif (root / member).exists() or (root / member).is_symlink():
            raise ValueError("Unregistered array member cannot be adopted")
    completed = []
    payload_reads = 0
    source_payload_bytes = 0
    recovered_payload_bytes = 0
    recovered = 0
    for row in rows:
        name, member = _files(row)
        previous = existing.get(row["id"])
        if previous is not None and (root / member).exists():
            record, checksum = previous
            current = _decode(root, member, record["content"], row, release.shape[-1])
            original = block_values(reader(row), row, release.shape[-1])
            source_payload_bytes += original.nbytes
            recovered_payload_bytes += current.nbytes
            payload_reads += 1
            if array_hash(original) != record["content"]["array_sha256"] or not np.array_equal(current, original):
                raise ValueError("Existing bank block differs from its original authorized source")
            recovered += 1
        else:
            if stop is not None and len(completed) >= stop:
                return dict(status="PARTIAL", registered=registration, source_reads=payload_reads,
                            committed=len(completed), recovered=recovered, source_payload_bytes=source_payload_bytes,
                            recovered_payload_bytes=recovered_payload_bytes)
            value = block_values(reader(row), row, release.shape[-1])
            source_payload_bytes += value.nbytes
            payload_reads += 1
            with io.BytesIO() as stream:
                np.save(stream, value, allow_pickle=False)
                raw = stream.getvalue()
            content = dict(npy_sha256=_sha(raw), array_sha256=array_hash(value), bytes=len(raw))
            expected_digest = row["origin"].get("array_sha256")
            if expected_digest is not None and content["array_sha256"] != expected_digest:
                raise ValueError("Original synthetic block content differs")
            record = dict(registration=registration, row=next(r for r in identity["rows"] if r["id"] == row["id"]),
                          member=member, dtype="<f4", content=content)
            packet = _json_bytes(record)
            checksum = _sha(packet)
            if previous is None:
                _atomic_new(root / name, packet)
            elif previous != (record, checksum):
                raise ValueError("Pending bank record differs from original source")
            _atomic_new(root / member, raw)
        completed.append(dict(id=row["id"], role=row["role"], name=name, sha256=checksum))
    arrays = [json.loads(_read_json(root, item["name"]))["content"]["array_sha256"] for item in completed]
    if len(arrays) != len(set(arrays)):
        raise ValueError("Duplicate array content cannot be admitted to the bank")
    body = _json_bytes(dict(kind=KIND, registration=registration, members=completed))
    _atomic_new(root / "bank.json", body)
    return dict(status="COMPLETE", root=str(root), sha256=_sha(body), registration=registration,
                source_reads=payload_reads, committed=len(completed), recovered=recovered,
                source_payload_bytes=source_payload_bytes, recovered_payload_bytes=recovered_payload_bytes)


class Bank:
    def __init__(self, pin, authorized, release):
        self.root = _bank_root(pin["root"], authorized)
        self.release = deepcopy(release)
        self.identity = self.release.identity()
        reg = _read_json(self.root, "registration.json")
        if reg != _json_bytes(self.identity) or _sha(reg) != pin["registration"]:
            raise ValueError("Bank registration/custody differs")
        body = _read_json(self.root, "bank.json")
        if _sha(body) != pin["sha256"]:
            raise ValueError("Bank root hash differs")
        bank = json.loads(body)
        if set(bank) != {"kind", "registration", "members"} or (bank["kind"], bank["registration"]) != (
                KIND, pin["registration"]):
            raise ValueError("Bank completion identity differs")
        self.rows = {row["id"]: row for row in self.release.rows if row["blocked"] is None}
        self.members = {}
        expected = []
        for row in self.rows.values():
            record, checksum = _manifest(self.root, self.identity, pin["registration"], row)
            _relative(self.root, record["member"])
            self.members[row["id"]] = record
            expected.append(dict(id=row["id"], role=row["role"], name=_files(row)[0], sha256=checksum))
        content = [entry["content"]["array_sha256"] for entry in self.members.values()]
        if bank["members"] != expected or len(content) != len(set(content)):
            raise ValueError("Bank completion member roster or content differs")
        expected_names = {"bank.json", "registration.json",
                          *(name for row in self.rows.values() for name in _files(row))}
        if {path.name for path in self.root.iterdir() if not path.name.startswith("tmp")} != expected_names:
            raise ValueError("Bank contains an unregistered member")
        self.pin = {key: pin[key] for key in ("root", "sha256", "registration")}
        self.reads = 0
        self.bytes = 0

    def role_rows(self, role):
        if role not in ROLES:
            raise ValueError("Only released train/validation roles may be iterated")
        return tuple(row for row in self.rows.values() if row["role"] == role)

    def batch(self, role, tiles, condition):
        if role not in ROLES or len(tiles) != len(set(tiles)) or any(
                type(i) is not int or i not in self.rows or self.rows[i]["role"] != role for i in tiles):
            raise ValueError("Batch escapes its authorized role or duplicates a tile")
        return (self.materialize(self.rows[i], condition) for i in tiles)

    def materialize(self, row, condition):
        if self.rows.get(row["id"]) != row:
            raise ValueError("Observation escapes its registered bank context")
        entry = self.members[row["id"]]
        release = self.release
        block = _decode(self.root, entry["member"], entry["content"], row, release.shape[-1])
        self.reads += 1
        self.bytes += block.nbytes
        tile = {key: row[key] for key in ("id", "partition", "box")}
        plan = {"shape": list(release.shape[:2]), "plan_hash": release.plan_hash, "tiles": {row["id"]: tile}}
        y0, y1, x0, x1 = row["box"]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        coords = np.stack((yy / max(release.shape[0] - 1, 1), xx / max(release.shape[1] - 1, 1)), -1).astype(np.float32)
        ids = yy * release.shape[1] + xx
        if row["representation"] == "raw_fp32":
            low, high = (np.asarray(release.statistics[key], dtype=np.float32) for key in ("minimum", "maximum"))
            values = (block - low) / np.where(high > low, high - low, 1)
        else:
            values = block
        for array in (values, coords, ids):
            array.setflags(write=False)
        original = PartitionObservation(tile["partition"], values, coords, ids, release.plan_hash)
        observed, selection = sample(original, release.bands, condition)
        stats = subset_statistics(release.statistics, release.bands, selection)
        actual = np.ascontiguousarray(block[:, :, selection["selected_positions"]])
        actual.setflags(write=False)
        scene = Scene(release.scene, BlockCube(release.shape, tile, actual), plan, observed.bands,
                      release.classes, stats)
        context = dict(bank=self.pin["sha256"], tile=row["id"], role=row["role"], box=row["box"],
            source=release.source["shared_state"], raw=array_hash(actual), normalized=array_hash(observed.observation.spectral),
            original=entry["content"]["array_sha256"], coordinates=array_hash(coords), ids=array_hash(ids),
            band_order=observed.bands.fingerprint, positions=selection["selected_positions"], statistics=stats,
            representation=row["representation"], samples=sample_identity(row, release.shape, release.classes))
        return scene, tile, observed, context

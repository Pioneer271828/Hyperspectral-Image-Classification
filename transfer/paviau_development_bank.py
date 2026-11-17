"""Complete the fixed PaviaU development release from its pinned image."""

from dataclasses import replace
import hashlib
import io

import numpy as np
from scipy.io import loadmat

from loadData.spatial_tiles import observe_tile, tile_slice, validate_tile_plan
from transfer import development_bank as bank, development_bank_custody as custody
from transfer import development_block_custody as prior
from transfer.development_block_probe import VIEWS
from transfer.development_blocks import array_hash, inside
from transfer.g2_observations import pack_source_observations


class FixedImage:
    """Only this producer decodes the MAT. Consumers open published rectangles."""

    def __init__(self, image, datasets, plan, release):
        self.image = image
        self.datasets = datasets
        self.plan = plan
        self.release = release
        self._rows = {row["id"]: row for row in release.rows}
        self._cube = None
        self.decode_count = 0

    def _image(self):
        if self._cube is None:
            path = inside(self.image["path"], self.datasets)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.image["sha256"]:
                raise ValueError("PaviaU image bytes differ from its fixed release")
            image = loadmat(io.BytesIO(raw), variable_names=["paviaU"])["paviaU"]
            if image.shape != self.release.shape or image.dtype.kind not in "iuf":
                raise ValueError("Unexpected PaviaU image layout")
            self._cube = image
            self.decode_count += 1
        return self._cube

    def __call__(self, row):
        if self._rows.get(row["id"]) != row or row["role"] not in ("devtrain", "devval"):
            raise ValueError("PaviaU image request escapes its fixed role and rectangle")
        original = np.asarray(self._image()[tile_slice(row)], dtype=np.float32)
        lo, hi = (np.asarray(self.release.statistics[key], dtype=np.float32) for key in ("minimum", "maximum"))
        return (original - lo) / np.where(hi > lo, hi - lo, 1)


def intake(root):
    """Reuse the immutable geometry, labels, source receipts and trained extrema."""
    old = custody.pavia(root)
    release = old.release
    registration = prior._json(prior.historical(root, "pavia"), root)
    plan = prior._json(release.custody["original_role_plan"], root)
    validate_tile_plan(plan)
    image = prior._pin(registration, registration["image"])
    staging = prior._json(release.custody["staging"], root)
    if staging["frozen_inputs"].get(image["path"]) != image["sha256"]:
        raise ValueError("PaviaU image was not the agreed fixed observation source")
    source, source_pins, baseline = prior._spatial_source(root, registration, "paviaU", state_required=True)
    rows = tuple({**row, "origin": image, "blocked": None} for row in release.rows)
    release = replace(release, rows=rows, source=source, custody={
        **release.custody, "selection": "all_fixed_spatial_pavia_devtrain_and_devval",
        "source_has_fresh_head": True, "legacy_e4b_source": release.source,
        "image_materialization": "pinned_mat_paviaU_fp32_frozen_train_extrema",
    })
    counts = release.coverage()
    if [(counts[role]["publishable_blocks"], counts[role]["publishable_per_class"])
            for role in ("devtrain", "devval")] != [(127, [5]*9), (41, [5]*9)]:
        raise ValueError("PaviaU fixed release lost geometry or original class coverage")
    reader = FixedImage(image, root.parent / "datasets", plan, release)
    return custody.Intake(release, baseline, reader, (*old.protected, *source_pins)), old


def verify_publication(opened, current, old, root):
    """Check every published observation against the original bytes and role plan."""
    reader = current.read
    plan, release = reader.plan, current.release
    shape = release.shape[:2]
    admitted_pixels = np.zeros(shape, dtype=bool)
    for row in release.rows:
        admitted_pixels[tile_slice(row)] = True
    test = [tile for tile in plan["tiles"] if tile["partition"] == "test"]
    if len(test) != 41 or any(admitted_pixels[tile_slice(tile)].any() for tile in test):
        raise ValueError("Held-out test geometry intersects the development release")
    # Only an in-memory copy is poisoned, including guard strips. Observed
    # values and the frozen role archives must be identical to the originals.
    poisoned = reader._image().astype(np.float32)
    poisoned[~admitted_pixels] = np.nan
    staging = prior._json(release.custody["staging"], root)
    expected_roles = {row["role"]: row["observation_sha256"] for row in staging["rows"]}
    raw_hashes = {}
    for role in ("train", "val"):
        payload = pack_source_observations("paviaU", reader.image["sha256"], plan, role, poisoned,
                                           release.bands.wavelengths)
        raw_hashes[role] = hashlib.sha256(payload).hexdigest()
    if raw_hashes != expected_roles:
        raise ValueError("Raw development values differ from the frozen role archives")
    legacy = {row["id"]: row for row in old.release.rows if row["origin"] is not None}
    manifests = opened.members
    audited = []
    for row in release.rows:
        reference = observe_tile(poisoned, plan, release.statistics, plan["tiles"][row["id"]])
        _, _, view, context = opened.materialize(row, VIEWS["full"])
        actual = view.observation
        for name in ("spectral", "coordinates", "global_ids"):
            if not np.array_equal(getattr(actual, name), getattr(reference, name)):
                raise ValueError(f"PaviaU published {name} differs in tile {row['id']}")
        if row["id"] in legacy and not np.array_equal(old.read(legacy[row["id"]]), actual.spectral):
            raise ValueError("Prior PaviaU NPZ is not semantically equivalent")
        audited.append({"tile": row["id"], "parent": row["parent_tile"], "role": row["role"],
                        "box": row["box"], "array": context["original"], "ids": context["ids"],
                        "coordinates": context["coordinates"], "manifest": manifests[row["id"]]["content"],
                        "values": actual.spectral.size, "raw": array_hash(poisoned[tile_slice(row)])})
    return {"coverage": release.coverage(), "blocks": audited, "legacy_equal": sorted(legacy),
            "test_blocks_excluded": len(test), "guard_and_test_poison_equal": True,
            "guard_and_test_pixels_excluded": int((~admitted_pixels).sum()),
            "raw_archive_sha256": raw_hashes, "image_decode_count": reader.decode_count,
            "statistics_refit": False, "protected_inputs": custody.protect(current)}


def publish(out, root, producer, *, resume=False, stop=None):
    current, old = intake(root)
    result = bank.publish(out, root, current.release, current.read, resume=resume, stop=stop)
    if result["status"] != "COMPLETE":
        return result
    opened = bank.Bank(result, root, current.release)
    return {**result, "kind": "wp02_paviau_completed_development_bank_v1", "producer": producer,
            "audit": verify_publication(opened, current, old, root),
            "raw_gt_loaded": False, "query_accessed": False, "optimization_updates": 0}


def open_publication(root, publication):
    receipt = prior._json(publication, root)
    if (receipt["kind"] != "wp02_paviau_completed_development_bank_v1" or receipt["status"] != "COMPLETE"
            or receipt["committed"] != 168 or receipt["optimization_updates"] != 0
            or receipt["query_accessed"] is not False or receipt["raw_gt_loaded"] is not False
            or receipt["audit"]["guard_and_test_poison_equal"] is not True
            or receipt["audit"]["legacy_equal"] != [0, 1]):
        raise ValueError("Expected the complete authenticated PaviaU bank publication")
    current, _ = intake(root)
    opened = bank.Bank({key: receipt[key] for key in ("root", "sha256", "registration")}, root, current.release)
    if current.release.coverage() != receipt["audit"]["coverage"]:
        raise ValueError("PaviaU publication and current frozen release disagree")
    return opened, publication, current

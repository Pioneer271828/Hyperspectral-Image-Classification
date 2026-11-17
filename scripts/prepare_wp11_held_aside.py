"""Export label-free candidate tiles from a pinned, previously audited image."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_confirmation import unused_tiles
from loadData.dataset_registry import DATASETS
from loadData.spatial_tiles import tile_slice
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.registered_development import _read


def prepare(registration, out, code):
    spec = json.loads(_read(registration))
    if set(spec) != {"scene", "preflight", "plan", "development", "image"}:
        raise ValueError("Exact image-only registration required")
    scene = spec["scene"]
    if scene not in DATASETS or DATASETS[scene][1] is None:
        raise ValueError("Registered MATLAB scene required")
    pins = [registration, spec["preflight"], spec["plan"], spec["development"], spec["image"]]
    image = external_directory(spec["image"]["path"]).resolve(strict=True)
    if out.exists() or any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pins):
        raise ValueError("New external output outside the input packages required")
    preflight, plan, development = (json.loads(_read(spec[key])) for key in ("preflight", "plan", "development"))
    rows = [row for row in preflight["rows"] if row["scene"] == scene]
    if (preflight["status"] != "passed" or preflight["raw_labels_read"] is not False
            or preflight["optimizer_updates"] != 0 or len(rows) != 1
            or rows[0]["plan_sha256"] != plan["plan_hash"]
            or rows[0]["image_sha256"] != spec["image"]["sha256"]
            or hash_file(image) != rows[0]["image_sha256"]):
        raise ValueError("Image and geometry must match the accepted image-only preflight")
    ids = unused_tiles(plan, development)
    if plan["gap"] < 1 or len(ids) < 2:
        raise ValueError("Insufficient disjoint held-aside geometry")
    variable = DATASETS[scene][1]
    cube = loadmat(image, variable_names=[variable])[variable]
    if cube.ndim != 3 or list(cube.shape) != rows[0]["shape"] or list(cube.shape[:2]) != plan["shape"]:
        raise ValueError("Image geometry differs from the preflight")
    out.mkdir()
    tiles = []
    for tile_id in ids:
        tile = plan["tiles"][tile_id]
        value = np.array(cube[tile_slice(tile)], dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite held-aside tile")
        path = out / f"tile-{tile_id}.npy"
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        tiles.append(dict(tile_id=tile_id, box=tile["box"], shape=list(value.shape),
                          file=dict(path=str(path), sha256=hash_file(path))))
    require_frozen_code(code)
    if any(hash_file(p["path"]) != p["sha256"] for p in pins):
        raise ValueError("Pinned preparation inputs changed")
    receipt = dict(status="candidates_prepared", scene=scene, plan_sha256=plan["plan_hash"],
                   development_sha256=development["validation_hash"], inputs=pins, tiles=tiles,
                   producer_commit=code["commit"], role_assignment=None, raw_labels_read=False,
                   optimizer_updates=0, query=0)
    write_json(out / "candidate-observations.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux preparation checkout required")
    prepare(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()

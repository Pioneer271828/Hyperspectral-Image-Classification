"""Run a fixed scikit-image SLIC release on synthetic and admitted development blocks."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
from tarfile import open as open_tar
from time import perf_counter
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from transfer.slic_assignment import slic_assignment


RELEASES = {
    "0.24.0": ("a160f384523ac70173d069954b7084bee79fb68e",
               "39ee0af13435c57351a3397eb379e72164ff85161923eec0c38849fecf1b4764",
               "62a4c03f60b1d6fa7fe035ab6bd0711247f8e2c7750f464fd796102c9956259f"),
    "0.25.2": ("e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae",
               "a4c464b90e978d137330be433df4e76d92ad3c5f46a22f159520ce0fdbea8a09",
               "134024267f3dde67d546b2464c1ec339ff1baa3393d3a21f1eb8315eb48edca6"),
}
BANKS = {
    "xuzhou": ("wp02-bank-xuzhou03.json", "8345449bd1a121672816bc23232587b20490e247f2885551b5c1e769fbe154c2", 100, 5),
    "paviaU": ("wp02-paviau-bank-nov18.json", "54c87a5f5d4a84bc4d6e4055dbf8874f633874dce7a01d82d79a66488073c8a4", 50, 41),
    "whu-honghu": ("wp02-bank-honghu03.json", "87c47c0eb04c3e93ec91b00c11ca0606266d41495a1550cd4167524842e52286", 100, 10),
}
SOURCE_FILES = ("LICENSE.txt", "skimage/segmentation/slic_superpixels.py",
                "skimage/segmentation/_slic.pyx", "skimage/segmentation/tests/test_slic.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, digest):
    require(hash_file(path) == digest, f"Wrong pinned bytes: {path.name}")
    return path


def check_release(args):
    import hashlib
    import skimage
    from skimage.segmentation import _slic, slic_superpixels

    version = args.release
    commit, wheel_hash, source_hash = RELEASES[version]
    wheel_path = pin(args.wheel, wheel_hash)
    official = json.loads(args.pypi.read_text())
    require(official["info"]["version"] == version and
            any(row["filename"] == wheel_path.name and row["digests"]["sha256"] == wheel_hash
                and row["url"].startswith("https://files.pythonhosted.org/") for row in official["urls"]),
            "Wheel digest not present in the official release listing")
    site = args.site.resolve(strict=True)
    require(skimage.__version__ == metadata.version("scikit-image") == version,
            "Select the locked scikit-image release in a separate process")
    loaded = [Path(module.__file__).resolve(strict=True) for module in (skimage, slic_superpixels, _slic)]
    require(all(path.is_relative_to(site) for path in loaded), "Unexpected scikit-image import location")
    with ZipFile(wheel_path) as wheel:
        members = [name for name in wheel.namelist()
                   if name.split("/")[0] in ("skimage", "scikit_image.libs") and not name.endswith("/")]
        for name in members:
            require((site / name).read_bytes() == wheel.read(name), f"Installed wheel member differs: {name}")
        packaged_license = [name for name in wheel.namelist() if ".dist-info/" in name
                            and name.endswith("/LICENSE.txt")]
        require(len(packaged_license) == 1, "Wheel license is missing")
        license_bytes = wheel.read(packaged_license[0])
        require((site / packaged_license[0]).read_bytes() == license_bytes, "Installed license bytes differ")
    with open_tar(pin(args.source, source_hash)) as archive:
        require(archive.pax_headers.get("comment") == commit and set(SOURCE_FILES) <= set(archive.getnames()),
                "Expected the fixed Git source archive")
        contents = {name: archive.extractfile(name).read() for name in SOURCE_FILES}
    require(contents["LICENSE.txt"] == license_bytes and
            contents["skimage/segmentation/slic_superpixels.py"] == loaded[1].read_bytes(),
            "Packaged license or Python SLIC layer differs from the fixed source")
    return dict(version=version, commit=commit, wheel=wheel_path.name, wheel_sha256=wheel_hash,
                source_sha256=source_hash, verified_package_members=len(members),
                source_files={name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
                installed_slic_extension_sha256=hash_file(loaded[2]))


def synthetic():
    rng = np.random.default_rng(512)
    for number, (h, w, bands, scale) in enumerate(
            ((15, 19, 3, 16), (28, 28, 9, 50), (28, 37, 22, 100),
             (7, 13, 1, 200), (14, 17, 3, 26), (28, 28, 103, 100)), 1):
        if number in (4, 5):
            cube = np.full((h, w, bands), 0.125 if number == 4 else 0, dtype=np.float32)
        else:
            cube = rng.random((h, w, bands), dtype=np.float32)
            cube[:h//2, :w//2] *= np.float32(0.2)
        yield f"fixture-{number:02d}", "synthetic", scale, cube, {}


def development(root):
    from transfer import development_bank as bank, development_bank_custody as custody
    from transfer import paviau_development_bank

    for scene, (name, checksum, scale, expected_tiles) in BANKS.items():
        path = pin(root / name, checksum)
        publication = json.loads(path.read_text())
        require(publication["status"] == "COMPLETE", "Expected a completed development bank")
        if scene == "paviaU":
            opened, _, _ = paviau_development_bank.open_publication(
                root, {"path": str(path), "sha256": checksum})
        else:
            admitted = custody.PROVIDERS[scene](root)
            opened = bank.Bank(publication, root, admitted.release)
        statistics = opened.release.statistics
        require(statistics["fit_partition"] == "train" and statistics.get("fit_role") in ("devtrain", None),
                "Expected the original training normalization authority")
        rows = opened.role_rows("devval")
        require(len(rows) == expected_tiles, f"Wrong released devval roster: {scene}")
        for ordinal, row in enumerate(rows, 1):
            _, _, view, context = opened.materialize(row, {"mode": "none"})
            require(context["role"] == "devval", "Only the released validation role may materialize")
            yield f"{scene}-{ordinal:02d}", scene, scale, view.observation.spectral, {
                "publication_sha256": checksum, "bank_sha256": opened.pin["sha256"],
                "observed_sha256": context["normalized"],
                "fit_partition": statistics["fit_partition"],
                "fit_role": statistics.get("fit_role"),
                "statistics_sha256": statistics["statistics_hash"],
            }
        require(opened.reads == expected_tiles, "Unexpected decoded block count")
        opened.verify_authority(opened.pin)


def capture(roster):
    from transfer.development_blocks import array_hash

    result = []
    for name, scene, scale, cube, input_pin in roster:
        before = array_hash(cube)
        if input_pin:
            require(before == input_pin["observed_sha256"], "Bank observation hash differs")
        samples = []
        for repeat in range(3):
            start = perf_counter()
            labels, q = slic_assignment(cube, scale=scale)
            samples.append(perf_counter() - start)
            if repeat == 0:
                exemplar = labels.copy()
            else:
                require(np.array_equal(exemplar, labels), "SLIC produced a different partition on repetition")
        require(array_hash(cube) == before and q.nnz == labels.size and q.shape[0] == labels.size and
                np.all(q.data == 1) and np.array_equal(q.indices, labels.ravel()),
                "Sparse membership or input snapshot changed")
        result.append(dict(name=name, scene=scene, role="synthetic" if not input_pin else "devval",
                           scale=scale, shape=list(cube.shape), input_sha256=before, input=input_pin,
                           labels=labels.ravel().tolist(), q_columns=q.shape[1], q_nnz=int(q.nnz),
                           elapsed_ms=[1000 * value for value in samples]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--release", choices=RELEASES, required=True)
    for path in ("runs", "site", "wheel", "pypi", "source", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    require(code["commit"] == args.expected_commit, "Check out the registered commit")
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    require(out.is_relative_to(root), "Evidence belongs under authorized runs")
    source = check_release(args)
    rows = capture(synthetic()) + capture(development(root))
    require(len(rows) == 6 + sum(row[3] for row in BANKS.values()), "Incorrect scene manifest")
    require_frozen_code(code)
    result = dict(schema="slic_portability_v1", commit=code["commit"], source=source, cases=rows,
                  real_updates=0, query_accessed=False, raw_gt_loaded=False, statistics_refit=False)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

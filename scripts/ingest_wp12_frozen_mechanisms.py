"""Publish the reviewed, de-identified WP05/WP06 mechanism supplement to WP12."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.wp12_frozen_mechanisms import ingest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import check, pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "cases"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("scope", "batch", "device"):
        parser.add_argument("--" + name, nargs=2, metavar=("PATH", "SHA256"), required=name != "device")
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected WP12 consumer checkout")
    root, destination = external_directory(args.root), external_directory(args.output)
    check(destination.is_relative_to(root) and destination != root and not destination.exists(),
          "Output must be a new child of the authorized runs directory")
    scope, batch, device = (pin(*pair) if pair else None for pair in (args.scope, args.batch, args.device))
    if device:
        check(Path(device["path"]).is_relative_to(root), "Device artifact escapes authorized runs")
    with TemporaryDirectory(dir=destination.parent, prefix=".wp12-mechanism-") as stage:
        publication = Path(stage) / "publication"
        ingest(root, scope, batch, args.cases, publication, code["commit"], device)
        require_frozen_code(code)
        os.rename(publication, destination)
    print(pin(destination / "receipt.json"))


if __name__ == "__main__":
    main()

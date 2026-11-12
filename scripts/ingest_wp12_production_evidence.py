"""Build the production WP12 table from pinned real WP10 and WP11 artifacts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_production_evidence import build, write_table
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer.registered_wp11_learning import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean exact Linux evidence adapter checkout required")
    pin = dict(path=args.registration, sha256=args.registration_sha256)
    spec = read(pin)
    out = external_directory(args.out)
    wp10_root = external_directory(spec["wp10"]["root"])
    protected = [wp10_root, Path(pin["path"]).resolve().parent,
                 *[Path(v["path"]).resolve().parent for group in ("wp11", "low_rank", "curriculum")
                   for v in spec.get(group, {}).values()]]
    if out.exists() or any(p == out or p in out.parents or out in p.parents for p in protected):
        raise ValueError("Fresh output separate from all evidence directories required")
    report = build(spec)
    require_frozen_code(code)
    read(pin)
    out.mkdir()
    write_table(out, dict(report, registration=pin, adapter_commit=code["commit"]))


if __name__ == "__main__":
    main()

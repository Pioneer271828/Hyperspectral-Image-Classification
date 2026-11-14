"""Import only fully pinned WP05–WP07 scopes and keep original WP12 table bytes."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_mechanism_admission as evidence
from research_runtime import code_snapshot, external_directory, require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    kinds = ("crash", "learning", "bn", "region")
    for key in (*kinds, "authorities"):
        parser.add_argument("--" + key, required=True, nargs=2)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    evidence.require(code["commit"] == args.expected_commit and root in out.parents, "Invalid adapter revision or output path")
    pin = lambda key: dict(zip(("path", "sha256"), getattr(args, key), strict=True))
    result = evidence.admit(out, root, {key: pin(key) for key in kinds}, pin("authorities"),
                            {k: code[k] for k in ("commit", "source_hash")})
    print(result, flush=True)


if __name__ == "__main__":
    main()

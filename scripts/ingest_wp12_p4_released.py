"""Consume reviewed P4 queue and released-point development preflights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_p4_released_evidence as evidence
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import registered_development as dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for name in ("runs", "p4-scope", "fit-template", "sparse", "queue-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("p4-review", "p4-release", "comparison"):
        parser.add_argument("--" + name, nargs=2, metavar=("PATH", "SHA256"))
    extension = ("duties-prepare", "duties-audit", "dual-qual", "intervention", "fit-diagnostic")
    for name in extension:
        parser.add_argument("--" + name, nargs=2, metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    if code["commit"] != args.expected_commit:
        raise ValueError("Unexpected WP12 adapter commit")
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    if not out.is_relative_to(root):
        raise ValueError("Output must remain inside the authorized runs root")
    def pinned(pair):
        return {"path": str(Path(pair[0]).resolve(strict=True)), "sha256": pair[1]} if pair else None
    pins = tuple(pinned(getattr(args, name.replace("-", "_"))) for name in extension)
    if any(pins) and not all(pins):
        raise ValueError("Complete native duties, qualification, intervention and diagnostics are required together")
    print(evidence.append(out, root, dev._pin(args.p4_scope), dev._pin(args.fit_template),
                          dev._pin(args.sparse), dev._pin(args.queue_receipt), code["commit"],
                          p4_review=pinned(args.p4_review), p4_release=pinned(args.p4_release),
                          comparison=pinned(args.comparison), native_duty=pins if any(pins) else None), flush=True)
    require_frozen_code(code)


if __name__ == "__main__":
    main()

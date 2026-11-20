"""Consume reviewed P4 queue and released-point development preflights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_p4_released_evidence as evidence
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import registered_development as dev


def short_science(argv):
    from evaluation.wp12_p4_short_review import require, run
    from research_runtime import write_json

    parser = argparse.ArgumentParser(description="Confirm P4 archived and migrated released-point outcomes")
    parser.add_argument("--short-science",action="store_true",required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    for name in ("migration","migration-approval","combined-acceptance"):
        parser.add_argument("--"+name,nargs=2,metavar=("PATH","SHA256"),required=name!="combined-acceptance")
    a = parser.parse_args(argv)
    code = code_snapshot()
    require_frozen_code(code)
    root = a.runs.resolve(strict=True)
    out = external_directory(a.out)
    require(sys.platform=="linux" and code["commit"]==a.expected_commit and root in out.parents
            and not out.exists(), "Use the expected clean checkout and a fresh authorized output")
    def pin(pair):
        return dict(zip(("path","sha256"),pair,strict=True)) if pair else None
    result = run(root,pin(a.migration),pin(a.migration_approval),pin(a.combined_acceptance))
    require_frozen_code(code)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,dict(**result,producer=code["commit"]))
    print(dev._pin(out),flush=True)


def main():
    if "--short-science" in sys.argv[1:]:
        return short_science(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for name in ("runs", "p4-scope", "fit-template", "sparse", "queue-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("p4-review", "p4-release", "comparison", "p4-migration", "paviau-preflight"):
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
                          comparison=pinned(args.comparison), native_duty=pins if any(pins) else None,
                          p4_migration=pinned(args.p4_migration), paviau=pinned(args.paviau_preflight)), flush=True)
    require_frozen_code(code)


if __name__ == "__main__":
    main()

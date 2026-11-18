"""Build the production WP12 table from pinned real WP10 and WP11 artifacts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_production_evidence import build, write_table
from evaluation import wp03_engineering_evidence as wp03
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
    if "wp03_light_engineering" in spec:
        if set(spec) != {"wp03_light_engineering"}:
            raise ValueError("Synthetic WP03 engineering and production evidence cannot be joined")
        inputs = spec["wp03_light_engineering"]
        report = wp03.build(inputs)
        paths = [Path(pin["path"]).resolve(), *(Path(inputs[k]["path"]) for k in ("worker","campaign"))]
        paths.extend(Path(run[k]["path"]) for run in report["runs"] for k in ("review","fixture","registration"))
        fixture_roots = {Path(run["fixture"]["path"]).parent for run in report["runs"]}
        if out.exists() or any(out == p or out in p.parents for p in paths) or any(
                out == root or root in out.parents for root in fixture_roots):
            raise ValueError("Fresh independent engineering output required")
        require_frozen_code(code)
        read(pin)
        wp03.write_table(out, {**report,"registration":pin,"adapter_commit":code["commit"]})
        return
    wp10_root = external_directory(spec["wp10"]["root"])
    protected = [wp10_root, Path(pin["path"]).resolve().parent,
                 *[Path(v["path"]).resolve().parent for group in ("wp11", "low_rank", "curriculum", "observation", "factorial", "real_factorial", "controlled_graph")
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

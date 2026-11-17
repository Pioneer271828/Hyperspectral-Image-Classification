"""Exercise the complete authenticated production comparison over synthetic banks."""
import argparse
from pathlib import Path

import torch

from research_runtime import code_snapshot
from scripts import run_native_released_comparison as cli
from tests.native_released_bank_fixture import fixture
from tests.test_native_released_production import release_for
from transfer import native_released_campaign as campaign, native_released_real as real, registered_development as dev
from transfer.native_scale_campaign import read


def pair(pin):
    return [pin["path"], pin["sha256"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--duties", action="store_true")
    args = parser.parse_args()
    root, code = args.out, code_snapshot()
    assert code["commit"] == args.expected_commit and not code["dirty"]
    root.mkdir()
    torch.set_num_threads(1)
    design = real.DUTY if args.duties else None
    pins, intake = fixture(root, code, design=design)
    manifest = dev._json_new(root / "fixture-preflights.json", {
        "schema": real.ENGINEERING, "status": "preflight_only", "scenes": pins,
        "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    cmd = ["--expected-commit", code["commit"], "--runs", str(root)]
    target = root / "matrix-execution"
    matrix = cli.main([*cmd, "prepare", "--preflight", *pair(manifest),
                       "--out", str(root / "registration"), "--execution", str(target),
                       *(["--duties"] if args.duties else [])], intake=intake)
    expected = read(matrix, root)
    count = 6 * len(real.comparisons(design))
    assert len(expected["cases"]) == count and expected["optimizer_updates"] == count * 12
    try:
        cli.main([*cmd, "run", "--matrix", *pair(matrix)], intake=intake)
    except ValueError as error:
        assert "Only prepare" in str(error) and not target.exists()
    else:
        raise AssertionError("Unreleased matrix entered execution")
    released = {}
    for i, case in enumerate(expected["cases"]):
        row = next(v for v in read(case["preflight"], root)["draft"]["cases"] if v["id"] == case["id"])
        released[case["id"]] = release_for(root, case["preflight"], row, Path(case["destination"]), code,
                                          "split" if i % 2 else "full")
    authorization = dev._json_new(root / "matrix-release.json", {
        "schema": campaign.RELEASE, "decision": "approved", "matrix": matrix, "execution": str(target),
        "synthetic": True, "optimizer_updates": count * 12, "cases": released, "query_authorized": False})
    run = [*cmd, "run", "--matrix", *pair(matrix), "--release", *pair(authorization)]
    first = cli.main(run, intake=intake)
    second = cli.main(run, intake=intake)
    assert first["new_updates"] == count * 12 and first["new_stages"] == count * 5 // 2
    assert second["new_updates"] == second["new_stages"] == 0
    reviewed = cli.main([*cmd, "audit", "--matrix", *pair(matrix), "--release", *pair(authorization),
                         "--out", str(root / "independent")], intake=intake)
    result = read(reviewed, root)
    assert result["status"] == "passed" and result["audit_optimizer_updates"] == 0 and len(result["cases"]) == count
    assert any(any(row["minority_points_per_class"]) for case in result["cases"]
               for row in case["training_exposure"])
    print(dev._json_new(root / "summary.json", {"first": first, "resumed": second, "audit": reviewed}), flush=True)


if __name__ == "__main__":
    main()

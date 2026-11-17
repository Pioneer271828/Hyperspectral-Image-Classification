"""Exercise the complete authenticated production comparison over synthetic banks."""
import argparse
from pathlib import Path

import torch

from research_runtime import code_snapshot
from scripts import run_native_released_comparison as cli
from tests.native_released_bank_fixture import fixture
from tests.test_native_released_production import release_for
from transfer import native_released_campaign as campaign, native_released_real as real, registered_development as dev
from transfer.native_scale_campaign import packet, read
from transfer.xuzhou_multiscale_audit import _exact


def pair(pin):
    return [pin["path"], pin["sha256"]]


def exercise(root, code, manifest, intake, design, *, mirror=False):
    paviau = design == real.PAVIA_DESIGN
    variant = "-mirror" if mirror else ""
    cmd = ["--expected-commit", code["commit"], "--runs", str(root)]
    target = root / ("matrix-execution" + variant)
    matrix = cli.main([*cmd, "prepare", "--preflight", *pair(manifest),
                       "--out", str(root / ("registration" + variant)), "--execution", str(target),
                       *(["--duties"] if design == real.DUTY else [])], intake=intake)
    expected = read(matrix, root)
    count = (3 if paviau else 6) * len(real.comparisons(design))
    budget = count * (4 if paviau else 12)
    assert len(expected["cases"]) == count and expected["optimizer_updates"] == budget
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
                                          "split" if (i + int(mirror)) % 2 else "full")
    authorization = dev._json_new(root / ("matrix-release" + variant + ".json"), {
        "schema": campaign.RELEASE, "decision": "approved", "matrix": matrix, "execution": str(target),
        "synthetic": True, "optimizer_updates": budget, "cases": released, "query_authorized": False})
    run = [*cmd, "run", "--matrix", *pair(matrix), "--release", *pair(authorization)]
    if paviau:
        try:
            cli.main(run)
        except ValueError as error:
            assert "Wrong matrix source" in str(error) and not target.exists()
        else:
            raise AssertionError("Fixture authorization entered real execution")
    first = cli.main(run, intake=intake)
    second = cli.main(run, intake=intake)
    assert first["new_updates"] == budget and first["new_stages"] == 2*count + (count+int(mirror))//2
    assert second["new_updates"] == second["new_stages"] == 0
    reviewed = cli.main([*cmd, "audit", "--matrix", *pair(matrix), "--release", *pair(authorization),
                         "--out", str(root / ("independent" + variant))], intake=intake)
    result = read(reviewed, root)
    assert result["status"] == "passed" and result["audit_optimizer_updates"] == 0 and len(result["cases"]) == count
    if not paviau:
        assert any(any(row["minority_points_per_class"]) for case in result["cases"]
                   for row in case["training_exposure"])
    return {"first": first, "resumed": second, "audit": reviewed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--duties", action="store_true")
    group.add_argument("--paviau", action="store_true")
    args = parser.parse_args()
    root, code = args.out, code_snapshot()
    assert code["commit"] == args.expected_commit and not code["dirty"]
    root.mkdir()
    torch.set_num_threads(1)
    design = real.PAVIA_DESIGN if args.paviau else real.DUTY if args.duties else None
    pins, intake = fixture(root, code, design=design)
    manifest = dev._json_new(root / "fixture-preflights.json", {
        "schema": real.ENGINEERING, "status": "preflight_only", "scenes": pins, "design": design,
        "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    result = exercise(root, code, manifest, intake, design)
    if args.paviau:
        mirror = exercise(root, code, manifest, intake, design, mirror=True)
        for lhs, rhs in zip(read(result["audit"], root)["cases"], read(mirror["audit"], root)["cases"], strict=True):
            assert lhs["case"] == rhs["case"] and lhs["selected_step"] == rhs["selected_step"]
            states = [packet(read(row["stages"][-1], root)["checkpoint"], root) for row in (lhs, rhs)]
            assert all(_exact(states[0][key], states[1][key]) for key in states[0] if key != "registration")
            rows = lhs["validation"]["rows"]
            assert len(rows) == 4 and sum(row["released"]["points"] == 0 for row in rows) == 2
            assert all(sum(row["scores"]["fused"]["count"] for row in rows if row["band"] == band) == 9
                       for band in {row["band"] for row in rows})
            assert all(row["condition_key"] is None and row["heads"] for row in rows)
        result.update(mirror=mirror, exact_full_split_cases=39, graph_scale=50)
    print(dev._json_new(root / "summary.json", result), flush=True)


if __name__ == "__main__":
    main()

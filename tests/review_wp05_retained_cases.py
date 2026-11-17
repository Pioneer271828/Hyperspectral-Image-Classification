"""Verify retained receipt integrity using the existing independent-queue CPU fixture."""

import argparse
from contextlib import contextmanager
from pathlib import Path

from research_runtime import code_snapshot, require_frozen_code
from scripts.prepare_wp05_resource_migration import verify_retained
from tests.test_p4_independent_queue import reject
from transfer import p4_independent_registration as scope, p4_runtime_device as device
from transfer.p4_resource_migration import seal


@contextmanager
def corrupted(path):
    data = path.read_bytes()
    try:
        path.write_bytes(data + b"\n")
        yield
    finally:
        path.write_bytes(data)


def run(fixture, out):
    out.mkdir()
    item = scope.pin(fixture / "matrix-resource/resource-scope.json")
    resource = scope.read(item)
    science = scope.read(resource["science"])
    assert resource["mode"] == scope.CPU and science["synthetic"]
    original = Path(resource["execution"])
    cases = {c["id"]: c for c in science["cases"]}
    kept = resource["slots"]["cpu-a"]["cases"]
    departed = resource["slots"]["cpu-b"]["cases"]
    review = seal(out / "fixture-partial-review.json", dict(
        resource=item, release=None, scientific_scope=resource["science"], status="partial",
        input_pins=science["protected_files"], committed_optimizer_updates=1200, audit_optimizer_updates=0,
        observed_cases=10, raw_gt_loaded=False, query_accessed=False,
        cases=[dict(case=cases[name], optimizer_updates=120, audit=scope.pin(original / "cases" / name / "audit.json"))
               for name in kept]))

    def current():
        return verify_retained(item, None, resource, science, review, "cpu-a", departed)

    with device.lease(original / ".coordinator.lock"):
        frozen = current()

        def revalidate():
            if current() != frozen:
                raise ValueError("Retained stage or artifact digest changed")

        folder = original / "cases" / kept[0]
        stage = original / "stages" / kept[0] / "00.receipt.json"
        negatives = {}
        for artifact, error in ((folder / "full/state.pt", "changed"),
                                (folder / "audit.json", "Pinned document changed"), (stage, "Retained stage or artifact digest changed")):
            before = scope.pin(artifact)
            with corrupted(artifact):
                try:
                    revalidate()
                except (ValueError, RuntimeError) as exc:
                    if error.lower() not in str(exc).lower():
                        raise
                    negatives[artifact.name] = str(exc)
                else:
                    raise AssertionError(f"Retained corruption went undetected: {artifact}")
            assert scope.pin(artifact) == before
            revalidate()
        overlap = reject(lambda: verify_retained(item, None, resource, science, review, "cpu-a",
                                                 [*departed[:-1], kept[0]]), "disjoint")
    return seal(out / "retained-rejection.json", dict(status="passed", retained_cases=len(frozen["cases"]),
                retained_pins=len(frozen["protected_files"]), new_updates=0, negative=negatives, overlap=overlap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    assert code["commit"] == args.expected_commit
    print(run(args.fixture, args.out))

"""Check the current strict CUDA stage under low-free/high-utilization telemetry."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

from scripts import run_wp04_factorial as wp04


def test_strict_synthetic_stage_records_telemetry_without_legacy_limits(monkeypatch, tmp_path):
    gpu = "GPU-11111111-1111-1111-1111-111111111111"
    row = {"physical_uuid": gpu, "free_mib": 128, "utilization": 100, "driver": "fixture"}
    case = {"seed": 64, "arm": "B0"}
    out = tmp_path / "64-B0"
    out.mkdir()
    args = SimpleNamespace(seed=64, arm="B0", phase="audit", gpu=gpu, matrix_fd=5, lease_fd=6)
    reg = {"scope": "synthetic_only", "cases": [case], "arithmetic": wp04.p4.CUDA_FP64_WEIGHT}
    seen = []

    @contextmanager
    def arithmetic(protocol, uuid, *, kernel_calls, enforce_resource_thresholds=False):
        assert not enforce_resource_thresholds
        assert protocol == reg["arithmetic"] and uuid == gpu
        yield {"protocol": protocol, "physical_uuid": uuid}

    def inherit(fd, path):
        assert (fd, path) == (5, tmp_path / ".matrix.lock")

    def lease(uuid, *, inherited_fd):
        assert (uuid, inherited_fd) == (gpu, 6)
        return nullcontext()

    def inventory(uuid):
        assert uuid == gpu
        seen.append(row)
        return row

    monkeypatch.setattr(wp04.p4, "check_inherited_lease", inherit)
    monkeypatch.setattr(wp04.p4, "arithmetic", arithmetic)
    monkeypatch.setattr(wp04.p4, "snapshot", inventory)
    monkeypatch.setattr(wp04, "physical_lease", lease)
    monkeypatch.setattr(wp04, "attest", lambda uuid: {"uuid": uuid})
    monkeypatch.setattr(wp04, "audit_case", lambda *args: {"verified": True})
    wp04.stage(args, {}, reg, out)
    assert seen == [row]
    assert wp04.read(out / "audit.json") == {
        "verified": True, "arithmetic": {"protocol": reg["arithmetic"], "physical_uuid": gpu}, "kernel_calls": {},
    }

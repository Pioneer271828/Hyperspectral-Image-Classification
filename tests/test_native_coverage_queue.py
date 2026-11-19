import numpy as np
import pytest
import torch

from evaluation.wp12_native_coverage import readout
from scripts.run_frozen_band_region import _read, pin
from transfer import frozen_mechanism_graph as graphs
from transfer import native_coverage_queue as queue, native_scale_training as native
from transfer.g3_training import state_hash
from transfer.native_coverage_replay import archived_receipts
from transfer.spatial_coverage_protocol import bank_view
from transfer.spatial_coverage_training import load


@pytest.fixture
def registered(tmp_path):
    torch.set_num_threads(1)
    code = {"commit": "unit-coverage"}
    queue.prepare(tmp_path, code)
    return tmp_path, code, queue.load_queue(tmp_path, code)


def test_native_queue_enforces_bank_role_budget_and_graph_cache(registered, tmp_path):
    root, code, plan = registered
    assert len(plan["cases"]) == 120 and len(plan["banks"]) == 24
    case = "whu-honghu-s512-k5-a-none"
    provider = queue.CoverageFixture(root, plan, case)
    producer = code["commit"]
    spec = provider.contract(22, plan["cases"][case]["model_seed"], "off", 1, producer)
    event = spec["training"][0]
    with pytest.raises(ValueError):
        provider.materialize("devval", event, producer)
    with pytest.raises(ValueError):
        provider.materialize("query", event, producer)
    with pytest.raises(ValueError, match="registered native exposure"):
        runner = native.Run(spec, producer, released=provider)
        runner.start()
        runner.update(dict(event, step=13))
    for other in ("whu-honghu-s512-k1-a-none", "whu-honghu-s2048-k5-a-none", "whu-honghu-s512-k5-b-none"):
        different = queue.CoverageFixture(root, plan, other)
        assert different.bank.pin["sha256"] != provider.bank.pin["sha256"]
        with pytest.raises(ValueError, match="registration|coverage case"):
            different_seed = plan["cases"][other]["model_seed"]
            native.Run(different.contract(22,different_seed,"off",1,producer), producer, released=provider)
    val = provider.bank.role_rows("devval")[0]
    full, _, full_targets = bank_view(provider.bank, val, producer, "full")
    gap, _, _ = bank_view(provider.bank, val, producer, "central_gap")
    assert len(gap.view.bands.band_ids) < len(full.view.bands.band_ids)
    assert 115 not in full.view.bands.band_ids
    assert full.view.bands.wavelength_quality == "approximate"
    assert provider.bank.release.coverage()["devval"]["released_per_class"][20] == 0
    model = provider.baseline(spec["seed"]).eval()
    state = state_hash(model)
    args = (model, "whu-honghu")
    opts = dict(role="devval", model_state=state, cache_only=True)
    args2 = (full_targets.cells.tolist(), 22, 100)
    fresh = graphs.execute(*args, full.view, *args2, **opts)
    cached = graphs.execute(*args, full.view, *args2, **opts, frozen=fresh.graph)
    graphs.assert_same_view(fresh, cached)
    with pytest.raises(ValueError, match="retained observation"):
        graphs.execute(*args, gap.view, *args2, **opts, frozen=fresh.graph)
    with pytest.raises(ValueError, match="Foreign"):
        graphs.execute(*args, full.view, *args2, **{**opts,"role":"devtrain"}, frozen=fresh.graph)
    poisoned = dict(plan)
    poisoned["banks"] = dict(plan["banks"], **{"whu-honghu-s512-k5-a":plan["banks"]["whu-honghu-s512-k5-b"]})
    with pytest.raises(ValueError):
        queue.CoverageFixture(root, poisoned, case)
    fake = tmp_path/"xuzhou.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        archived_receipts(root, [(fake, pin(fake)["sha256"])])


def test_native_phase_resume_reuses_completed_and_rejects_replaced_bytes(registered, monkeypatch):
    root, code, _ = registered
    monkeypatch.setattr(queue, "require_frozen_code", lambda _: None)
    case = "paviaU-s512-k1-a-continuous"
    result = queue.run_case(root, code, case)
    receipt = _read(result["receipt"])
    place = root/"cases"/"cpu"/case
    assert receipt["optimizer_updates"] == 24
    assert queue.run_case(root, code, case)["new_updates"] == 0
    # Simulate interruption after a valid stage was written and before its receipt was published.
    (place/"receipt.json").unlink()
    (place/"completed.json").unlink()
    recovered = queue.run_case(root, code, case)
    assert list(place.glob("completed.orphan-*.pt")) and _read(recovered["receipt"])["stages"]["completed"]["updates"] == 0
    assert native._exact(load(place/"completed.pt"), load(next(place.glob("completed.orphan-*.pt"))))
    packet = place/"prefix.pt"
    packet.chmod(0o644)
    with packet.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="receipt or checkpoint"):
        queue.run_case(root, code, case)


def test_independent_class_audit_preserves_absent_class_and_rejects_nonfinite():
    confusion, nll = readout([([1], [[0.8, 0.2]])], 2)
    assert confusion.tolist() == [[1, 0], [0, 0]] and nll > 0
    with pytest.raises(ValueError, match="Overflow"):
        readout([([1], [[np.nan, 0.2]])], 2)

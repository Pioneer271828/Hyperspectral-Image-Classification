from dataclasses import replace

import numpy as np
import pytest
import torch

from evaluation.frozen_mechanism import apply_temperature, calibration, fit_algebraic_temperature, point_scores
from evaluation.wp12_frozen_mechanisms import _digest_trace, check_aliases, ingest
from loadData.observation_sampling import BandMetadata, ObservationView, _apply
from loadData.spatial_protocol import PartitionObservation
from scripts.run_frozen_band_region import _write
from scripts.run_frozen_mechanisms import PREVIOUS_SCOPE, PREVIOUS_SUMMARY
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.frozen_mechanism_graph import assert_same_view, execute
from transfer.frozen_mechanism_plan import POLICY, random_control, windows
from transfer.frozen_mechanism_runner import _window_outputs
from transfer.frozen_band_region import region_boxes
from transfer.g3_training import state_hash


@pytest.mark.parametrize("bands", (103, 436))
def test_registered_windows_include_mirrors_nonoverlap_and_fixed_counts(bands):
    plan = windows(bands)
    assert plan == windows(bands)
    for width in (bands//4, bands//2):
        selected = [row for row in plan if row["width"] == width]
        ends = [row["first"] for row in selected if "end" in row["families"]]
        assert ends == [0, bands - width]
        assert {row["seed"] for row in selected if row["seed"] is not None} == {7719, 7720}
        assert all(len(row["deleted"]) == width and len(row["retained"]) == bands-width for row in selected)
    assert not any(row["width"] == bands//2 and row["first"] == (bands-bands//2)//2 for row in plan)
    starts = [x["first"] for x in plan if x["width"] == bands//4 and "disjoint" in x["families"]]
    assert all(b-a >= bands//4 for a,b in zip(starts, starts[1:]))


@pytest.mark.parametrize("mode", ("off", "mean"))
def test_graph_exchange_through_original_heads_and_negative(mode):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    model = ConditionalMultiscaleVCASAHGCN({"xuzhou": 3}, graph_multiscale=mode, latent_channels=8,
                                           adapter_embed_dim=8, norm_type="group_norm", graph_mode="learned").eval()
    model.backbone.adapter.cache_attention = False
    rng = np.random.default_rng(88)
    values = rng.random((9, 9, 16), dtype=np.float32)
    yy, xx = np.mgrid[:9,:9]
    coords = np.stack((yy/8, xx/8), -1).astype(np.float32)
    axis = BandMetadata("test", tuple(range(16)), tuple(float(400+20*i) for i in range(16)),
                        "nm", "approximate", (True,)*16)
    observation = PartitionObservation("val", values, coords, yy*9+xx, "plan")
    view = ObservationView(observation, axis)
    identity = state_hash(model)
    opts = dict(role="devval", model_state=identity, registered_units=False)
    full = execute(model, "xuzhou", view, [0, 20, 45], 3, 100, **opts)
    replay = execute(model, "xuzhou", view, [0, 20, 45], 3, 100, frozen=full.graph, **opts)
    assert_same_view(full, replay)
    sub = _apply(observation, axis, tuple(range(4,16)))
    boxes = region_boxes((9,9), [0,20,45], [2,1,3], [1])
    boxes["random"] = random_control((9,9), boxes, 2)
    outputs, calls = _window_outputs(model, "xuzhou", sub, boxes, [0,20,45], 3, {"model": identity}, full)
    target = outputs["target_clean_graph"]
    assert target.trace["graph"] == outputs["clean"].trace["graph"]
    assert target.trace["feature_input"] != outputs["clean"].trace["feature_input"]
    assert calls in (9,10)
    assert (target.trace["graph"]["wide_a"] is not None) == (mode == "mean")
    with pytest.raises(ValueError, match="Foreign"):
        execute(model, "xuzhou", view, [0,20,45], 3, 100, frozen=replace(full.graph, geometry="foreign"), **opts)
    with pytest.raises(ValueError, match="projection identity"):
        wrong = {**target.trace, "source_graph_features": "mismatch"}
        _digest_trace(target.probabilities, wrong, outputs["clean"].trace, outputs["target"].trace)
    assert state_hash(model) == identity


def test_calibration_is_order_preserving_and_reports_empty():
    values = np.array([[.75, .24, .01], [.20, .71, .09], [.20, .35, .45]], dtype=np.float64)
    labels = [2, 2, 3]
    temp = fit_algebraic_temperature(values, labels)
    softened = apply_temperature(values, temp["temperature"])
    a, b = (point_scores(p, labels, {2: (1,), 3: (2,)}) for p in (values, softened))
    assert [x["rank"] for x in a] == [x["rank"] for x in b]
    assert [x["predicted"] for x in a] == [x["predicted"] for x in b]
    assert a[0]["foils"]["1"]["margin"] < 0
    assert calibration(softened[:0], [])["ece10"] is None


def test_wp12_refuses_missing_and_mixed_receipts(tmp_path):
    kwargs = dict(schema=POLICY, query_accessed=False, raw_gt_loaded=False, optimizer_updates=0)
    scope = _write(tmp_path/"scope.json", {**kwargs,"code":{"commit":"a"}, "model_selection":False,
                                         "parents":{"scope":{"sha256":PREVIOUS_SCOPE},
                                                    "summary":{"sha256":PREVIOUS_SUMMARY}},
                                         "cases":{"xuzhou":["off"],"paviaU":[],"honghu24":[],"honghu60":[]}})
    bad_scope = _write(tmp_path/"foreign.json", dict(**kwargs,status="complete",scope=scope["sha256"],commit="b"))
    with pytest.raises(ValueError, match="altered or training"):
        ingest(tmp_path, scope, bad_scope, tmp_path, tmp_path/"bad", "x")
    incomplete = _write(tmp_path/"partial.json", dict(**kwargs,status="complete",scope=scope["sha256"],commit="a",
                                                     cases=1,artifacts={}))
    with pytest.raises(ValueError, match="complete registered"):
        ingest(tmp_path, scope, incomplete, tmp_path, tmp_path/"partial", "x")
    for vector in (["plain-coordinate"], ["f"*32, "f"*32]):
        with pytest.raises(ValueError, match="point identity"):
            check_aliases(vector, [1] * len(vector), len(vector), 3)

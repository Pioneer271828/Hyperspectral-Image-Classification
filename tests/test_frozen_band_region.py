from dataclasses import replace

import numpy as np
import pytest

from evaluation.frozen_band_region import paired, summarize
from loadData.observation_sampling import BandMetadata, sample
from loadData.spatial_protocol import PartitionObservation
from scripts.ingest_wp12_frozen_band_region import _audit_pair
from transfer.frozen_band_region import band_specs, region_boxes, same_region_groups, _spatial_view


def test_paired_tracks_error_destinations_and_empty_classes():
    old = [1, 2, 2, 3, 2]
    new = [3, 1, 3, 2, 2]
    values = paired(new, old, [1, 1, 1, 2, 2], 3)
    first, second, empty = values["classes"]
    assert (first["denominator"], first["correct_to_error"], first["error_to_correct"]) == (3, 1, 1)
    assert first["lost_to"] == {3: 1} and first["gained_from"] == {2: 1}
    assert second["transitions"] == [[2, 2, 1], [3, 2, 1]]
    assert empty["denominator"] == 0 and empty["target_recall"] is None
    _audit_pair(values)
    values["classes"][0]["lost_to"] = {2: 1}
    with pytest.raises(ValueError, match="reconstruction"):
        _audit_pair(values)


@pytest.mark.parametrize("bands", [103, 270, 436])
def test_equal_count_band_recipes_and_spatial_control(bands):
    axis = BandMetadata("test", tuple(range(bands)), tuple(400. + i for i in range(bands)),
                        "nm", "approximate", (True,) * bands)
    size = 12
    original = PartitionObservation("train", np.ones((size, size, bands), dtype=np.float32),
                                    np.zeros((size, size, 2), dtype=np.float32),
                                    np.arange(size ** 2).reshape(size, size), "test")
    recipe = band_specs(axis)
    retained, record = sample(original, axis, recipe["contiguous"])
    _, scattered = sample(original, axis, recipe["scattered"])
    deleted = sorted(set(range(bands)) - set(record["selected_positions"]))
    assert deleted == list(range(deleted[0], deleted[0] + bands // 2))
    assert len(record["selected_positions"]) == len(scattered["selected_positions"])
    changed = original.spectral.copy()
    changed[..., deleted] = -102.
    result, _ = sample(replace(original, spectral=changed), axis, recipe["contiguous"])
    assert np.array_equal(retained.observation.spectral, result.observation.spectral)
    boxes = region_boxes((size, size), [1, 70], [5, 6], [6])
    target, control = (_spatial_view(original, boxes[name]).spectral for name in ("target", "matched"))
    assert (target == 0).sum() == (control == 0).sum() == 25 * bands
    assert not ((target == 0) & (control == 0)).any()
    assert target.reshape(-1, bands)[70].sum() == 0
    groups = same_region_groups([1, 70, 143], boxes["target"], (size, size),
                                np.array([[1, 70, 143], [0, 0, 1]]))
    assert groups["outside_sharing_native_q"].tolist() == [True, False, False]
    assert groups["outside_other_native_q"].tolist() == [False, False, True]


def test_model_seed_sd_and_duplicate_rejection():
    values = paired([2, 2], [1, 2], [1, 2], 2)
    case = dict(scene="paviaU", cohort="factor1/off", population="fixed",
                model_seed=64, paired={"change": {head: values for head in ("fused", "graph", "cnn")}})
    assert summarize([case])[0]["metrics"]["delta_oa"]["sample_sd"] is None
    assert summarize([case, {**case, "model_seed": 512}])[0]["metrics"]["delta_oa"]["sample_sd"] == 0
    with pytest.raises(ValueError, match="duplicated seeds"):
        summarize([case, case])

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from evaluation.frozen_stage_archives import read_stage_probabilities
from evaluation.frozen_training_stages import HEADS, paired, pinned, scores
from loadData.observation_sampling import BandMetadata
from transfer.spatial_coverage_protocol import retained_positions


def _points(ids, truths, vectors):
    return [{"id": [3, index], "truth": label, "probabilities": dict.fromkeys(HEADS, p)}
            for index, label, p in zip(ids, truths, vectors, strict=True)]


def test_equal_deleted_physical_bands_preserve_historical_odd_view():
    bands = BandMetadata("synthetic", tuple(range(7)),
                         tuple(range(401, 408)), "nm", "verified", (True,)*7)
    contiguous = retained_positions(bands, "central_gap")
    dispersed = retained_positions(bands, "matched_dispersed")
    assert contiguous == (0, 1, 5, 6)
    assert dispersed == (0, 2, 4, 6)
    assert len(retained_positions(bands, "dispersed")) == 3


def test_paired_heads_reject_wrong_region_and_keep_class_totals():
    full = _points([20, 21, 22], [1, 1, 2], [[.9, .1], [.6, .4], [.3, .7]])
    altered = _points([20, 21, 22], [1, 1, 2], [[.3, .7], [.6, .4], [.8, .2]])
    assert scores(altered, 2)["fused"]["confusion"] == [[1, 1], [1, 0]]
    expected = paired(full, altered, 2, {1: 2})["cnn"]
    assert expected["transitions"] == [[1, 1, 1, 1], [1, 1, 2, 1], [2, 2, 1, 1]]
    np.testing.assert_allclose(expected["locked_probability_margins"]["1"]["before"], [.8, .2])
    mismatch = deepcopy(altered)
    mismatch[1]["id"] = [4, 21]
    with pytest.raises(ValueError, match="roster"):
        paired(full, mismatch, 2, {})
    mismatch = deepcopy(altered)
    mismatch[0]["probabilities"]["cnn"] = [-.1, 1.1]
    with pytest.raises(ValueError, match="probabilities"):
        scores(mismatch, 2)


def test_only_pinned_run_files_and_development_phases_are_admitted(tmp_path):
    from evaluation.frozen_training_stages import file_pin

    root = tmp_path/"runs"
    root.mkdir()
    checkpoint = root/"weights.pt"
    foreign = root/"foreign.pt"
    rows = dict.fromkeys(HEADS, torch.tensor([[.4, .6]]))
    torch.save({"xuzhou": {"train": rows, "val": rows}}, checkpoint)
    torch.save({"xuzhou": {"train": rows, "val": rows, "query": rows}}, foreign)
    pin = file_pin(checkpoint)
    assert pinned(root, pin) == checkpoint and set(read_stage_probabilities(checkpoint, "xuzhou")) == {"train", "val"}
    with pytest.raises(ValueError, match="hash"):
        pinned(root, {**pin, "path": str(foreign)})
    with pytest.raises(ValueError, match="train/val"):
        read_stage_probabilities(foreign, "xuzhou")
    unrelated = tmp_path/"weights.json"
    unrelated.write_text(json.dumps({"scope": "foreign"}))
    with pytest.raises(ValueError, match="runs root"):
        pinned(root, file_pin(unrelated))

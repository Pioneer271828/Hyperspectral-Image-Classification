import numpy as np
import pytest
import torch

from evaluation.frozen_band_region import HEADS
from evaluation.wp12_transfer_readout import guard
from evaluation.frozen_band_region import paired
from scripts.run_transfer_readout import stratified
from transfer.support_readout import SupportReadout, apply_distance
from transfer.support_readout_fixture import exercise


def test_planted_support_gain_and_shift_with_exact_restore():
    torch.set_num_threads(1)
    evidence = exercise("fixture")
    assert evidence["optimizer_updates"] == evidence["learned_parameters"] == 0
    assert len(evidence["cases"]) == 6
    assert all(row["candidate_correct"] == 30 and row["native_correct"] == 20
               for row in evidence["cases"].values())
    assert all(row["shifted_candidate_correct"] < row["candidate_correct"] for row in evidence["cases"].values())


def test_partial_support_preserves_absent_raw_weight_and_rejects_bad_packet():
    baseline = np.array([[.5, .3, .2]])
    d = np.array([[0., 2., 100.]])
    correction = apply_distance(baseline, d, [1, 1, 0], 2.)
    assert correction[0,0] > .5
    np.testing.assert_array_equal(apply_distance(baseline, d, [1, 0, 0], 0.), baseline)
    binding = dict(scene="scene", checkpoint="checkpoint", observation="view", support="support")
    features = {h: [[0., 1.], [1., 0.]] for h in HEADS}
    scores = {h: [[1., 0.], [0., 1.]] for h in HEADS}
    readout = SupportReadout.fit(binding, features, scores, [1, 2], 2)
    with pytest.raises(ValueError, match="features or distance scale"):
        bad = readout.state()
        bad["scales"]["cnn"] += 1
        SupportReadout.from_state(bad, binding)


def test_support_replicates_are_disjoint_and_class_guard_uses_correct_to_error():
    labels = [1] * 5 + [2] * 5
    sample = stratified(list(map(str, range(10))), labels)
    assert len(sample["all"]) == 10 and len(sample["k2a"]) == len(sample["k2b"]) == 4
    assert not set(sample["k2a"]) & set(sample["k2b"])
    values = paired([2, 1, 2, 2], [1, 2, 2, 2], [1, 1, 2, 2], 2)
    assert guard(values, [1], {"aggregate_min_delta": -.005, "focus_min_delta_recall": -.02}, True) == ["new_error_1"]

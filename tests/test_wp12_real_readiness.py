import numpy as np
import pytest

from evaluation.wp06_capacity_released import (
    WIDTH, dual_predictions, primal_predictions, project, sample_design,
)
from evaluation.wp12_real_readiness import transition


def test_same_point_transition_keeps_wrong_to_wrong_and_rejects_roster_mismatch():
    before = {1: (5, 5), 2: (5, 3), 3: (5, 1), 4: (6, 6)}
    after = {3: (5, 1), 1: (5, 3), 2: (5, 5), 4: (6, 6)}
    assert transition(before, after, 5) == dict(
        n=3, before=1, after=1, baseline_wrong=2, gains=1, losses=1,
        destinations=[[1, 1, 1], [3, 5, 1], [5, 3, 1]],
    )
    with pytest.raises(ValueError, match="class identity"):
        transition(before, {**after, 3: (6, 1)}, 5)
    with pytest.raises(ValueError, match="omit points"):
        transition(before, {i: v for i, v in after.items() if i != 2}, 5)


def test_nested_support_and_parent_control_use_equal_per_class_counts():
    labels = np.repeat([1, 2, 3], 5)
    parent = np.array([1, 2, 2, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2])
    selections, draws, controls = sample_design(labels, parent, (1, 2))
    assert controls["1"]["missing_classes"] == [2]
    assert controls["2"]["status"] == "paired"
    for record in (controls["2"], controls["3"]):
        a, b = (selections[record[key]] for key in ("spatial", "count_control"))
        assert np.bincount(labels[list(a)]).tolist() == np.bincount(labels[list(b)]).tolist()
    assert all(value == "all" for key, value in draws.items() if key.startswith("k5"))
    assert all(len(set(labels[list(selections[value])])) == 3 for key, value in draws.items()
               if key.startswith("k1"))


def test_equal_width_dual_and_primal_match_with_constant_training_columns():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(21, 79))
    data[:15, :4] = 2
    x, columns = project(data)
    assert x.shape == (21, WIDTH)
    assert columns[0] == 0 and columns[-1] == 78 and len(set(columns)) == WIDTH
    train, val, y = {"layer": x[:15]}, {"layer": x[15:]}, np.repeat([1, 2, 3], 5)
    np.testing.assert_array_equal(dual_predictions(train, val, y)["layer"],
                                  primal_predictions(train, val, y)["layer"])

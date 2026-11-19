import numpy as np
import pytest
import torch

from evaluation.wp12_coverage_readout import softmax
from transfer.frozen_branch_readout import centroid_distances, centroid_inputs
from transfer.spectral_spacing import normalize_attention


def test_retained_quadrature_ignores_outside_columns_and_preserves_empty_window():
    positions = torch.tensor([.12, .13, .16, 0.])
    native = torch.tensor([[.25, .25, .25, .25], [.31, .22, .31, .16]], requires_grad=True)
    anchors = torch.tensor([.14, .83])
    valid = torch.tensor([1., 0.])
    result, support = normalize_attention(native, valid, anchors, positions, .16,
                                          torch.tensor([True, True, True, False]), 1.)
    assert torch.equal(support, valid)
    assert torch.equal(result[1], native[1])
    assert result[0, 3].item() == 0
    torch.testing.assert_close(result[0].sum(), torch.tensor(1.))
    assert result[0, 0] < result[0, 1] < result[0, 2]
    result[0, 0].backward()
    assert native.grad[0].abs().sum() > 0
    assert native.grad[0, 3] == 0


def test_prototype_has_fixed_capacity_and_no_validation_fit():
    train = np.array([[0., 0.], [2., 0.], [8., 0.], [10., 0.]])
    centroids, counts = centroid_inputs(train, [1, 1, 2, 2], 2)
    assert counts == [2, 2]
    val = np.array([[.003, .7], [9.3, -2.]])
    probabilities = softmax(-centroid_distances(centroids, val))
    np.testing.assert_array_equal(probabilities.argmax(1), [0, 1])
    np.testing.assert_array_equal(centroids, [[1., 0.], [9., 0.]])
    with pytest.raises(ValueError, match="released support"):
        centroid_inputs(train, [1, 1, 1, 1], 2)

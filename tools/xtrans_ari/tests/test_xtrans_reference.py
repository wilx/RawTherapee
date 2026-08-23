import numpy as np
import pytest

from tools.xtrans_ari.analysis import (
    label_spatial_metrics,
    oracle_reconstruction,
    ranking_metrics,
)
from tools.xtrans_ari.xtrans_reference import (
    _interpolate_observed,
    green_candidate_bank,
)
from tools.xtrans_mlri_internal.dataset import mosaic, origin_cells


def test_directional_interpolation_preserves_observations():
    values = np.zeros((5, 7), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    mask[:, (0, 3, 6)] = True
    values[mask] = np.arange(np.count_nonzero(mask))
    result = _interpolate_observed(values, mask, axis=1)
    assert np.array_equal(result[mask], values[mask])
    assert np.isfinite(result).all()


def test_constant_contract_all_18_phase_cells():
    truth = np.full((3, 36, 36), 0.37, dtype=np.float64)
    seen = set()
    for origin_x, origin_y in origin_cells():
        scalar, cfa = mosaic(truth, origin_x, origin_y)
        cell = bytes(cfa[:6, :6].ravel())
        assert cell not in seen
        seen.add(cell)
        first = green_candidate_bank(scalar, cfa)
        second = green_candidate_bank(scalar, cfa)
        assert first.candidates.shape == (2, 2, 11, 36, 36)
        assert np.array_equal(first.candidates, second.candidates)
        assert np.array_equal(first.criteria, second.criteria)
        assert np.max(np.abs(first.adaptive - truth[1])) < 1e-8
        assert np.array_equal(first.adaptive[cfa == 1], scalar[cfa == 1])
    assert len(seen) == 18


def test_oracle_and_ranking_contracts():
    y, x = np.mgrid[:48, :48]
    truth = np.stack((0.1 + 0.7 * x / 47, 0.2 + 0.6 * y / 47, 0.8 - 0.5 * x / 47))
    scalar, cfa = mosaic(truth, 0, 0)
    bank = green_candidate_bank(scalar, cfa)
    candidates = bank.candidates.reshape(44, 48, 48)
    criteria = bank.criteria.reshape(44, 48, 48)
    output, labels = oracle_reconstruction(truth[1], candidates, cfa, 7)
    assert output.shape == labels.shape == cfa.shape
    assert labels.min() >= 0 and labels.max() < 44
    metrics = ranking_metrics(
        truth[1], candidates, criteria, cfa, maximum_pixels=64
    )
    assert metrics["pixel_count"] == 64
    assert 0.0 <= metrics["pairwise_auc"] <= 1.0
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0


def test_spatial_label_metrics_measure_coherence_and_scale_agreement():
    cfa = np.zeros((40, 40), dtype=np.uint8)
    pixel = np.indices(cfa.shape).sum(axis=0) % 2
    coherent = np.zeros_like(pixel)
    fragmented = label_spatial_metrics(pixel, cfa, reference=pixel)
    smooth = label_spatial_metrics(coherent, cfa, reference=pixel)
    assert fragmented["boundary_density"] > smooth["boundary_density"]
    assert fragmented["agreement_with_pixel_oracle"] == 1.0
    assert smooth["agreement_with_pixel_oracle"] == 0.5
    assert fragmented["component_count"] > smooth["component_count"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(bad):
    truth = np.full((3, 36, 36), 0.4)
    scalar, cfa = mosaic(truth, 0, 0)
    scalar[12, 12] = bad
    with pytest.raises(ValueError):
        green_candidate_bank(scalar, cfa)

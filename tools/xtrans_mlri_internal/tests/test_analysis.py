import numpy as np

from tools.xtrans_mlri_internal.analysis import (
    alpha_oracle,
    classification_metrics,
    local_mad,
    operating_threshold,
    roc_auc,
    scalar_metrics,
)
from tools.xtrans_mlri_internal.dataset import MLRI_CFA, cfa_for_origin, origin_cells


def test_origin_cells_are_the_18_unique_internal_phase_representations():
    origins = origin_cells()
    assert len(origins) == 18
    cells = {cfa_for_origin(6, 6, x, y).tobytes() for x, y in origins}
    assert len(cells) == 18
    assert np.array_equal(cfa_for_origin(6, 6, 0, 0), MLRI_CFA)


def test_alpha_oracle_bounds_and_exact_interior_solution():
    base = np.array([0.2, 0.2, 0.2, 0.2])
    correction = np.array([0.4, 0.4, -0.4, 0.0])
    truth = np.array([0.2, 0.4, 0.9, 0.7])
    alpha = alpha_oracle(base, correction, truth)
    np.testing.assert_allclose(alpha, [0.0, 0.5, 0.0, 0.0])


def test_local_mad_detects_an_impulse():
    values = np.zeros((9, 9))
    values[4, 4] = 1.0
    mad = local_mad(values, 5)
    assert mad[4, 4] == 0.0
    assert np.isfinite(mad).all()


def test_roc_and_operating_threshold_are_deterministic():
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert roc_auc(scores, labels) == 1.0
    threshold = operating_threshold(scores, labels, 1.0)
    assert threshold == 0.7
    metrics = classification_metrics(scores, labels, threshold)
    assert metrics["severe_recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_scalar_metrics_reports_expected_rms():
    metrics = scalar_metrics(np.array([-1.0, 1.0]))
    assert metrics["rms"] == 1.0
    assert metrics["maximum_abs"] == 1.0

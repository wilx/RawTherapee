from __future__ import annotations

import numpy as np

from tools.xtrans_lmmse_mixture.training import (
    _PhaseAccumulator,
    definition_from_quantiles,
)


def test_phase_accumulator_matches_direct_statistics() -> None:
    rng = np.random.default_rng(0x4D495833)
    x = rng.normal(size=(80, 7))
    target = rng.normal(size=(80, 3))
    accumulator = _PhaseAccumulator(7)
    accumulator.add(x[:31], target[:31])
    accumulator.add(x[31:], target[31:])
    row = accumulator.statistics()
    np.testing.assert_allclose(row.x_mean, np.mean(x, axis=0), atol=1e-14)
    np.testing.assert_allclose(row.target_mean, np.mean(target, axis=0), atol=1e-14)
    np.testing.assert_allclose(row.covariance, np.cov(x, rowvar=False), atol=1e-13)
    expected_cross = (target - np.mean(target, axis=0)).T @ (x - np.mean(x, axis=0)) / 79
    np.testing.assert_allclose(row.cross_covariance, expected_cross, atol=1e-13)


def test_definition_thresholds_come_from_training_quantiles() -> None:
    survey = {
        "raw_variance": np.linspace(-2.0, 2.0, 101),
        "highpass_variance": np.linspace(-3.0, 1.0, 101),
        "chroma_variation": np.linspace(-4.0, 0.0, 101),
        "anisotropy": np.linspace(0.0, 1.0, 101),
    }
    definition = definition_from_quantiles(
        "test", "set1", "raw_variance", survey, 0.4, 0.6
    )
    assert definition.activity_threshold == np.quantile(survey["raw_variance"], 0.4)
    active = survey["raw_variance"] >= definition.activity_threshold
    assert definition.anisotropy_threshold == np.quantile(survey["anisotropy"][active], 0.6)

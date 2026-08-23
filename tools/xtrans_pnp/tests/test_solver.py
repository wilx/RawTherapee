from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_pnp.operator import xtrans_operator
from tools.xtrans_pnp.solver import pnp_admm, pnp_pgm


def _identity(value: np.ndarray, sigma: float) -> np.ndarray:
    assert sigma >= 0.0
    return value.copy()


def test_exact_admm_identity_is_projection_and_records_zero_cfa_error() -> None:
    operator = xtrans_operator(13, 14, 1, 5)
    rng = np.random.default_rng(3)
    truth = rng.random((3, 13, 14))
    measured = operator.forward(truth)
    initial = rng.random(truth.shape)
    result = pnp_admm(
        operator, measured, initial, _identity, 0.01, 1.0, 3,
        exact_projection=True, capture_iterations=(1, 3),
    )
    expected = operator.project(initial, measured)
    assert np.array_equal(result.image, expected)
    assert np.array_equal(result.snapshots[1], expected)
    assert all(row.cfa_rms == 0.0 and row.cfa_maximum == 0.0 for row in result.records)


def test_soft_admm_identity_converges_measured_components() -> None:
    operator = xtrans_operator(8, 9)
    truth = np.ones((3, 8, 9), dtype=np.float64)
    measured = operator.forward(truth)
    initial = np.zeros_like(truth)
    result = pnp_admm(
        operator, measured, initial, _identity, 0.0, 1.0, 4,
        exact_projection=False, capture_iterations=(1, 4),
    )
    assert result.records[-1].cfa_rms < result.records[0].cfa_rms
    assert np.all(result.image[operator.mask == 0] == 0.0)


def test_continuation_schedule_reaches_denoiser() -> None:
    seen = []

    def denoiser(value: np.ndarray, sigma: float) -> np.ndarray:
        seen.append(sigma)
        return value

    operator = xtrans_operator(6, 6)
    initial = np.zeros((3, 6, 6))
    pnp_admm(
        operator, np.zeros((6, 6)), initial, denoiser, 0.04, 1.0, 4,
        exact_projection=True, continuation=0.5,
    )
    assert seen == pytest.approx([0.04, 0.02, 0.01, 0.005])


def test_pgm_exact_projection_and_bad_output() -> None:
    operator = xtrans_operator(7, 8)
    measured = np.arange(56, dtype=np.float64).reshape(7, 8) / 56.0
    initial = np.zeros((3, 7, 8))
    result = pnp_pgm(
        operator, measured, initial, _identity, 0.01, 1.0, 2,
        exact_projection=True, capture_iterations=(2,),
    )
    assert np.array_equal(operator.forward(result.image), measured)

    def bad(value: np.ndarray, sigma: float) -> np.ndarray:
        return np.full(value.shape, np.nan)

    with pytest.raises(ValueError, match="denoiser returned invalid"):
        pnp_pgm(operator, measured, initial, bad, 0.01, 1.0, 1, exact_projection=False)

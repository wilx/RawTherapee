from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_mlri_internal.dataset import origin_cells
from tools.xtrans_pnp.operator import CFAOperator, bayer_bggr_operator, xtrans_operator


@pytest.mark.parametrize("origin", origin_cells())
def test_xtrans_forward_adjoint_identity(origin: tuple[int, int]) -> None:
    operator = xtrans_operator(17, 19, *origin)
    rng = np.random.default_rng(0x504E50 + origin[0] * 7 + origin[1])
    rgb = rng.standard_normal((3, 17, 19))
    scalar = rng.standard_normal((17, 19))
    left = float(np.vdot(operator.forward(rgb), scalar))
    right = float(np.vdot(rgb, operator.adjoint(scalar)))
    assert left == pytest.approx(right, rel=1e-14, abs=1e-14)


def test_mask_is_exact_diagonal_of_adjoint_forward() -> None:
    operator = xtrans_operator(13, 11, 5, 4)
    rgb = np.arange(3 * 13 * 11, dtype=np.float64).reshape(3, 13, 11)
    assert np.array_equal(operator.adjoint(operator.forward(rgb)), operator.mask * rgb)
    assert np.all(np.sum(operator.mask, axis=0) == 1.0)


def test_projection_preserves_only_physical_samples() -> None:
    operator = xtrans_operator(9, 10, 2, 3)
    rng = np.random.default_rng(9)
    estimate = rng.standard_normal((3, 9, 10))
    measured = rng.standard_normal((9, 10))
    projected = operator.project(estimate, measured)
    assert np.array_equal(operator.forward(projected), measured)
    assert np.array_equal(projected * (1.0 - operator.mask), estimate * (1.0 - operator.mask))


def test_soft_update_matches_closed_form() -> None:
    operator = bayer_bggr_operator(7, 8)
    rng = np.random.default_rng(7)
    consensus = rng.standard_normal((3, 7, 8))
    measured = rng.standard_normal((7, 8))
    rho = 0.3
    output = operator.soft_update(consensus, measured, rho)
    expected = consensus.copy()
    for y in range(7):
        for x in range(8):
            channel = int(operator.cfa[y, x])
            expected[channel, y, x] = (measured[y, x] + rho * consensus[channel, y, x]) / (1.0 + rho)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "bad",
    (np.zeros((2, 2, 2)), np.asarray([[0, 3]], dtype=np.int64), np.asarray([0, 1, 2])),
)
def test_bad_cfa_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        CFAOperator(bad)

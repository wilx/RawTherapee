from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_pnp.experiment import _region, metric, pooled


def test_metric_and_pooling() -> None:
    truth = np.zeros((3, 64, 66), dtype=np.float64)
    first = truth.copy()
    first[:, 12:-12, 12:-12] = 0.01
    second = truth.copy()
    second[:, 12:-12, 12:-12] = 0.02
    a = metric(first, truth)
    b = metric(second, truth)
    combined = pooled([a, b])
    assert a["mse"] == pytest.approx(0.0001)
    assert b["mse"] == pytest.approx(0.0004)
    assert combined["mse"] == pytest.approx(0.00025)
    assert combined["case_count"] == 2


def test_crop_regions_are_bounded_and_phase_aligned() -> None:
    for index in range(9):
        x0, y0, x1, y1 = _region((321, 481), 96, index)
        assert x0 % 6 == 0
        assert y0 % 6 == 0
        assert x1 - x0 == y1 - y0 == 96
        assert 0 <= x0 < x1 <= 481
        assert 0 <= y0 < y1 <= 321

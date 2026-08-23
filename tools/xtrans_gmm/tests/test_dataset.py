from __future__ import annotations

import numpy as np

from tools.xtrans_gmm.dataset import grid_samples


def test_grid_sampling_is_dense_and_deterministic() -> None:
    rgb = np.arange(3 * 32 * 35, dtype=np.float64).reshape(3, 32, 35) / 4000.0
    first = grid_samples(rgb, "source", "group", 5, 7)
    second = grid_samples(rgb, "source", "group", 5, 7)
    assert len(first) == 49
    assert [(row.x, row.y, row.phase) for row in first] == [
        (row.x, row.y, row.phase) for row in second
    ]
    for one, two in zip(first, second):
        np.testing.assert_array_equal(one.vector, two.vector)
        np.testing.assert_array_equal(one.indices, two.indices)
    centers = {(row.x + 2, row.y + 2) for row in first}
    assert len({x for x, _ in centers}) == 7
    assert len({y for _, y in centers}) == 7

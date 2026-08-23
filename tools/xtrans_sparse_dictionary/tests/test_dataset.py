from __future__ import annotations

import numpy as np

from tools.xtrans_sparse_dictionary.dataset import (
    sample_bright_patches,
    sample_rgb_patches,
)


def test_patch_sampling_is_deterministic():
    rgb = np.arange(3 * 13 * 15, dtype=np.float64).reshape(3, 13, 15) / 1000.0
    first = sample_rgb_patches(rgb, "source", "group", 5, 12)
    second = sample_rgb_patches(rgb, "source", "group", 5, 12)
    assert [(row.x, row.y) for row in first] == [(row.x, row.y) for row in second]
    assert all(np.array_equal(a.vector, b.vector) for a, b in zip(first, second))


def test_bright_sampling_targets_brightest_separated_points():
    rgb = np.zeros((3, 21, 21), dtype=np.float64)
    rgb[:, 5, 5] = 1.0
    rgb[:, 15, 15] = 0.9
    rows = sample_bright_patches(rgb, "stars", 5, 2)
    centers = {(row.x + 2, row.y + 2) for row in rows}
    assert centers == {(5, 5), (15, 15)}

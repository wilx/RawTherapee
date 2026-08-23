from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_ulri.analysis import method_metrics, patch_oracle


def test_metrics_identical_rgb_is_perfect():
    truth = np.full((3, 40, 42), 0.25)
    cfa = np.resize(np.asarray(((0, 1, 1, 2, 1, 1),)), (40, 42))
    metrics = method_metrics(truth, truth.copy(), cfa)
    assert metrics["rgb_rms"] == 0.0
    assert metrics["psnr_db"] == float("inf")
    assert metrics["maximum_abs"] == 0.0


def test_metrics_separate_missing_channels():
    truth = np.zeros((3, 40, 42))
    output = truth.copy()
    cfa = np.resize(np.asarray((0, 1, 2), dtype=np.uint8), (40, 42))
    output[0] = 0.25
    output[1] = 0.5
    metrics = method_metrics(truth, output, cfa)
    assert metrics["missing_red_blue_rms"] == pytest.approx(0.25 / np.sqrt(2))
    assert metrics["missing_green_rms"] == 0.5
    interior = cfa[16:-16, 16:-16]
    sampled = np.concatenate((
        output[0, 16:-16, 16:-16][interior == 0],
        output[1, 16:-16, 16:-16][interior == 1],
        output[2, 16:-16, 16:-16][interior == 2],
    ))
    assert metrics["sampled_rms"] == pytest.approx(np.sqrt(np.mean(sampled**2)))


def test_patch_oracle_selects_lower_error_candidate():
    truth = np.zeros((3, 40, 42))
    first = np.full_like(truth, 0.4)
    second = np.full_like(truth, 0.1)
    output, selected = patch_oracle(truth, (first, second), 7)
    assert np.all(selected == 1)
    assert np.array_equal(output, second)

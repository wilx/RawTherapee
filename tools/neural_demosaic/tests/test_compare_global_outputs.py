from __future__ import annotations

import numpy as np
import pytest

from tools.neural_demosaic.compare_gamma22_outputs import ComparisonError
from tools.xtrans_global.compare_outputs import analyze


def test_identical_global_outputs_have_zero_deltas() -> None:
    rows, columns = np.indices((48, 48))
    image = np.stack((
        rows * 101 + columns,
        rows * 31 + columns * 17,
        rows * 7 + columns * 89,
    ), axis=2).astype(np.uint16)
    images = {name: image.copy() for name in ("a", "b", "c", "markesteijn")}
    result = analyze(images, (0, 0, 48, 48))
    for key in ("a_minus_markesteijn", "b_minus_markesteijn",
                "c_minus_markesteijn", "b_minus_a", "c_minus_b"):
        assert result[key]["crop_mean_rgb_delta"] == [0.0, 0.0, 0.0]
        assert result[key]["rms_pixel_delta"] == 0.0
        assert result[key]["observed_sample_delta_rms"] == 0.0


def test_global_analyzer_detects_phase_and_color_errors() -> None:
    mark = np.full((48, 48, 3), 20000, dtype=np.uint16)
    phase_a = mark.copy()
    phase_b = mark.copy()
    phase_c = mark.copy()
    for y in range(48):
        for x in range(48):
            phase_a[y, x, 0] += (y % 3) * 60 + (x % 3) * 20
            phase_b[y, x, 1] += 100
            phase_c[y, x, 2] += 150
    result = analyze(
        {"a": phase_a, "b": phase_b, "c": phase_c, "markesteijn": mark},
        (0, 0, 48, 48),
    )
    assert result["methods"]["a"]["phase"]["rms_rgb"][0] > 0
    assert result["b_minus_markesteijn"]["crop_mean_rgb_delta"][1] == pytest.approx(
        100 / 65535
    )
    assert result["c_minus_markesteijn"]["rgb_delta_range"] > 0


def test_global_analyzer_rejects_invalid_geometry() -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint16)
    images = {name: image for name in ("a", "b", "c", "markesteijn")}
    with pytest.raises(ComparisonError, match="outside"):
        analyze(images, (40, 40, 20, 20))
    bad = dict(images)
    bad["c"] = np.zeros((47, 48, 3), dtype=np.uint16)
    with pytest.raises(ComparisonError, match="shapes differ"):
        analyze(bad, (0, 0, 24, 24))

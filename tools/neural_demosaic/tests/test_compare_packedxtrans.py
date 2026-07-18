from __future__ import annotations

import numpy as np
import pytest

from tools.neural_demosaic.compare_gamma22_outputs import ComparisonError
from tools.neural_demosaic.compare_packedxtrans_outputs import analyze_arrays


def images() -> dict[str, np.ndarray]:
    base = np.full((24, 24, 3), 20000, np.uint16)
    return {"packedxtransnet": base.copy(), "xveon": base.copy(), "markesteijn": base.copy()}


def test_identical_images_pass() -> None:
    result = analyze_arrays(images(), (0, 0, 24, 24))
    assert result["objective"]["pass"]
    assert result["objective"]["color"]["common_luminance_delta"] == 0


def test_phase_and_color_failures_are_detected() -> None:
    values = images()
    rows, columns = np.indices((24, 24))
    values["packedxtransnet"][..., 0] += ((rows % 3 == 0) & (columns % 3 == 0)).astype(np.uint16) * 1000
    values["packedxtransnet"][..., 2] += 1000
    result = analyze_arrays(values, (0, 0, 24, 24))
    assert not result["objective"]["pass"]
    assert not result["objective"]["color"]["pass"]


def test_contract_and_bounds_are_rejected() -> None:
    with pytest.raises(ComparisonError):
        analyze_arrays({"packedxtransnet": np.zeros((4, 4, 3), np.uint16)}, (0, 0, 4, 4))
    with pytest.raises(ComparisonError):
        analyze_arrays(images(), (20, 20, 8, 8))

from __future__ import annotations

import struct

import pytest

from tools.neural_demosaic.compare_xveon_backends import (
    OUTPUT_FLOATS,
    BackendComparisonError,
    compare,
)


def write_values(path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def test_identical_outputs(tmp_path):
    left = tmp_path / "left.f32le"
    right = tmp_path / "right.f32le"
    values = [float(index % 17) / 17 for index in range(OUTPUT_FLOATS)]
    write_values(left, values)
    write_values(right, values)
    result = compare(left, right)
    assert result["exact_match_count"] == OUTPUT_FLOATS
    assert result["maximum_absolute_error"] == 0
    assert all(result["acceptance"].values())


def test_records_known_error_and_gate(tmp_path):
    left = tmp_path / "left.f32le"
    right = tmp_path / "right.f32le"
    write_values(left, [1.0] * OUTPUT_FLOATS)
    write_values(right, [1.01] * OUTPUT_FLOATS)
    result = compare(left, right)
    assert result["maximum_absolute_error"] == pytest.approx(0.01)
    assert not any(result["acceptance"].values())


def test_rejects_wrong_size_and_nonfinite(tmp_path):
    good = tmp_path / "good.f32le"
    bad = tmp_path / "bad.f32le"
    write_values(good, [0.0] * OUTPUT_FLOATS)
    bad.write_bytes(b"short")
    with pytest.raises(BackendComparisonError):
        compare(good, bad)
    values = [0.0] * OUTPUT_FLOATS
    values[50] = float("nan")
    write_values(bad, values)
    with pytest.raises(BackendComparisonError):
        compare(good, bad)

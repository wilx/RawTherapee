from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.neural_demosaic.xveon_reference import (
    PATTERN,
    XVeonReferenceError,
    load_session,
    run_wrapper,
    transformed_cfa,
    unique_cfas,
)


def echo(tile: np.ndarray) -> np.ndarray:
    return np.concatenate((tile[:, :1], tile[:, :1] + 0.25, tile[:, :1] - 0.25), axis=1)


def test_all_18_cfa_mappings_and_signed_output():
    assert len(unique_cfas()) == 18
    raw = np.arange(1, 36, dtype=np.float32).reshape(5, 7) * 1000
    for cfa in unique_cfas():
        output = run_wrapper(raw, cfa, echo)
        assert output.shape == (5, 7, 3)
        assert np.isfinite(output).all()
    ordinary = run_wrapper(np.full((100, 100), 32767.5, np.float32), PATTERN, echo)
    assert ordinary[50, 50] == pytest.approx((0.5, 0.75, 0.25), abs=2e-6)


def test_right_bottom_zero_extension_and_leading_reflection():
    captured = []

    def capture(tile):
        captured.append(tile.copy())
        return np.zeros((1, 3, 288, 288), np.float32)

    raw = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
    run_wrapper(raw, transformed_cfa((1, 0, 0, 1), 1, 1), capture)
    assert len(captured) == 1
    assert np.count_nonzero(captured[0][0, 0]) == 20  # 12 samples plus deterministic leading phase reflection.
    assert captured[0][0, 0, -1, -1] == 0
    assert np.all(captured[0][0, 1:].sum(axis=0) == 1)


def test_invalid_cfa_and_nonfinite_values_are_rejected():
    with pytest.raises(XVeonReferenceError, match="phase/orientation"):
        run_wrapper(np.ones((2, 2), np.float32), np.zeros((6, 6), np.uint8), echo)
    bad = np.ones((2, 2), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(XVeonReferenceError, match="non-finite"):
        run_wrapper(bad, PATTERN, echo)


@pytest.mark.skipif(not os.environ.get("XVEON_XTRANS_ONNX"), reason="XVEON_XTRANS_ONNX is not set")
def test_reviewed_model_is_deterministic():
    session = load_session(Path(os.environ["XVEON_XTRANS_ONNX"]))
    tile = np.zeros((1, 4, 288, 288), np.float32)
    rows, columns = np.indices((288, 288))
    channels = PATTERN[rows % 6, columns % 6]
    tile[0, 0] = ((rows * 31 + columns * 17) % 1024) / 1023
    for channel in range(3):
        tile[0, channel + 1] = channels == channel
    first = session.run(["output"], {"input": tile})[0]
    second = session.run(["output"], {"input": tile})[0]
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


@pytest.mark.skipif(
    not os.environ.get("XVEON_XTRANS_ONNX") or not os.environ.get("XVEON_NATIVE_TILE_TOOL"),
    reason="XVEON_XTRANS_ONNX and XVEON_NATIVE_TILE_TOOL are not set",
)
def test_native_tile_matches_independent_onnxruntime(tmp_path):
    model = Path(os.environ["XVEON_XTRANS_ONNX"])
    tool = Path(os.environ["XVEON_NATIVE_TILE_TOOL"])
    output_path = tmp_path / "native.f32le"
    subprocess.run([str(tool), str(model), str(output_path)], check=True)
    native = np.fromfile(output_path, dtype="<f4").reshape(1, 3, 288, 288)

    session = load_session(model)
    rows, columns = np.indices((288, 288))
    channels = PATTERN[rows % 6, columns % 6]
    tile = np.zeros((1, 4, 288, 288), np.float32)
    tile[0, 0] = ((rows * 31 + columns * 17) % 1024) / 1023
    for channel in range(3):
        tile[0, channel + 1] = channels == channel
    reference = session.run(["output"], {"input": tile})[0]
    tolerance = 5e-6 + 1e-5 * np.abs(reference)
    assert np.all(np.abs(native - reference) <= tolerance)


@pytest.mark.skipif(
    not os.environ.get("XVEON_XTRANS_ONNX") or not os.environ.get("XVEON_NATIVE_WRAPPER_TOOL"),
    reason="XVEON_XTRANS_ONNX and XVEON_NATIVE_WRAPPER_TOOL are not set",
)
def test_native_wrapper_matches_independent_geometry_and_onnxruntime(tmp_path):
    model = Path(os.environ["XVEON_XTRANS_ONNX"])
    output_path = tmp_path / "native-wrapper.f32le"
    subprocess.run(
        [os.environ["XVEON_NATIVE_WRAPPER_TOOL"], str(model), str(output_path)],
        check=True,
    )
    native = np.fromfile(output_path, dtype="<f4").reshape(31, 173, 3)

    rows, columns = np.indices((31, 173))
    raw = ((rows * 197 + columns * 113) % 65536).astype(np.float32)
    cfa = transformed_cfa((0, -1, 1, 0), 2, 3)
    session = load_session(model)
    reference = run_wrapper(raw, cfa, lambda tile: session.run(["output"], {"input": tile})[0])
    tolerance = 5e-6 + 1e-5 * np.abs(reference)
    assert np.all(np.abs(native - reference) <= tolerance)

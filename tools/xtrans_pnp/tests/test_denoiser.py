from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tools.xtrans_pnp.denoiser import BM3DDenoiser, authenticate_bm3d


@pytest.mark.skipif(importlib.util.find_spec("bm3d") is None, reason="external research BM3D absent")
def test_authenticated_bm3d_and_joint_color_call() -> None:
    identity = authenticate_bm3d()
    assert identity["packages"]["bm3d"]["version"] == "4.0.3"
    assert identity["packages"]["bm4d"]["version"] == "4.2.5"
    rng = np.random.default_rng(1)
    data = rng.random((3, 24, 25))
    for mode in ("rgb", "linear-luma"):
        denoiser = BM3DDenoiser(mode)
        output = denoiser(data, 0.002)
        assert output.shape == data.shape
        assert np.isfinite(output).all()
        assert denoiser.calls == 1
        assert denoiser.seconds > 0.0

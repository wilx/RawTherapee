from __future__ import annotations

import numpy as np

from tools.xtrans_mlri_green_fusion.analysis import (
    fit_ridge_score,
    patch_blend_alpha,
    patch_labels,
    preserve_measured_green,
    scalar_convex_hull,
    temperature_fusion,
)
from tools.xtrans_mlri_internal.dataset import cfa_for_origin
from tools.xtrans_mlri_green_fusion.dataset import _coherence_truth


def _case() -> dict[str, object]:
    height = width = 36
    y, x = np.mgrid[:height, :width]
    cfa = cfa_for_origin(height, width, 0, 0)
    mosaic = 0.1 + 0.01 * x + 0.005 * y
    trace: dict[str, np.ndarray] = {
        "pass0-green": mosaic + 0.01,
        "pass1-green": mosaic + 0.02,
    }
    for pass_index in (0, 1):
        for direction in range(8):
            trace[f"pass{pass_index}-green-direction-{direction}"] = (
                mosaic + 0.01 * pass_index + 0.001 * direction
            )
            trace[f"pass{pass_index}-green-energy-{direction}"] = np.full(
                mosaic.shape, direction + 1.0,
            )
            weight = np.full(mosaic.shape, 1.0 / 8)
            weight[cfa == 1] = 0
            trace[f"pass{pass_index}-green-weight-{direction}"] = weight
    truth = np.stack((mosaic, mosaic + 0.015, mosaic))
    return {
        "cfa": cfa,
        "mosaic": mosaic,
        "origin_x": 0,
        "origin_y": 0,
        "trace": trace,
        "truth": truth,
        "case_id": "unit",
    }


def test_preserve_measured_green() -> None:
    case = _case()
    values = np.zeros_like(case["mosaic"])
    result = preserve_measured_green(case, values)
    measured = case["cfa"] == 1
    assert np.array_equal(result[measured], case["mosaic"][measured])
    assert np.all(result[~measured] == 0)


def test_patch_label_prefers_exact_candidate() -> None:
    truth = np.full((9, 9), 0.4)
    candidates = np.stack((np.full_like(truth, 0.1), truth, np.full_like(truth, 0.8)))
    for window in (1, 3, 7):
        assert np.all(patch_labels(candidates, truth, window) == 1)


def test_convex_hull_contains_and_preserves_samples() -> None:
    case = _case()
    candidates = np.stack((
        np.asarray(case["truth"])[1] - 0.1,
        np.asarray(case["truth"])[1] + 0.1,
    ))
    guide, contained = scalar_convex_hull(case, candidates)
    missing = case["cfa"] != 1
    assert np.all(contained)
    assert np.allclose(guide[missing], np.asarray(case["truth"])[1][missing])
    measured = ~missing
    assert np.array_equal(guide[measured], case["mosaic"][measured])


def test_patch_blend_alpha_recovers_midpoint() -> None:
    first = np.zeros((11, 11))
    second = np.ones((11, 11))
    truth = np.full((11, 11), 0.25)
    assert np.allclose(patch_blend_alpha(first, second, truth, 1), 0.25)
    assert np.allclose(patch_blend_alpha(first, second, truth, 7), 0.25)


def test_temperature_zero_is_uniform_mean() -> None:
    case = _case()
    result = temperature_fusion(case, 1, 0)
    candidates = np.stack([
        case["trace"][f"pass1-green-direction-{direction}"] for direction in range(8)
    ])
    missing = case["cfa"] != 1
    assert np.allclose(result[missing], np.mean(candidates, axis=0)[missing])


def test_ridge_score_orders_linear_target() -> None:
    x = np.arange(40, dtype=np.float64)[:, None]
    target = 2.0 * x[:, 0] + 3.0
    model = fit_ridge_score(x, target, np.ones(40), 1e-10)
    prediction = model.score(x)
    assert np.max(np.abs(prediction - target)) < 1e-8


def test_coherence_transition_increases_support() -> None:
    levels = (
        "isolated-points", "aligned-sparse-points", "dotted-line",
        "continuous-line", "repeated-texture",
    )
    changed = [
        int(np.sum(np.any(_coherence_truth(level) != 0.12, axis=0)))
        for level in levels
    ]
    assert changed[:4] == sorted(changed[:4])
    assert changed[-1] > changed[-2]

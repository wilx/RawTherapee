import numpy as np

from tools.xtrans_color_line.analysis import oracle_gate, reconstruction_metrics
from tools.xtrans_color_line.model import (
    fit_color_line,
    fit_color_line_huber,
    reconstruct_from_line,
)
from tools.xtrans_mlri_internal.dataset import mosaic, origin_cells


def _two_color(size=36):
    y, x = np.mgrid[:size, :size]
    alpha = (x + 2 * y) / (3 * (size - 1))
    first = np.asarray((0.08, 0.24, 0.7))[:, None, None]
    second = np.asarray((0.9, 0.6, 0.1))[:, None, None]
    return first * (1 - alpha) + second * alpha


def test_exact_color_line_reconstructs_from_every_phase():
    truth = _two_color()
    for origin_x, origin_y in origin_cells():
        scalar, cfa = mosaic(truth, origin_x, origin_y)
        fit = fit_color_line(truth, 7)
        result = reconstruct_from_line(fit, scalar, cfa, truth=truth)
        assert np.max(np.abs(result.rgb[:, 8:-8, 8:-8] - truth[:, 8:-8, 8:-8])) < 1e-11
        for channel in range(3):
            mask = cfa == channel
            assert np.array_equal(result.rgb[channel, mask], scalar[mask])


def test_rank_one_and_rank_two_model_metrics_are_ordered():
    line = _two_color()
    fit = fit_color_line(line, 7)
    assert np.max(fit.eigenvalues[..., 0]) < 1e-12
    assert np.max(fit.eigenvalues[..., 1]) < 1e-12
    three_color = line.copy()
    three_color[1, 14:22, 14:22] += 0.2
    three_fit = fit_color_line(three_color, 7)
    assert np.mean(three_fit.eigenvalues[..., 1]) > np.mean(fit.eigenvalues[..., 1])
    robust = fit_color_line_huber(three_color, 7)
    assert np.isfinite(robust.direction).all()
    assert np.allclose(np.linalg.norm(robust.direction, axis=-1), 1.0)


def test_degenerate_sample_channel_is_reported_and_preserved():
    size = 36
    y, x = np.mgrid[:size, :size]
    alpha = x / (size - 1)
    truth = np.stack((0.1 + 0.8 * alpha, np.full_like(alpha, 0.4), 0.8 - 0.6 * alpha))
    scalar, cfa = mosaic(truth, 0, 0)
    result = reconstruct_from_line(fit_color_line(truth, 7), scalar, cfa)
    assert np.mean(~result.identifiable[cfa == 1]) > 0.99
    assert np.mean(result.identifiable[cfa != 1]) > 0.99
    assert np.array_equal(result.rgb[1, cfa == 1], scalar[cfa == 1])
    plane = reconstruct_from_line(
        fit_color_line(truth, 7), scalar, cfa, degenerate_fallback="plane"
    )
    assert np.isfinite(plane.rgb).all()
    assert np.array_equal(plane.rgb[1, cfa == 1], scalar[cfa == 1])


def test_oracle_gate_and_metrics_contracts():
    truth = _two_color(40)
    scalar, cfa = mosaic(truth, 0, 0)
    baseline = truth + 0.05
    line = reconstruct_from_line(fit_color_line(truth, 7), scalar, cfa).rgb
    output, labels = oracle_gate(truth, baseline, line, cfa, 7)
    assert labels.shape == cfa.shape
    assert np.mean(labels) > 0.99
    metrics = reconstruction_metrics(truth, output, cfa, margin=8)
    assert metrics["psnr_db"] > 100
    assert metrics["missing_channel_psnr_db"] > 100


def test_cfa_only_unconstrained_two_color_model_is_not_unique():
    truth = _two_color(36)
    scalar, cfa = mosaic(truth, 0, 0)
    observed_min = np.asarray([np.min(scalar[cfa == channel]) for channel in range(3)])
    observed_max = np.asarray([np.max(scalar[cfa == channel]) for channel in range(3)])
    solutions = []
    for expansion in (0.05, 0.15):
        low = observed_min - expansion
        high = observed_max + expansion
        alpha = (scalar - low[cfa]) / (high[cfa] - low[cfa])
        reconstructed = low[:, None, None] + alpha[None] * (
            high - low
        )[:, None, None]
        sampled = np.take_along_axis(
            np.moveaxis(reconstructed, 0, -1), cfa[..., None], axis=2
        )[..., 0]
        assert np.all((alpha >= 0.0) & (alpha <= 1.0))
        assert np.max(np.abs(sampled - scalar)) < 1e-12
        solutions.append(reconstructed)
    assert np.max(np.abs(solutions[0] - solutions[1])) > 0.05

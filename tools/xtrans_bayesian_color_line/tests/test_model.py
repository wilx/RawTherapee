import numpy as np

from tools.xtrans_bayesian_color_line.model import (
    bennett_alpha,
    direct_alpha,
    oracle_neighbor_alpha,
    oracle_neighbor_alpha_from_rgb,
    oracle_plane_alpha,
    oracle_plane_alpha_from_rgb,
    reconstruct,
)
from tools.xtrans_mlri_internal.dataset import mosaic, origin_cells


def _line_truth(size=36):
    y, x = np.mgrid[:size, :size]
    alpha = (x + 2.0 * y) / (3.0 * (size - 1))
    first_color = np.asarray((0.08, 0.24, 0.70))
    second_color = np.asarray((0.90, 0.60, 0.10))
    rgb = first_color[:, None, None] * (1.0 - alpha) + second_color[:, None, None] * alpha
    first = np.broadcast_to(first_color, (size, size, 3)).copy()
    second = np.broadcast_to(second_color, (size, size, 3)).copy()
    return rgb, first, second, alpha


def test_direct_alpha_is_exact_for_all_xtrans_phases():
    truth, first, second, expected = _line_truth()
    for origin_x, origin_y in origin_cells():
        scalar, cfa = mosaic(truth, origin_x, origin_y)
        alpha, identifiable = direct_alpha(scalar, cfa, first, second)
        assert np.all(identifiable)
        assert np.max(np.abs(alpha - expected)) < 1e-14


def test_bennett_constant_patch_reproduces_exact_bayer_behavior():
    size = 21
    alpha_value = 0.37
    first = np.broadcast_to((0.1, 0.3, 0.8), (size, size, 3)).copy()
    second = np.broadcast_to((0.9, 0.7, 0.2), (size, size, 3)).copy()
    truth = np.moveaxis(first + alpha_value * (second - first), -1, 0)
    y, x = np.mgrid[:size, :size]
    bayer = np.where(y % 2 == 0, np.where(x % 2 == 0, 0, 1), np.where(x % 2 == 0, 1, 2))
    scalar = np.take_along_axis(np.moveaxis(truth, 0, -1), bayer[..., None], axis=2)[..., 0]
    result = bennett_alpha(scalar, bayer, first, second, eta=1.0)
    assert np.all(result.identifiable)
    assert np.max(np.abs(result.alpha - alpha_value)) < 1e-14
    rgb = reconstruct(first, second, result.alpha, scalar=scalar, cfa=bayer)
    assert np.max(np.abs(rgb - truth)) < 1e-14


def test_published_endpoint_prior_compares_zero_one_and_alpha_star():
    truth, first, second, _ = _line_truth(21)
    scalar, cfa = mosaic(truth, 0, 0)
    flat = bennett_alpha(scalar, cfa, first, second, eta=1.0)
    biased = bennett_alpha(scalar, cfa, first, second, eta=1e-12)
    assert np.all(flat.selected_candidate == 1)
    assert np.mean(biased.selected_candidate != 1) > 0.05
    assert set(np.unique(biased.selected_candidate)) == {0, 1, 2}
    assert np.all((biased.alpha >= 0.0) & (biased.alpha <= 1.0))


def test_oracle_plane_and_neighbor_intercepts_are_equivalent_on_symmetric_support():
    truth, first, second, alpha = _line_truth(25)
    neighbor = oracle_neighbor_alpha(alpha, support_radius=2)
    plane = oracle_plane_alpha(alpha, support_radius=2)
    assert np.max(np.abs(neighbor - plane)) < 1e-14
    neighbor_rgb = oracle_neighbor_alpha_from_rgb(
        truth, first, second, support_radius=2
    )
    plane_rgb = oracle_plane_alpha_from_rgb(
        truth, first, second, support_radius=2
    )
    assert np.max(np.abs(neighbor_rgb - plane_rgb)) < 1e-14
    assert np.max(np.abs(neighbor_rgb - neighbor)) < 1e-14
    assert np.max(np.abs(plane_rgb[2:-2, 2:-2] - alpha[2:-2, 2:-2])) < 1e-14


def test_bennett_rejects_degenerate_endpoint_channels():
    truth, first, second, _ = _line_truth(24)
    second[..., 1] = first[..., 1]
    scalar, cfa = mosaic(truth, 0, 0)
    result = bennett_alpha(scalar, cfa, first, second)
    assert np.all(result.identifiable)
    assert np.isfinite(result.alpha).all()


def test_bennett_rejects_invalid_cfa_and_nonfinite_data():
    truth, first, second, _ = _line_truth(18)
    scalar, cfa = mosaic(truth, 0, 0)
    invalid_cfa = cfa.copy()
    invalid_cfa[4, 4] = 3
    with np.testing.assert_raises(ValueError):
        bennett_alpha(scalar, invalid_cfa, first, second)
    invalid_scalar = scalar.copy()
    invalid_scalar[4, 4] = np.nan
    with np.testing.assert_raises(ValueError):
        bennett_alpha(invalid_scalar, cfa, first, second)

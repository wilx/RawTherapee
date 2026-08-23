import numpy as np

from tools.xtrans_bayesian_color_line.bayer_reference import (
    bennett_endpoints,
    full_bayer_reference,
    malvar_hqli,
    mosaic_bayer,
)


def test_malvar_bootstrap_is_exact_on_constant_and_preserves_samples():
    truth = np.broadcast_to(
        np.asarray((0.2, 0.4, 0.7))[:, None, None], (3, 24, 26)
    ).copy()
    scalar, cfa = mosaic_bayer(truth)
    result = malvar_hqli(scalar, cfa)
    assert np.max(np.abs(result[:, 3:-3, 3:-3] - truth[:, 3:-3, 3:-3])) < 1e-14
    for channel in range(3):
        assert np.array_equal(result[channel, cfa == channel], scalar[cfa == channel])


def test_weighted_clustering_recovers_two_exact_regions_away_from_edge():
    truth = np.empty((3, 30, 30), dtype=np.float64)
    truth[:, :, :15] = np.asarray((0.1, 0.3, 0.8))[:, None, None]
    truth[:, :, 15:] = np.asarray((0.9, 0.6, 0.2))[:, None, None]
    first, second = bennett_endpoints(truth)
    for x in (7, 22):
        assert np.max(np.abs(first[15, x] - second[15, x])) < 1e-14
        assert np.max(np.abs(first[15, x] - truth[:, 15, x])) < 1e-14


def test_full_bayer_reference_is_finite_and_native_sample_exact():
    size = 31
    y, x = np.mgrid[:size, :size]
    alpha = np.clip((x + y - 12.0) / 20.0, 0.0, 1.0)
    first = np.asarray((0.08, 0.23, 0.72))[:, None, None]
    second = np.asarray((0.91, 0.62, 0.11))[:, None, None]
    truth = first * (1.0 - alpha) + second * alpha
    scalar, cfa = mosaic_bayer(truth)
    result = full_bayer_reference(scalar, cfa)
    assert np.isfinite(result.rgb).all()
    for channel in range(3):
        assert np.array_equal(result.rgb[channel, cfa == channel], scalar[cfa == channel])

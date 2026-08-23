import numpy as np

from tools.xtrans_lmmse.model import phase_statistics
from tools.xtrans_lmmse.population import _Accumulator


def test_parallel_moment_merge_matches_direct_statistics():
    rng = np.random.default_rng(0x4C4D4D53)
    observations = [rng.normal(size=(40, 9)) for _ in range(18)]
    targets = [rng.normal(size=(40, 3)) for _ in range(18)]
    accumulator = _Accumulator(9)
    accumulator.add(
        [row[:17] for row in observations],
        [row[:17] for row in targets],
        support=3,
        dc_mode="m0",
    )
    accumulator.add(
        [row[17:] for row in observations],
        [row[17:] for row in targets],
        support=3,
        dc_mode="m0",
    )
    merged = accumulator.statistics()
    direct = phase_statistics(observations, targets)
    for actual, expected in zip(merged, direct):
        assert actual.count == expected.count
        assert np.allclose(actual.x_mean, expected.x_mean, atol=1e-14)
        assert np.allclose(actual.target_mean, expected.target_mean, atol=1e-14)
        assert np.allclose(actual.covariance, expected.covariance, atol=1e-13)
        assert np.allclose(
            actual.cross_covariance, expected.cross_covariance, atol=1e-13
        )

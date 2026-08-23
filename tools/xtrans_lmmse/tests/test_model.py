import numpy as np

from tools.xtrans_lmmse.model import (
    LC_BASIS,
    collect_phase_samples,
    derive_filter_bank,
    geometry_interpolate_region,
    phase_index_table,
    phase_map,
    phase_statistics,
    predict_region,
)
from tools.xtrans_mlri_internal.dataset import cfa_for_origin, origin_cells


def _scene(height: int = 72, width: int = 78) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    result = np.empty((3, height, width), dtype=np.float64)
    result[0] = 0.10 + 0.55 * x / (width - 1)
    result[1] = 0.18 + 0.42 * y / (height - 1)
    result[2] = 0.08 + 0.28 * (x + y) / (width + height - 2)
    return result


def test_phase_lookup_has_eighteen_cells_and_tracks_origin():
    table = phase_index_table()
    assert table.shape == (6, 6)
    assert set(table.ravel()) == set(range(18))
    for origin_x, origin_y in origin_cells():
        phases = phase_map(6, 6, origin_x, origin_y)
        expected = phase_index_table()[(np.arange(6)[:, None] + origin_y) % 6,
                                       (np.arange(6)[None, :] + origin_x) % 6]
        assert np.array_equal(phases, expected)


def test_lc_basis_is_orthogonal():
    assert np.allclose(LC_BASIS @ LC_BASIS.T, np.eye(3), atol=1e-15)


def test_rgb_and_orthogonal_target_fits_are_equivalent():
    truth = _scene()
    train = (0, 0, 42, truth.shape[1])
    evaluate = (48, 5, truth.shape[2], truth.shape[1] - 5)
    observations, targets = collect_phase_samples(truth, 5, train)
    rgb = derive_filter_bank(
        phase_statistics(observations, targets, basis="rgb"), 5, 1e-5,
        basis="rgb",
    )
    lc = derive_filter_bank(
        phase_statistics(observations, targets, basis="lc"), 5, 1e-5,
        basis="lc",
    )
    rgb_result = predict_region(rgb, truth, evaluate)
    lc_result = predict_region(lc, truth, evaluate)
    # The two formulations are the same linear subspace; the small residual is
    # solely the different floating-point order in covariance formation/solve.
    assert np.max(np.abs(rgb_result - lc_result)) < 1e-9


def test_predictor_and_geometry_preserve_native_samples():
    truth = _scene()
    origin_x, origin_y = origin_cells()[7]
    observations, targets = collect_phase_samples(
        truth, 5, (0, 0, 42, truth.shape[1]),
        origin_x=origin_x, origin_y=origin_y,
    )
    bank = derive_filter_bank(
        phase_statistics(observations, targets), 5, 1e-5
    )
    region = (48, 4, truth.shape[2], truth.shape[1] - 4)
    predicted = predict_region(
        bank, truth, region, origin_x=origin_x, origin_y=origin_y
    )
    geometry = geometry_interpolate_region(
        truth, 5, region, origin_x=origin_x, origin_y=origin_y
    )
    x0, y0, x1, y1 = region
    cfa = cfa_for_origin(truth.shape[1], truth.shape[2], origin_x, origin_y)[
        y0:y1, x0:x1
    ]
    expected = truth[:, y0:y1, x0:x1]
    for channel in range(3):
        selected = cfa == channel
        assert np.array_equal(predicted[channel, selected], expected[channel, selected])
        assert np.array_equal(geometry[channel, selected], expected[channel, selected])


def test_filter_contract_and_finite_output():
    truth = _scene()
    observations, targets = collect_phase_samples(truth, 3, (0, 0, 42, 72))
    statistics = phase_statistics(observations, targets)
    bank = derive_filter_bank(statistics, 3, 1e-4)
    assert bank.weights.shape == (18, 3, 9)
    assert np.all(bank.sample_counts > 0)
    assert np.isfinite(bank.condition_numbers).all()
    assert np.isfinite(bank.filter_norms).all()
    output = predict_region(bank, truth, (48, 3, 78, 69), clip=True)
    assert output.shape == (3, 66, 30)
    assert np.isfinite(output).all()
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0


def test_observable_dc_modes_reconstruct_constant_fields():
    truth = np.empty((3, 72, 78), dtype=np.float64)
    truth[0] = 0.17
    truth[1] = 0.43
    truth[2] = 0.81
    train = (0, 0, 42, 72)
    test = (48, 4, 73, 68)
    observations, targets = collect_phase_samples(truth, 5, train)
    for dc_mode in ("m0", "m1", "m2"):
        statistics = phase_statistics(
            observations, targets, dc_mode=dc_mode, support=5
        )
        bank = derive_filter_bank(
            statistics, 5, 1e-4, dc_mode=dc_mode
        )
        output = predict_region(bank, truth, test)
        expected = truth[:, test[1] : test[3], test[0] : test[2]]
        assert np.max(np.abs(output - expected)) < 1e-6

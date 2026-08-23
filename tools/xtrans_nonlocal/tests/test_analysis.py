import numpy as np
import pytest

from tools.xtrans_nonlocal.analysis import (
    PATCH_SIZE,
    _candidate_centers,
    analyze_target,
    cfa_for_context,
    direct_nonlocal,
    low_rank_completion,
    mosaic,
)


def test_cfa_context_origin_and_mosaic():
    cfa = cfa_for_context(12, 12, (5, -7))
    shifted = cfa_for_context(12, 12, (11, -1))
    assert np.array_equal(cfa, shifted)
    truth = np.stack((np.full((12, 12), 0.1), np.full((12, 12), 0.5), np.full((12, 12), 0.9)))
    sampled = mosaic(truth, cfa)
    for channel, value in enumerate((0.1, 0.5, 0.9)):
        assert np.all(sampled[cfa == channel] == value)


def test_same_phase_has_one_pattern_and_no_missing_donors():
    cfa = cfa_for_context(80, 80, (0, 0))
    ys, xs = _candidate_centers((80, 80), (40, 40), same_phase=True)
    patches = [cfa[y - 3:y + 4, x - 3:x + 4] for y, x in zip(ys, xs)]
    assert len({patch.tobytes() for patch in patches}) == 1
    target = cfa[37:44, 37:44]
    for patch in patches:
        assert not np.any(patch != target)


def test_cross_phase_direct_donation_reinjects_samples():
    cfa = cfa_for_context(80, 80, (0, 0))
    y, x = np.mgrid[:80, :80]
    truth = np.stack((0.1 + x / 200, 0.3 + y / 250, 0.6 + (x + y) / 500))
    scalar = mosaic(truth, cfa)
    center = (40, 40)
    ys, xs = _candidate_centers((80, 80), center, same_phase=False)
    candidates = np.stack([scalar[y0 - 3:y0 + 4, x0 - 3:x0 + 4] for y0, x0 in zip(ys, xs)])
    masks = np.stack([cfa[y0 - 3:y0 + 4, x0 - 3:x0 + 4] for y0, x0 in zip(ys, xs)])
    target_scalar = scalar[37:44, 37:44]
    target_cfa = cfa[37:44, 37:44]
    indices = np.arange(min(128, len(ys)))
    result, coverage = direct_nonlocal(
        target_scalar, target_cfa, candidates, masks, indices,
        np.linspace(0.001, 0.1, len(ys)),
    )
    assert coverage == pytest.approx(1.0)
    for channel in range(3):
        observed = target_cfa == channel
        assert np.array_equal(result[channel, observed], target_scalar[observed])


def test_low_rank_completion_is_finite_and_reinjects():
    rng = np.random.default_rng(20260822)
    cfa = cfa_for_context(60, 60, (0, 0))
    scalar = rng.random((60, 60))
    target_scalar = scalar[27:34, 27:34]
    target_cfa = cfa[27:34, 27:34]
    centers = [(10 + (3 * i) % 40, 10 + (7 * i) % 40) for i in range(24)]
    candidates = np.stack([scalar[y - 3:y + 4, x - 3:x + 4] for y, x in centers])
    masks = np.stack([cfa[y - 3:y + 4, x - 3:x + 4] for y, x in centers])
    result, coverage = low_rank_completion(
        target_scalar, target_cfa, candidates, masks, np.arange(24), iterations=2,
    )
    assert coverage > 0.8
    assert np.isfinite(result).all()
    for channel in range(3):
        observed = target_cfa == channel
        assert np.array_equal(result[channel, observed], target_scalar[observed])


def test_small_repeated_scene_has_pure_observable_neighbors():
    size = 96
    y, x = np.mgrid[:size, :size]
    motif = ((x // 8 + y // 8) % 2).astype(np.float64)
    truth = np.stack((0.2 + 0.5 * motif, 0.3 + 0.3 * motif, 0.4 + 0.2 * motif))
    cfa = cfa_for_context(size, size, (0, 0))
    scalar = mosaic(truth, cfa)
    baselines = {"markesteijn": truth, "mlri": truth, "practical_blender": truth}
    row, _ = analyze_target(truth, scalar, cfa, (48, 48), baselines)
    assert row["same_phase"]["phase_pattern_count"] == 1
    assert row["cross_phase"]["phase_pattern_count"] == 18
    assert row["same_phase"]["best_cfa_selected_rgb_mse"] < 1e-15
    assert row["cross_phase"]["direct_donor_coverage"] == pytest.approx(1.0)

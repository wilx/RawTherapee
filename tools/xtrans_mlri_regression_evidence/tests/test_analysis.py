import numpy as np

from tools.xtrans_mlri_regression_evidence.analysis import (
    centered,
    modulated_guide,
    pairwise_metrics,
    within_pixel_rank_metrics,
)


def test_centered_removes_pixel_difficulty():
    values = np.asarray([[10, 11, 12], [100, 101, 102]], dtype=np.float64)
    np.testing.assert_allclose(centered(values), [[-1, 0, 1], [-1, 0, 1]])


def test_perfect_cost_ranking_has_exact_top_one():
    error = np.asarray([[4, 1, 9], [0, 2, 3]], dtype=np.float64)
    row = within_pixel_rank_metrics(error, error)
    assert row["mean_within_pixel_spearman"] == 1.0
    assert row["top1_accuracy"] == 1.0
    assert row["top2_inclusion"] == 1.0


def test_inverted_pairwise_score_is_rejected():
    ascending = np.arange(16, dtype=np.float64)
    error = np.stack((ascending, ascending[::-1], np.roll(ascending, 5)))
    good = pairwise_metrics(error, error)
    bad = pairwise_metrics(-error, error)
    assert good["all"]["all"]["auc"] == 1.0
    assert bad["all"]["all"]["auc"] == 0.0


def test_modulation_preserves_measured_green_and_is_bounded():
    height = width = 4
    candidates = np.stack([
        np.full((height, width), index / 20.0) for index in range(16)
    ])
    trace = {}
    for pass_index in (0, 1):
        for direction in range(8):
            trace[f"pass{pass_index}-green-direction-{direction}"] = candidates[pass_index * 8 + direction]
            trace[f"pass{pass_index}-green-energy-{direction}"] = np.ones((height, width))
            trace[f"pass{pass_index}-green-weight-{direction}"] = np.full((height, width), 1 / 8)
    cfa = np.zeros((height, width), dtype=np.int32)
    cfa[::2, ::2] = 1
    mosaic = np.full((height, width), 0.37)
    case = {"trace": trace, "cfa": cfa, "mosaic": mosaic}
    scores = np.broadcast_to(np.arange(16)[:, None, None], candidates.shape)
    guide = modulated_guide(case, scores, 1.0)
    assert np.all(guide[cfa == 1] == np.float32(0.37))
    assert np.all((guide >= 0) & (guide <= 0.75))

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.xtrans_alias.analysis import CANONICAL_XTRANS
from tools.xtrans_oracle.analysis import (
    block_oracle,
    convex_blend_oracle,
    from_opponent,
    label_consistency,
    method_metrics,
    normalized_correlation,
    opponent,
    winner_statistics,
)
from tools.xtrans_oracle.generate import synthetic_scenes, tiled_cfa


EXPECTED_MANIFEST_SHA256 = "f433c4a3d241b3ab46c9f95a01edd160a39a96db823e92dbd8316d3d15c2c260"


def test_opponent_basis_round_trips_exactly_within_float64_precision():
    rng = np.random.default_rng(0x4F5241434C45)
    rgb = rng.random((3, 17, 19))
    assert np.max(np.abs(from_opponent(opponent(rgb)) - rgb)) < 5e-16


def test_block_oracle_selects_the_lower_error_method_per_block():
    truth = np.zeros((3, 8, 8))
    left = np.ones_like(truth)
    right = np.ones_like(truth)
    left[:, :, :4] = 0.1
    right[:, :, 4:] = 0.2
    selected, labels = block_oracle(
        truth, {"left": left, "right": right}, ("left", "right"),
        4, margin=0,
    )
    assert np.all(labels[:, :4] == 0)
    assert np.all(labels[:, 4:] == 1)
    assert np.max(selected[:, :, :4]) == pytest.approx(0.1)
    assert np.max(selected[:, :, 4:]) == pytest.approx(0.2)


def test_patch_granularity_prevents_pixel_scale_switching():
    truth = np.zeros((3, 8, 8))
    checker = np.indices((8, 8)).sum(axis=0) % 2
    left = np.broadcast_to(checker, truth.shape).copy()
    right = 1 - left
    pixel, _ = block_oracle(
        truth, {"left": left, "right": right}, ("left", "right"), 1, margin=0
    )
    blocks, labels = block_oracle(
        truth, {"left": left, "right": right}, ("left", "right"), 4, margin=0
    )
    assert np.max(np.abs(pixel)) == 0
    assert np.sqrt(np.mean(blocks**2)) > 0.6
    assert np.unique(labels).tolist() == [0]


def test_convex_oracle_finds_midpoint_better_than_hard_selection():
    truth = np.full((3, 6, 6), 0.5)
    left = np.ones_like(truth)
    right = np.zeros_like(truth)
    blended, alpha = convex_blend_oracle(truth, left, right, 3, margin=0)
    assert np.max(np.abs(blended - truth)) < 1e-15
    assert np.nanmin(alpha) == pytest.approx(0.5)
    assert np.nanmax(alpha) == pytest.approx(0.5)


def test_winner_statistics_distinguish_coherent_and_fragmented_maps():
    coherent = np.zeros((8, 8), dtype=np.int16)
    coherent[:, 4:] = 1
    checker = np.indices((8, 8)).sum(axis=0).astype(np.int16) % 2
    coherent_stats = winner_statistics(coherent, 2, margin=0)
    checker_stats = winner_statistics(checker, 2, margin=0)
    assert coherent_stats["connected_regions"] == 2
    assert checker_stats["connected_regions"] == 64
    assert checker_stats["boundary_fraction"] > coherent_stats["boundary_fraction"]
    assert label_consistency(coherent, coherent.copy(), margin=0) == 1


def test_method_metrics_separate_native_and_interpolated_samples():
    truth = np.full((3, 12, 12), 0.4)
    cfa = tiled_cfa(12, 12)
    output = truth.copy()
    for channel in range(3):
        output[channel][cfa != channel] += 0.1
    result = method_metrics(truth, output, cfa, margin=0)
    assert result["native_sample_rms"] == 0
    assert result["interpolation_only_rms"] == pytest.approx(0.1)
    assert result["interpolation_only_psnr_db"] == pytest.approx(20)


def test_correlation_reports_centered_bias_difference():
    values = np.arange(20, dtype=np.float64)
    shifted = values + 100
    assert normalized_correlation(values, shifted, centered=True) == pytest.approx(1)
    assert normalized_correlation(values, shifted, centered=False) < 0.9


def test_required_synthetic_scene_contract():
    scenes = synthetic_scenes(96)
    assert len(scenes) == 13
    assert {
        "smooth_gradient",
        "correlated_field",
        "vertical_red_green",
        "horizontal_red_green",
        "diagonal_red_green",
        "saturated_red_gray",
        "saturated_blue_gray",
        "saturated_quadrants",
        "isolated_impulse",
        "frequency_sweep",
        "high_frequency_monochrome",
        "periodic_chromatic_texture",
        "support_transition",
    } == set(scenes)
    assert np.asarray(CANONICAL_XTRANS).shape == (6, 6)
    for scene in scenes.values():
        assert scene["truth"].shape == (3, 96, 96)
        assert np.isfinite(scene["truth"]).all()


def test_tracked_corpus_is_canonical_and_complete():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes" / "images" / "xtrans-oracle"
    manifest = corpus / "complementarity.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    from tools.xtrans_alias.analysis import canonical_json_bytes

    assert manifest.read_bytes() == canonical_json_bytes(payload)
    assert payload["format"] == "rawtherapee-xtrans-demosaicer-complementarity-v1"
    assert len(payload["methods"]) == 6
    assert len(payload["synthetic_scenes"]) == 13
    assert len(payload["natural_sources"]) == 4
    for plot in payload["plots"]:
        path = corpus / plot["filename"]
        assert path.stat().st_size == plot["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plot["sha256"]

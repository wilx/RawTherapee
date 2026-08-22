from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from tools.xtrans_alias.analysis import CANONICAL_XTRANS, canonical_json_bytes
from tools.xtrans_danger.analysis import (
    DangerTable,
    FROZEN_RESULTS_SHA256,
    impulse_features,
    load_frozen_blender,
    safety_metrics,
    select_operating_point,
)
from tools.xtrans_danger.dataset import cfa_variants, mosaic_with_cfa, synthetic_cases


EXPECTED_MANIFEST_SHA256 = "a75bb6b94e225c06ac16bedac7e7ec0f16433269322664c58abee58433d99cc5"


def test_impulse_feature_api_has_no_ground_truth():
    assert "truth" not in inspect.signature(impulse_features).parameters


def test_impulse_energy_concentration_and_coherence_distinguish_point_and_line():
    mark = np.zeros((3, 7, 7))
    point = mark.copy()
    point[0, 3, 3] = 1.0
    line = mark.copy()
    line[0, 3, :] = 1.0
    point_features = impulse_features(mark, point)
    line_features = impulse_features(mark, line)
    assert point_features["impulse_chroma_top1_energy_fraction"] > line_features["impulse_chroma_top1_energy_fraction"]
    assert point_features["impulse_isolated_fraction"] > line_features["impulse_isolated_fraction"]
    assert line_features["impulse_mean_neighbor_fraction"] > point_features["impulse_mean_neighbor_fraction"]
    assert line_features["impulse_structure_coherence"] > point_features["impulse_structure_coherence"]
    assert all(np.isfinite(value) for value in point_features.values())


def test_all_18_cfa_variants_are_unique_and_preserve_color_counts():
    variants = cfa_variants()
    assert len(variants) == 18
    assert len({variant.tobytes() for variant in variants}) == 18
    for variant in variants:
        assert sorted(np.bincount(variant.ravel(), minlength=3)) == [8, 8, 20]
        truth = np.stack((
            np.full((12, 12), 0.2),
            np.full((12, 12), 0.4),
            np.full((12, 12), 0.7),
        ))
        scalar, tiled = mosaic_with_cfa(truth, variant)
        for channel, value in enumerate((0.2, 0.4, 0.7)):
            assert np.all(scalar[tiled == channel] == value)


def test_hard_negative_corpus_covers_exact_dot_sizes_and_transitions():
    cases = synthetic_cases()
    by_name = {case["source_id"]: case for case in cases}
    red = np.asarray(by_name["red-dot-2"]["truth"])
    green = np.asarray(by_name["green-dot-3"]["truth"])
    assert np.sum(np.all(red == np.asarray((1.0, 0.0, 0.0))[:, None, None], axis=0)) == 2
    assert np.sum(np.all(green == np.asarray((0.0, 1.0, 0.0))[:, None, None], axis=0)) == 3
    assert {case["cfa_variant"] for case in cases} == set(range(18))
    transitions = [case["transition"] for case in cases if case["transition"]]
    assert transitions == [
        "points:1", "points:3", "points:8", "points:24", "points:72",
        "line:1", "line:2", "line:3", "line:5",
    ]
    assert all(case["split"] == "test" for case in cases if case["transition"])


def test_frozen_blender_identity_and_selection_are_authenticated():
    root = Path(__file__).resolve().parents[3]
    path = root / "devnotes/images/xtrans-hybrid/results.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_RESULTS_SHA256
    blender = load_frozen_blender(path)
    assert len(blender.feature_names) == 44
    assert blender.coefficients.shape == blender.mean.shape == blender.scale.shape
    assert np.all(blender.scale > 0)


def _table() -> DangerTable:
    rows = 10
    mark = np.full(rows, 3.0)
    blend = np.asarray((2.0, 2.2, 2.5, 2.8, 3.1, 3.2, 3.5, 4.0, 5.0, 6.0))
    return DangerTable(
        features=np.arange(rows * 2, dtype=np.float64).reshape(rows, 2),
        feature_names=("a", "b"), feature_groups={"core": ("a",), "impulse": ("b",)},
        mark_sse=mark, mlri_sse=blend + 0.1, fixed_sse=blend + 0.05,
        blend_sse=blend, pixels=np.ones(rows, dtype=np.int32),
        source_ids=np.asarray(["source"] * rows), crop_ids=np.asarray(["crop"] * rows),
        case_indices=np.zeros(rows, dtype=np.int32), bounds=np.zeros((rows, 4), dtype=np.int32),
        splits=np.asarray(["test"] * rows), kinds=np.asarray(["synthetic"] * rows),
    )


def test_oracle_safety_gate_cannot_be_worse_than_either_choice():
    table = _table()
    danger = table.blend_sse > table.mark_sse
    result = safety_metrics(table, danger, severe_tau=0.1)
    assert result["gated"]["mse"] <= result["markesteijn"]["mse"]
    assert result["gated"]["mse"] <= result["ungated_blender"]["mse"]
    assert result["catastrophic_patches_missed"] == 0


def test_operating_point_prioritizes_recall_and_retained_gain_then_cost():
    rows = [
        {"threshold": 0.2, "severe_recall": 0.95, "aggregate_gain_retained_fraction": 0.90, "fallback_fraction": 0.40},
        {"threshold": 0.4, "severe_recall": 0.91, "aggregate_gain_retained_fraction": 0.88, "fallback_fraction": 0.20},
        {"threshold": 0.6, "severe_recall": 0.85, "aggregate_gain_retained_fraction": 0.95, "fallback_fraction": 0.10},
    ]
    assert select_operating_point(rows)["threshold"] == 0.4


def test_tracked_results_are_canonical_and_authenticated():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes/images/xtrans-danger"
    manifest = corpus / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.read_bytes() == canonical_json_bytes(payload)
    for asset in payload["assets"]:
        path = corpus / asset["filename"]
        assert path.stat().st_size == asset["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]

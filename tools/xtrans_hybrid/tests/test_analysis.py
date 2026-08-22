from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from tools.xtrans_alias.analysis import CANONICAL_XTRANS, canonical_json_bytes
from tools.xtrans_hybrid.analysis import (
    build_patch_table,
    feature_indices,
    feature_maps,
    fit_ridge,
    quality_metrics,
)
from tools.xtrans_hybrid.dataset import (
    CROP_SIZE,
    NATURAL_SOURCES,
    crop_coordinates,
    dataset_manifest,
    load_sources,
)


EXPECTED_MANIFEST_SHA256 = "3a59d351c0276301fa7be0b68a79b0292bdab10ce6d598d99f501b53bbe1949f"


def _case() -> dict[str, object]:
    size = 48
    y, x = np.mgrid[:size, :size]
    truth = np.stack((0.2 + x / 100, 0.3 + y / 120, 0.25 + (x + y) / 180))
    cfa = np.tile(CANONICAL_XTRANS, (8, 8))
    mosaic = np.take_along_axis(np.moveaxis(truth, 0, -1), cfa[..., None], axis=2)[..., 0]
    mark = truth + 0.02 * np.sin(x / 3)[None]
    mlri = truth + 0.015 * np.cos(y / 4)[None]
    for channel in range(3):
        mark[channel][cfa == channel] = mosaic[cfa == channel]
        mlri[channel][cfa == channel] = mosaic[cfa == channel]
    return {
        "cfa": cfa, "crop_id": "case", "markesteijn": mark,
        "mlri": mlri, "mosaic": mosaic, "source_id": "source",
        "truth": truth,
    }


def test_feature_api_cannot_receive_ground_truth():
    assert "truth" not in inspect.signature(feature_maps).parameters


def test_feature_groups_are_compact_finite_and_native_samples_are_zero():
    case = _case()
    maps, groups = feature_maps(
        case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"],
        window=7,
    )
    assert 25 <= len(maps) <= 60
    assert set(groups) == set("ABCDEFG")
    assert all(np.isfinite(values).all() for values in maps.values())
    assert np.max(np.abs(maps["mark_native_sample_rms"])) < 1e-14
    assert np.max(np.abs(maps["mlri_native_sample_rms"])) < 1e-14


def test_patch_oracle_and_convex_oracle_are_consistent():
    table = build_patch_table((_case(),), 7)
    mark = quality_metrics(table, np.zeros(table.features.shape[0]))
    hard = quality_metrics(table, table.labels)
    convex = quality_metrics(table, table.oracle_alpha)
    assert hard["mse"] <= mark["mse"]
    assert convex["mse"] <= hard["mse"] + 1e-15
    assert np.all((table.oracle_alpha >= 0) & (table.oracle_alpha <= 1))


def test_ridge_normalization_is_fitted_only_from_supplied_training_rows():
    table = build_patch_table((_case(),), 7)
    indices = feature_indices(table, ("A", "D"))
    model = fit_ridge(
        table.features, table.oracle_alpha, table.quadratic, indices, 0.01
    )
    assert np.array_equal(
        model.normalizer.mean,
        np.round(table.features[:, indices].mean(axis=0), decimals=12),
    )
    prediction = model.alpha(table.features)
    assert np.all((prediction >= 0) & (prediction <= 1))


def test_dataset_split_is_source_level_and_stereo_views_stay_together():
    sources = load_sources()
    assert len(sources) == 20
    assert {row["split"] for row in sources} == {"train", "validation", "test"}
    assert sum(row["split"] == "train" for row in sources) == 12
    assert sum(row["split"] == "validation" for row in sources) == 4
    assert sum(row["split"] == "test" for row in sources) == 4
    motorcycles = [row for row in sources if row["group"] == "middlebury-motorcycle"]
    assert len(motorcycles) == 2
    assert {row["split"] for row in motorcycles} == {"validation"}
    assert all(len(row["crops"]) == 3 for row in sources)
    assert dataset_manifest(sources)["crop_count"] == 60
    assert dataset_manifest(sources)["source_split_counts"] == {
        "train": 12, "validation": 4, "test": 4,
    }


def test_crop_coordinates_are_deterministic_and_in_range():
    digest = "01" * 32
    first = crop_coordinates(512, 300, digest)
    second = crop_coordinates(512, 300, digest)
    assert first == second
    assert len(set(first)) == 3
    for x, y in first:
        assert 0 <= x <= 512 - CROP_SIZE
        assert 0 <= y <= 300 - CROP_SIZE


def test_tracked_corpus_is_canonical_and_authenticated():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes" / "images" / "xtrans-hybrid"
    manifest = corpus / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.read_bytes() == canonical_json_bytes(payload)
    for asset in payload["assets"]:
        path = corpus / asset["filename"]
        assert path.stat().st_size == asset["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]

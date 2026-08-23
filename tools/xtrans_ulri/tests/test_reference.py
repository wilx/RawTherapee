from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.xtrans_ulri.generate_reference import CFA, mosaic, scenes


GOLDEN = Path(__file__).parents[1] / "reference_golden"
MANIFEST_SHA256 = "2441cb88637a5c50375093d11e97b6cb46912d68201b338c3f8d4a5c4e7114c3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_scenes_are_finite_normalized_rgb():
    required = {
        "constant-gray", "constant-red", "constant-green", "constant-blue",
        "horizontal-gradient", "vertical-gradient", "diagonal-gradient",
        "low-frequency-sinusoid", "white-impulse", "isolated-red-point",
        "vertical-chromatic-edge",
    }
    values = scenes()
    assert set(values) == required
    for truth in values.values():
        assert truth.shape == (3, 48, 48)
        assert np.isfinite(truth).all()
        assert np.min(truth) >= 0.0
        assert np.max(truth) <= 1.0


def test_mosaic_uses_only_observed_cfa_channel():
    truth = scenes()["horizontal-gradient"]
    scalar = mosaic(truth)
    for y in range(scalar.shape[0]):
        for x in range(scalar.shape[1]):
            assert scalar[y, x] == pytest.approx(truth[CFA[y % 6, x % 6], y, x])


def test_committed_reference_corpus_identities():
    manifest_path = GOLDEN / "manifest.json"
    assert _sha256(manifest_path) == MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "rawtherapee-xtrans-ulri-reference-v1"
    assert manifest["cfa"] == CFA.tolist()
    assert len(manifest["cases"]) == 11
    for case in manifest["cases"]:
        artifacts = [case["truth"], case["mosaic"], *case["outputs"]]
        for artifact in artifacts:
            path = GOLDEN / artifact["file"]
            assert path.stat().st_size == artifact["bytes"]
            assert _sha256(path) == artifact["sha256"]

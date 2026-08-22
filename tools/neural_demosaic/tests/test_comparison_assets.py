from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[3] / "devnotes" / "images" / "xtrans-neural" / "DSCF0771"


def test_tracked_comparison_assets_match_manifest():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes().endswith(b"\n")
    assert manifest["format"] == "rawtherapee-xtrans-comparison-assets-v1"
    assert len(manifest["assets"]) >= 6
    for entry in manifest["assets"]:
        path = ROOT / entry["filename"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        with Image.open(path) as image:
            assert [image.height, image.width, 3] == entry["shape"]
            assert hashlib.sha256(image.info["icc_profile"]).hexdigest() == manifest["icc_profile"]["sha256"]


def test_triangulation_comparison_assets_match_manifest():
    manifest_path = ROOT / "triangulation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes().endswith(b"\n")
    assert manifest["format"] == "rawtherapee-xtrans-triangulation-comparison-v1"
    assert manifest["comparison_order"] == [
        "xtrans-triangulated-rgb",
        "xtrans-triangulated-chroma",
        "3-pass (Markesteijn)",
    ]
    assert len(manifest["assets"]) == 8
    for entry in manifest["assets"]:
        path = ROOT / entry["filename"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        with Image.open(path) as image:
            assert [image.height, image.width, 3] == entry["shape"]
            assert (
                hashlib.sha256(image.info["icc_profile"]).hexdigest()
                == manifest["metadata"]["icc_profile_sha256"]
            )

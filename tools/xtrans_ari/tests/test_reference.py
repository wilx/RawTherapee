import hashlib
import json
from pathlib import Path

import numpy as np

from tools.xtrans_ari.bayer_reference import bayer_masks, green_ari, mosaic_bayer
from tools.xtrans_ari.generate_reference import scenes


GOLDEN = Path(__file__).parents[1] / "reference_golden"
MANIFEST_SHA256 = "0b45395709c1b0be6527970443b158cb1d17a72b69548c8d5ab6a9d83a6a4ded"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_manifest_and_payloads_are_frozen():
    manifest_path = GOLDEN / "manifest.json"
    assert not (GOLDEN / "octave-reference").exists()
    assert sum(path.is_file() for path in GOLDEN.iterdir()) == 96
    assert _sha256(manifest_path) == MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "rawtherapee-ari-bayer-reference-v1"
    assert len(manifest["cases"]) == 19
    assert manifest["archive"]["version"] == "1.0"
    assert manifest["license"] == "research purpose only; all rights reserved"
    assert manifest["independent_green_parity"]["maximum_abs"] < 7e-6
    assert manifest["independent_green_parity"]["interior_maximum_abs"] < 4e-8
    for case in manifest["cases"]:
        for field in ("truth", "mosaic", "official_green", "official_output", "independent_green"):
            identity = case[field]
            assert identity is not None
            path = GOLDEN / identity["file"]
            assert path.stat().st_size == identity["bytes"]
            assert _sha256(path) == identity["sha256"]


def test_independent_green_reproduces_every_official_fixture():
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    for case in manifest["cases"]:
        height, width = case["height"], case["width"]
        mosaic = np.fromfile(GOLDEN / case["mosaic"]["file"], dtype="<f4").reshape(3, height, width)
        official = np.fromfile(
            GOLDEN / case["official_green"]["file"], dtype="<f4"
        ).reshape(height, width)
        actual = green_ari(
            mosaic, bayer_masks(height, width, case["pattern"]), case["pattern"]
        ).adaptive
        difference = np.abs(actual - official)
        assert np.max(difference[16:-16, 16:-16]) < 4e-8
        assert np.max(difference) < 7e-6


def test_reference_mask_typo_is_isolated_and_controlled():
    case = next(
        row for row in json.loads((GOLDEN / "manifest.json").read_text())["cases"]
        if row["pattern"] == "grbg" and row["name"] == "periodic-chromatic"
    )
    mosaic = np.fromfile(GOLDEN / case["mosaic"]["file"], dtype="<f4").reshape(3, 48, 48)
    masks = bayer_masks(48, 48, "grbg")
    preserved = green_ari(mosaic, masks, "grbg").adaptive
    corrected = green_ari(
        mosaic, masks, "grbg", preserve_reference_vertical_mask_typo=False
    ).adaptive
    official = np.fromfile(
        GOLDEN / case["official_green"]["file"], dtype="<f4"
    ).reshape(48, 48)
    assert np.max(np.abs(preserved - official)) < 4e-8
    assert np.max(np.abs(corrected - official)) > 0.5


def test_bayer_constants_and_samples_are_preserved():
    truth = scenes(40)["constant-gray"]
    for pattern in ("grbg", "rggb", "gbrg", "bggr"):
        mosaic, masks = mosaic_bayer(truth, pattern)
        result = green_ari(mosaic, masks, pattern)
        assert np.max(np.abs(result.adaptive - truth[1])) < 1e-8
        assert np.array_equal(result.adaptive[masks[1] != 0], truth[1][masks[1] != 0])

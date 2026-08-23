"""Frozen population and external-control bindings for the MIX3 experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.xtrans_hybrid.dataset import load_sources
from tools.xtrans_lmmse.dataset import (
    BSDS_MIRROR_REVISION,
    BSDS_MIRROR_URL,
    BSDS_OFFICIAL_PAGE,
    BSDS_TERMS,
    load_rgb,
    sha256_file,
)
from tools.xtrans_hybrid.dataset import srgb_decode


TRAIN_COUNTS = (100, 150, 200)
VALIDATION_COUNT = 20
TEST_COUNT = 20
EXTERNAL_CHROMATIC_IDS = (
    "chelsea",
    "coffee",
    "ihc",
    "motorcycle-left",
    "retina",
    "rocket",
)


def _row(path: Path, split: str) -> dict[str, object]:
    digest = sha256_file(path)
    with Image.open(path) as image:
        width, height = image.size
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = np.moveaxis(srgb_decode(encoded), -1, 0)
    opponent = np.stack(
        (
            (linear[0] - linear[2]) / np.sqrt(2.0),
            (linear[0] - 2.0 * linear[1] + linear[2]) / np.sqrt(6.0),
        ),
        axis=0,
    )
    chroma_rms = float(np.sqrt(np.mean(opponent * opponent)))
    return {
        "chroma_rms": chroma_rms,
        "filename": path.name,
        "height": height,
        "path": path,
        "sha256": digest,
        "split": split,
        "true_chromatic": chroma_rms >= 0.005,
        "width": width,
    }


def discover_expanded(root: Path) -> dict[str, list[dict[str, object]]]:
    base = root.resolve() / "BSDS500" / "data" / "images"
    counts = {"train": max(TRAIN_COUNTS), "val": VALIDATION_COUNT, "test": TEST_COUNT}
    result = {}
    for split, count in counts.items():
        paths = sorted((base / split).glob("*.jpg"))
        if len(paths) < count:
            raise RuntimeError(f"BSDS500 {split} has only {len(paths)} images")
        result[split] = [_row(path, split) for path in paths[:count]]
    return result


def external_chromatic_sources() -> list[dict[str, object]]:
    by_id = {str(source["id"]): source for source in load_sources()}
    result = []
    for source_id in EXTERNAL_CHROMATIC_IDS:
        if source_id not in by_id:
            raise RuntimeError(f"missing pinned chromatic source {source_id}")
        result.append(by_id[source_id])
    return result


def manifest(
    splits: dict[str, list[dict[str, object]]],
    external: list[dict[str, object]],
) -> dict[str, object]:
    bsds_rows = []
    for split in ("train", "val", "test"):
        for source in splits[split]:
            bsds_rows.append({
                key: source[key]
                for key in (
                    "chroma_rms", "filename", "height", "sha256", "split",
                    "true_chromatic", "width",
                )
            })
    external_rows = []
    for source in external:
        external_rows.append({
            key: source[key]
            for key in (
                "category", "crops", "filename", "group", "height", "id",
                "license", "sha256", "width",
            )
        })
    return {
        "bsds": {
            "format": "rawtherapee-xtrans-lmmse-mixture-bsds500-v1",
            "linearization": "JPEG values inverse-EOTF decoded as IEC 61966-2-1 sRGB",
            "mirror_revision": BSDS_MIRROR_REVISION,
            "mirror_url": BSDS_MIRROR_URL,
            "official_page": BSDS_OFFICIAL_PAGE,
            "selection": "first sorted 200 train, 20 validation, and 20 test sources",
            "sources": bsds_rows,
            "terms": BSDS_TERMS,
        },
        "external_chromatic": {
            "format": "rawtherapee-xtrans-lmmse-mixture-skimage-controls-v1",
            "note": "six source-held-out pinned scikit-image 0.26.0 controls; never used for fitting or threshold selection",
            "sources": external_rows,
        },
    }


__all__ = (
    "EXTERNAL_CHROMATIC_IDS",
    "TRAIN_COUNTS",
    "discover_expanded",
    "external_chromatic_sources",
    "load_rgb",
    "manifest",
)

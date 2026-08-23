"""Authenticated source-level BSDS500 split for population covariance fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from tools.xtrans_hybrid.dataset import srgb_decode


BSDS_MIRROR_REVISION = "a04b7c6c3a9f0ace74bf205c72a43d32e1c72722"
BSDS_MIRROR_URL = "https://github.com/BIDS/BSDS500.git"
BSDS_OFFICIAL_PAGE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/"
BSDS_TERMS = "non-commercial research and educational use; cite Martin et al., ICCV 2001"
SPLIT_COUNTS = {"train": 100, "val": 20, "test": 20}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path) -> dict[str, list[dict[str, object]]]:
    base = root.resolve() / "BSDS500" / "data" / "images"
    result = {}
    for split, count in SPLIT_COUNTS.items():
        candidates = sorted((base / split).glob("*.jpg"))
        if len(candidates) < count:
            raise RuntimeError(f"BSDS500 {split} has only {len(candidates)} images")
        rows = []
        for path in candidates[:count]:
            digest = sha256_file(path)
            with Image.open(path) as image:
                width, height = image.size
                if image.mode not in ("RGB", "L"):
                    raise RuntimeError(f"unexpected BSDS mode {image.mode}: {path}")
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
            rows.append({
                "chroma_rms": chroma_rms,
                "filename": path.name,
                "height": height,
                "path": path,
                "sha256": digest,
                "split": split,
                "true_chromatic": chroma_rms >= 0.005,
                "width": width,
            })
        result[split] = rows
    return result


def load_rgb(row: dict[str, object]) -> np.ndarray:
    path = Path(row["path"])
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"BSDS source changed: {path}")
    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return np.moveaxis(srgb_decode(encoded), -1, 0)


def manifest(splits: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    rows = []
    for split in ("train", "val", "test"):
        for source in splits[split]:
            rows.append({
                key: source[key]
                for key in (
                    "chroma_rms", "filename", "height", "sha256", "split",
                    "true_chromatic", "width",
                )
            })
    return {
        "format": "rawtherapee-xtrans-lmmse-bsds500-corpus-v1",
        "image_count": len(rows),
        "linearization": "JPEG code values treated as IEC 61966-2-1 sRGB and inverse-EOTF decoded",
        "mirror_revision": BSDS_MIRROR_REVISION,
        "mirror_url": BSDS_MIRROR_URL,
        "official_page": BSDS_OFFICIAL_PAGE,
        "selection": "lexicographically first 100 train, 20 val, and 20 test filenames",
        "sources": rows,
        "split_counts": {
            split: len(splits[split]) for split in ("train", "val", "test")
        },
        "terms": BSDS_TERMS,
        "true_chromatic_fraction": float(
            np.mean([bool(row["true_chromatic"]) for row in rows])
        ),
    }

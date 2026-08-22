"""Leakage-safe natural and deterministic hard-negative safety corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tools.xtrans_alias.analysis import CANONICAL_XTRANS
from tools.xtrans_hybrid.dataset import CROP_SIZE, crop_coordinates, iter_crops, load_sources, srgb_decode


STARFIELD_BINDING = {
    "id": "nasa-hydra-starfield",
    "filename": "grail_free_air_stars1.tif",
    "source_url": "https://svs.gsfc.nasa.gov/vis/a000000/a004000/a004041/grail_free_air_stars1.tif",
    "landing_page": "https://svs.gsfc.nasa.gov/4041/",
    "credit": "NASA's Goddard Space Flight Center Scientific Visualization Studio",
    "license": "NASA media; use subject to NASA reproduction guidelines",
    "sha256": "bc5ae5f68fc1977545c0d16c1c60c20dafb4fdbeb045f55774e7a5d03498f6d0",
    "width": 3000,
    "height": 3000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cfa_variants() -> tuple[np.ndarray, ...]:
    """Return all unique X-Trans phase/orientation cells deterministically."""

    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for reflection in (False, True):
        reflected = np.fliplr(CANONICAL_XTRANS) if reflection else CANONICAL_XTRANS
        for rotation in range(4):
            oriented = np.rot90(reflected, rotation)
            for y in range(6):
                for x in range(6):
                    candidate = np.roll(oriented, (y, x), axis=(0, 1)).astype(np.uint8)
                    key = candidate.tobytes()
                    if key not in seen:
                        seen.add(key)
                        variants.append(candidate)
    if len(variants) != 18:
        raise RuntimeError(f"expected 18 X-Trans variants, found {len(variants)}")
    return tuple(variants)


def tiled_cfa(cell: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.tile(np.asarray(cell, dtype=np.uint8), (
        (height + 5) // 6, (width + 5) // 6,
    ))[:height, :width]


def mosaic_with_cfa(truth: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(truth, dtype=np.float64)
    cfa = tiled_cfa(cell, checked.shape[2], checked.shape[1])
    scalar = np.take_along_axis(np.moveaxis(checked, 0, -1), cfa[..., None], axis=2)[..., 0]
    return scalar, cfa


def load_natural_cases(starfield_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    sources = load_sources()
    cases = [dict(case, kind="natural") for case in iter_crops(sources)]
    path = starfield_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != STARFIELD_BINDING["sha256"]:
        raise ValueError(f"NASA starfield digest mismatch: {digest}")
    with Image.open(path) as image:
        if image.size != (STARFIELD_BINDING["width"], STARFIELD_BINDING["height"]):
            raise ValueError(f"NASA starfield dimensions changed: {image.size}")
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = np.moveaxis(srgb_decode(encoded), -1, 0)
    coordinates = crop_coordinates(
        int(STARFIELD_BINDING["width"]), int(STARFIELD_BINDING["height"]), digest
    )
    for index, (x, y) in enumerate(coordinates):
        cases.append({
            "category": "independent catalog-derived sparse star field",
            "crop": [x, y, CROP_SIZE, CROP_SIZE],
            "crop_id": f"nasa-hydra-starfield-{index}",
            "group": "nasa-hydra-starfield",
            "source_id": "nasa-hydra-starfield",
            "split": "test",
            "kind": "natural",
            "truth": linear[:, y:y + CROP_SIZE, x:x + CROP_SIZE].copy(),
        })
    return cases, {
        **STARFIELD_BINDING,
        "crops": [list(coordinate) for coordinate in coordinates],
        "crop_size": CROP_SIZE,
    }


def _background(size: int, brightness: float, color: tuple[float, float, float]) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    gradient = 0.015 * (x + 0.7 * y) / max(1, 2 * size - 2)
    result = np.empty((3, size, size), dtype=np.float64)
    for channel, value in enumerate(color):
        result[channel] = np.clip(brightness * value + gradient, 0, 1)
    return result


def _dot(image: np.ndarray, y: int, x: int, color: tuple[float, float, float], radius: int) -> None:
    yy, xx = np.mgrid[:image.shape[1], :image.shape[2]]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
    image[:, mask] = np.asarray(color)[:, None]


def _dot_pixels(
    image: np.ndarray,
    y: int,
    x: int,
    color: tuple[float, float, float],
    count: int,
) -> None:
    offsets = ((0, 0), (0, 1), (1, 0))
    if count not in (1, 2, 3):
        raise ValueError("dot pixel count must be one, two, or three")
    for dy, dx in offsets[:count]:
        image[:, y + dy, x + dx] = color


def _line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[float, float, float],
    thickness: int,
) -> None:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    ys = np.rint(np.linspace(start[0], end[0], steps)).astype(int)
    xs = np.rint(np.linspace(start[1], end[1], steps)).astype(int)
    for y, x in zip(ys, xs):
        _dot(image, int(y), int(x), color, max(0, thickness - 1))


def _case(
    name: str,
    split: str,
    family: str,
    image: np.ndarray,
    cfa_index: int,
    transition: str | None = None,
) -> dict[str, object]:
    return {
        "category": family,
        "crop_id": name,
        "group": name,
        "source_id": name,
        "split": split,
        "kind": "synthetic",
        "truth": np.clip(image, 0, 1),
        "cfa_cell": cfa_variants()[cfa_index % 18],
        "cfa_variant": cfa_index % 18,
        "transition": transition,
    }


def synthetic_cases(size: int = 72) -> list[dict[str, object]]:
    """Generate hard negatives and untouched coherence transitions."""

    variants = cfa_variants()
    del variants  # Assert the 18-cell contract before constructing cases.
    cases: list[dict[str, object]] = []
    colors = ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    # Every CFA representation sees a CFA-relative subpixel impulse.  Parameter
    # sets, rather than CFA geometry, are split so orientation itself is not a
    # source identity shortcut.
    for index in range(18):
        split = "train" if index < 12 else "validation" if index < 15 else "test"
        image = _background(size, 0.015 + 0.01 * (index % 3), (0.9, 1.0, 0.85))
        color = colors[index % len(colors)]
        _dot(image, size // 2 + index % 3 - 1, size // 2 + (index * 5) % 5 - 2, color, index % 3)
        cases.append(_case(
            f"cfa-impulse-{index:02d}", split, "isolated colored point across CFA variants",
            image, index,
        ))

    definitions = (
        ("white-impulse", "train", "isolated white impulse", 1),
        ("red-dot-2", "train", "two-pixel saturated red dot", 2),
        ("green-dot-3", "train", "three-pixel saturated green dot", 3),
        ("blue-dark-point", "train", "colored point on dark background", 4),
        ("tiny-specular", "train", "tiny saturated specular highlight", 5),
        ("thin-red-line", "train", "thin colored line", 6),
        ("line-intersection", "train", "intersection of thin lines", 7),
        ("sparse-microtexture", "train", "sparse bright microtexture", 8),
        ("medium-points", "train", "progressively denser point pattern", 9),
        ("point-next-edge", "train", "tiny colored structure adjacent to edge", 10),
        ("star-field-a", "train", "sparse star field", 11),
        ("star-field-b", "validation", "sparse star field", 12),
        ("dim-blue-dot", "validation", "low-intensity colored point", 13),
        ("bright-edge-point", "validation", "tiny saturated structure adjacent to edge", 14),
        ("dense-microtexture", "validation", "dense bright microtexture", 15),
    )
    for serial, (name, split, family, cfa_index) in enumerate(definitions):
        brightness = 0.01 if "dark" in name or "star" in name else 0.12 + 0.04 * (serial % 3)
        image = _background(size, brightness, (0.9, 1.0, 0.8))
        center = size // 2
        if "line-intersection" in name:
            _line(image, (12, 10), (size - 13, size - 11), (1.0, 0.1, 0.1), 1)
            _line(image, (12, size - 11), (size - 13, 10), (0.1, 0.4, 1.0), 1)
        elif "line" in name:
            _line(image, (9, 12), (size - 10, size - 16), (1.0, 0.05, 0.1), 1)
        elif "edge" in name:
            image[:, :, center:] += np.asarray((0.25, 0.18, 0.12))[:, None, None]
            _dot(image, center, center - 2, (0.0, 0.2, 1.0), 1)
        elif "microtexture" in name or "points" in name or "star-field" in name:
            seed = 0x58445200 + serial
            rng = np.random.default_rng(seed)
            count = 8 if "sparse" in name or "star" in name else 48 if "dense" in name else 24
            for point in range(count):
                y = int(rng.integers(8, size - 8))
                x = int(rng.integers(8, size - 8))
                color = colors[int(rng.integers(0, len(colors)))]
                intensity = 0.35 + 0.65 * float(rng.random())
                _dot(image, y, x, tuple(intensity * value for value in color), 0 if point % 4 else 1)
        else:
            color = (1.0, 1.0, 1.0)
            if "red" in name:
                color = (1.0, 0.0, 0.0)
            elif "green" in name:
                color = (0.0, 1.0, 0.0)
            elif "blue" in name:
                color = (0.0, 0.0, 1.0)
            count = 1
            if name == "red-dot-2":
                count = 2
            elif name == "green-dot-3":
                count = 3
            _dot_pixels(image, center, center + serial % 3 - 1, color, count)
        cases.append(_case(name, split, family, image, cfa_index))

    # Untouched test-only interpolation paths: isolated points become texture,
    # and single-pixel lines become coherent edges.  These are never used for
    # model or operating-threshold selection.
    for count in (1, 3, 8, 24, 72):
        image = _background(size, 0.02, (1.0, 0.9, 0.8))
        rng = np.random.default_rng(0x54524100 + count)
        for point in range(count):
            y = int(rng.integers(8, size - 8))
            x = int(rng.integers(8, size - 8))
            _dot(image, y, x, colors[1 + point % 3], 0)
        cases.append(_case(
            f"transition-points-{count:02d}", "test", "point-density transition",
            image, count, f"points:{count}",
        ))
    for thickness in (1, 2, 3, 5):
        image = _background(size, 0.08, (0.9, 1.0, 0.85))
        _line(image, (9, 11), (size - 10, size - 15), (1.0, 0.05, 0.1), thickness)
        cases.append(_case(
            f"transition-line-{thickness}", "test", "line-thickness transition",
            image, 13 + thickness, f"line:{thickness}",
        ))
    return cases


def safety_dataset_manifest(
    natural_cases: Iterable[dict[str, object]],
    synthetic: Iterable[dict[str, object]],
    starfield: dict[str, object],
) -> dict[str, object]:
    natural = list(natural_cases)
    synthetic_rows = list(synthetic)
    return {
        "format": "rawtherapee-xtrans-danger-dataset-v1",
        "frozen_blender_source": "devnotes/images/xtrans-hybrid/results.json",
        "hard_negative_synthetic_cases": [
            {
                "id": row["source_id"], "category": row["category"],
                "split": row["split"], "cfa_variant": row["cfa_variant"],
                "transition": row["transition"],
            }
            for row in synthetic_rows
        ],
        "natural_case_count": len(natural),
        "natural_source_splits": {
            split: sorted({str(row["source_id"]) for row in natural if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "starfield": starfield,
        "test_sources_never_used_for_detector_fitting_or_threshold_selection": [
            "astronaut", "brick", "hubble", "page", "nasa-hydra-starfield",
        ],
        "synthetic_case_count": len(synthetic_rows),
        "xtrans_cfa_variant_count": len(cfa_variants()),
    }

"""Authenticated natural and deterministic diagnostic recurrence scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tools.xtrans_danger.dataset import STARFIELD_BINDING, sha256_file
from tools.xtrans_hybrid.dataset import load_sources, srgb_decode


TARGET_SIZE = 168
CONTEXT_HALO = 78
CONTEXT_SIZE = TARGET_SIZE + 2 * CONTEXT_HALO

SOURCE_SELECTION = (
    ("astronaut", "mixed natural detail"),
    ("brick", "repeated texture"),
    ("chelsea", "foliage-like natural texture"),
    ("grass", "foliage-like natural texture"),
    ("gravel", "irregular natural texture"),
    ("hubble", "sparse colored point field"),
    ("page", "repeated text and edges"),
    ("rocket", "mixed natural detail"),
)


def _context_case(source: dict[str, object], scene_class: str) -> dict[str, object]:
    rgb = np.asarray(source["rgb"], dtype=np.float64)
    height, width = rgb.shape[1:]
    x = (width - TARGET_SIZE) // 2
    y = (height - TARGET_SIZE) // 2
    x0, y0 = x - CONTEXT_HALO, y - CONTEXT_HALO
    padded = np.pad(
        rgb,
        ((0, 0), (CONTEXT_HALO, CONTEXT_HALO), (CONTEXT_HALO, CONTEXT_HALO)),
        mode="reflect",
    )
    context = padded[:, y:y + CONTEXT_SIZE, x:x + CONTEXT_SIZE].copy()
    if context.shape != (3, CONTEXT_SIZE, CONTEXT_SIZE):
        raise ValueError(f"source {source['id']} has incomplete recurrence context")
    return {
        "id": str(source["id"]),
        "kind": "natural",
        "scene_class": scene_class,
        "split": "safety" if source["id"] == "hubble" else "analysis",
        "context": context,
        "context_global_origin": (x0, y0),
        "target_origin": (CONTEXT_HALO, CONTEXT_HALO),
        "source": {
            key: source[key]
            for key in ("category", "filename", "height", "id", "license", "sha256", "width")
        },
    }


def _starfield_case(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if digest != STARFIELD_BINDING["sha256"]:
        raise ValueError(f"NASA starfield digest mismatch: {digest}")
    with Image.open(resolved) as image:
        if image.size != (STARFIELD_BINDING["width"], STARFIELD_BINDING["height"]):
            raise ValueError(f"NASA starfield dimensions changed: {image.size}")
        x = (image.width - TARGET_SIZE) // 2
        y = (image.height - TARGET_SIZE) // 2
        x0, y0 = x - CONTEXT_HALO, y - CONTEXT_HALO
        encoded = np.asarray(
            image.crop((x0, y0, x0 + CONTEXT_SIZE, y0 + CONTEXT_SIZE)).convert("RGB"),
            dtype=np.float64,
        ) / 255.0
    context = np.moveaxis(srgb_decode(encoded), -1, 0)
    case = {
        "id": "nasa-hydra-starfield",
        "kind": "natural",
        "scene_class": "sparse colored point field",
        "split": "safety",
        "context": context,
        "context_global_origin": (x0, y0),
        "target_origin": (CONTEXT_HALO, CONTEXT_HALO),
        "source": dict(STARFIELD_BINDING),
    }
    binding = dict(STARFIELD_BINDING)
    binding["linear_context"] = [x0, y0, CONTEXT_SIZE, CONTEXT_SIZE]
    return case, binding


def _synthetic_case(name: str) -> dict[str, object]:
    size = CONTEXT_SIZE
    y, x = np.mgrid[:size, :size].astype(np.float64)
    if name == "saturated-edge":
        # Several translated copies of the same chromatic edge create genuine
        # nonlocal support while retaining subpixel-scale saturated details.
        stripe = ((x + 0.63 * y) % 54.0) >= 27.0
        truth = np.stack((
            np.where(stripe, 1.0, 0.08),
            np.where(stripe, 0.12, 0.72),
            np.where(stripe, 0.06, 0.18),
        ))
        scene_class = "saturated repeated edge"
    elif name == "frequency-sweep":
        phase = 2 * np.pi * (x * (0.01 + 0.44 * y / (size - 1)) + 0.17 * y)
        truth = np.stack((
            0.5 + 0.45 * np.sin(phase),
            0.5 + 0.45 * np.sin(phase + 2 * np.pi / 3),
            0.5 + 0.45 * np.sin(phase + 4 * np.pi / 3),
        ))
        scene_class = "frequency sweep"
    else:
        raise ValueError(name)
    return {
        "id": name,
        "kind": "synthetic",
        "scene_class": scene_class,
        "split": "analysis",
        "context": np.clip(truth, 0, 1),
        "context_global_origin": (0, 0),
        "target_origin": (CONTEXT_HALO, CONTEXT_HALO),
        "source": {
            "id": name,
            "license": "generated deterministically by this experiment",
            "width": size,
            "height": size,
        },
    }


def load_cases(starfield_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = dict(SOURCE_SELECTION)
    cases = [
        _context_case(source, selected[str(source["id"])])
        for source in load_sources()
        if source["id"] in selected
    ]
    if len(cases) != len(SOURCE_SELECTION):
        raise RuntimeError("not all pinned recurrence sources were loaded")
    starfield, binding = _starfield_case(starfield_path)
    cases.extend((starfield, _synthetic_case("saturated-edge"), _synthetic_case("frequency-sweep")))
    return cases, binding


def dataset_manifest(cases: Iterable[dict[str, object]], starfield: dict[str, object]) -> dict[str, object]:
    rows = []
    for case in cases:
        source = dict(case["source"])
        source.pop("source_url", None)
        rows.append({
            "id": case["id"],
            "kind": case["kind"],
            "scene_class": case["scene_class"],
            "split": case["split"],
            "source": source,
            "context_global_origin": list(case["context_global_origin"]),
            "target_origin": list(case["target_origin"]),
        })
    return {
        "format": "rawtherapee-xtrans-nonlocal-dataset-v1",
        "linearization": "IEC 61966-2-1 sRGB inverse EOTF",
        "context_shape_chw": [3, CONTEXT_SIZE, CONTEXT_SIZE],
        "target_shape_chw": [3, TARGET_SIZE, TARGET_SIZE],
        "context_halo": CONTEXT_HALO,
        "search_is_within_same_source_context": True,
        "hubble_and_hydra_are_untuned_safety_sources": True,
        "starfield_binding": starfield,
        "cases": rows,
    }

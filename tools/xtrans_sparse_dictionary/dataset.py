"""Frozen corpus bindings and deterministic RGB patch sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from tools.xtrans_lmmse.dataset import load_rgb
from tools.xtrans_lmmse_mixture.dataset import (
    discover_expanded,
    external_chromatic_sources,
    manifest as lmmse_dataset_manifest,
)
from tools.xtrans_lmmse.experiment import _load_experiment_sources

from .model import observation_indices, phase_index


@dataclass(frozen=True)
class PatchSample:
    source_id: str
    group: str
    vector: np.ndarray
    indices: np.ndarray
    phase: int
    x: int
    y: int


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def sample_rgb_patches(
    rgb: np.ndarray,
    source_id: str,
    group: str,
    patch_size: int,
    count: int,
    *,
    region: tuple[int, int, int, int] | None = None,
) -> list[PatchSample]:
    checked = np.asarray(rgb, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3 or not np.isfinite(checked).all():
        raise ValueError("RGB source must be finite CHW")
    height, width = checked.shape[1:]
    if region is None:
        region = (0, 0, width, height)
    x0, y0, x1, y1 = region
    maximum_x = x1 - patch_size
    maximum_y = y1 - patch_size
    if x0 < 0 or y0 < 0 or maximum_x < x0 or maximum_y < y0 or x1 > width or y1 > height:
        raise ValueError("source is too small for requested patch region")
    candidates = (maximum_x - x0 + 1) * (maximum_y - y0 + 1)
    if count > candidates:
        raise ValueError("too many patches requested")
    rng = np.random.default_rng(_seed(f"{source_id}:{group}:{patch_size}:{count}:{region}"))
    flat = rng.choice(candidates, size=count, replace=False)
    span_x = maximum_x - x0 + 1
    result = []
    for value in flat:
        x = x0 + int(value % span_x)
        y = y0 + int(value // span_x)
        patch = np.ascontiguousarray(checked[:, y:y + patch_size, x:x + patch_size])
        indices = observation_indices(patch_size, x % 6, y % 6)
        result.append(PatchSample(
            source_id=source_id,
            group=group,
            vector=patch.reshape(-1),
            indices=indices,
            phase=phase_index(x % 6, y % 6),
            x=x,
            y=y,
        ))
    return result


def sample_bright_patches(
    rgb: np.ndarray,
    source_id: str,
    patch_size: int,
    count: int,
) -> list[PatchSample]:
    """Select deterministic separated bright targets for sparse-star safety."""

    checked = np.asarray(rgb, dtype=np.float64)
    height, width = checked.shape[1:]
    radius = patch_size // 2
    luminance = np.mean(checked, axis=0)
    valid = np.zeros((height, width), dtype=bool)
    valid[radius:height - radius, radius:width - radius] = True
    ranked = np.argsort(luminance.ravel(), kind="stable")[::-1]
    centers: list[tuple[int, int]] = []
    minimum_distance = max(2, patch_size // 2)
    for flat in ranked:
        y, x = divmod(int(flat), width)
        if not valid[y, x]:
            continue
        if any(
            abs(x - previous_x) <= minimum_distance
            and abs(y - previous_y) <= minimum_distance
            for previous_x, previous_y in centers
        ):
            continue
        centers.append((x, y))
        if len(centers) == count:
            break
    if len(centers) != count:
        raise RuntimeError(f"could not select {count} bright patches from {source_id}")
    result = []
    for x, y in centers:
        left = x - radius
        top = y - radius
        patch = np.ascontiguousarray(
            checked[:, top:top + patch_size, left:left + patch_size]
        )
        indices = observation_indices(patch_size, left % 6, top % 6)
        result.append(PatchSample(
            source_id=f"{source_id}-bright",
            group="sparse-bright-targets",
            vector=patch.reshape(-1),
            indices=indices,
            phase=phase_index(left % 6, top % 6),
            x=left,
            y=top,
        ))
    return result


def bsds_samples(
    rows: list[dict[str, object]],
    patch_size: int,
    per_source: int,
    group: str,
) -> list[PatchSample]:
    result = []
    for row in rows:
        result.extend(sample_rgb_patches(
            load_rgb(row), str(row["filename"]), group, patch_size, per_source,
        ))
    return result


def external_samples(patch_size: int, per_crop: int = 12) -> list[PatchSample]:
    result = []
    for source in external_chromatic_sources():
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        for crop_index, (x, y) in enumerate(source["crops"]):
            result.extend(sample_rgb_patches(
                rgb,
                f"{source['id']}-{crop_index}",
                "external-chromatic",
                patch_size,
                per_crop,
                region=(int(x), int(y), int(x) + 168, int(y) + 168),
            ))
    return result


def established_samples(
    starfield: Path, patch_size: int, per_source: int = 96,
) -> dict[str, list[PatchSample]]:
    result = {}
    for source in _load_experiment_sources(starfield):
        source_id = str(source["id"])
        if source_id not in ("astronaut", "hubble", "nasa-hydra-starfield"):
            continue
        result[source_id] = sample_rgb_patches(
            np.asarray(source["rgb"], dtype=np.float64),
            source_id,
            "established",
            patch_size,
            per_source,
        )
        if source_id in ("hubble", "nasa-hydra-starfield"):
            result[f"{source_id}-bright"] = sample_bright_patches(
                np.asarray(source["rgb"], dtype=np.float64), source_id,
                patch_size, per_source,
            )
    return result


def corpus_manifest(
    bsds_root: Path,
    patch_size: int,
    training_per_source: int,
    evaluation_per_source: int,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    splits = discover_expanded(bsds_root)
    external = external_chromatic_sources()
    return ({
        "format": "rawtherapee-xtrans-sparse-dictionary-corpus-v1",
        "parent": lmmse_dataset_manifest(splits, external),
        "patch_sampling": {
            "evaluation_per_bsds_source": evaluation_per_source,
            "patch_sizes": [3, 5, 7, 9],
            "primary_patch_size": patch_size,
            "selection": "SHA-256-seeded sampling without replacement within each source",
            "training_per_bsds_source": training_per_source,
            "vector_order": "contiguous CHW RGB",
        },
        "split_discipline": "BSDS train only for dictionary fitting; validation for configuration; test, external chromatic, Hubble, and Hydra untouched",
    }, splits)


__all__ = (
    "PatchSample", "bsds_samples", "corpus_manifest", "established_samples",
    "external_samples", "sample_bright_patches", "sample_rgb_patches",
)

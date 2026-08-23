"""Frozen corpus bindings and deterministic patch sampling for the GMM study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools.xtrans_lmmse.dataset import load_rgb
from tools.xtrans_lmmse_mixture.dataset import (
    discover_expanded,
    external_chromatic_sources,
    manifest as parent_manifest,
)
from tools.xtrans_lmmse.experiment import _load_experiment_sources
from tools.xtrans_sparse_dictionary.dataset import (
    PatchSample,
    sample_bright_patches,
    sample_rgb_patches,
)
from tools.xtrans_sparse_dictionary.model import observation_indices, phase_index


TRAINING_PER_SOURCE = 512
VALIDATION_GRID_SIDE = 24
TEST_GRID_SIDE = 24
SEED_DESCRIPTION = "SHA-256 label-derived NumPy PCG64 sampling"


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def grid_samples(
    rgb: np.ndarray,
    source_id: str,
    group: str,
    patch_size: int,
    side: int,
) -> list[PatchSample]:
    """Select one deterministic dense grid, enabling block-oracle metrics."""

    checked = np.asarray(rgb, dtype=np.float64)
    height, width = checked.shape[1:]
    radius = patch_size // 2
    if side < 1 or width < side + 2 * radius or height < side + 2 * radius:
        raise ValueError("source is too small for requested evaluation grid")
    rng = np.random.default_rng(_seed(f"grid:{source_id}:{group}:{patch_size}:{side}"))
    left = int(rng.integers(radius, width - radius - side + 1))
    top = int(rng.integers(radius, height - radius - side + 1))
    result = []
    for center_y in range(top, top + side):
        for center_x in range(left, left + side):
            x = center_x - radius
            y = center_y - radius
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


def training_samples(rows: list[dict[str, object]], patch_size: int) -> list[PatchSample]:
    result = []
    for row in rows:
        result.extend(sample_rgb_patches(
            load_rgb(row), str(row["filename"]), "bsds-train", patch_size,
            TRAINING_PER_SOURCE,
        ))
    return result


def evaluation_samples(
    rows: list[dict[str, object]], patch_size: int, group: str, side: int,
) -> list[PatchSample]:
    result = []
    for row in rows:
        result.extend(grid_samples(
            load_rgb(row), str(row["filename"]), group, patch_size, side,
        ))
    return result


def external_samples(patch_size: int, side: int = 12) -> list[PatchSample]:
    result = []
    for source in external_chromatic_sources():
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        for crop_index, (x, y) in enumerate(source["crops"]):
            crop = rgb[:, int(y):int(y) + 168, int(x):int(x) + 168]
            result.extend(grid_samples(
                crop, f"{source['id']}-{crop_index}", "external-chromatic",
                patch_size, side,
            ))
    return result


def established_samples(starfield: Path, patch_size: int) -> dict[str, list[PatchSample]]:
    result = {}
    for source in _load_experiment_sources(starfield):
        source_id = str(source["id"])
        if source_id not in ("hubble", "nasa-hydra-starfield"):
            continue
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        result[source_id] = grid_samples(
            rgb, source_id, "established-uniform", patch_size, 24,
        )
        result[f"{source_id}-bright"] = sample_bright_patches(
            rgb, source_id, patch_size, 96,
        )
    return result


def corpus_manifest(bsds_root: Path) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    splits = discover_expanded(bsds_root)
    external = external_chromatic_sources()
    return ({
        "format": "rawtherapee-xtrans-joint-color-gmm-corpus-v1",
        "parent": parent_manifest(splits, external),
        "sampling": {
            "evaluation": "one SHA-256-seeded dense 24x24 center grid per BSDS source",
            "patch_sizes": [3, 5, 7],
            "seed": SEED_DESCRIPTION,
            "training": f"{TRAINING_PER_SOURCE} random patches per each of 200 BSDS train sources",
            "training_patch_count_per_configuration": TRAINING_PER_SOURCE * 200,
            "vector_order": "contiguous CHW RGB",
        },
        "split_discipline": "BSDS train only for EM; validation for all choices; test/external/stars/synthetics untouched",
    }, splits)


__all__ = (
    "TEST_GRID_SIDE", "TRAINING_PER_SOURCE", "VALIDATION_GRID_SIDE",
    "corpus_manifest", "established_samples", "evaluation_samples",
    "external_samples", "grid_samples", "training_samples",
)

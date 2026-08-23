"""Metrics for the ULRI structural-comparison experiment."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from tools.xtrans_oracle.analysis import OPPONENT_BASIS


MARGIN = 16


def check_rgb(value: np.ndarray) -> np.ndarray:
    checked = np.asarray(value, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("RGB data must have shape (3,height,width)")
    if not np.isfinite(checked).all():
        raise ValueError("RGB data must be finite")
    return checked


def opponent(value: np.ndarray) -> np.ndarray:
    return np.einsum("dc,cyx->dyx", OPPONENT_BASIS, check_rgb(value))


def _psnr(rms: float) -> float:
    return math.inf if rms == 0 else float(-20 * math.log10(rms))


def error_vectors(
    truth: np.ndarray, reconstruction: np.ndarray, cfa: np.ndarray, margin: int = MARGIN
) -> dict[str, np.ndarray]:
    truth = check_rgb(truth)
    reconstruction = check_rgb(reconstruction)
    if truth.shape != reconstruction.shape:
        raise ValueError("truth and reconstruction shapes differ")
    cfa = np.asarray(cfa, dtype=np.uint8)
    if cfa.shape != truth.shape[1:]:
        raise ValueError("CFA shape differs from RGB")
    height, width = cfa.shape
    if margin < 0 or 2 * margin >= min(height, width):
        raise ValueError("margin leaves no evaluation interior")
    ys, xs = slice(margin, height - margin), slice(margin, width - margin)
    error = reconstruction[:, ys, xs] - truth[:, ys, xs]
    component_error = opponent(reconstruction)[:, ys, xs] - opponent(truth)[:, ys, xs]
    interior_cfa = cfa[ys, xs]
    sampled = []
    interpolated = []
    missing_green = []
    missing_red_blue = []
    for channel in range(3):
        sampled.append(error[channel][interior_cfa == channel])
        interpolated.append(error[channel][interior_cfa != channel])
        if channel == 1:
            missing_green.append(error[channel][interior_cfa != channel])
        else:
            missing_red_blue.append(error[channel][interior_cfa != channel])
    return {
        "rgb": error.ravel(),
        "R": error[0].ravel(),
        "G": error[1].ravel(),
        "B": error[2].ravel(),
        "L": component_error[0].ravel(),
        "C1": component_error[1].ravel(),
        "C2": component_error[2].ravel(),
        "sampled": np.concatenate(sampled),
        "interpolated": np.concatenate(interpolated),
        "missing-green": np.concatenate(missing_green),
        "missing-red-blue": np.concatenate(missing_red_blue),
    }


def summarize_vectors(vectors: Mapping[str, np.ndarray]) -> dict[str, object]:
    absolute = np.abs(np.asarray(vectors["rgb"], dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.asarray(vectors["rgb"], dtype=np.float64) ** 2)))
    return {
        "channel_rms": {
            name: float(np.sqrt(np.mean(np.asarray(vectors[name]) ** 2)))
            for name in ("R", "G", "B")
        },
        "component_rms": {
            name: float(np.sqrt(np.mean(np.asarray(vectors[name]) ** 2)))
            for name in ("L", "C1", "C2")
        },
        "interpolated_rms": float(np.sqrt(np.mean(vectors["interpolated"] ** 2))),
        "maximum_abs": float(np.max(absolute)),
        "missing_green_rms": float(np.sqrt(np.mean(vectors["missing-green"] ** 2))),
        "missing_red_blue_rms": float(np.sqrt(np.mean(vectors["missing-red-blue"] ** 2))),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": _psnr(rms),
        "rgb_rms": rms,
        "sampled_maximum_abs": float(np.max(np.abs(vectors["sampled"]))),
        "sampled_rms": float(np.sqrt(np.mean(vectors["sampled"] ** 2))),
    }


def method_metrics(
    truth: np.ndarray, reconstruction: np.ndarray, cfa: np.ndarray, margin: int = MARGIN
) -> dict[str, object]:
    return summarize_vectors(error_vectors(truth, reconstruction, cfa, margin))


def pooled_metrics(
    cases: Sequence[Mapping[str, object]], method: str, margin: int = MARGIN
) -> dict[str, object]:
    gathered: dict[str, list[np.ndarray]] = {}
    for case in cases:
        vectors = error_vectors(
            np.asarray(case["truth"]), np.asarray(case["outputs"][method]),
            np.asarray(case["cfa"]), margin,
        )
        for name, values in vectors.items():
            gathered.setdefault(name, []).append(values)
    return summarize_vectors({name: np.concatenate(values) for name, values in gathered.items()})


def patch_oracle(
    truth: np.ndarray, candidates: Sequence[np.ndarray], window: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    truth = check_rgb(truth)
    bank = np.stack([check_rgb(candidate) for candidate in candidates])
    squared = np.mean((bank - truth[None]) ** 2, axis=1)
    local = np.stack([
        ndimage.uniform_filter(row, size=window, mode="nearest") for row in squared
    ])
    selected = np.argmin(local, axis=0)
    output = np.take_along_axis(
        np.moveaxis(bank, 1, -1), selected[None, ..., None], axis=0
    )[0]
    return np.moveaxis(output, -1, 0), selected

"""CFA-observable features and bounded three-bank LMMSE selection.

This module deliberately keeps the selector interpretable.  Classification
uses only scalar values physically measured through the X-Trans CFA.  It never
uses an initializer, reconstructed RGB, source identity, or reconstruction
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tools.xtrans_lmmse.model import (
    FilterBank,
    _check_rgb,
    _dc_transform,
    _mosaic,
    _windows,
    observation_colors,
    phase_map,
    positions_for_phase,
)


EPSILON = np.finfo(np.float64).eps
CLASS_COUNT = 3


@dataclass(frozen=True)
class FeatureNormalization:
    raw_variance_median: np.ndarray
    highpass_variance_median: np.ndarray
    chroma_variation_median: np.ndarray


@dataclass(frozen=True)
class MixtureDefinition:
    name: str
    partition: str
    activity_feature: str
    activity_threshold: float
    anisotropy_threshold: float
    activity_width: float
    anisotropy_width: float
    class_names: tuple[str, str, str]
    activity_quantile: float
    anisotropy_quantile: float


@dataclass(frozen=True)
class MixtureBank:
    definition: MixtureDefinition
    normalization: FeatureNormalization
    banks: tuple[FilterBank, FilterBank, FilterBank]
    fallback_to_global: np.ndarray


def _same_color_pairs(support: int, phase: int) -> tuple[np.ndarray, ...]:
    """Return horizontal/vertical same-color pairs at distances one to three.

    Each row is ``(first_flat_index, second_flat_index, distance)``.  Comparing
    only equal CFA colors avoids turning red/green/blue level differences into
    false spatial gradients.
    """

    colors = observation_colors(support, phase).reshape(support, support)
    horizontal = []
    vertical = []
    for distance in (1, 2, 3):
        for y in range(support):
            for x in range(support - distance):
                if colors[y, x] == colors[y, x + distance]:
                    horizontal.append((y * support + x, y * support + x + distance, distance))
        for y in range(support - distance):
            for x in range(support):
                if colors[y, x] == colors[y + distance, x]:
                    vertical.append((y * support + x, (y + distance) * support + x, distance))
    if not horizontal or not vertical:
        raise RuntimeError("X-Trans support has no directional same-color pairs")
    return (
        np.asarray(horizontal, dtype=np.int64),
        np.asarray(vertical, dtype=np.int64),
    )


def observable_features(
    observations: np.ndarray,
    phase: int,
    support: int,
) -> dict[str, np.ndarray]:
    """Calculate activity, anisotropy, and chroma from physical samples only."""

    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != support * support:
        raise ValueError("invalid CFA observation matrix")
    if not np.isfinite(x).all():
        raise ValueError("non-finite CFA observations")
    raw_mean = np.mean(x, axis=1, keepdims=True)
    raw_variance = np.mean((x - raw_mean) ** 2, axis=1)

    colors = observation_colors(support, phase)
    residual = np.empty_like(x)
    color_means = np.empty((x.shape[0], 3), dtype=np.float64)
    for channel in range(3):
        selected = colors == channel
        color_means[:, channel] = np.mean(x[:, selected], axis=1)
        residual[:, selected] = x[:, selected] - color_means[:, channel, None]
    highpass_variance = np.mean(residual * residual, axis=1)
    chroma_variation = (
        (color_means[:, 0] - color_means[:, 1]) ** 2
        + (color_means[:, 2] - color_means[:, 1]) ** 2
    )

    horizontal, vertical = _same_color_pairs(support, phase)

    def energy(pairs: np.ndarray) -> np.ndarray:
        difference = x[:, pairs[:, 0]] - x[:, pairs[:, 1]]
        scaled = difference / pairs[:, 2][None, :]
        return np.mean(scaled * scaled, axis=1)

    horizontal_energy = energy(horizontal)
    vertical_energy = energy(vertical)
    anisotropy = np.abs(horizontal_energy - vertical_energy) / (
        horizontal_energy + vertical_energy + EPSILON
    )
    return {
        "anisotropy": anisotropy,
        "chroma_variation": chroma_variation,
        "highpass_variance": highpass_variance,
        "horizontal_energy": horizontal_energy,
        "raw_variance": raw_variance,
        "vertical_energy": vertical_energy,
    }


def normalize_features(
    features: dict[str, np.ndarray],
    phase: int,
    normalization: FeatureNormalization,
) -> dict[str, np.ndarray]:
    """Normalize scale features by training-only phase medians and take log10."""

    medians = {
        "raw_variance": normalization.raw_variance_median[phase],
        "highpass_variance": normalization.highpass_variance_median[phase],
        "chroma_variation": normalization.chroma_variation_median[phase],
    }
    result = {"anisotropy": np.asarray(features["anisotropy"], dtype=np.float64)}
    for name, median in medians.items():
        scale = max(float(median), 1e-14)
        result[name] = np.log10(np.asarray(features[name], dtype=np.float64) / scale + 1e-12)
    return result


def classify_features(
    features: dict[str, np.ndarray], definition: MixtureDefinition
) -> np.ndarray:
    activity = np.asarray(features[definition.activity_feature], dtype=np.float64)
    anisotropy = np.asarray(features["anisotropy"], dtype=np.float64)
    if activity.shape != anisotropy.shape:
        raise ValueError("feature shapes differ")
    labels = np.full(activity.shape, 2, dtype=np.uint8)
    labels[activity < definition.activity_threshold] = 0
    labels[
        (activity >= definition.activity_threshold)
        & (anisotropy >= definition.anisotropy_threshold)
    ] = 1
    return labels


def soft_weights(
    features: dict[str, np.ndarray], definition: MixtureDefinition
) -> np.ndarray:
    """A fixed logistic smoothing of the two explicit threshold boundaries."""

    activity = np.asarray(features[definition.activity_feature], dtype=np.float64)
    anisotropy = np.asarray(features["anisotropy"], dtype=np.float64)
    activity_width = max(definition.activity_width, 1e-6)
    anisotropy_width = max(definition.anisotropy_width, 1e-6)
    smooth = 1.0 / (
        1.0 + np.exp(np.clip((activity - definition.activity_threshold) / activity_width, -60.0, 60.0))
    )
    edge_conditional = 1.0 / (
        1.0 + np.exp(np.clip((definition.anisotropy_threshold - anisotropy) / anisotropy_width, -60.0, 60.0))
    )
    remainder = 1.0 - smooth
    weights = np.stack(
        (smooth, remainder * edge_conditional, remainder * (1.0 - edge_conditional)),
        axis=1,
    )
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def _feature_rows(
    rgb: np.ndarray,
    support: int,
    region: tuple[int, int, int, int],
    normalization: FeatureNormalization,
    *,
    origin_x: int,
    origin_y: int,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]]:
    checked = _check_rgb(rgb)
    scalar, _ = _mosaic(checked, origin_x, origin_y)
    windows = _windows(scalar, support)
    x0, y0, x1, y1 = region
    class_shape = (y1 - y0, x1 - x0)
    class_map = np.empty(class_shape, dtype=np.uint8)
    rows = {}
    for phase in range(18):
        ys, xs = positions_for_phase(
            scalar.shape, phase, region, origin_x=origin_x, origin_y=origin_y
        )
        observations = windows[ys, xs].reshape(ys.size, support * support)
        raw = observable_features(observations, phase, support)
        normalized = normalize_features(raw, phase, normalization)
        rows[phase] = (ys, xs, normalized)
    return class_map, rows


def classify_region(
    rgb: np.ndarray,
    mixture: MixtureBank,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    class_map, rows = _feature_rows(
        rgb, mixture.banks[0].support, region, mixture.normalization,
        origin_x=origin_x, origin_y=origin_y,
    )
    x0, y0, _, _ = region
    feature_maps = {
        name: np.empty(class_map.shape, dtype=np.float64)
        for name in ("raw_variance", "highpass_variance", "chroma_variation", "anisotropy")
    }
    for _, (ys, xs, features) in rows.items():
        local_y = ys - y0
        local_x = xs - x0
        class_map[local_y, local_x] = classify_features(features, mixture.definition)
        for name in feature_maps:
            feature_maps[name][local_y, local_x] = features[name]
    return class_map, feature_maps


def predict_mixture(
    mixture: MixtureBank,
    rgb: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    soft: bool = False,
    clip: bool = False,
    individual_outputs: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply three fixed banks and select/blend them from observable features."""

    from tools.xtrans_lmmse.model import predict_region

    outputs = (
        tuple(individual_outputs)
        if individual_outputs is not None
        else tuple(
            predict_region(
                bank, rgb, region, origin_x=origin_x, origin_y=origin_y, clip=False
            )
            for bank in mixture.banks
        )
    )
    if len(outputs) != CLASS_COUNT or any(output.shape != outputs[0].shape for output in outputs):
        raise ValueError("expected three equal-sized bank outputs")
    class_map, feature_maps = classify_region(
        rgb, mixture, region, origin_x=origin_x, origin_y=origin_y
    )
    if soft:
        # Convert maps to the same row representation used by soft_weights.
        flat = {name: values.ravel() for name, values in feature_maps.items()}
        weights = soft_weights(flat, mixture.definition)
        output = sum(
            outputs[index] * weights[:, index].reshape(class_map.shape)[None, :, :]
            for index in range(CLASS_COUNT)
        )
    else:
        output = np.empty_like(outputs[0])
        for class_index in range(CLASS_COUNT):
            selected = class_map == class_index
            output[:, selected] = outputs[class_index][:, selected]
    if clip:
        np.clip(output, 0.0, 1.0, out=output)
        # Every bank already restores the same native sample, so clipping only
        # needs to preserve it explicitly for out-of-range research inputs.
        checked = _check_rgb(rgb)
        scalar, cfa = _mosaic(checked, origin_x, origin_y)
        x0, y0, x1, y1 = region
        local_cfa = cfa[y0:y1, x0:x1]
        local_scalar = scalar[y0:y1, x0:x1]
        for channel in range(3):
            selected = local_cfa == channel
            output[channel, selected] = local_scalar[selected]
    if not np.isfinite(output).all():
        raise ValueError("non-finite mixture output")
    return output, class_map


def oracle_select(
    outputs: Iterable[np.ndarray], truth: np.ndarray, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the minimum-RGB-MSE bank in non-overlapping square blocks."""

    candidates = tuple(np.asarray(output, dtype=np.float64) for output in outputs)
    checked_truth = _check_rgb(truth)
    if len(candidates) != CLASS_COUNT or any(value.shape != checked_truth.shape for value in candidates):
        raise ValueError("invalid oracle candidates")
    if block < 1:
        raise ValueError("block must be positive")
    height, width = checked_truth.shape[1:]
    selected_map = np.empty((height, width), dtype=np.uint8)
    result = np.empty_like(checked_truth)
    for y0 in range(0, height, block):
        y1 = min(height, y0 + block)
        for x0 in range(0, width, block):
            x1 = min(width, x0 + block)
            errors = [
                float(np.sum((value[:, y0:y1, x0:x1] - checked_truth[:, y0:y1, x0:x1]) ** 2))
                for value in candidates
            ]
            chosen = int(np.argmin(errors))
            result[:, y0:y1, x0:x1] = candidates[chosen][:, y0:y1, x0:x1]
            selected_map[y0:y1, x0:x1] = chosen
    return result, selected_map

"""Local affine RGB line fitting and CFA-constrained reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


DEGENERATE_CHANNEL_DELTA = 1e-4


@dataclass(frozen=True)
class ColorLineFit:
    mean: np.ndarray
    direction: np.ndarray
    eigenvalues: np.ndarray
    minimum_projection: np.ndarray
    maximum_projection: np.ndarray
    two_means_low: np.ndarray
    two_means_high: np.ndarray
    patch_size: int


@dataclass(frozen=True)
class LineReconstruction:
    rgb: np.ndarray
    identifiable: np.ndarray
    direct_coordinate: np.ndarray
    used_coordinate: np.ndarray
    projected_truth_coordinate: np.ndarray | None
    spatial_support_identifiable: np.ndarray


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    checked = np.asarray(rgb, dtype=np.float64)
    if (
        checked.ndim != 3
        or checked.shape[0] != 3
        or min(checked.shape[1:]) < 3
        or not np.isfinite(checked).all()
    ):
        raise ValueError("RGB input must be finite CHW data")
    return checked


def _canonicalize_direction(direction: np.ndarray) -> np.ndarray:
    dominant = np.argmax(np.abs(direction), axis=-1)
    sign = np.take_along_axis(direction, dominant[..., None], axis=-1)[..., 0]
    return direction * np.where(sign < 0.0, -1.0, 1.0)[..., None]


def _moments(
    rgb_hwc: np.ndarray,
    patch_size: int,
    center_exclusion_radius: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size < 3 or patch_size % 2 != 1:
        raise ValueError("patch size must be odd and at least three")
    if (
        center_exclusion_radius is not None
        and (
            center_exclusion_radius < 0
            or 2 * center_exclusion_radius + 1 >= patch_size
        )
    ):
        raise ValueError("invalid center exclusion")
    if center_exclusion_radius is None:
        mean = np.stack(
            [
                ndimage.uniform_filter(
                    rgb_hwc[..., channel], size=patch_size, mode="reflect"
                )
                for channel in range(3)
            ],
            axis=-1,
        )
        second = np.empty((*rgb_hwc.shape[:2], 3, 3), dtype=np.float64)
        for left in range(3):
            for right in range(left, 3):
                value = ndimage.uniform_filter(
                    rgb_hwc[..., left] * rgb_hwc[..., right],
                    size=patch_size,
                    mode="reflect",
                )
                second[..., left, right] = value
                second[..., right, left] = value
        return mean, second

    kernel = np.ones((patch_size, patch_size), dtype=np.float64)
    center = patch_size // 2
    radius = center_exclusion_radius
    kernel[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ] = 0.0
    count = float(np.sum(kernel))
    mean = np.stack(
        [
            ndimage.convolve(rgb_hwc[..., channel], kernel, mode="reflect") / count
            for channel in range(3)
        ],
        axis=-1,
    )
    second = np.empty((*rgb_hwc.shape[:2], 3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(left, 3):
            value = ndimage.convolve(
                rgb_hwc[..., left] * rgb_hwc[..., right], kernel, mode="reflect"
            ) / count
            second[..., left, right] = value
            second[..., right, left] = value
    return mean, second


def _local_projections(
    rgb_hwc: np.ndarray,
    mean: np.ndarray,
    direction: np.ndarray,
    patch_size: int,
    center_exclusion_radius: int | None,
) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(
        rgb_hwc, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (patch_size, patch_size), axis=(0, 1)
    )
    projections = np.einsum(
        "hwcij,hwc->hwij",
        windows - mean[..., None, None],
        direction,
        optimize=True,
    )
    if center_exclusion_radius is not None:
        center = patch_size // 2
        radius = center_exclusion_radius
        projections[
            ...,
            center - radius : center + radius + 1,
            center - radius : center + radius + 1,
        ] = np.nan
    return projections


def _two_means(projections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanmin(projections, axis=(-2, -1))
    high = np.nanmax(projections, axis=(-2, -1))
    left = low.copy()
    right = high.copy()
    finite = np.isfinite(projections)
    for _ in range(8):
        threshold = (left + right) * 0.5
        choose_left = finite & (projections <= threshold[..., None, None])
        choose_right = finite & ~choose_left
        left_count = np.sum(choose_left, axis=(-2, -1))
        right_count = np.sum(choose_right, axis=(-2, -1))
        left_sum = np.nansum(np.where(choose_left, projections, 0.0), axis=(-2, -1))
        right_sum = np.nansum(np.where(choose_right, projections, 0.0), axis=(-2, -1))
        left = np.divide(left_sum, left_count, out=left, where=left_count != 0)
        right = np.divide(right_sum, right_count, out=right, where=right_count != 0)
    return np.minimum(left, right), np.maximum(left, right)


def fit_color_line(
    rgb: np.ndarray,
    patch_size: int,
    *,
    center_exclusion_radius: int | None = None,
    compute_two_means: bool = False,
) -> ColorLineFit:
    checked = _check_rgb(rgb)
    rgb_hwc = np.moveaxis(checked, 0, -1)
    mean, second = _moments(rgb_hwc, patch_size, center_exclusion_radius)
    covariance = second - mean[..., :, None] * mean[..., None, :]
    covariance = (covariance + np.swapaxes(covariance, -1, -2)) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    direction = _canonicalize_direction(eigenvectors[..., :, -1])
    projections = _local_projections(
        rgb_hwc, mean, direction, patch_size, center_exclusion_radius
    )
    minimum = np.nanmin(projections, axis=(-2, -1))
    maximum = np.nanmax(projections, axis=(-2, -1))
    if compute_two_means:
        two_low, two_high = _two_means(projections)
    else:
        two_low, two_high = minimum.copy(), maximum.copy()
    if not all(
        np.isfinite(values).all()
        for values in (mean, direction, eigenvalues, minimum, maximum, two_low, two_high)
    ):
        raise ValueError("color-line fit produced non-finite values")
    return ColorLineFit(
        mean=mean,
        direction=direction,
        eigenvalues=eigenvalues,
        minimum_projection=minimum,
        maximum_projection=maximum,
        two_means_low=two_low,
        two_means_high=two_high,
        patch_size=patch_size,
    )


def fit_color_line_huber(rgb: np.ndarray, patch_size: int) -> ColorLineFit:
    """One deterministic Huber reweighting of an ordinary local PCA line."""
    checked = _check_rgb(rgb)
    rgb_hwc = np.moveaxis(checked, 0, -1)
    initial = fit_color_line(checked, patch_size)
    coordinate = np.sum(
        (rgb_hwc - initial.mean) * initial.direction, axis=-1
    )
    projected = initial.mean + coordinate[..., None] * initial.direction
    residual = np.sqrt(np.sum((rgb_hwc - projected) ** 2, axis=-1))
    scale = 1.4826 * ndimage.median_filter(
        residual, size=patch_size, mode="reflect"
    )
    cutoff = 1.345 * np.maximum(scale, 1e-12)
    weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
    normalizer = ndimage.uniform_filter(
        weights, size=patch_size, mode="reflect"
    )
    usable = normalizer > 1e-12
    mean = np.empty_like(rgb_hwc)
    for channel in range(3):
        numerator = ndimage.uniform_filter(
            weights * rgb_hwc[..., channel], size=patch_size, mode="reflect"
        )
        mean[..., channel] = np.divide(
            numerator,
            normalizer,
            out=initial.mean[..., channel].copy(),
            where=usable,
        )
    second = np.empty((*rgb_hwc.shape[:2], 3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(left, 3):
            numerator = ndimage.uniform_filter(
                weights * rgb_hwc[..., left] * rgb_hwc[..., right],
                size=patch_size,
                mode="reflect",
            )
            value = np.divide(
                numerator,
                normalizer,
                out=(
                    initial.mean[..., left] * initial.mean[..., right]
                ).copy(),
                where=usable,
            )
            second[..., left, right] = value
            second[..., right, left] = value
    covariance = second - mean[..., :, None] * mean[..., None, :]
    covariance = (covariance + np.swapaxes(covariance, -1, -2)) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    direction = _canonicalize_direction(eigenvectors[..., :, -1])
    projections = _local_projections(rgb_hwc, mean, direction, patch_size, None)
    minimum = np.min(projections, axis=(-2, -1))
    maximum = np.max(projections, axis=(-2, -1))
    two_low, two_high = _two_means(projections)
    if not all(
        np.isfinite(values).all()
        for values in (mean, direction, eigenvalues, minimum, maximum, two_low, two_high)
    ):
        raise ValueError("robust color-line fit produced non-finite values")
    return ColorLineFit(
        mean=mean,
        direction=direction,
        eigenvalues=eigenvalues,
        minimum_projection=minimum,
        maximum_projection=maximum,
        two_means_low=two_low,
        two_means_high=two_high,
        patch_size=patch_size,
    )


def _sampled(values: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, cfa[..., None], axis=-1)[..., 0]


def _neighbor_coordinate(
    fit: ColorLineFit,
    scalar: np.ndarray,
    cfa: np.ndarray,
    separation: np.ndarray,
    threshold: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = scalar.shape
    padded_scalar = np.pad(scalar, radius, mode="reflect")
    padded_cfa = np.pad(cfa, radius, mode="reflect")
    weighted_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    sigma = max(radius / 1.5, 1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_cfa = padded_cfa[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            neighbor_scalar = padded_scalar[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            mean = _sampled(fit.mean, neighbor_cfa)
            direction = _sampled(fit.direction, neighbor_cfa)
            usable = np.abs(separation * direction) >= threshold
            coordinate = np.divide(
                neighbor_scalar - mean,
                direction,
                out=np.zeros_like(neighbor_scalar),
                where=usable,
            )
            weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            weighted_sum += weight * coordinate * usable
            weight_sum += weight * usable
    identifiable = weight_sum > 0.0
    coordinate = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum),
        where=identifiable,
    )
    return coordinate, identifiable


def _plane_coordinate(
    fit: ColorLineFit,
    scalar: np.ndarray,
    cfa: np.ndarray,
    separation: np.ndarray,
    threshold: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = scalar.shape
    padded_scalar = np.pad(scalar, radius, mode="reflect")
    padded_cfa = np.pad(cfa, radius, mode="reflect")
    normal = np.zeros((height, width, 3, 3), dtype=np.float64)
    right = np.zeros((height, width, 3), dtype=np.float64)
    sigma = max(radius / 1.5, 1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor_cfa = padded_cfa[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            neighbor_scalar = padded_scalar[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            mean = _sampled(fit.mean, neighbor_cfa)
            direction = _sampled(fit.direction, neighbor_cfa)
            usable = np.abs(separation * direction) >= threshold
            coordinate = np.divide(
                neighbor_scalar - mean,
                direction,
                out=np.zeros_like(neighbor_scalar),
                where=usable,
            )
            weight = (
                np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * usable
            )
            design = np.asarray((1.0, float(dx), float(dy)))
            normal += weight[..., None, None] * np.outer(design, design)
            right += weight[..., None] * coordinate[..., None] * design
    eigenvalues = np.linalg.eigvalsh(normal)
    identifiable = eigenvalues[..., 0] > np.maximum(eigenvalues[..., 2], 1.0) * 1e-10
    safe = normal.copy()
    safe[~identifiable] = np.eye(3)
    coefficients = np.linalg.solve(safe, right[..., None])[..., 0]
    return coefficients[..., 0], identifiable


def reconstruct_from_line(
    fit: ColorLineFit,
    scalar: np.ndarray,
    cfa: np.ndarray,
    *,
    endpoint: str = "extrema",
    bounded: bool = True,
    degenerate_threshold: float = DEGENERATE_CHANNEL_DELTA,
    degenerate_fallback: str = "neighbor",
    neighbor_radius: int = 2,
    truth: np.ndarray | None = None,
) -> LineReconstruction:
    observed = np.asarray(scalar, dtype=np.float64)
    phases = np.asarray(cfa, dtype=np.int64)
    if observed.shape != phases.shape or observed.shape != fit.mean.shape[:2]:
        raise ValueError("line reconstruction shapes differ")
    if not np.isfinite(observed).all() or np.any((phases < 0) | (phases > 2)):
        raise ValueError("invalid CFA input")
    if endpoint == "extrema":
        low, high = fit.minimum_projection, fit.maximum_projection
    elif endpoint == "two-means":
        low, high = fit.two_means_low, fit.two_means_high
    else:
        raise ValueError("unknown endpoint definition")
    separation = high - low
    sampled_mean = _sampled(fit.mean, phases)
    sampled_direction = _sampled(fit.direction, phases)
    identifiable = np.abs(separation * sampled_direction) >= degenerate_threshold
    direct = np.divide(
        observed - sampled_mean,
        sampled_direction,
        out=np.zeros_like(observed),
        where=identifiable,
    )
    if degenerate_fallback == "neighbor":
        fallback, spatial_identifiable = _neighbor_coordinate(
            fit,
            observed,
            phases,
            separation,
            degenerate_threshold,
            neighbor_radius,
        )
    elif degenerate_fallback == "plane":
        fallback, spatial_identifiable = _plane_coordinate(
            fit,
            observed,
            phases,
            separation,
            degenerate_threshold,
            neighbor_radius,
        )
    elif degenerate_fallback == "centroid":
        fallback = np.zeros_like(observed)
        spatial_identifiable = np.zeros_like(identifiable)
    else:
        raise ValueError("unknown degenerate fallback")
    used = np.where(identifiable, direct, fallback)
    if bounded:
        used = np.clip(used, low, high)
    rgb_hwc = fit.mean + used[..., None] * fit.direction
    for channel in range(3):
        mask = phases == channel
        rgb_hwc[..., channel][mask] = observed[mask]
    projected_truth_coordinate = None
    if truth is not None:
        checked_truth = np.moveaxis(_check_rgb(truth), 0, -1)
        if checked_truth.shape != rgb_hwc.shape:
            raise ValueError("truth shape differs")
        projected_truth_coordinate = np.sum(
            (checked_truth - fit.mean) * fit.direction, axis=-1
        )
    if not np.isfinite(rgb_hwc).all():
        raise ValueError("line reconstruction produced non-finite values")
    return LineReconstruction(
        rgb=np.moveaxis(rgb_hwc, -1, 0),
        identifiable=identifiable,
        direct_coordinate=direct,
        used_coordinate=used,
        projected_truth_coordinate=projected_truth_coordinate,
        spatial_support_identifiable=spatial_identifiable,
    )


def project_truth_to_line(fit: ColorLineFit, truth: np.ndarray) -> np.ndarray:
    checked = np.moveaxis(_check_rgb(truth), 0, -1)
    coordinate = np.sum((checked - fit.mean) * fit.direction, axis=-1)
    return np.moveaxis(fit.mean + coordinate[..., None] * fit.direction, -1, 0)

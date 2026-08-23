"""Small independent Bayer front-end reproduction for Bennett et al.

The paper specifies Malvar--He--Cutler HQLI, weighted RGB two-means in a 5x5
window, and a one-standard-deviation rejection/re-fit.  It does not specify
K-means initialization, iteration count, the finite center weight for inverse
distance, or the scalar definition of RGB cluster variance.  Those choices are
isolated and documented below; this module is not presented as author-code
parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .model import bennett_alpha, reconstruct


_GREEN_AT_RB = np.asarray(
    (
        (0, 0, -1, 0, 0),
        (0, 0, 2, 0, 0),
        (-1, 2, 4, 2, -1),
        (0, 0, 2, 0, 0),
        (0, 0, -1, 0, 0),
    ),
    dtype=np.float64,
) / 8.0
_RB_AT_G_HORIZONTAL = np.asarray(
    (
        (0, 0, 0.5, 0, 0),
        (0, -1, 0, -1, 0),
        (-1, 4, 5, 4, -1),
        (0, -1, 0, -1, 0),
        (0, 0, 0.5, 0, 0),
    ),
    dtype=np.float64,
) / 8.0
_RB_AT_G_VERTICAL = _RB_AT_G_HORIZONTAL.T
_RB_AT_OPPOSITE = np.asarray(
    (
        (0, 0, -1.5, 0, 0),
        (0, 2, 0, 2, 0),
        (-1.5, 0, 6, 0, -1.5),
        (0, 2, 0, 2, 0),
        (0, 0, -1.5, 0, 0),
    ),
    dtype=np.float64,
) / 8.0


@dataclass(frozen=True)
class BayerReferenceResult:
    bootstrap: np.ndarray
    first: np.ndarray
    second: np.ndarray
    rgb: np.ndarray


def rggb_cfa(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.where(
        y % 2 == 0,
        np.where(x % 2 == 0, 0, 1),
        np.where(x % 2 == 0, 1, 2),
    ).astype(np.uint8)


def mosaic_bayer(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(rgb, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3 or not np.isfinite(checked).all():
        raise ValueError("Bayer truth must be finite CHW RGB")
    cfa = rggb_cfa(checked.shape[1], checked.shape[2])
    scalar = np.take_along_axis(
        np.moveaxis(checked, 0, -1), cfa[..., None], axis=-1
    )[..., 0]
    return scalar, cfa


def malvar_hqli(scalar: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    """Malvar--He--Cutler 5x5 linear interpolation for an RGGB mosaic."""
    observed = np.asarray(scalar, dtype=np.float64)
    phases = np.asarray(cfa, dtype=np.uint8)
    if observed.shape != phases.shape or np.any(phases != rggb_cfa(*phases.shape)):
        raise ValueError("the Bayer reference requires canonical RGGB")
    filtered_green = ndimage.convolve(observed, _GREEN_AT_RB, mode="reflect")
    filtered_horizontal = ndimage.convolve(
        observed, _RB_AT_G_HORIZONTAL, mode="reflect"
    )
    filtered_vertical = ndimage.convolve(
        observed, _RB_AT_G_VERTICAL, mode="reflect"
    )
    filtered_opposite = ndimage.convolve(
        observed, _RB_AT_OPPOSITE, mode="reflect"
    )
    y, _ = np.mgrid[: observed.shape[0], : observed.shape[1]]
    red = np.empty(observed.shape, dtype=np.float64)
    green = np.where(phases == 1, observed, filtered_green)
    blue = np.empty(observed.shape, dtype=np.float64)
    red[phases == 0] = observed[phases == 0]
    blue[phases == 2] = observed[phases == 2]
    opposite_red = phases == 2
    opposite_blue = phases == 0
    red[opposite_red] = filtered_opposite[opposite_red]
    blue[opposite_blue] = filtered_opposite[opposite_blue]
    green_mask = phases == 1
    green_red_row = green_mask & (y % 2 == 0)
    green_blue_row = green_mask & ~green_red_row
    red[green_red_row] = filtered_horizontal[green_red_row]
    blue[green_red_row] = filtered_vertical[green_red_row]
    red[green_blue_row] = filtered_vertical[green_blue_row]
    blue[green_blue_row] = filtered_horizontal[green_blue_row]
    result = np.stack((red, green, blue))
    if not np.isfinite(result).all():
        raise ValueError("HQLI produced non-finite output")
    return result


def _cluster_windows(bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_hwc = np.moveaxis(np.asarray(bootstrap, dtype=np.float64), 0, -1)
    padded = np.pad(rgb_hwc, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (5, 5), axis=(0, 1)
    )
    points = np.moveaxis(windows, 2, -1).reshape(*rgb_hwc.shape[:2], 25, 3)
    offsets = np.asarray(
        [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)],
        dtype=np.float64,
    )
    # Bennett says inverse Euclidean distance but the center distance is zero.
    # 1/(1+d) is the finite, monotonic interpretation used for this reproduction.
    weights = 1.0 / (1.0 + np.sqrt(np.sum(offsets * offsets, axis=1)))
    return points, weights


def _weighted_two_means(
    points: np.ndarray,
    spatial_weights: np.ndarray,
    active: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = np.zeros(points.shape[:-1], dtype=bool)
    for _ in range(iterations):
        first_distance = np.sum((points - first[..., None, :]) ** 2, axis=-1)
        second_distance = np.sum((points - second[..., None, :]) ** 2, axis=-1)
        assignment = second_distance < first_distance
        for choose_second in (False, True):
            selected = active & (assignment == choose_second)
            weights = selected * spatial_weights
            denominator = np.sum(weights, axis=-1)
            numerator = np.sum(weights[..., None] * points, axis=-2)
            old = second if choose_second else first
            updated = np.divide(
                numerator,
                denominator[..., None],
                out=old.copy(),
                where=denominator[..., None] > 0.0,
            )
            if choose_second:
                second = updated
            else:
                first = updated
    return first, second, assignment


def bennett_endpoints(bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent 5x5 weighted two-means plus one-sigma rejection."""
    points, spatial_weights = _cluster_windows(bootstrap)
    luminance = np.mean(points, axis=-1)
    low_index = np.argmin(luminance, axis=-1)
    high_index = np.argmax(luminance, axis=-1)
    first = np.take_along_axis(points, low_index[..., None, None], axis=-2)[..., 0, :]
    second = np.take_along_axis(points, high_index[..., None, None], axis=-2)[..., 0, :]
    active = np.ones(points.shape[:-1], dtype=bool)
    first, second, assignment = _weighted_two_means(
        points, spatial_weights, active, first, second
    )

    first_distance = np.sqrt(np.sum((points - first[..., None, :]) ** 2, axis=-1))
    second_distance = np.sqrt(np.sum((points - second[..., None, :]) ** 2, axis=-1))
    distance = np.where(assignment, second_distance, first_distance)
    keep = np.zeros(active.shape, dtype=bool)
    for choose_second in (False, True):
        selected = assignment == choose_second
        count = np.sum(selected, axis=-1)
        mean = np.divide(
            np.sum(np.where(selected, distance, 0.0), axis=-1),
            count,
            out=np.zeros(count.shape, dtype=np.float64),
            where=count > 0,
        )
        variance = np.divide(
            np.sum(
                np.where(selected, (distance - mean[..., None]) ** 2, 0.0),
                axis=-1,
            ),
            count,
            out=np.zeros(count.shape, dtype=np.float64),
            where=count > 0,
        )
        sigma = np.sqrt(np.maximum(variance, 0.0))
        cluster_keep = distance <= mean[..., None] + sigma[..., None]
        keep |= selected & cluster_keep
    first, second, _ = _weighted_two_means(
        points, spatial_weights, keep, first, second
    )
    swap = np.mean(first, axis=-1) > np.mean(second, axis=-1)
    low = np.where(swap[..., None], second, first)
    high = np.where(swap[..., None], first, second)
    return low, high


def full_bayer_reference(
    scalar: np.ndarray,
    cfa: np.ndarray,
    *,
    eta: float = 1.0,
) -> BayerReferenceResult:
    bootstrap = malvar_hqli(scalar, cfa)
    first, second = bennett_endpoints(bootstrap)
    alpha = bennett_alpha(
        scalar,
        cfa,
        first,
        second,
        support_radius=2,
        eta=eta,
    )
    rgb = reconstruct(first, second, alpha.alpha, scalar=scalar, cfa=cfa)
    return BayerReferenceResult(
        bootstrap=bootstrap,
        first=first,
        second=second,
        rgb=rgb,
    )

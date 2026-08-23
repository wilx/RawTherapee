"""Independent Bayer green-stage implementation of Sensors ARI v1.0.

The implementation follows equations (5)-(16) and the operation ordering of
the authenticated 2017 reference.  It intentionally does not import or copy
the research-only MATLAB source.  Arrays use normalized linear values; ARI is
homogeneous apart from its documented numerical floors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage


EPSILON = 1e-10
PATTERNS: Mapping[str, np.ndarray] = {
    "grbg": np.asarray(((1, 0), (2, 1)), dtype=np.uint8),
    "rggb": np.asarray(((0, 1), (1, 2)), dtype=np.uint8),
    "gbrg": np.asarray(((1, 2), (0, 1)), dtype=np.uint8),
    "bggr": np.asarray(((2, 1), (1, 0)), dtype=np.uint8),
}


@dataclass(frozen=True)
class BayerGreenTrace:
    candidates: np.ndarray  # branch,direction,iteration,height,width
    criteria: np.ndarray
    selected_iterations: np.ndarray  # branch,direction,height,width
    selected: np.ndarray  # branch,direction,height,width
    adaptive: np.ndarray


def _filter(value: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.correlate(
        np.asarray(value, dtype=np.float64),
        np.asarray(kernel, dtype=np.float64),
        mode="nearest",
    )


def _box(value: np.ndarray, horizontal: int, vertical: int) -> np.ndarray:
    size = (2 * vertical + 1, 2 * horizontal + 1)
    return ndimage.uniform_filter(
        np.asarray(value, dtype=np.float64), size=size, mode="constant", cval=0.0
    ) * float(size[0] * size[1])


def _gaussian5() -> np.ndarray:
    y, x = np.mgrid[-2:3, -2:3]
    kernel = np.exp(-(x * x + y * y) / 8.0)
    return kernel / np.sum(kernel)


GAUSSIAN5 = _gaussian5()


def bayer_masks(height: int, width: int, pattern: str) -> np.ndarray:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown Bayer pattern: {pattern}")
    y, x = np.mgrid[:height, :width]
    cfa = PATTERNS[pattern][y % 2, x % 2]
    return np.stack([cfa == channel for channel in range(3)]).astype(np.float64)


def mosaic_bayer(truth: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(truth, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3 or not np.isfinite(checked).all():
        raise ValueError("truth must be finite CHW RGB")
    masks = bayer_masks(checked.shape[1], checked.shape[2], pattern)
    return checked * masks, masks


def _guided_ri(
    guide: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    horizontal: int,
    vertical: int,
) -> np.ndarray:
    count = _box(mask, horizontal, vertical)
    count[count == 0.0] = 1.0
    mean_g = _box(guide * mask, horizontal, vertical) / count
    mean_o = _box(observed * mask, horizontal, vertical) / count
    mean_go = _box(guide * observed * mask, horizontal, vertical) / count
    covariance = mean_go - mean_g * mean_o
    variance = _box(guide * guide * mask, horizontal, vertical) / count - mean_g**2
    gain = covariance / (variance + EPSILON)
    offset = mean_o - gain * mean_g

    fit = (
        _box(guide * guide * mask, horizontal, vertical) * gain**2
        + offset**2 * count
        + _box(observed * observed * mask, horizontal, vertical)
        + 2.0 * gain * offset * _box(guide * mask, horizontal, vertical)
        - 2.0 * offset * _box(observed * mask, horizontal, vertical)
        - 2.0 * gain * _box(observed * guide * mask, horizontal, vertical)
    ) / count
    weight = 1.0 / np.maximum(np.sqrt(np.maximum(fit, 0.0)), 1e-3)
    weight_sum = _box(weight, horizontal, vertical)
    mean_gain = _box(gain * weight, horizontal, vertical) / (
        weight_sum + 1e-4
    )
    mean_offset = _box(offset * weight, horizontal, vertical) / (
        weight_sum + 1e-4
    )
    return mean_gain * guide + mean_offset


def _guided_mlri(
    guide: np.ndarray,
    observed: np.ndarray,
    line_mask: np.ndarray,
    lap_guide: np.ndarray,
    lap_observed: np.ndarray,
    lap_mask: np.ndarray,
    horizontal: int,
    vertical: int,
) -> np.ndarray:
    count = _box(lap_mask, horizontal, vertical)
    count[count == 0.0] = 1.0
    gain = (
        _box(lap_guide * lap_observed * lap_mask, horizontal, vertical) / count
    ) / (
        _box(lap_guide * lap_guide * lap_mask, horizontal, vertical) / count
        + EPSILON
    )
    samples = _box(line_mask, horizontal, vertical)
    samples[samples == 0.0] = 1.0
    mean_g = _box(guide * line_mask, horizontal, vertical) / samples
    mean_o = _box(observed * line_mask, horizontal, vertical) / samples
    offset = mean_o - gain * mean_g
    fit = (
        _box(guide * guide * line_mask, horizontal, vertical) * gain**2
        + offset**2 * samples
        + _box(observed * observed * line_mask, horizontal, vertical)
        + 2.0 * gain * offset * _box(guide * line_mask, horizontal, vertical)
        - 2.0 * offset * _box(observed * line_mask, horizontal, vertical)
        - 2.0 * gain * _box(observed * guide * line_mask, horizontal, vertical)
    ) / samples
    weight = 1.0 / np.maximum(np.sqrt(np.maximum(fit, 0.0)), 1e-3)
    weight_sum = _box(weight, horizontal, vertical)
    mean_gain = _box(gain * weight, horizontal, vertical) / (
        weight_sum + 1e-4
    )
    mean_offset = _box(offset * weight, horizontal, vertical) / (
        weight_sum + 1e-4
    )
    return mean_gain * guide + mean_offset


def green_ari(
    mosaic: np.ndarray,
    masks: np.ndarray,
    pattern: str,
    *,
    iterations: int = 11,
    preserve_reference_vertical_mask_typo: bool = True,
) -> BayerGreenTrace:
    """Run the independently implemented ARI green stage.

    Branch index 0 is RI and 1 is MLRI; direction 0 is horizontal and 1 is
    vertical.  Every iteration candidate and criterion is retained.
    """
    # Execute in the reference's native 0..255 double domain so every fixed
    # epsilon and regression-residual floor retains its published meaning.
    value = np.asarray(mosaic, dtype=np.float64) * 255.0
    mask = np.asarray(masks, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 3 or value.shape != mask.shape:
        raise ValueError("mosaic and masks must have equal CHW shapes")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    raw = np.sum(value, axis=0)
    height, width = raw.shape
    y, x = np.mgrid[:height, :width]
    cfa = PATTERNS[pattern][y % 2, x % 2]
    # G sites horizontally adjacent to R/B respectively.
    cell = PATTERNS[pattern]
    cell_gr = np.zeros((2, 2), dtype=bool)
    for cy in range(2):
        for cx in range(2):
            cell_gr[cy, cx] = cell[cy, cx] == 1 and (
                cell[cy, (cx - 1) % 2] == 0
                or cell[cy, (cx + 1) % 2] == 0
            )
    mask_gr = cell_gr[y % 2, x % 2].astype(np.float64)
    mask_gb = mask[1] - mask_gr
    # Roll-derived boundary labels are overwritten from the two-periodic CFA.
    kh = np.asarray([[0.5, 0.0, 0.5]], dtype=np.float64)
    kv = kh.T
    raw_h = _filter(raw, kh)
    raw_v = _filter(raw, kv)

    # axis,color: matching green-site mask.
    green_masks = ((mask_gr, mask_gb), (mask_gb, mask_gr))
    color_masks = (mask[0], mask[2])
    kernels = (kh, kv)
    residual_kernels = (
        np.asarray([[0.5, 1.0, 0.5]]),
        np.asarray([[0.5], [1.0], [0.5]]),
    )
    gradient_kernels = (
        np.asarray([[-1.0, 0.0, 1.0]]),
        np.asarray([[-1.0], [0.0], [1.0]]),
    )
    laplacians = (
        np.asarray([[-1.0, 0.0, 2.0, 0.0, -1.0]]),
        np.asarray([[-1.0], [0.0], [2.0], [0.0], [-1.0]]),
    )
    raw_linear = (raw_h, raw_v)

    candidates = np.empty((2, 2, iterations, height, width), dtype=np.float64)
    criteria = np.empty_like(candidates)
    selected = np.empty((2, 2, height, width), dtype=np.float64)
    minima = np.full((2, 2, height, width), 1e32, dtype=np.float64)
    selected_iterations = np.zeros((2, 2, height, width), dtype=np.int16)

    # Each branch owns independent direction/color guide pairs.
    states: list[list[list[tuple[np.ndarray, np.ndarray]]]] = []
    for _branch in range(2):
        branch_states = []
        for direction in range(2):
            axis_states = []
            for color in range(2):
                gm = green_masks[direction][color]
                cm = color_masks[color]
                gguide = value[1] * gm + raw_linear[direction] * cm
                cguide = value[color * 2] + raw_linear[direction] * gm
                axis_states.append((gguide, cguide))
            branch_states.append(axis_states)
        states.append(branch_states)
    for branch in range(2):
        for direction in range(2):
            selected[branch, direction] = sum(pair[0] for pair in states[branch][direction])

    for iteration in range(iterations):
        ri_h = 2 + iteration
        ri_v = 1 + iteration
        mlri_h = 4 + iteration
        mlri_v = iteration
        for branch in range(2):
            for direction in range(2):
                radii = (
                    (ri_h, ri_v) if direction == 0 else (ri_v, ri_h)
                ) if branch == 0 else (
                    (mlri_h, mlri_v) if direction == 0 else (mlri_v, mlri_h)
                )
                new_pairs = []
                criterion_magnitude = np.zeros((height, width), dtype=np.float64)
                criterion_gradient = np.zeros((height, width), dtype=np.float64)
                for color in range(2):
                    gm = green_masks[direction][color]
                    cm = color_masks[color]
                    line = gm + cm
                    gguide, cguide = states[branch][direction][color]
                    if branch == 0:
                        tentative_g = _guided_ri(
                            cguide, gguide, line, radii[0], radii[1]
                        )
                        tentative_c = _guided_ri(
                            gguide, cguide, line, radii[0], radii[1]
                        )
                    else:
                        lap_g = _filter(gguide, laplacians[direction])
                        lap_c = _filter(cguide, laplacians[direction])
                        tentative_g = _guided_mlri(
                            cguide, gguide, line, lap_c, lap_g, gm,
                            radii[0], radii[1],
                        )
                        tentative_c = _guided_mlri(
                            gguide, cguide, line, lap_g, lap_c, cm,
                            radii[0], radii[1],
                        )
                    residual_g = _filter(
                        (value[1] - tentative_g) * gm,
                        residual_kernels[direction],
                    )
                    residual_c = _filter(
                        (value[color * 2] - tentative_c) * cm,
                        residual_kernels[direction],
                    )
                    reconstructed_g = (tentative_g + residual_g) * cm
                    reconstruction_green_mask = gm
                    if (
                        preserve_reference_vertical_mask_typo
                        and branch == 1
                        and direction == 1
                        and color == 0
                    ):
                        # Sensors ARI v1.0 line 178 uses maskGr for MLRI_Rv,
                        # although the RI branch and the geometry require
                        # maskGb.  Reference mode preserves that authenticated
                        # behavior; X-Trans feasibility uses the corrected
                        # equation and has no Bayer mask names to inherit.
                        reconstruction_green_mask = green_masks[1][1]
                    reconstructed_c = (
                        tentative_c + residual_c
                    ) * reconstruction_green_mask
                    next_g = value[1] * gm + reconstructed_g
                    next_c = value[color * 2] + reconstructed_c
                    delta_g = (gguide - tentative_g) * line
                    delta_c = (cguide - tentative_c) * line
                    criterion_magnitude += (
                        np.abs(delta_g) + np.abs(delta_c)
                    ) * line
                    criterion_gradient += (
                        np.abs(_filter(delta_g, gradient_kernels[direction]))
                        + np.abs(_filter(delta_c, gradient_kernels[direction]))
                    ) * line
                    new_pairs.append((next_g, next_c))
                states[branch][direction] = new_pairs
                candidate = new_pairs[0][0] + new_pairs[1][0]
                axis_criterion = _filter(criterion_magnitude, GAUSSIAN5) ** 2
                axis_criterion *= _filter(criterion_gradient, GAUSSIAN5)
                candidates[branch, direction, iteration] = candidate
                criteria[branch, direction, iteration] = axis_criterion
                improve = axis_criterion < minima[branch, direction]
                selected[branch, direction][improve] = candidate[improve]
                selected_iterations[branch, direction][improve] = iteration
                minima[branch, direction][improve] = axis_criterion[improve]

    weights = 1.0 / (minima + 1e-10)
    adaptive = np.sum(weights * selected, axis=(0, 1)) / (
        np.sum(weights, axis=(0, 1)) + 1e-32
    )
    adaptive = adaptive * (1.0 - mask[1]) + value[1]
    adaptive = np.clip(adaptive, 0.0, 255.0)
    return BayerGreenTrace(
        candidates / 255.0,
        criteria,
        selected_iterations,
        selected / 255.0,
        adaptive / 255.0,
    )

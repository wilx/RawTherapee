"""Offline, missing-green X-Trans adaptation of the published ARI machinery.

This is deliberately not a complete demosaicer.  It transfers ARI's parallel
RI/MLRI branches, growing guided-filter supports, residual reconstruction,
per-iteration criterion, minimum-criterion retention, and reciprocal-criterion
combination to phase-aware horizontal/vertical X-Trans guide pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from tools.xtrans_ari.bayer_reference import (
    GAUSSIAN5,
    _filter,
    _guided_mlri,
    _guided_ri,
)


BRANCHES = ("ri", "mlri")
DIRECTIONS = ("horizontal", "vertical")
ITERATIONS = 11


@dataclass(frozen=True)
class XTransGreenBank:
    candidates: np.ndarray  # branch,direction,iteration,height,width
    criteria: np.ndarray
    selected: np.ndarray  # branch,direction,height,width
    selected_iterations: np.ndarray
    ri_adaptive: np.ndarray
    mlri_adaptive: np.ndarray
    adaptive: np.ndarray
    branch_seconds: tuple[float, float]
    criterion_seconds: float


def _interpolate_observed(
    values: np.ndarray, mask: np.ndarray, axis: int
) -> np.ndarray:
    """Linearly fill a sampled plane along one image axis, preserving samples."""
    source = np.asarray(values, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool)
    if source.shape != observed.shape or source.ndim != 2:
        raise ValueError("directional interpolation planes differ")
    transposed = axis == 0
    work = source.T if transposed else source
    support = observed.T if transposed else observed
    result = np.empty_like(work)
    positions = np.arange(work.shape[1], dtype=np.float64)
    for row in range(work.shape[0]):
        indices = np.flatnonzero(support[row])
        if indices.size == 0:
            raise ValueError("X-Trans line has no observed support")
        result[row] = np.interp(positions, indices, work[row, indices])
    return result.T if transposed else result


def _adaptive(
    selected: np.ndarray, minima: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    weights = 1.0 / (minima + epsilon)
    return np.sum(weights * selected, axis=0) / (np.sum(weights, axis=0) + 1e-32)


def green_candidate_bank(
    scalar_mosaic: np.ndarray,
    cfa: np.ndarray,
    *,
    iterations: int = ITERATIONS,
) -> XTransGreenBank:
    """Generate the corrected X-Trans RI/MLRI green candidate bank.

    The input is normalized linear scalar CFA data.  The implementation runs
    internally in ARI's native 0..255 domain.  Unlike Sensors ARI v1.0 line
    178, branch updates always use the geometrically correct green-site mask.
    """
    scalar = np.asarray(scalar_mosaic, dtype=np.float64)
    phases = np.asarray(cfa, dtype=np.uint8)
    if scalar.ndim != 2 or phases.shape != scalar.shape:
        raise ValueError("mosaic and CFA must have equal two-dimensional shapes")
    if not np.isfinite(scalar).all() or np.any((phases < 0) | (phases > 2)):
        raise ValueError("invalid X-Trans input")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    value = scalar * 255.0
    masks = np.stack([phases == channel for channel in range(3)]).astype(np.float64)
    green_raw = value * masks[1]
    color_raw = (value * masks[0], value * masks[2])
    height, width = value.shape
    candidates = np.empty((2, 2, iterations, height, width), dtype=np.float64)
    criteria = np.empty_like(candidates)
    minima = np.full((2, 2, height, width), 1e32, dtype=np.float64)
    selected = np.zeros((2, 2, height, width), dtype=np.float64)
    selected_iterations = np.zeros((2, 2, height, width), dtype=np.int16)

    laplacians = (
        np.asarray([[-1.0, 0.0, 2.0, 0.0, -1.0]]),
        np.asarray([[-1.0], [0.0], [2.0], [0.0], [-1.0]]),
    )
    gradient_kernels = (
        np.asarray([[-1.0, 0.0, 1.0]]),
        np.asarray([[-1.0], [0.0], [1.0]]),
    )

    # branch,direction,color -> (green guide, color guide)
    states: list[list[list[tuple[np.ndarray, np.ndarray]]]] = []
    for _branch in range(2):
        branch_states = []
        for direction in range(2):
            axis = 1 if direction == 0 else 0
            green_linear = _interpolate_observed(green_raw, masks[1], axis)
            pair_states = []
            for color in range(2):
                cm = masks[color * 2]
                color_linear = _interpolate_observed(color_raw[color], cm, axis)
                pair_states.append((
                    green_raw + green_linear * cm,
                    color_raw[color] + color_linear * masks[1],
                ))
            branch_states.append(pair_states)
        states.append(branch_states)
    initial = green_raw.copy()
    for color in range(2):
        initial += states[0][0][color][0] * masks[color * 2]
    selected[:] = initial

    branch_seconds = [0.0, 0.0]
    criterion_seconds = 0.0
    for iteration in range(iterations):
        ri_h, ri_v = 2 + iteration, 1 + iteration
        mlri_h, mlri_v = 4 + iteration, iteration
        for branch in range(2):
            branch_start = time.perf_counter()
            for direction in range(2):
                axis = 1 if direction == 0 else 0
                horizontal, vertical = (
                    (ri_h, ri_v) if direction == 0 else (ri_v, ri_h)
                ) if branch == 0 else (
                    (mlri_h, mlri_v) if direction == 0 else (mlri_v, mlri_h)
                )
                next_pairs = []
                criterion_magnitude = np.zeros((height, width), dtype=np.float64)
                criterion_gradient = np.zeros((height, width), dtype=np.float64)
                for color in range(2):
                    cm = masks[color * 2]
                    line = masks[1] + cm
                    green_guide, color_guide = states[branch][direction][color]
                    if branch == 0:
                        tentative_green = _guided_ri(
                            color_guide, green_guide, line, horizontal, vertical
                        )
                        tentative_color = _guided_ri(
                            green_guide, color_guide, line, horizontal, vertical
                        )
                    else:
                        lap_green = _filter(green_guide, laplacians[direction])
                        lap_color = _filter(color_guide, laplacians[direction])
                        tentative_green = _guided_mlri(
                            color_guide, green_guide, line,
                            lap_color, lap_green, masks[1], horizontal, vertical,
                        )
                        tentative_color = _guided_mlri(
                            green_guide, color_guide, line,
                            lap_green, lap_color, cm, horizontal, vertical,
                        )
                    green_residual = (green_raw - tentative_green) * masks[1]
                    color_residual = (color_raw[color] - tentative_color) * cm
                    green_residual = _interpolate_observed(
                        green_residual, masks[1], axis
                    )
                    color_residual = _interpolate_observed(color_residual, cm, axis)
                    reconstructed_green = (
                        tentative_green + green_residual
                    ) * cm
                    reconstructed_color = (
                        tentative_color + color_residual
                    ) * masks[1]
                    next_pairs.append((
                        green_raw + reconstructed_green,
                        color_raw[color] + reconstructed_color,
                    ))
                    delta_green = (green_guide - tentative_green) * line
                    delta_color = (color_guide - tentative_color) * line
                    criterion_magnitude += np.abs(delta_green) + np.abs(delta_color)
                    criterion_gradient += (
                        np.abs(_filter(delta_green, gradient_kernels[direction]))
                        + np.abs(_filter(delta_color, gradient_kernels[direction]))
                    )
                states[branch][direction] = next_pairs
                candidate = green_raw.copy()
                candidate += next_pairs[0][0] * masks[0]
                candidate += next_pairs[1][0] * masks[2]
                criterion_start = time.perf_counter()
                score = _filter(criterion_magnitude, GAUSSIAN5) ** 2
                score *= _filter(criterion_gradient, GAUSSIAN5)
                criterion_seconds += time.perf_counter() - criterion_start
                candidates[branch, direction, iteration] = candidate
                criteria[branch, direction, iteration] = score
                improve = score < minima[branch, direction]
                selected[branch, direction][improve] = candidate[improve]
                selected_iterations[branch, direction][improve] = iteration
                minima[branch, direction][improve] = score[improve]
            branch_seconds[branch] += time.perf_counter() - branch_start

    ri_adaptive = _adaptive(selected[0], minima[0])
    mlri_adaptive = _adaptive(selected[1], minima[1])
    adaptive = _adaptive(selected.reshape(4, height, width), minima.reshape(4, height, width))
    for output in (ri_adaptive, mlri_adaptive, adaptive):
        output *= 1.0 - masks[1]
        output += green_raw
        np.clip(output, 0.0, 255.0, out=output)
    if not all(np.isfinite(array).all() for array in (candidates, criteria, adaptive)):
        raise FloatingPointError("ARI candidate bank produced a non-finite value")
    return XTransGreenBank(
        candidates / 255.0,
        criteria,
        selected / 255.0,
        selected_iterations,
        ri_adaptive / 255.0,
        mlri_adaptive / 255.0,
        adaptive / 255.0,
        (branch_seconds[0], branch_seconds[1]),
        criterion_seconds,
    )

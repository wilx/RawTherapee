"""Phase-specific linear estimators operating on physical X-Trans samples.

This is deliberately an offline research implementation.  It follows the
localized joint spatial-chromatic estimator of Portilla, Otaduy, and
Dorronsoro (ICIP 2005): each target CFA phase has its own small linear map
from the scalar mosaic neighborhood to RGB.  The physically observed target
component is restored after prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tools.xtrans_mlri_internal.dataset import MLRI_CFA, cfa_for_origin, origin_cells


LC_BASIS = np.asarray(
    (
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
        (1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0)),
        (1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class PhaseStatistics:
    count: int
    x_mean: np.ndarray
    target_mean: np.ndarray
    covariance: np.ndarray
    cross_covariance: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float


@dataclass(frozen=True)
class FilterBank:
    support: int
    ridge_ratio: float
    basis: str
    dc_mode: str
    x_means: np.ndarray
    target_means: np.ndarray
    weights: np.ndarray
    condition_numbers: np.ndarray
    minimum_eigenvalues: np.ndarray
    filter_norms: np.ndarray
    sample_counts: np.ndarray


def _check_support(support: int) -> int:
    if support < 3 or support % 2 != 1:
        raise ValueError("support must be odd and at least three")
    return support


def phase_origins() -> tuple[tuple[int, int], ...]:
    return origin_cells()


def phase_index_table() -> np.ndarray:
    origins = phase_origins()
    by_cell = {
        cfa_for_origin(6, 6, origin_x, origin_y).tobytes(): index
        for index, (origin_x, origin_y) in enumerate(origins)
    }
    result = np.empty((6, 6), dtype=np.uint8)
    for y in range(6):
        for x in range(6):
            key = cfa_for_origin(6, 6, x, y).tobytes()
            result[y, x] = by_cell[key]
    if set(int(value) for value in result.ravel()) != set(range(18)):
        raise RuntimeError("invalid 18-phase X-Trans lookup")
    return result


PHASE_TABLE = phase_index_table()


def phase_map(
    height: int, width: int, origin_x: int = 0, origin_y: int = 0
) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return PHASE_TABLE[(y + origin_y) % 6, (x + origin_x) % 6]


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


def _mosaic(rgb: np.ndarray, origin_x: int, origin_y: int) -> tuple[np.ndarray, np.ndarray]:
    checked = _check_rgb(rgb)
    cfa = cfa_for_origin(checked.shape[1], checked.shape[2], origin_x, origin_y)
    scalar = np.take_along_axis(
        np.moveaxis(checked, 0, -1), cfa[..., None], axis=2
    )[..., 0]
    return scalar, cfa


def _windows(scalar: np.ndarray, support: int) -> np.ndarray:
    radius = _check_support(support) // 2
    padded = np.pad(np.asarray(scalar, dtype=np.float64), radius, mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, (support, support))


def positions_for_phase(
    shape: tuple[int, int],
    phase: int,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    maximum: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("invalid sample region")
    phases = phase_map(height, width, origin_x, origin_y)
    local_y, local_x = np.nonzero(phases[y0:y1, x0:x1] == phase)
    ys = local_y.astype(np.int64) + y0
    xs = local_x.astype(np.int64) + x0
    if maximum is not None and ys.size > maximum:
        selected = np.linspace(0, ys.size - 1, maximum, dtype=np.int64)
        ys = ys[selected]
        xs = xs[selected]
    return ys, xs


def collect_phase_samples(
    rgb: np.ndarray,
    support: int,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    maximum_per_phase: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    checked = _check_rgb(rgb)
    scalar, _ = _mosaic(checked, origin_x, origin_y)
    windows = _windows(scalar, support)
    x0, y0, x1, y1 = region
    phases = phase_map(checked.shape[1], checked.shape[2], origin_x, origin_y)
    region_phases = phases[y0:y1, x0:x1]
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for phase in range(18):
        local_y, local_x = np.nonzero(region_phases == phase)
        ys = local_y.astype(np.int64) + y0
        xs = local_x.astype(np.int64) + x0
        if maximum_per_phase is not None and ys.size > maximum_per_phase:
            selected = np.linspace(
                0, ys.size - 1, maximum_per_phase, dtype=np.int64
            )
            ys = ys[selected]
            xs = xs[selected]
        observations.append(
            np.ascontiguousarray(windows[ys, xs].reshape(ys.size, support * support))
        )
        targets.append(np.ascontiguousarray(checked[:, ys, xs].T))
    return observations, targets


def phase_statistics(
    observations: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
    *,
    basis: str = "rgb",
    dc_mode: str = "m0",
    support: int | None = None,
) -> tuple[PhaseStatistics, ...]:
    if basis not in ("rgb", "lc"):
        raise ValueError("basis must be rgb or lc")
    if dc_mode not in ("m0", "m1", "m2"):
        raise ValueError("dc_mode must be m0, m1, or m2")
    if dc_mode == "m2" and support is None:
        raise ValueError("M2 requires the filter support")
    rows = []
    for phase, (x_values, target_values) in enumerate(zip(observations, targets)):
        x = np.asarray(x_values, dtype=np.float64)
        target = np.asarray(target_values, dtype=np.float64)
        if x.ndim != 2 or target.shape != (x.shape[0], 3) or x.shape[0] < 2:
            raise ValueError("invalid phase samples")
        x, target, _ = _dc_transform(x, target, phase, dc_mode, support)
        if basis == "lc":
            target = target @ LC_BASIS.T
        x_mean = np.mean(x, axis=0)
        target_mean = np.mean(target, axis=0)
        centered_x = x - x_mean
        centered_target = target - target_mean
        denominator = float(x.shape[0] - 1)
        covariance = centered_x.T @ centered_x / denominator
        covariance = (covariance + covariance.T) * 0.5
        cross = centered_target.T @ centered_x / denominator
        eigenvalues = np.linalg.eigvalsh(covariance)
        rows.append(
            PhaseStatistics(
                count=x.shape[0],
                x_mean=x_mean,
                target_mean=target_mean,
                covariance=covariance,
                cross_covariance=cross,
                minimum_eigenvalue=float(eigenvalues[0]),
                maximum_eigenvalue=float(eigenvalues[-1]),
            )
        )
    if len(rows) != 18:
        raise ValueError("expected exactly 18 phase sample sets")
    return tuple(rows)


def derive_filter_bank(
    statistics: Iterable[PhaseStatistics],
    support: int,
    ridge_ratio: float,
    *,
    basis: str = "rgb",
    dc_mode: str = "m0",
) -> FilterBank:
    _check_support(support)
    if ridge_ratio < 0.0 or not np.isfinite(ridge_ratio):
        raise ValueError("invalid ridge ratio")
    rows = tuple(statistics)
    if len(rows) != 18:
        raise ValueError("expected 18 phase statistics")
    feature_count = support * support
    x_means = np.empty((18, feature_count), dtype=np.float64)
    target_means = np.empty((18, 3), dtype=np.float64)
    weights = np.empty((18, 3, feature_count), dtype=np.float64)
    conditions = np.empty(18, dtype=np.float64)
    minimum_eigenvalues = np.empty(18, dtype=np.float64)
    norms = np.empty((18, 3), dtype=np.float64)
    counts = np.empty(18, dtype=np.int64)
    for phase, row in enumerate(rows):
        scale = float(np.trace(row.covariance) / feature_count)
        ridge = ridge_ratio * max(scale, np.finfo(np.float64).tiny)
        system = row.covariance + ridge * np.eye(feature_count)
        # Solving the transposed normal equation avoids explicitly forming an inverse.
        weight = np.linalg.solve(system, row.cross_covariance.T).T
        if not np.isfinite(weight).all():
            raise ValueError("non-finite LMMSE filter")
        x_means[phase] = row.x_mean
        target_means[phase] = row.target_mean
        weights[phase] = weight
        minimum_eigenvalues[phase] = row.minimum_eigenvalue
        adjusted_minimum = row.minimum_eigenvalue + ridge
        adjusted_maximum = row.maximum_eigenvalue + ridge
        conditions[phase] = adjusted_maximum / max(
            adjusted_minimum, np.finfo(np.float64).tiny
        )
        norms[phase] = np.linalg.norm(weight, axis=1)
        counts[phase] = row.count
    return FilterBank(
        support=support,
        ridge_ratio=ridge_ratio,
        basis=basis,
        dc_mode=dc_mode,
        x_means=x_means,
        target_means=target_means,
        weights=weights,
        condition_numbers=conditions,
        minimum_eigenvalues=minimum_eigenvalues,
        filter_norms=norms,
        sample_counts=counts,
    )


def predict_region(
    bank: FilterBank,
    rgb: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    clip: bool = False,
) -> np.ndarray:
    checked = _check_rgb(rgb)
    scalar, cfa = _mosaic(checked, origin_x, origin_y)
    windows = _windows(scalar, bank.support)
    x0, y0, x1, y1 = region
    output = np.empty((3, y1 - y0, x1 - x0), dtype=np.float64)
    for phase in range(18):
        ys, xs = positions_for_phase(
            scalar.shape,
            phase,
            region,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        x = windows[ys, xs].reshape(ys.size, bank.support * bank.support)
        x, _, dc = _dc_transform(
            x, None, phase, bank.dc_mode, bank.support
        )
        predicted = bank.target_means[phase] + (
            x - bank.x_means[phase]
        ) @ bank.weights[phase].T
        if bank.basis == "lc":
            predicted = predicted @ LC_BASIS
        predicted += dc
        local_y = ys - y0
        local_x = xs - x0
        output[:, local_y, local_x] = predicted.T
    local_cfa = cfa[y0:y1, x0:x1]
    local_scalar = scalar[y0:y1, x0:x1]
    for channel in range(3):
        selected = local_cfa == channel
        output[channel, selected] = local_scalar[selected]
    if clip:
        np.clip(output, 0.0, 1.0, out=output)
        for channel in range(3):
            selected = local_cfa == channel
            output[channel, selected] = local_scalar[selected]
    if not np.isfinite(output).all():
        raise ValueError("non-finite LMMSE reconstruction")
    return output


def observation_colors(support: int, phase: int) -> np.ndarray:
    _check_support(support)
    if not 0 <= phase < 18:
        raise ValueError("invalid X-Trans phase")
    origin_x, origin_y = phase_origins()[phase]
    radius = support // 2
    colors = np.empty((support, support), dtype=np.uint8)
    for y in range(support):
        for x in range(support):
            colors[y, x] = MLRI_CFA[
                (origin_y + y - radius) % 6,
                (origin_x + x - radius) % 6,
            ]
    return colors.ravel()


def _dc_transform(
    observations: np.ndarray,
    targets: np.ndarray | None,
    phase: int,
    mode: str,
    support: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    x = np.asarray(observations, dtype=np.float64)
    if mode == "m0":
        base = np.zeros((x.shape[0], 3), dtype=np.float64)
        return x, targets, base
    if mode == "m1":
        mean = np.mean(x, axis=1, keepdims=True)
        base = np.repeat(mean, 3, axis=1)
        transformed_target = None if targets is None else targets - base
        return x - mean, transformed_target, base
    if mode != "m2" or support is None:
        raise ValueError("invalid DC transformation")
    colors = observation_colors(support, phase)
    base = np.empty((x.shape[0], 3), dtype=np.float64)
    transformed_x = np.empty_like(x)
    for channel in range(3):
        selected = colors == channel
        if not np.any(selected):
            raise ValueError("support has no observed sample for a color")
        mean = np.mean(x[:, selected], axis=1)
        base[:, channel] = mean
        transformed_x[:, selected] = x[:, selected] - mean[:, None]
    transformed_target = None if targets is None else targets - base
    return transformed_x, transformed_target, base


def geometry_interpolate_region(
    rgb: np.ndarray,
    support: int,
    region: tuple[int, int, int, int],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
) -> np.ndarray:
    """Inverse-distance same-color interpolation with exact sample retention."""

    checked = _check_rgb(rgb)
    scalar, cfa = _mosaic(checked, origin_x, origin_y)
    radius = _check_support(support) // 2
    padded_scalar = np.pad(scalar, radius, mode="reflect")
    padded_cfa = np.pad(cfa, radius, mode="reflect")
    x0, y0, x1, y1 = region
    height = y1 - y0
    width = x1 - x0
    output = np.zeros((3, height, width), dtype=np.float64)
    normalizer = np.zeros_like(output)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = np.hypot(dx, dy)
            weight = 1.0 if distance == 0.0 else 1.0 / distance
            source = padded_scalar[
                radius + y0 + dy : radius + y1 + dy,
                radius + x0 + dx : radius + x1 + dx,
            ]
            colors = padded_cfa[
                radius + y0 + dy : radius + y1 + dy,
                radius + x0 + dx : radius + x1 + dx,
            ]
            for channel in range(3):
                selected = colors == channel
                output[channel, selected] += weight * source[selected]
                normalizer[channel, selected] += weight
    if np.any(normalizer == 0.0):
        raise ValueError("geometry support contains no sample for a color")
    output /= normalizer
    local_cfa = cfa[y0:y1, x0:x1]
    local_scalar = scalar[y0:y1, x0:x1]
    for channel in range(3):
        selected = local_cfa == channel
        output[channel, selected] = local_scalar[selected]
    return output


def native_sample_error(
    reconstruction: np.ndarray, rgb: np.ndarray, cfa: np.ndarray
) -> float:
    truth = _check_rgb(rgb)
    checked = np.asarray(reconstruction, dtype=np.float64)
    return float(
        max(
            np.max(np.abs(checked[channel, cfa == channel] - truth[channel, cfa == channel]))
            for channel in range(3)
        )
    )

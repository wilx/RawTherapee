"""Observed-CFA patch matching and nonlocal reconstruction controls.

The same-phase study deliberately stops at an information upper bound: every
same-phase patch has exactly the same missing rows, so neither direct donation
nor unconstrained low-rank completion can identify those rows.  Cross-phase
matching is therefore the first implementable reconstruction control.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from tools.xtrans_alias.analysis import CANONICAL_XTRANS


PATCH_SIZE = 7
PATCH_RADIUS = PATCH_SIZE // 2
SEARCH_RADIUS = 72
NEIGHBOR_COUNT = 32
TARGET_STEP = 24
TARGET_MARGIN = 24
LOW_RANK = 4
LOW_RANK_ITERATIONS = 10
EPSILON = 1e-12


def cfa_for_context(height: int, width: int, global_origin: tuple[int, int]) -> np.ndarray:
    x0, y0 = global_origin
    y, x = np.mgrid[:height, :width]
    return CANONICAL_XTRANS[(y + y0) % 6, (x + x0) % 6].astype(np.uint8)


def mosaic(truth: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    checked = np.asarray(truth, dtype=np.float64)
    masks = np.asarray(cfa, dtype=np.uint8)
    if checked.ndim != 3 or checked.shape[0] != 3 or masks.shape != checked.shape[1:]:
        raise ValueError("truth/CFA shape mismatch")
    if not np.isfinite(checked).all() or np.any((masks < 0) | (masks > 2)):
        raise ValueError("invalid truth or CFA")
    return np.take_along_axis(np.moveaxis(checked, 0, -1), masks[..., None], axis=2)[..., 0]


def target_centers(target_origin: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x0, y0 = target_origin
    positions = range(TARGET_MARGIN, 168 - TARGET_MARGIN, TARGET_STEP)
    return tuple((y0 + y, x0 + x) for y in positions for x in positions)


def _candidate_centers(
    shape: tuple[int, int], center: tuple[int, int], *, same_phase: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    cy, cx = center
    y0 = max(PATCH_RADIUS, cy - SEARCH_RADIUS)
    y1 = min(height - PATCH_RADIUS - 1, cy + SEARCH_RADIUS)
    x0 = max(PATCH_RADIUS, cx - SEARCH_RADIUS)
    x1 = min(width - PATCH_RADIUS - 1, cx + SEARCH_RADIUS)
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    separated = np.maximum(np.abs(yy - cy), np.abs(xx - cx)) >= PATCH_SIZE
    if same_phase:
        separated &= (yy % 6 == cy % 6) & (xx % 6 == cx % 6)
    return yy[separated].astype(np.int32), xx[separated].astype(np.int32)


def _patches(values: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    checked = np.asarray(values)
    if checked.ndim == 2:
        windows = np.lib.stride_tricks.sliding_window_view(checked, (PATCH_SIZE, PATCH_SIZE))
        return windows[ys - PATCH_RADIUS, xs - PATCH_RADIUS]
    if checked.ndim == 3:
        windows = np.lib.stride_tricks.sliding_window_view(
            checked, (PATCH_SIZE, PATCH_SIZE), axis=(1, 2)
        )
        return np.moveaxis(windows[:, ys - PATCH_RADIUS, xs - PATCH_RADIUS], 0, 1)
    raise ValueError("patch source must be 2D or CHW")


def _one_patch(values: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    y, x = center
    if values.ndim == 2:
        return values[y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1]
    return values[:, y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.ptp(left) <= EPSILON or np.ptp(right) <= EPSILON:
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def _weights(distances: np.ndarray) -> np.ndarray:
    checked = np.asarray(distances, dtype=np.float64)
    positive = checked[checked > EPSILON]
    scale = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-checked / max(scale, EPSILON))
    total = float(np.sum(weights))
    return np.full(checked.shape, 1.0 / checked.size) if total <= EPSILON else weights / total


def _enforce_observed(result: np.ndarray, scalar: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    output = np.asarray(result, dtype=np.float64).copy()
    for channel in range(3):
        mask = cfa == channel
        output[channel, mask] = scalar[mask]
    return output


def direct_nonlocal(
    target_scalar: np.ndarray,
    target_cfa: np.ndarray,
    candidate_scalar: np.ndarray,
    candidate_cfa: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate missing samples only from physically observed donor samples."""

    selected_scalar = candidate_scalar[indices]
    selected_cfa = candidate_cfa[indices]
    weights = _weights(distances[indices])
    result = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    covered = 0
    missing = 0
    for channel in range(3):
        for y in range(PATCH_SIZE):
            for x in range(PATCH_SIZE):
                if target_cfa[y, x] == channel:
                    result[channel, y, x] = target_scalar[y, x]
                    continue
                missing += 1
                available = selected_cfa[:, y, x] == channel
                if np.any(available):
                    local_weights = weights[available]
                    result[channel, y, x] = float(
                        np.sum(local_weights * selected_scalar[available, y, x])
                        / np.sum(local_weights)
                    )
                    covered += 1
                else:
                    result[channel, y, x] = target_scalar[y, x]
    return _enforce_observed(result, target_scalar, target_cfa), covered / max(1, missing)


def low_rank_completion(
    target_scalar: np.ndarray,
    target_cfa: np.ndarray,
    candidate_scalar: np.ndarray,
    candidate_cfa: np.ndarray,
    indices: np.ndarray,
    *,
    rank: int = LOW_RANK,
    iterations: int = LOW_RANK_ITERATIONS,
) -> tuple[np.ndarray, float]:
    """Iterative rank projection with exact CFA sample reinjection."""

    selected_scalar = candidate_scalar[indices]
    selected_cfa = candidate_cfa[indices]
    columns = len(indices) + 1
    values = np.zeros((3, PATCH_SIZE, PATCH_SIZE, columns), dtype=np.float64)
    known = np.zeros(values.shape, dtype=bool)
    for channel in range(3):
        target_mask = target_cfa == channel
        values[channel, target_mask, 0] = target_scalar[target_mask]
        known[channel, target_mask, 0] = True
        for column in range(len(indices)):
            mask = selected_cfa[column] == channel
            values[channel, mask, column + 1] = selected_scalar[column, mask]
            known[channel, mask, column + 1] = True
    matrix = values.reshape(3 * PATCH_SIZE * PATCH_SIZE, columns)
    observed = known.reshape(matrix.shape)
    row_counts = np.sum(observed, axis=1)
    coverage = float(np.mean(row_counts > 0))
    channel_means = []
    for channel in range(3):
        channel_values = matrix[channel * PATCH_SIZE * PATCH_SIZE:(channel + 1) * PATCH_SIZE * PATCH_SIZE]
        channel_known = observed[channel * PATCH_SIZE * PATCH_SIZE:(channel + 1) * PATCH_SIZE * PATCH_SIZE]
        channel_means.append(float(np.mean(channel_values[channel_known])))
    for row in range(matrix.shape[0]):
        channel = row // (PATCH_SIZE * PATCH_SIZE)
        fill = (
            float(np.mean(matrix[row, observed[row]]))
            if row_counts[row]
            else channel_means[channel]
        )
        matrix[row, ~observed[row]] = fill
    fixed = matrix.copy()
    for _ in range(iterations):
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        retained = min(rank, singular.size)
        projected = (u[:, :retained] * singular[:retained]) @ vh[:retained]
        matrix[~observed] = projected[~observed]
        matrix[observed] = fixed[observed]
    result = matrix[:, 0].reshape(3, PATCH_SIZE, PATCH_SIZE)
    return _enforce_observed(result, target_scalar, target_cfa), coverage


def reconstruction_error(result: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    delta = np.asarray(result, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    mse = float(np.mean(delta * delta))
    return {
        "mse": mse,
        "rms": math.sqrt(mse),
        "psnr_db": math.inf if mse == 0 else 10 * math.log10(1.0 / mse),
        "maximum_absolute": float(np.max(np.abs(delta))),
    }


def classify_patch(truth: np.ndarray) -> str:
    checked = np.asarray(truth, dtype=np.float64)
    luma = np.mean(checked, axis=0)
    gy, gx = np.gradient(luma)
    energy = gx * gx + gy * gy
    total = float(np.sum(energy))
    if math.sqrt(float(np.mean(energy))) < 0.008:
        return "smooth"
    top = np.sort(energy.ravel())[-4:]
    if total > EPSILON and float(np.sum(top) / total) > 0.55:
        return "sparse point"
    jxx, jyy, jxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    discriminant = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4 * jxy * jxy))
    coherence = discriminant / (jxx + jyy + EPSILON)
    if coherence > 0.72:
        return "edge"
    if float(np.max(checked) - np.min(checked)) > 0.8:
        return "saturated detail"
    return "texture/corner"


def _summary_indices(distances: np.ndarray, count: int = NEIGHBOR_COUNT) -> np.ndarray:
    return np.argsort(distances, kind="stable")[:min(count, distances.size)]


def _donor_average(
    candidate_rgb: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    target_scalar: np.ndarray,
    target_cfa: np.ndarray,
) -> np.ndarray:
    weights = _weights(distances[indices])
    result = np.sum(candidate_rgb[indices] * weights[:, None, None, None], axis=0)
    return _enforce_observed(result, target_scalar, target_cfa)


def analyze_target(
    truth: np.ndarray,
    scalar: np.ndarray,
    cfa: np.ndarray,
    center: tuple[int, int],
    baselines: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target_truth = _one_patch(truth, center)
    target_scalar = _one_patch(scalar, center)
    target_cfa = _one_patch(cfa, center)
    errors = {
        name: reconstruction_error(_one_patch(values, center), target_truth)
        for name, values in baselines.items()
    }
    reconstructions: dict[str, np.ndarray] = {}

    same_y, same_x = _candidate_centers(scalar.shape, center, same_phase=True)
    same_scalar = _patches(scalar, same_y, same_x)
    same_rgb = _patches(truth, same_y, same_x)
    same_cfa = _patches(cfa, same_y, same_x)
    same_cfa_distance = np.mean((same_scalar - target_scalar) ** 2, axis=(1, 2))
    same_rgb_distance = np.mean((same_rgb - target_truth) ** 2, axis=(1, 2, 3))
    same_observable = _summary_indices(same_cfa_distance)
    same_oracle = _summary_indices(same_rgb_distance)
    same_observable_reconstruction = _donor_average(
        same_rgb, same_observable, same_cfa_distance, target_scalar, target_cfa
    )
    same_oracle_reconstruction = _donor_average(
        same_rgb, same_oracle, same_rgb_distance, target_scalar, target_cfa
    )
    for label, indices, distances in (
        ("observable", same_observable, same_cfa_distance),
        ("oracle", same_oracle, same_rgb_distance),
    ):
        for count, suffix in ((1, "best1"), (8, "top8"), (NEIGHBOR_COUNT, "top32")):
            reconstruction = _donor_average(
                same_rgb, indices[:count], distances, target_scalar, target_cfa
            )
            name = f"same_phase_{label}_{suffix}_donor_upper_bound"
            errors[name] = reconstruction_error(reconstruction, target_truth)
            reconstructions[name] = reconstruction

    cross_y, cross_x = _candidate_centers(scalar.shape, center, same_phase=False)
    cross_scalar = _patches(scalar, cross_y, cross_x)
    cross_rgb = _patches(truth, cross_y, cross_x)
    cross_cfa = _patches(cfa, cross_y, cross_x)
    comparable = cross_cfa == target_cfa[None]
    counts = np.sum(comparable, axis=(1, 2))
    valid = counts >= 8
    cross_scalar, cross_rgb, cross_cfa = cross_scalar[valid], cross_rgb[valid], cross_cfa[valid]
    cross_y, cross_x, comparable, counts = (
        cross_y[valid], cross_x[valid], comparable[valid], counts[valid]
    )
    cross_cfa_distance = np.sum(
        (cross_scalar - target_scalar) ** 2 * comparable, axis=(1, 2)
    ) / counts
    cross_rgb_distance = np.mean((cross_rgb - target_truth) ** 2, axis=(1, 2, 3))
    cross_observable = _summary_indices(cross_cfa_distance)
    cross_oracle = _summary_indices(cross_rgb_distance)
    direct_observable, direct_coverage = direct_nonlocal(
        target_scalar, target_cfa, cross_scalar, cross_cfa,
        cross_observable, cross_cfa_distance,
    )
    direct_oracle, _ = direct_nonlocal(
        target_scalar, target_cfa, cross_scalar, cross_cfa,
        cross_oracle, cross_rgb_distance,
    )
    lowrank_observable, lowrank_coverage = low_rank_completion(
        target_scalar, target_cfa, cross_scalar, cross_cfa, cross_observable
    )
    lowrank_oracle, _ = low_rank_completion(
        target_scalar, target_cfa, cross_scalar, cross_cfa, cross_oracle
    )
    for label, indices, distances in (
        ("observable", cross_observable, cross_cfa_distance),
        ("oracle", cross_oracle, cross_rgb_distance),
    ):
        for count, suffix in ((1, "best1"), (8, "top8"), (NEIGHBOR_COUNT, "top32")):
            reconstruction = _donor_average(
                cross_rgb, indices[:count], distances, target_scalar, target_cfa
            )
            name = f"cross_phase_{label}_{suffix}_donor_upper_bound"
            errors[name] = reconstruction_error(reconstruction, target_truth)
            reconstructions[name] = reconstruction
    for name, values in (
        ("cross_phase_observable_direct_nlm", direct_observable),
        ("cross_phase_oracle_direct_nlm", direct_oracle),
        ("cross_phase_observable_low_rank", lowrank_observable),
        ("cross_phase_oracle_low_rank", lowrank_oracle),
    ):
        errors[name] = reconstruction_error(values, target_truth)
        reconstructions[name] = values

    same_phase_patterns = {patch.tobytes() for patch in same_cfa}
    cross_phase_patterns = {patch.tobytes() for patch in cross_cfa}
    oracle_overlap = len(set(same_observable[:8]) & set(same_oracle[:8])) / 8.0
    row = {
        "center_yx": list(center),
        "texture_class": classify_patch(target_truth),
        "errors": errors,
        "same_phase": {
            "candidate_count": int(same_cfa_distance.size),
            "phase_pattern_count": len(same_phase_patterns),
            "cfa_rgb_spearman": _spearman(same_cfa_distance, same_rgb_distance),
            "best_cfa_selected_rgb_mse": float(same_rgb_distance[same_observable[0]]),
            "best_oracle_rgb_mse": float(same_rgb_distance[same_oracle[0]]),
            "top8_oracle_overlap_fraction": oracle_overlap,
            "corresponding_missing_color_donor_coverage": 0.0,
        },
        "cross_phase": {
            "candidate_count": int(cross_cfa_distance.size),
            "phase_pattern_count": len(cross_phase_patterns),
            "cfa_rgb_spearman": _spearman(cross_cfa_distance, cross_rgb_distance),
            "best_cfa_selected_rgb_mse": float(cross_rgb_distance[cross_observable[0]]),
            "best_oracle_rgb_mse": float(cross_rgb_distance[cross_oracle[0]]),
            "direct_donor_coverage": direct_coverage,
            "matrix_row_coverage": lowrank_coverage,
        },
    }
    return row, reconstructions


def pooled_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    names = tuple(rows[0]["errors"])
    result = {}
    for name in names:
        mse = np.asarray([row["errors"][name]["mse"] for row in rows], dtype=np.float64)
        pooled = float(np.mean(mse))
        result[name] = {
            "mean_patch_mse": pooled,
            "pooled_psnr_db": math.inf if pooled == 0 else 10 * math.log10(1 / pooled),
            "median_patch_rms": float(np.median(np.sqrt(mse))),
            "p95_patch_rms": float(np.percentile(np.sqrt(mse), 95)),
            "improved_over_markesteijn_fraction": float(np.mean(
                mse < np.asarray([row["errors"]["markesteijn"]["mse"] for row in rows])
            )),
        }
    return result


def matching_metrics(rows: Sequence[Mapping[str, object]], phase: str) -> dict[str, float]:
    purity = np.asarray([row[phase]["best_cfa_selected_rgb_mse"] for row in rows])
    oracle = np.asarray([row[phase]["best_oracle_rgb_mse"] for row in rows])
    ratio = purity / np.maximum(oracle, EPSILON)
    return {
        "mean_cfa_rgb_spearman": float(np.mean([row[phase]["cfa_rgb_spearman"] for row in rows])),
        "median_best_cfa_selected_rgb_rms": float(np.median(np.sqrt(purity))),
        "median_best_oracle_rgb_rms": float(np.median(np.sqrt(oracle))),
        "median_selected_to_oracle_mse_ratio": float(np.median(ratio)),
        "useful_recurrence_fraction_rgb_rms_below_0_02": float(np.mean(np.sqrt(purity) < 0.02)),
        "useful_recurrence_fraction_rgb_rms_below_0_05": float(np.mean(np.sqrt(purity) < 0.05)),
    }

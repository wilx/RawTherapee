"""Metrics for the ARI X-Trans candidate-bank feasibility experiment."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import ndimage, stats


MARGIN = 16


def _psnr(rms: float) -> float:
    return math.inf if rms == 0.0 else float(-20.0 * math.log10(rms))


def green_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    cfa: np.ndarray,
    margin: int = MARGIN,
) -> dict[str, float]:
    target = np.asarray(truth, dtype=np.float64)
    output = np.asarray(reconstruction, dtype=np.float64)
    phases = np.asarray(cfa)
    if target.shape != output.shape or target.shape != phases.shape:
        raise ValueError("green metric planes differ")
    interior = np.zeros_like(phases, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    missing = interior & (phases != 1)
    error = output[missing] - target[missing]
    absolute = np.abs(error)
    rms = float(np.sqrt(np.mean(error**2)))
    return {
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": _psnr(rms),
        "rms": rms,
    }


def pooled_green_metrics(
    cases: Sequence[dict[str, object]], method: str, margin: int = MARGIN
) -> dict[str, float]:
    errors = []
    for case in cases:
        truth = np.asarray(case["truth"])[1]
        output = np.asarray(case["green_outputs"][method])
        cfa = np.asarray(case["cfa"])
        interior = np.zeros_like(cfa, dtype=bool)
        interior[margin:-margin, margin:-margin] = True
        missing = interior & (cfa != 1)
        errors.append((output - truth)[missing])
    error = np.concatenate(errors)
    absolute = np.abs(error)
    rms = float(np.sqrt(np.mean(error**2)))
    return {
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": _psnr(rms),
        "rms": rms,
    }


def label_spatial_metrics(
    labels: np.ndarray,
    cfa: np.ndarray,
    *,
    class_divisor: int = 1,
    reference: np.ndarray | None = None,
    margin: int = MARGIN,
) -> dict[str, float | int]:
    """Measure coherence of an oracle label map on missing-green samples.

    ``class_divisor`` collapses candidate labels into a coarser semantic class,
    for example combined-bank labels into RI/MLRI branch labels. Components
    use 8-connectivity so X-Trans's diagonally adjacent missing-green sites
    belong to the same spatial region rather than artificial one-pixel islands.
    """
    raw = np.asarray(labels)
    phases = np.asarray(cfa)
    if raw.shape != phases.shape or class_divisor < 1:
        raise ValueError("label metric contract differs")
    classes = raw // class_divisor
    interior = np.zeros_like(phases, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    valid = interior & (phases != 1)
    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    pair_count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    differences = int(
        np.count_nonzero((classes[:, :-1] != classes[:, 1:]) & horizontal)
        + np.count_nonzero((classes[:-1, :] != classes[1:, :]) & vertical)
    )

    component_sizes: list[int] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for value in np.unique(classes[valid]):
        components, count = ndimage.label((classes == value) & valid, structure)
        if count:
            sizes = np.bincount(components.ravel())[1:]
            component_sizes.extend(int(size) for size in sizes)
    sizes_array = np.asarray(component_sizes, dtype=np.float64)
    result: dict[str, float | int] = {
        "boundary_density": float(differences / pair_count) if pair_count else 0.0,
        "component_count": len(component_sizes),
        "component_maximum_pixels": int(np.max(sizes_array)) if sizes_array.size else 0,
        "component_mean_pixels": float(np.mean(sizes_array)) if sizes_array.size else 0.0,
        "component_median_pixels": float(np.median(sizes_array)) if sizes_array.size else 0.0,
        "component_p90_pixels": float(np.quantile(sizes_array, 0.90)) if sizes_array.size else 0.0,
        "pixel_count": int(np.count_nonzero(valid)),
        "singleton_component_fraction": (
            float(np.mean(sizes_array == 1.0)) if sizes_array.size else 0.0
        ),
    }
    if reference is not None:
        baseline = np.asarray(reference)
        if baseline.shape != raw.shape:
            raise ValueError("reference label map shape differs")
        result["agreement_with_pixel_oracle"] = float(
            np.mean(classes[valid] == (baseline // class_divisor)[valid])
        )
    return result


def oracle_reconstruction(
    truth: np.ndarray,
    candidates: np.ndarray,
    cfa: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(truth, dtype=np.float64)
    bank = np.asarray(candidates, dtype=np.float64)
    phases = np.asarray(cfa)
    if bank.ndim != 3 or bank.shape[1:] != target.shape:
        raise ValueError("candidate bank shape differs")
    missing = phases != 1
    squared = (bank - target[None]) ** 2 * missing[None]
    if window > 1:
        numerator = np.stack([
            ndimage.uniform_filter(row, size=window, mode="nearest")
            for row in squared
        ])
        denominator = ndimage.uniform_filter(
            missing.astype(np.float64), size=window, mode="nearest"
        )
        local = numerator / np.maximum(denominator[None], 1e-12)
    else:
        local = squared
    labels = np.argmin(local, axis=0)
    output = np.take_along_axis(bank, labels[None], axis=0)[0]
    output = np.where(missing, output, target)
    return output, labels


def ranking_metrics(
    truth: np.ndarray,
    candidates: np.ndarray,
    criteria: np.ndarray,
    cfa: np.ndarray,
    *,
    margin: int = MARGIN,
    maximum_pixels: int = 2048,
) -> dict[str, float]:
    target = np.asarray(truth, dtype=np.float64)
    bank = np.asarray(candidates, dtype=np.float64)
    score = np.asarray(criteria, dtype=np.float64)
    phases = np.asarray(cfa)
    if bank.shape != score.shape or bank.ndim != 3:
        raise ValueError("ranking banks differ")
    interior = np.zeros_like(phases, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    locations = np.flatnonzero(interior & (phases != 1))
    if locations.size > maximum_pixels:
        locations = locations[
            np.linspace(0, locations.size - 1, maximum_pixels, dtype=np.int64)
        ]
    candidate_values = bank.reshape(bank.shape[0], -1)[:, locations].T
    criterion_values = score.reshape(score.shape[0], -1)[:, locations].T
    error = (candidate_values - target.ravel()[locations, None]) ** 2
    error_rank = stats.rankdata(error, axis=1)
    criterion_rank = stats.rankdata(criterion_values, axis=1)
    centered_error = error_rank - np.mean(error_rank, axis=1, keepdims=True)
    centered_criterion = criterion_rank - np.mean(
        criterion_rank, axis=1, keepdims=True
    )
    denominator = np.sqrt(
        np.sum(centered_error**2, axis=1)
        * np.sum(centered_criterion**2, axis=1)
    )
    spearman = np.divide(
        np.sum(centered_error * centered_criterion, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    chosen = np.argmin(criterion_values, axis=1)
    oracle = np.argmin(error, axis=1)
    top2 = np.argpartition(criterion_values, 2, axis=1)[:, :2]

    # Exact pairwise concordance on the deterministic pixel sample.  Lower
    # criterion and lower error are both treated as the positive ordering.
    concordant = 0.0
    comparisons = 0
    for left in range(bank.shape[0] - 1):
        criterion_difference = criterion_values[:, left, None] - criterion_values[:, left + 1 :]
        error_difference = error[:, left, None] - error[:, left + 1 :]
        product = criterion_difference * error_difference
        concordant += float(np.sum(product > 0.0)) + 0.5 * float(np.sum(product == 0.0))
        comparisons += product.size
    selected_error = error[np.arange(error.shape[0]), chosen]
    oracle_error = error[np.arange(error.shape[0]), oracle]
    pooled = stats.spearmanr(criterion_values.ravel(), error.ravel()).statistic
    return {
        "candidate_regret_mse": float(np.mean(selected_error - oracle_error)),
        "pairwise_auc": float(concordant / comparisons),
        "pixel_count": int(locations.size),
        "pooled_spearman": float(pooled),
        "top1_accuracy": float(np.mean(chosen == oracle)),
        "top2_inclusion": float(np.mean(np.any(top2 == oracle[:, None], axis=1))),
        "within_pixel_spearman": float(np.mean(spearman)),
    }


def calibration(
    truth: np.ndarray,
    candidates: np.ndarray,
    criteria: np.ndarray,
    cfa: np.ndarray,
    bins: int = 10,
    margin: int = MARGIN,
) -> list[dict[str, float]]:
    target = np.asarray(truth)
    bank = np.asarray(candidates)
    score = np.asarray(criteria)
    phases = np.asarray(cfa)
    interior = np.zeros_like(phases, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    locations = np.flatnonzero(interior & (phases != 1))
    criterion = score.reshape(score.shape[0], -1)[:, locations].ravel()
    error = (
        bank.reshape(bank.shape[0], -1)[:, locations]
        - target.ravel()[locations][None]
    ).ravel() ** 2
    edges = np.quantile(criterion, np.linspace(0.0, 1.0, bins + 1))
    rows = []
    for index in range(bins):
        selected = (criterion >= edges[index]) & (
            criterion <= edges[index + 1] if index + 1 == bins else criterion < edges[index + 1]
        )
        rows.append({
            "criterion_mean": float(np.mean(criterion[selected])),
            "error_rms": float(np.sqrt(np.mean(error[selected]))),
            "fraction": float(np.mean(selected)),
        })
    return rows

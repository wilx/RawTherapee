"""Candidate banks, observable features, oracles, and deterministic models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, stats

from tools.xtrans_hybrid.analysis import Normalizer
from tools.xtrans_mlri_internal.analysis import MARGIN, scalar_metrics


EPSILON = 1e-10
DIRECTION_NAMES = (
    "north", "south", "west", "east",
    "diagonal-northwest", "diagonal-southeast",
    "antidiagonal-northeast", "antidiagonal-southwest",
)
DIRECTION_ANGLES = np.asarray((
    math.pi / 2, math.pi / 2, 0, 0,
    math.pi / 4, math.pi / 4, -math.pi / 4, -math.pi / 4,
))

FEATURE_NAMES = (
    "candidate_value",
    "directional_energy",
    "log_directional_energy",
    "current_weight",
    "energy_rank",
    "abs_to_local_green",
    "abs_to_pass_fused",
    "abs_to_pass_counterpart",
    "abs_to_candidate_median",
    "candidate_gradient",
    "candidate_laplacian",
    "candidate_local_variation",
    "gradient_alignment",
    "direction_cos2",
    "direction_sin2",
    "direction_side",
    "pass_is_one",
    "pass_disagreement",
    "pass_direction_spread",
    "combined_direction_spread",
    "robust_direction_spread",
    "weight_entropy",
    "dominant_weight_ratio",
    "mosaic_gradient",
    "raw_green_variation",
    "chroma_site_variation",
    "cfa_phase_sin",
    "cfa_phase_cos",
)


def candidate_bank(case: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = case["trace"]
    candidates = np.stack([
        trace[f"pass{pass_index}-green-direction-{direction}"]
        for pass_index in (0, 1) for direction in range(8)
    ]).astype(np.float64)
    energies = np.stack([
        trace[f"pass{pass_index}-green-energy-{direction}"]
        for pass_index in (0, 1) for direction in range(8)
    ]).astype(np.float64)
    weights = np.stack([
        trace[f"pass{pass_index}-green-weight-{direction}"]
        for pass_index in (0, 1) for direction in range(8)
    ]).astype(np.float64)
    return candidates, energies, weights


def interior_missing_green(case: Mapping[str, object]) -> np.ndarray:
    cfa = np.asarray(case["cfa"])
    result = np.zeros(cfa.shape, dtype=bool)
    result[MARGIN:-MARGIN, MARGIN:-MARGIN] = cfa[MARGIN:-MARGIN, MARGIN:-MARGIN] != 1
    return result


def preserve_measured_green(case: Mapping[str, object], values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    measured = np.asarray(case["cfa"]) == 1
    result[measured] = np.asarray(case["mosaic"], dtype=np.float64)[measured]
    return result


def local_measured_green(case: Mapping[str, object], size: int = 5) -> np.ndarray:
    mosaic = np.asarray(case["mosaic"], dtype=np.float64)
    mask = (np.asarray(case["cfa"]) == 1).astype(np.float64)
    numerator = ndimage.uniform_filter(mosaic * mask, size=size, mode="reflect")
    denominator = ndimage.uniform_filter(mask, size=size, mode="reflect")
    return numerator / np.maximum(denominator, EPSILON)


def _variation(values: np.ndarray, size: int = 5) -> np.ndarray:
    mean = ndimage.uniform_filter(values, size=size, mode="reflect")
    square = ndimage.uniform_filter(values * values, size=size, mode="reflect")
    return np.sqrt(np.maximum(0.0, square - mean * mean))


def feature_maps(case: Mapping[str, object]) -> np.ndarray:
    """Return [candidate, y, x, feature] observable feature maps."""

    candidates, energies, weights = candidate_bank(case)
    trace = case["trace"]
    mosaic = np.asarray(case["mosaic"], dtype=np.float64)
    cfa = np.asarray(case["cfa"])
    height, width = mosaic.shape
    local_green = local_measured_green(case)
    fused = np.stack((trace["pass0-green"], trace["pass1-green"]))
    pass_spread = np.stack((
        np.ptp(candidates[:8], axis=0), np.ptp(candidates[8:], axis=0),
    ))
    combined_spread = np.ptp(candidates, axis=0)
    q75 = np.percentile(candidates, 75, axis=0)
    q25 = np.percentile(candidates, 25, axis=0)
    robust_spread = q75 - q25
    median = np.median(candidates, axis=0)
    pass_disagreement = np.abs(fused[1] - fused[0])

    normalized_by_pass = np.maximum(weights.reshape(2, 8, height, width), 0.0)
    normalized_by_pass /= np.maximum(
        np.sum(normalized_by_pass, axis=1, keepdims=True), EPSILON,
    )
    normalized = normalized_by_pass.reshape(16, height, width)
    entropy_by_pass = -np.sum(
        normalized.reshape(2, 8, height, width)
        * np.log(np.maximum(normalized.reshape(2, 8, height, width), EPSILON)),
        axis=1,
    ) / math.log(8)
    sorted_weights = np.sort(normalized.reshape(2, 8, height, width), axis=1)
    dominant_ratio = sorted_weights[:, -1] / np.maximum(sorted_weights[:, -2], EPSILON)

    mosaic_gx = ndimage.sobel(mosaic, axis=1, mode="reflect") / 8.0
    mosaic_gy = ndimage.sobel(mosaic, axis=0, mode="reflect") / 8.0
    mosaic_gradient = np.hypot(mosaic_gx, mosaic_gy)
    green_mask = cfa == 1
    green_variation = _variation(np.where(green_mask, mosaic, local_green))
    chroma_variation = _variation(np.where(~green_mask, mosaic, local_green))
    y, x = np.mgrid[:height, :width]
    phase = ((y + int(case["origin_y"])) % 6) * 6 + ((x + int(case["origin_x"])) % 6)
    phase_angle = phase * (2 * math.pi / 36)

    result = np.empty((16, height, width, len(FEATURE_NAMES)), dtype=np.float32)
    for index in range(16):
        pass_index = index // 8
        direction = index % 8
        candidate = candidates[index]
        counterpart = candidates[(1 - pass_index) * 8 + direction]
        candidate_gx = ndimage.sobel(candidate, axis=1, mode="reflect") / 8.0
        candidate_gy = ndimage.sobel(candidate, axis=0, mode="reflect") / 8.0
        candidate_gradient = np.hypot(candidate_gx, candidate_gy)
        laplacian = np.abs(ndimage.laplace(candidate, mode="reflect"))
        angle = DIRECTION_ANGLES[direction]
        alignment = np.abs(mosaic_gx * math.cos(angle) + mosaic_gy * math.sin(angle))
        alignment /= mosaic_gradient + 1e-6
        ranks = np.argsort(np.argsort(energies[pass_index * 8:(pass_index + 1) * 8], axis=0), axis=0)[direction] / 7.0
        maps = (
            candidate,
            energies[index],
            np.log1p(energies[index]),
            normalized[index],
            ranks,
            np.abs(candidate - local_green),
            np.abs(candidate - fused[pass_index]),
            np.abs(candidate - counterpart),
            np.abs(candidate - median),
            candidate_gradient,
            laplacian,
            _variation(candidate),
            alignment,
            np.full((height, width), math.cos(2 * angle)),
            np.full((height, width), math.sin(2 * angle)),
            np.full((height, width), -1.0 if direction % 2 == 0 else 1.0),
            np.full((height, width), float(pass_index)),
            pass_disagreement,
            pass_spread[pass_index],
            combined_spread,
            robust_spread,
            entropy_by_pass[pass_index],
            dominant_ratio[pass_index],
            mosaic_gradient,
            green_variation,
            chroma_variation,
            np.sin(phase_angle),
            np.cos(phase_angle),
        )
        result[index] = np.stack(maps, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite candidate feature for {case['case_id']}")
    return result


@dataclass(frozen=True)
class CandidateTable:
    features: np.ndarray
    candidates: np.ndarray
    truth: np.ndarray
    current0: np.ndarray
    current1: np.ndarray
    case_indices: np.ndarray
    source_ids: np.ndarray
    splits: np.ndarray

    def select(self, mask: np.ndarray) -> "CandidateTable":
        checked = np.asarray(mask, dtype=bool)
        return CandidateTable(*(
            np.asarray(getattr(self, name))[checked]
            for name in self.__dataclass_fields__
        ))


def build_candidate_table(cases: Sequence[Mapping[str, object]]) -> CandidateTable:
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in CandidateTable.__dataclass_fields__
    }
    for case_index, case in enumerate(cases):
        selected = interior_missing_green(case)
        candidates, _, _ = candidate_bank(case)
        features = feature_maps(case)
        count = int(selected.sum())
        columns["features"].append(np.moveaxis(features[:, selected, :], 0, 1))
        columns["candidates"].append(candidates[:, selected].T.astype(np.float32))
        columns["truth"].append(np.asarray(case["truth"])[1][selected].astype(np.float32))
        columns["current0"].append(np.asarray(case["trace"]["pass0-green"])[selected].astype(np.float32))
        columns["current1"].append(np.asarray(case["trace"]["pass1-green"])[selected].astype(np.float32))
        columns["case_indices"].append(np.full(count, case_index, dtype=np.int32))
        columns["source_ids"].append(np.full(count, str(case["source_id"]), dtype=object))
        columns["splits"].append(np.full(count, str(case["split"]), dtype=object))
    return CandidateTable(*(np.concatenate(columns[name]) for name in CandidateTable.__dataclass_fields__))


@dataclass(frozen=True)
class RidgeScore:
    normalizer: Normalizer
    coefficients: np.ndarray
    intercept: float
    regularization: float

    def score(self, features: np.ndarray) -> np.ndarray:
        shape = features.shape[:-1]
        flat = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
        result = self.normalizer.apply(flat) @ self.coefficients + self.intercept
        return result.reshape(shape)


def fit_ridge_score(features: np.ndarray, target: np.ndarray, weights: np.ndarray,
                    regularization: float = 1e-2) -> RidgeScore:
    checked = np.asarray(features, dtype=np.float64)
    normalizer = Normalizer.fit(checked)
    normalized = normalizer.apply(checked)
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    sqrt_weight = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), EPSILON))
    weighted_design = design * sqrt_weight[:, None]
    weighted_target = np.asarray(target, dtype=np.float64) * sqrt_weight
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0
    solution = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return RidgeScore(normalizer, solution[1:], float(solution[0]), regularization)


def source_balanced_weights(source_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(source_ids)
    result = np.empty(values.size, dtype=np.float64)
    unique, counts = np.unique(values, return_counts=True)
    for source, count in zip(unique, counts):
        result[values == source] = values.size / (len(unique) * count)
    return result


def deterministic_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, maximum).astype(np.int64)


def patch_labels(candidates: np.ndarray, truth: np.ndarray, window: int) -> np.ndarray:
    error = (np.asarray(candidates, dtype=np.float64) - np.asarray(truth)[None]) ** 2
    if window > 1:
        error = ndimage.uniform_filter(error, size=(1, window, window), mode="reflect")
    return np.argmin(error, axis=0).astype(np.uint8)


def select_candidates(case: Mapping[str, object], candidates: np.ndarray,
                      labels: np.ndarray) -> np.ndarray:
    selected = np.take_along_axis(candidates, labels[None], axis=0)[0]
    return preserve_measured_green(case, selected)


def label_coherence(labels: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    selected = np.asarray(selected, dtype=bool)
    horizontal = selected[:, 1:] & selected[:, :-1]
    vertical = selected[1:, :] & selected[:-1, :]
    boundaries = int(np.sum((labels[:, 1:] != labels[:, :-1]) & horizontal))
    boundaries += int(np.sum((labels[1:, :] != labels[:-1, :]) & vertical))
    adjacency = int(horizontal.sum() + vertical.sum())
    sizes: list[int] = []
    for identity in np.unique(labels[selected]):
        components, count = ndimage.label(selected & (labels == identity))
        if count:
            sizes.extend(np.bincount(components.ravel())[1:].tolist())
    return {
        "boundary_density": boundaries / max(1, adjacency),
        "component_count": len(sizes),
        "mean_component_pixels": float(np.mean(sizes)) if sizes else 0.0,
        "median_component_pixels": float(np.median(sizes)) if sizes else 0.0,
        "p90_component_pixels": float(np.percentile(sizes, 90)) if sizes else 0.0,
    }


def scalar_convex_hull(case: Mapping[str, object], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(case["truth"])[1]
    low = np.min(candidates, axis=0)
    high = np.max(candidates, axis=0)
    result = np.clip(truth, low, high)
    inside = (truth >= low) & (truth <= high)
    return preserve_measured_green(case, result), inside


def patch_blend_alpha(first: np.ndarray, second: np.ndarray, truth: np.ndarray,
                      window: int) -> np.ndarray:
    delta = np.asarray(second) - np.asarray(first)
    numerator = (np.asarray(truth) - np.asarray(first)) * delta
    denominator = delta * delta
    if window > 1:
        numerator = ndimage.uniform_filter(numerator, size=window, mode="reflect")
        denominator = ndimage.uniform_filter(denominator, size=window, mode="reflect")
    return np.clip(numerator / np.maximum(denominator, EPSILON), 0, 1)


def temperature_fusion(case: Mapping[str, object], pass_index: int, exponent: float) -> np.ndarray:
    candidates, _, weights = candidate_bank(case)
    subset = slice(pass_index * 8, (pass_index + 1) * 8)
    adjusted = np.maximum(weights[subset], EPSILON) ** exponent
    adjusted /= np.maximum(np.sum(adjusted, axis=0, keepdims=True), EPSILON)
    return preserve_measured_green(case, np.sum(adjusted * candidates[subset], axis=0))


def score_calibration(table: CandidateTable, scores: np.ndarray) -> dict[str, float]:
    actual = (table.candidates - table.truth[:, None]) ** 2
    selected = np.argmin(scores, axis=1)
    oracle = np.argmin(actual, axis=1)
    order = np.argsort(scores, axis=1)
    rows = np.arange(table.truth.size)
    selected_error = actual[rows, selected]
    oracle_error = actual[rows, oracle]
    sample = deterministic_indices(actual.size, 200000)
    correlation = stats.spearmanr(scores.ravel()[sample], actual.ravel()[sample]).statistic
    return {
        "spearman_cost_vs_squared_error": float(correlation),
        "top1_accuracy": float(np.mean(selected == oracle)),
        "top2_oracle_inclusion": float(np.mean(np.any(order[:, :2] == oracle[:, None], axis=1))),
        "mean_squared_error": float(np.mean(selected_error)),
        "mean_oracle_squared_error": float(np.mean(oracle_error)),
        "mean_regret": float(np.mean(selected_error - oracle_error)),
        "p95_regret": float(np.percentile(selected_error - oracle_error, 95)),
        "psnr_db": math.inf if np.mean(selected_error) == 0 else float(-10 * math.log10(np.mean(selected_error))),
    }


def green_metrics(cases: Sequence[Mapping[str, object]], guides: Sequence[np.ndarray]) -> dict[str, float]:
    errors = []
    for case, guide in zip(cases, guides):
        selected = interior_missing_green(case)
        errors.append((np.asarray(guide) - np.asarray(case["truth"])[1])[selected])
    return scalar_metrics(np.concatenate(errors))

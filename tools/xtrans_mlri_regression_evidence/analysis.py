"""Candidate-specific guided-regression features and deterministic metrics."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from tools.xtrans_mlri_green_fusion.analysis import (
    candidate_bank,
    interior_missing_green,
    preserve_measured_green,
)
from tools.xtrans_mlri_internal.analysis import scalar_metrics


EPSILON = 1e-10
REGRESSION_STATISTICS = (
    "fit-mse", "min-fit-mse", "gain", "gain-variance",
    "gain-min", "gain-max", "offset", "gain-numerator",
    "gain-denominator", "guide-energy", "laplacian-count",
    "sample-count", "prediction-variance", "prediction-range",
    "prediction-mad", "prediction-iqr", "model-weight-entropy",
    "effective-model-count", "max-model-weight", "total-model-weight",
    "loo-prediction-variance", "max-model-influence",
)

FEATURE_NAMES = (
    "log-directional-energy", "current-weight", "log-fit-mse",
    "log-min-fit-mse", "gain", "abs-gain", "abs-gain-minus-one",
    "offset", "log-gain-denominator", "inverse-gain-denominator",
    "log-guide-energy", "log-prediction-variance",
    "log-prediction-range", "log-prediction-mad", "log-prediction-iqr",
    "gain-variance", "gain-range", "model-weight-entropy",
    "inverse-effective-model-count", "max-model-weight",
    "log-total-model-weight", "log-loo-prediction-variance",
    "log-max-model-influence", "pass-counterpart-disagreement",
)


def regression_bank(case: Mapping[str, object]) -> np.ndarray:
    trace = case["trace"]
    return np.stack([
        np.stack([
            trace[f"pass{pass_index}-green-regression-{name}-{direction}"]
            for name in REGRESSION_STATISTICS
        ], axis=-1)
        for pass_index in (0, 1) for direction in range(8)
    ]).astype(np.float64)


def feature_maps(case: Mapping[str, object]) -> np.ndarray:
    candidates, energies, weights = candidate_bank(case)
    regression = regression_bank(case)
    index = {name: REGRESSION_STATISTICS.index(name) for name in REGRESSION_STATISTICS}
    gain = regression[..., index["gain"]]
    denominator = np.abs(regression[..., index["gain-denominator"]])
    effective = regression[..., index["effective-model-count"]]
    counterpart = np.concatenate((candidates[8:], candidates[:8]))
    values = (
        np.log1p(np.maximum(energies, 0)),
        weights,
        np.log1p(np.maximum(regression[..., index["fit-mse"]], 0)),
        np.log1p(np.maximum(regression[..., index["min-fit-mse"]], 0)),
        gain,
        np.abs(gain),
        np.abs(gain - 1.0),
        regression[..., index["offset"]] / 255.0,
        np.log1p(denominator),
        1.0 / (denominator + 0.01),
        np.log1p(np.maximum(regression[..., index["guide-energy"]], 0)),
        np.log1p(np.maximum(regression[..., index["prediction-variance"]], 0)),
        np.log1p(np.maximum(regression[..., index["prediction-range"]], 0)),
        np.log1p(np.maximum(regression[..., index["prediction-mad"]], 0)),
        np.log1p(np.maximum(regression[..., index["prediction-iqr"]], 0)),
        np.log1p(np.maximum(regression[..., index["gain-variance"]], 0)),
        regression[..., index["gain-max"]] - regression[..., index["gain-min"]],
        regression[..., index["model-weight-entropy"]],
        1.0 / np.maximum(effective, 1.0),
        regression[..., index["max-model-weight"]],
        np.log1p(np.maximum(regression[..., index["total-model-weight"]], 0)),
        np.log1p(np.maximum(regression[..., index["loo-prediction-variance"]], 0)),
        np.log1p(np.maximum(regression[..., index["max-model-influence"]], 0)),
        np.abs(candidates - counterpart),
    )
    result = np.stack(values, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite regression feature for {case['case_id']}")
    return result


@dataclass(frozen=True)
class EvidenceTable:
    features: np.ndarray
    candidates: np.ndarray
    truth: np.ndarray
    current0: np.ndarray
    current1: np.ndarray
    case_indices: np.ndarray
    source_ids: np.ndarray
    splits: np.ndarray
    families: np.ndarray

    def select(self, mask: np.ndarray) -> "EvidenceTable":
        checked = np.asarray(mask, dtype=bool)
        return EvidenceTable(*(
            np.asarray(getattr(self, name))[checked]
            for name in self.__dataclass_fields__
        ))


def build_table(cases: Sequence[Mapping[str, object]]) -> EvidenceTable:
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in EvidenceTable.__dataclass_fields__
    }
    for case_index, case in enumerate(cases):
        selected = interior_missing_green(case)
        candidates, _, _ = candidate_bank(case)
        count = int(np.sum(selected))
        columns["features"].append(
            feature_maps(case)[:, selected].transpose(1, 0, 2)
        )
        columns["candidates"].append(candidates[:, selected].T)
        columns["truth"].append(np.asarray(case["truth"])[1][selected])
        columns["current0"].append(np.asarray(case["trace"]["pass0-green"])[selected])
        columns["current1"].append(np.asarray(case["trace"]["pass1-green"])[selected])
        columns["case_indices"].append(np.full(count, case_index, dtype=np.int32))
        columns["source_ids"].append(np.full(count, str(case["source_id"]), dtype=object))
        columns["splits"].append(np.full(count, str(case["split"]), dtype=object))
        columns["families"].append(np.full(count, str(case["family"]), dtype=object))
    return EvidenceTable(*(np.concatenate(columns[name]) for name in columns))


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - np.mean(array, axis=1, keepdims=True)


def within_pixel_rank_metrics(scores: np.ndarray, squared_error: np.ndarray) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    error = np.asarray(squared_error, dtype=np.float64)
    score_rank = stats.rankdata(score, axis=1)
    error_rank = stats.rankdata(error, axis=1)
    score_rank -= np.mean(score_rank, axis=1, keepdims=True)
    error_rank -= np.mean(error_rank, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(score_rank * score_rank, axis=1)
        * np.sum(error_rank * error_rank, axis=1)
    )
    correlations = np.divide(
        np.sum(score_rank * error_rank, axis=1), denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    selected = np.argmin(score, axis=1)
    actual = np.argmin(error, axis=1)
    top2 = np.argpartition(score, 1, axis=1)[:, :2]
    chosen_error = np.take_along_axis(error, selected[:, None], axis=1)[:, 0]
    best_error = np.min(error, axis=1)
    return {
        "mean_within_pixel_spearman": float(np.mean(correlations)),
        "median_within_pixel_spearman": float(np.median(correlations)),
        "top1_accuracy": float(np.mean(selected == actual)),
        "top2_inclusion": float(np.mean(np.any(top2 == actual[:, None], axis=1))),
        "regret": scalar_metrics(np.sqrt(np.maximum(chosen_error - best_error, 0))),
    }


def pooled_correlations(scores: np.ndarray, squared_error: np.ndarray) -> dict[str, float]:
    score = np.asarray(scores, dtype=np.float64)
    error = np.asarray(squared_error, dtype=np.float64)
    return {
        "absolute_spearman": float(stats.spearmanr(score.ravel(), error.ravel()).statistic),
        "centered_spearman": float(stats.spearmanr(centered(score).ravel(), centered(error).ravel()).statistic),
    }


def pairwise_metrics(scores: np.ndarray, squared_error: np.ndarray) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    error = np.asarray(squared_error, dtype=np.float64)
    groups: dict[str, list[tuple[int, int]]] = {
        "all": list(itertools.combinations(range(16), 2)),
        "corresponding-pass0-pass1": [(i, i + 8) for i in range(8)],
        "within-pass": [
            pair for start in (0, 8)
            for pair in itertools.combinations(range(start, start + 8), 2)
        ],
    }
    result: dict[str, object] = {}
    for name, pairs in groups.items():
        score_difference = np.concatenate([score[:, i] - score[:, j] for i, j in pairs])
        error_difference = np.concatenate([error[:, i] - error[:, j] for i, j in pairs])
        magnitude = np.abs(error_difference)
        lower, upper = np.percentile(magnitude, (25, 75))
        rows = {}
        for subset_name, mask in (
            ("all", np.ones(magnitude.shape, dtype=bool)),
            ("close", magnitude <= lower),
            ("high-margin", magnitude >= upper),
        ):
            labels = error_difference[mask] > 0
            selected_scores = score_difference[mask]
            positive = int(np.sum(labels))
            negative = int(labels.size - positive)
            if positive == 0 or negative == 0:
                auc = 0.5
            else:
                ranks = stats.rankdata(selected_scores, method="average")
                rank_sum = float(np.sum(ranks[labels]))
                auc = (
                    rank_sum - positive * (positive + 1) / 2
                ) / (positive * negative)
            rows[subset_name] = {
                "auc": float(auc),
                "accuracy": float(np.mean((score_difference[mask] > 0) == labels)),
                "count": int(np.sum(mask)),
            }
        result[name] = rows
    return result


def select_guide(case: Mapping[str, object], scores: np.ndarray) -> np.ndarray:
    candidates, _, _ = candidate_bank(case)
    labels = np.argmin(scores, axis=0)
    guide = np.take_along_axis(candidates, labels[None], axis=0)[0]
    return preserve_measured_green(case, guide).astype(np.float32)


def fuse_survivors(
    case: Mapping[str, object], scores: np.ndarray, retain: int,
    mode: str = "current-weight",
) -> np.ndarray:
    candidates, _, weights = candidate_bank(case)
    identities = np.argpartition(scores, retain - 1, axis=0)[:retain]
    selected_candidates = np.take_along_axis(candidates, identities, axis=0)
    if mode == "median":
        guide = np.median(selected_candidates, axis=0)
    else:
        if mode == "uniform":
            selected_weights = np.ones_like(selected_candidates)
        elif mode == "current-weight":
            selected_weights = np.take_along_axis(weights, identities, axis=0)
        else:
            raise ValueError(mode)
        selected_weights /= np.maximum(np.sum(selected_weights, axis=0, keepdims=True), EPSILON)
        guide = np.sum(selected_weights * selected_candidates, axis=0)
    return preserve_measured_green(case, guide).astype(np.float32)


def modulated_guide(case: Mapping[str, object], scores: np.ndarray, strength: float) -> np.ndarray:
    candidates, _, weights = candidate_bank(case)
    normalized = centered(scores.reshape(16, -1)).reshape(scores.shape)
    scale = np.std(normalized, axis=0, keepdims=True)
    z_score = normalized / np.maximum(scale, 1e-6)
    reliability = np.exp(-np.clip(strength * z_score, -20, 20))
    combined = np.maximum(weights, 0) * reliability
    combined /= np.maximum(np.sum(combined, axis=0, keepdims=True), EPSILON)
    guide = np.sum(combined * candidates, axis=0)
    return preserve_measured_green(case, guide).astype(np.float32)

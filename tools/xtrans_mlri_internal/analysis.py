"""Stage, residual-oracle, and internal-predictability analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage


EPSILON = 1e-10
MARGIN = 16
FEATURE_NAMES = (
    "correction_abs",
    "correction_local_rms",
    "correction_local_mad",
    "correction_to_rms",
    "correction_to_mad",
    "correction_to_cfa_variation",
    "raw_residual_local_rms",
    "raw_residual_sign_consistency",
    "pass1_to_tentative_abs",
    "pass0_to_pass1_abs",
    "pass0_to_final_abs",
    "green_pass_change",
    "pass0_green_direction_spread",
    "pass1_green_direction_spread",
    "pass1_green_direction_stddev",
    "green_local_variation",
    "mosaic_local_variation",
    "channel_is_blue",
)


def local_mean(values: np.ndarray, size: int = 5) -> np.ndarray:
    return ndimage.uniform_filter(np.asarray(values, dtype=np.float64), size=size, mode="reflect")


def local_rms(values: np.ndarray, size: int = 5) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.maximum(0.0, local_mean(checked * checked, size)))


def local_mad(values: np.ndarray, size: int = 5) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    median = ndimage.median_filter(checked, size=size, mode="reflect")
    return ndimage.median_filter(np.abs(checked - median), size=size, mode="reflect")


def local_variation(values: np.ndarray, size: int = 5) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.maximum(0.0, local_mean(checked * checked, size) - local_mean(checked, size) ** 2))


def alpha_oracle(base: np.ndarray, correction: np.ndarray, truth: np.ndarray,
                 low: float = 0.0, high: float = 1.0) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    result = np.divide(
        truth - base, correction,
        out=np.zeros_like(base), where=np.abs(correction) > EPSILON,
    )
    return np.clip(result, low, high)


def scalar_metrics(error: np.ndarray) -> dict[str, float]:
    values = np.asarray(error, dtype=np.float64).ravel()
    absolute = np.abs(values)
    mse = float(np.mean(values * values)) if values.size else 0.0
    return {
        "count": int(values.size),
        "maximum_abs": float(np.max(absolute)) if values.size else 0.0,
        "median_abs": float(np.percentile(absolute, 50)) if values.size else 0.0,
        "mse": mse,
        "p90_abs": float(np.percentile(absolute, 90)) if values.size else 0.0,
        "p95_abs": float(np.percentile(absolute, 95)) if values.size else 0.0,
        "p99_abs": float(np.percentile(absolute, 99)) if values.size else 0.0,
        "psnr_db": math.inf if mse == 0 else float(10 * math.log10(1.0 / mse)),
        "rms": math.sqrt(mse),
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def operating_threshold(scores: np.ndarray, labels: np.ndarray,
                        minimum_recall: float = 0.9) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return math.inf
    candidates = np.unique(scores)
    best = float(np.min(candidates))
    best_rejection = 1.0
    for threshold in candidates:
        predicted = scores >= threshold
        recall = float(np.mean(predicted[labels]))
        rejection = float(np.mean(predicted))
        if recall >= minimum_recall and rejection <= best_rejection:
            best = float(threshold)
            best_rejection = rejection
    return best


def classification_metrics(scores: np.ndarray, labels: np.ndarray,
                           threshold: float) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    predicted = scores >= threshold
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((~labels).sum()))
    bins = []
    edges = np.linspace(0, 1, 6)
    clipped = np.clip(scores, 0, 1)
    for index in range(5):
        mask = (clipped >= edges[index]) & (
            clipped <= edges[index + 1] if index == 4 else clipped < edges[index + 1]
        )
        bins.append({
            "count": int(mask.sum()),
            "observed_rate": None if not mask.any() else float(np.mean(labels[mask])),
            "predicted_mean": None if not mask.any() else float(np.mean(clipped[mask])),
        })
    return {
        "calibration_bins": bins,
        "false_positive_rate": float(np.sum(predicted & ~labels) / negatives),
        "rejection_fraction": float(np.mean(predicted)),
        "roc_auc": float(roc_auc(scores, labels)),
        "severe_recall": float(np.sum(predicted & labels) / positives),
        "severe_count": int(labels.sum()),
        "threshold": float(threshold),
    }


def internal_feature_maps(case: Mapping[str, object], channel: int) -> dict[str, np.ndarray]:
    trace = case["trace"]
    suffix = "red" if channel == 0 else "blue"
    correction = np.asarray(trace[f"final-correction-{suffix}"], dtype=np.float64)
    raw_residual = np.asarray(trace[f"final-raw-residual-{suffix}"], dtype=np.float64)
    tentative = np.asarray(trace[f"final-tentative-{suffix}"], dtype=np.float64)
    pass0 = np.asarray(trace[f"pass0-provisional-{suffix}"], dtype=np.float64)
    pass1 = np.asarray(trace[f"pass1-provisional-{suffix}"], dtype=np.float64)
    final = np.asarray(trace[f"final-{suffix}"], dtype=np.float64)
    green0 = np.asarray(trace["pass0-green"], dtype=np.float64)
    green1 = np.asarray(trace["pass1-green"], dtype=np.float64)
    green_directions0 = np.stack([
        trace[f"pass0-green-direction-{index}"] for index in range(8)
    ])
    green_directions1 = np.stack([
        trace[f"pass1-green-direction-{index}"] for index in range(8)
    ])
    mosaic = np.asarray(case["mosaic"], dtype=np.float64)
    correction_rms = local_rms(correction)
    correction_mad = local_mad(correction)
    cfa_variation = local_variation(mosaic)
    raw_rms = local_rms(raw_residual)
    sign_consistency = np.abs(local_mean(raw_residual)) / (local_mean(np.abs(raw_residual)) + EPSILON)
    return dict(zip(FEATURE_NAMES, (
        np.abs(correction),
        correction_rms,
        correction_mad,
        np.abs(correction) / (correction_rms + EPSILON),
        np.abs(correction) / (correction_mad + 1e-5),
        np.abs(correction) / (cfa_variation + 1e-5),
        raw_rms,
        sign_consistency,
        np.abs(pass1 - tentative),
        np.abs(pass0 - pass1),
        np.abs(pass0 - final),
        np.abs(green1 - green0),
        np.ptp(green_directions0, axis=0),
        np.ptp(green_directions1, axis=0),
        np.std(green_directions1, axis=0),
        local_variation(green1),
        cfa_variation,
        np.full(correction.shape, float(channel == 2)),
    )))


@dataclass(frozen=True)
class SampleTable:
    features: np.ndarray
    base: np.ndarray
    correction: np.ndarray
    final: np.ndarray
    pass0_final: np.ndarray
    truth: np.ndarray
    pass0: np.ndarray
    pass1: np.ndarray
    source_ids: np.ndarray
    case_ids: np.ndarray
    splits: np.ndarray
    kinds: np.ndarray

    @property
    def harm(self) -> np.ndarray:
        return (self.final - self.truth) ** 2 - (self.base - self.truth) ** 2

    def select(self, mask: np.ndarray) -> "SampleTable":
        checked = np.asarray(mask, dtype=bool)
        return SampleTable(*(
            np.asarray(getattr(self, field))[checked]
            for field in self.__dataclass_fields__
        ))


def build_sample_table(cases: Sequence[Mapping[str, object]]) -> SampleTable:
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in SampleTable.__dataclass_fields__
    }
    for case in cases:
        cfa = np.asarray(case["cfa"])
        truth = np.asarray(case["truth"], dtype=np.float64)
        trace = case["trace"]
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        for channel, suffix in ((0, "red"), (2, "blue")):
            selected = interior & (cfa != channel)
            maps = internal_feature_maps(case, channel)
            count = int(selected.sum())
            columns["features"].append(np.column_stack([maps[name][selected] for name in FEATURE_NAMES]))
            columns["base"].append(np.asarray(trace[f"final-tentative-{suffix}"])[selected])
            columns["correction"].append(np.asarray(trace[f"final-correction-{suffix}"])[selected])
            columns["final"].append(np.asarray(trace[f"final-{suffix}"])[selected])
            columns["pass0_final"].append(np.asarray(case["pass0_guide_final"])[channel][selected])
            columns["truth"].append(truth[channel][selected])
            columns["pass0"].append(np.asarray(trace[f"pass0-provisional-{suffix}"])[selected])
            columns["pass1"].append(np.asarray(trace[f"pass1-provisional-{suffix}"])[selected])
            columns["source_ids"].append(np.full(count, str(case["source_id"]), dtype=object))
            columns["case_ids"].append(np.full(count, str(case["case_id"]), dtype=object))
            columns["splits"].append(np.full(count, str(case["split"]), dtype=object))
            columns["kinds"].append(np.full(count, str(case["kind"]), dtype=object))
    return SampleTable(*(np.concatenate(columns[name], axis=0) for name in SampleTable.__dataclass_fields__))


def reconstruction_metrics(table: SampleTable, values: np.ndarray) -> dict[str, float]:
    return scalar_metrics(np.asarray(values) - table.truth)

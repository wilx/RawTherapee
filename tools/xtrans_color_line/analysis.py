"""Metrics for the X-Trans local color-line feasibility experiment."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, stats

from .model import ColorLineFit


MARGIN = 16
EPSILON = 1e-15


def _psnr(mse: float) -> float:
    return math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def reconstruction_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    cfa: np.ndarray,
    *,
    margin: int = MARGIN,
) -> dict[str, object]:
    target = np.asarray(truth, dtype=np.float64)
    output = np.asarray(reconstruction, dtype=np.float64)
    phases = np.asarray(cfa)
    if target.shape != output.shape or target.shape[1:] != phases.shape:
        raise ValueError("metric shapes differ")
    interior = np.zeros(phases.shape, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    error = output - target
    selected = error[:, interior].ravel()
    missing_mask = np.broadcast_to(
        np.arange(3)[:, None, None] != phases[None], target.shape
    ) & interior[None]
    missing = error[missing_mask]
    absolute = np.abs(selected)
    mse = float(np.mean(selected * selected))
    missing_mse = float(np.mean(missing * missing))
    transform = np.asarray(
        (
            (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
            (1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)),
            (1 / math.sqrt(6), -2 / math.sqrt(6), 1 / math.sqrt(6)),
        )
    )
    lcc = np.einsum("kc,cyx->kyx", transform, error)
    return {
        "l_c1_c2_rms": [
            float(np.sqrt(np.mean(lcc[channel, interior] ** 2)))
            for channel in range(3)
        ],
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "missing_channel_psnr_db": _psnr(missing_mse),
        "missing_channel_rms": math.sqrt(missing_mse),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": _psnr(mse),
        "rms": math.sqrt(mse),
        "sample_count": int(selected.size),
    }


def pooled_reconstruction_metrics(
    cases: Sequence[Mapping[str, object]],
    method: str,
    *,
    margin: int = MARGIN,
) -> dict[str, object]:
    all_error = []
    all_missing = []
    all_lcc = []
    transform = np.asarray(
        (
            (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
            (1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)),
            (1 / math.sqrt(6), -2 / math.sqrt(6), 1 / math.sqrt(6)),
        )
    )
    for case in cases:
        truth = np.asarray(case["truth"], dtype=np.float64)
        output = np.asarray(case["outputs"][method], dtype=np.float64)
        cfa = np.asarray(case["cfa"])
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[margin:-margin, margin:-margin] = True
        error = output - truth
        all_error.append(error[:, interior].ravel())
        missing_mask = np.broadcast_to(
            np.arange(3)[:, None, None] != cfa[None], truth.shape
        ) & interior[None]
        all_missing.append(error[missing_mask])
        all_lcc.append(np.einsum("kc,cyx->kyx", transform, error)[:, interior])
    error = np.concatenate(all_error)
    missing = np.concatenate(all_missing)
    lcc = np.concatenate(all_lcc, axis=1)
    absolute = np.abs(error)
    mse = float(np.mean(error * error))
    missing_mse = float(np.mean(missing * missing))
    return {
        "l_c1_c2_rms": [
            float(np.sqrt(np.mean(lcc[channel] ** 2))) for channel in range(3)
        ],
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "missing_channel_psnr_db": _psnr(missing_mse),
        "missing_channel_rms": math.sqrt(missing_mse),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": _psnr(mse),
        "rms": math.sqrt(mse),
        "sample_count": int(error.size),
    }


def line_model_samples(
    fit: ColorLineFit,
    cfa: np.ndarray,
    *,
    margin: int = MARGIN,
    maximum_samples: int = 4096,
) -> dict[str, np.ndarray]:
    eigenvalues = np.asarray(fit.eigenvalues)
    phases = np.asarray(cfa)
    interior = np.zeros(phases.shape, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    locations = np.flatnonzero(interior)
    if locations.size > maximum_samples:
        locations = locations[
            np.linspace(0, locations.size - 1, maximum_samples, dtype=np.int64)
        ]
    values = eigenvalues.reshape(-1, 3)[locations]
    total = np.sum(values, axis=1)
    squared = np.sum(values * values, axis=1)
    separation = (
        fit.maximum_projection - fit.minimum_projection
    ).ravel()[locations]
    return {
        "effective_rank": np.divide(
            total * total,
            squared,
            out=np.ones_like(total),
            where=squared > EPSILON,
        ),
        "endpoint_separation": separation,
        "explained_variance": np.divide(
            values[:, 2], total, out=np.ones_like(total), where=total > EPSILON
        ),
        "rank1_residual_ratio": np.divide(
            values[:, 0] + values[:, 1],
            total,
            out=np.zeros_like(total),
            where=total > EPSILON,
        ),
        "rank2_residual_ratio": np.divide(
            values[:, 0], total, out=np.zeros_like(total), where=total > EPSILON
        ),
    }


def summarize_samples(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, object]:
    result = {}
    for name in (
        "effective_rank",
        "endpoint_separation",
        "explained_variance",
        "rank1_residual_ratio",
        "rank2_residual_ratio",
    ):
        values = np.concatenate([np.asarray(row[name]) for row in rows])
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
        }
    result["sample_count"] = int(sum(len(row["effective_rank"]) for row in rows))
    return result


def oracle_gate(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    cfa: np.ndarray,
    support: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(truth, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    alternative = np.asarray(candidate, dtype=np.float64)
    if target.shape != base.shape or target.shape != alternative.shape:
        raise ValueError("oracle gate shapes differ")
    base_error = np.mean((base - target) ** 2, axis=0)
    candidate_error = np.mean((alternative - target) ** 2, axis=0)
    if support > 1:
        base_error = ndimage.uniform_filter(base_error, support, mode="reflect")
        candidate_error = ndimage.uniform_filter(
            candidate_error, support, mode="reflect"
        )
    choose_line = candidate_error < base_error
    output = np.where(choose_line[None], alternative, base)
    phases = np.asarray(cfa)
    scalar = np.take_along_axis(
        np.moveaxis(target, 0, -1), phases[..., None], axis=2
    )[..., 0]
    for channel in range(3):
        mask = phases == channel
        output[channel, mask] = scalar[mask]
    return output, choose_line


def confidence_metrics(confidence: np.ndarray, error: np.ndarray) -> dict[str, float]:
    score = np.asarray(confidence, dtype=np.float64).ravel()
    loss = np.asarray(error, dtype=np.float64).ravel()
    return {
        "spearman": float(stats.spearmanr(score, loss).statistic),
    }

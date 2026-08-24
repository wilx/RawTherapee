"""Shared deterministic metrics for Student-t GMR reduction candidates."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

from tools.xtrans_gmr.experiment import _metric, _rows, _truth


def quality_retention(
    gmax_mse: float, reference_mse: float, candidate_mse: float
) -> float | None:
    denominator = gmax_mse - reference_mse
    if denominator <= 0.0:
        return None
    return (gmax_mse - candidate_mse) / denominator


def summarize_batch(samples, batch, patch_size: int = 7) -> dict[str, object]:
    truth = _truth(samples, patch_size)
    squared = np.mean((batch.mmse_rgb - truth) ** 2, axis=1)
    correlation = float(spearmanr(batch.predictive_risk, squared).statistic)
    measured_exact = True
    area = patch_size * patch_size
    center = area // 2
    for index, sample in enumerate(samples):
        measured = sample.indices[sample.indices % area == center]
        if measured.size != 1:
            raise ValueError("invalid measured-center contract")
        channel = int(measured[0] // area)
        measured_exact = measured_exact and bool(
            batch.mmse_rgb[index, channel]
            == sample.vector[channel * area + center]
        )
    return {
        "metric": _metric(batch.mmse_rgb - truth),
        "rows": _rows(samples, {"candidate": batch.mmse_rgb}, truth),
        "posterior": {
            "effective_component_count_mean": float(
                np.mean(batch.effective_component_count)
            ),
            "entropy_mean": float(np.mean(batch.entropy)),
            "maximum_responsibility_mean": float(
                np.mean(batch.maximum_responsibility)
            ),
            "risk_error_spearman": (
                correlation if math.isfinite(correlation) else None
            ),
        },
        "native_center_exact": measured_exact,
    }


def summarize_shortlist(samples, batch, patch_size: int = 7) -> dict[str, object]:
    truth = _truth(samples, patch_size)
    responsibilities = batch.responsibilities
    positive = responsibilities > 0.0
    entropy = -np.sum(
        np.where(positive, responsibilities * np.log(np.maximum(
            responsibilities, np.finfo(np.float64).tiny
        )), 0.0),
        axis=1,
    )
    area = patch_size * patch_size
    center = area // 2
    measured_exact = True
    for index, sample in enumerate(samples):
        measured = sample.indices[sample.indices % area == center]
        channel = int(measured[0] // area)
        measured_exact = measured_exact and bool(
            batch.mmse_rgb[index, channel]
            == sample.vector[channel * area + center]
        )
    return {
        "metric": _metric(batch.mmse_rgb - truth),
        "rows": _rows(samples, {"candidate": batch.mmse_rgb}, truth),
        "posterior": {
            "effective_component_count_mean": float(np.mean(np.exp(entropy))),
            "entropy_mean": float(np.mean(entropy)),
            "maximum_responsibility_mean": float(
                np.mean(np.max(responsibilities, axis=1))
            ),
            "risk_error_spearman": None,
        },
        "shortlist": {
            "exact_top_inclusion": float(np.mean(batch.exact_top_included)),
            "retained_exact_mass_mean": float(np.mean(batch.retained_exact_mass)),
            "retained_exact_mass_p05": float(
                np.quantile(batch.retained_exact_mass, 0.05)
            ),
            "retained_exact_mass_minimum": float(
                np.min(batch.retained_exact_mass)
            ),
        },
        "native_center_exact": measured_exact,
    }


def pooled_metric(rows: dict[str, object], method: str) -> dict[str, float | int]:
    count = sum(int(row["methods"][method]["count"]) for row in rows.values())
    sse = sum(float(row["methods"][method]["sse"]) for row in rows.values())
    mse = sse / count
    return {
        "count": count,
        "mse": mse,
        "psnr_db": -10.0 * math.log10(max(mse, 1e-30)),
        "sse": sse,
    }


def candidate_row_metric(summary: dict[str, object], name: str) -> dict[str, object]:
    return summary["rows"][name]["candidate"]


__all__ = (
    "candidate_row_metric",
    "pooled_metric",
    "quality_retention",
    "summarize_batch",
    "summarize_shortlist",
)

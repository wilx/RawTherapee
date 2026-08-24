#!/usr/bin/env python3
"""Run validation-only Student-t GMR reduction screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmm.dataset import (
    VALIDATION_GRID_SIDE,
    corpus_manifest,
    evaluation_samples,
)
from tools.xtrans_gmr.experiment import _logical_digest, _metric, _truth, load_model
from tools.xtrans_gmr.model import model_summary, prepare_gmr_cache
from tools.xtrans_lmmse.dataset import load_rgb
from tools.xtrans_lmmse.model import predict_region
from tools.xtrans_lmmse_mixture.training import survey_and_train_globals
from tools.xtrans_tgmr.experiment import (
    FROZEN_TAU,
    FROZEN_TEMPERATURE,
    evaluate_student,
)
from tools.xtrans_tgmr.model import conditional_student_t_predict

from .model import (
    approximate_factor_model,
    coarse_component_order,
    factor_mac_estimate,
    factor_parameter_count,
    factor_student_t_predict,
    prepare_factor_cache,
    prune_components,
    shortlist_student_t_predict,
    truncate_exact_posterior,
)


DEGREES_OF_FREEDOM = 3.0
FACTOR_RANKS = (2, 4, 8, 12, 16, 24, 32)
PRUNED_COUNTS = (8, 16, 24, 32, 48)
TRUNCATED_COUNTS = (1, 2, 4, 8, 16, 32)
SHORTLIST_COUNTS = (4, 8, 16, 24)


def _quality_retention(gmax_mse: float, full_mse: float, candidate_mse: float) -> float:
    denominator = gmax_mse - full_mse
    if denominator <= 0.0:
        raise ValueError("frozen t-GMR must improve validation GMAX")
    return (gmax_mse - candidate_mse) / denominator


def _gmax_validation(
    splits: dict[str, list[dict[str, object]]],
    samples,
    cache_path: Path,
) -> dict[str, object]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    globals_by_size, _, _ = survey_and_train_globals(splits["train"])
    bank = globals_by_size[200]
    by_source: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_source.setdefault(sample.source_id, []).append(index)
    errors = []
    for source in splits["val"]:
        source_id = str(source["filename"])
        if source_id not in by_source:
            continue
        truth = load_rgb(source)
        output = predict_region(
            bank, truth, (0, 0, truth.shape[2], truth.shape[1]), clip=False
        )
        for index in by_source[source_id]:
            sample = samples[index]
            center = 3
            errors.append(
                output[:, sample.y + center, sample.x + center]
                - truth[:, sample.y + center, sample.x + center]
            )
    result = {
        "format": "rawtherapee-xtrans-tgmr-reduce-gmax-validation-v1",
        "metric": _metric(errors),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(canonical_json_bytes(result))
    return result


def _posterior_summary(batch, truth: np.ndarray) -> dict[str, float]:
    error = np.mean((batch.mmse_rgb - truth) ** 2, axis=1)
    correlation = float(spearmanr(batch.predictive_risk, error).statistic)
    return {
        "effective_component_count_mean": float(
            np.mean(batch.effective_component_count)
        ),
        "entropy_mean": float(np.mean(batch.entropy)),
        "maximum_responsibility_mean": float(
            np.mean(batch.maximum_responsibility)
        ),
        "risk_error_spearman": correlation if math.isfinite(correlation) else None,
    }


def _factor_row(model, samples, truth, full_batch, rank: int) -> dict[str, object]:
    factor = approximate_factor_model(model, rank)
    batch = factor_student_t_predict(
        prepare_factor_cache(factor, FROZEN_TAU),
        samples,
        degrees_of_freedom=DEGREES_OF_FREEDOM,
        temperature=FROZEN_TEMPERATURE,
    )
    return {
        "rank": rank,
        "approximation": factor.diagnostics,
        "conditional_component_rgb_rms_difference": float(math.sqrt(np.mean(
            (batch.component_rgb - full_batch.component_rgb) ** 2
        ))),
        "log_responsibility_rms_difference": float(math.sqrt(np.mean(
            (batch.log_responsibilities - full_batch.log_responsibilities) ** 2
        ))),
        "metric": _metric(batch.mmse_rgb - truth),
        "posterior": _posterior_summary(batch, truth),
        "parameter_count": factor_parameter_count(factor),
        "estimated_mac_per_pixel": factor_mac_estimate(factor),
    }


def run(
    output: Path,
    full_model_path: Path,
    component_models: dict[int, Path],
    bsds_root: Path,
    gmax_cache: Path,
    parent_results: Path,
) -> dict[str, object]:
    parent = json.loads(parent_results.read_text(encoding="utf-8"))
    corpus, splits = corpus_manifest(bsds_root)
    validation = evaluation_samples(
        splits["val"], 7, "bsds-validation", VALIDATION_GRID_SIDE
    )
    truth = _truth(validation)
    gmax = _gmax_validation(splits, validation, gmax_cache)
    full_model = load_model(full_model_path)
    full_result, full_batch = evaluate_student(
        prepare_gmr_cache(full_model, FROZEN_TAU), validation, DEGREES_OF_FREEDOM
    )
    expected = parent["ecm_checkpoints"]
    expected_full = next(
        row for row in expected if int(row["ecm_iterations"]) == 30
    )["validation"]["methods"]["mmse"]
    full_metric = full_result["methods"]["mmse"]
    if full_metric != expected_full:
        raise RuntimeError("TGMR64-FULL does not reproduce the frozen validation result")
    gmax_mse = float(gmax["metric"]["mse"])
    full_mse = float(full_metric["mse"])

    components = []
    component_state = {}
    for count in (8, 16, 32):
        model = load_model(component_models[count])
        evaluation, batch = evaluate_student(
            prepare_gmr_cache(model, FROZEN_TAU), validation, DEGREES_OF_FREEDOM
        )
        component_state[count] = (model, batch)
        metric = evaluation["methods"]["mmse"]
        components.append({
            "component_count": count,
            "model": {**model_summary(model), "logical_sha256": _logical_digest(model)},
            "metric": metric,
            "quality_retention": _quality_retention(
                gmax_mse, full_mse, float(metric["mse"])
            ),
            "relative_dense_cost": count / 64.0,
        })
    components.append({
        "component_count": 64,
        "model": {
            **model_summary(full_model),
            "logical_sha256": _logical_digest(full_model),
        },
        "metric": full_metric,
        "quality_retention": 1.0,
        "relative_dense_cost": 1.0,
    })
    components.sort(key=lambda row: int(row["component_count"]))

    pruning = []
    for policy in ("weight", "training-responsibility"):
        for count in PRUNED_COUNTS:
            reduced = prune_components(full_model, count, policy=policy)
            evaluation, _ = evaluate_student(
                prepare_gmr_cache(reduced, FROZEN_TAU),
                validation,
                DEGREES_OF_FREEDOM,
            )
            metric = evaluation["methods"]["mmse"]
            pruning.append({
                "policy": policy,
                "retained": count,
                "metric": metric,
                "quality_retention": _quality_retention(
                    gmax_mse, full_mse, float(metric["mse"])
                ),
                "relative_dense_cost": count / 64.0,
            })

    factors = []
    for rank in FACTOR_RANKS:
        print(f"factor rank={rank}", flush=True)
        row = _factor_row(full_model, validation, truth, full_batch, rank)
        row["quality_retention"] = _quality_retention(
            gmax_mse, full_mse, float(row["metric"]["mse"])
        )
        factors.append(row)

    truncation = []
    for retained in TRUNCATED_COUNTS:
        batch = truncate_exact_posterior(full_batch, validation, 7, retained)
        metric = _metric(batch.mmse_rgb - truth)
        truncation.append({
            "retained": retained,
            "metric": metric,
            "quality_retention": _quality_retention(
                gmax_mse, full_mse, float(metric["mse"])
            ),
            "posterior_mass": {
                "mean": float(np.mean(batch.retained_exact_mass)),
                "p05": float(np.quantile(batch.retained_exact_mass, 0.05)),
                "minimum": float(np.min(batch.retained_exact_mass)),
            },
        })

    shortlist_trigger = any(
        int(row["retained"]) <= 16 and float(row["quality_retention"]) >= 0.99
        for row in truncation
    )
    shortlists = []
    if shortlist_trigger:
        full_cache = prepare_gmr_cache(full_model, FROZEN_TAU)
        for support in (3, 5):
            order = coarse_component_order(
                full_cache,
                validation,
                degrees_of_freedom=DEGREES_OF_FREEDOM,
                support=support,
            )
            for retained in SHORTLIST_COUNTS:
                batch = shortlist_student_t_predict(
                    full_cache,
                    full_batch,
                    validation,
                    degrees_of_freedom=DEGREES_OF_FREEDOM,
                    support=support,
                    retained=retained,
                    component_order=order,
                )
                metric = _metric(batch.mmse_rgb - truth)
                shortlists.append({
                    "support": support,
                    "retained": retained,
                    "metric": metric,
                    "quality_retention": _quality_retention(
                        gmax_mse, full_mse, float(metric["mse"])
                    ),
                    "exact_top_inclusion": float(np.mean(batch.exact_top_included)),
                    "posterior_mass": {
                        "mean": float(np.mean(batch.retained_exact_mass)),
                        "p05": float(np.quantile(batch.retained_exact_mass, 0.05)),
                        "minimum": float(np.min(batch.retained_exact_mass)),
                    },
                })

    qualifying_components = [
        row for row in components
        if int(row["component_count"]) <= 32
        and float(row["quality_retention"]) >= 0.9
    ]
    selected_component = (
        min(qualifying_components, key=lambda row: int(row["component_count"]))
        if qualifying_components else None
    )
    qualifying_factors = [
        row for row in factors
        if int(row["rank"]) <= 24 and float(row["quality_retention"]) >= 0.9
    ]
    selected_factor = (
        min(qualifying_factors, key=lambda row: int(row["rank"]))
        if qualifying_factors else None
    )
    qualifying_pruning = [
        row for row in pruning
        if int(row["retained"]) <= 32
        and float(row["quality_retention"]) >= 0.9
    ]
    selected_pruning = (
        min(
            qualifying_pruning,
            key=lambda row: (
                int(row["retained"]),
                0 if row["policy"] == "training-responsibility" else 1,
            ),
        )
        if qualifying_pruning else None
    )
    qualifying_shortlists = [
        row for row in shortlists if float(row["quality_retention"]) >= 0.9
    ]
    selected_shortlist = (
        min(
            qualifying_shortlists,
            key=lambda row: (int(row["support"]), int(row["retained"])),
        )
        if qualifying_shortlists else None
    )

    combined_shortlists = []
    if selected_component is not None and shortlist_trigger:
        shortlist_grid = {
            8: (2, 4),
            16: (4, 8),
            32: (4, 8, 16),
        }
        dense_component_cost = 163072.0 / 64.0
        for count, retained_values in shortlist_grid.items():
            model, batch = component_state[count]
            cache = prepare_gmr_cache(model, FROZEN_TAU)
            for support in (3, 5):
                order = coarse_component_order(
                    cache,
                    validation,
                    degrees_of_freedom=DEGREES_OF_FREEDOM,
                    support=support,
                )
                dimension = support * support
                for retained in retained_values:
                    shortlisted = shortlist_student_t_predict(
                        cache,
                        batch,
                        validation,
                        degrees_of_freedom=DEGREES_OF_FREEDOM,
                        support=support,
                        retained=retained,
                        component_order=order,
                    )
                    metric = _metric(shortlisted.mmse_rgb - truth)
                    estimated_cost = (
                        count * (dimension * dimension + dimension)
                        + retained * dense_component_cost
                    )
                    combined_shortlists.append({
                        "component_count": count,
                        "support": support,
                        "retained": retained,
                        "metric": metric,
                        "quality_retention": _quality_retention(
                            gmax_mse, full_mse, float(metric["mse"])
                        ),
                        "exact_top_inclusion": float(np.mean(
                            shortlisted.exact_top_included
                        )),
                        "posterior_mass_mean": float(np.mean(
                            shortlisted.retained_exact_mass
                        )),
                        "estimated_mac_per_pixel": estimated_cost,
                        "arithmetic_reduction": 163072.0 / estimated_cost,
                    })
    qualifying_combined = [
        row for row in combined_shortlists
        if float(row["quality_retention"]) >= 0.9
    ]
    selected_combined = (
        min(
            qualifying_combined,
            key=lambda row: (
                float(row["estimated_mac_per_pixel"]),
                int(row["component_count"]),
                int(row["support"]),
                int(row["retained"]),
            ),
        )
        if qualifying_combined else None
    )

    result = {
        "format": "rawtherapee-xtrans-tgmr-reduction-screen-v1",
        "parents": {
            "tgmr_results_sha256": hashlib.sha256(parent_results.read_bytes()).hexdigest(),
            "full_model_logical_sha256": _logical_digest(full_model),
            "dataset_format": corpus["format"],
        },
        "frozen": {
            "degrees_of_freedom": DEGREES_OF_FREEDOM,
            "tau": FROZEN_TAU,
            "temperature": FROZEN_TEMPERATURE,
            "gmax_validation": gmax["metric"],
            "tgmr64_validation": full_metric,
        },
        "component_count": components,
        "posthoc_pruning": pruning,
        "factor_rank": factors,
        "exact_posterior_truncation": truncation,
        "shortlist_triggered": shortlist_trigger,
        "coarse_shortlist": shortlists,
        "combined_component_shortlist": combined_shortlists,
        "validation_selection": {
            "component": selected_component,
            "pruning": selected_pruning,
            "factor": selected_factor,
            "shortlist": selected_shortlist,
            "combined": selected_combined,
            "rules": {
                "component": "smallest K<=32 retaining >=0.9",
                "pruning": "smallest K'<=32 retaining >=0.9; training responsibility wins ties",
                "factor": "smallest rank<=24 retaining >=0.9",
                "shortlist_trigger": "exact top-q q<=16 retaining >=0.99",
                "shortlist": "cheapest S9/S25 then q retaining >=0.9",
                "combined": "lowest estimated cost retaining >=0.9 after independent K and shortlist triggers",
            },
        },
        "triggers": {
            "true_factor_training": selected_factor is not None,
            "shortlist": shortlist_trigger,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--k8", type=Path, required=True)
    parser.add_argument("--k16", type=Path, required=True)
    parser.add_argument("--k32", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--gmax-cache", type=Path,
        default=Path("/tmp/xtrans-tgmr-reduce/gmax-validation.json"),
    )
    parser.add_argument(
        "--parent-results", type=Path,
        default=Path("devnotes/images/xtrans-tgmr/results.json"),
    )
    arguments = parser.parse_args()
    result = run(
        arguments.output,
        arguments.full_model,
        {8: arguments.k8, 16: arguments.k16, 32: arguments.k32},
        arguments.bsds_root,
        arguments.gmax_cache,
        arguments.parent_results,
    )
    print(json.dumps(result["validation_selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

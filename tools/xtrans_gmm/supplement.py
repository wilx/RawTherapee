#!/usr/bin/env python3
"""Run frozen post-selection diagnostics for the X-Trans GMM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from tools.xtrans_alias.analysis import canonical_json_bytes

from .dataset import (
    TEST_GRID_SIDE,
    VALIDATION_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
    training_samples,
)
from .experiment import (
    COVARIANCE_FLOOR,
    SEED,
    TAUS,
    TEMPERATURES,
    _evaluate,
    _load_model,
    _save_model,
    _write_json,
)
from .freeze import freeze
from .model import (
    JointColorGMM,
    fit_joint_gmm,
    full_rgb_log_likelihood,
    model_summary,
    prepare_conditional_cache,
    sample_matrix,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(
    model: JointColorGMM,
    path: Path,
    covariance_type: str,
) -> dict[str, object]:
    result = model_summary(model)
    dimension = 3 * model.patch_size * model.patch_size
    if covariance_type == "diagonal":
        parameters = model.component_count * (2 * dimension) + model.component_count - 1
    else:
        parameters = (
            model.component_count * dimension
            + model.component_count * dimension * (dimension + 1) // 2
            + model.component_count - 1
        )
    result.update({
        "artifact_sha256": _digest(path),
        "covariance_type": covariance_type,
        "independent_parameter_count": parameters,
    })
    return result


def _fit_or_load(
    path: Path,
    rows: list[dict[str, object]],
    patch_size: int,
    components: int,
    dc_mode: str,
    *,
    covariance_type: str,
    seed: int,
) -> JointColorGMM:
    if path.exists():
        return _load_model(path)
    matrix = sample_matrix(training_samples(rows, patch_size), dc_mode)
    model = fit_joint_gmm(
        matrix,
        patch_size,
        components,
        dc_mode=dc_mode,
        covariance_floor=COVARIANCE_FLOOR,
        seed=seed,
        maximum_iterations=30,
        tolerance=1e-3,
        covariance_type=covariance_type,
    )
    _save_model(path, model)
    return model


def _select_validation(
    model: JointColorGMM,
    validation: list,
) -> dict[str, object]:
    tau_rows = [_evaluate(model, validation, tau, 1.0)[0] for tau in TAUS]
    tau = float(min(
        tau_rows, key=lambda row: float(row["methods"]["cfa-mmse"]["mse"])
    )["tau"])
    rows = [_evaluate(model, validation, tau, value)[0] for value in TEMPERATURES]
    return min(rows, key=lambda row: float(row["methods"]["cfa-mmse"]["mse"]))


def _mean_log_evidence(model: JointColorGMM, samples: list) -> dict[str, float | int]:
    joint = full_rgb_log_likelihood(model, samples)
    maximum = np.max(joint, axis=1)
    evidence = maximum + np.log(np.sum(np.exp(joint - maximum[:, None]), axis=1))
    return {
        "count": int(evidence.size),
        "maximum": float(np.max(evidence)),
        "mean": float(np.mean(evidence)),
        "minimum": float(np.min(evidence)),
        "standard_deviation": float(np.std(evidence)),
    }


def _projected_overlap(model: JointColorGMM, tau: float) -> dict[str, object]:
    cache = prepare_conditional_cache(model, tau)
    phase_rows = []
    all_distances = []
    for phase_index, phase in enumerate(cache.phases):
        observed = phase.observed_indices
        distances = []
        for left in range(model.component_count):
            left_covariance = model.covariances[left][np.ix_(observed, observed)].copy()
            left_covariance.flat[::left_covariance.shape[0] + 1] += tau * tau
            for right in range(left + 1, model.component_count):
                right_covariance = model.covariances[right][np.ix_(observed, observed)].copy()
                right_covariance.flat[::right_covariance.shape[0] + 1] += tau * tau
                average = 0.5 * (left_covariance + right_covariance)
                factor = cho_factor(average, lower=True, check_finite=True)
                delta = phase.means_observed[left] - phase.means_observed[right]
                quadratic = float(delta @ cho_solve(factor, delta, check_finite=True)) / 8.0
                logdet_average = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
                determinant = 0.5 * (
                    logdet_average
                    - 0.5 * (
                        phase.log_determinants[left]
                        + phase.log_determinants[right]
                    )
                )
                distances.append(max(0.0, quadratic + determinant))
        values = np.asarray(distances, dtype=np.float64)
        all_distances.extend(distances)
        phase_rows.append({
            "distance_minimum": float(np.min(values)),
            "distance_median": float(np.median(values)),
            "distance_p10": float(np.quantile(values, 0.10)),
            "pairs_below_1": int(np.sum(values < 1.0)),
            "phase": phase_index,
        })
    values = np.asarray(all_distances, dtype=np.float64)
    return {
        "distance_interpretation": "Bhattacharyya distance between component distributions after physical X-Trans projection",
        "distance_minimum": float(np.min(values)),
        "distance_median": float(np.median(values)),
        "distance_p10": float(np.quantile(values, 0.10)),
        "pair_phase_count": int(values.size),
        "pairs_below_1": int(np.sum(values < 1.0)),
        "phases": phase_rows,
    }


def run(
    output: Path,
    results_path: Path,
    bsds_root: Path,
    starfield: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads(results_path.read_text(encoding="utf-8"))
    _, splits = corpus_manifest(bsds_root)
    selected = result["selected_configuration"]
    patch_size = int(selected["patch_size"])
    components = int(selected["component_count"])
    dc_mode = str(selected["dc_mode"])
    tau = float(selected["tau"])
    temperature = float(selected["temperature"])
    model_root = output / "models"
    selected_path = model_root / f"p{patch_size}-k{components}-{dc_mode}.npz"
    selected_model = _load_model(selected_path)
    validation = evaluation_samples(
        splits["val"], patch_size, "bsds-validation", VALIDATION_GRID_SIDE
    )
    test = evaluation_samples(
        splits["test"], patch_size, "bsds-test", TEST_GRID_SIDE
    )

    diagonal_path = model_root / f"p{patch_size}-k{components}-{dc_mode}-diagonal.npz"
    diagonal = _fit_or_load(
        diagonal_path, splits["train"], patch_size, components, dc_mode,
        covariance_type="diagonal", seed=SEED + 0xD1A6,
    )
    diagonal_validation = _select_validation(diagonal, validation)
    diagonal_test = _evaluate(
        diagonal, test, float(diagonal_validation["tau"]),
        float(diagonal_validation["temperature"]),
    )[0]

    training_curve = []
    for source_count in (25, 50, 100, 200):
        if source_count == 200:
            model = selected_model
            path = selected_path
        else:
            path = model_root / (
                f"p{patch_size}-k{components}-{dc_mode}-n{source_count}.npz"
            )
            model = _fit_or_load(
                path, splits["train"][:source_count], patch_size, components,
                dc_mode, covariance_type="full", seed=SEED + source_count,
            )
        training_curve.append({
            "model": _summary(model, path, "full"),
            "source_count": source_count,
            "test": _evaluate(model, test, tau, temperature)[0]["methods"]["cfa-mmse"],
            "validation": _evaluate(model, validation, tau, temperature)[0]["methods"]["cfa-mmse"],
        })

    established = established_samples(starfield, patch_size)
    likelihood = {
        "bsds-test": _mean_log_evidence(selected_model, test),
        "external-chromatic": _mean_log_evidence(
            selected_model, external_samples(patch_size)
        ),
    }
    for name, samples in established.items():
        likelihood[name] = _mean_log_evidence(selected_model, samples)

    supplement = freeze({
        "diagonal_control": {
            "model": _summary(diagonal, diagonal_path, "diagonal"),
            "test": diagonal_test,
            "validation": diagonal_validation,
        },
        "format": "rawtherapee-xtrans-joint-color-gmm-supplement-v1",
        "full_selected_model": _summary(selected_model, selected_path, "full"),
        "mean_full_rgb_log_evidence": likelihood,
        "projected_component_overlap": _projected_overlap(selected_model, tau),
        "selection_frozen_from": hashlib.sha256(
            canonical_json_bytes(freeze(result))
        ).hexdigest(),
        "training_size_curve": training_curve,
    })
    _write_json(output / "supplement.json", supplement)
    return supplement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    arguments = parser.parse_args()
    result = run(
        arguments.output, arguments.results, arguments.bsds_root, arguments.starfield
    )
    print(json.dumps({
        "diagonal_test_psnr_db": result["diagonal_control"]["test"]["methods"]["cfa-mmse"]["psnr_db"],
        "projected_distance_minimum": result["projected_component_overlap"]["distance_minimum"],
        "training_curve": [
            [row["source_count"], row["test"]["psnr_db"]]
            for row in result["training_size_curve"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

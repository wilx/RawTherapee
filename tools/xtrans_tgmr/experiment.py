#!/usr/bin/env python3
"""Run the phase-conditioned Student-t GMR safety experiment."""

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
from scipy.stats import spearmanr

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmm.dataset import (
    TEST_GRID_SIDE,
    VALIDATION_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
)
from tools.xtrans_gmr.experiment import (
    PATCH_SIZE,
    _metric,
    _rows,
    _synthetic_samples,
    _truth,
    load_model,
)
from tools.xtrans_gmr.model import conditional_predict, prepare_gmr_cache

from .model import conditional_student_t_predict


DEGREES_OF_FREEDOM = (3.0, 5.0, 10.0, 20.0, 50.0, math.inf)
FROZEN_TAU = 3e-4
FROZEN_TEMPERATURE = 4.0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _uncertainty_bins(risk: np.ndarray, error: np.ndarray) -> list[dict[str, float | int]]:
    order = np.argsort(risk, kind="stable")
    rows = []
    for group, selected in enumerate(np.array_split(order, 10)):
        rows.append({
            "bin": group,
            "count": int(selected.size),
            "risk_mean": float(np.mean(risk[selected])),
            "error_rms": float(math.sqrt(np.mean(error[selected]))),
        })
    return rows


def _posterior_rows(samples, batch, squared: np.ndarray) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.source_id, []).append(index)
    return {
        source: {
            "count": len(indices),
            "entropy_mean": float(np.mean(batch.entropy[indices])),
            "maximum_responsibility_mean": float(
                np.mean(batch.maximum_responsibility[indices])
            ),
            "predictive_risk_mean": float(np.mean(batch.predictive_risk[indices])),
            "rgb_error_rms": float(math.sqrt(np.mean(squared[indices]))),
        }
        for source, indices in groups.items()
    }


def evaluate_student(cache, samples, degrees_of_freedom: float) -> tuple[dict[str, object], object]:
    batch = conditional_student_t_predict(
        cache,
        samples,
        degrees_of_freedom=degrees_of_freedom,
        temperature=FROZEN_TEMPERATURE,
    )
    truth = _truth(samples, cache.model.patch_size)
    squared = np.mean((batch.mmse_rgb - truth) ** 2, axis=1)
    return ({
        "degrees_of_freedom": "infinity" if math.isinf(degrees_of_freedom) else degrees_of_freedom,
        "methods": {"mmse": _metric(batch.mmse_rgb - truth)},
        "rows": _rows(samples, {"mmse": batch.mmse_rgb}, truth),
        "posterior": {
            "effective_component_count_mean": float(np.mean(batch.effective_component_count)),
            "entropy_mean": float(np.mean(batch.entropy)),
            "entropy_p95": float(np.quantile(batch.entropy, 0.95)),
            "maximum_responsibility_mean": float(np.mean(batch.maximum_responsibility)),
            "maximum_responsibility_p05": float(np.quantile(batch.maximum_responsibility, 0.05)),
            "top_two_ratio_median": float(np.median(batch.top_two_ratio)),
        },
        "posterior_rows": _posterior_rows(samples, batch, squared),
        "predictive_uncertainty": {
            "risk_error_spearman": _finite_correlation(batch.predictive_risk, squared),
            "risk_minimum": float(np.min(batch.predictive_risk)),
            "risk_median": float(np.median(batch.predictive_risk)),
            "risk_maximum": float(np.max(batch.predictive_risk)),
            "decile_calibration": _uncertainty_bins(batch.predictive_risk, squared),
        },
    }, batch)


def _phase_range(result: dict[str, object]) -> float:
    values = [float(row["mmse"]["psnr_db"]) for row in result["rows"].values()]
    return max(values) - min(values)


def _synthetic_scenes() -> dict[str, np.ndarray]:
    size = 48
    y, x = np.mgrid[:size, :size]
    center = size // 2
    scenes = {
        "gray-gradient": np.stack([0.1 + 0.8 * x / (size - 1)] * 3),
        "chromatic-gradient": np.stack((
            0.1 + 0.8 * x / (size - 1),
            0.15 + 0.7 * y / (size - 1),
            0.2 + 0.6 * (x + y) / (2 * size - 2),
        )),
        "red-gray-edge": np.stack((
            0.2 + 0.8 * (x >= center),
            0.2 + 0.3 * (x >= center),
            0.2 + 0.3 * (x >= center),
        )),
        "periodic-chromatic": np.stack((
            0.15 + 0.75 * ((x + y) % 4 == 0),
            0.15 + 0.75 * ((x - y) % 5 == 0),
            0.15 + 0.75 * (x % 3 == 0),
        )),
        "tiny-star": np.full((3, size, size), 0.03),
        "saturated-point": np.full((3, size, size), 0.1),
    }
    scenes["tiny-star"][:, center, center] = (1.0, 0.55, 0.15)
    scenes["saturated-point"][:, center, center] = (1.0, 0.0, 0.9)
    return scenes


def run_stage_a(
    output: Path,
    bsds_root: Path,
    starfield: Path,
    model_path: Path,
    gmr_results_path: Path,
    full_gmm_results_path: Path,
) -> dict[str, object]:
    gmr_results = json.loads(gmr_results_path.read_text(encoding="utf-8"))
    full_gmm_results = json.loads(full_gmm_results_path.read_text(encoding="utf-8"))
    if gmr_results.get("format") != "rawtherapee-xtrans-phase-conditioned-gmr-v1":
        raise RuntimeError("unexpected frozen GMR result format")
    model = load_model(model_path)
    selected = gmr_results["selected"]
    if (
        model.component_count != int(selected["component_count"])
        or min(phase.iterations for phase in model.phases) != int(selected["iterations"])
        or float(selected["tau"]) != FROZEN_TAU
        or float(selected["temperature"]) != FROZEN_TEMPERATURE
    ):
        raise RuntimeError("model does not match frozen GMR selection")
    cache = prepare_gmr_cache(model, FROZEN_TAU)
    corpus, splits = corpus_manifest(bsds_root)
    validation = evaluation_samples(
        splits["val"], PATCH_SIZE, "bsds-validation", VALIDATION_GRID_SIDE
    )
    validation_rows = []
    validation_batches = {}
    for nu in DEGREES_OF_FREEDOM:
        result, batch = evaluate_student(cache, validation, nu)
        validation_rows.append(result)
        validation_batches[str(nu)] = batch
        print(f"stage A nu={nu}: {result['methods']['mmse']['psnr_db']:.4f} dB", flush=True)

    gaussian = conditional_predict(
        cache, validation, temperature=FROZEN_TEMPERATURE, compute_joint_map=False
    )
    infinity = validation_batches[str(math.inf)]
    parity = {
        "all_18_phases_present": len({sample.phase for sample in validation}) == 18,
        "maximum_output_abs_difference": float(np.max(np.abs(infinity.mmse_rgb - gaussian.mmse_rgb))),
        "maximum_responsibility_abs_difference": float(
            np.max(np.abs(infinity.responsibilities - gaussian.responsibilities))
        ),
        "bit_identical_output": bool(np.array_equal(infinity.mmse_rgb, gaussian.mmse_rgb)),
        "bit_identical_responsibilities": bool(
            np.array_equal(infinity.responsibilities, gaussian.responsibilities)
        ),
    }
    finite_rows = [row for row in validation_rows if row["degrees_of_freedom"] != "infinity"]
    chosen = min(finite_rows, key=lambda row: float(row["methods"]["mmse"]["mse"]))
    chosen_nu = float(chosen["degrees_of_freedom"])
    gaussian_validation = float(validation_rows[-1]["methods"]["mmse"]["psnr_db"])
    chosen_validation = float(chosen["methods"]["mmse"]["psnr_db"])
    stage_a_promising = chosen_validation >= gaussian_validation - 0.2

    external = external_samples(PATCH_SIZE)
    external_result, _ = evaluate_student(cache, external, chosen_nu)
    external_gaussian, _ = evaluate_student(cache, external, math.inf)
    stars = established_samples(starfield, PATCH_SIZE)
    star_results = {
        name: evaluate_student(cache, samples, chosen_nu)[0]
        for name, samples in stars.items()
    }
    synthetic = []
    previous_synthetic = {
        row["scene"]: row for row in gmr_results["synthetic"]["scenes"]
    }
    for name, scene in _synthetic_scenes().items():
        result, _ = evaluate_student(cache, _synthetic_samples(scene), chosen_nu)
        synthetic.append({
            "scene": name,
            "gmax": previous_synthetic[name]["gmax"],
            "gaussian_gmr": previous_synthetic[name]["gmr"],
            "student_t": result["methods"]["mmse"],
            "phase_range_db": {
                "gmax": previous_synthetic[name]["phase_range_db"]["gmax"],
                "gaussian_gmr": previous_synthetic[name]["phase_range_db"]["gmr"],
                "student_t": _phase_range(result),
            },
        })

    gmax_rows = full_gmm_results["external_reference"]
    external_deltas = {
        source: float(row["mmse"]["psnr_db"])
        - float(gmax_rows[source]["methods"]["gmax"]["psnr_db"])
        for source, row in external_result["rows"].items()
    }
    result = {
        "format": "rawtherapee-xtrans-student-t-gmr-stage-a-v1",
        "parent": {
            "gmr_results_sha256": hashlib.sha256(gmr_results_path.read_bytes()).hexdigest(),
            "full_gmm_results_sha256": hashlib.sha256(full_gmm_results_path.read_bytes()).hexdigest(),
            "dataset_format": corpus["format"],
        },
        "frozen": {
            "patch_size": PATCH_SIZE,
            "phase_count": 18,
            "component_count": model.component_count,
            "iterations": min(phase.iterations for phase in model.phases),
            "tau": FROZEN_TAU,
            "temperature": FROZEN_TEMPERATURE,
            "model_logical_sha256": selected["model"]["logical_sha256"],
        },
        "gaussian_limit_parity": parity,
        "validation_sweep": validation_rows,
        "selected_finite_nu": chosen_nu,
        "selected_validation_delta_vs_gaussian_db": chosen_validation - gaussian_validation,
        "stage_a_promising": stage_a_promising,
        "external": external_result,
        "external_gaussian": external_gaussian,
        "external_delta_vs_gmax_db": external_deltas,
        "stars": star_results,
        "synthetic": synthetic,
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("/tmp/xtrans-gmr/models/p7-k64-seed1481067858-i10-n200.npz"),
    )
    parser.add_argument(
        "--gmr-results", type=Path,
        default=Path("devnotes/images/xtrans-gmr/results.json"),
    )
    parser.add_argument(
        "--full-gmm-results", type=Path,
        default=Path("devnotes/images/xtrans-gmm/results.json"),
    )
    arguments = parser.parse_args()
    result = run_stage_a(
        arguments.output,
        arguments.bsds_root,
        arguments.starfield,
        arguments.model,
        arguments.gmr_results,
        arguments.full_gmm_results,
    )
    print(json.dumps({
        "selected_finite_nu": result["selected_finite_nu"],
        "validation_delta_db": result["selected_validation_delta_vs_gaussian_db"],
        "stage_a_promising": result["stage_a_promising"],
        "worst_external_delta_db": min(result["external_delta_vs_gmax_db"].values()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

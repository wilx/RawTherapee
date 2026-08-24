#!/usr/bin/env python3
"""Collect frozen Student-t GMR checkpoints into one canonical evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

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
    _logical_digest,
    _metric,
    _oracle,
    _synthetic_samples,
    _truth,
    load_model,
)
from tools.xtrans_gmr.model import model_summary, prepare_gmr_cache
from tools.xtrans_sparse_dictionary.dataset import PatchSample

from .experiment import (
    FROZEN_TAU,
    _phase_range,
    _synthetic_scenes,
    evaluate_student,
)


MODEL_PATTERN = re.compile(r"nu(?P<nu>[0-9.]+)-i(?P<iterations>[0-9]+)-n(?P<sources>[0-9]+)\.npz$")


def _models(directory: Path) -> dict[tuple[float, int, int], Path]:
    rows = {}
    for path in directory.glob("nu*-i*-n*.npz"):
        match = MODEL_PATTERN.fullmatch(path.name)
        if match:
            rows[(
                float(match.group("nu")),
                int(match.group("iterations")),
                int(match.group("sources")),
            )] = path
    return rows


def _evaluate(path: Path, nu: float, samples) -> tuple[dict[str, object], object, object]:
    model = load_model(path)
    result, batch = evaluate_student(prepare_gmr_cache(model, FROZEN_TAU), samples, nu)
    return result, batch, model


def _model_row(path: Path, nu: float, iterations: int, samples) -> dict[str, object]:
    result, _, model = _evaluate(path, nu, samples)
    return {
        "degrees_of_freedom": nu,
        "ecm_iterations": iterations,
        "model": {**model_summary(model), "logical_sha256": _logical_digest(model)},
        "validation": result,
    }


def _training_weight_diagnostic(
    vectors_path: Path,
    model,
    degrees_of_freedom: float,
) -> dict[str, object]:
    vectors = np.load(vectors_path, mmap_mode="r")
    selected_rows = np.linspace(0, vectors.shape[0] - 1, 256, dtype=np.int64)
    samples = []
    for phase_index, phase in enumerate(model.phases):
        for row in selected_rows:
            samples.append(PatchSample(
                source_id=f"training-phase-{phase_index}",
                group="training-robustness-diagnostic",
                vector=np.asarray(vectors[row]),
                indices=phase.observed_indices,
                phase=phase_index,
                x=int(row),
                y=phase_index,
            ))
    result, batch = evaluate_student(
        prepare_gmr_cache(model, FROZEN_TAU), samples, degrees_of_freedom
    )
    truth = np.stack([
        sample.vector[np.asarray([24, 73, 122])] for sample in samples
    ])
    error = np.mean((batch.mmse_rgb - truth) ** 2, axis=1)
    latent = (degrees_of_freedom + 49.0) / (
        degrees_of_freedom + batch.observed_mahalanobis
    )
    expected = np.sum(batch.responsibilities * latent, axis=1)
    boundary = float(np.quantile(error, 0.99))

    def summary(values: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(values.size),
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }

    return {
        "sampling": "256 deterministic evenly spaced training patches per phase",
        "prediction": result["methods"]["mmse"],
        "top_one_percent_error_threshold": boundary,
        "ordinary_expected_latent_weight": summary(expected[error < boundary]),
        "top_one_percent_expected_latent_weight": summary(expected[error >= boundary]),
    }


def run(
    output: Path,
    models_directory: Path,
    stage_a_path: Path,
    bsds_root: Path,
    starfield: Path,
    gmr_results_path: Path,
    full_gmm_results_path: Path,
    gaussian_model_path: Path,
    training_vectors_path: Path,
    learned_model_path: Path,
    learned_nu_path: Path,
) -> dict[str, object]:
    available = _models(models_directory)
    required_fixed = [(nu, 1, 200) for nu in (3.0, 5.0, 10.0, 20.0)]
    missing = [key for key in required_fixed if key not in available]
    if missing:
        raise RuntimeError(f"missing fixed-nu checkpoints: {missing}")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    gmr = json.loads(gmr_results_path.read_text(encoding="utf-8"))
    full = json.loads(full_gmm_results_path.read_text(encoding="utf-8"))
    corpus, splits = corpus_manifest(bsds_root)
    validation = evaluation_samples(
        splits["val"], PATCH_SIZE, "bsds-validation", VALIDATION_GRID_SIDE
    )
    test = evaluation_samples(splits["test"], PATCH_SIZE, "bsds-test", TEST_GRID_SIDE)

    fixed_rows = [
        _model_row(available[key], key[0], key[1], validation)
        for key in required_fixed
    ]
    selected_fixed = min(
        fixed_rows, key=lambda row: float(row["validation"]["methods"]["mmse"]["mse"])
    )
    selected_nu = float(selected_fixed["degrees_of_freedom"])
    checkpoint_keys = sorted(
        key for key in available if key[0] == selected_nu and key[2] == 200
    )
    checkpoint_rows = [
        _model_row(available[key], key[0], key[1], validation)
        for key in checkpoint_keys
    ]
    selected_checkpoint = min(
        checkpoint_rows,
        key=lambda row: float(row["validation"]["methods"]["mmse"]["mse"]),
    )
    selected_iterations = int(selected_checkpoint["ecm_iterations"])
    selected_key = (selected_nu, selected_iterations, 200)
    if selected_key not in available:
        raise RuntimeError("selected t-GMR checkpoint unavailable")

    source_rows = []
    for sources in (25, 50, 100, 200):
        key = (selected_nu, selected_iterations, sources)
        if key not in available:
            raise RuntimeError(f"missing source-curve checkpoint: {key}")
        validation_result, _, model = _evaluate(
            available[key], selected_nu, validation
        )
        test_result, _, _ = _evaluate(available[key], selected_nu, test)
        source_rows.append({
            "sources": sources,
            "model": {**model_summary(model), "logical_sha256": _logical_digest(model)},
            "validation": validation_result["methods"]["mmse"],
            "test": test_result["methods"]["mmse"],
        })

    selected_path = available[selected_key]
    selected_test, selected_batch, selected_model = _evaluate(
        selected_path, selected_nu, test
    )
    test_truth = _truth(test, PATCH_SIZE)
    trained_oracles = {}
    for block in (1, 3, 7, 15):
        prediction, _ = _oracle(
            selected_batch.component_rgb, test_truth, test, block
        )
        trained_oracles[f"oracle-{block}"] = _metric(prediction - test_truth)
    responsibility_model_path = gaussian_model_path
    responsibility_test, responsibility_batch, _ = _evaluate(
        responsibility_model_path, float(stage_a["selected_finite_nu"]), test
    )
    learned_nu = float(learned_nu_path.read_text(encoding="utf-8"))
    learned_validation, _, learned_model = _evaluate(
        learned_model_path, learned_nu, validation
    )
    learned_test, _, _ = _evaluate(learned_model_path, learned_nu, test)

    external = external_samples(PATCH_SIZE)
    selected_external, _, _ = _evaluate(selected_path, selected_nu, external)
    responsibility_external, _, _ = _evaluate(
        responsibility_model_path, float(stage_a["selected_finite_nu"]), external
    )
    learned_external, _, _ = _evaluate(learned_model_path, learned_nu, external)
    stars = established_samples(starfield, PATCH_SIZE)
    star_results = {
        name: {
            "trained_t": _evaluate(selected_path, selected_nu, samples)[0],
            "responsibility_only": _evaluate(
                responsibility_model_path,
                float(stage_a["selected_finite_nu"]),
                samples,
            )[0],
            "learned_shared_nu": _evaluate(
                learned_model_path, learned_nu, samples
            )[0],
        }
        for name, samples in stars.items()
    }

    previous_synthetic = {row["scene"]: row for row in gmr["synthetic"]["scenes"]}
    synthetic = []
    for name, scene in _synthetic_scenes().items():
        samples = _synthetic_samples(scene)
        trained = _evaluate(selected_path, selected_nu, samples)[0]
        responsibility = _evaluate(
            responsibility_model_path, float(stage_a["selected_finite_nu"]), samples
        )[0]
        previous = previous_synthetic[name]
        synthetic.append({
            "scene": name,
            "gmax": previous["gmax"],
            "gaussian_gmr": previous["gmr"],
            "responsibility_only": responsibility["methods"]["mmse"],
            "trained_t": trained["methods"]["mmse"],
            "phase_range_db": {
                "gmax": previous["phase_range_db"]["gmax"],
                "gaussian_gmr": previous["phase_range_db"]["gmr"],
                "responsibility_only": _phase_range(responsibility),
                "trained_t": _phase_range(trained),
            },
        })

    gmax_test = float(full["sampled_baselines"]["pooled"]["gmax"]["psnr_db"])
    gaussian_test = float(gmr["bsds_test"]["gmr"]["methods"]["mmse"]["psnr_db"])
    trained_test = float(selected_test["methods"]["mmse"]["psnr_db"])
    external_deltas = {
        source: float(row["mmse"]["psnr_db"])
        - float(full["external_reference"][source]["methods"]["gmax"]["psnr_db"])
        for source, row in selected_external["rows"].items()
    }
    bright_deltas = {}
    for name in ("hubble-bright", "nasa-hydra-starfield-bright"):
        bright_deltas[name] = (
            float(star_results[name]["trained_t"]["methods"]["mmse"]["psnr_db"])
            - float(full["established_reference"][name]["methods"]["gmax"]["psnr_db"])
        )
    analytical_deltas = {
        row["scene"]: float(row["trained_t"]["psnr_db"])
        - float(row["gmax"]["psnr_db"])
        for row in synthetic
    }
    source_values = [float(row["validation"]["psnr_db"]) for row in source_rows]
    checkpoint_values = [
        float(row["validation"]["methods"]["mmse"]["psnr_db"])
        for row in checkpoint_rows
    ]
    gate_a = bool(stage_a["gaussian_limit_parity"]["bit_identical_output"])
    gate_b = float(stage_a["selected_validation_delta_vs_gaussian_db"]) >= -0.2
    gate_c = trained_test - gmax_test >= 2.0 and trained_test >= gaussian_test - 0.2
    gate_d = min(external_deltas.values()) >= -0.5
    gate_e = min(bright_deltas.values()) >= 0.0
    gate_f = min(analytical_deltas.values()) >= -2.0
    gate_g = (
        all(right >= left - 0.1 for left, right in zip(source_values, source_values[1:]))
        and max(checkpoint_values) - checkpoint_values[-1] <= 0.3
    )
    if all((gate_a, gate_b, gate_c, gate_d, gate_e, gate_f, gate_g)):
        outcome = "GO - robust Student-t GMR"
    elif all((gate_a, gate_b, gate_c, gate_e, gate_f, gate_g)):
        outcome = "PARTIAL - heavy-tail safety tradeoff"
    else:
        outcome = "NO-GO - Student-t stabilization"

    result = {
        "format": "rawtherapee-xtrans-student-t-gmr-v1",
        "parents": {
            "stage_a_sha256": hashlib.sha256(stage_a_path.read_bytes()).hexdigest(),
            "gmr_results_sha256": hashlib.sha256(gmr_results_path.read_bytes()).hexdigest(),
            "full_gmm_results_sha256": hashlib.sha256(full_gmm_results_path.read_bytes()).hexdigest(),
            "dataset_format": corpus["format"],
        },
        "reference": {
            "conditional_distribution": "Peng Ding, arXiv:1604.00561",
            "mixture_ecm": "Peel and McLachlan, DOI 10.1023/A:1008981510081",
            "implementation": "independent; no reference source copied",
        },
        "stage_a": stage_a,
        "fixed_nu_training_screen": fixed_rows,
        "selected": {
            "degrees_of_freedom": selected_nu,
            "ecm_iterations": selected_iterations,
            "model": {
                **model_summary(selected_model),
                "logical_sha256": _logical_digest(selected_model),
            },
        },
        "ecm_checkpoints": checkpoint_rows,
        "learned_shared_nu": {
            "degrees_of_freedom": learned_nu,
            "model": {
                **model_summary(learned_model),
                "logical_sha256": _logical_digest(learned_model),
            },
            "validation": learned_validation,
            "test": learned_test,
            "external": learned_external,
            "component_specific_nu_deferred": (
                "phase component labels have no shared cross-phase identity"
            ),
        },
        "training_source_curve": source_rows,
        "training_latent_weight_diagnostic": _training_weight_diagnostic(
            training_vectors_path, selected_model, selected_nu
        ),
        "bsds_test": {
            "responsibility_only": responsibility_test,
            "trained_t": selected_test,
            "trained_t_component_oracles": trained_oracles,
            "per_sample_difference": {
                "maximum_abs": float(np.max(np.abs(
                    selected_batch.mmse_rgb - responsibility_batch.mmse_rgb
                ))),
            },
        },
        "external": {
            "responsibility_only": responsibility_external,
            "trained_t": selected_external,
            "delta_vs_gmax_db": external_deltas,
        },
        "stars": star_results,
        "synthetic": synthetic,
        "decision": {
            "outcome": outcome,
            "gate_a_gaussian_parity": gate_a,
            "gate_b_robust_likelihood_value": gate_b,
            "gate_c_bsds": gate_c,
            "gate_d_external_safety": gate_d,
            "gate_e_bright_stars": gate_e,
            "gate_f_analytical": gate_f,
            "gate_g_training_stability": gate_g,
            "trained_t_minus_gmax_test_db": trained_test - gmax_test,
            "trained_t_minus_gaussian_gmr_test_db": trained_test - gaussian_test,
            "trained_t_minus_responsibility_only_test_db": (
                trained_test - float(responsibility_test["methods"]["mmse"]["psnr_db"])
            ),
            "worst_external_minus_gmax_db": min(external_deltas.values()),
            "worst_bright_star_minus_gmax_db": min(bright_deltas.values()),
            "worst_analytical_minus_gmax_db": min(analytical_deltas.values()),
        },
        "complexity": {
            "parameter_count": model_summary(selected_model)["parameter_count"] + 1,
            "additional_runtime_operations_vs_gmr": "one log1p and scalar tail expression per component",
            "component_count_per_pixel": selected_model.component_count,
            "estimated_dense_mac_per_pixel": gmr["complexity"]["estimated_dense_mac_per_pixel"],
        },
        "deferred": {
            "epll": True,
            "production_cpp": True,
            "optimization": True,
            "external_selector": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    parser.add_argument(
        "--gmr-results", type=Path,
        default=Path("devnotes/images/xtrans-gmr/results.json"),
    )
    parser.add_argument(
        "--full-gmm-results", type=Path,
        default=Path("devnotes/images/xtrans-gmm/results.json"),
    )
    parser.add_argument(
        "--gaussian-model", type=Path,
        default=Path("/tmp/xtrans-gmr/models/p7-k64-seed1481067858-i10-n200.npz"),
    )
    parser.add_argument(
        "--training-vectors", type=Path,
        default=Path("/tmp/xtrans-gmr/training-vectors.npy"),
    )
    parser.add_argument("--learned-model", type=Path, required=True)
    parser.add_argument("--learned-nu", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.output,
        arguments.models,
        arguments.stage_a,
        arguments.bsds_root,
        arguments.starfield,
        arguments.gmr_results,
        arguments.full_gmm_results,
        arguments.gaussian_model,
        arguments.training_vectors,
        arguments.learned_model,
        arguments.learned_nu,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

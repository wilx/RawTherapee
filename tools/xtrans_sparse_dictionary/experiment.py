#!/usr/bin/env python3
"""Run the staged CFA-constrained sparse RGB dictionary feasibility study."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_lmmse_mixture.dataset import load_rgb
from tools.xtrans_lmmse.experiment import _run_baselines
from tools.xtrans_lmmse.model import predict_region
from tools.xtrans_lmmse_mixture.training import survey_and_train_globals
from tools.xtrans_mlri_internal.dataset import origin_cells

from .dataset import (
    PatchSample,
    bsds_samples,
    corpus_manifest,
    established_samples,
    external_samples,
)
from .model import (
    blind_cfa_reconstruction,
    dct_dictionary,
    dictionary_coherence,
    lasso_cfa_reconstruction,
    masked_coherence,
    observation_indices,
    omp,
    oracle_support_reconstruction,
    remove_observable_dc,
    restore_observed,
    train_mod_dictionary,
)


TRAINING_PER_SOURCE = 24
EVALUATION_PER_SOURCE = 24
SPARSITIES = (1, 2, 4, 8, 16)
CONFIGURATIONS = (
    (5, 128, "absolute"),
    (5, 128, "observable-dc"),
    (5, 256, "observable-dc"),
    (7, 256, "observable-dc"),
)
TRAINING_COUNTS = (25, 50, 100, 200)
SEED = 0x58445247


def _metric(differences: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate([np.asarray(row, dtype=np.float64).ravel() for row in differences])
    absolute = np.abs(values)
    mse = float(np.mean(values * values))
    return {
        "count": int(values.size),
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "mse": mse,
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
    }


def _normalize_sample(
    sample: PatchSample, normalization: str,
) -> tuple[np.ndarray, float]:
    if normalization == "observable-dc":
        return remove_observable_dc(sample.vector, sample.indices)
    if normalization == "absolute":
        return np.asarray(sample.vector, dtype=np.float64), 0.0
    raise ValueError(normalization)


def _matrix(
    samples: list[PatchSample], normalization: str = "observable-dc",
) -> np.ndarray:
    rows = []
    for sample in samples:
        normalized, _ = _normalize_sample(sample, normalization)
        rows.append(normalized)
    return np.ascontiguousarray(np.stack(rows, axis=1))


def _representation(
    dictionary: np.ndarray,
    samples: list[PatchSample],
    sparsities: tuple[int, ...] = SPARSITIES,
    normalization: str = "observable-dc",
) -> dict[str, object]:
    by_sparsity = {}
    for sparsity in sparsities:
        differences = []
        captured = []
        nonzero = []
        for sample in samples:
            normalized, dc = _normalize_sample(sample, normalization)
            coded = omp(dictionary, normalized, sparsity)
            reconstruction = dictionary @ coded.coefficients + dc
            differences.append(reconstruction - sample.vector)
            denominator = float(np.dot(normalized, normalized))
            residual = reconstruction - sample.vector
            captured.append(1.0 - float(np.dot(residual, residual)) / max(denominator, 1e-30))
            nonzero.append(int(coded.support.size))
        by_sparsity[str(sparsity)] = {
            "energy_captured_mean": float(np.mean(captured)),
            "metric": _metric(differences),
            "nonzero_mean": float(np.mean(nonzero)),
            "patch_target_coverage": {
                str(target): float(np.mean([
                    -10.0 * math.log10(
                        max(float(np.mean(row * row)), 1e-30)
                    ) >= target
                    for row in differences
                ]))
                for target in (30, 40, 50)
            },
        }
    return by_sparsity


def _three_stage(
    dictionary: np.ndarray,
    samples: list[PatchSample],
    sparsity: int,
    normalization: str = "observable-dc",
) -> dict[str, object]:
    differences = {name: [] for name in (
        "representation", "oracle-before", "oracle-enforced",
        "blind-before", "blind-enforced",
    )}
    center_differences = {name: [] for name in differences}
    conditions = []
    ranks = []
    sigma_min = []
    precision = []
    recall = []
    jaccard = []
    regret = []
    elapsed = 0.0
    patch_size = int(round(math.sqrt(dictionary.shape[0] / 3.0)))
    center_spatial = (patch_size * patch_size) // 2
    center_indices = np.asarray(
        [channel * patch_size * patch_size + center_spatial for channel in range(3)],
        dtype=np.int64,
    )
    for sample in samples:
        normalized, dc = _normalize_sample(sample, normalization)
        representation = omp(dictionary, normalized, sparsity)
        represented = dictionary @ representation.coefficients + dc
        oracle, diagnostic = oracle_support_reconstruction(
            dictionary, normalized, sample.indices, representation.support,
        )
        oracle += dc
        oracle_enforced = restore_observed(oracle, sample.vector, sample.indices)
        started = time.perf_counter()
        blind, inferred = blind_cfa_reconstruction(
            dictionary, normalized, sample.indices, sparsity,
        )
        elapsed += time.perf_counter() - started
        blind += dc
        blind_enforced = restore_observed(blind, sample.vector, sample.indices)
        outputs = {
            "representation": represented,
            "oracle-before": oracle,
            "oracle-enforced": oracle_enforced,
            "blind-before": blind,
            "blind-enforced": blind_enforced,
        }
        for name, output in outputs.items():
            error = output - sample.vector
            differences[name].append(error)
            center_differences[name].append(error[center_indices])
        conditions.append(float(diagnostic["condition_number"]))
        ranks.append(int(diagnostic["rank"]))
        sigma_min.append(float(diagnostic["sigma_min"]))
        true_support = set(int(value) for value in representation.support)
        blind_support = set(int(value) for value in inferred.support)
        intersection = len(true_support & blind_support)
        precision.append(intersection / max(len(blind_support), 1))
        recall.append(intersection / max(len(true_support), 1))
        jaccard.append(intersection / max(len(true_support | blind_support), 1))
        oracle_mse = float(np.mean((oracle_enforced - sample.vector) ** 2))
        blind_mse = float(np.mean((blind_enforced - sample.vector) ** 2))
        regret.append(blind_mse - oracle_mse)
    regret_array = np.asarray(regret)
    return {
        "center_metrics": {name: _metric(rows) for name, rows in center_differences.items()},
        "complexity": {
            "dictionary_bytes_float32": int(dictionary.size * 4),
            "dictionary_shape": list(dictionary.shape),
            "mean_blind_seconds_per_patch": elapsed / len(samples),
            "nominal_omp_iterations": sparsity,
        },
        "conditioning": {
            "condition_median": float(np.median(conditions)),
            "condition_p99": float(np.quantile(conditions, 0.99)),
            "full_rank_fraction": float(np.mean(np.asarray(ranks) == sparsity)),
            "sigma_min_median": float(np.median(sigma_min)),
        },
        "metrics": {name: _metric(rows) for name, rows in differences.items()},
        "regret": {
            "mean_mse": float(np.mean(regret_array)),
            "median_mse": float(np.median(regret_array)),
            "p90_mse": float(np.quantile(regret_array, 0.90)),
            "p99_mse": float(np.quantile(regret_array, 0.99)),
        },
        "support_recovery": {
            "jaccard_mean": float(np.mean(jaccard)),
            "precision_mean": float(np.mean(precision)),
            "recall_mean": float(np.mean(recall)),
        },
    }


def _blind_only(
    dictionary: np.ndarray,
    samples: list[PatchSample],
    solver: str,
    value: float,
    normalization: str = "observable-dc",
) -> dict[str, object]:
    patch_size = int(round(math.sqrt(dictionary.shape[0] / 3.0)))
    center = patch_size * patch_size // 2
    center_indices = np.asarray(
        [channel * patch_size * patch_size + center for channel in range(3)],
        dtype=np.int64,
    )
    differences = []
    center_differences = []
    nonzero = []
    elapsed = 0.0
    for sample in samples:
        normalized, dc = _normalize_sample(sample, normalization)
        started = time.perf_counter()
        if solver == "omp":
            reconstruction, sparse = blind_cfa_reconstruction(
                dictionary, normalized, sample.indices, int(value),
            )
        elif solver == "lasso":
            reconstruction, sparse = lasso_cfa_reconstruction(
                dictionary, normalized, sample.indices, value,
            )
        elif solver == "elastic-net":
            reconstruction, sparse = lasso_cfa_reconstruction(
                dictionary, normalized, sample.indices, value,
                elastic_ridge=1e-3,
            )
        else:
            raise ValueError(solver)
        elapsed += time.perf_counter() - started
        reconstruction = restore_observed(
            reconstruction + dc, sample.vector, sample.indices,
        )
        error = reconstruction - sample.vector
        differences.append(error)
        center_differences.append(error[center_indices])
        nonzero.append(int(sparse.support.size))
    return {
        "center_metric": _metric(center_differences),
        "mean_nonzero": float(np.mean(nonzero)),
        "metric": _metric(differences),
        "seconds_per_patch": elapsed / len(samples),
        "solver": solver,
        "value": value,
    }


def _grouped_three_stage(
    dictionary: np.ndarray,
    samples: list[PatchSample],
    sparsity: int,
    normalization: str = "observable-dc",
) -> dict[str, object]:
    grouped: dict[str, list[PatchSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_id, []).append(sample)
    return {
        source_id: _three_stage(dictionary, rows, sparsity, normalization)
        for source_id, rows in sorted(grouped.items())
    }


def _grouped_blind_only(
    dictionary: np.ndarray,
    samples: list[PatchSample],
    solver: str,
    value: float,
    normalization: str,
) -> dict[str, object]:
    grouped: dict[str, list[PatchSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_id, []).append(sample)
    return {
        source_id: _blind_only(
            dictionary, rows, solver, value, normalization,
        )
        for source_id, rows in sorted(grouped.items())
    }


def _sampled_population_comparison(
    training_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    test_samples: list[PatchSample],
    runner: Path,
) -> dict[str, object]:
    """Evaluate baselines on exactly the dictionary's sampled center pixels."""

    globals_by_size, _, _ = survey_and_train_globals(training_rows)
    gmax = globals_by_size[200]
    samples_by_source: dict[str, list[PatchSample]] = {}
    for sample in test_samples:
        samples_by_source.setdefault(sample.source_id, []).append(sample)
    pooled: dict[str, list[np.ndarray]] = {
        "markesteijn": [], "corrected-final": [], "gmax": [],
    }
    rows = []
    for source in test_rows:
        source_id = str(source["filename"])
        truth = load_rgb(source)
        region = (0, 0, truth.shape[2], truth.shape[1])
        gmax_output = predict_region(gmax, truth, region, clip=False)
        with tempfile.TemporaryDirectory(prefix="xtrans-sparse-baseline-") as directory:
            baselines, timings = _run_baselines(
                runner, truth, 0, 0, Path(directory),
            )
        outputs = {
            "markesteijn": baselines["markesteijn"],
            "corrected-final": baselines["corrected-final"],
            "gmax": gmax_output,
        }
        source_errors = {name: [] for name in outputs}
        for sample in samples_by_source[source_id]:
            sample_size = int(round(math.sqrt(sample.vector.size / 3.0)))
            center_x = sample.x + sample_size // 2
            center_y = sample.y + sample_size // 2
            target = truth[:, center_y, center_x]
            for name, output in outputs.items():
                error = output[:, center_y, center_x] - target
                source_errors[name].append(error)
                pooled[name].append(error)
        rows.append({
            "baseline_seconds": timings,
            "methods": {
                name: _metric(errors) for name, errors in source_errors.items()
            },
            "source_id": source_id,
        })
        print(f"sampled-baseline source={source_id}", flush=True)
    return {
        "pooled": {name: _metric(errors) for name, errors in pooled.items()},
        "rows": rows,
        "sampling": "identical 24 deterministic center coordinates per held-out BSDS source",
    }


def _synthetic_patch(name: str, patch_size: int) -> np.ndarray:
    y, x = np.mgrid[:patch_size, :patch_size]
    center = patch_size // 2
    if name == "gray-gradient":
        value = 0.1 + 0.8 * x / max(patch_size - 1, 1)
        return np.stack((value, value, value))
    if name == "chromatic-gradient":
        return np.stack((
            0.1 + 0.8 * x / max(patch_size - 1, 1),
            0.15 + 0.7 * y / max(patch_size - 1, 1),
            0.2 + 0.6 * (x + y) / max(2 * patch_size - 2, 1),
        ))
    if name == "red-gray-edge":
        right = x >= center
        return np.stack((0.25 + 0.75 * right, 0.25 + 0.25 * right, 0.25 + 0.25 * right))
    if name == "blue-gray-diagonal":
        right = x + y >= 2 * center
        return np.stack((0.2 + 0.2 * right, 0.2 + 0.2 * right, 0.2 + 0.8 * right))
    if name == "saturated-point":
        result = np.full((3, patch_size, patch_size), 0.03, dtype=np.float64)
        result[:, center, center] = (1.0, 0.0, 0.9)
        return result
    if name == "periodic-chromatic":
        return np.stack((
            0.1 + 0.8 * ((x + y) % 3 == 0),
            0.1 + 0.8 * ((x - y) % 4 == 0),
            0.1 + 0.8 * (x % 2 == 0),
        ))
    if name.startswith("points-spacing-"):
        spacing = int(name.rsplit("-", 1)[1])
        result = np.full((3, patch_size, patch_size), 0.05, dtype=np.float64)
        selected = (x % spacing == 0) & (y % spacing == 0)
        result[0, selected] = 1.0
        result[1, selected] = 0.15
        result[2, selected] = 0.75
        return result
    raise ValueError(name)


def _synthetic_study(
    dictionary: np.ndarray, sparsity: int, normalization: str,
) -> dict[str, object]:
    patch_size = int(round(math.sqrt(dictionary.shape[0] / 3.0)))
    names = (
        "gray-gradient", "chromatic-gradient", "red-gray-edge",
        "blue-gray-diagonal", "saturated-point", "periodic-chromatic",
        "points-spacing-4", "points-spacing-2",
    )
    result = []
    for name in names:
        truth = _synthetic_patch(name, patch_size).reshape(-1)
        phase_rows = []
        supports = []
        for origin_x, origin_y in origin_cells():
            indices = observation_indices(patch_size, origin_x, origin_y)
            sample = PatchSample(
                source_id=name, group="synthetic", vector=truth,
                indices=indices, phase=0, x=origin_x, y=origin_y,
            )
            stage = _three_stage(dictionary, [sample], sparsity, normalization)
            phase_rows.append(stage["metrics"]["blind-enforced"])
            normalized, _ = _normalize_sample(sample, normalization)
            _, inferred = blind_cfa_reconstruction(dictionary, normalized, indices, sparsity)
            supports.append(tuple(int(value) for value in inferred.support))
        psnr = [float(row["psnr_db"]) for row in phase_rows]
        result.append({
            "blind_psnr_max_db": max(psnr),
            "blind_psnr_min_db": min(psnr),
            "blind_psnr_phase_range_db": max(psnr) - min(psnr),
            "case": name,
            "distinct_supports": len(set(supports)),
            "maximum_abs": max(float(row["maximum_abs"]) for row in phase_rows),
        })
    return {"cases": result, "phase_count": len(origin_cells())}


def _coherence(dictionary: np.ndarray, patch_size: int) -> dict[str, object]:
    rows = []
    for phase, (origin_x, origin_y) in enumerate(origin_cells()):
        indices = observation_indices(patch_size, origin_x, origin_y)
        rows.append({
            "coherence": masked_coherence(dictionary, indices),
            "origin": [origin_x, origin_y],
            "phase": phase,
        })
    return {
        "full_rgb": dictionary_coherence(dictionary),
        "masked_by_phase": rows,
        "masked_maximum": max(float(row["coherence"]) for row in rows),
        "masked_minimum": min(float(row["coherence"]) for row in rows),
    }


def _render_atoms(dictionary: np.ndarray, destination: Path, maximum: int = 64) -> None:
    patch_size = int(round(math.sqrt(dictionary.shape[0] / 3.0)))
    count = min(dictionary.shape[1], maximum)
    columns = 8
    scale = 18
    rows = int(math.ceil(count / columns))
    canvas = np.zeros((rows * patch_size * scale, columns * patch_size * scale, 3), dtype=np.uint8)
    for index in range(count):
        atom = dictionary[:, index].reshape(3, patch_size, patch_size)
        maximum_abs = max(float(np.max(np.abs(atom))), 1e-12)
        rgb = np.moveaxis(0.5 + 0.5 * atom / maximum_abs, 0, -1)
        tile = np.asarray(np.clip(np.rint(rgb * 255.0), 0, 255), dtype=np.uint8)
        tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
        y = (index // columns) * patch_size * scale
        x = (index % columns) * patch_size * scale
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(destination, optimize=True)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "infinity" if value > 0 else "-infinity"
        return float(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_safe(value)))


def run(
    output: Path, bsds_root: Path, starfield: Path, runner: Path,
) -> dict[str, object]:
    started = time.monotonic()
    corpus, splits = corpus_manifest(
        bsds_root, 5, TRAINING_PER_SOURCE, EVALUATION_PER_SOURCE,
    )
    _write_json(output / "dataset.json", corpus)

    configuration_rows = []
    dictionaries: dict[tuple[int, int, str], np.ndarray] = {}
    for patch_size, atom_count, normalization in CONFIGURATIONS:
        print(
            f"prepare patch={patch_size} atoms={atom_count} normalization={normalization}",
            flush=True,
        )
        training = bsds_samples(
            splits["train"], patch_size, TRAINING_PER_SOURCE, "bsds-train",
        )
        validation = bsds_samples(
            splits["val"], patch_size, EVALUATION_PER_SOURCE, "bsds-validation",
        )
        trained = train_mod_dictionary(
            _matrix(training, normalization), atom_count,
            seed=SEED + patch_size * 1000 + atom_count + (1 if normalization == "absolute" else 0),
        )
        dictionaries[(patch_size, atom_count, normalization)] = trained.dictionary
        learned_representation = _representation(
            trained.dictionary, validation, normalization=normalization,
        )
        dct_rgb_representation = _representation(
            dct_dictionary(patch_size, color_basis="rgb"), validation,
            normalization=normalization,
        )
        dct_opponent_representation = _representation(
            dct_dictionary(patch_size, color_basis="opponent"), validation,
            normalization=normalization,
        )
        dct_representation = max(
            (dct_rgb_representation, dct_opponent_representation),
            key=lambda row: float(row["8"]["metric"]["psnr_db"]),
        )
        selected_dct_basis = (
            "rgb" if dct_representation is dct_rgb_representation else "opponent"
        )
        configuration_rows.append({
            "atom_count": atom_count,
            "dct_representation": dct_representation,
            "dct_rgb_representation": dct_rgb_representation,
            "dct_opponent_representation": dct_opponent_representation,
            "dictionary_training": {
                "coding": "FISTA L1",
                "coding_iterations": 30,
                "effective_nonzero": list(trained.effective_nonzero),
                "objectives_per_patch": list(trained.objectives),
                "outer_iterations": 6,
                "regularization": 0.015,
                "update": "ridge-regularized method of optimal directions",
            },
            "learned_representation": learned_representation,
            "normalization": normalization,
            "patch_size": patch_size,
            "selected_dct_basis": selected_dct_basis,
        })
        print(
            f"representation patch={patch_size} atoms={atom_count} normalization={normalization} "
            f"learned_s8={learned_representation['8']['metric']['psnr_db']:.3f} "
            f"dct_s8={dct_representation['8']['metric']['psnr_db']:.3f}",
            flush=True,
        )

    analytic_patch_screen = {}
    for analytic_size in (3, 5, 7, 9):
        validation = bsds_samples(
            splits["val"], analytic_size, EVALUATION_PER_SOURCE,
            "bsds-validation",
        )
        analytic_patch_screen[str(analytic_size)] = {
            basis: _representation(
                dct_dictionary(analytic_size, color_basis=basis), validation,
                tuple(
                    value for value in SPARSITIES
                    if value <= 3 * analytic_size * analytic_size
                ),
                "observable-dc",
            )
            for basis in ("rgb", "opponent")
        }

    selected = max(
        configuration_rows,
        key=lambda row: float(row["learned_representation"]["8"]["metric"]["psnr_db"]),
    )
    selected_key = (
        int(selected["patch_size"]), int(selected["atom_count"]),
        str(selected["normalization"]),
    )
    dictionary = dictionaries[selected_key]
    patch_size = selected_key[0]
    normalization = selected_key[2]
    dct = dct_dictionary(
        patch_size, color_basis=str(selected["selected_dct_basis"]),
    )
    learned_gain = float(selected["learned_representation"]["8"]["metric"]["psnr_db"]) - float(selected["dct_representation"]["8"]["metric"]["psnr_db"])
    gate1 = learned_gain >= 0.5

    training_curve = []
    if gate1:
        for count in TRAINING_COUNTS:
            training = bsds_samples(
                splits["train"][:count], 5, TRAINING_PER_SOURCE, "bsds-train",
            )
            validation = bsds_samples(
                splits["val"], 5, EVALUATION_PER_SOURCE, "bsds-validation",
            )
            trained = train_mod_dictionary(
                _matrix(training, "observable-dc"), 128, outer_iterations=4,
                seed=SEED + count,
            )
            score = _representation(
                trained.dictionary, validation, (8,), "observable-dc",
            )["8"]["metric"]
            training_curve.append({"atom_count": 128, "metric": score, "source_count": count})
            print(f"training-curve sources={count} psnr={score['psnr_db']:.3f}", flush=True)

    result: dict[str, object] = {
        "configuration_screen": configuration_rows,
        "analytic_patch_size_screen": analytic_patch_screen,
        "decision": {
            "gate1_dictionary_representation": {
                "learned_minus_dct_db_at_s8": learned_gain,
                "pass": gate1,
                "rule": "selected learned dictionary must beat matched patch-size DCT by at least 0.5 dB at S=8",
            },
        },
        "experiment_format": "rawtherapee-xtrans-sparse-dictionary-v1",
        "formulation": {
            "dictionary": "one phase-independent full-RGB dictionary",
            "inference": "OMP on physically observed X-Trans components",
            "native_samples": "restored exactly after patch reconstruction",
            "normalization": "N0 absolute and N1 observable scalar-CFA DC controls; selected mode recorded below",
            "precision": "float64 research implementation",
            "seed": SEED,
        },
        "selected_configuration": {
            "atom_count": selected_key[1],
            "normalization": normalization,
            "patch_size": selected_key[0],
            "selection": "highest BSDS-validation learned representation PSNR at S=8",
        },
        "training_size_curve": training_curve,
    }
    if not gate1:
        result["decision"].update({
            "outcome": "NO-GO - representation",
            "stopped_after": "Gate 1",
        })
        _write_json(output / "results.json", result)
        return result

    test_samples = bsds_samples(
        splits["test"], patch_size, EVALUATION_PER_SOURCE, "bsds-test",
    )
    validation_samples = bsds_samples(
        splits["val"], patch_size, EVALUATION_PER_SOURCE, "bsds-validation",
    )
    validation_stage = {
        str(sparsity): _three_stage(
            dictionary, validation_samples, sparsity, normalization,
        )
        for sparsity in SPARSITIES if sparsity <= patch_size * patch_size
    }
    selected_sparsity = min(
        (sparsity for sparsity in SPARSITIES if sparsity <= patch_size * patch_size),
        key=lambda sparsity: float(validation_stage[str(sparsity)]["metrics"]["blind-enforced"]["mse"]),
    )
    learned_test = _three_stage(
        dictionary, test_samples, selected_sparsity, normalization,
    )
    dct_test = _three_stage(dct, test_samples, selected_sparsity, normalization)
    solver_controls = [
        _blind_only(
            dictionary, validation_samples, "omp", selected_sparsity,
            normalization,
        ),
    ]
    for regularization in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        solver_controls.append(
            _blind_only(
                dictionary, validation_samples, "lasso", regularization,
                normalization,
            )
        )
    for regularization in (3e-4, 1e-3, 3e-3):
        solver_controls.append(
            _blind_only(
                dictionary, validation_samples, "elastic-net", regularization,
                normalization,
            )
        )
    selected_solver = min(
        solver_controls, key=lambda row: float(row["metric"]["mse"]),
    )
    selected_blind_test = _blind_only(
        dictionary, test_samples, str(selected_solver["solver"]),
        float(selected_solver["value"]), normalization,
    )
    representation_mse = float(learned_test["metrics"]["representation"]["mse"])
    oracle_before_mse = float(learned_test["metrics"]["oracle-before"]["mse"])
    oracle_mse = float(learned_test["metrics"]["oracle-enforced"]["mse"])
    survival = representation_mse / max(oracle_before_mse, 1e-30)
    # MSE ratio is one at equal quality; target preservation means oracle error
    # is no more than 1/0.7 of the representation error.
    gate2 = survival >= 0.70
    result.update({
        "coherence": {
            "dct": _coherence(dct, patch_size),
            "learned": _coherence(dictionary, patch_size),
        },
        "selected_sparsity": selected_sparsity,
        "three_stage": {
            "bsds_test": {"dct": dct_test, "learned": learned_test},
            "selected_blind_test": selected_blind_test,
            "solver_validation": solver_controls,
            "validation_by_sparsity": validation_stage,
        },
    })
    result["decision"]["gate2_oracle_support"] = {
        "mse_quality_survival_ratio_before_native_enforcement": survival,
        "pass": gate2,
        "rule": "representation/oracle-support MSE ratio must be at least 0.70",
    }
    _render_atoms(dictionary, output / "atoms.png")
    if not gate2:
        result["decision"].update({
            "outcome": "NO-GO - X-Trans projection",
            "stopped_after": "Gate 2",
        })
        _write_json(output / "results.json", result)
        return result

    blind_mse = float(learned_test["metrics"]["blind-enforced"]["mse"])
    dct_blind_mse = float(dct_test["metrics"]["blind-enforced"]["mse"])
    oracle_advantage = dct_blind_mse - oracle_mse
    recovered = (
        (dct_blind_mse - blind_mse) / oracle_advantage
        if oracle_advantage > 1e-30 else 0.0
    )
    gate3 = recovered >= 0.30
    result["decision"]["gate3_blind_recovery"] = {
        "oracle_advantage_recovered_fraction": recovered,
        "pass": gate3,
        "rule": "blind learned coding must recover at least 30% of the oracle-support MSE advantage over blind DCT",
    }
    result["external_chromatic"] = _three_stage(
        dictionary, external_samples(patch_size), selected_sparsity,
        normalization,
    )
    result["external_chromatic_selected_blind"] = _blind_only(
        dictionary, external_samples(patch_size), str(selected_solver["solver"]),
        float(selected_solver["value"]), normalization,
    )
    established = established_samples(starfield, patch_size)
    result["established"] = {
        source_id: _three_stage(
            dictionary, samples, selected_sparsity, normalization,
        )
        for source_id, samples in established.items()
    }
    result["established_selected_blind"] = {
        source_id: _blind_only(
            dictionary, samples, str(selected_solver["solver"]),
            float(selected_solver["value"]), normalization,
        )
        for source_id, samples in established.items()
    }
    result["bsds_test_by_source"] = _grouped_three_stage(
        dictionary, test_samples, selected_sparsity, normalization,
    )
    result["bsds_test_selected_blind_by_source"] = _grouped_blind_only(
        dictionary, test_samples, str(selected_solver["solver"]),
        float(selected_solver["value"]), normalization,
    )
    result["sampled_baseline_comparison"] = _sampled_population_comparison(
        splits["train"], splits["test"], test_samples, runner,
    )
    synthetic = _synthetic_study(dictionary, selected_sparsity, normalization)
    _write_json(output / "synthetic.json", synthetic)
    result["synthetic_summary"] = synthetic
    if not gate3:
        result["decision"].update({
            "outcome": "PARTIAL - sparse-support inference problem",
            "stopped_after": "Gate 3",
        })
    else:
        result["global_lmmse_reference"] = {
            "artifact": "devnotes/images/xtrans-lmmse-mixture/mixture.json",
            "bsds_test_psnr_db": 33.454,
            "comparison_limit": "historical full-image metric only; the gate uses the exact sampled-coordinate comparison",
        }
        center_blind = float(selected_blind_test["center_metric"]["psnr_db"])
        sampled_gmax = float(
            result["sampled_baseline_comparison"]["pooled"]["gmax"]["psnr_db"]
        )
        gate4 = center_blind > sampled_gmax + 0.1
        result["decision"]["gate4_population_value"] = {
            "pass": gate4,
            "selected_solver": {
                "name": selected_solver["solver"],
                "value": selected_solver["value"],
            },
            "selected_center_psnr_db": center_blind,
            "sampled_gmax_psnr_db": sampled_gmax,
            "rule": "sampled test center PSNR must exceed GMAX on the identical coordinates by at least 0.1 dB before whole-image overlap evaluation",
        }
        result["decision"].update({
            "outcome": "GO - sparse dictionary" if gate4 else "NO-GO - practical",
            "stopped_after": "Gate 4" if not gate4 else "none",
        })
    result["elapsed_seconds"] = time.monotonic() - started
    _write_json(output / "results.json", result)
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
        "--runner", type=Path,
        default=Path("build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests"),
    )
    arguments = parser.parse_args()
    result = run(
        arguments.output, arguments.bsds_root, arguments.starfield,
        arguments.runner.resolve(),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

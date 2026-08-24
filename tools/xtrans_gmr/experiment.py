#!/usr/bin/env python3
"""Run the phase-conditioned X-Trans Gaussian-mixture-regression study."""

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
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmm.dataset import (
    TEST_GRID_SIDE,
    VALIDATION_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
    training_samples,
)
from tools.xtrans_sparse_dictionary.dataset import PatchSample
from tools.xtrans_sparse_dictionary.model import observation_indices, phase_index
from tools.xtrans_mlri_internal.dataset import origin_cells

from .model import (
    PhaseConditionedGMR,
    PhaseMixture,
    conditional_predict,
    fit_phase_conditioned_gmr,
    model_summary,
    phase_contract,
    phase_training_matrix,
    prepare_gmr_cache,
    refine_phase_conditioned_gmr,
    shrink_covariances,
)


PATCH_SIZE = 7
COMPONENT_COUNTS = (1, 4, 8, 16, 32, 64)
SEEDS = (0x58474D52, 0x58474D53, 0x58474D54)
TAUS = (3e-4, 1e-3, 3e-3, 1e-2)
TEMPERATURES = (1.0, 2.0, 4.0)
SCREEN_ITERATIONS = 20
CHECKPOINTS = (5, 10, 20, 30, 50)
COVARIANCE_FLOOR = 1e-6


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _metric(errors: np.ndarray | list[np.ndarray]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    mse = float(np.mean(values * values))
    return {
        "count": int(values.size),
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "mse": mse,
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": float(np.sum(values * values)),
    }


def _truth(samples: list[PatchSample], patch_size: int = PATCH_SIZE) -> np.ndarray:
    area = patch_size * patch_size
    center = area // 2
    indices = np.asarray([channel * area + center for channel in range(3)])
    return np.stack([sample.vector[indices] for sample in samples])


def _oracle(
    predictions: np.ndarray,
    truth: np.ndarray,
    samples: list[PatchSample],
    block: int,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.mean((predictions - truth[:, None, :]) ** 2, axis=2)
    selected = np.empty(len(samples), dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.source_id, []).append(index)
    for indices in groups.values():
        if block == 1:
            selected[indices] = np.argmin(errors[indices], axis=1)
            continue
        side = int(round(math.sqrt(len(indices))))
        if side * side != len(indices):
            selected[indices] = np.argmin(errors[indices], axis=1)
            continue
        local = errors[indices].reshape(side, side, errors.shape[1])
        local_phases = np.asarray(
            [samples[index].phase for index in indices], dtype=np.int64
        ).reshape(side, side)
        local_selected = np.empty((side, side), dtype=np.int64)
        window = min(block, side)
        # Component labels are architecture-local to each independently
        # trained phase GMM.  Never smooth error for component k across a
        # neighboring pixel belonging to another phase, where k has unrelated
        # statistical meaning.  A block oracle may choose one expert per phase
        # population, but not pretend that all 18 banks share aligned labels.
        for phase in np.unique(local_phases):
            mask = local_phases == phase
            weights = uniform_filter(
                mask.astype(np.float64), size=window, mode="reflect"
            )
            numerator = uniform_filter(
                local * mask[:, :, None],
                size=(window, window, 1), mode="reflect",
            )
            averaged = numerator / np.maximum(weights[:, :, None], 1e-30)
            choices = np.argmin(averaged, axis=2)
            local_selected[mask] = choices[mask]
        selected[indices] = local_selected.ravel()
    return predictions[np.arange(len(samples)), selected], selected


def _rows(samples: list[PatchSample], outputs: dict[str, np.ndarray], truth: np.ndarray) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.source_id, []).append(index)
    return {
        source: {
            name: _metric(output[indices] - truth[indices])
            for name, output in outputs.items()
        }
        for source, indices in groups.items()
    }


def evaluate(
    model: PhaseConditionedGMR,
    samples: list[PatchSample],
    tau: float,
    temperature: float,
    *,
    joint: bool,
) -> tuple[dict[str, object], object]:
    batch = conditional_predict(
        prepare_gmr_cache(model, tau), samples,
        temperature=temperature, compute_joint_map=joint,
    )
    target = _truth(samples, model.patch_size)
    outputs = {"map": batch.map_rgb, "mmse": batch.mmse_rgb}
    oracle_components = None
    for block in (1, 3, 7, 15):
        oracle_output, selected = _oracle(batch.component_rgb, target, samples, block)
        outputs[f"oracle-{block}"] = oracle_output
        if block == 1:
            oracle_components = selected
    assert oracle_components is not None
    component_errors = np.mean((batch.component_rgb - target[:, None, :]) ** 2, axis=2)
    actual = component_errors.reshape(-1)
    predicted = batch.conditional_risks.reshape(-1)
    correlation = spearmanr(predicted, actual).statistic
    result = {
        "agreements": {
            "observed_map_vs_oracle": float(np.mean(batch.map_components == oracle_components)),
            "joint_map_vs_oracle": (
                float(np.mean(batch.joint_map_components == oracle_components)) if joint else None
            ),
            "observed_map_vs_joint_map": (
                float(np.mean(batch.map_components == batch.joint_map_components)) if joint else None
            ),
        },
        "conditional_risk": {
            "spearman_actual_component_error": (
                float(correlation) if math.isfinite(float(correlation)) else None
            ),
            "minimum": float(np.min(predicted)),
            "median": float(np.median(predicted)),
            "maximum": float(np.max(predicted)),
        },
        "methods": {name: _metric(output - target) for name, output in outputs.items()},
        "rows": _rows(samples, outputs, target),
        "tau": tau,
        "temperature": temperature,
    }
    return result, batch


def _logical_digest(model: PhaseConditionedGMR) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"gmr-v1:{model.patch_size}:{model.component_count}:{model.dc_mode}:"
        f"{model.covariance_floor:.17g}:{model.training_sample_count}:{model.seed}:"
        f"{model.shrinkage}".encode("ascii")
    )
    for phase in model.phases:
        for value in (phase.weights, phase.means, phase.covariances, phase.effective_counts):
            digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def save_model(path: Path, model: PhaseConditionedGMR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        patch_size=model.patch_size,
        component_count=model.component_count,
        dc_mode=model.dc_mode,
        covariance_floor=model.covariance_floor,
        training_sample_count=model.training_sample_count,
        seed=model.seed,
        shrinkage=model.shrinkage,
        weights=np.stack([phase.weights for phase in model.phases]),
        means=np.stack([phase.means for phase in model.phases]),
        covariances=np.stack([phase.covariances for phase in model.phases]),
        converged=np.asarray([phase.converged for phase in model.phases], dtype=np.uint8),
        iterations=np.asarray([phase.iterations for phase in model.phases]),
        lower_bounds=np.asarray([phase.lower_bound for phase in model.phases]),
        effective_counts=np.stack([phase.effective_counts for phase in model.phases]),
    )


def load_model(path: Path) -> PhaseConditionedGMR:
    with np.load(path, allow_pickle=False) as values:
        patch_size = int(values["patch_size"])
        component_count = int(values["component_count"])
        phases = []
        for phase, (origin_x, origin_y) in enumerate(origin_cells()):
            observed, sampled, targets = phase_contract(patch_size, phase)
            phases.append(PhaseMixture(
                origin_x=origin_x, origin_y=origin_y,
                observed_indices=observed, sampled_center_channel=sampled,
                target_channels=targets,
                weights=np.ascontiguousarray(values["weights"][phase]),
                means=np.ascontiguousarray(values["means"][phase]),
                covariances=np.ascontiguousarray(values["covariances"][phase]),
                converged=bool(values["converged"][phase]),
                iterations=int(values["iterations"][phase]),
                lower_bound=float(values["lower_bounds"][phase]),
                effective_counts=np.ascontiguousarray(values["effective_counts"][phase]),
            ))
        return PhaseConditionedGMR(
            patch_size=patch_size, component_count=component_count,
            dc_mode=str(values["dc_mode"]), covariance_floor=float(values["covariance_floor"]),
            training_sample_count=int(values["training_sample_count"]), seed=int(values["seed"]),
            phases=tuple(phases), shrinkage=str(values["shrinkage"]),
        )


def _model_path(output: Path, components: int, seed: int, iterations: int, sources: int) -> Path:
    return output / "models" / f"p7-k{components}-seed{seed}-i{iterations}-n{sources}.npz"


def fit_or_load(
    output: Path,
    vectors: np.ndarray,
    components: int,
    seed: int,
    iterations: int,
    sources: int = 200,
) -> PhaseConditionedGMR:
    path = _model_path(output, components, seed, iterations, sources)
    if path.exists():
        return load_model(path)
    print(f"train GMR K={components} seed={seed} iterations={iterations} sources={sources}", flush=True)
    model = fit_phase_conditioned_gmr(
        vectors[: sources * 512], PATCH_SIZE, components,
        dc_mode="observed-rgb", covariance_floor=COVARIANCE_FLOOR,
        seed=seed, maximum_iterations=iterations, tolerance=0.0,
    )
    save_model(path, model)
    return model


def refine_or_load(
    output: Path,
    vectors: np.ndarray,
    base: PhaseConditionedGMR,
    target_iterations: int,
    sources: int = 200,
) -> PhaseConditionedGMR:
    path = _model_path(
        output, base.component_count, base.seed, target_iterations, sources
    )
    if path.exists():
        return load_model(path)
    current = min(phase.iterations for phase in base.phases)
    additional = target_iterations - current
    if additional < 1:
        raise ValueError("refinement target must exceed the base checkpoint")
    print(
        f"continue GMR K={base.component_count} seed={base.seed} "
        f"iterations={current}->{target_iterations} sources={sources}",
        flush=True,
    )
    model = refine_phase_conditioned_gmr(
        np.asarray(vectors[: sources * 512]), base,
        additional_iterations=additional, tolerance=0.0,
    )
    save_model(path, model)
    return model


def _vectors(output: Path, splits: dict[str, list[dict[str, object]]]) -> np.ndarray:
    path = output / "training-vectors.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    print("decode and cache 102,400 BSDS training patches", flush=True)
    rows = training_samples(splits["train"], PATCH_SIZE)
    vectors = np.ascontiguousarray(np.stack([sample.vector for sample in rows]), dtype=np.float64)
    np.save(path, vectors)
    return vectors


def _select_hyperparameters(
    model: PhaseConditionedGMR,
    samples: list[PatchSample],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for tau in TAUS:
        for temperature in TEMPERATURES:
            result, _ = evaluate(model, samples, tau, temperature, joint=False)
            rows.append({
                "tau": tau, "temperature": temperature,
                "mmse": result["methods"]["mmse"],
                "map": result["methods"]["map"],
                "oracle_7": result["methods"]["oracle-7"],
            })
    selected = min(rows, key=lambda row: float(row["mmse"]["mse"]))
    return selected, rows


def _k1_parity(model: PhaseConditionedGMR, vectors: np.ndarray, tau: float) -> dict[str, object]:
    mean_difference = 0.0
    covariance_difference = 0.0
    for phase_index, phase in enumerate(model.phases):
        matrix = phase_training_matrix(vectors, PATCH_SIZE, phase_index, model.dc_mode)
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        covariance = centered.T @ centered / matrix.shape[0]
        covariance.flat[:: covariance.shape[0] + 1] += model.covariance_floor
        mean_difference = max(mean_difference, float(np.max(np.abs(mean - phase.means[0]))))
        covariance_difference = max(
            covariance_difference,
            float(np.max(np.abs(covariance - phase.covariances[0]))),
        )
    # Conditional parity is also covered independently by test_model.py.
    return {
        "covariance_maximum_abs_difference": covariance_difference,
        "mean_maximum_abs_difference": mean_difference,
        "pass": mean_difference <= 1e-12 and covariance_difference <= 1e-10,
        "tau": tau,
    }


def _load_previous(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != "rawtherapee-xtrans-joint-color-gmm-v1":
        raise RuntimeError("unexpected frozen FULL-GMM result format")
    return result


def _synthetic_samples(scene: np.ndarray) -> list[PatchSample]:
    size = scene.shape[1]
    center = size // 2
    radius = PATCH_SIZE // 2
    patch = np.ascontiguousarray(scene[:, center - radius:center + radius + 1, center - radius:center + radius + 1])
    rows = []
    for origin_x, origin_y in origin_cells():
        patch_x = (origin_x + center - radius) % 6
        patch_y = (origin_y + center - radius) % 6
        rows.append(PatchSample(
            source_id=f"phase-{origin_x}-{origin_y}", group="synthetic-phase",
            vector=patch.reshape(-1), indices=observation_indices(PATCH_SIZE, patch_x, patch_y),
            phase=phase_index(patch_x, patch_y), x=center - radius, y=center - radius,
        ))
    return rows


def synthetic_study(
    model: PhaseConditionedGMR,
    shrink: PhaseConditionedGMR,
    tau: float,
    temperature: float,
    previous: dict[str, object],
) -> dict[str, object]:
    size = 48
    y, x = np.mgrid[:size, :size]
    center = size // 2
    scenes = {
        "gray-gradient": np.stack([0.1 + 0.8 * x / (size - 1)] * 3),
        "chromatic-gradient": np.stack((
            0.1 + 0.8 * x / (size - 1), 0.15 + 0.7 * y / (size - 1),
            0.2 + 0.6 * (x + y) / (2 * size - 2),
        )),
        "red-gray-edge": np.stack((
            0.2 + 0.8 * (x >= center), 0.2 + 0.3 * (x >= center),
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
    prior_rows = {row["scene"]: row for row in previous["synthetic_phase"]["scenes"]}
    rows = []
    for name, scene in scenes.items():
        samples = _synthetic_samples(scene)
        ordinary, _ = evaluate(model, samples, tau, temperature, joint=True)
        stabilized, _ = evaluate(shrink, samples, tau, temperature, joint=False)
        row = prior_rows[name]
        rows.append({
            "scene": name,
            "gmax": row["methods"]["gmax"],
            "full_gmm": row["methods"]["gmm-mmse"],
            "gmr": ordinary["methods"]["mmse"],
            "shrink_gmr": stabilized["methods"]["mmse"],
            "phase_range_db": {
                "full_gmm": row["psnr_range_db"]["gmm-mmse"],
                "gmr": _phase_range(ordinary),
                "shrink_gmr": _phase_range(stabilized),
                "gmax": row["psnr_range_db"]["gmax"],
            },
        })
    return {"scenes": rows}


def _phase_range(result: dict[str, object]) -> float:
    values = [float(row["mmse"]["psnr_db"]) for row in result["rows"].values()]
    return max(values) - min(values)


def run(output: Path, bsds_root: Path, starfield: Path, previous_path: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    previous = _load_previous(previous_path)
    corpus, splits = corpus_manifest(bsds_root)
    vectors = _vectors(output, splits)
    validation = evaluation_samples(splits["val"], PATCH_SIZE, "bsds-validation", VALIDATION_GRID_SIDE)

    screen = []
    models: dict[tuple[int, int, int], PhaseConditionedGMR] = {}
    for components in COMPONENT_COUNTS:
        model = fit_or_load(output, vectors, components, SEEDS[0], SCREEN_ITERATIONS)
        models[(components, SEEDS[0], SCREEN_ITERATIONS)] = model
        selected, controls = _select_hyperparameters(model, validation)
        screen.append({
            "component_count": components,
            "model": {**model_summary(model), "logical_sha256": _logical_digest(model)},
            "selected": selected,
            "hyperparameter_screen": controls,
        })
        print(f"screen K={components}: {selected['mmse']['psnr_db']:.3f} dB", flush=True)

    k1_model = models[(1, SEEDS[0], SCREEN_ITERATIONS)]
    parity = _k1_parity(k1_model, np.asarray(vectors), float(screen[0]["selected"]["tau"]))
    selected_k = int(min(screen, key=lambda row: float(row["selected"]["mmse"]["mse"]))["component_count"])

    # The prescribed seed study is limited to K=16 and K=32.  The base-seed
    # screen rows are reused; only the two additional seeds are trained.
    seed_rows = []
    for components in (16, 32):
        for seed in SEEDS:
            model = models.get((components, seed, SCREEN_ITERATIONS))
            if model is None:
                model = fit_or_load(output, vectors, components, seed, SCREEN_ITERATIONS)
                models[(components, seed, SCREEN_ITERATIONS)] = model
            selected, _ = _select_hyperparameters(model, validation)
            seed_rows.append({
                "component_count": components, "seed": seed,
                "model": model_summary(model), "validation": selected,
            })

    eligible_seeds = [row for row in seed_rows if row["component_count"] == selected_k]
    if not eligible_seeds:
        eligible_seeds = [
            {"component_count": selected_k, "seed": SEEDS[0],
             "validation": next(row["selected"] for row in screen if row["component_count"] == selected_k)}
        ]
    selected_seed = int(min(eligible_seeds, key=lambda row: float(row["validation"]["mmse"]["mse"]))["seed"])

    checkpoint_rows = []
    checkpoint_models = {}
    for iterations in CHECKPOINTS:
        model = models.get((selected_k, selected_seed, iterations))
        if model is None:
            if iterations == 30:
                model = refine_or_load(
                    output, vectors, checkpoint_models[20], iterations
                )
            elif iterations == 50:
                model = refine_or_load(
                    output, vectors, checkpoint_models[30], iterations
                )
            else:
                model = fit_or_load(
                    output, vectors, selected_k, selected_seed, iterations
                )
        checkpoint_models[iterations] = model
        selected, _ = _select_hyperparameters(model, validation)
        checkpoint_rows.append({
            "iterations": iterations, "model": model_summary(model), "validation": selected,
        })
    selected_checkpoint = min(checkpoint_rows, key=lambda row: float(row["validation"]["mmse"]["mse"]))
    selected_iterations = int(selected_checkpoint["iterations"])
    selected_model = checkpoint_models[selected_iterations]
    selected_hyper = selected_checkpoint["validation"]
    tau = float(selected_hyper["tau"])
    temperature = float(selected_hyper["temperature"])

    # Bounded global-covariance backoff.  No external or test datum enters the
    # choice; all policies are ranked solely on the frozen validation grid.
    shrink_rows = []
    shrink_models: dict[str, PhaseConditionedGMR] = {"none": selected_model}
    for alpha in (0.05, 0.1, 0.2, 0.4):
        candidate = shrink_covariances(selected_model, k1_model, alpha=alpha)
        result, _ = evaluate(candidate, validation, tau, temperature, joint=False)
        shrink_models[candidate.shrinkage] = candidate
        shrink_rows.append({"policy": candidate.shrinkage, "validation": result["methods"]["mmse"]})
    for kappa in (64.0, 256.0, 1024.0):
        candidate = shrink_covariances(selected_model, k1_model, kappa=kappa)
        result, _ = evaluate(candidate, validation, tau, temperature, joint=False)
        shrink_models[candidate.shrinkage] = candidate
        shrink_rows.append({"policy": candidate.shrinkage, "validation": result["methods"]["mmse"]})
    selected_shrink_row = min(
        [{"policy": "none", "validation": selected_hyper["mmse"]}] + shrink_rows,
        key=lambda row: float(row["validation"]["mse"]),
    )
    shrink_model = shrink_models[str(selected_shrink_row["policy"])]

    validation_result, _ = evaluate(selected_model, validation, tau, temperature, joint=True)
    shrink_validation, _ = evaluate(shrink_model, validation, tau, temperature, joint=False)
    k1_validation, _ = evaluate(k1_model, validation, float(screen[0]["selected"]["tau"]), 1.0, joint=True)
    oracle1_headroom = (
        float(validation_result["methods"]["oracle-1"]["psnr_db"])
        - float(k1_validation["methods"]["mmse"]["psnr_db"])
    )
    oracle7_headroom = (
        float(validation_result["methods"]["oracle-7"]["psnr_db"])
        - float(k1_validation["methods"]["mmse"]["psnr_db"])
    )
    posterior_headroom = (
        float(validation_result["methods"]["mmse"]["psnr_db"])
        - float(k1_validation["methods"]["mmse"]["psnr_db"])
    )

    # Freeze the configuration here.  Everything below is post-selection.
    test = evaluation_samples(splits["test"], PATCH_SIZE, "bsds-test", TEST_GRID_SIDE)
    test_result, _ = evaluate(selected_model, test, tau, temperature, joint=True)
    shrink_test, _ = evaluate(shrink_model, test, tau, temperature, joint=False)
    k1_test, _ = evaluate(k1_model, test, float(screen[0]["selected"]["tau"]), 1.0, joint=False)

    # Training-source stability uses the already frozen architecture/seed/EM
    # length.  Test is diagnostic only and cannot alter selection.
    source_rows = []
    for sources in (25, 50, 100, 200):
        model = fit_or_load(output, vectors, selected_k, selected_seed, selected_iterations, sources)
        val_result, _ = evaluate(model, validation, tau, temperature, joint=False)
        tst_result, _ = evaluate(model, test, tau, temperature, joint=False)
        source_rows.append({
            "sources": sources, "model": model_summary(model),
            "validation": val_result["methods"]["mmse"],
            "test": tst_result["methods"]["mmse"],
        })

    external = external_samples(PATCH_SIZE)
    external_result, _ = evaluate(selected_model, external, tau, temperature, joint=True)
    external_shrink, _ = evaluate(shrink_model, external, tau, temperature, joint=False)
    stars = established_samples(starfield, PATCH_SIZE)
    star_results = {
        name: {
            "gmr": evaluate(selected_model, samples, tau, temperature, joint=True)[0],
            "shrink_gmr": evaluate(shrink_model, samples, tau, temperature, joint=False)[0],
        }
        for name, samples in stars.items()
    }
    synthetic = synthetic_study(selected_model, shrink_model, tau, temperature, previous)

    previous_baselines = previous["sampled_baselines"]
    gmax_test = float(previous_baselines["pooled"]["gmax"]["psnr_db"])
    full_test = float(previous["bsds_test"]["methods"]["cfa-mmse"]["psnr_db"])
    practical_test = float(shrink_test["methods"]["mmse"]["psnr_db"])
    external_deltas = []
    for source, row in external_shrink["rows"].items():
        gmax = float(previous["external_reference"][source]["methods"]["gmax"]["psnr_db"])
        external_deltas.append(float(row["mmse"]["psnr_db"]) - gmax)
    star_deltas = []
    for name in ("hubble-bright", "nasa-hydra-starfield-bright"):
        gmax = float(previous["established_reference"][name]["methods"]["gmax"]["psnr_db"])
        star_deltas.append(float(star_results[name]["shrink_gmr"]["methods"]["mmse"]["psnr_db"]) - gmax)
    synthetic_deltas = [
        float(row["shrink_gmr"]["psnr_db"]) - float(row["gmax"]["psnr_db"])
        for row in synthetic["scenes"]
    ]
    seed_spreads = {}
    for components in (16, 32):
        values = [float(row["validation"]["mmse"]["psnr_db"]) for row in seed_rows if row["component_count"] == components]
        seed_spreads[str(components)] = max(values) - min(values)
    checkpoint_psnr = [float(row["validation"]["mmse"]["psnr_db"]) for row in checkpoint_rows]
    em_reversal = max(checkpoint_psnr) - checkpoint_psnr[-1]

    gate_a = bool(parity["pass"])
    gate_b = oracle1_headroom >= 0.3 and posterior_headroom >= 0.3
    gate_c = practical_test - gmax_test >= 0.3
    gate_d = practical_test >= full_test - 0.2
    gate_e = min(external_deltas) >= -0.5 and min(star_deltas) >= -0.2 and min(synthetic_deltas) >= -2.0
    gate_f = max(seed_spreads.values()) <= 0.3 and em_reversal <= 0.3
    if all((gate_a, gate_b, gate_c, gate_d, gate_e, gate_f)):
        outcome = "GO - phase-conditioned GMR"
    elif gate_a and gate_b and gate_c and gate_d and gate_f:
        outcome = "PARTIAL - safety stabilization incomplete"
    elif gate_a and gate_b and not gate_c:
        outcome = "PARTIAL - generative selection"
    else:
        outcome = "NO-GO - phase-conditioned GMR"

    result = {
        "format": "rawtherapee-xtrans-phase-conditioned-gmr-v1",
        "corpus": {
            "parent_format": corpus["format"],
            "parent_dataset_sha256": hashlib.sha256(
                Path("devnotes/images/xtrans-gmm/dataset.json").read_bytes()
            ).hexdigest(),
            "training_patches_per_phase": int(vectors.shape[0]),
            "phase_count": 18,
            "split_discipline": corpus["split_discipline"],
        },
        "contract": {
            "patch": "7x7 RGB",
            "observed_dimension": 49,
            "target_dimension": 2,
            "joint_dimension": 51,
            "target_order": {"R": ["G", "B"], "G": ["R", "B"], "B": ["R", "G"]},
            "dc": "common mean of phase-specific physically observed R/G/B means",
            "native_center": "restored exactly after posterior averaging",
        },
        "previous_full_gmm": {
            "results_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            "frozen_configuration": {"patch_size": 7, "components": 16, "dc": "observed-rgb", "tau": 0.003, "temperature": 4.0},
        },
        "k1_parity": parity,
        "component_screen": screen,
        "seed_stability": {"rows": seed_rows, "validation_spread_db": seed_spreads},
        "em_checkpoints": {"rows": checkpoint_rows, "late_reversal_db": em_reversal},
        "shrinkage_screen": {"rows": shrink_rows, "selected": selected_shrink_row},
        "selected": {
            "component_count": selected_k, "seed": selected_seed,
            "iterations": selected_iterations, "tau": tau, "temperature": temperature,
            "model": {**model_summary(selected_model), "logical_sha256": _logical_digest(selected_model)},
            "shrinkage": shrink_model.shrinkage,
            "shrink_model_logical_sha256": _logical_digest(shrink_model),
        },
        "validation": {"k1": k1_validation, "gmr": validation_result, "shrink_gmr": shrink_validation},
        "bsds_test": {"k1": k1_test, "gmr": test_result, "shrink_gmr": shrink_test},
        "training_source_curve": source_rows,
        "external": {"gmr": external_result, "shrink_gmr": external_shrink},
        "stars": star_results,
        "synthetic": synthetic,
        "decision": {
            "outcome": outcome,
            "gate_a_k1_parity": gate_a,
            "gate_b_component_capacity": gate_b,
            "gate_c_nonlinear_value": gate_c,
            "gate_d_full_gmm_competitiveness": gate_d,
            "gate_e_external_star_synthetic_safety": gate_e,
            "gate_f_training_stability": gate_f,
            "gmr_minus_gmax_test_db": practical_test - gmax_test,
            "gmr_minus_full_gmm_test_db": practical_test - full_test,
            "validation_oracle1_minus_k1_db": oracle1_headroom,
            "validation_oracle7_minus_k1_db": oracle7_headroom,
            "validation_posterior_minus_k1_db": posterior_headroom,
            "worst_external_minus_gmax_db": min(external_deltas),
            "worst_bright_star_minus_gmax_db": min(star_deltas),
            "worst_synthetic_minus_gmax_db": min(synthetic_deltas),
            "maximum_seed_spread_db": max(seed_spreads.values()),
            "late_em_reversal_db": em_reversal,
        },
        "complexity": {
            "gmax_mac_per_pixel": 242,
            "gmr_parameter_count": model_summary(selected_model)["parameter_count"],
            "gmr_covariance_values": 18 * selected_k * 51 * 52 // 2,
            "gmr_covariance_float32_mib": 18 * selected_k * 51 * 52 // 2 * 4 / (1024 * 1024),
            "likelihood_dimension": 49,
            "component_count_per_pixel": selected_k,
            "estimated_dense_mac_per_pixel": selected_k * (49 * 49 + 49 + 2 * 49),
            "softmax_exponentials_per_pixel": selected_k,
        },
        "deferred": {"epll": True, "production_cpp": True, "discriminative_gating": True},
    }
    _write_json(output / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument("--starfield", type=Path, default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"))
    parser.add_argument("--previous", type=Path, default=Path("devnotes/images/xtrans-gmm/results.json"))
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.bsds_root, arguments.starfield, arguments.previous)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

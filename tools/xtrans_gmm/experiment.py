#!/usr/bin/env python3
"""Run the staged X-Trans joint-color GMM feasibility experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from scipy.ndimage import uniform_filter

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_lmmse.experiment import _run_baselines
from tools.xtrans_lmmse.experiment import _load_experiment_sources
from tools.xtrans_lmmse.model import predict_region
from tools.xtrans_lmmse_mixture.dataset import external_chromatic_sources
from tools.xtrans_lmmse_mixture.training import survey_and_train_globals
from tools.xtrans_sparse_dictionary.dataset import PatchSample
from tools.xtrans_sparse_dictionary.model import observation_indices, phase_index
from tools.xtrans_mlri_internal.dataset import mosaic, origin_cells

from .dataset import (
    TEST_GRID_SIDE,
    TRAINING_PER_SOURCE,
    VALIDATION_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
    training_samples,
)
from .model import (
    JointColorGMM,
    center_indices,
    conditional_predict,
    fit_joint_gmm,
    full_rgb_log_likelihood,
    model_summary,
    prepare_conditional_cache,
    refine_joint_gmm,
    sample_matrix,
)


SEED = 0x58474D4D
PRIMARY_PATCH_SIZE = 5
COMPONENT_COUNTS = (1, 4, 8, 16, 32, 64)
TAUS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
COVARIANCE_FLOOR = 1e-6
MINIMUM_EFFECTIVE_COUNT = 32.0
CONTROL_COMPONENT_COUNT = 16


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "infinity" if value > 0 else "-infinity"
        return value
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_safe(value)))


def _metric(errors: np.ndarray | list[np.ndarray]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
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
        "sse": float(np.sum(values * values)),
    }


def _truth_centers(samples: list[PatchSample], patch_size: int) -> np.ndarray:
    target = center_indices(patch_size)
    return np.stack([sample.vector[target] for sample in samples])


def _oracle_outputs(
    component_rgb: np.ndarray,
    truth: np.ndarray,
    samples: list[PatchSample],
    block: int,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.mean((component_rgb - truth[:, None, :]) ** 2, axis=2)
    selected = np.empty(errors.shape[0], dtype=np.int64)
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.source_id, []).append(index)
    for indices in grouped.values():
        if block == 1:
            selected[indices] = np.argmin(errors[indices], axis=1)
            continue
        side = int(round(math.sqrt(len(indices))))
        if side * side != len(indices):
            selected[indices] = np.argmin(errors[indices], axis=1)
            continue
        local = errors[indices].reshape(side, side, errors.shape[1])
        smoothed = uniform_filter(
            local, size=(min(block, side), min(block, side), 1), mode="reflect"
        )
        selected[indices] = np.argmin(smoothed, axis=2).ravel()
    return component_rgb[np.arange(errors.shape[0]), selected], selected


def _calibration(
    entropy: np.ndarray,
    responsibilities: np.ndarray,
    errors: np.ndarray,
    oracle_components: np.ndarray,
    map_components: np.ndarray,
) -> dict[str, object]:
    certainty = np.max(responsibilities, axis=1)
    result = {}
    for name, values, reverse in (
        ("maximum_responsibility", certainty, False),
        ("entropy", entropy, True),
    ):
        if float(np.max(values) - np.min(values)) <= 1e-15:
            result[name] = [{
                "actual_rgb_rms": float(np.sqrt(np.mean(errors ** 2))),
                "component_agreement_with_oracle": float(np.mean(
                    map_components == oracle_components
                )),
                "count": int(values.size),
                "high": float(values[0]),
                "low": float(values[0]),
            }]
            continue
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, 6))
        bins = []
        for index in range(5):
            selected = (values >= quantiles[index]) & (
                values <= quantiles[index + 1] if index == 4 else values < quantiles[index + 1]
            )
            bins.append({
                "actual_rgb_rms": float(np.sqrt(np.mean(errors[selected] ** 2))),
                "component_agreement_with_oracle": float(np.mean(
                    map_components[selected] == oracle_components[selected]
                )),
                "count": int(np.sum(selected)),
                "high": float(quantiles[index + 1]),
                "low": float(quantiles[index]),
            })
        if reverse:
            bins = list(reversed(bins))
        result[name] = bins
    return result


def _evaluate(
    model: JointColorGMM,
    samples: list[PatchSample],
    tau: float,
    temperature: float,
) -> tuple[dict[str, object], object]:
    started = time.perf_counter()
    batch = conditional_predict(
        prepare_conditional_cache(model, tau), samples, temperature=temperature
    )
    elapsed = time.perf_counter() - started
    truth = _truth_centers(samples, model.patch_size)
    rgb_map_components = np.argmax(full_rgb_log_likelihood(model, samples), axis=1)
    rgb_map = batch.component_rgb[np.arange(len(samples)), rgb_map_components]
    outputs = {
        "cfa-map": batch.map_rgb,
        "cfa-mmse": batch.mmse_rgb,
        "rgb-map": rgb_map,
    }
    oracle_components = {}
    for block in (1, 3, 7, 15):
        output, selected = _oracle_outputs(
            batch.component_rgb, truth, samples, block
        )
        outputs[f"oracle-{block}"] = output
        oracle_components[str(block)] = selected
    rows = {}
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.source_id, []).append(index)
    for source_id, indices in sorted(grouped.items()):
        rows[source_id] = {
            name: _metric(output[indices] - truth[indices])
            for name, output in outputs.items()
        }
    methods = {name: _metric(output - truth) for name, output in outputs.items()}
    oracle1 = oracle_components["1"]
    map_errors = batch.mmse_rgb - truth
    result = {
        "calibration": _calibration(
            batch.entropy, batch.responsibilities, map_errors,
            oracle1, batch.map_components,
        ),
        "component_selection": {
            "cfa_map_agreement_with_oracle1": float(np.mean(batch.map_components == oracle1)),
            "cfa_map_agreement_with_rgb_map": float(np.mean(batch.map_components == rgb_map_components)),
            "effective_component_count_mean": float(np.mean(np.exp(batch.entropy))),
            "entropy_mean": float(np.mean(batch.entropy)),
            "maximum_responsibility_mean": float(np.mean(np.max(batch.responsibilities, axis=1))),
            "rgb_map_agreement_with_oracle1": float(np.mean(rgb_map_components == oracle1)),
        },
        "mean_inference_seconds_per_patch": elapsed / len(samples),
        "methods": methods,
        "rows": rows,
        "sample_count": len(samples),
        "tau": tau,
        "temperature": temperature,
    }
    return result, batch


def _k1_parity(model: JointColorGMM, samples: list[PatchSample], tau: float) -> dict[str, object]:
    if model.component_count != 1:
        raise ValueError("K=1 model required")
    matrix = sample_matrix(samples, model.dc_mode)
    direct_mean = np.mean(matrix, axis=0)
    centered = matrix - direct_mean
    direct_covariance = centered.T @ centered / matrix.shape[0]
    direct_covariance.flat[:: direct_covariance.shape[0] + 1] += model.covariance_floor
    return {
        "covariance_maximum_abs_difference": float(
            np.max(np.abs(direct_covariance - model.covariances[0]))
        ),
        "mean_maximum_abs_difference": float(np.max(np.abs(direct_mean - model.means[0]))),
        "weight": float(model.weights[0]),
        "tau": tau,
    }


def _save_model(path: Path, model: JointColorGMM) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        patch_size=model.patch_size,
        component_count=model.component_count,
        dc_mode=model.dc_mode,
        covariance_floor=model.covariance_floor,
        weights=model.weights,
        means=model.means,
        covariances=model.covariances,
        converged=int(model.converged),
        iterations=model.iterations,
        lower_bound=model.lower_bound,
        effective_counts=model.effective_counts,
        training_sample_count=model.training_sample_count,
        seed=model.seed,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_model(path: Path) -> JointColorGMM:
    with np.load(path, allow_pickle=False) as values:
        return JointColorGMM(
            patch_size=int(values["patch_size"]),
            component_count=int(values["component_count"]),
            dc_mode=str(values["dc_mode"]),
            covariance_floor=float(values["covariance_floor"]),
            weights=np.ascontiguousarray(values["weights"]),
            means=np.ascontiguousarray(values["means"]),
            covariances=np.ascontiguousarray(values["covariances"]),
            converged=bool(values["converged"]),
            iterations=int(values["iterations"]),
            lower_bound=float(values["lower_bound"]),
            effective_counts=np.ascontiguousarray(values["effective_counts"]),
            training_sample_count=int(values["training_sample_count"]),
            seed=int(values["seed"]),
        )


def _train_or_load(
    output: Path,
    splits: dict[str, list[dict[str, object]]],
    patch_size: int,
    components: int,
    dc_mode: str,
) -> tuple[JointColorGMM, dict[str, object]]:
    model_path = output / "models" / f"p{patch_size}-k{components}-{dc_mode}.npz"
    started = time.perf_counter()
    if model_path.exists():
        model = _load_model(model_path)
        source = "cache"
    else:
        samples = training_samples(splits["train"], patch_size)
        matrix = sample_matrix(samples, dc_mode)
        model = fit_joint_gmm(
            matrix,
            patch_size,
            components,
            dc_mode=dc_mode,
            covariance_floor=COVARIANCE_FLOOR,
            seed=SEED + patch_size * 1000 + components * 10 + len(dc_mode),
            maximum_iterations=30,
            tolerance=1e-3,
        )
        _save_model(model_path, model)
        source = "trained"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = model_summary(model)
    summary.update({
        "artifact_sha256": digest,
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_occupancy_pass": float(np.min(model.effective_counts)) >= MINIMUM_EFFECTIVE_COUNT,
        "source": source,
    })
    return model, summary


def _configuration_screen(
    output: Path,
    splits: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[tuple[int, int, str], JointColorGMM]]:
    models = {}
    rows = []
    validation_cache: dict[int, list[PatchSample]] = {}

    def evaluate_configuration(patch_size: int, components: int, dc_mode: str) -> None:
        key = (patch_size, components, dc_mode)
        if key in models:
            return
        print(f"fit patch={patch_size} K={components} dc={dc_mode}", flush=True)
        model, training = _train_or_load(
            output, splits, patch_size, components, dc_mode
        )
        models[key] = model
        validation = validation_cache.setdefault(
            patch_size,
            evaluation_samples(
                splits["val"], patch_size, "bsds-validation", VALIDATION_GRID_SIDE
            ),
        )
        candidates = []
        for tau in TAUS:
            result, _ = _evaluate(model, validation, tau, 1.0)
            candidates.append(result)
        selected_tau = min(
            candidates,
            key=lambda row: float(row["methods"]["cfa-mmse"]["mse"]),
        )["tau"]
        selected_temperature_rows = []
        for temperature in TEMPERATURES:
            result, _ = _evaluate(model, validation, float(selected_tau), temperature)
            selected_temperature_rows.append(result)
        selected = min(
            selected_temperature_rows,
            key=lambda row: float(row["methods"]["cfa-mmse"]["mse"]),
        )
        rows.append({
            "dc_mode": dc_mode,
            "model": training,
            "patch_size": patch_size,
            "selected": selected,
            "tau_screen": [
                {"tau": row["tau"], "methods": row["methods"]}
                for row in candidates
            ],
            "temperature_screen": [
                {"temperature": row["temperature"], "methods": row["methods"]}
                for row in selected_temperature_rows
            ],
        })
        print(
            f"screen patch={patch_size} K={components} dc={dc_mode} "
            f"mmse={selected['methods']['cfa-mmse']['psnr_db']:.3f} "
            f"oracle7={selected['methods']['oracle-7']['psnr_db']:.3f}",
            flush=True,
        )

    for components in COMPONENT_COUNTS:
        evaluate_configuration(PRIMARY_PATCH_SIZE, components, "observed-rgb")
    selected_primary = max(
        (row for row in rows if row["patch_size"] == PRIMARY_PATCH_SIZE),
        key=lambda row: float(row["selected"]["methods"]["cfa-mmse"]["psnr_db"]),
    )
    # Keep the support/DC controls bounded and comparable.  Applying the
    # primary K=64 winner to 7x7 full covariance would multiply the expensive
    # control by roughly four without isolating support from model capacity.
    selected_components = CONTROL_COMPONENT_COUNT
    for patch_size in (3, 7):
        evaluate_configuration(patch_size, selected_components, "observed-rgb")
    selected_control_support = min(
        (
            row for row in rows
            if int(row["model"]["component_count"]) == selected_components
            and row["dc_mode"] == "observed-rgb"
        ),
        key=lambda row: float(row["selected"]["methods"]["cfa-mmse"]["mse"]),
    )["patch_size"]
    for dc_mode in ("absolute", "observed-scalar"):
        evaluate_configuration(int(selected_control_support), selected_components, dc_mode)
    return rows, models


def _sampled_baselines(
    splits: dict[str, list[dict[str, object]]],
    samples: list[PatchSample],
    runner: Path,
    cache_path: Path | None = None,
) -> tuple[dict[str, object], object]:
    globals_by_size, _, _ = survey_and_train_globals(splits["train"])
    gmax = globals_by_size[200]
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), gmax
    grouped: dict[str, list[PatchSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_id, []).append(sample)
    pooled = {name: [] for name in ("markesteijn", "corrected-final", "gmax")}
    rows = {}
    for source in splits["test"]:
        source_id = str(source["filename"])
        if source_id not in grouped:
            continue
        from tools.xtrans_lmmse.dataset import load_rgb
        truth = load_rgb(source)
        region = (0, 0, truth.shape[2], truth.shape[1])
        gmax_output = predict_region(gmax, truth, region, clip=False)
        with tempfile.TemporaryDirectory(prefix="xtrans-gmm-baseline-") as directory:
            baselines, timings = _run_baselines(runner, truth, 0, 0, Path(directory))
        outputs = {
            "markesteijn": baselines["markesteijn"],
            "corrected-final": baselines["corrected-final"],
            "gmax": gmax_output,
        }
        local = {name: [] for name in outputs}
        for sample in grouped[source_id]:
            center_x = sample.x + int(round(math.sqrt(sample.vector.size / 3))) // 2
            center_y = sample.y + int(round(math.sqrt(sample.vector.size / 3))) // 2
            target = truth[:, center_y, center_x]
            for name, output in outputs.items():
                error = output[:, center_y, center_x] - target
                local[name].append(error)
                pooled[name].append(error)
        rows[source_id] = {
            "baseline_seconds": timings,
            "methods": {name: _metric(values) for name, values in local.items()},
        }
        print(f"baseline {source_id}", flush=True)
    result = {
        "pooled": {name: _metric(values) for name, values in pooled.items()},
        "rows": rows,
    }
    if cache_path is not None:
        _write_json(cache_path, result)
    return result, gmax


def _external_contexts() -> dict[str, np.ndarray]:
    result = {}
    for source in external_chromatic_sources():
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        for crop_index, (x, y) in enumerate(source["crops"]):
            result[f"{source['id']}-{crop_index}"] = np.ascontiguousarray(
                rgb[:, int(y):int(y) + 168, int(x):int(x) + 168]
            )
    return result


def _established_contexts(starfield: Path) -> dict[str, np.ndarray]:
    return {
        str(source["id"]): np.asarray(source["rgb"], dtype=np.float64)
        for source in _load_experiment_sources(starfield)
        if str(source["id"]) in ("hubble", "nasa-hydra-starfield")
    }


def _run_selected_baselines(
    runner: Path,
    truth: np.ndarray,
    origin_x: int,
    origin_y: int,
    directory: Path,
    methods: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run only the native references required by a focused experiment."""

    scalar, _ = mosaic(truth, origin_x, origin_y)
    input_path = directory / "input.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            str(runner), "run-selected", str(input_path), str(directory),
            str(truth.shape[2]), str(truth.shape[1]), str(origin_x), str(origin_y),
            ",".join(methods),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"selected baseline runner failed:\n{completed.stdout}\n{completed.stderr}"
        )
    outputs = {}
    for method in methods:
        values = np.fromfile(directory / f"{method}.f32le", dtype="<f4")
        if values.size != truth.size or not np.isfinite(values).all():
            raise RuntimeError(f"invalid selected baseline output: {method}")
        outputs[method] = values.reshape(truth.shape).astype(np.float64)
    timings = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] in methods:
            timings[fields[0]] = float(fields[1])
    return outputs, timings


def _reference_for_groups(
    gmax,
    groups: dict[str, list[PatchSample]],
    contexts: dict[str, np.ndarray],
    *,
    runner: Path | None = None,
) -> dict[str, object]:
    def context_for(group_name: str, sample: PatchSample) -> str:
        # Bright-sample helpers vary between retaining the base id and adding a
        # suffix.  Resolve both representations without changing coordinates.
        source_id = sample.source_id.removesuffix("-bright")
        if source_id in contexts:
            return source_id
        return group_name.removesuffix("-bright")

    bounds: dict[str, list[int]] = {}
    for group_name, samples in groups.items():
        for sample in samples:
            context_id = context_for(group_name, sample)
            patch_size = int(round(math.sqrt(sample.vector.size / 3.0)))
            center_x = sample.x + patch_size // 2
            center_y = sample.y + patch_size // 2
            if context_id not in bounds:
                bounds[context_id] = [center_x, center_y, center_x + 1, center_y + 1]
            else:
                row = bounds[context_id]
                row[0] = min(row[0], center_x)
                row[1] = min(row[1], center_y)
                row[2] = max(row[2], center_x + 1)
                row[3] = max(row[3], center_y + 1)

    by_context = {}
    for context_id, truth in contexts.items():
        if context_id not in bounds:
            continue
        region = tuple(bounds[context_id])
        outputs = {"gmax": (predict_region(gmax, truth, region, clip=False), region)}
        timings = {}
        if runner is not None:
            with tempfile.TemporaryDirectory(prefix="xtrans-gmm-control-") as directory:
                native, timings = _run_selected_baselines(
                    runner, truth, 0, 0, Path(directory),
                    ("markesteijn", "corrected-final"),
                )
            outputs.update({
                "markesteijn": (native["markesteijn"], (0, 0, truth.shape[2], truth.shape[1])),
                "corrected-final": (native["corrected-final"], (0, 0, truth.shape[2], truth.shape[1])),
            })
        by_context[context_id] = (truth, outputs, timings)
    result = {}
    for group_name, samples in groups.items():
        errors: dict[str, list[np.ndarray]] = {}
        timing_rows = {}
        for sample in samples:
            context_id = context_for(group_name, sample)
            truth, outputs, timings = by_context[context_id]
            patch_size = int(round(math.sqrt(sample.vector.size / 3.0)))
            center_x = sample.x + patch_size // 2
            center_y = sample.y + patch_size // 2
            target = truth[:, center_y, center_x]
            for name, (output, region) in outputs.items():
                errors.setdefault(name, []).append(
                    output[:, center_y - region[1], center_x - region[0]] - target
                )
            timing_rows[context_id] = timings
        result[group_name] = {
            "methods": {name: _metric(values) for name, values in errors.items()},
            "native_timings": timing_rows,
        }
    return result


def _synthetic_phase_study(model: JointColorGMM, gmax, tau: float, temperature: float) -> dict[str, object]:
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
        "tiny-star": np.full((3, size, size), 0.03, dtype=np.float64),
        "saturated-point": np.full((3, size, size), 0.1, dtype=np.float64),
    }
    scenes["tiny-star"][:, center, center] = (1.0, 0.55, 0.15)
    scenes["saturated-point"][:, center, center] = (1.0, 0.0, 0.9)
    rows = []
    radius = model.patch_size // 2
    for name, truth in scenes.items():
        samples = []
        gmax_predictions = []
        target = truth[:, center, center]
        patch = truth[:, center - radius:center + radius + 1, center - radius:center + radius + 1]
        for origin_x, origin_y in origin_cells():
            patch_origin_x = (origin_x + center - radius) % 6
            patch_origin_y = (origin_y + center - radius) % 6
            indices = observation_indices(model.patch_size, patch_origin_x, patch_origin_y)
            samples.append(PatchSample(
                source_id=f"{name}-{origin_x}-{origin_y}",
                group="synthetic-phase",
                vector=np.ascontiguousarray(patch).reshape(-1),
                indices=indices,
                phase=phase_index(patch_origin_x, patch_origin_y),
                x=center - radius,
                y=center - radius,
            ))
            output = predict_region(
                gmax, truth, (center, center, center + 1, center + 1),
                origin_x=origin_x, origin_y=origin_y,
            )
            gmax_predictions.append(output[:, 0, 0])
        gmm = conditional_predict(
            prepare_conditional_cache(model, tau), samples, temperature=temperature
        )
        gmax_values = np.stack(gmax_predictions)
        methods = {
            "gmax": _metric(gmax_values - target),
            "gmm-map": _metric(gmm.map_rgb - target),
            "gmm-mmse": _metric(gmm.mmse_rgb - target),
        }
        phase_psnr = {
            method: [
                float(-10.0 * math.log10(max(float(np.mean((value - target) ** 2)), 1e-30)))
                for value in values
            ]
            for method, values in (
                ("gmax", gmax_values), ("gmm-map", gmm.map_rgb), ("gmm-mmse", gmm.mmse_rgb)
            )
        }
        rows.append({
            "methods": methods,
            "posterior_entropy_range": [float(np.min(gmm.entropy)), float(np.max(gmm.entropy))],
            "psnr_range_db": {
                method: max(values) - min(values) for method, values in phase_psnr.items()
            },
            "scene": name,
        })
    return {"scenes": rows}


def _refine_selection(
    output: Path,
    splits: dict[str, list[dict[str, object]]],
    selected: dict[str, object],
    model: JointColorGMM,
) -> tuple[JointColorGMM, dict[str, object], dict[str, object]]:
    """Converge only the validation winner and repeat its hyperparameter screen."""

    key = (model.patch_size, model.component_count, model.dc_mode)
    refined_path = output / "models" / (
        f"p{key[0]}-k{key[1]}-{key[2]}-refined.npz"
    )
    started = time.perf_counter()
    if refined_path.exists():
        refined = _load_model(refined_path)
        source = "refined-cache"
    elif model.converged:
        refined = model
        _save_model(refined_path, refined)
        source = "screen-model-already-converged"
    else:
        samples = training_samples(splits["train"], model.patch_size)
        matrix = sample_matrix(samples, model.dc_mode)
        refined = refine_joint_gmm(
            matrix, model, additional_iterations=70, tolerance=1e-4
        )
        _save_model(refined_path, refined)
        source = "continued-em"
    summary = model_summary(refined)
    summary.update({
        "artifact_sha256": hashlib.sha256(refined_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_occupancy_pass": float(np.min(refined.effective_counts)) >= MINIMUM_EFFECTIVE_COUNT,
        "screen_lower_bound": float(model.lower_bound),
        "source": source,
    })
    validation = evaluation_samples(
        splits["val"], refined.patch_size, "bsds-validation", VALIDATION_GRID_SIDE
    )
    tau_rows = [_evaluate(refined, validation, tau, 1.0)[0] for tau in TAUS]
    tau = float(min(
        tau_rows, key=lambda row: float(row["methods"]["cfa-mmse"]["mse"])
    )["tau"])
    temperature_rows = [
        _evaluate(refined, validation, tau, temperature)[0]
        for temperature in TEMPERATURES
    ]
    validation_result = min(
        temperature_rows,
        key=lambda row: float(row["methods"]["cfa-mmse"]["mse"]),
    )
    return refined, summary, validation_result


def run(
    output: Path,
    bsds_root: Path,
    starfield: Path,
    runner: Path,
) -> dict[str, object]:
    started = time.monotonic()
    corpus, splits = corpus_manifest(bsds_root)
    _write_json(output / "dataset.json", corpus)
    configurations, models = _configuration_screen(output, splits)
    primary_k1 = next(
        row for row in configurations
        if row["patch_size"] == PRIMARY_PATCH_SIZE
        and int(row["model"]["component_count"]) == 1
        and row["dc_mode"] == "observed-rgb"
    )
    k1_model = models[(PRIMARY_PATCH_SIZE, 1, "observed-rgb")]
    train_for_parity = training_samples(splits["train"], PRIMARY_PATCH_SIZE)
    parity = _k1_parity(k1_model, train_for_parity, float(primary_k1["selected"]["tau"]))
    gate_a = (
        parity["mean_maximum_abs_difference"] <= 1e-12
        and parity["covariance_maximum_abs_difference"] <= 1e-10
    )
    selected = min(
        configurations,
        key=lambda row: float(row["selected"]["methods"]["cfa-mmse"]["mse"]),
    )
    selected_key = (
        int(selected["patch_size"]),
        int(selected["model"]["component_count"]),
        str(selected["dc_mode"]),
    )
    screen_model = models[selected_key]
    screen_validation = selected["selected"]
    refined_model, refined_summary, refined_validation = _refine_selection(
        output, splits, selected, screen_model
    )
    use_refined = float(refined_validation["methods"]["cfa-mmse"]["mse"]) < float(
        screen_validation["methods"]["cfa-mmse"]["mse"]
    )
    selected_model = refined_model if use_refined else screen_model
    selected_validation = refined_validation if use_refined else screen_validation
    selected_summary = refined_summary if use_refined else selected["model"]
    k1_psnr = float(primary_k1["selected"]["methods"]["cfa-mmse"]["psnr_db"])
    oracle7_psnr = float(selected_validation["methods"]["oracle-7"]["psnr_db"])
    gate_b = oracle7_psnr - k1_psnr >= 0.3
    posterior_psnr = float(selected_validation["methods"]["cfa-mmse"]["psnr_db"])
    k1_mse = float(primary_k1["selected"]["methods"]["cfa-mmse"]["mse"])
    oracle_mse = float(selected_validation["methods"]["oracle-7"]["mse"])
    posterior_mse = float(selected_validation["methods"]["cfa-mmse"]["mse"])
    recovered = (
        (k1_mse - posterior_mse) / (k1_mse - oracle_mse)
        if k1_mse > oracle_mse else 0.0
    )
    gate_c = posterior_psnr >= k1_psnr and recovered >= 0.30

    result: dict[str, object] = {
        "configuration_screen": configurations,
        "corpus": corpus,
        "decision": {
            "gate_a_k1_parity": gate_a,
            "gate_b_mixture_oracle": gate_b,
            "gate_c_posterior_usefulness": gate_c,
            "selected_em_converged": selected_model.converged,
            "k1_psnr_db": k1_psnr,
            "oracle7_headroom_db": oracle7_psnr - k1_psnr,
            "posterior_headroom_recovered_fraction": recovered,
            "rules": {
                "mixture_oracle_headroom_db": 0.3,
                "posterior_headroom_recovery_fraction": 0.30,
            },
        },
        "format": "rawtherapee-xtrans-joint-color-gmm-v1",
        "em_refinement": {
            "refined_model": refined_summary,
            "refined_validation": refined_validation,
            "screen_model": selected["model"],
            "screen_validation": screen_validation,
            "selected": "refined" if use_refined else "validation-early-stopped-screen",
            "selection_rule": "lower validation conditional-MMSE MSE; likelihood alone does not select a demosaicer",
        },
        "k1_parity": parity,
        "references": {
            "jcs_gmm": {
                "doi": "10.1109/LSP.2018.2886466",
                "local_pdf_sha256": "4b83b2e0d1c6bc2e529c4f2c2ef065b44e2f7347f83c0163563287c4a1128b6b",
                "paper_contract": "hard observed-data MAP component, conditional Gaussian missing-pixel estimate, stride-1 overlap, K=150, 6x6, one million patches, five/ten adaptation iterations",
                "source_code": "no author/reference code found; IEEE states supplementary material exists but it was not available with the supplied PDF",
            },
            "epll": {
                "doi": "10.1109/ICCV.2011.6126278",
                "role": "whole-image GMM patch-prior framework; deliberately not implemented before local gates",
            },
        },
        "selected_configuration": {
            "dc_mode": selected_key[2],
            "model": selected_summary,
            "patch_size": selected_key[0],
            "component_count": selected_key[1],
            "tau": selected_validation["tau"],
            "temperature": selected_validation["temperature"],
            "validation": selected_validation,
        },
    }
    if not gate_a:
        result["decision"].update({"outcome": "NO-GO - K=1 parity", "stopped_after": "Gate A"})
    elif not gate_b:
        result["decision"].update({"outcome": "NO-GO - mixture capacity", "stopped_after": "Gate B"})
    elif not gate_c:
        result["decision"].update({"outcome": "PARTIAL - generative selection", "stopped_after": "Gate C"})
    else:
        test_samples = evaluation_samples(
            splits["test"], selected_key[0], "bsds-test", TEST_GRID_SIDE
        )
        test_result, _ = _evaluate(
            selected_model, test_samples, float(selected_validation["tau"]),
            float(selected_validation["temperature"]),
        )
        baselines, gmax = _sampled_baselines(
            splits, test_samples, runner, output / "sampled-baselines.json"
        )
        result["bsds_test"] = test_result
        result["sampled_baselines"] = baselines
        gmax_psnr = float(baselines["pooled"]["gmax"]["psnr_db"])
        gmm_psnr = float(test_result["methods"]["cfa-mmse"]["psnr_db"])
        gate_d = gmm_psnr - gmax_psnr >= 0.2
        result["decision"]["gate_d_held_out_population"] = gate_d
        result["decision"]["gmm_minus_gmax_db"] = gmm_psnr - gmax_psnr
        if not gate_d:
            result["decision"].update({"outcome": "NO-GO - population value", "stopped_after": "Gate D"})
        else:
            external_rows = external_samples(selected_key[0])
            external_evaluation = _evaluate(
                selected_model, external_rows,
                float(selected_validation["tau"]), float(selected_validation["temperature"]),
            )[0]
            external_groups: dict[str, list[PatchSample]] = {}
            for sample in external_rows:
                external_groups.setdefault(sample.source_id, []).append(sample)
            external_reference = _reference_for_groups(
                gmax, external_groups, _external_contexts()
            )
            established_rows = established_samples(starfield, selected_key[0])
            established_evaluation = {
                name: _evaluate(
                    selected_model, samples, float(selected_validation["tau"]),
                    float(selected_validation["temperature"]),
                )[0]
                for name, samples in established_rows.items()
            }
            established_reference = _reference_for_groups(
                gmax, established_rows, _established_contexts(starfield), runner=runner
            )
            synthetic = _synthetic_phase_study(
                selected_model, gmax, float(selected_validation["tau"]),
                float(selected_validation["temperature"]),
            )
            result["external_chromatic"] = external_evaluation
            result["external_reference"] = external_reference
            result["established"] = established_evaluation
            result["established_reference"] = established_reference
            result["synthetic_phase"] = synthetic
            external_deltas = [
                float(external_evaluation["rows"][source_id]["cfa-mmse"]["psnr_db"])
                - float(reference["methods"]["gmax"]["psnr_db"])
                for source_id, reference in external_reference.items()
            ]
            star_deltas = [
                float(established_evaluation[name]["methods"]["cfa-mmse"]["psnr_db"])
                - float(established_reference[name]["methods"]["gmax"]["psnr_db"])
                for name in ("hubble-bright", "nasa-hydra-starfield-bright")
            ]
            synthetic_deltas = [
                float(row["methods"]["gmm-mmse"]["psnr_db"])
                - float(row["methods"]["gmax"]["psnr_db"])
                for row in synthetic["scenes"]
            ]
            gate_e = (
                min(external_deltas) >= -0.5
                and min(star_deltas) >= -0.2
                and min(synthetic_deltas) >= -2.0
            )
            result["decision"].update({
                "external_worst_gmm_minus_gmax_db": min(external_deltas),
                "gate_e_external_sparse_synthetic_safety": gate_e,
                "sparse_bright_worst_gmm_minus_gmax_db": min(star_deltas),
                "synthetic_worst_gmm_minus_gmax_db": min(synthetic_deltas),
            })
            if gate_e:
                result["decision"].update({
                    "outcome": "GO - local joint-color GMM; EPLL evaluation justified",
                    "stopped_after": "none",
                })
            else:
                result["decision"].update({
                    "outcome": "PARTIAL - population prior is strong but sparse/synthetic safety failed",
                    "stopped_after": "Gate E",
                })
    result["elapsed_seconds"] = time.monotonic() - started
    result["epll"] = {
        "implemented": False,
        "reason": (
            "local GMM must pass Gates A-E before EPLL"
            if result["decision"].get("stopped_after") != "none"
            else "local gates pass; EPLL remains a separate subsequent experiment"
        ),
    }
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
        arguments.output,
        arguments.bsds_root,
        arguments.starfield,
        arguments.runner.resolve(),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

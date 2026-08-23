#!/usr/bin/env python3
"""Run the MLRI green-candidate selection and bounded-fusion experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Sequence

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_hybrid.analysis import fit_logistic, fit_tree
from tools.xtrans_hybrid.generate import _json_safe
from tools.xtrans_mlri_internal.analysis import MARGIN, roc_auc, scalar_metrics
from tools.xtrans_mlri_internal.dataset import mosaic
from tools.xtrans_mlri_internal.generate import _run_case, _runtime, srgb_encode
from tools.xtrans_oracle.analysis import opponent
from .analysis import (
    DIRECTION_NAMES,
    FEATURE_NAMES,
    CandidateTable,
    RidgeScore,
    build_candidate_table,
    candidate_bank,
    deterministic_indices,
    feature_maps,
    fit_ridge_score,
    green_metrics,
    interior_missing_green,
    label_coherence,
    patch_blend_alpha,
    patch_labels,
    preserve_measured_green,
    scalar_convex_hull,
    score_calibration,
    select_candidates,
    source_balanced_weights,
    temperature_fusion,
)
from .dataset import dataset_manifest, load_cases


PATCH_WINDOWS = (1, 3, 5, 7, 11, 15)
BANKS = {
    "pass0": tuple(range(8)),
    "pass1": tuple(range(8, 16)),
    "combined": tuple(range(16)),
}
PAIRWISE_PAIRS = tuple(
    [(direction, direction + 8) for direction in range(8)]
    + [
        (pass_index * 8 + first, pass_index * 8 + second)
        for pass_index in (0, 1)
        for first, second in (
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
        )
    ]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(_json_safe(value)))


def _tree_scores(model, features: np.ndarray) -> np.ndarray:
    shape = features.shape[:-1]
    flat = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    selected = flat[:, model.feature_indices]
    return model.root.predict(selected).reshape(shape)


def _pairwise_scores(model, features: np.ndarray) -> np.ndarray:
    result = np.zeros(features.shape[:-1], dtype=np.float64)
    for first, second in PAIRWISE_PAIRS:
        difference = features[first] - features[second]
        probability_first_worse = model.alpha(difference.reshape(-1, difference.shape[-1])).reshape(difference.shape[:-1])
        result[first] += probability_first_worse
        result[second] += 1.0 - probability_first_worse
    return result


def _fit_models(table: CandidateTable) -> dict[str, object]:
    train = table.select(table.splits == "train")
    validation = table.select(table.splits == "validation")
    train_error = (train.candidates - train.truth[:, None]) ** 2
    train_target = np.log10(train_error + 1e-8)
    train_relative_target = train_target - np.mean(train_target, axis=1, keepdims=True)
    flat_features = train.features.reshape(-1, train.features.shape[-1])
    flat_target = train_target.ravel()
    pixel_weights = source_balanced_weights(train.source_ids)
    flat_weights = np.repeat(pixel_weights, 16)
    selected = deterministic_indices(flat_features.shape[0], 250000)

    ridge_rows = []
    best_ridge = None
    best_mse = math.inf
    for regularization in (1e-3, 1e-2, 1e-1, 1.0):
        model = fit_ridge_score(
            flat_features[selected], flat_target[selected], flat_weights[selected], regularization,
        )
        scores = model.score(validation.features)
        row = score_calibration(validation, scores)
        row["regularization"] = regularization
        ridge_rows.append(row)
        if row["mean_squared_error"] < best_mse:
            best_mse = row["mean_squared_error"]
            best_ridge = model
    assert best_ridge is not None

    relative_rows = []
    best_relative_ridge = None
    best_relative_mse = math.inf
    flat_relative_target = train_relative_target.ravel()
    for regularization in (1e-3, 1e-2, 1e-1, 1.0):
        model = fit_ridge_score(
            flat_features[selected], flat_relative_target[selected],
            flat_weights[selected], regularization,
        )
        scores = model.score(validation.features)
        row = score_calibration(validation, scores)
        row["regularization"] = regularization
        relative_rows.append(row)
        if row["mean_squared_error"] < best_relative_mse:
            best_relative_mse = row["mean_squared_error"]
            best_relative_ridge = model
    assert best_relative_ridge is not None

    low, high = np.percentile(flat_target[selected], (1, 99))
    tree_target = np.clip((flat_target - low) / max(high - low, 1e-8), 0, 1)
    tree_selected = deterministic_indices(flat_features.shape[0], 60000)
    tree = fit_tree(
        flat_features[tree_selected], tree_target[tree_selected], flat_weights[tree_selected],
        np.arange(len(FEATURE_NAMES), dtype=np.int32), FEATURE_NAMES, 3,
        classification=False,
    )

    pixel_selected = deterministic_indices(train.truth.size, 9000)
    pair_features = []
    pair_target = []
    pair_weights = []
    balanced = source_balanced_weights(train.source_ids)
    for first, second in PAIRWISE_PAIRS:
        difference = train.features[pixel_selected, first] - train.features[pixel_selected, second]
        margin = train_error[pixel_selected, first] - train_error[pixel_selected, second]
        pair_features.append(difference)
        pair_target.append((margin > 0).astype(np.float64))
        pair_weights.append(balanced[pixel_selected] * (np.abs(margin) + 1e-5))
    pair_features_array = np.concatenate(pair_features)
    pair_target_array = np.concatenate(pair_target)
    pair_weights_array = np.concatenate(pair_weights)
    pair_selected = deterministic_indices(pair_target_array.size, 180000)
    pairwise = fit_logistic(
        pair_features_array[pair_selected], pair_target_array[pair_selected],
        pair_weights_array[pair_selected], np.arange(len(FEATURE_NAMES), dtype=np.int32), 1e-2,
    )

    context_indices = [
        FEATURE_NAMES.index(name) for name in (
            "pass_disagreement", "combined_direction_spread", "robust_direction_spread",
            "mosaic_gradient", "raw_green_variation", "chroma_site_variation",
            "cfa_phase_sin", "cfa_phase_cos",
        )
    ]

    def context(source: CandidateTable) -> np.ndarray:
        shared = source.features[:, 0, context_indices]
        energy_index = FEATURE_NAMES.index("log_directional_energy")
        weight_index = FEATURE_NAMES.index("current_weight")
        extra = np.column_stack((
            np.min(source.features[:, :, energy_index], axis=1),
            np.mean(source.features[:, :, energy_index], axis=1),
            np.max(source.features[:, :, weight_index], axis=1),
            np.ptp(source.candidates[:, :8], axis=1),
            np.ptp(source.candidates[:, 8:], axis=1),
        ))
        return np.column_stack((shared, extra))

    delta = train.current1 - train.current0
    alpha_target = np.clip(
        np.divide(
            train.truth - train.current0, delta,
            out=np.zeros_like(delta), where=np.abs(delta) > 1e-10,
        ), 0, 1,
    )
    pass_model = fit_ridge_score(
        context(train), alpha_target, pixel_weights, 1e-2,
    )

    return {
        "ridge": best_ridge,
        "ridge_relative": best_relative_ridge,
        "tree": tree,
        "pairwise": pairwise,
        "pass_blend": pass_model,
        "context": context,
        "model_report": {
            "ridge_validation_regularization": ridge_rows,
            "ridge_relative_validation_regularization": relative_rows,
            "ridge": {
                "regularization": best_ridge.regularization,
                "intercept": best_ridge.intercept,
                "coefficients": dict(zip(FEATURE_NAMES, map(float, best_ridge.coefficients))),
                "normalizer_mean": best_ridge.normalizer.mean.tolist(),
                "normalizer_scale": best_ridge.normalizer.scale.tolist(),
            },
            "ridge_relative": {
                "regularization": best_relative_ridge.regularization,
                "intercept": best_relative_ridge.intercept,
                "coefficients": dict(zip(FEATURE_NAMES, map(float, best_relative_ridge.coefficients))),
                "normalizer_mean": best_relative_ridge.normalizer.mean.tolist(),
                "normalizer_scale": best_relative_ridge.normalizer.scale.tolist(),
                "target": "candidate log squared error minus the same pixel's 16-candidate mean",
            },
            "tree": tree.root.to_dict(FEATURE_NAMES),
            "pairwise": {
                "regularization": pairwise.regularization,
                "intercept": pairwise.intercept,
                "coefficients": dict(zip(FEATURE_NAMES, map(float, pairwise.coefficients))),
                "pair_count": len(PAIRWISE_PAIRS),
            },
            "pass_blend": {
                "regularization": pass_model.regularization,
                "intercept": pass_model.intercept,
                "coefficients": pass_model.coefficients.tolist(),
            },
        },
    }


def _pairwise_quality(table: CandidateTable, model) -> dict[str, object]:
    result = {}
    for split in ("validation", "test"):
        subset = table.select(table.splits == split)
        actual = (subset.candidates - subset.truth[:, None]) ** 2
        scores = []
        labels = []
        margins = []
        for first, second in PAIRWISE_PAIRS:
            probability = model.alpha(subset.features[:, first] - subset.features[:, second])
            margin = actual[:, first] - actual[:, second]
            scores.append(probability)
            labels.append(margin > 0)
            margins.append(np.abs(margin))
        score = np.concatenate(scores)
        label = np.concatenate(labels)
        margin = np.concatenate(margins)
        predicted = score >= 0.5
        result[split] = {
            "accuracy": float(np.mean(predicted == label)),
            "margin_weighted_accuracy": float(np.sum(margin * (predicted == label)) / max(np.sum(margin), 1e-12)),
            "roc_auc": float(roc_auc(score, label)),
        }
    return result


def _scorer_guides(
    cases: Sequence[Mapping[str, object]],
    scorer: Callable[[np.ndarray], np.ndarray],
    bank: Sequence[int],
    window: int,
    temperature: float | None = None,
) -> list[np.ndarray]:
    result = []
    bank_array = np.asarray(bank, dtype=np.int64)
    for case in cases:
        candidates, _, _ = candidate_bank(case)
        features = feature_maps(case)
        scores = scorer(features)[bank_array]
        if window > 1:
            scores = ndimage.uniform_filter(scores, size=(1, window, window), mode="reflect")
        if temperature is None:
            labels = np.argmin(scores, axis=0)
            guide = np.take_along_axis(candidates[bank_array], labels[None], axis=0)[0]
        else:
            centered = scores - np.min(scores, axis=0, keepdims=True)
            weights = np.exp(-np.clip(centered / temperature, 0, 60))
            weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
            guide = np.sum(weights * candidates[bank_array], axis=0)
        result.append(preserve_measured_green(case, guide).astype(np.float32))
    return result


def _pass_model_guides(cases: Sequence[Mapping[str, object]], model: RidgeScore,
                       context_function, window: int) -> list[np.ndarray]:
    result = []
    for case in cases:
        candidates, _, _ = candidate_bank(case)
        features = feature_maps(case)
        trace = case["trace"]
        # Recreate the compact per-pixel context contract used for fitting.
        shared_names = (
            "pass_disagreement", "combined_direction_spread", "robust_direction_spread",
            "mosaic_gradient", "raw_green_variation", "chroma_site_variation",
            "cfa_phase_sin", "cfa_phase_cos",
        )
        shared = np.stack([features[0, :, :, FEATURE_NAMES.index(name)] for name in shared_names], axis=-1)
        energy = features[:, :, :, FEATURE_NAMES.index("log_directional_energy")]
        weight = features[:, :, :, FEATURE_NAMES.index("current_weight")]
        extra = np.stack((
            np.min(energy, axis=0), np.mean(energy, axis=0), np.max(weight, axis=0),
            np.ptp(candidates[:8], axis=0), np.ptp(candidates[8:], axis=0),
        ), axis=-1)
        context = np.concatenate((shared, extra), axis=-1)
        alpha = np.clip(model.score(context), 0, 1)
        if window > 1:
            alpha = ndimage.uniform_filter(alpha, size=window, mode="reflect")
        guide = (1 - alpha) * np.asarray(trace["pass0-green"]) + alpha * np.asarray(trace["pass1-green"])
        result.append(preserve_measured_green(case, guide).astype(np.float32))
    return result


def _run_final_bank(case: Mapping[str, object], guides: Sequence[np.ndarray], runner: Path,
                    work: Path) -> np.ndarray:
    count = len(guides)
    truth = np.asarray(case["truth"])
    height, width = truth.shape[1:]
    input_path = work / "mosaic.f32le"
    green_path = work / "greens.f32le"
    output_path = work / "rgb.f32le"
    np.asarray(case["mosaic"], dtype="<f4").tofile(input_path)
    np.asarray(guides, dtype="<f4").tofile(green_path)
    completed = subprocess.run([
        str(runner), "final-bank", str(input_path), str(green_path), str(output_path),
        str(count), str(width), str(height), str(case["origin_x"]), str(case["origin_y"]),
    ], text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"native final-bank failed for {case['case_id']}:\n{completed.stdout}\n{completed.stderr}"
        )
    values = np.fromfile(output_path, dtype="<f4")
    if values.size != count * 3 * height * width:
        raise RuntimeError("native final-bank returned the wrong payload size")
    result = values.reshape(count, 3, height, width).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("native final-bank returned non-finite values")
    return result


def _metrics_from_errors(values: Sequence[np.ndarray]) -> dict[str, float]:
    return scalar_metrics(np.concatenate([np.asarray(value).ravel() for value in values]))


def _final_metrics(
    cases: Sequence[Mapping[str, object]],
    guides: Mapping[str, Sequence[np.ndarray]],
    runner: Path,
    root: Path,
) -> tuple[dict[str, object], dict[str, dict[str, list[np.ndarray]]]]:
    accumulators = {
        name: {
            split: {"rgb": [], "missing_rb": [], "opponent": []}
            for split in ("train", "validation", "test")
        }
        for name in guides
    }
    source_errors: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in guides
    }
    names = tuple(guides)
    for case_index, case in enumerate(cases):
        work = root / f"final-{case_index:03d}"
        work.mkdir(exist_ok=True)
        outputs = _run_final_bank(case, [guides[name][case_index] for name in names], runner, work)
        truth = np.asarray(case["truth"])
        cfa = np.asarray(case["cfa"])
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        split = str(case["split"])
        for method_index, name in enumerate(names):
            error = outputs[method_index] - truth
            accumulators[name][split]["rgb"].append(error[:, interior].astype(np.float32))
            missing = []
            for channel in (0, 2):
                mask = interior & (cfa != channel)
                missing.append(error[channel][mask])
            accumulators[name][split]["missing_rb"].append(np.concatenate(missing).astype(np.float32))
            component = opponent(outputs[method_index]) - opponent(truth)
            accumulators[name][split]["opponent"].append(component[:, interior].astype(np.float32))
            source_errors[name][str(case["source_id"])].append(error[:, interior].astype(np.float32))
    result = {}
    for name in names:
        result[name] = {}
        for split in ("train", "validation", "test"):
            rgb = _metrics_from_errors(accumulators[name][split]["rgb"])
            rgb["missing_red_blue"] = _metrics_from_errors(accumulators[name][split]["missing_rb"])
            opponent_values = np.concatenate(accumulators[name][split]["opponent"], axis=1)
            rgb["opponent_rms"] = {
                component: float(np.sqrt(np.mean(opponent_values[index] ** 2)))
                for index, component in enumerate(("L", "C1", "C2"))
            }
            result[name][split] = rgb
    return result, source_errors


def _case_output(case: Mapping[str, object], guide: np.ndarray, runner: Path, work: Path) -> np.ndarray:
    work.mkdir(exist_ok=True)
    return _run_final_bank(case, [guide], runner, work)[0]


def _oracle_guides(cases: Sequence[Mapping[str, object]]) -> tuple[dict[str, list[np.ndarray]], dict[str, object]]:
    guides: dict[str, list[np.ndarray]] = defaultdict(list)
    details: dict[str, dict[str, object]] = {}
    for bank_name, identities in {
        **BANKS,
        "combined-plus-fused": tuple(range(18)),
    }.items():
        details[bank_name] = {}
        counts_by_window = {}
        margins_by_window = {}
        coherence_by_window = {}
        for window in PATCH_WINDOWS:
            counts = np.zeros(len(identities), dtype=np.int64)
            margins = []
            coherence = []
            for case in cases:
                candidates, _, _ = candidate_bank(case)
                if bank_name == "combined-plus-fused":
                    candidates = np.concatenate((
                        candidates,
                        np.asarray(case["trace"]["pass0-green"])[None],
                        np.asarray(case["trace"]["pass1-green"])[None],
                    ))
                else:
                    candidates = candidates[np.asarray(identities)]
                truth = np.asarray(case["truth"])[1]
                error = (candidates - truth[None]) ** 2
                if window > 1:
                    error = ndimage.uniform_filter(error, size=(1, window, window), mode="reflect")
                labels = np.argmin(error, axis=0).astype(np.uint8)
                guide = select_candidates(case, candidates, labels)
                guides[f"oracle-green-{bank_name}-w{window}"].append(guide.astype(np.float32))
                selected = interior_missing_green(case)
                counts += np.bincount(labels[selected], minlength=len(identities))
                sorted_error = np.sort(error[:, selected], axis=0)
                margins.append((sorted_error[1] - sorted_error[0]).astype(np.float32))
                coherence.append(label_coherence(labels, selected))
            counts_by_window[str(window)] = counts.tolist()
            margins_by_window[str(window)] = scalar_metrics(np.concatenate(margins))
            coherence_by_window[str(window)] = {
                key: float(np.mean([row[key] for row in coherence])) for key in coherence[0]
            }
        details[bank_name] = {
            "selection_counts_by_window": counts_by_window,
            "best_second_squared_error_margin": margins_by_window,
            "label_coherence_by_window": coherence_by_window,
        }
    return dict(guides), details


def _candidate_rgb_oracles(
    cases: Sequence[Mapping[str, object]], runner: Path, root: Path,
) -> tuple[dict[str, list[np.ndarray]], dict[str, object]]:
    guides: dict[str, list[np.ndarray]] = defaultdict(list)
    agreements = {str(window): [] for window in PATCH_WINDOWS}
    for case_index, case in enumerate(cases):
        candidates, _, _ = candidate_bank(case)
        work = root / f"candidate-rgb-{case_index:03d}"
        work.mkdir()
        candidate_rgb = _run_final_bank(case, candidates, runner, work)
        truth = np.asarray(case["truth"])
        selected = interior_missing_green(case)
        green_error = (candidates - truth[1][None]) ** 2
        rgb_error = np.sum((candidate_rgb - truth[None]) ** 2, axis=1)
        for window in PATCH_WINDOWS:
            ge = green_error if window == 1 else ndimage.uniform_filter(
                green_error, size=(1, window, window), mode="reflect")
            re = rgb_error if window == 1 else ndimage.uniform_filter(
                rgb_error, size=(1, window, window), mode="reflect")
            green_label = np.argmin(ge, axis=0).astype(np.uint8)
            rgb_label = np.argmin(re, axis=0).astype(np.uint8)
            agreements[str(window)].append(float(np.mean(green_label[selected] == rgb_label[selected])))
            guides[f"oracle-final-rgb-combined-w{window}"].append(
                select_candidates(case, candidates, rgb_label).astype(np.float32)
            )
    return dict(guides), {
        "green_vs_final_rgb_label_agreement": {
            window: float(np.mean(values)) for window, values in agreements.items()
        }
    }


def _controls_and_convex(cases: Sequence[Mapping[str, object]]) -> tuple[dict[str, list[np.ndarray]], dict[str, object]]:
    guides: dict[str, list[np.ndarray]] = defaultdict(list)
    inside: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        candidates, energies, _ = candidate_bank(case)
        trace = case["trace"]
        guides["current-pass1"].append(
            preserve_measured_green(case, trace["pass1-green"]).astype(np.float32)
        )
        guides["pass0-current"].append(
            preserve_measured_green(case, trace["pass0-green"]).astype(np.float32)
        )
        guides["true-green"].append(
            preserve_measured_green(case, np.asarray(case["truth"])[1]).astype(np.float32)
        )
        for bank_name, identities in BANKS.items():
            bank = candidates[np.asarray(identities)]
            guides[f"uniform-{bank_name}"].append(preserve_measured_green(case, np.mean(bank, axis=0)).astype(np.float32))
            guides[f"median-{bank_name}"].append(preserve_measured_green(case, np.median(bank, axis=0)).astype(np.float32))
            labels = np.argmin(energies[np.asarray(identities)], axis=0)
            guides[f"min-energy-{bank_name}"].append(select_candidates(case, bank, labels).astype(np.float32))
            convex, contained = scalar_convex_hull(case, bank)
            guides[f"oracle-convex-{bank_name}"].append(convex.astype(np.float32))
            selected = interior_missing_green(case)
            inside[bank_name].append(float(np.mean(contained[selected])))
        for exponent in (0.0, 0.5, 1.0, 2.0, 4.0):
            guides[f"temperature-pass1-p{exponent:g}"].append(
                temperature_fusion(case, 1, exponent).astype(np.float32)
            )
        for window in PATCH_WINDOWS:
            alpha = patch_blend_alpha(
                np.asarray(trace["pass0-green"]), np.asarray(trace["pass1-green"]),
                np.asarray(case["truth"])[1], window,
            )
            blend = (1 - alpha) * np.asarray(trace["pass0-green"]) + alpha * np.asarray(trace["pass1-green"])
            guides[f"oracle-pass-blend-w{window}"].append(preserve_measured_green(case, blend).astype(np.float32))
            pass0_error = np.sum((np.asarray(case["pass0_guide_final"]) - np.asarray(case["truth"])) ** 2, axis=0)
            current = np.stack((trace["final-red"], trace["final-green"], trace["final-blue"]))
            pass1_error = np.sum((current - np.asarray(case["truth"])) ** 2, axis=0)
            if window > 1:
                pass0_error = ndimage.uniform_filter(pass0_error, size=window, mode="reflect")
                pass1_error = ndimage.uniform_filter(pass1_error, size=window, mode="reflect")
            use_pass0 = pass0_error <= pass1_error
            adaptive = np.where(use_pass0, np.asarray(trace["pass0-green"]), np.asarray(trace["pass1-green"]))
            guides[f"oracle-adaptive-pass-w{window}"].append(preserve_measured_green(case, adaptive).astype(np.float32))
            selected = interior_missing_green(case)
            inside[f"adaptive-pass0-w{window}"].append(float(np.mean(use_pass0[selected])))
    return dict(guides), {
        "true_green_inside_candidate_interval": {
            name: float(np.mean(values)) for name, values in inside.items() if not name.startswith("adaptive")
        },
        "oracle_adaptive_pass0_fraction": {
            name.removeprefix("adaptive-pass0-"): float(np.mean(values))
            for name, values in inside.items() if name.startswith("adaptive")
        },
    }


def _practical_guides(cases: Sequence[Mapping[str, object]], models: Mapping[str, object]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    scorers = {
        "ridge": models["ridge"].score,
        "ridge-relative": models["ridge_relative"].score,
        "tree": lambda features: _tree_scores(models["tree"], features),
        "pairwise": lambda features: _pairwise_scores(models["pairwise"], features),
    }
    shared_names = (
        "pass_disagreement", "combined_direction_spread", "robust_direction_spread",
        "mosaic_gradient", "raw_green_variation", "chroma_site_variation",
        "cfa_phase_sin", "cfa_phase_cos",
    )
    for case in cases:
        candidates, _, _ = candidate_bank(case)
        features = feature_maps(case)
        scores_by_model = {name: scorer(features) for name, scorer in scorers.items()}
        for name, scores_all in scores_by_model.items():
            for bank_name, bank in BANKS.items():
                bank_array = np.asarray(bank, dtype=np.int64)
                for window in (1, 3, 7):
                    scores = scores_all[bank_array]
                    if window > 1:
                        scores = ndimage.uniform_filter(
                            scores, size=(1, window, window), mode="reflect",
                        )
                    labels = np.argmin(scores, axis=0)
                    guide = np.take_along_axis(
                        candidates[bank_array], labels[None], axis=0,
                    )[0]
                    result[f"practical-{name}-{bank_name}-hard-w{window}"].append(
                        preserve_measured_green(case, guide).astype(np.float32)
                    )
        ridge_scores = scores_by_model["ridge"]
        for window in (1, 3, 7):
            scores = ridge_scores
            if window > 1:
                scores = ndimage.uniform_filter(
                    scores, size=(1, window, window), mode="reflect",
                )
            for temperature in (0.1, 0.25, 0.5, 1.0):
                centered = scores - np.min(scores, axis=0, keepdims=True)
                weights = np.exp(-np.clip(centered / temperature, 0, 60))
                weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
                guide = np.sum(weights * candidates, axis=0)
                result[f"practical-ridge-combined-soft-w{window}-t{temperature:g}"].append(
                    preserve_measured_green(case, guide).astype(np.float32)
                )

        shared = np.stack([
            features[0, :, :, FEATURE_NAMES.index(name)] for name in shared_names
        ], axis=-1)
        energy = features[:, :, :, FEATURE_NAMES.index("log_directional_energy")]
        weight = features[:, :, :, FEATURE_NAMES.index("current_weight")]
        context = np.concatenate((shared, np.stack((
            np.min(energy, axis=0), np.mean(energy, axis=0), np.max(weight, axis=0),
            np.ptp(candidates[:8], axis=0), np.ptp(candidates[8:], axis=0),
        ), axis=-1)), axis=-1)
        alpha = np.clip(models["pass_blend"].score(context), 0, 1)
        trace = case["trace"]
        for window in (1, 3, 7):
            smoothed = alpha if window == 1 else ndimage.uniform_filter(
                alpha, size=window, mode="reflect",
            )
            guide = (
                (1 - smoothed) * np.asarray(trace["pass0-green"])
                + smoothed * np.asarray(trace["pass1-green"])
            )
            result[f"practical-pass-blend-ridge-w{window}"].append(
                preserve_measured_green(case, guide).astype(np.float32)
            )
    return dict(result)


def _source_rows(
    cases: Sequence[Mapping[str, object]], source_errors: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    methods: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for source in sorted({str(case["source_id"]) for case in cases}):
        selected_cases = [case for case in cases if str(case["source_id"]) == source]
        row: dict[str, object] = {
            "source_id": source,
            "split": str(selected_cases[0]["split"]),
            "kind": str(selected_cases[0]["kind"]),
        }
        for method in methods:
            row[method] = _metrics_from_errors(source_errors[method][source])
        rows.append(row)
    return rows


def _direct_baseline_metrics(
    cases: Sequence[Mapping[str, object]], getter: Callable[[Mapping[str, object]], np.ndarray],
) -> tuple[dict[str, object], dict[str, list[np.ndarray]]]:
    accumulators = {
        split: {"rgb": [], "missing_rb": [], "opponent": []}
        for split in ("train", "validation", "test")
    }
    source_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    for case in cases:
        output = np.asarray(getter(case), dtype=np.float64)
        truth = np.asarray(case["truth"], dtype=np.float64)
        cfa = np.asarray(case["cfa"])
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        error = output - truth
        split = str(case["split"])
        accumulators[split]["rgb"].append(error[:, interior].astype(np.float32))
        accumulators[split]["missing_rb"].append(np.concatenate([
            error[channel][interior & (cfa != channel)] for channel in (0, 2)
        ]).astype(np.float32))
        components = opponent(output) - opponent(truth)
        accumulators[split]["opponent"].append(components[:, interior].astype(np.float32))
        source_errors[str(case["source_id"])].append(error[:, interior].astype(np.float32))
    result = {}
    for split in ("train", "validation", "test"):
        row = _metrics_from_errors(accumulators[split]["rgb"])
        row["missing_red_blue"] = _metrics_from_errors(accumulators[split]["missing_rb"])
        components = np.concatenate(accumulators[split]["opponent"], axis=1)
        row["opponent_rms"] = {
            name: float(np.sqrt(np.mean(components[index] ** 2)))
            for index, name in enumerate(("L", "C1", "C2"))
        }
        result[split] = row
    return result, dict(source_errors)


def _diagnostic_map(
    case: Mapping[str, object], practical: np.ndarray, practical_output: np.ndarray,
    ridge_model: RidgeScore, path: Path,
) -> None:
    candidates, _, _ = candidate_bank(case)
    truth = np.asarray(case["truth"])
    current = np.stack((case["trace"]["final-red"], case["trace"]["final-green"], case["trace"]["final-blue"]))
    features = feature_maps(case)
    scores = ridge_model.score(features)
    selected_label = np.argmin(scores, axis=0)
    oracle_label = patch_labels(candidates, truth[1], 7)
    selected_error = np.take_along_axis((candidates - truth[1][None]) ** 2, selected_label[None], axis=0)[0]
    oracle_error = np.min((candidates - truth[1][None]) ** 2, axis=0)
    regret = np.maximum(0, selected_error - oracle_error)
    spread = np.ptp(candidates, axis=0)
    min_cost = np.min(scores, axis=0)

    def gray(values: np.ndarray) -> Image.Image:
        clipped = np.clip(values, 0, 1)
        encoded = np.rint(clipped * 255).astype(np.uint8)
        return Image.fromarray(np.stack((encoded,) * 3, axis=-1), "RGB")

    def heat(values: np.ndarray, scale: float | None = None) -> Image.Image:
        checked = np.asarray(values, dtype=np.float64)
        if scale is None:
            scale = max(float(np.percentile(checked, 99)), 1e-8)
        mapped = np.clip(checked / scale, 0, 1)
        rgb = np.stack((mapped, np.sqrt(mapped), 1 - mapped), axis=-1)
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), "RGB")

    def labels(values: np.ndarray) -> Image.Image:
        palette = np.asarray([
            (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
            (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
            (170, 110, 40), (128, 0, 0), (170, 255, 195), (0, 0, 128),
            (128, 128, 0), (255, 215, 180), (0, 128, 128), (128, 128, 128),
        ], dtype=np.uint8)
        return Image.fromarray(palette[np.asarray(values, dtype=np.int64) % len(palette)], "RGB")

    panels = (
        ("truth green", gray(truth[1])),
        ("pass0 green", gray(case["trace"]["pass0-green"])),
        ("pass1 green", gray(case["trace"]["pass1-green"])),
        ("current RGB error", heat(np.mean(np.abs(current - truth), axis=0))),
        ("ridge identity", labels(selected_label)),
        ("oracle 7x7 identity", labels(oracle_label)),
        ("selection regret", heat(regret)),
        ("direction spread", heat(spread)),
        ("predicted min cost", heat(min_cost - np.min(min_cost))),
        ("practical RGB error", heat(np.mean(np.abs(practical_output - truth), axis=0))),
    )
    scale = max(1, min(4, 220 // truth.shape[1]))
    width = truth.shape[2] * scale
    height = truth.shape[1] * scale
    output = Image.new("RGB", (len(panels) * width, height + 24), "white")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        output.paste(panel.resize((width, height), Image.Resampling.NEAREST), (index * width, 24))
        draw.text((index * width + 3, 5), label, fill=(10, 10, 10), font=font)
    output.save(path, format="PNG", compress_level=9)


def _select_map_cases(cases: Sequence[Mapping[str, object]]) -> list[int]:
    wanted = (
        "hubble-0", "nasa-hydra-starfield-0", "brick-0", "page-0", "grass-0",
        "white-impulse", "control-saturated_red_gray",
    )
    result = []
    for name in wanted:
        matches = [index for index, case in enumerate(cases) if str(case["case_id"]) == name]
        if matches:
            result.append(matches[0])
    return result


def _oracle_conditionals(cases: Sequence[Mapping[str, object]], window: int = 7) -> dict[str, object]:
    groups: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(16, dtype=np.int64))
    for case in cases:
        candidates, _, _ = candidate_bank(case)
        labels = patch_labels(candidates, np.asarray(case["truth"])[1], window)
        selected = interior_missing_green(case)
        counts = np.bincount(labels[selected], minlength=16)
        groups[f"kind:{case['kind']}"] += counts
        groups[f"family:{case['family']}"] += counts
    result = {}
    for name, counts in sorted(groups.items()):
        total = int(np.sum(counts))
        result[name] = {
            "count": total,
            "pass0_fraction": float(np.sum(counts[:8]) / max(1, total)),
            "pass1_fraction": float(np.sum(counts[8:]) / max(1, total)),
            "direction_fraction": [float(value / max(1, total)) for value in counts],
        }
    return result


def _practical_label_map(case: Mapping[str, object], method: str,
                         models: Mapping[str, object]) -> np.ndarray | None:
    if method.startswith("min-energy-"):
        bank_name = method.removeprefix("min-energy-")
        if bank_name not in BANKS:
            return None
        _, energies, _ = candidate_bank(case)
        identities = np.asarray(BANKS[bank_name], dtype=np.int64)
        return identities[np.argmin(energies[identities], axis=0)]
    if not method.startswith("practical-") or "-hard-w" not in method:
        return None
    prefix, window_text = method.rsplit("-hard-w", 1)
    window = int(window_text)
    matched = None
    for model_name in ("ridge-relative", "ridge", "tree", "pairwise"):
        marker = f"practical-{model_name}-"
        if prefix.startswith(marker):
            matched = model_name
            bank_name = prefix[len(marker):]
            break
    if matched is None or bank_name not in BANKS:
        return None
    features = feature_maps(case)
    if matched == "ridge-relative":
        scores = models["ridge_relative"].score(features)
    elif matched == "ridge":
        scores = models["ridge"].score(features)
    elif matched == "tree":
        scores = _tree_scores(models["tree"], features)
    else:
        scores = _pairwise_scores(models["pairwise"], features)
    identities = np.asarray(BANKS[bank_name], dtype=np.int64)
    selected_scores = scores[identities]
    if window > 1:
        selected_scores = ndimage.uniform_filter(
            selected_scores, size=(1, window, window), mode="reflect",
        )
    return identities[np.argmin(selected_scores, axis=0)]


def _transition_rows(
    cases: Sequence[Mapping[str, object]], guides: Mapping[str, Sequence[np.ndarray]],
    selected_method: str, models: Mapping[str, object], source_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    source_by_name = {str(row["source_id"]): row for row in source_rows}
    result = []
    for case_index, case in enumerate(cases):
        case_id = str(case["case_id"])
        if not (
            case_id.startswith("transition-points-")
            or case_id.startswith("transition-line-")
            or case_id.startswith("transition-coherence-")
        ):
            continue
        selected = interior_missing_green(case)
        label = _practical_label_map(case, selected_method, models)
        _, _, weights = candidate_bank(case)
        row: dict[str, object] = {
            "case_id": case_id,
            "family": case["family"],
            "current_directional_weight_mean": np.mean(weights[:, selected], axis=1).tolist(),
            "current_green": scalar_metrics(
                (np.asarray(case["trace"]["pass1-green"]) - np.asarray(case["truth"])[1])[selected]
            ),
            "practical_green": scalar_metrics(
                (np.asarray(guides[selected_method][case_index]) - np.asarray(case["truth"])[1])[selected]
            ),
            "current_final_rgb": source_by_name[str(case["source_id"])]["current-pass1"],
            "practical_final_rgb": source_by_name[str(case["source_id"])][selected_method],
        }
        if label is not None:
            counts = np.bincount(label[selected], minlength=16)
            row["selected_pass0_fraction"] = float(np.mean(label[selected] < 8))
            row["selected_direction_fraction"] = (
                counts / max(1, int(np.sum(counts)))
            ).tolist()
        result.append(row)
    return result


def run(runner: Path, starfield: Path, output: Path) -> dict[str, object]:
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(starfield)
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-mlri-green-fusion-") as temporary:
        root = Path(temporary)
        trace_start = time.perf_counter()
        for index, case in enumerate(cases):
            work = root / f"trace-{index:03d}"
            work.mkdir()
            _run_case(case, runner, work)
            print(f"[trace {index + 1:02d}/{len(cases):02d}] {case['case_id']}", flush=True)
        timings["trace_and_baselines_seconds"] = time.perf_counter() - trace_start

        feature_start = time.perf_counter()
        print("[analysis] building candidate feature table", flush=True)
        table = build_candidate_table(cases)
        timings["candidate_feature_seconds"] = time.perf_counter() - feature_start
        model_start = time.perf_counter()
        print("[analysis] fitting ridge, tree, pairwise, and pass-blend models", flush=True)
        models = _fit_models(table)
        timings["model_fit_seconds"] = time.perf_counter() - model_start

        oracle_start = time.perf_counter()
        print("[analysis] constructing spatial green oracles", flush=True)
        guides, oracle_structure = _oracle_guides(cases)
        print("[analysis] running 16 candidate guides through native final R/B", flush=True)
        rgb_oracle_guides, rgb_oracle_details = _candidate_rgb_oracles(cases, runner, root)
        guides.update(rgb_oracle_guides)
        controls, convex_details = _controls_and_convex(cases)
        guides.update(controls)
        practical = _practical_guides(cases, models)
        guides.update(practical)
        timings["guide_generation_and_candidate_rgb_seconds"] = time.perf_counter() - oracle_start

        for name, values in guides.items():
            if len(values) != len(cases):
                raise RuntimeError(f"guide count mismatch for {name}")
            for case, guide in zip(cases, values):
                measured = np.asarray(case["cfa"]) == 1
                if not np.array_equal(
                    np.asarray(guide, dtype=np.float32)[measured],
                    np.asarray(case["mosaic"], dtype=np.float32)[measured],
                ):
                    raise RuntimeError(f"{name} changed a measured green sample")

        final_start = time.perf_counter()
        print(f"[analysis] evaluating {len(guides)} guide methods through native final R/B", flush=True)
        final_metrics, source_errors = _final_metrics(cases, guides, runner, root)
        mark_metrics, mark_source_errors = _direct_baseline_metrics(
            cases, lambda case: np.asarray(case["markesteijn"]),
        )
        final_metrics["markesteijn"] = mark_metrics
        source_errors["markesteijn"] = mark_source_errors
        timings["all_final_red_blue_evaluations_seconds"] = time.perf_counter() - final_start
        green_results = {
            name: {
                split: green_metrics(
                    [case for case in cases if case["split"] == split],
                    [guide for case, guide in zip(cases, values) if case["split"] == split],
                )
                for split in ("train", "validation", "test")
            }
            for name, values in guides.items()
        }

        practical_names = tuple(practical)
        deployable_names = practical_names + tuple(
            name for name in guides
            if name.startswith(("uniform-", "median-", "min-energy-", "temperature-"))
        ) + ("pass0-current", "current-pass1")
        selected_pooled = max(
            deployable_names,
            key=lambda name: final_metrics[name]["validation"]["psnr_db"],
        )
        validation_sources = sorted({
            str(case["source_id"]) for case in cases if case["split"] == "validation"
        })
        selection_rows = []
        brick_current = _metrics_from_errors(source_errors["current-pass1"]["brick"])
        brick_pass0 = _metrics_from_errors(source_errors["pass0-current"]["brick"])
        brick_denominator = brick_pass0["mse"] - brick_current["mse"]
        hubble_current = _metrics_from_errors(source_errors["current-pass1"]["hubble"])
        for name in deployable_names:
            source_psnr = {}
            source_regression = {}
            for source in validation_sources:
                method_metrics = _metrics_from_errors(source_errors[name][source])
                current_metrics = _metrics_from_errors(source_errors["current-pass1"][source])
                source_psnr[source] = method_metrics["psnr_db"]
                source_regression[source] = method_metrics["psnr_db"] - current_metrics["psnr_db"]
            brick_method = _metrics_from_errors(source_errors[name]["brick"])
            brick_retention = None if brick_denominator <= 0 else float(
                (brick_pass0["mse"] - brick_method["mse"]) / brick_denominator
            )
            hubble_method = _metrics_from_errors(source_errors[name]["hubble"])
            hubble_gain = hubble_method["psnr_db"] - hubble_current["psnr_db"]
            selection_rows.append({
                "method": name,
                "mean_source_psnr_db": float(np.mean(list(source_psnr.values()))),
                "minimum_source_psnr_db": float(np.min(list(source_psnr.values()))),
                "mean_gain_from_current_db": float(np.mean(list(source_regression.values()))),
                "worst_gain_from_current_db": float(np.min(list(source_regression.values()))),
                "source_psnr_db": source_psnr,
                "source_gain_from_current_db": source_regression,
                "brick_pass1_gain_retention": brick_retention,
                "hubble_gain_db": hubble_gain,
                "eligible": (
                    (brick_retention is None or brick_retention >= 0.8)
                    and hubble_gain >= 0
                ),
            })
        selection_rows.sort(key=lambda row: row["mean_source_psnr_db"], reverse=True)
        eligible_rows = [row for row in selection_rows if row["eligible"]]
        if not eligible_rows:
            raise RuntimeError("no validation method preserves Brick and Hubble")
        selected_practical = str(eligible_rows[0]["method"])
        selected_test = final_metrics[selected_practical]["test"]
        current_test = final_metrics["current-pass1"]["test"]
        oracle_test = final_metrics["oracle-green-combined-w7"]["test"]
        denominator = current_test["mse"] - oracle_test["mse"]
        recovered = 0.0 if denominator <= 0 else (
            current_test["mse"] - selected_test["mse"]
        ) / denominator

        calibrations = {}
        for split in ("validation", "test"):
            subset = table.select(table.splits == split)
            calibrations[split] = {
                "ridge": score_calibration(subset, models["ridge"].score(subset.features)),
                "ridge_relative": score_calibration(
                    subset, models["ridge_relative"].score(subset.features),
                ),
                "tree": score_calibration(subset, _tree_scores(models["tree"], subset.features)),
                "pairwise": score_calibration(subset, _pairwise_scores(models["pairwise"], subset.features)),
            }

        feature_importance = sorted(
            (
                {"feature": name, "absolute_standardized_coefficient": abs(float(value))}
                for name, value in zip(FEATURE_NAMES, models["ridge"].coefficients)
            ),
            key=lambda row: row["absolute_standardized_coefficient"], reverse=True,
        )
        relative_feature_importance = sorted(
            (
                {"feature": name, "absolute_standardized_coefficient": abs(float(value))}
                for name, value in zip(FEATURE_NAMES, models["ridge_relative"].coefficients)
            ),
            key=lambda row: row["absolute_standardized_coefficient"], reverse=True,
        )
        core_methods = (
            "markesteijn", "current-pass1", "pass0-current", selected_practical,
            "oracle-green-combined-w7", "oracle-final-rgb-combined-w7",
            "oracle-convex-combined", "true-green",
        )
        sources = _source_rows(cases, source_errors, core_methods)
        transitions = _transition_rows(
            cases, guides, selected_practical, models, sources,
        )

        runtime = _runtime(cases)
        pass1_fraction = sum(
            runtime["fraction_of_traced_mlri_time"][name]
            for name in ("pass1-guide", "pass1-green", "pass1-chroma")
        )
        skip_fraction = convex_details["oracle_adaptive_pass0_fraction"]["w7"]
        adaptive_saving = skip_fraction * pass1_fraction

        results: dict[str, object] = {
            "format": "rawtherapee-xtrans-mlri-green-fusion-results-v1",
            "decision": {},
            "dataset": dataset_manifest(cases),
            "candidate_contract": {
                "candidate_count": 16,
                "directions": list(DIRECTION_NAMES),
                "passes": [0, 1],
                "feature_names": list(FEATURE_NAMES),
                "guided_regression_fit_statistics": (
                    "not retained per final directional candidate by guidedMlri; exact directional energy and normalized fusion weight are traced"
                ),
            },
            "oracle_structure": oracle_structure,
            "oracle_conditionals_w7": _oracle_conditionals(cases, 7),
            "green_vs_final_rgb_oracle": rgb_oracle_details,
            "convex_and_adaptive_pass": convex_details,
            "models": models["model_report"],
            "pairwise_ranking": _pairwise_quality(table, models["pairwise"]),
            "candidate_score_calibration": calibrations,
            "ridge_feature_importance": feature_importance,
            "ridge_relative_feature_importance": relative_feature_importance,
            "green_metrics": green_results,
            "final_rgb_metrics": final_metrics,
            "selected_practical_by_validation_final_rgb": selected_practical,
            "pooled_validation_selection": selected_pooled,
            "source_balanced_validation_selection": selection_rows,
            "validation_selection_policy": (
                "maximize mean per-source validation PSNR among methods retaining at least 80% of Brick's pass-1 MSE gain and not worsening Hubble"
            ),
            "held_out_oracle_gap_recovery_fraction": recovered,
            "source_metrics": sources,
            "transition_analysis": transitions,
            "runtime": {
                "native_trace": runtime,
                "experiment_seconds": timings,
                "oracle_pass1_work_fraction": pass1_fraction,
                "oracle_w7_pass0_fraction": skip_fraction,
                "oracle_estimated_total_runtime_saving_fraction": adaptive_saving,
                "note": "research Python feature/model time and traced scalar stage time are not production benchmarks",
            },
        }

        # A GO requires a held-out final-RGB win and positive practical recovery.
        sparse_sources = {"hubble", "nasa-hydra-starfield"}
        source_by_name = {row["source_id"]: row for row in sources}
        sparse_improvements = {
            source: source_by_name[source][selected_practical]["psnr_db"]
            - source_by_name[source]["current-pass1"]["psnr_db"]
            for source in sparse_sources
        }
        texture_sources = ("brick", "grass", "gravel", "page")
        retained = {}
        for source in texture_sources:
            row = source_by_name[source]
            # Retention is measured against pass-0 rather than Markesteijn so
            # the value isolates the benefit of the second MLRI pass.
            current_mse = row["current-pass1"]["mse"]
            pass0_mse = row["pass0-current"]["mse"]
            practical_mse = row[selected_practical]["mse"]
            denominator_texture = pass0_mse - current_mse
            retained[source] = None if denominator_texture <= 0 else float(
                (pass0_mse - practical_mse) / denominator_texture
            )
        held_out_gain = selected_test["psnr_db"] - current_test["psnr_db"]
        substantial = recovered >= 0.30 and held_out_gain > 0
        sparse_ok = all(value > 0 for value in sparse_improvements.values())
        texture_ok = all(value is None or value >= 0.8 for value in retained.values())
        if substantial and sparse_ok and texture_ok:
            decision = "GO — directional candidate fusion"
        elif held_out_gain > 0 or any(value > 0 for value in sparse_improvements.values()):
            decision = "PARTIAL"
        else:
            decision = "NO-GO"
        results["decision"] = {
            "classification": decision,
            "held_out_final_rgb_gain_db": held_out_gain,
            "held_out_oracle_gap_recovery_fraction": recovered,
            "sparse_source_gain_db": sparse_improvements,
            "coherent_texture_pass1_gain_retention": retained,
            "criteria": {
                "positive_held_out_final_rgb": held_out_gain > 0,
                "recover_at_least_30_percent": recovered >= 0.30,
                "both_sparse_sources_improve": sparse_ok,
                "retain_at_least_80_percent_texture_gain": texture_ok,
            },
        }

        # Re-run only the selected/core guides for the diagnostic maps.
        for case_index in _select_map_cases(cases):
            case = cases[case_index]
            work = root / f"map-{case_index:03d}"
            practical_output = _case_output(
                case, guides[selected_practical][case_index], runner, work,
            )
            _diagnostic_map(
                case, guides[selected_practical][case_index], practical_output,
                models["ridge"], output / f"map-{case['case_id']}.png",
            )

        _write(output / "dataset.json", dataset_manifest(cases))
        _write(output / "results.json", results)
        manifest = {
            "format": "rawtherapee-xtrans-mlri-green-fusion-manifest-v1",
            "dataset_sha256": _sha256(output / "dataset.json"),
            "results_sha256": _sha256(output / "results.json"),
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(output.glob("map-*.png"))
            ],
        }
        _write(output / "manifest.json", manifest)
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--starfield", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    results = run(arguments.runner, arguments.starfield, arguments.output)
    decision = results["decision"]
    print(
        f"{decision['classification']}: selected={results['selected_practical_by_validation_final_rgb']} "
        f"held-out gain={decision['held_out_final_rgb_gain_db']:.4f} dB "
        f"recovery={100 * decision['held_out_oracle_gap_recovery_fraction']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

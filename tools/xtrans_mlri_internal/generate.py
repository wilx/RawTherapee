#!/usr/bin/env python3
"""Run the corrected-final MLRI internal failure decomposition experiment."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
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
from tools.xtrans_oracle.analysis import opponent
from tools.xtrans_oracle.generate import srgb_encode
from .analysis import (
    FEATURE_NAMES,
    MARGIN,
    SampleTable,
    alpha_oracle,
    build_sample_table,
    classification_metrics,
    local_mad,
    local_rms,
    operating_threshold,
    reconstruction_metrics,
    roc_auc,
    scalar_metrics,
)
from .dataset import dataset_manifest, load_cases, mosaic


TRACE_PLANES = (
    "pass0-green", "pass0-provisional-red", "pass0-provisional-blue",
    "pass1-green", "pass1-provisional-red", "pass1-provisional-blue",
    "final-tentative-red", "final-tentative-blue",
    "final-raw-residual-red", "final-raw-residual-blue",
    "final-correction-red", "final-correction-blue",
    "final-unclipped-red", "final-unclipped-blue",
    "final-red", "final-green", "final-blue",
) + tuple(
    f"pass{pass_index}-green-direction-{direction}"
    for pass_index in (0, 1) for direction in range(8)
) + tuple(
    f"pass{pass_index}-green-{statistic}-{direction}"
    for pass_index in (0, 1)
    for statistic in ("energy", "weight")
    for direction in range(8)
 ) + tuple(
    f"pass{pass_index}-green-regression-{statistic}-{direction}"
    for pass_index in (0, 1)
    for statistic in (
        "fit-mse", "min-fit-mse", "gain", "gain-variance",
        "gain-min", "gain-max", "offset", "gain-numerator",
        "gain-denominator", "guide-energy", "laplacian-count",
        "sample-count", "prediction-variance", "prediction-range",
        "prediction-mad", "prediction-iqr", "model-weight-entropy",
        "effective-model-count", "max-model-weight", "total-model-weight",
        "loo-prediction-variance", "max-model-influence",
    )
    for direction in range(8)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_case(case: dict[str, object], runner: Path, work: Path) -> None:
    truth = np.asarray(case["truth"], dtype=np.float64)
    scalar, cfa = mosaic(truth, int(case["origin_x"]), int(case["origin_y"]))
    input_path = work / "mosaic.f32le"
    truth_path = work / "truth.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    np.asarray(truth, dtype="<f4").tofile(truth_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            str(runner), "run", str(input_path), str(truth_path), str(work),
            str(truth.shape[2]), str(truth.shape[1]),
            str(case["origin_x"]), str(case["origin_y"]),
        ],
        env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native internal runner failed for {case['case_id']}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    pixels = truth.shape[1] * truth.shape[2]
    trace = {}
    for name in TRACE_PLANES:
        values = np.fromfile(work / f"{name}.f32le", dtype="<f4")
        if values.size != pixels:
            raise RuntimeError(f"incorrect trace size for {name}")
        trace[name] = values.astype(np.float64).reshape(truth.shape[1:])
    mark = np.fromfile(work / "markesteijn.f32le", dtype="<f4")
    if mark.size != truth.size:
        raise RuntimeError("incorrect Markesteijn output size")
    oracle_green = []
    pass0_guide = []
    for channel in ("red", "blue"):
        values = np.fromfile(work / f"oracle-green-{channel}.f32le", dtype="<f4")
        if values.size != pixels:
            raise RuntimeError("incorrect oracle-green output size")
        oracle_green.append(values.astype(np.float64).reshape(truth.shape[1:]))
        values = np.fromfile(work / f"pass0-guide-{channel}.f32le", dtype="<f4")
        if values.size != pixels:
            raise RuntimeError("incorrect pass-0-guide output size")
        pass0_guide.append(values.astype(np.float64).reshape(truth.shape[1:]))
    timings = {}
    for line in completed.stdout.splitlines():
        name, seconds = line.split("\t")
        timings[name] = float(seconds)
    decomposition_error = max(
        float(np.max(np.abs(
            trace[f"final-tentative-{suffix}"]
            + trace[f"final-correction-{suffix}"]
            - trace[f"final-unclipped-{suffix}"]
        )))
        for suffix in ("red", "blue")
    )
    if decomposition_error > 2e-6:
        raise RuntimeError(f"trace decomposition mismatch: {decomposition_error}")
    case["mosaic"] = scalar
    case["cfa"] = cfa
    case["trace"] = trace
    case["markesteijn"] = mark.astype(np.float64).reshape(truth.shape)
    case["oracle_green"] = np.stack((oracle_green[0], truth[1], oracle_green[1]))
    case["pass0_guide_final"] = np.stack((
        pass0_guide[0], np.clip(trace["pass0-green"], 0, 1), pass0_guide[1]
    ))
    case["timings"] = timings


def _rgb(case: Mapping[str, object], stage: str) -> np.ndarray:
    trace = case["trace"]
    if stage == "pass0":
        return np.stack((trace["pass0-provisional-red"], trace["pass0-green"], trace["pass0-provisional-blue"]))
    if stage == "pass1":
        return np.stack((trace["pass1-provisional-red"], trace["pass1-green"], trace["pass1-provisional-blue"]))
    if stage == "guide_only":
        return np.stack((trace["final-tentative-red"], trace["final-green"], trace["final-tentative-blue"]))
    if stage == "final":
        return np.stack((trace["final-red"], trace["final-green"], trace["final-blue"]))
    if stage == "markesteijn":
        return np.asarray(case["markesteijn"])
    if stage == "oracle_green":
        return np.asarray(case["oracle_green"])
    if stage == "pass0_guide_final":
        return np.asarray(case["pass0_guide_final"])
    raise KeyError(stage)


def _interior(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[..., MARGIN:-MARGIN, MARGIN:-MARGIN]


def _quality(cases: Sequence[Mapping[str, object]], getter: Callable[[Mapping[str, object]], np.ndarray]) -> dict[str, object]:
    errors = []
    opponent_errors = []
    for case in cases:
        error = _interior(getter(case) - np.asarray(case["truth"]))
        errors.append(error.ravel())
        opponent_errors.append(_interior(opponent(getter(case)) - opponent(np.asarray(case["truth"]))))
    values = np.concatenate(errors)
    components = np.concatenate([values.reshape(3, -1) for values in opponent_errors], axis=1)
    result: dict[str, object] = scalar_metrics(values)
    result["opponent_rms"] = {
        name: float(np.sqrt(np.mean(components[index] ** 2)))
        for index, name in enumerate(("L", "C1", "C2"))
    }
    return result


def _row_reconstruction(table: SampleTable, name: str) -> np.ndarray:
    if name == "pass0":
        return table.pass0
    if name == "pass1":
        return table.pass1
    if name == "guide_only":
        return table.base
    if name == "final":
        return table.final
    if name == "pass0_final":
        return table.pass0_final
    if name == "oracle_alpha":
        alpha = alpha_oracle(table.base, table.correction, table.truth)
        return np.clip(table.base + alpha * table.correction, 0, 1)
    if name == "oracle_alpha_wide":
        alpha = alpha_oracle(table.base, table.correction, table.truth, -0.5, 1.5)
        return np.clip(table.base + alpha * table.correction, 0, 1)
    if name == "oracle_reject":
        helps = (table.final - table.truth) ** 2 <= (table.base - table.truth) ** 2
        return np.where(helps, table.final, table.base)
    if name == "oracle_refinement_reject":
        helps = (table.final - table.truth) ** 2 <= (table.pass0_final - table.truth) ** 2
        return np.where(helps, table.final, table.pass0_final)
    if name == "candidate_oracle":
        candidates = np.stack((
            table.pass0, table.pass1, table.base, table.pass0_final, table.final,
        ))
        errors = np.abs(candidates - table.truth[None])
        return np.take_along_axis(candidates, np.argmin(errors, axis=0)[None], axis=0)[0]
    if name == "candidate_median":
        return np.median(np.stack((
            table.pass0, table.pass1, table.base, table.pass0_final, table.final,
        )), axis=0)
    if name == "candidate_trimmed":
        candidates = np.sort(np.stack((
            table.pass0, table.pass1, table.base, table.pass0_final, table.final,
        )), axis=0)
        return np.mean(candidates[1:-1], axis=0)
    raise KeyError(name)


def _split_metrics(table: SampleTable, values: np.ndarray) -> dict[str, object]:
    return {
        split: reconstruction_metrics(table.select(table.splits == split), values[table.splits == split])
        for split in ("train", "validation", "test")
    }


def _missing_channel_quality(
    cases: Sequence[Mapping[str, object]],
    getter: Callable[[Mapping[str, object]], np.ndarray],
    channels: Sequence[int],
) -> dict[str, float]:
    errors = []
    for case in cases:
        output = getter(case)
        truth = np.asarray(case["truth"])
        cfa = np.asarray(case["cfa"])
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        for channel in channels:
            selected = interior & (cfa != channel)
            errors.append((output[channel] - truth[channel])[selected])
    return scalar_metrics(np.concatenate(errors))


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    positive = max(1, int(labels.sum()))
    negative = max(1, int((~labels).sum()))
    return {
        "false_positive_rate": float(np.sum(predicted & ~labels) / negative),
        "rejection_fraction": float(np.mean(predicted)),
        "roc_auc": float(roc_auc(scores, labels)),
        "severe_recall": float(np.sum(predicted & labels) / positive),
        "severe_count": int(labels.sum()),
        "threshold": float(threshold),
    }


def _model_dict(model) -> dict[str, object]:
    if hasattr(model, "coefficients"):
        return {
            "coefficients": model.coefficients.tolist(),
            "feature_indices": model.feature_indices.tolist(),
            "feature_names": [FEATURE_NAMES[index] for index in model.feature_indices],
            "intercept": model.intercept,
            "kind": model.kind,
            "normalizer_mean": model.normalizer.mean.tolist(),
            "normalizer_scale": model.normalizer.scale.tolist(),
            "regularization": model.regularization,
        }
    return model.root.to_dict(FEATURE_NAMES)


def _predictability(
    table: SampleTable,
    severe_tau: float,
    baseline: np.ndarray,
    target_name: str,
) -> dict[str, object]:
    harm = (table.final - table.truth) ** 2 - (baseline - table.truth) ** 2
    labels = harm > severe_tau
    train_mask = table.splits == "train"
    validation_mask = table.splits == "validation"
    test_mask = table.splits == "test"
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size > 150000:
        train_indices = train_indices[np.linspace(0, train_indices.size - 1, 150000).astype(int)]
    target = labels[train_indices].astype(np.float64)
    weights = np.where(target > 0, 20.0, np.where(harm[train_indices] > 0, 5.0, 1.0))
    all_features = np.arange(len(FEATURE_NAMES), dtype=np.int32)
    logistic = fit_logistic(
        table.features[train_indices], target, weights, all_features, 1e-3,
    )
    tree_indices = train_indices
    if tree_indices.size > 40000:
        tree_indices = tree_indices[np.linspace(0, tree_indices.size - 1, 40000).astype(int)]
    tree_target = labels[tree_indices].astype(np.float64)
    tree_weights = np.where(tree_target > 0, 20.0, np.where(harm[tree_indices] > 0, 5.0, 1.0))
    tree = fit_tree(
        table.features[tree_indices], tree_target, tree_weights,
        all_features, FEATURE_NAMES, 3, classification=True,
    )
    candidates = {
        "logistic": logistic.alpha(table.features),
        "depth3_tree": tree.alpha(table.features),
    }
    # A one-statistic rule establishes whether model complexity is buying
    # anything over the most obvious residual-outlier hypothesis.
    best_feature = max(
        range(len(FEATURE_NAMES)),
        key=lambda index: roc_auc(table.features[validation_mask, index], labels[validation_mask]),
    )
    candidates[f"threshold_{FEATURE_NAMES[best_feature]}"] = table.features[:, best_feature]
    rows = {}
    best_name = None
    best_test_recall = -1.0
    for name, scores in candidates.items():
        threshold = operating_threshold(scores[validation_mask], labels[validation_mask], 0.9)
        predicted = scores >= threshold
        predicted_values = np.where(predicted, baseline, table.final)
        metrics_function = classification_metrics if name in ("logistic", "depth3_tree") else _threshold_metrics
        row = {
            "validation": metrics_function(scores[validation_mask], labels[validation_mask], threshold),
            "test": metrics_function(scores[test_mask], labels[test_mask], threshold),
            "test_reconstruction": reconstruction_metrics(
                table.select(test_mask), predicted_values[test_mask]
            ),
        }
        rows[name] = row
        recall = row["test"]["severe_recall"]
        if recall > best_test_recall:
            best_name, best_test_recall = name, recall
    return {
        "external_detector_reference": {
            "held_out_severe_recall": 0.771,
            "notes": "previous final-output 7x7 detector; poor Hydra recall and unsafe p99",
        },
        "fitted_models": {
            "logistic": _model_dict(logistic),
            "depth3_tree": _model_dict(tree),
        },
        "models": rows,
        "best_by_untouched_test_recall_for_reporting_only": best_name,
        "severe_definition": (
            f"squared-error increase from {target_name} exceeds train-only p90 of positive increases"
        ),
        "severe_tau": severe_tau,
        "training_sample_count": int(train_indices.size),
    }


def _robust_controls(table: SampleTable) -> dict[str, object]:
    configurations: list[tuple[str, np.ndarray]] = [("none", table.correction)]
    for threshold in (0.005, 0.01, 0.02, 0.05, 0.1):
        configurations.append((f"absolute_{threshold:g}", np.clip(table.correction, -threshold, threshold)))
    rms = table.features[:, FEATURE_NAMES.index("correction_local_rms")]
    mad = table.features[:, FEATURE_NAMES.index("correction_local_mad")]
    cfa = table.features[:, FEATURE_NAMES.index("mosaic_local_variation")]
    for scale in (0.5, 1.0, 2.0, 4.0):
        for label, local in (("rms", rms), ("mad", mad), ("cfa", cfa)):
            limit = scale * local
            configurations.append((f"{label}_{scale:g}", np.clip(table.correction, -limit, limit)))
    validation = table.splits == "validation"
    test = table.splits == "test"
    rows = []
    for name, correction in configurations:
        output = np.clip(table.base + correction, 0, 1)
        rows.append({
            "name": name,
            "validation": reconstruction_metrics(table.select(validation), output[validation]),
            "test": reconstruction_metrics(table.select(test), output[test]),
        })
    selected = min(rows, key=lambda row: row["validation"]["mse"])
    return {
        "configurations": rows,
        "selected_on_validation": selected["name"],
        "selected_test": selected["test"],
    }


def _candidate_and_residual(table: SampleTable, catastrophic_tau: float) -> dict[str, object]:
    final_abs = np.abs(table.final - table.truth)
    catastrophic = final_abs > catastrophic_tau
    alternatives = np.stack((table.pass0, table.pass1, table.base, table.pass0_final))
    best_alternative = np.min(np.abs(alternatives - table.truth[None]), axis=0)
    type_a = catastrophic & (best_alternative <= 0.5 * final_abs)
    helps = (table.final - table.truth) ** 2 < (table.base - table.truth) ** 2
    harm = table.harm
    positive = harm[harm > 0]
    severe_harm_tau = float(np.percentile(positive, 90)) if positive.size else 0.0
    alpha = alpha_oracle(table.base, table.correction, table.truth)
    alpha_wide = alpha_oracle(table.base, table.correction, table.truth, -0.5, 1.5)

    def failure_types(mask: np.ndarray) -> dict[str, object]:
        selected_catastrophic = catastrophic & mask
        selected_type_a = type_a & mask
        return {
            "catastrophic_count": int(selected_catastrophic.sum()),
            "type_a_fraction": float(selected_type_a.sum() / max(1, selected_catastrophic.sum())),
            "type_b_fraction": float((selected_catastrophic.sum() - selected_type_a.sum()) / max(1, selected_catastrophic.sum())),
        }

    def alpha_distribution(mask: np.ndarray) -> dict[str, float]:
        selected = alpha[mask]
        return {
            "at_zero": float(np.mean(selected == 0)),
            "interior": float(np.mean((selected > 0) & (selected < 1))),
            "at_one": float(np.mean(selected == 1)),
            "mean": float(np.mean(selected)),
        }

    return {
        "candidate_failure_types": {
            "catastrophic_count": int(catastrophic.sum()),
            "catastrophic_threshold_abs": catastrophic_tau,
            "type_a_good_candidate_exists_count": int(type_a.sum()),
            "type_a_good_candidate_exists_fraction": float(type_a.sum() / max(1, catastrophic.sum())),
            "type_b_all_candidates_wrong_fraction": float((catastrophic.sum() - type_a.sum()) / max(1, catastrophic.sum())),
        },
        "correction_outcome": {
            "helps_fraction": float(np.mean(helps)),
            "hurts_fraction": float(np.mean(~helps)),
            "hurts_severely_fraction": float(np.mean(harm > severe_harm_tau)),
            "severe_harm_tau": severe_harm_tau,
        },
        "oracle_alpha_distribution": {
            "at_zero": float(np.mean(alpha == 0)),
            "interior": float(np.mean((alpha > 0) & (alpha < 1))),
            "at_one": float(np.mean(alpha == 1)),
            "mean": float(np.mean(alpha)),
            "wide_below_zero": float(np.mean(alpha_wide < 0)),
            "wide_above_one": float(np.mean(alpha_wide > 1)),
        },
        "failure_types_by_kind": {
            kind: failure_types(table.kinds == kind)
            for kind in sorted(set(table.kinds.tolist()))
        },
        "alpha_by_kind": {
            kind: alpha_distribution(table.kinds == kind)
            for kind in sorted(set(table.kinds.tolist()))
        },
        "reconstructions": {
            name: _split_metrics(table, _row_reconstruction(table, name))
            for name in (
                "pass0", "pass1", "pass0_final", "guide_only", "final", "candidate_oracle",
                "candidate_median", "candidate_trimmed", "oracle_alpha",
                "oracle_alpha_wide", "oracle_reject",
                "oracle_refinement_reject",
            )
        },
    }


def _patch_candidate_oracle(cases: Sequence[Mapping[str, object]], block: int = 7) -> dict[str, float]:
    errors = []
    for case in cases:
        truth = np.asarray(case["truth"])
        cfa = np.asarray(case["cfa"])
        trace = case["trace"]
        for channel, suffix in ((0, "red"), (2, "blue")):
            candidates = np.stack((
                trace[f"pass0-provisional-{suffix}"],
                trace[f"pass1-provisional-{suffix}"],
                trace[f"final-tentative-{suffix}"],
                trace[f"final-{suffix}"],
            ))
            for y in range(MARGIN, truth.shape[1] - MARGIN, block):
                y2 = min(truth.shape[1] - MARGIN, y + block)
                for x in range(MARGIN, truth.shape[2] - MARGIN, block):
                    x2 = min(truth.shape[2] - MARGIN, x + block)
                    missing = cfa[y:y2, x:x2] != channel
                    if not missing.any():
                        continue
                    candidate_block = candidates[:, y:y2, x:x2][:, missing]
                    truth_block = truth[channel, y:y2, x:x2][missing]
                    winner = int(np.argmin(np.sum((candidate_block - truth_block) ** 2, axis=1)))
                    errors.append(candidate_block[winner] - truth_block)
    return scalar_metrics(np.concatenate(errors))


def _green_directional_analysis(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = []
    for case in cases:
        truth = np.asarray(case["truth"])[1]
        cfa = np.asarray(case["cfa"])
        trace = case["trace"]
        interior = np.zeros(cfa.shape, dtype=bool)
        interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        selected = interior & (cfa != 1)
        pass0 = np.stack([trace[f"pass0-green-direction-{index}"] for index in range(8)])
        pass1 = np.stack([trace[f"pass1-green-direction-{index}"] for index in range(8)])
        current0 = np.asarray(trace["pass0-green"])
        current = np.asarray(trace["pass1-green"])
        oracle0 = np.take_along_axis(
            pass0, np.argmin(np.abs(pass0 - truth[None]), axis=0)[None], axis=0,
        )[0]
        oracle1 = np.take_along_axis(
            pass1, np.argmin(np.abs(pass1 - truth[None]), axis=0)[None], axis=0,
        )[0]
        measured_green = np.asarray(case["mosaic"]) * (cfa == 1)
        measured_mask = (cfa == 1).astype(np.float64)
        nearby_green = ndimage.uniform_filter(measured_green, size=5, mode="reflect") / np.maximum(
            ndimage.uniform_filter(measured_mask, size=5, mode="reflect"), 1e-8,
        )
        median0 = np.median(pass0, axis=0)
        median1 = np.median(pass1, axis=0)
        combined = np.concatenate((pass0, pass1), axis=0)
        combined_median = np.median(combined, axis=0)
        rows.append({
            "case": case,
            "selected": selected,
            "truth": truth,
            "current0": current0,
            "current": current,
            "oracle0": oracle0,
            "oracle1": oracle1,
            "spread1": np.ptp(pass1, axis=0),
            "median0": median0,
            "median1": median1,
            "trimmed0": np.mean(np.sort(pass0, axis=0)[1:-1], axis=0),
            "trimmed1": np.mean(np.sort(pass1, axis=0)[1:-1], axis=0),
            "combined_median": combined_median,
            "medoid0": np.take_along_axis(
                pass0, np.argmin(np.abs(pass0 - median0[None]), axis=0)[None], axis=0,
            )[0],
            "local_green_selector0": np.take_along_axis(
                pass0, np.argmin(np.abs(pass0 - nearby_green[None]), axis=0)[None], axis=0,
            )[0],
            "pass1": pass1,
            "pass0": pass0,
        })

    train_error = np.concatenate([
        np.abs(row["current"] - row["truth"])[row["selected"]]
        for row in rows if row["case"]["split"] == "train"
    ])
    catastrophic_tau = max(0.02, float(np.percentile(train_error, 99)))

    def metrics(split: str, key: str) -> dict[str, float]:
        return scalar_metrics(np.concatenate([
            (row[key] - row["truth"])[row["selected"]]
            for row in rows if row["case"]["split"] == split
        ]))

    catastrophic_parts = []
    type_a_parts = []
    spread_parts = []
    for row in rows:
        selected = row["selected"]
        current_error = np.abs(row["current"] - row["truth"])[selected]
        best_error = np.minimum(
            np.abs(row["oracle0"] - row["truth"])[selected],
            np.abs(row["oracle1"] - row["truth"])[selected],
        )
        catastrophic = current_error > catastrophic_tau
        catastrophic_parts.append(catastrophic)
        type_a_parts.append(catastrophic & (best_error <= 0.5 * current_error))
        spread_parts.append(row["spread1"][selected])
    catastrophic = np.concatenate(catastrophic_parts)
    type_a = np.concatenate(type_a_parts)
    spread = np.concatenate(spread_parts)

    patch_errors: dict[str, list[np.ndarray]] = {"pass0": [], "pass1": [], "combined": []}
    for row in rows:
        truth = row["truth"]
        cfa = np.asarray(row["case"]["cfa"])
        for y in range(MARGIN, truth.shape[0] - MARGIN, 7):
            y2 = min(truth.shape[0] - MARGIN, y + 7)
            for x in range(MARGIN, truth.shape[1] - MARGIN, 7):
                x2 = min(truth.shape[1] - MARGIN, x + 7)
                missing = cfa[y:y2, x:x2] != 1
                truth_block = truth[y:y2, x:x2][missing]
                for name, candidates in (
                    ("pass0", row["pass0"]),
                    ("pass1", row["pass1"]),
                    ("combined", np.concatenate((row["pass0"], row["pass1"]), axis=0)),
                ):
                    candidate_block = candidates[:, y:y2, x:x2][:, missing]
                    winner = int(np.argmin(np.sum((candidate_block - truth_block) ** 2, axis=1)))
                    patch_errors[name].append(candidate_block[winner] - truth_block)
    return {
        "catastrophic_threshold_abs": catastrophic_tau,
        "catastrophic_count": int(catastrophic.sum()),
        "good_direction_exists_fraction": float(type_a.sum() / max(1, catastrophic.sum())),
        "all_directions_wrong_fraction": float((catastrophic.sum() - type_a.sum()) / max(1, catastrophic.sum())),
        "directional_spread_auc_for_catastrophic_green": float(roc_auc(spread, catastrophic)),
        "pixel_oracles": {
            split: {
                "current_pass1_green": metrics(split, "current"),
                "pass0_direction_oracle": metrics(split, "oracle0"),
                "pass1_direction_oracle": metrics(split, "oracle1"),
            }
            for split in ("train", "validation", "test")
        },
        "observable_robust_controls": {
            split: {
                name: metrics(split, name)
                for name in (
                    "current0", "current", "median0", "median1", "trimmed0",
                    "trimmed1", "combined_median", "medoid0", "local_green_selector0",
                )
            }
            for split in ("train", "validation", "test")
        },
        "direction_patch_oracles_7x7": {
            name: scalar_metrics(np.concatenate(values))
            for name, values in patch_errors.items()
        },
    }


def _residual_diagnostics(table: SampleTable, catastrophic_tau: float) -> dict[str, object]:
    catastrophic = np.abs(table.final - table.truth) > catastrophic_tau
    names = (
        "correction_abs", "correction_to_rms", "correction_to_mad",
        "correction_to_cfa_variation", "raw_residual_local_rms",
        "raw_residual_sign_consistency", "pass1_to_tentative_abs",
        "pass0_to_pass1_abs",
    )
    result = {}
    for name in names:
        values = table.features[:, FEATURE_NAMES.index(name)]
        result[name] = {
            "catastrophic_median": float(np.median(values[catastrophic])) if catastrophic.any() else 0.0,
            "catastrophic_p90": float(np.percentile(values[catastrophic], 90)) if catastrophic.any() else 0.0,
            "ordinary_median": float(np.median(values[~catastrophic])),
            "ordinary_p90": float(np.percentile(values[~catastrophic], 90)),
            "roc_auc_for_catastrophic_final_error": float(roc_auc(values, catastrophic)),
        }
    return result


def _error_growth(table: SampleTable, catastrophic_tau: float) -> dict[str, object]:
    errors = {
        name: np.abs(_row_reconstruction(table, name) - table.truth)
        for name in ("pass0", "pass1", "guide_only", "final")
    }
    catastrophic = errors["final"] > catastrophic_tau
    rows = {}
    previous = "pass0"
    for current in ("pass1", "guide_only", "final"):
        growth = errors[current] - errors[previous]
        rows[f"{previous}_to_{current}"] = {
            "catastrophic_mean_abs_error_growth": float(np.mean(growth[catastrophic])) if catastrophic.any() else 0.0,
            "ordinary_mean_abs_error_growth": float(np.mean(growth[~catastrophic])),
            "catastrophic_fraction_worsened": float(np.mean(growth[catastrophic] > 0)) if catastrophic.any() else 0.0,
        }
        previous = current
    return rows


def _source_rows(cases: Sequence[Mapping[str, object]], table: SampleTable) -> list[dict[str, object]]:
    rows = []
    for source_id in sorted({str(case["source_id"]) for case in cases}):
        selected_cases = [case for case in cases if case["source_id"] == source_id]
        mask = table.source_ids == source_id
        subset = table.select(mask)
        rows.append({
            "source_id": source_id,
            "split": str(selected_cases[0]["split"]),
            "kind": str(selected_cases[0]["kind"]),
            "markesteijn": _missing_channel_quality(
                selected_cases, lambda case: _rgb(case, "markesteijn"), (0, 2),
            ),
            "pass0_guide_final": _missing_channel_quality(
                selected_cases, lambda case: _rgb(case, "pass0_guide_final"), (0, 2),
            ),
            "guide_only": reconstruction_metrics(subset, subset.base),
            "final": reconstruction_metrics(subset, subset.final),
            "oracle_alpha": reconstruction_metrics(subset, _row_reconstruction(subset, "oracle_alpha")),
            "oracle_reject": reconstruction_metrics(subset, _row_reconstruction(subset, "oracle_reject")),
        })
    return rows


def _diagnostic_map(case: Mapping[str, object], path: Path) -> None:
    truth = np.asarray(case["truth"])
    final = _rgb(case, "final")
    guide = _rgb(case, "guide_only")
    correction = np.stack((
        case["trace"]["final-correction-red"],
        np.zeros(truth.shape[1:]),
        case["trace"]["final-correction-blue"],
    ))
    direction_spread = np.ptp(np.stack([
        case["trace"][f"pass1-green-direction-{index}"] for index in range(8)
    ]), axis=0)
    alpha = np.stack((
        alpha_oracle(guide[0], correction[0], truth[0]),
        np.ones(truth.shape[1:]),
        alpha_oracle(guide[2], correction[2], truth[2]),
    ))
    harmful = ((final - truth) ** 2 > (guide - truth) ** 2).astype(np.float64)

    def rgb(values):
        encoded = np.moveaxis(srgb_encode(values), 0, -1)
        return Image.fromarray(np.rint(encoded * 255).astype(np.uint8), "RGB")

    def error(values):
        magnitude = np.mean(np.abs(values), axis=0)
        mapped = np.clip(magnitude * 8, 0, 1)
        image = np.stack((mapped, np.sqrt(mapped), 1 - mapped), axis=-1)
        return Image.fromarray(np.rint(image * 255).astype(np.uint8), "RGB")

    panels = (
        ("truth", rgb(truth)),
        ("final MLRI", rgb(final)),
        ("final error x8", error(final - truth)),
        ("guide error x8", error(guide - truth)),
        ("green direction spread", error(np.stack((direction_spread,) * 3))),
        ("correction x8", error(correction)),
        ("oracle alpha", rgb(alpha)),
        ("harmful correction", rgb(harmful)),
    )
    scale = max(1, min(4, 240 // truth.shape[1]))
    width = truth.shape[2] * scale
    height = truth.shape[1] * scale
    output = Image.new("RGB", (len(panels) * width, height + 24), "white")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        panel = panel.resize((width, height), Image.Resampling.NEAREST)
        output.paste(panel, (index * width, 24))
        draw.text((index * width + 3, 5), label, fill=(15, 15, 15), font=font)
    output.save(path, format="PNG", compress_level=9)


def _runtime(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stage_names = (
        "pass0-guide", "pass0-green", "pass0-chroma",
        "pass1-guide", "pass1-green", "pass1-chroma", "final-red-blue",
    )
    totals = {name: sum(float(case["timings"][name]) for case in cases) for name in stage_names}
    total = sum(totals.values())
    return {
        "aggregate_seconds": totals,
        "fraction_of_traced_mlri_time": {
            name: value / total for name, value in totals.items()
        },
        "markesteijn_aggregate_seconds": sum(float(case["timings"]["markesteijn"]) for case in cases),
        "traced_mlri_aggregate_seconds": total,
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(_json_safe(value)))


def run(runner: Path, starfield: Path, output: Path) -> dict[str, object]:
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(starfield)
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-mlri-internal-") as temporary:
        root = Path(temporary)
        for index, case in enumerate(cases):
            work = root / f"{index:03d}"
            work.mkdir()
            _run_case(case, runner, work)
            print(f"[{index + 1:02d}/{len(cases):02d}] {case['case_id']}", flush=True)

    table = build_sample_table(cases)
    train = table.select(table.splits == "train")
    train_final_abs = np.abs(train.final - train.truth)
    catastrophic_tau = max(0.02, float(np.percentile(train_final_abs, 99)))
    positive_harm = train.harm[train.harm > 0]
    severe_tau = float(np.percentile(positive_harm, 90)) if positive_harm.size else 0.0
    refinement_harm = (
        (train.final - train.truth) ** 2
        - (train.pass0_final - train.truth) ** 2
    )
    positive_refinement_harm = refinement_harm[refinement_harm > 0]
    refinement_severe_tau = (
        float(np.percentile(positive_refinement_harm, 90))
        if positive_refinement_harm.size else 0.0
    )

    stage_quality = {
        split: {
            stage: _quality(
                [case for case in cases if case["split"] == split],
                lambda case, stage=stage: _rgb(case, stage),
            )
            for stage in (
                "markesteijn", "pass0", "pass1", "pass0_guide_final",
                "guide_only", "final", "oracle_green",
            )
        }
        for split in ("train", "validation", "test")
    }
    candidate_residual = _candidate_and_residual(table, catastrophic_tau)
    results = {
        "format": "rawtherapee-xtrans-mlri-internal-results-v1",
        "scope": "corrected blue diagonal guides, two-pass, final-only MLRI",
        "stage_quality": stage_quality,
        "source_rows": _source_rows(cases, table),
        "candidate_and_residual": candidate_residual,
        "green_directional_candidates": _green_directional_analysis(cases),
        "patch_candidate_oracle_7x7": _patch_candidate_oracle(cases),
        "error_growth": _error_growth(table, catastrophic_tau),
        "residual_diagnostics": _residual_diagnostics(table, catastrophic_tau),
        "predictability": {
            "final_residual": _predictability(
                table, severe_tau, table.base, "applying the final residual correction",
            ),
            "second_pass_refinement": _predictability(
                table, refinement_severe_tau, table.pass0_final,
                "replacing the pass-0 guide reconstruction with pass 1",
            ),
        },
        "robust_clipping": _robust_controls(table),
        "runtime": _runtime(cases),
        "oracle_substitutions": {
            "oracle_green": "true green guide passed through the real final guided MLRI and residual interpolation",
            "oracle_residual": "analytical truth-minus-real-guide residual; exact R/B by definition, retained only as a sensitivity ceiling",
            "oracle_correction": "per-sample alpha in [0,1] applied to the real final correction",
            "missing_red_blue_quality": {
                split: {
                    "final": _missing_channel_quality(
                        [case for case in cases if case["split"] == split],
                        lambda case: _rgb(case, "final"), (0, 2),
                    ),
                    "oracle_green": _missing_channel_quality(
                        [case for case in cases if case["split"] == split],
                        lambda case: _rgb(case, "oracle_green"), (0, 2),
                    ),
                    "pass0_guide_final": _missing_channel_quality(
                        [case for case in cases if case["split"] == split],
                        lambda case: _rgb(case, "pass0_guide_final"), (0, 2),
                    ),
                    "green_stage": _missing_channel_quality(
                        [case for case in cases if case["split"] == split],
                        lambda case: _rgb(case, "final"), (1,),
                    ),
                }
                for split in ("train", "validation", "test")
            },
        },
    }

    selected_map_ids = (
        "hubble-0", "nasa-hydra-starfield-0", "white-impulse",
        "brick-0", "page-0", "control-saturated_red_gray",
    )
    image_rows = []
    for case_id in selected_map_ids:
        case = next(case for case in cases if case["case_id"] == case_id)
        path = output / f"map-{case_id}.png"
        _diagnostic_map(case, path)
        image_rows.append({"file": path.name, "sha256": _sha256(path)})

    dataset_path = output / "dataset.json"
    results_path = output / "results.json"
    _write(dataset_path, dataset_manifest(cases))
    _write(results_path, results)
    manifest = {
        "format": "rawtherapee-xtrans-mlri-internal-manifest-v1",
        "dataset": {"file": dataset_path.name, "sha256": _sha256(dataset_path)},
        "results": {"file": results_path.name, "sha256": _sha256(results_path)},
        "images": image_rows,
        "runner": str(runner.name),
    }
    _write(output / "manifest.json", manifest)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = run(args.runner.resolve(), args.starfield.resolve(), args.output)
    print(f"results: {args.output / 'results.json'}")
    print(f"MLRI traced seconds: {results['runtime']['traced_mlri_aggregate_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the deterministic Markesteijn/MLRI selector and blender experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

# Parallel BLAS reductions can change fitted coefficients in their final bits.
# Freeze the development analysis before NumPy/SciPy load; the native runner
# separately receives its intended four-thread OpenMP setting in _run_pair().
for _thread_variable in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from tools.xtrans_alias.analysis import CANONICAL_XTRANS, canonical_json_bytes
from tools.xtrans_oracle.generate import mosaic, srgb_encode, synthetic_scenes
from .analysis import (
    FEATURE_DECIMALS,
    MARGIN,
    LinearModel,
    PatchTable,
    TreeModel,
    alpha_fragmentation,
    apply_block_alpha,
    build_patch_table,
    calibration_metrics,
    classification_metrics,
    dense_quality_metrics,
    feature_indices,
    feature_maps,
    fit_logistic,
    fit_ridge,
    fit_tree,
    quality_metrics,
    sse_for_alpha,
)
from .dataset import NATURAL_SOURCES, dataset_manifest, iter_crops, load_sources


REGULARIZATIONS = (1e-4, 1e-2, 1.0, 100.0)
ALPHA_DECIMALS = 3
FEATURE_ABLATIONS = {
    "disagreement_only": ("A",),
    "mosaic_gradients_only": ("B",),
    "candidate_gradients_only": ("C",),
    "chroma_saturation_only": ("D",),
    "texture_frequency_only": ("E",),
    "disagreement_plus_gradients": ("A", "B", "C"),
    "disagreement_plus_chroma": ("A", "D"),
    "all_core": ("A", "B", "C", "D", "E"),
    "all_core_plus_phase": ("A", "B", "C", "D", "E", "G"),
}
COLORS = ((38, 111, 181), (225, 126, 34), (43, 145, 84), (190, 38, 43))
SOURCE_SPLITS = {str(row["id"]): str(row["split"]) for row in NATURAL_SOURCES}


@dataclass
class Candidate:
    name: str
    family: str
    mode: str
    groups: tuple[str, ...]
    hyperparameter: float | int | str
    model: object
    config: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "undefined"
        if math.isinf(number):
            return "infinity" if number > 0 else "-infinity"
        return number
    if isinstance(value, np.integer):
        return int(value)
    return value


def _run_pair(runner: Path, truth: np.ndarray, work: Path) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, np.ndarray]:
    scalar, cfa = mosaic(truth)
    input_path = work / "mosaic.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            str(runner), "run-pair", str(input_path), str(work),
            str(truth.shape[2]), str(truth.shape[1]),
        ],
        env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"native pair runner failed:\n{completed.stdout}\n{completed.stderr}")
    timings = {}
    for line in completed.stdout.splitlines():
        name, seconds = line.split("\t")
        timings[name] = float(seconds)
    outputs = {}
    pixels = truth.shape[1] * truth.shape[2]
    for name in ("markesteijn", "mlri-final"):
        values = np.fromfile(work / f"{name}.f32le", dtype="<f4")
        if values.size != 3 * pixels:
            raise RuntimeError(f"incorrect output size for {name}")
        outputs[name] = values.astype(np.float64).reshape(3, truth.shape[1], truth.shape[2])
    return outputs, timings, scalar, cfa


def _split(table: PatchTable, name: str) -> PatchTable:
    mask = np.asarray([SOURCE_SPLITS[source] == name for source in table.source_ids])
    return table.subset(mask)


def _indices(table: PatchTable, groups: Sequence[str]) -> np.ndarray:
    return feature_indices(table, groups)


def predict(candidate: Candidate, table: PatchTable) -> np.ndarray:
    if candidate.family == "constant":
        alpha = np.full(table.features.shape[0], float(candidate.model))
    elif candidate.family == "threshold":
        index = int(candidate.model["feature_index"])
        threshold = float(candidate.model["threshold"])
        above = table.features[:, index] >= threshold
        alpha = above.astype(np.float64) if candidate.model["direction"] == "above" else (~above).astype(np.float64)
    elif isinstance(candidate.model, (LinearModel, TreeModel)):
        alpha = candidate.model.alpha(table.features)
        if candidate.mode == "hard":
            alpha = (alpha >= 0.5).astype(np.float64)
    else:
        raise TypeError(f"unknown candidate model {type(candidate.model)!r}")
    return np.round(alpha, decimals=ALPHA_DECIMALS)


def _quality_row(candidate: Candidate, tables: Mapping[str, PatchTable]) -> dict[str, object]:
    result = {
        "family": candidate.family,
        "feature_groups": list(candidate.groups),
        "hyperparameter": candidate.hyperparameter,
        "mode": candidate.mode,
        "name": candidate.name,
        "splits": {},
    }
    for split, table in tables.items():
        alpha = predict(candidate, table)
        metrics = quality_metrics(table, alpha)
        metrics["classification"] = classification_metrics(table, alpha)
        result["splits"][split] = metrics
    result["validation_minus_test_db"] = (
        result["splits"]["validation"]["psnr_db"]
        - result["splits"]["test"]["psnr_db"]
    )
    return result


def _best_by_validation(candidates: Sequence[Candidate], validation: PatchTable) -> Candidate:
    return max(candidates, key=lambda candidate: quality_metrics(validation, predict(candidate, validation))["psnr_db"])


def _fixed_blend(train: PatchTable) -> Candidate:
    alpha = float(np.clip(-np.sum(train.cross) / max(np.sum(train.quadratic), 1e-30), 0, 1))
    return Candidate(
        "fixed-global-blend", "constant", "blend", (), alpha, alpha,
        {"family": "constant"},
    )


def _threshold_candidates(train: PatchTable, validation: PatchTable) -> Candidate:
    feature_name = "disagreement_rgb_rms"
    index = train.feature_names.index(feature_name)
    candidates = []
    for quantile in np.linspace(0.05, 0.95, 37):
        threshold = float(np.quantile(train.features[:, index], quantile))
        for direction in ("above", "below"):
            model = {"feature_index": index, "threshold": threshold, "direction": direction}
            candidates.append(Candidate(
                "disagreement-threshold", "threshold", "hard", ("A",),
                f"q={quantile:.3f},{direction}", model,
                {"family": "threshold", "quantile": float(quantile), "direction": direction},
            ))
    return _best_by_validation(candidates, validation)


def _linear_candidates(
    family: str,
    mode: str,
    groups: tuple[str, ...],
    train: PatchTable,
    validation: PatchTable,
) -> Candidate:
    indices = _indices(train, groups)
    candidates = []
    for regularization in REGULARIZATIONS:
        if family == "ridge":
            model = fit_ridge(
                train.features, train.oracle_alpha, train.quadratic,
                indices, regularization,
            )
        elif family == "logistic-blend":
            model = fit_logistic(
                train.features, train.oracle_alpha, train.quadratic,
                indices, regularization, soft_target=True,
            )
        elif family == "logistic-hard":
            model = fit_logistic(
                train.features, train.labels, np.abs(train.margins),
                indices, regularization,
            )
        else:
            raise ValueError(family)
        candidates.append(Candidate(
            family, family, mode, groups, regularization, model,
            {"family": family, "groups": list(groups), "regularization": regularization},
        ))
    return _best_by_validation(candidates, validation)


def _tree_candidates(
    family: str,
    mode: str,
    groups: tuple[str, ...],
    train: PatchTable,
    validation: PatchTable,
) -> Candidate:
    indices = _indices(train, groups)
    candidates = []
    for depth in (2, 3, 4):
        classification = family == "tree-hard"
        target = train.labels if classification else train.oracle_alpha
        weights = np.abs(train.margins) if classification else train.quadratic
        model = fit_tree(
            train.features, target, weights, indices, train.feature_names,
            depth, classification=classification,
        )
        candidates.append(Candidate(
            family, family, mode, groups, depth, model,
            {"family": family, "groups": list(groups), "depth": depth},
        ))
    return _best_by_validation(candidates, validation)


def fit_config(config: Mapping[str, object], train: PatchTable) -> Candidate:
    family = str(config["family"])
    groups = tuple(config.get("groups", ()))
    if family == "constant":
        return _fixed_blend(train)
    if family == "threshold":
        index = train.feature_names.index("disagreement_rgb_rms")
        threshold = float(np.quantile(train.features[:, index], float(config["quantile"])))
        model = {"feature_index": index, "threshold": threshold, "direction": config["direction"]}
        return Candidate("disagreement-threshold", family, "hard", ("A",), config["quantile"], model, dict(config))
    indices = _indices(train, groups)
    if family == "ridge":
        model = fit_ridge(train.features, train.oracle_alpha, train.quadratic, indices, float(config["regularization"]))
        mode = "blend"
    elif family == "logistic-blend":
        model = fit_logistic(train.features, train.oracle_alpha, train.quadratic, indices, float(config["regularization"]), soft_target=True)
        mode = "blend"
    elif family == "logistic-hard":
        model = fit_logistic(train.features, train.labels, np.abs(train.margins), indices, float(config["regularization"]))
        mode = "hard"
    elif family in ("tree-hard", "tree-blend"):
        classification = family == "tree-hard"
        model = fit_tree(
            train.features, train.labels if classification else train.oracle_alpha,
            np.abs(train.margins) if classification else train.quadratic,
            indices, train.feature_names, int(config["depth"]),
            classification=classification,
        )
        mode = "hard" if classification else "blend"
    else:
        raise ValueError(family)
    return Candidate(family, family, mode, groups, config.get("regularization", config.get("depth")), model, dict(config))


def _serialize_candidate(candidate: Candidate, feature_names: Sequence[str]) -> dict[str, object]:
    result = {
        "config": candidate.config,
        "family": candidate.family,
        "feature_groups": list(candidate.groups),
        "mode": candidate.mode,
        "name": candidate.name,
    }
    if isinstance(candidate.model, LinearModel):
        selected_names = [feature_names[index] for index in candidate.model.feature_indices]
        result["parameters"] = {
            "coefficients": dict(zip(selected_names, map(float, candidate.model.coefficients))),
            "intercept": candidate.model.intercept,
            "normalization_mean": dict(zip(selected_names, map(float, candidate.model.normalizer.mean))),
            "normalization_scale": dict(zip(selected_names, map(float, candidate.model.normalizer.scale))),
        }
    elif isinstance(candidate.model, TreeModel):
        result["parameters"] = candidate.model.root.to_dict(candidate.model.feature_names)
    else:
        result["parameters"] = candidate.model
    return result


def _oracle_rows(tables: Mapping[str, PatchTable]) -> dict[str, object]:
    result = {}
    for split, table in tables.items():
        result[split] = {
            "markesteijn": quality_metrics(table, np.zeros(table.features.shape[0])),
            "mlri": quality_metrics(table, np.ones(table.features.shape[0])),
            "hard_oracle": quality_metrics(table, table.labels),
            "convex_oracle": quality_metrics(table, table.oracle_alpha),
        }
    return result


def _headroom(row: dict[str, object], oracle: dict[str, object]) -> dict[str, float]:
    baseline = float(oracle["markesteijn"]["psnr_db"])
    practical = float(row["psnr_db"])
    hard = float(oracle["hard_oracle"]["psnr_db"])
    convex = float(oracle["convex_oracle"]["psnr_db"])
    return {
        "gain_over_markesteijn_db": practical - baseline,
        "hard_headroom_recovered": (practical - baseline) / max(hard - baseline, 1e-12),
        "convex_headroom_recovered": (practical - baseline) / max(convex - baseline, 1e-12),
    }


def _gate_candidate(
    candidate: Candidate,
    validation: PatchTable,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    base = predict(candidate, validation)
    mark = quality_metrics(validation, np.zeros(base.size))
    convex_oracle = quality_metrics(validation, validation.oracle_alpha)
    mark_patch_rms = np.sqrt(validation.mark_sse / (3 * validation.pixels))
    rows = []
    best = None
    for lower in np.linspace(0, 1, 21):
        for upper in (0.75, 0.90, 1.01):
            if lower >= upper:
                continue
            gated = base.copy()
            gated[gated < lower] = 0
            gated[gated >= upper] = 1
            metrics = quality_metrics(validation, gated)
            patch_rms = np.sqrt(
                np.maximum(0.0, sse_for_alpha(validation, gated))
                / (3 * validation.pixels)
            )
            gain = metrics["psnr_db"] - mark["psnr_db"]
            row = {
                "lower": float(lower), "upper": float(upper),
                "coverage": metrics["coverage_nonzero"],
                "convex_headroom_recovered": gain / max(
                    convex_oracle["psnr_db"] - mark["psnr_db"], 1e-12
                ),
                "gain_db": gain,
                "p95": metrics["patch_rms"]["p95"],
                "worst_paired_patch_regression": float(
                    np.max(patch_rms - mark_patch_rms)
                ),
            }
            rows.append(row)
            safe = metrics["patch_rms"]["p95"] <= mark["patch_rms"]["p95"] * 1.001
            key = metrics["psnr_db"] if safe else -math.inf
            if best is None or key > best[0]:
                best = (key, row)
    assert best is not None
    if not math.isfinite(best[0]):
        return {"lower": 1.01, "upper": 1.01}, rows
    return {"lower": best[1]["lower"], "upper": best[1]["upper"]}, rows


def _apply_gate(alpha: np.ndarray, gate: Mapping[str, float]) -> np.ndarray:
    result = np.asarray(alpha, dtype=np.float64).copy()
    result[result < gate["lower"]] = 0
    result[result >= gate["upper"]] = 1
    return result


def _cross_validation(
    development: PatchTable,
    config: Mapping[str, object],
    source_groups: Mapping[str, str],
) -> list[dict[str, object]]:
    groups = sorted({source_groups[source] for source in development.source_ids})
    folds = [groups[index::4] for index in range(4)]
    rows = []
    for fold_index, held_out_groups in enumerate(folds):
        held_out = sorted(
            source for source in set(development.source_ids)
            if source_groups[source] in held_out_groups
        )
        validation_mask = np.isin(development.source_ids, held_out)
        train = development.subset(~validation_mask)
        validation = development.subset(validation_mask)
        candidate = fit_config(config, train)
        mark = quality_metrics(validation, np.zeros(validation.features.shape[0]))
        practical = quality_metrics(validation, predict(candidate, validation))
        oracle = quality_metrics(validation, validation.oracle_alpha)
        rows.append({
            "fold": fold_index,
            "held_out_groups": held_out_groups,
            "held_out_sources": held_out,
            "gain_db": practical["psnr_db"] - mark["psnr_db"],
            "convex_headroom_recovered": (
                (practical["psnr_db"] - mark["psnr_db"])
                / max(oracle["psnr_db"] - mark["psnr_db"], 1e-12)
            ),
        })
    return rows


def _extend_alpha_border(field: np.ndarray) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    interior = result[MARGIN:-MARGIN, MARGIN:-MARGIN]
    result[:MARGIN, MARGIN:-MARGIN] = interior[0]
    result[-MARGIN:, MARGIN:-MARGIN] = interior[-1]
    result[:, :MARGIN] = result[:, MARGIN : MARGIN + 1]
    result[:, -MARGIN:] = result[:, -MARGIN - 1 : -MARGIN]
    return result


def _dense_alpha(candidate: Candidate, cases: Sequence[Mapping[str, object]], window: int) -> list[np.ndarray]:
    result = []
    for case in cases:
        maps, _ = feature_maps(case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"], window=window)
        matrix = np.round(
            np.column_stack([maps[name].ravel() for name in maps]),
            decimals=FEATURE_DECIMALS,
        )
        proxy = type("DenseTable", (), {"features": matrix})()
        if candidate.family == "constant":
            alpha = np.full(matrix.shape[0], float(candidate.model))
        elif candidate.family == "threshold":
            index = int(candidate.model["feature_index"])
            above = matrix[:, index] >= float(candidate.model["threshold"])
            alpha = above.astype(np.float64) if candidate.model["direction"] == "above" else (~above).astype(np.float64)
        else:
            alpha = candidate.model.alpha(proxy.features)
            if candidate.mode == "hard":
                alpha = (alpha >= 0.5).astype(np.float64)
        alpha = np.round(alpha, decimals=ALPHA_DECIMALS)
        result.append(alpha.reshape(np.asarray(case["mosaic"]).shape))
    return result


def _spatial_variants(
    candidate: Candidate,
    gate: Mapping[str, float],
    all_cases: Sequence[Mapping[str, object]],
    split_cases: Sequence[Mapping[str, object]],
    split_table: PatchTable,
    split_name: str,
) -> dict[str, tuple[list[np.ndarray], dict[str, object]]]:
    base_block = _apply_gate(predict(candidate, split_table), gate)
    all_block_fields = apply_block_alpha(all_cases, split_table, base_block)
    split_indices = [index for index, case in enumerate(all_cases) if case["split"] == split_name]
    block_fields = [_extend_alpha_border(all_block_fields[index]) for index in split_indices]
    dense_fields = [_extend_alpha_border(field) for field in _dense_alpha(candidate, split_cases, split_table.block_size)]
    fields_by_name = {
        f"constant_{split_table.block_size}x{split_table.block_size}_blocks": block_fields,
        "block_alpha_box_7": [ndimage.uniform_filter(field, 7, mode="reflect") for field in block_fields],
        "block_alpha_median_7": [ndimage.median_filter(field, 7, mode="reflect") for field in block_fields],
        "per_pixel": [_apply_gate(field, gate) for field in dense_fields],
        "per_pixel_box_7": [ndimage.uniform_filter(_apply_gate(field, gate), 7, mode="reflect") for field in dense_fields],
        "per_pixel_box_15": [ndimage.uniform_filter(_apply_gate(field, gate), 15, mode="reflect") for field in dense_fields],
    }
    return {
        name: (
            fields,
            {
                "metrics": dense_quality_metrics(split_cases, fields, tail_block_size=7),
                "fragmentation": alpha_fragmentation(fields),
            },
        )
        for name, fields in fields_by_name.items()
    }


def _per_source_rows(
    table: PatchTable,
    practical_alpha: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for source in sorted(set(table.source_ids)):
        mask = table.source_ids == source
        selected = table.subset(mask)
        alpha = np.asarray(practical_alpha)[mask]
        mark = quality_metrics(selected, np.zeros(alpha.size))
        mlri = quality_metrics(selected, np.ones(alpha.size))
        hard = quality_metrics(selected, selected.labels)
        convex = quality_metrics(selected, selected.oracle_alpha)
        practical = quality_metrics(selected, alpha)
        rows.append({
            "source": source, "markesteijn": mark, "mlri": mlri,
            "hard_oracle": hard, "convex_oracle": convex,
            "practical": practical, "headroom": _headroom(practical, {
                "markesteijn": mark, "hard_oracle": hard, "convex_oracle": convex,
            }),
        })
    return rows


def _benchmark_inference(
    candidate: Candidate,
    cases: Sequence[Mapping[str, object]],
    window: int,
) -> tuple[float, float]:
    feature_time = 0.0
    selector_time = 0.0
    for case in cases:
        started = time.perf_counter()
        maps, _ = feature_maps(
            case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"],
            window=window,
        )
        matrix = np.round(
            np.column_stack([maps[name].ravel() for name in maps]),
            decimals=FEATURE_DECIMALS,
        )
        feature_time += time.perf_counter() - started
        started = time.perf_counter()
        if candidate.family == "constant":
            np.full(matrix.shape[0], float(candidate.model))
        elif candidate.family == "threshold":
            index = int(candidate.model["feature_index"])
            matrix[:, index] >= float(candidate.model["threshold"])
        else:
            candidate.model.alpha(matrix)
        selector_time += time.perf_counter() - started
    return feature_time, selector_time


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _bar_plot(title: str, rows: Sequence[tuple[str, float]], path: Path) -> None:
    width, height = 900, 90 + 42 * len(rows)
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill=(20, 20, 20), font=_font())
    minimum = min(0.0, min(value for _, value in rows))
    maximum = max(0.01, max(value for _, value in rows))
    left, span = 260, 560
    zero = left + int((-minimum) / (maximum - minimum) * span)
    for index, (name, value) in enumerate(rows):
        y = 48 + 42 * index
        draw.text((12, y + 8), name, fill=(30, 30, 30), font=_font())
        end = left + int((value - minimum) / (maximum - minimum) * span)
        draw.rectangle((min(zero, end), y, max(zero, end), y + 24), fill=COLORS[index % len(COLORS)])
        draw.text((830, y + 8), f"{value:+.4f} dB", fill=(30, 30, 30), font=_font())
    image.save(path, format="PNG", compress_level=9)


def _gating_plot(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    image = Image.new("RGB", (850, 480), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 40, 810, 420
    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30))
    min_gain = min(-0.01, min(float(row["gain_db"]) for row in rows))
    max_gain = max(0.01, max(float(row["gain_db"]) for row in rows))
    for row in rows:
        x = left + float(row["coverage"]) * (right - left)
        y = bottom - (float(row["gain_db"]) - min_gain) / (max_gain - min_gain) * (bottom - top)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS[0])
    draw.text((12, 10), "Validation quality gain versus modified coverage", fill=(20, 20, 20), font=_font())
    draw.text((left, bottom + 12), "0%", fill=(20, 20, 20), font=_font())
    draw.text((right - 25, bottom + 12), "100%", fill=(20, 20, 20), font=_font())
    image.save(path, format="PNG", compress_level=9)


def _calibration_plot(calibration: Mapping[str, object], path: Path) -> None:
    image = Image.new("RGB", (600, 600), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 50, 550, 530
    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30))
    draw.line((left, bottom, right, top), fill=(150, 150, 150), width=2)
    for row in calibration["bins"]:
        if row["count"]:
            x = left + float(row["predicted_mean"]) * (right - left)
            y = bottom - float(row["oracle_mean"]) * (bottom - top)
            radius = 3 + min(10, math.sqrt(int(row["count"])) / 3)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[1])
    draw.text((12, 12), "Predicted versus oracle MLRI blend weight", fill=(20, 20, 20), font=_font())
    image.save(path, format="PNG", compress_level=9)


def _diagnostic_map(
    case: Mapping[str, object],
    predicted: np.ndarray,
    oracle: np.ndarray,
    path: Path,
) -> None:
    truth = np.asarray(case["truth"])
    mark = np.asarray(case["markesteijn"])
    mlri = np.asarray(case["mlri"])
    hybrid = mark + predicted[None] * (mlri - mark)
    regression = np.sum((hybrid - truth) ** 2, axis=0) - np.sum((mark - truth) ** 2, axis=0)
    scale = max(float(np.percentile(np.abs(regression[MARGIN:-MARGIN, MARGIN:-MARGIN]), 99)), 1e-12)
    regression_rgb = np.full((*regression.shape, 3), 0.5)
    normalized = np.clip(regression / scale, -1, 1)
    regression_rgb[..., 0] += 0.5 * np.maximum(normalized, 0)
    regression_rgb[..., 1] -= 0.35 * np.abs(normalized)
    regression_rgb[..., 2] += 0.5 * np.maximum(-normalized, 0)
    disagreement = np.sqrt(np.mean((mlri - mark) ** 2, axis=0))
    disagreement /= max(float(np.percentile(disagreement, 99)), 1e-12)
    panels = (
        ("truth", np.moveaxis(srgb_encode(truth), 0, -1)),
        ("predicted alpha", np.repeat(predicted[..., None], 3, axis=2)),
        ("oracle alpha", np.repeat(oracle[..., None], 3, axis=2)),
        ("blue improves / red regresses", regression_rgb),
        ("candidate disagreement", np.repeat(np.clip(disagreement[..., None], 0, 1), 3, axis=2)),
    )
    zoom = 2
    height, width = predicted.shape
    image = Image.new("RGB", (len(panels) * width * zoom, height * zoom + 24), "white")
    draw = ImageDraw.Draw(image)
    for index, (label, values) in enumerate(panels):
        panel = Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8), "RGB")
        panel = panel.resize((width * zoom, height * zoom), Image.Resampling.NEAREST)
        image.paste(panel, (index * width * zoom, 24))
        draw.text((index * width * zoom + 4, 5), label, fill=(20, 20, 20), font=_font())
    image.save(path, format="PNG", compress_level=9)


def generate(output: Path, runner: Path, *, force: bool) -> tuple[Path, dict[str, float]]:
    output = output.resolve()
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    if force:
        for existing in output.iterdir():
            if existing.is_file():
                existing.unlink()

    sources = load_sources()
    dataset_payload = dataset_manifest(sources)
    dataset_bytes = canonical_json_bytes(dataset_payload)
    (output / "dataset.json").write_bytes(dataset_bytes)
    dataset_digest = hashlib.sha256(dataset_bytes).hexdigest()

    cases = []
    native_timings = {"markesteijn": 0.0, "mlri-final": 0.0}
    started = time.perf_counter()
    crops = list(iter_crops(sources))
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-hybrid-") as temporary:
        temporary_root = Path(temporary)
        for index, crop in enumerate(crops, start=1):
            print(f"[{index}/{len(crops)}] {crop['crop_id']}", flush=True)
            work = temporary_root / str(index)
            work.mkdir()
            outputs, timing, scalar, cfa = _run_pair(runner, crop["truth"], work)
            for name in native_timings:
                native_timings[name] += timing[name]
            cases.append({
                **crop, "cfa": cfa, "markesteijn": outputs["markesteijn"],
                "mlri": outputs["mlri-final"], "mosaic": scalar,
            })
    reconstruction_seconds = time.perf_counter() - started

    tables_by_size = {}
    feature_seconds = 0.0
    model_started = time.perf_counter()
    for block_size in (7, 15):
        feature_started = time.perf_counter()
        full = build_patch_table(cases, block_size)
        feature_seconds += time.perf_counter() - feature_started
        tables = {split: _split(full, split) for split in ("train", "validation", "test")}
        oracle = _oracle_rows(tables)
        train, validation = tables["train"], tables["validation"]

        candidates = [
            _fixed_blend(train),
            _threshold_candidates(train, validation),
            _linear_candidates("logistic-hard", "hard", FEATURE_ABLATIONS["all_core"], train, validation),
            _tree_candidates("tree-hard", "hard", FEATURE_ABLATIONS["all_core"], train, validation),
            _linear_candidates("ridge", "blend", FEATURE_ABLATIONS["all_core"], train, validation),
            _linear_candidates("logistic-blend", "blend", FEATURE_ABLATIONS["all_core"], train, validation),
            _tree_candidates("tree-blend", "blend", FEATURE_ABLATIONS["all_core"], train, validation),
            _linear_candidates("logistic-blend", "blend", ("A",), train, validation),
        ]
        candidates[-1].name = "disagreement-weighted-logistic-blend"
        candidates[-1].config["name"] = candidates[-1].name
        model_rows = [_quality_row(candidate, tables) for candidate in candidates]

        ablation_rows = []
        ablation_candidates = {}
        for name, groups in FEATURE_ABLATIONS.items():
            candidate = _linear_candidates("ridge", "blend", groups, train, validation)
            candidate.name = f"ridge-{name}"
            ablation_candidates[name] = candidate
            ablation_rows.append(_quality_row(candidate, tables))

        best = _best_by_validation([*candidates, *ablation_candidates.values()], validation)
        gate, gating_rows = _gate_candidate(best, validation)
        final_rows = {}
        for split, table in tables.items():
            alpha = _apply_gate(predict(best, table), gate)
            metrics = quality_metrics(table, alpha)
            metrics["classification"] = classification_metrics(table, alpha)
            metrics["calibration"] = calibration_metrics(table, alpha)
            metrics["headroom"] = _headroom(metrics, oracle[split])
            final_rows[split] = metrics

        tables_by_size[block_size] = {
            "full": full, "tables": tables, "oracle": oracle,
            "expanded_dataset_oracle": _oracle_rows({"all": full})["all"],
            "models": model_rows, "ablations": ablation_rows,
            "best": best, "gate": gate, "gating_rows": gating_rows,
            "final": final_rows,
        }

    selected_size = max(
        (7, 15),
        key=lambda size: tables_by_size[size]["final"]["validation"]["psnr_db"],
    )
    selected = tables_by_size[selected_size]
    best = selected["best"]
    gate = selected["gate"]
    development = selected["full"].subset(
        np.isin(
            selected["full"].source_ids,
            [row["id"] for row in sources if row["split"] != "test"],
        )
    )
    source_groups = {str(row["id"]): str(row["group"]) for row in NATURAL_SOURCES}
    cv_rows = _cross_validation(development, best.config, source_groups)

    validation_cases = [case for case in cases if case["split"] == "validation"]
    test_cases = [case for case in cases if case["split"] == "test"]
    validation_variants = _spatial_variants(
        best, gate, cases, validation_cases, selected["tables"]["validation"],
        "validation",
    )
    test_variants = _spatial_variants(
        best, gate, cases, test_cases, selected["tables"]["test"], "test",
    )
    selected_spatial_name = max(
        validation_variants,
        key=lambda name: validation_variants[name][1]["metrics"]["psnr_db"],
    )
    spatial_rows = [
        {
            "name": name,
            "validation": validation_variants[name][1],
            "test": test_variants[name][1],
        }
        for name in validation_variants
    ]

    # The block fields used for maps match the frozen practical operating point,
    # even if a diagnostic dense variant happens to score slightly higher.
    test_alpha = _apply_gate(predict(best, selected["tables"]["test"]), gate)
    all_predicted_fields = apply_block_alpha(cases, selected["tables"]["test"], test_alpha)
    all_oracle_fields = apply_block_alpha(cases, selected["tables"]["test"], selected["tables"]["test"].oracle_alpha)

    synthetic_rows = []
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-hybrid-synthetic-") as temporary:
        temporary_root = Path(temporary)
        for index, (name, metadata) in enumerate(synthetic_scenes().items()):
            work = temporary_root / str(index)
            work.mkdir()
            outputs, _, scalar, cfa = _run_pair(runner, metadata["truth"], work)
            synthetic_case = {
                "category": metadata["category"], "crop_id": name,
                "source_id": name, "split": "synthetic", "truth": metadata["truth"],
                "mosaic": scalar, "cfa": cfa, "markesteijn": outputs["markesteijn"],
                "mlri": outputs["mlri-final"],
            }
            table = build_patch_table((synthetic_case,), selected_size)
            practical_alpha = _apply_gate(predict(best, table), gate)
            synthetic_rows.append({
                "category": metadata["category"], "scene": name,
                "markesteijn": quality_metrics(table, np.zeros(table.features.shape[0])),
                "mlri": quality_metrics(table, np.ones(table.features.shape[0])),
                "hard_oracle": quality_metrics(table, table.labels),
                "convex_oracle": quality_metrics(table, table.oracle_alpha),
                "practical": quality_metrics(table, practical_alpha),
            })

    model_seconds = time.perf_counter() - model_started - feature_seconds
    inference_feature_seconds, inference_selector_seconds = _benchmark_inference(
        best, test_cases, selected_size,
    )
    per_source_test = _per_source_rows(
        selected["tables"]["test"], test_alpha,
    )
    native_feature_maxima = {
        name: float(np.max(np.abs(selected["tables"]["train"].features[:, selected["full"].feature_names.index(name)])))
        for name in selected["full"].feature_groups["F"]
    }
    results = {
        "dataset_sha256": dataset_digest,
        "experiment": "Markesteijn plus corrected-final MLRI deterministic selector/blender",
        "feature_contract": {
            "groups": {name: list(values) for name, values in selected["full"].feature_groups.items()},
            "ground_truth_is_not_an_argument": "feature_maps accepts only mosaic, CFA, Markesteijn, and MLRI",
            "normalization": "training-set mean and standard deviation only",
        },
        "format": "rawtherapee-xtrans-hybrid-results-v1",
        "image_level_cross_validation": cv_rows,
        "native_sample_feature_maxima": native_feature_maxima,
        "per_source_test": per_source_test,
        "scales": {
            str(size): {
                "feature_ablations": tables_by_size[size]["ablations"],
                "expanded_dataset_oracle": tables_by_size[size]["expanded_dataset_oracle"],
                "final": tables_by_size[size]["final"],
                "gate": tables_by_size[size]["gate"],
                "gating_sweep": tables_by_size[size]["gating_rows"],
                "model_ablations": tables_by_size[size]["models"],
                "oracle": tables_by_size[size]["oracle"],
                "selected_model": _serialize_candidate(tables_by_size[size]["best"], tables_by_size[size]["full"].feature_names),
            }
            for size in (7, 15)
        },
        "selected": {
            "block_size": selected_size,
            "gate": gate,
            "model": _serialize_candidate(best, selected["full"].feature_names),
            "spatial_variants": spatial_rows,
            "spatial_variant_selection": "highest validation PSNR; test metrics not used",
            "selected_spatial_variant": selected_spatial_name,
        },
        "synthetic_sanity": synthetic_rows,
    }
    result_bytes = canonical_json_bytes(_json_safe(results))
    (output / "results.json").write_bytes(result_bytes)

    _bar_plot(
        "Held-out feature-ablation gain over Markesteijn",
        [
            (
                row["name"],
                row["splits"]["test"]["psnr_db"] - selected["oracle"]["test"]["markesteijn"]["psnr_db"],
            )
            for row in selected["ablations"]
        ],
        output / "feature-ablation.png",
    )
    _bar_plot(
        "Development grouped-fold quality gain",
        [(f"fold {row['fold']}", row["gain_db"]) for row in cv_rows],
        output / "grouped-folds.png",
    )
    _gating_plot(selected["gating_rows"], output / "gating-coverage.png")
    selected_test_alpha = _apply_gate(predict(best, selected["tables"]["test"]), gate)
    _calibration_plot(
        calibration_metrics(selected["tables"]["test"], selected_test_alpha),
        output / "blend-calibration.png",
    )

    first_test_case_by_source = {}
    for case_index, case in enumerate(cases):
        if case["split"] == "test" and case["source_id"] not in first_test_case_by_source:
            first_test_case_by_source[case["source_id"]] = case_index
    for source_id, case_index in first_test_case_by_source.items():
        predicted = _extend_alpha_border(all_predicted_fields[case_index])
        oracle = _extend_alpha_border(all_oracle_fields[case_index])
        _diagnostic_map(cases[case_index], predicted, oracle, output / f"map-{source_id}.png")

    assets = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        assets.append({"bytes": path.stat().st_size, "filename": path.name, "sha256": _sha256(path)})
    manifest = {
        "assets": assets,
        "dataset_sha256": dataset_digest,
        "format": "rawtherapee-xtrans-hybrid-corpus-v1",
        "oracle_study_commit": "dba78e8f2",
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    timing = {
        "feature_seconds": feature_seconds,
        "inference_feature_seconds_for_12_test_crops": inference_feature_seconds,
        "inference_selector_seconds_for_12_test_crops": inference_selector_seconds,
        "markesteijn_seconds": native_timings["markesteijn"],
        "maximum_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "mlri_seconds": native_timings["mlri-final"],
        "model_and_analysis_seconds": model_seconds,
        "reconstruction_wall_seconds": reconstruction_seconds,
        "total_wall_seconds": time.perf_counter() - started,
    }
    return manifest_path, timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest, timing = generate(args.output, args.runner, force=args.force)
    print(f"manifest: {manifest}")
    print(f"sha256: {_sha256(manifest)}")
    print("timing: " + json.dumps(timing, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

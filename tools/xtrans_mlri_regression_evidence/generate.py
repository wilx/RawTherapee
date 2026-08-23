#!/usr/bin/env python3
"""Run the faithful MLRI guided-regression evidence experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
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
from tools.xtrans_mlri_green_fusion.analysis import (
    candidate_bank,
    interior_missing_green,
    patch_labels,
    preserve_measured_green,
    scalar_convex_hull,
    source_balanced_weights,
)
from tools.xtrans_mlri_green_fusion.dataset import dataset_manifest, load_cases
from tools.xtrans_mlri_green_fusion.generate import (
    _direct_baseline_metrics,
    _final_metrics,
    _run_case,
    _run_final_bank,
    _source_rows,
)
from tools.xtrans_mlri_internal.analysis import scalar_metrics
from tools.xtrans_mlri_internal.generate import srgb_encode
from .analysis import (
    FEATURE_NAMES,
    REGRESSION_STATISTICS,
    EvidenceTable,
    build_table,
    centered,
    feature_maps,
    fuse_survivors,
    modulated_guide,
    pairwise_metrics,
    pooled_correlations,
    regression_bank,
    select_guide,
    within_pixel_rank_metrics,
)


PAIRWISE_PAIRS = tuple((i, j) for i in range(16) for j in range(i + 1, 16))


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(_json_safe(value)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _fit_ridge(features: np.ndarray, target: np.ndarray, weights: np.ndarray,
               regularization: float):
    from tools.xtrans_mlri_green_fusion.analysis import fit_ridge_score
    return fit_ridge_score(features, target, weights, regularization)


def _tree_scores(model, features: np.ndarray) -> np.ndarray:
    shape = features.shape[:-1]
    flat = features.reshape(-1, features.shape[-1])
    selected = flat[:, model.feature_indices]
    return model.root.predict(selected).reshape(shape)


def _pairwise_scores(model, features: np.ndarray) -> np.ndarray:
    result = np.zeros(features.shape[:-1], dtype=np.float64)
    for first, second in PAIRWISE_PAIRS:
        difference = features[first] - features[second]
        probability_first_worse = model.alpha(
            difference.reshape(-1, difference.shape[-1])
        ).reshape(difference.shape[:-1])
        result[first] += probability_first_worse
        result[second] += 1.0 - probability_first_worse
    return result


def _fit_models(table: EvidenceTable) -> dict[str, object]:
    train = table.select(table.splits == "train")
    error = (train.candidates - train.truth[:, None]) ** 2
    target = np.log10(error + 1e-8)
    relative_target = centered(target)
    features = train.features.reshape(-1, len(FEATURE_NAMES))
    flattened_target = relative_target.ravel()
    pixel_weights = source_balanced_weights(train.source_ids)
    weights = np.repeat(pixel_weights, 16)
    selected = _indices(features.shape[0], 300000)

    ridge_rows = []
    ridge = None
    best = math.inf
    validation = table.select(table.splits == "validation")
    validation_target = centered(np.log10(
        (validation.candidates - validation.truth[:, None]) ** 2 + 1e-8
    ))
    for regularization in (1e-3, 1e-2, 1e-1, 1.0):
        candidate = _fit_ridge(
            features[selected], flattened_target[selected], weights[selected],
            regularization,
        )
        prediction = candidate.score(validation.features)
        mse = float(np.mean((prediction - validation_target) ** 2))
        ridge_rows.append({"regularization": regularization, "validation_mse": mse})
        if mse < best:
            best = mse
            ridge = candidate
    assert ridge is not None

    pixel_selection = _indices(train.truth.size, 7000)
    pair_features = []
    pair_labels = []
    pair_weights = []
    for first, second in PAIRWISE_PAIRS:
        difference = train.features[pixel_selection, first] - train.features[pixel_selection, second]
        margin = error[pixel_selection, first] - error[pixel_selection, second]
        pair_features.append(difference)
        pair_labels.append((margin > 0).astype(np.float64))
        pair_weights.append(pixel_weights[pixel_selection] * (np.abs(margin) + 1e-6))
    pair_features_array = np.concatenate(pair_features)
    pair_labels_array = np.concatenate(pair_labels)
    pair_weights_array = np.concatenate(pair_weights)
    pair_selection = _indices(pair_labels_array.size, 220000)
    pairwise = fit_logistic(
        pair_features_array[pair_selection], pair_labels_array[pair_selection],
        pair_weights_array[pair_selection],
        np.arange(len(FEATURE_NAMES), dtype=np.int32), 1e-2,
    )

    tree_selection = _indices(features.shape[0], 90000)
    low, high = np.percentile(flattened_target[tree_selection], (1, 99))
    tree_target = np.clip(
        (flattened_target - low) / max(high - low, 1e-8), 0, 1,
    )
    tree = fit_tree(
        features[tree_selection], tree_target[tree_selection], weights[tree_selection],
        np.arange(len(FEATURE_NAMES), dtype=np.int32), FEATURE_NAMES, 3,
        classification=False,
    )
    return {
        "ridge": ridge,
        "pairwise": pairwise,
        "tree": tree,
        "report": {
            "ridge_validation": ridge_rows,
            "ridge": {
                "regularization": ridge.regularization,
                "intercept": ridge.intercept,
                "coefficients": dict(zip(FEATURE_NAMES, map(float, ridge.coefficients))),
            },
            "pairwise": {
                "regularization": pairwise.regularization,
                "intercept": pairwise.intercept,
                "coefficients": dict(zip(FEATURE_NAMES, map(float, pairwise.coefficients))),
            },
            "tree": tree.root.to_dict(FEATURE_NAMES),
        },
    }


def _standardized_cost(*values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(values[0]), dtype=np.float64)
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        shifted = array - np.mean(array, axis=0, keepdims=True)
        result += shifted / np.maximum(np.std(shifted, axis=0, keepdims=True), 1e-6)
    return result


def _score_maps(case: Mapping[str, object], models: Mapping[str, object]) -> dict[str, np.ndarray]:
    features = feature_maps(case).astype(np.float64)
    at = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    scores = {
        "directional-energy": features[..., at["log-directional-energy"]],
        "fit-mse": features[..., at["log-fit-mse"]],
        "min-fit-mse": features[..., at["log-min-fit-mse"]],
        "gain-instability": features[..., at["abs-gain"]]
            + features[..., at["gain-variance"]],
        "denominator-instability": features[..., at["inverse-gain-denominator"]],
        "prediction-variance": features[..., at["log-prediction-variance"]],
        "prediction-range": features[..., at["log-prediction-range"]],
        "prediction-mad": features[..., at["log-prediction-mad"]],
        "loo-influence": features[..., at["log-max-model-influence"]],
        "weak-support": features[..., at["inverse-effective-model-count"]]
            + features[..., at["max-model-weight"]],
    }
    scores["stability-composite"] = _standardized_cost(
        scores["fit-mse"], scores["prediction-variance"],
        scores["gain-instability"], scores["loo-influence"],
        scores["weak-support"],
    )
    scores["low-residual-high-disagreement"] = _standardized_cost(
        -scores["fit-mse"], scores["prediction-variance"],
        scores["gain-instability"],
    )
    scores["ridge"] = models["ridge"].score(features)
    scores["tree"] = _tree_scores(models["tree"], features)
    scores["pairwise"] = _pairwise_scores(models["pairwise"], features)
    return scores


def _table_score(table: EvidenceTable, name: str, models: Mapping[str, object]) -> np.ndarray:
    at = {feature: FEATURE_NAMES.index(feature) for feature in FEATURE_NAMES}
    features = table.features.astype(np.float64)
    direct = {
        "directional-energy": features[..., at["log-directional-energy"]],
        "fit-mse": features[..., at["log-fit-mse"]],
        "min-fit-mse": features[..., at["log-min-fit-mse"]],
        "gain-instability": features[..., at["abs-gain"]] + features[..., at["gain-variance"]],
        "denominator-instability": features[..., at["inverse-gain-denominator"]],
        "prediction-variance": features[..., at["log-prediction-variance"]],
        "prediction-range": features[..., at["log-prediction-range"]],
        "prediction-mad": features[..., at["log-prediction-mad"]],
        "loo-influence": features[..., at["log-max-model-influence"]],
        "weak-support": features[..., at["inverse-effective-model-count"]]
            + features[..., at["max-model-weight"]],
    }
    if name == "stability-composite":
        return _standardized_cost(
            direct["fit-mse"].T, direct["prediction-variance"].T,
            direct["gain-instability"].T, direct["loo-influence"].T,
            direct["weak-support"].T,
        ).T
    if name == "low-residual-high-disagreement":
        return _standardized_cost(
            -direct["fit-mse"].T, direct["prediction-variance"].T,
            direct["gain-instability"].T,
        ).T
    if name == "ridge":
        return models["ridge"].score(features)
    if name == "tree":
        shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        return models["tree"].root.predict(flat[:, models["tree"].feature_indices]).reshape(shape)
    if name == "pairwise":
        result = np.zeros(features.shape[:-1], dtype=np.float64)
        for first, second in PAIRWISE_PAIRS:
            probability = models["pairwise"].alpha(features[:, first] - features[:, second])
            result[:, first] += probability
            result[:, second] += 1.0 - probability
        return result
    return direct[name]


def _diagnostic_analysis(table: EvidenceTable, models: Mapping[str, object]) -> dict[str, object]:
    names = (
        "directional-energy", "fit-mse", "min-fit-mse", "gain-instability",
        "denominator-instability", "prediction-variance", "prediction-range",
        "prediction-mad", "loo-influence", "weak-support",
        "stability-composite", "low-residual-high-disagreement",
        "ridge", "tree", "pairwise",
    )
    result = {}
    for split in ("validation", "test"):
        subset = table.select(table.splits == split)
        error = (subset.candidates - subset.truth[:, None]) ** 2
        result[split] = {}
        for name in names:
            score = _table_score(subset, name, models)
            result[split][name] = {
                **pooled_correlations(score, error),
                **within_pixel_rank_metrics(score, error),
                "pairwise": pairwise_metrics(score, error),
            }
    return result


def _guides(cases: Sequence[Mapping[str, object]], models: Mapping[str, object]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    hard = (
        "fit-mse", "prediction-variance", "gain-instability",
        "denominator-instability", "loo-influence", "weak-support",
        "stability-composite", "low-residual-high-disagreement",
        "ridge", "tree", "pairwise",
    )
    for case in cases:
        candidates, energies, _ = candidate_bank(case)
        trace = case["trace"]
        result["current-pass1"].append(preserve_measured_green(case, trace["pass1-green"]).astype(np.float32))
        result["pass0-current"].append(preserve_measured_green(case, trace["pass0-green"]).astype(np.float32))
        result["min-energy-pass1"].append(select_guide(case, np.concatenate((
            np.full_like(energies[:8], np.inf), energies[8:]
        ))))
        result["true-green"].append(preserve_measured_green(case, np.asarray(case["truth"])[1]).astype(np.float32))
        oracle = patch_labels(candidates, np.asarray(case["truth"])[1], 7)
        result["oracle-combined-7x7"].append(preserve_measured_green(
            case, np.take_along_axis(candidates, oracle[None], axis=0)[0]
        ).astype(np.float32))
        convex, _ = scalar_convex_hull(case, candidates)
        result["oracle-convex"].append(convex.astype(np.float32))

        scores = _score_maps(case, models)
        for name in hard:
            result[f"hard-{name}"].append(select_guide(case, scores[name]))
        pass1_fit = np.concatenate((np.full_like(scores["fit-mse"][:8], np.inf), scores["fit-mse"][8:]))
        pass1_stability = np.concatenate((
            np.full_like(scores["stability-composite"][:8], np.inf),
            scores["stability-composite"][8:],
        ))
        result["hard-fit-mse-pass1"].append(select_guide(case, pass1_fit))
        result["hard-stability-pass1"].append(select_guide(case, pass1_stability))

        for score_name in ("stability-composite", "ridge"):
            for retained in (2, 4, 8):
                result[f"prune-{score_name}-weighted-k{retained}"].append(
                    fuse_survivors(case, scores[score_name], retained, "current-weight")
                )
            result[f"prune-{score_name}-uniform-k4"].append(
                fuse_survivors(case, scores[score_name], 4, "uniform")
            )
            result[f"prune-{score_name}-median-k4"].append(
                fuse_survivors(case, scores[score_name], 4, "median")
            )
            for strength in (0.25, 0.5, 1.0, 2.0):
                result[f"modulate-{score_name}-{strength:g}"].append(
                    modulated_guide(case, scores[score_name], strength)
                )
    return dict(result)


def _source_lookup(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(row["source_id"]): row for row in rows}


def _select_practical(source_rows: Sequence[Mapping[str, object]], methods: Sequence[str]) -> tuple[str, list[dict[str, object]]]:
    sources = _source_lookup(source_rows)
    current_brick = sources["brick"]["current-pass1"]["mse"]
    pass0_brick = sources["brick"]["pass0-current"]["mse"]
    current_hubble = sources["hubble"]["current-pass1"]["mse"]
    rows = []
    for method in methods:
        brick = sources["brick"][method]["mse"]
        denominator = pass0_brick - current_brick
        retention = None if denominator <= 0 else (pass0_brick - brick) / denominator
        hubble = sources["hubble"][method]["mse"]
        admissible = (
            (retention is None or retention >= 0.90)
            and hubble <= current_hubble
        )
        validation_psnr = np.mean([
            sources[source][method]["psnr_db"] for source in ("brick", "hubble")
        ])
        rows.append({
            "method": method,
            "brick_pass1_gain_retention": retention,
            "hubble_mse_ratio": hubble / current_hubble,
            "mean_source_validation_psnr_db": float(validation_psnr),
            "admissible": bool(admissible),
        })
    eligible = [row for row in rows if row["admissible"]]
    eligible.sort(key=lambda row: (-row["mean_source_validation_psnr_db"], row["method"]))
    return (eligible[0]["method"] if eligible else "current-pass1"), rows


def _transition_rows(cases: Sequence[Mapping[str, object]], selected: str,
                     guides: Mapping[str, Sequence[np.ndarray]],
                     source_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    sources = _source_lookup(source_rows)
    result = []
    statistic_index = {name: REGRESSION_STATISTICS.index(name) for name in REGRESSION_STATISTICS}
    for case_index, case in enumerate(cases):
        case_id = str(case["case_id"])
        if not (
            case_id.startswith("transition-points-")
            or case_id.startswith("transition-line-")
            or case_id.startswith("transition-coherence-")
        ):
            continue
        mask = interior_missing_green(case)
        regression = regression_bank(case)
        pass_means = {}
        for pass_index in (0, 1):
            bank = regression[pass_index * 8:(pass_index + 1) * 8]
            pass_means[f"pass{pass_index}"] = {
                name: float(np.mean(bank[..., index][:, mask]))
                for name, index in statistic_index.items()
                if name in (
                    "fit-mse", "gain", "gain-denominator",
                    "prediction-variance", "prediction-range",
                    "effective-model-count", "max-model-influence",
                )
            }
        result.append({
            "case_id": case_id,
            "family": case["family"],
            "diagnostic_means": {
                name: float(np.mean(regression[..., index][:, mask]))
                for name, index in statistic_index.items()
                if name in (
                    "fit-mse", "gain", "gain-denominator",
                    "prediction-variance", "prediction-range",
                    "effective-model-count", "max-model-influence",
                )
            },
            "diagnostic_means_by_pass": pass_means,
            "current_final_rgb": sources[str(case["source_id"])]["current-pass1"],
            "selected_final_rgb": sources[str(case["source_id"])][selected],
            "selected_green": scalar_metrics(
                (np.asarray(guides[selected][case_index]) - np.asarray(case["truth"])[1])[mask]
            ),
        })
    return result


def _source_regression_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for case in cases:
        if case["kind"] == "natural":
            grouped[str(case["source_id"])].append(case)
    statistic_index = {name: REGRESSION_STATISTICS.index(name) for name in REGRESSION_STATISTICS}
    wanted = (
        "fit-mse", "gain", "gain-variance", "gain-denominator",
        "prediction-variance", "prediction-range", "prediction-mad",
        "effective-model-count", "max-model-weight", "max-model-influence",
    )
    rows = []
    for source, source_cases in sorted(grouped.items()):
        row: dict[str, object] = {
            "source_id": source,
            "split": source_cases[0]["split"],
        }
        for pass_index in (0, 1):
            accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
            candidate_errors = []
            fused_errors = []
            for case in source_cases:
                mask = interior_missing_green(case)
                regression = regression_bank(case)[pass_index * 8:(pass_index + 1) * 8]
                candidates, _, _ = candidate_bank(case)
                candidates = candidates[pass_index * 8:(pass_index + 1) * 8]
                truth = np.asarray(case["truth"])[1]
                for name in wanted:
                    accumulators[name].append(regression[..., statistic_index[name]][:, mask])
                candidate_errors.append((candidates[:, mask] - truth[mask]) ** 2)
                fused_errors.append((
                    np.asarray(case["trace"][f"pass{pass_index}-green"])[mask]
                    - truth[mask]
                ) ** 2)
            error = np.concatenate(candidate_errors, axis=1)
            row[f"pass{pass_index}"] = {
                "diagnostic_means": {
                    name: float(np.mean(np.concatenate(values, axis=1)))
                    for name, values in accumulators.items()
                },
                "all_candidate_green_mse": float(np.mean(error)),
                "best_candidate_green_mse": float(np.mean(np.min(error, axis=0))),
                "fused_green_mse": float(np.mean(np.concatenate(fused_errors))),
            }
        rows.append(row)
    return rows


def _phase_stability_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    fit_index = FEATURE_NAMES.index("log-fit-mse")
    for case in cases:
        if not str(case["case_id"]).startswith("cfa-impulse-"):
            continue
        mask = interior_missing_green(case)
        candidates, _, _ = candidate_bank(case)
        truth = np.asarray(case["truth"])[1][mask]
        error = (candidates[:, mask].T - truth[:, None]) ** 2
        score = feature_maps(case)[..., fit_index][:, mask].T
        ranking = within_pixel_rank_metrics(score, error)
        rows.append({
            "case_id": case["case_id"],
            "origin": [case["origin_x"], case["origin_y"]],
            "fit_mse_ranking": ranking,
            "mean_fit_mse": float(np.mean(score)),
            "mean_best_candidate_squared_error": float(np.mean(np.min(error, axis=1))),
        })
    return rows


def _diagnostic_map(case: Mapping[str, object], selected_guide: np.ndarray,
                    selected_rgb: np.ndarray, scores: Mapping[str, np.ndarray],
                    path: Path) -> None:
    truth = np.asarray(case["truth"])
    candidates, _, _ = candidate_bank(case)
    regression = regression_bank(case)
    at = {name: REGRESSION_STATISTICS.index(name) for name in REGRESSION_STATISTICS}
    current = np.stack((
        case["trace"]["final-red"], case["trace"]["final-green"],
        case["trace"]["final-blue"],
    ))
    labels = np.argmin(scores["ridge"], axis=0)
    oracle = patch_labels(candidates, truth[1], 7)
    fit = np.min(np.maximum(regression[..., at["fit-mse"]], 0), axis=0)
    spread = np.min(regression[..., at["prediction-range"]], axis=0)
    influence = np.min(regression[..., at["max-model-influence"]], axis=0)

    def gray(values: np.ndarray) -> Image.Image:
        encoded = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(np.stack((encoded,) * 3, axis=-1), "RGB")

    def heat(values: np.ndarray) -> Image.Image:
        checked = np.maximum(np.asarray(values, dtype=np.float64), 0)
        scale = max(float(np.percentile(checked, 99)), 1e-8)
        mapped = np.clip(checked / scale, 0, 1)
        rgb = np.stack((mapped, np.sqrt(mapped), 1 - mapped), axis=-1)
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), "RGB")

    def label_image(values: np.ndarray) -> Image.Image:
        palette = np.asarray([
            (230,25,75),(60,180,75),(255,225,25),(0,130,200),
            (245,130,48),(145,30,180),(70,240,240),(240,50,230),
            (170,110,40),(128,0,0),(170,255,195),(0,0,128),
            (128,128,0),(255,215,180),(0,128,128),(128,128,128),
        ], dtype=np.uint8)
        return Image.fromarray(palette[np.asarray(values, dtype=np.int64) % 16], "RGB")

    panels = (
        ("truth green", gray(truth[1])),
        ("selected green", gray(selected_guide)),
        ("current RGB error", heat(np.mean(np.abs(current - truth), axis=0))),
        ("selected RGB error", heat(np.mean(np.abs(selected_rgb - truth), axis=0))),
        ("min fit MSE", heat(fit)),
        ("min prediction range", heat(spread)),
        ("min model influence", heat(influence)),
        ("ridge identity", label_image(labels)),
        ("7x7 oracle identity", label_image(oracle)),
    )
    scale = max(1, min(4, 220 // truth.shape[1]))
    panel_width = truth.shape[2] * scale
    panel_height = truth.shape[1] * scale
    output = Image.new("RGB", (len(panels) * panel_width, panel_height + 24), "white")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        output.paste(panel.resize((panel_width, panel_height), Image.Resampling.NEAREST),
                     (index * panel_width, 24))
        draw.text((index * panel_width + 3, 5), label, fill=(10,10,10), font=font)
    output.save(path, format="PNG", compress_level=9)


def run(runner: Path, starfield: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(starfield)
    timings = {}
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-mlri-regression-") as temporary:
        root = Path(temporary)
        started = time.perf_counter()
        for index, case in enumerate(cases):
            work = root / f"trace-{index:03d}"
            work.mkdir()
            _run_case(case, runner, work)
            print(f"[trace {index + 1:02d}/{len(cases):02d}] {case['case_id']}", flush=True)
        timings["trace_seconds"] = time.perf_counter() - started

        print("[analysis] building candidate table", flush=True)
        phase_started = time.perf_counter()
        table = build_table(cases)
        timings["candidate_table_seconds"] = time.perf_counter() - phase_started
        print("[analysis] fitting compact candidate models", flush=True)
        phase_started = time.perf_counter()
        models = _fit_models(table)
        timings["model_fit_seconds"] = time.perf_counter() - phase_started
        print("[analysis] measuring within-pixel and pairwise evidence", flush=True)
        phase_started = time.perf_counter()
        diagnostics = _diagnostic_analysis(table, models)
        timings["diagnostic_analysis_seconds"] = time.perf_counter() - phase_started
        print("[analysis] constructing hard, pruned, and modulated guides", flush=True)
        phase_started = time.perf_counter()
        guides = _guides(cases, models)
        timings["guide_construction_seconds"] = time.perf_counter() - phase_started
        print("[analysis] running all guides through native final R/B", flush=True)
        phase_started = time.perf_counter()
        final, source_errors = _final_metrics(cases, guides, runner, root)
        timings["final_rgb_seconds"] = time.perf_counter() - phase_started
        mark_metrics, mark_source = _direct_baseline_metrics(
            cases, lambda case: np.asarray(case["markesteijn"]),
        )
        source_errors["markesteijn"] = mark_source
        source_rows = _source_rows(
            cases, source_errors, ("markesteijn",) + tuple(guides),
        )
        selected, selection_rows = _select_practical(
            source_rows,
            tuple(name for name in guides if not name.startswith("oracle-") and name != "true-green"),
        )

        current_mse = final["current-pass1"]["test"]["mse"]
        selected_mse = final[selected]["test"]["mse"]
        oracle_mse = final["oracle-combined-7x7"]["test"]["mse"]
        recovery = (current_mse - selected_mse) / max(current_mse - oracle_mse, 1e-20)
        sources = _source_lookup(source_rows)
        texture_retention = {}
        for source in ("brick", "grass", "gravel", "page"):
            pass0 = sources[source]["pass0-current"]["mse"]
            current = sources[source]["current-pass1"]["mse"]
            practical = sources[source][selected]["mse"]
            texture_retention[source] = None if pass0 <= current else (pass0 - practical) / (pass0 - current)
        sparse_delta = {
            source: sources[source][selected]["psnr_db"] - sources[source]["current-pass1"]["psnr_db"]
            for source in ("hubble", "nasa-hydra-starfield")
        }

        map_names = (
            "hubble-0", "nasa-hydra-starfield-0", "brick-0", "page-0",
            "transition-coherence-isolated-points",
            "transition-coherence-continuous-line",
        )
        image_files = []
        for case_id in map_names:
            matches = [i for i, case in enumerate(cases) if case["case_id"] == case_id]
            if not matches:
                continue
            index = matches[0]
            work = root / f"map-final-{index:03d}"
            work.mkdir()
            rgb = _run_final_bank(cases[index], [guides[selected][index]], runner, work)[0]
            name = f"map-{case_id}.png"
            _diagnostic_map(
                cases[index], guides[selected][index], rgb,
                _score_maps(cases[index], models), output / name,
            )
            image_files.append(name)

        transition = _transition_rows(cases, selected, guides, source_rows)
        decision = {
            "classification": "GO" if (
                recovery >= 0.30
                and all(value >= 0 for value in sparse_delta.values())
                and all(value is None or value >= 0.90 for value in texture_retention.values())
            ) else "PARTIAL" if (
                recovery > 0.0258 and selected_mse < current_mse
            ) else "NO-GO",
            "selected_method": selected,
            "test_oracle_gap_recovery": float(recovery),
            "sparse_source_psnr_delta_db": sparse_delta,
            "coherent_texture_pass1_gain_retention": texture_retention,
        }
        results = {
            "format": "rawtherapee-xtrans-mlri-regression-evidence-results-v1",
            "case_count": len(cases),
            "regression_statistics": list(REGRESSION_STATISTICS),
            "candidate_feature_names": list(FEATURE_NAMES),
            "candidate_attribution": (
                "each statistic follows the same chromatic/green CFA-site pairing, opposite-color completion, and phase-dependent half-Gaussian support as its final directional G-C candidate"
            ),
            "diagnostic_analysis": diagnostics,
            "models": models["report"],
            "final_metrics": final,
            "markesteijn_metrics": mark_metrics,
            "source_metrics": source_rows,
            "validation_selection": selection_rows,
            "selected_practical": selected,
            "transitions": transition,
            "natural_source_regression": _source_regression_rows(cases),
            "cfa_phase_stability": _phase_stability_rows(cases),
            "decision": decision,
            "runtime_policy": (
                "wall-clock timings are reported in the reviewed Markdown report and excluded from canonical JSON reproducibility identities"
            ),
            "instrumentation_limit": (
                "leave-one-model-out influence is exact for overlapping fitted models; full leave-one-support-sample regression refits were not run because guidedMlri uses overlapping 7x7 fits and would require a separate high-cost reconstruction path"
            ),
        }
        _write(output / "results.json", results)
        dataset = dataset_manifest(cases)
        dataset["format"] = "rawtherapee-xtrans-mlri-regression-evidence-dataset-v1"
        _write(output / "dataset.json", dataset)
        manifest = {
            "format": "rawtherapee-xtrans-mlri-regression-evidence-manifest-v1",
            "files": [
                {"name": name, "bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
                for name in ("dataset.json", "results.json", *image_files)
            ],
        }
        _write(output / "manifest.json", manifest)
    print(
        f"{results['decision']['classification']}: selected={selected} "
        f"recovery={results['decision']['test_oracle_gap_recovery']:.4f}",
        flush=True,
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.runner.resolve(), args.starfield.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the bounded three-bank adaptive X-Trans LMMSE experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_lmmse.experiment import (
    BASELINE_METHODS,
    _load_experiment_sources,
    _run_baselines,
)
from tools.xtrans_lmmse.model import (
    _dc_transform,
    collect_phase_samples,
    observation_colors,
    phase_origins,
    predict_region,
)
from tools.xtrans_lmmse_mixture.dataset import (
    TRAIN_COUNTS,
    discover_expanded,
    external_chromatic_sources,
    load_rgb,
    manifest as dataset_manifest,
)
from tools.xtrans_lmmse_mixture.model import (
    MixtureBank,
    classify_features,
    classify_region,
    normalize_features,
    observable_features,
    oracle_select,
    predict_mixture,
    soft_weights,
)
from tools.xtrans_lmmse_mixture.training import (
    MINIMUM_CLASS_PHASE_SAMPLES,
    RIDGE_RATIO,
    SAMPLES_PER_PHASE_PER_SOURCE,
    SUPPORT,
    definition_json,
    initial_definitions,
    normalization_json,
    sensitivity_definitions,
    survey_and_train_globals,
    survey_summary,
    train_mixtures,
)


METRIC_MARGIN = 12
VALIDATION_SAMPLES_PER_PHASE_PER_SOURCE = 256
POOLED_TAIL_METHODS = {
    "markesteijn", "corrected-final", "g100", "gmax", "mix3-hard",
    "mix3-soft", "oracle-7",
}


def _selected_mask(shape: tuple[int, int], margin: int = METRIC_MARGIN) -> np.ndarray:
    height, width = shape
    checked_margin = min(margin, max(0, min(height, width) // 4))
    selected = np.zeros((height, width), dtype=bool)
    if checked_margin == 0:
        selected[:] = True
    else:
        selected[checked_margin:-checked_margin, checked_margin:-checked_margin] = True
    return selected


def _metric(reconstruction: np.ndarray, truth: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    selected = _selected_mask(truth.shape[1:])
    difference = np.asarray(reconstruction, dtype=np.float64)[:, selected] - truth[:, selected]
    absolute = np.abs(difference).ravel()
    squared = difference * difference
    mse = float(np.mean(squared))
    return ({
        "count": int(difference.size),
        "maximum_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "mse": mse,
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": float(np.sum(squared)),
    }, absolute.astype(np.float32))


@dataclass
class _Collector:
    rows: list[dict[str, float | int]] = field(default_factory=list)
    tails: list[np.ndarray] = field(default_factory=list)

    def add(self, reconstruction: np.ndarray, truth: np.ndarray, keep_tail: bool) -> dict[str, float | int]:
        metric, absolute = _metric(reconstruction, truth)
        self.rows.append(metric)
        if keep_tail:
            self.tails.append(absolute)
        return metric

    def pooled(self) -> dict[str, float | int]:
        count = sum(int(row["count"]) for row in self.rows)
        sse = sum(float(row["sse"]) for row in self.rows)
        mse = sse / count
        result: dict[str, float | int] = {
            "case_count": len(self.rows),
            "count": count,
            "maximum_abs": max(float(row["maximum_abs"]) for row in self.rows),
            "mse": mse,
            "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
            "sse": sse,
        }
        if self.tails:
            absolute = np.concatenate(self.tails)
            result.update({
                "median_abs": float(np.median(absolute)),
                "p90_abs": float(np.quantile(absolute, 0.90)),
                "p95_abs": float(np.quantile(absolute, 0.95)),
                "p99_abs": float(np.quantile(absolute, 0.99)),
            })
        return result


def _sample_prediction(
    bank, phase: int, observations: np.ndarray, targets: np.ndarray,
) -> np.ndarray:
    transformed, _, dc = _dc_transform(observations, None, phase, bank.dc_mode, bank.support)
    predicted = bank.target_means[phase] + (
        transformed - bank.x_means[phase]
    ) @ bank.weights[phase].T
    predicted += dc
    sampled_channel = int(
        observation_colors(bank.support, phase)[bank.support * bank.support // 2]
    )
    predicted[:, sampled_channel] = targets[:, sampled_channel]
    return predicted


def _validation_result(
    mixture: MixtureBank, validation_rows: list[dict[str, object]],
) -> dict[str, object]:
    hard_sse = 0.0
    soft_sse = 0.0
    count = 0
    class_counts = np.zeros(3, dtype=np.int64)
    for source in validation_rows:
        truth = load_rgb(source)
        radius = SUPPORT // 2
        observations, targets = collect_phase_samples(
            truth, SUPPORT,
            (radius, radius, truth.shape[2] - radius, truth.shape[1] - radius),
            maximum_per_phase=VALIDATION_SAMPLES_PER_PHASE_PER_SOURCE,
        )
        for phase in range(18):
            raw = observable_features(observations[phase], phase, SUPPORT)
            features = normalize_features(raw, phase, mixture.normalization)
            labels = classify_features(features, mixture.definition)
            class_counts += np.bincount(labels, minlength=3)
            candidates = np.stack(
                [
                    _sample_prediction(bank, phase, observations[phase], targets[phase])
                    for bank in mixture.banks
                ],
                axis=0,
            )
            selected = candidates[labels, np.arange(labels.size)]
            hard_sse += float(np.sum((selected - targets[phase]) ** 2))
            weights = soft_weights(features, mixture.definition)
            blended = np.sum(np.moveaxis(candidates, 0, 1) * weights[:, :, None], axis=1)
            sampled_channel = int(
                observation_colors(SUPPORT, phase)[SUPPORT * SUPPORT // 2]
            )
            blended[:, sampled_channel] = targets[phase][:, sampled_channel]
            soft_sse += float(np.sum((blended - targets[phase]) ** 2))
            count += int(targets[phase].size)
    return {
        "class_counts": class_counts.tolist(),
        "class_fractions": (class_counts / np.sum(class_counts)).tolist(),
        "count": count,
        "hard": {
            "mse": hard_sse / count,
            "psnr_db": float(-10.0 * math.log10(max(hard_sse / count, 1e-30))),
            "sse": hard_sse,
        },
        "soft": {
            "mse": soft_sse / count,
            "psnr_db": float(-10.0 * math.log10(max(soft_sse / count, 1e-30))),
            "sse": soft_sse,
        },
    }


def _filter_analysis(mixture: MixtureBank) -> dict[str, object]:
    pair_rows = []
    radius_rows: dict[str, dict[str, list[float]]] = {}
    strongest = []
    for first in range(3):
        for second in range(first + 1, 3):
            first_weights = mixture.banks[first].weights
            second_weights = mixture.banks[second].weights
            difference = first_weights - second_weights
            flat_first = first_weights.reshape(54, -1)
            flat_second = second_weights.reshape(54, -1)
            norms = np.linalg.norm(difference.reshape(54, -1), axis=1)
            cosine = np.sum(flat_first * flat_second, axis=1) / (
                np.linalg.norm(flat_first, axis=1) * np.linalg.norm(flat_second, axis=1) + 1e-30
            )
            pair_name = f"{mixture.definition.class_names[first]}--{mixture.definition.class_names[second]}"
            pair_rows.append({
                "average_cosine_similarity": float(np.mean(cosine)),
                "average_l2_difference": float(np.mean(norms)),
                "maximum_l2_difference": float(np.max(norms)),
                "pair": pair_name,
            })
            radius_rows[pair_name] = {}
            for phase in range(18):
                colors = observation_colors(SUPPORT, phase)
                y, x = np.mgrid[:SUPPORT, :SUPPORT]
                radial = np.maximum(np.abs(x - SUPPORT // 2), np.abs(y - SUPPORT // 2)).ravel()
                for output_channel in range(3):
                    coefficients = difference[phase, output_channel]
                    top = np.argsort(np.abs(coefficients), kind="stable")[-3:][::-1]
                    for index in top:
                        strongest.append({
                            "absolute_difference": float(abs(coefficients[index])),
                            "coefficient_difference": float(coefficients[index]),
                            "dx": int(index % SUPPORT - SUPPORT // 2),
                            "dy": int(index // SUPPORT - SUPPORT // 2),
                            "input_color": int(colors[index]),
                            "output_channel": output_channel,
                            "pair": pair_name,
                            "phase": phase,
                        })
                    for sample_color in range(3):
                        for sample_radius in range(SUPPORT // 2 + 1):
                            selected = (colors == sample_color) & (radial == sample_radius)
                            if np.any(selected):
                                key = f"color{sample_color}-radius{sample_radius}"
                                radius_rows[pair_name].setdefault(key, []).append(
                                    float(np.mean(np.abs(coefficients[selected])))
                                )
    return {
        "pairwise": pair_rows,
        "strongest_coefficient_changes": sorted(
            strongest, key=lambda row: float(row["absolute_difference"]), reverse=True
        )[:24],
        "mean_absolute_difference_by_input_color_and_radius": {
            pair: {key: float(np.mean(values)) for key, values in rows.items()}
            for pair, rows in radius_rows.items()
        },
    }


def _case(
    case_id: str,
    context_truth: np.ndarray,
    region: tuple[int, int, int, int],
    group: str,
) -> dict[str, object]:
    x0, y0, x1, y1 = region
    return {
        "case_id": case_id,
        "context_truth": context_truth,
        "group": group,
        "origin_x": x0 % 6,
        "origin_y": y0 % 6,
        "region": region,
        "truth": context_truth[:, y0:y1, x0:x1],
    }


def _bsds_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        truth = load_rgb(row)
        result.append(_case(str(row["filename"]), truth, (0, 0, truth.shape[2], truth.shape[1]), "bsds-test"))
    return result


def _chromatic_cases() -> list[dict[str, object]]:
    result = []
    for source in external_chromatic_sources():
        truth = np.asarray(source["rgb"], dtype=np.float64)
        for crop_index, (x, y) in enumerate(source["crops"]):
            result.append(_case(
                f"{source['id']}-{crop_index}", truth,
                (int(x), int(y), int(x) + 168, int(y) + 168),
                str(source["id"]),
            ))
    return result


def _established_cases(
    source_oracle: dict[str, object], starfield: Path,
) -> list[dict[str, object]]:
    sources = {str(row["id"]): row for row in _load_experiment_sources(starfield)}
    result = []
    for oracle_source in source_oracle["sources"]:
        source_id = str(oracle_source["source_id"])
        selected_support = int(oracle_source["selected_support"])
        selected = next(
            row for row in oracle_source["supports"]
            if int(row["support"]) == selected_support
        )
        region = tuple(int(value) for value in selected["test_region"])
        result.append(_case(
            source_id,
            np.asarray(sources[source_id]["rgb"], dtype=np.float64),
            region,
            "established",
        ))
    return result


def _evaluate_cases(
    cases: list[dict[str, object]],
    globals_by_size: dict[int, object],
    mixture: MixtureBank,
    runner: Path,
    *,
    run_baselines: bool,
) -> dict[str, object]:
    collectors: dict[str, _Collector] = {}
    rows = []
    feature_samples = []
    for case_index, case in enumerate(cases):
        truth = np.asarray(case["truth"], dtype=np.float64)
        context = np.asarray(case["context_truth"], dtype=np.float64)
        region = tuple(int(value) for value in case["region"])
        individual = tuple(
            predict_region(bank, context, region, clip=False)
            for bank in mixture.banks
        )
        outputs: dict[str, np.ndarray] = {
            "g100": predict_region(globals_by_size[100], context, region, clip=False),
            "g150": predict_region(globals_by_size[150], context, region, clip=False),
            "gmax": predict_region(globals_by_size[200], context, region, clip=False),
        }
        outputs["gmax-clipped"] = np.clip(outputs["gmax"], 0.0, 1.0)
        for class_index, class_name in enumerate(mixture.definition.class_names):
            outputs[f"bank-{class_name}"] = individual[class_index]
        hard, class_map = predict_mixture(
            mixture, context, region, soft=False, individual_outputs=individual
        )
        soft, _ = predict_mixture(
            mixture, context, region, soft=True, individual_outputs=individual
        )
        outputs["mix3-hard"] = hard
        outputs["mix3-soft"] = soft
        outputs["mix3-hard-clipped"] = np.clip(hard, 0.0, 1.0)
        outputs["mix3-soft-clipped"] = np.clip(soft, 0.0, 1.0)
        for block in (1, 7, 15):
            outputs[f"oracle-{block}"], _ = oracle_select(individual, truth, block)
        baseline_seconds = {}
        if run_baselines:
            with tempfile.TemporaryDirectory(prefix="xtrans-lmmse-mixture-baseline-") as directory:
                baseline_outputs, baseline_seconds = _run_baselines(
                    runner,
                    truth,
                    int(case["origin_x"]),
                    int(case["origin_y"]),
                    Path(directory),
                )
            outputs.update(baseline_outputs)
        methods = {}
        for name, output in outputs.items():
            collector = collectors.setdefault(name, _Collector())
            methods[name] = collector.add(output, truth, name in POOLED_TAIL_METHODS)
        _, feature_maps = classify_region(context, mixture, region)
        stride = max(1, class_map.size // 1024)
        indices = np.arange(0, class_map.size, stride, dtype=np.int64)[:1024]
        for index in indices:
            feature_samples.append({
                "activity": float(feature_maps[mixture.definition.activity_feature].ravel()[index]),
                "anisotropy": float(feature_maps["anisotropy"].ravel()[index]),
                "dataset": str(case["group"]),
            })
        class_counts = np.bincount(class_map.ravel(), minlength=3)
        row = {
            "baseline_seconds": baseline_seconds,
            "case_id": case["case_id"],
            "class_counts": class_counts.tolist(),
            "class_fractions": (class_counts / np.sum(class_counts)).tolist(),
            "group": case["group"],
            "methods": methods,
        }
        if str(case["case_id"]) in ("hubble", "nasa-hydra-starfield"):
            luminance = np.mean(truth, axis=0)
            bright = luminance >= np.quantile(luminance, 0.99)
            row["bright_point_rgb_error"] = {
                name: {
                    "maximum_abs": float(np.max(np.abs(output[:, bright] - truth[:, bright]))),
                    "rms": float(np.sqrt(np.mean((output[:, bright] - truth[:, bright]) ** 2))),
                }
                for name, output in outputs.items()
            }
        rows.append(row)
        print(
            f"evaluate group={case['group']} case={case_index + 1}/{len(cases)} id={case['case_id']}",
            flush=True,
        )
    return {
        "feature_samples": feature_samples,
        "pooled": {name: collector.pooled() for name, collector in collectors.items()},
        "rows": rows,
    }


def _compact_feature_samples(
    evaluation: dict[str, object], maximum_per_dataset: int = 256
) -> None:
    """Keep the diagnostic scatter useful without making the JSON enormous."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in evaluation["feature_samples"]:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    compact = []
    for dataset in sorted(grouped):
        rows = grouped[dataset]
        if len(rows) > maximum_per_dataset:
            indices = np.linspace(0, len(rows) - 1, maximum_per_dataset, dtype=np.int64)
            rows = [rows[int(index)] for index in indices]
        compact.extend(rows)
    evaluation["feature_samples"] = compact


def _candidate_training_summary(rows: dict[str, object]) -> dict[str, object]:
    return {
        name: {
            "class_sample_counts": [int(row["sample_count"]) for row in value["classes"]],
            "fallback_count": int(value["fallback_count"]),
        }
        for name, value in rows.items()
    }


def _phase_stability(mixture: MixtureBank, gmax) -> dict[str, object]:
    y, x = np.mgrid[:48, :48]
    scenes = {
        "gradient": np.stack((0.1 + 0.7 * x / 47, 0.15 + 0.6 * y / 47, 0.2 + 0.5 * (x + y) / 94)),
        "vertical-edge": np.stack((x >= 24, 0.2 + 0.6 * (x >= 24), 1.0 - 0.8 * (x >= 24))).astype(np.float64),
        "saturated-point": np.full((3, 48, 48), 0.05, dtype=np.float64),
        "periodic": np.stack((0.2 + 0.7 * ((x + y) % 4 == 0), 0.2 + 0.7 * ((x - y) % 5 == 0), 0.2 + 0.7 * (x % 3 == 0))),
    }
    scenes["saturated-point"][:, 24, 24] = (1.0, 0.0, 0.8)
    results = []
    for name, truth in scenes.items():
        class_maps = []
        mix_metrics = []
        global_metrics = []
        for origin_x, origin_y in phase_origins():
            region = (0, 0, truth.shape[2], truth.shape[1])
            global_output = predict_region(
                gmax, truth, region, origin_x=origin_x, origin_y=origin_y
            )
            mix_output, class_map = predict_mixture(
                mixture, truth, region, origin_x=origin_x, origin_y=origin_y
            )
            global_metrics.append(_metric(global_output, truth)[0])
            mix_metrics.append(_metric(mix_output, truth)[0])
            class_maps.append(class_map)
        reference = class_maps[0]
        change_fractions = [float(np.mean(values != reference)) for values in class_maps]
        results.append({
            "assignment_change_fraction_maximum": max(change_fractions),
            "gmax_psnr_range_db": max(float(row["psnr_db"]) for row in global_metrics) - min(float(row["psnr_db"]) for row in global_metrics),
            "mix3_psnr_range_db": max(float(row["psnr_db"]) for row in mix_metrics) - min(float(row["psnr_db"]) for row in mix_metrics),
            "scene": name,
        })
    return {"scenes": results}


def _source_oracle_gap(
    established: dict[str, object], source_oracle: dict[str, object]
) -> dict[str, object]:
    oracle_by_id = {}
    for source in source_oracle["sources"]:
        selected_support = int(source["selected_support"])
        selected = next(
            row for row in source["supports"] if int(row["support"]) == selected_support
        )
        oracle_by_id[str(source["source_id"])] = selected["test"]
    rows = []
    pooled_global_sse = pooled_mix_sse = pooled_oracle_sse = 0.0
    pooled_count = 0
    for row in established["rows"]:
        source_id = str(row["case_id"])
        global_metric = row["methods"]["gmax"]
        mix_metric = row["methods"]["mix3-hard"]
        oracle_metric = oracle_by_id[source_id]
        global_mse = float(global_metric["mse"])
        mix_mse = float(mix_metric["mse"])
        oracle_mse = float(oracle_metric["mse"])
        denominator = global_mse - oracle_mse
        recovery = (global_mse - mix_mse) / denominator if abs(denominator) > 1e-30 else 0.0
        rows.append({
            "gmax_mse": global_mse,
            "mix3_mse": mix_mse,
            "recovered_remaining_mse_fraction": recovery,
            "source_id": source_id,
            "source_oracle_mse": oracle_mse,
        })
        count = int(global_metric["count"])
        pooled_global_sse += global_mse * count
        pooled_mix_sse += mix_mse * count
        pooled_oracle_sse += oracle_mse * count
        pooled_count += count
    denominator = pooled_global_sse - pooled_oracle_sse
    pooled_recovery = (
        (pooled_global_sse - pooled_mix_sse) / denominator
        if abs(denominator) > 1e-30 else 0.0
    )
    return {"pooled_recovered_fraction": pooled_recovery, "rows": rows}


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


def run(
    output: Path,
    bsds_root: Path,
    runner: Path,
    starfield: Path,
    source_oracle_path: Path,
    previous_population_path: Path,
) -> dict[str, object]:
    started = time.monotonic()
    splits = discover_expanded(bsds_root)
    external_sources = external_chromatic_sources()
    corpus = dataset_manifest(splits, external_sources)
    source_oracle = json.loads(source_oracle_path.read_text(encoding="utf-8"))
    previous_population = json.loads(previous_population_path.read_text(encoding="utf-8"))

    globals_by_size, normalization, survey = survey_and_train_globals(splits["train"])
    definitions = initial_definitions(survey)
    initial_mixtures, initial_training = train_mixtures(
        splits["train"], definitions, normalization, globals_by_size[200]
    )
    validation = {
        name: _validation_result(mixture, splits["val"])
        for name, mixture in initial_mixtures.items()
    }
    initial_selected_name = min(
        validation,
        key=lambda name: float(validation[name]["hard"]["mse"]),
    )
    sensitivity_defs = sensitivity_definitions(
        initial_mixtures[initial_selected_name].definition, survey
    )
    sensitivity_mixtures, sensitivity_training = train_mixtures(
        splits["train"], sensitivity_defs, normalization, globals_by_size[200]
    )
    sensitivity_validation = {
        name: _validation_result(mixture, splits["val"])
        for name, mixture in sensitivity_mixtures.items()
    }
    all_mixtures = {**initial_mixtures, **sensitivity_mixtures}
    all_validation = {**validation, **sensitivity_validation}
    selected_name = min(
        all_validation,
        key=lambda name: float(all_validation[name]["hard"]["mse"]),
    )
    selected = all_mixtures[selected_name]
    print(f"selected mixture={selected_name}", flush=True)

    bsds = _evaluate_cases(
        _bsds_cases(splits["test"]), globals_by_size, selected, runner,
        run_baselines=True,
    )
    established = _evaluate_cases(
        _established_cases(source_oracle, starfield), globals_by_size, selected,
        runner, run_baselines=True,
    )
    chromatic = _evaluate_cases(
        _chromatic_cases(), globals_by_size, selected, runner,
        run_baselines=True,
    )
    for evaluation in (bsds, established, chromatic):
        _compact_feature_samples(evaluation)
    filter_analysis = _filter_analysis(selected)
    phase_stability = _phase_stability(selected, globals_by_size[200])
    oracle_gap = _source_oracle_gap(established, source_oracle)

    previous_g100 = float(previous_population["dc_study"]["m2"]["test"]["psnr_db"])
    current_g100 = float(bsds["pooled"]["g100"]["psnr_db"])
    if abs(previous_g100 - current_g100) > 1e-10:
        raise RuntimeError(
            f"G100 baseline changed: previous={previous_g100}, current={current_g100}"
        )
    g100_psnr = current_g100
    gmax_psnr = float(bsds["pooled"]["gmax"]["psnr_db"])
    mix_psnr = float(bsds["pooled"]["mix3-hard"]["psnr_db"])
    oracle7_psnr = float(bsds["pooled"]["oracle-7"]["psnr_db"])
    chromatic_gmax = float(chromatic["pooled"]["gmax"]["psnr_db"])
    chromatic_mix = float(chromatic["pooled"]["mix3-hard"]["psnr_db"])
    pairwise = filter_analysis["pairwise"]
    diversity = any(
        float(row["average_cosine_similarity"]) < 0.995
        or float(row["average_l2_difference"]) > 0.05
        for row in pairwise
    )
    gate_a = diversity and oracle7_psnr - gmax_psnr >= 0.1
    gate_b = oracle7_psnr - gmax_psnr >= 0.2
    gate_c = mix_psnr - gmax_psnr >= 0.2 and chromatic_mix - chromatic_gmax >= 0.2
    star_rows = {
        str(row["case_id"]): row for row in established["rows"]
        if str(row["case_id"]) in ("hubble", "nasa-hydra-starfield")
    }
    star_safe = all(
        float(row["methods"]["mix3-hard"]["psnr_db"])
        >= float(row["methods"]["gmax"]["psnr_db"]) - 0.2
        for row in star_rows.values()
    )
    gate_d = star_safe and all(
        float(row["methods"]["mix3-hard"]["psnr_db"])
        >= float(row["methods"]["gmax"]["psnr_db"]) - 1.0
        for row in chromatic["rows"]
    )
    if not gate_a:
        decision = "NO-GO - covariance partition lacks useful bank diversity"
    elif not gate_b:
        decision = "NO-GO - three-bank oracle headroom is too small"
    elif not gate_c:
        decision = "PARTIAL - bank selection does not generalize"
    elif not gate_d:
        decision = "NO-GO - external sparse/chromatic safety failed"
    elif float(oracle_gap["pooled_recovered_fraction"]) < 0.20:
        decision = "PARTIAL - insufficient source-oracle gap recovery"
    else:
        decision = "GO - bounded adaptive LMMSE candidate"

    result = {
        "complexity": {
            "classifier": "one normalized activity/chroma statistic plus same-color directional anisotropy",
            "coefficient_count": 3 * 18 * 2 * SUPPORT * SUPPORT,
            "estimated_filter_storage_float32_bytes": 3 * (4356 + 36) * 4,
            "selected_bank_macs_per_pixel": 2 * SUPPORT * SUPPORT,
        },
        "corpus": corpus,
        "decision": {
            "gate_a_covariance_diversity": gate_a,
            "gate_b_oracle_mixture": gate_b,
            "gate_c_observable_selection": gate_c,
            "gate_d_external_safety": gate_d,
            "result": decision,
            "rules": {
                "oracle_headroom_material_db": 0.2,
                "practical_gain_each_bsds_and_external_db": 0.2,
                "source_oracle_gap_recovery_fraction": 0.20,
                "star_material_regression_db": 0.2,
            },
        },
        "elapsed_seconds": time.monotonic() - started,
        "evaluation": {
            "bsds_test": bsds,
            "established": established,
            "external_chromatic": chromatic,
        },
        "filter_analysis": filter_analysis,
        "format": "rawtherapee-xtrans-lmmse-mixture-v1",
        "global_training_scale": {
            "100": bsds["pooled"]["g100"],
            "150": bsds["pooled"]["g150"],
            "200": bsds["pooled"]["gmax"],
            "data_gain_g200_minus_g100_db": gmax_psnr - g100_psnr,
            "mixture_gain_minus_g200_db": mix_psnr - gmax_psnr,
        },
        "normalization": normalization_json(normalization),
        "phase_stability": phase_stability,
        "selected": {
            "definition": definition_json(selected.definition),
            "name": selected_name,
            "validation": all_validation[selected_name],
        },
        "source_oracle_gap": oracle_gap,
        "survey": survey_summary(survey),
        "training": {
            "candidate_summary": {
                **_candidate_training_summary(initial_training),
                **_candidate_training_summary(sensitivity_training),
            },
            "minimum_class_phase_samples": MINIMUM_CLASS_PHASE_SAMPLES,
            "sample_count_per_phase_per_source": SAMPLES_PER_PHASE_PER_SOURCE,
            "selected": (
                sensitivity_training[selected_name]
                if selected_name in sensitivity_training
                else initial_training[selected_name]
            ),
        },
        "validation_candidates": {
            name: {
                "definition": definition_json(all_mixtures[name].definition),
                "result": all_validation[name],
            }
            for name in sorted(all_validation)
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_bytes(canonical_json_bytes(_json_safe(corpus)))
    (output / "mixture.json").write_bytes(canonical_json_bytes(_json_safe(result)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--source-oracle", type=Path, required=True)
    parser.add_argument("--previous-population", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.output,
        arguments.bsds_root,
        arguments.runner,
        arguments.starfield,
        arguments.source_oracle,
        arguments.previous_population,
    )
    print(json.dumps({
        "decision": result["decision"],
        "global_training_scale": result["global_training_scale"],
        "selected": result["selected"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

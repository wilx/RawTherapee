#!/usr/bin/env python3
"""Population-trained X-Trans LMMSE and Gate 2 evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time

for _name in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes

from .dataset import discover, load_rgb, manifest as dataset_manifest
from .experiment import (
    BASELINE_METHODS,
    RIDGE_RATIOS,
    SUPPORTS,
    _load_experiment_sources,
    _metrics,
    _pooled,
    _run_baselines,
)
from .model import (
    LC_BASIS,
    FilterBank,
    PhaseStatistics,
    _dc_transform,
    collect_phase_samples,
    derive_filter_bank,
    geometry_interpolate_region,
    observation_colors,
    predict_region,
)


TRAIN_SAMPLES_PER_PHASE_PER_SOURCE = 192
VALIDATION_SAMPLES_PER_PHASE_PER_SOURCE = 256


@dataclass
class _Moment:
    count: int
    x_mean: np.ndarray
    target_mean: np.ndarray
    x_m2: np.ndarray
    target_x_m2: np.ndarray


class _Accumulator:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.rows: list[_Moment | None] = [None] * 18

    def add(
        self,
        observations: list[np.ndarray],
        targets: list[np.ndarray],
        *,
        support: int,
        dc_mode: str,
    ) -> None:
        for phase, (x_values, target_values) in enumerate(zip(observations, targets)):
            x, target, _ = _dc_transform(
                np.asarray(x_values, dtype=np.float64),
                np.asarray(target_values, dtype=np.float64),
                phase,
                dc_mode,
                support,
            )
            if target is None:
                raise RuntimeError("missing target samples")
            count = x.shape[0]
            x_mean = np.mean(x, axis=0)
            target_mean = np.mean(target, axis=0)
            centered_x = x - x_mean
            centered_target = target - target_mean
            incoming = _Moment(
                count=count,
                x_mean=x_mean,
                target_mean=target_mean,
                x_m2=centered_x.T @ centered_x,
                target_x_m2=centered_target.T @ centered_x,
            )
            current = self.rows[phase]
            if current is None:
                self.rows[phase] = incoming
                continue
            total = current.count + incoming.count
            ratio = current.count * incoming.count / total
            x_delta = incoming.x_mean - current.x_mean
            target_delta = incoming.target_mean - current.target_mean
            self.rows[phase] = _Moment(
                count=total,
                x_mean=current.x_mean + x_delta * incoming.count / total,
                target_mean=(
                    current.target_mean
                    + target_delta * incoming.count / total
                ),
                x_m2=(
                    current.x_m2
                    + incoming.x_m2
                    + ratio * np.outer(x_delta, x_delta)
                ),
                target_x_m2=(
                    current.target_x_m2
                    + incoming.target_x_m2
                    + ratio * np.outer(target_delta, x_delta)
                ),
            )

    def statistics(self) -> tuple[PhaseStatistics, ...]:
        result = []
        for row in self.rows:
            if row is None or row.count < 2:
                raise RuntimeError("incomplete population moments")
            covariance = row.x_m2 / (row.count - 1)
            covariance = (covariance + covariance.T) * 0.5
            cross = row.target_x_m2 / (row.count - 1)
            eigenvalues = np.linalg.eigvalsh(covariance)
            result.append(PhaseStatistics(
                count=row.count,
                x_mean=row.x_mean.copy(),
                target_mean=row.target_mean.copy(),
                covariance=covariance,
                cross_covariance=cross,
                minimum_eigenvalue=float(eigenvalues[0]),
                maximum_eigenvalue=float(eigenvalues[-1]),
            ))
        return tuple(result)


def _training_statistics(
    rows: list[dict[str, object]], support: int, dc_mode: str,
) -> tuple[PhaseStatistics, ...]:
    accumulator = _Accumulator(support * support)
    radius = support // 2
    for index, row in enumerate(rows):
        truth = load_rgb(row)
        observations, targets = collect_phase_samples(
            truth,
            support,
            (radius, radius, truth.shape[2] - radius, truth.shape[1] - radius),
            maximum_per_phase=TRAIN_SAMPLES_PER_PHASE_PER_SOURCE,
        )
        accumulator.add(observations, targets, support=support, dc_mode=dc_mode)
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                f"population support={support} dc={dc_mode} "
                f"training_sources={index + 1}",
                flush=True,
            )
    return accumulator.statistics()


def _sample_prediction(
    bank: FilterBank,
    phase: int,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, int]:
    x, _, dc = _dc_transform(
        observations, None, phase, bank.dc_mode, bank.support
    )
    predicted = bank.target_means[phase] + (
        x - bank.x_means[phase]
    ) @ bank.weights[phase].T
    if bank.basis == "lc":
        predicted = predicted @ LC_BASIS
    predicted += dc
    sampled_channel = int(
        observation_colors(bank.support, phase)[bank.support * bank.support // 2]
    )
    predicted[:, sampled_channel] = targets[:, sampled_channel]
    difference = predicted - targets
    return float(np.sum(difference * difference)), int(difference.size)


def _select_ridge(
    banks: list[FilterBank], validation_rows: list[dict[str, object]]
) -> tuple[FilterBank, list[dict[str, float | int]]]:
    totals = [[0.0, 0] for _ in banks]
    support = banks[0].support
    radius = support // 2
    for row in validation_rows:
        truth = load_rgb(row)
        observations, targets = collect_phase_samples(
            truth,
            support,
            (radius, radius, truth.shape[2] - radius, truth.shape[1] - radius),
            maximum_per_phase=VALIDATION_SAMPLES_PER_PHASE_PER_SOURCE,
        )
        for bank_index, bank in enumerate(banks):
            for phase in range(18):
                sse, count = _sample_prediction(
                    bank, phase, observations[phase], targets[phase]
                )
                totals[bank_index][0] += sse
                totals[bank_index][1] += count
    rows = []
    for bank, (sse, count) in zip(banks, totals):
        mse = sse / count
        rows.append({
            "count": count,
            "mse": mse,
            "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
            "ridge_ratio": bank.ridge_ratio,
            "sse": sse,
        })
    selected_index = min(range(len(rows)), key=lambda index: float(rows[index]["mse"]))
    return banks[selected_index], rows


def _train_bank(
    training_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    support: int,
    dc_mode: str,
) -> tuple[FilterBank, dict[str, object]]:
    statistics = _training_statistics(training_rows, support, dc_mode)
    banks = [
        derive_filter_bank(
            statistics, support, ridge, basis="rgb", dc_mode=dc_mode
        )
        for ridge in RIDGE_RATIOS
    ]
    selected, validation = _select_ridge(banks, validation_rows)
    return selected, {
        "condition_number_maximum": float(np.max(selected.condition_numbers)),
        "filter_norm_maximum": float(np.max(selected.filter_norms)),
        "minimum_covariance_eigenvalue": float(
            np.min(selected.minimum_eigenvalues)
        ),
        "ridge_ratio": selected.ridge_ratio,
        "sample_count": int(np.sum(selected.sample_counts)),
        "validation": validation,
    }


def _test_banks(
    test_rows: list[dict[str, object]],
    banks: dict[int, FilterBank],
    runner: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    support_metrics = {support: [] for support in banks}
    geometry_metrics = {support: [] for support in banks}
    clipped_metrics = {support: [] for support in banks}
    baseline_metrics = {method: [] for method in BASELINE_METHODS}
    with tempfile.TemporaryDirectory(prefix="xtrans-lmmse-test-") as temporary:
        root = Path(temporary)
        for source_index, source in enumerate(test_rows):
            truth = load_rgb(source)
            methods = {}
            for support, bank in banks.items():
                region = (0, 0, truth.shape[2], truth.shape[1])
                result = predict_region(bank, truth, region, clip=False)
                clipped = np.clip(result, 0.0, 1.0)
                geometry = geometry_interpolate_region(truth, support, region)
                methods[f"lmmse-{support}"] = _metrics(result, truth)
                methods[f"lmmse-{support}-clipped"] = _metrics(clipped, truth)
                methods[f"geometry-{support}"] = _metrics(geometry, truth)
                support_metrics[support].append(methods[f"lmmse-{support}"])
                clipped_metrics[support].append(methods[f"lmmse-{support}-clipped"])
                geometry_metrics[support].append(methods[f"geometry-{support}"])
            work = root / str(source["filename"])
            work.mkdir()
            outputs, timings = _run_baselines(runner, truth, 0, 0, work)
            for method, output in outputs.items():
                methods[method] = _metrics(output, truth)
                baseline_metrics[method].append(methods[method])
            rows.append({
                "baseline_seconds": timings,
                "chroma_rms": source["chroma_rms"],
                "filename": source["filename"],
                "methods": methods,
                "sha256": source["sha256"],
            })
            print(
                f"population test_source={source_index + 1}/{len(test_rows)} "
                f"filename={source['filename']}",
                flush=True,
            )
    pooled = {
        "baselines": {
            method: _pooled(metrics) for method, metrics in baseline_metrics.items()
        },
        "geometry_by_support": {
            str(support): _pooled(metrics)
            for support, metrics in geometry_metrics.items()
        },
        "lmmse_by_support": {
            str(support): _pooled(metrics)
            for support, metrics in support_metrics.items()
        },
        "lmmse_clipped_by_support": {
            str(support): _pooled(metrics)
            for support, metrics in clipped_metrics.items()
        },
    }
    return rows, pooled


def _training_size_curve(
    training_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    support: int,
    ridge_ratio: float,
) -> list[dict[str, object]]:
    milestones = (10, 25, 50, 100)
    accumulator = _Accumulator(support * support)
    radius = support // 2
    result = []
    for index, row in enumerate(training_rows):
        truth = load_rgb(row)
        observations, targets = collect_phase_samples(
            truth, support,
            (radius, radius, truth.shape[2] - radius, truth.shape[1] - radius),
            maximum_per_phase=TRAIN_SAMPLES_PER_PHASE_PER_SOURCE,
        )
        accumulator.add(observations, targets, support=support, dc_mode="m0")
        source_count = index + 1
        if source_count not in milestones:
            continue
        bank = derive_filter_bank(
            accumulator.statistics(), support, ridge_ratio,
            basis="rgb", dc_mode="m0",
        )
        metrics = []
        for test in test_rows:
            test_truth = load_rgb(test)
            output = predict_region(
                bank, test_truth, (0, 0, test_truth.shape[2], test_truth.shape[1])
            )
            metrics.append(_metrics(output, test_truth))
        result.append({
            "source_count": source_count,
            "test": _pooled(metrics),
        })
        print(f"training_size_curve sources={source_count}", flush=True)
    return result


def _dc_study(
    training_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    support: int,
    m0_bank: FilterBank,
) -> tuple[dict[str, object], dict[str, FilterBank]]:
    banks = {"m0": m0_bank}
    rows = {}
    for dc_mode in ("m1", "m2"):
        bank, training = _train_bank(
            training_rows, validation_rows, support, dc_mode
        )
        banks[dc_mode] = bank
        metrics = []
        for source in test_rows:
            truth = load_rgb(source)
            output = predict_region(bank, truth, (0, 0, truth.shape[2], truth.shape[1]))
            metrics.append(_metrics(output, truth))
        rows[dc_mode] = {"training": training, "test": _pooled(metrics)}
    rows["m0"] = {
        "note": "phase-centered affine LMMSE; mathematically equivalent to an unregularized constant column M3",
    }
    return rows, banks


def _source_oracle_retention(
    bank: FilterBank,
    source_oracle: dict[str, object],
    starfield: Path,
) -> dict[str, object]:
    sources = {
        str(row["id"]): row for row in _load_experiment_sources(starfield)
    }
    population_metrics = []
    oracle_metrics = []
    geometry_metrics = []
    rows = []
    for oracle_source in source_oracle["sources"]:
        source_id = str(oracle_source["source_id"])
        truth = np.asarray(sources[source_id]["rgb"], dtype=np.float64)
        selected_support = int(oracle_source["selected_support"])
        selected = next(
            row for row in oracle_source["supports"]
            if int(row["support"]) == selected_support
        )
        x0, y0, x1, y1 = [int(value) for value in selected["test_region"]]
        local_truth = truth[:, y0:y1, x0:x1]
        population = predict_region(bank, truth, (x0, y0, x1, y1))
        metric = _metrics(population, local_truth)
        population_metrics.append(metric)
        oracle_metrics.append(selected["test"])
        geometry_metrics.append(selected["geometry"])
        rows.append({
            "geometry_psnr_db": selected["geometry"]["psnr_db"],
            "population_psnr_db": metric["psnr_db"],
            "source_id": source_id,
            "source_oracle_psnr_db": selected["test"]["psnr_db"],
        })
    pooled_population = _pooled(population_metrics)
    pooled_oracle = _pooled(oracle_metrics)
    pooled_geometry = _pooled(geometry_metrics)
    denominator = float(pooled_oracle["psnr_db"]) - float(pooled_geometry["psnr_db"])
    retained = (
        float(pooled_population["psnr_db"]) - float(pooled_geometry["psnr_db"])
    ) / denominator
    return {
        "geometry": pooled_geometry,
        "population": pooled_population,
        "retained_oracle_gain_fraction_db": retained,
        "rows": rows,
        "source_oracle": pooled_oracle,
    }


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
) -> dict[str, object]:
    started = time.monotonic()
    splits = discover(bsds_root)
    corpus = dataset_manifest(splits)
    banks = {}
    training = {}
    for support in SUPPORTS:
        bank, diagnostics = _train_bank(
            splits["train"], splits["val"], support, "m0"
        )
        banks[support] = bank
        training[str(support)] = diagnostics
        print(
            f"population support={support} selected_ridge={bank.ridge_ratio:g}",
            flush=True,
        )
    test_rows, pooled = _test_banks(splits["test"], banks, runner.resolve())
    selected_support = max(
        SUPPORTS,
        key=lambda support: max(
            float(row["psnr_db"])
            for row in training[str(support)]["validation"]
        ),
    )
    selected_bank = banks[selected_support]
    dc_study, dc_banks = _dc_study(
        splits["train"], splits["val"], splits["test"],
        selected_support, selected_bank,
    )
    best_dc_mode = max(
        dc_banks,
        key=lambda mode: (
            float(pooled["lmmse_by_support"][str(selected_support)]["psnr_db"])
            if mode == "m0"
            else float(dc_study[mode]["test"]["psnr_db"])
        ),
    )
    final_bank = dc_banks[best_dc_mode]
    source_oracle = json.loads(source_oracle_path.read_text(encoding="utf-8"))
    retention = _source_oracle_retention(final_bank, source_oracle, starfield)
    size_curve = _training_size_curve(
        splits["train"], splits["val"], splits["test"],
        selected_support, selected_bank.ridge_ratio,
    )
    selected_test = (
        pooled["lmmse_by_support"][str(selected_support)]
        if best_dc_mode == "m0"
        else dc_study[best_dc_mode]["test"]
    )
    corrected = pooled["baselines"]["corrected-final"]
    geometry = pooled["geometry_by_support"][str(selected_support)]
    population_gain = float(selected_test["psnr_db"]) - float(geometry["psnr_db"])
    corrected_gap = float(selected_test["psnr_db"]) - float(corrected["psnr_db"])
    gate2_pass = float(retention["retained_oracle_gain_fraction_db"]) >= 0.5
    gate4_pass = population_gain >= 1.0 and corrected_gap >= -0.5
    decision = {
        "gate_2_population_generalization": {
            "pass": gate2_pass,
            "retained_source_oracle_gain_fraction_db": retention["retained_oracle_gain_fraction_db"],
            "rule": "retain at least half of source-oracle gain over geometry on the same strips",
        },
        "gate_4_held_out_benefit": {
            "corrected_final_gap_db": corrected_gap,
            "geometry_gain_db": population_gain,
            "pass": gate4_pass,
            "rule": "gain at least 1 dB over geometry and finish within 0.5 dB of corrected-final MLRI",
        },
    }
    if not gate2_pass:
        final_result = "PARTIAL - covariance/domain adaptation problem"
    elif not gate4_pass:
        final_result = "NO-GO - population model not competitive"
    else:
        final_result = "CONTINUE - sparse/coherent safety gate required"
    result = {
        "corpus": corpus,
        "dc_study": dc_study,
        "decision": {"result": final_result, **decision},
        "elapsed_seconds": time.monotonic() - started,
        "format": "rawtherapee-xtrans-lmmse-population-v1",
        "pooled": pooled,
        "selected": {
            "dc_mode": best_dc_mode,
            "support": selected_support,
            "test": selected_test,
        },
        "source_oracle_retention": retention,
        "test_sources": test_rows,
        "training": training,
        "training_size_curve": size_curve,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_bytes(
        canonical_json_bytes(_json_safe(corpus))
    )
    (output / "population.json").write_bytes(
        canonical_json_bytes(_json_safe(result))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--source-oracle", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.output, arguments.bsds_root, arguments.runner,
        arguments.starfield, arguments.source_oracle,
    )
    print(json.dumps({
        "decision": result["decision"],
        "selected": result["selected"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

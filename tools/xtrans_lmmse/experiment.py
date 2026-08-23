#!/usr/bin/env python3
"""Run the source-specific model-capacity gate for X-Trans LMMSE.

The large external population corpus is intentionally deferred until this
generous oracle proves that a compact phase-specific linear model is capable
of competitive reconstruction at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
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
from PIL import Image

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_danger.dataset import STARFIELD_BINDING, sha256_file
from tools.xtrans_hybrid.dataset import load_sources, srgb_decode
from tools.xtrans_mlri_internal.dataset import MLRI_CFA, mosaic

from .model import (
    collect_phase_samples,
    derive_filter_bank,
    geometry_interpolate_region,
    phase_statistics,
    predict_region,
)


SUPPORTS = (3, 5, 7, 9, 11)
RIDGE_RATIOS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SOURCE_IDS = ("astronaut", "brick", "grass", "gravel", "page", "hubble")
BASELINE_METHODS = ("markesteijn", "corrected-final", "ulri-slow0", "ulri-slow3")
TIMING_RE = re.compile(r"^(\S+)\t([0-9.eE+-]+)$")
METRIC_MARGIN = 12


PAPERS = (
    {
        "authors": "Javier Portilla; Deitze Otaduy; Carlos Dorronsoro",
        "doi": "10.1109/ICIP.2005.1529687",
        "local_sha256": "78d32398151b88666f1a4cc253858c0932cb98b2e24ce8668426df8e3c005079",
        "local_size": 861105,
        "title": "Low-complexity linear demosaicing using joint spatial-chromatic image statistics",
        "year": 2005,
    },
    {
        "authors": "Brice Chaix de Lavarene; David Alleysson; Jeanny Herault",
        "doi": "10.1016/j.cviu.2006.11.016",
        "local_sha256": "0731ed722f1e560af37133b7a494d85038cf7608521543ad9eb38c75d999318e",
        "local_size": 860407,
        "title": "Practical implementation of LMMSE demosaicing using luminance and chrominance spaces",
        "year": 2007,
    },
    {
        "authors": "Prakhar Amba; David Alleysson",
        "doi": "10.2352/ISSN.2169-2629.2018.26.151",
        "local_sha256": "3165a4267c4a698865990989f0d2bb8f11cf7f8622574f27bcbe36d5734b18b9",
        "local_size": 5270348,
        "title": "LMMSE Demosaicing for multicolor CFAs",
        "year": 2018,
    },
)


def _regions(width: int, height: int, support: int) -> dict[str, tuple[int, int, int, int]]:
    radius = support // 2
    train_end = max(radius + 12, int(width * 0.44))
    validation_start = max(train_end + 2 * radius + 1, int(width * 0.51))
    validation_end = max(validation_start + 12, int(width * 0.66))
    test_start = max(validation_end + 2 * radius + 1, int(width * 0.73), width - 168)
    test_end = width - radius
    if test_end - test_start < 24:
        test_start = test_end - 24
    vertical_span = min(192, height - 2 * radius)
    y0 = max(radius, (height - vertical_span) // 2)
    y1 = min(height - radius, y0 + vertical_span)
    return {
        "train": (radius, radius, train_end, height - radius),
        "validation": (validation_start, y0, validation_end, y1),
        "test": (test_start, y0, test_end, y1),
    }


def _selected_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    margin = min(METRIC_MARGIN, max(0, min(height, width) // 4))
    selected = np.zeros((height, width), dtype=bool)
    if margin == 0:
        selected[:] = True
    else:
        selected[margin:-margin, margin:-margin] = True
    return selected


def _metrics(reconstruction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    selected = _selected_mask(truth.shape[1:])
    difference = np.asarray(reconstruction, dtype=np.float64)[:, selected] - truth[:, selected]
    squared = difference * difference
    absolute = np.abs(difference)
    mse = float(np.mean(squared))
    return {
        "count": int(difference.size),
        "maximum_abs": float(np.max(absolute)),
        "mse": mse,
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": float(np.sum(squared)),
    }


def _pooled(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(row["count"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    mse = sse / count
    return {
        "case_count": len(rows),
        "count": count,
        "maximum_abs": max(float(row["maximum_abs"]) for row in rows),
        "mse": mse,
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": sse,
    }


def _load_hydra(path: Path) -> dict[str, object]:
    checked = path.resolve()
    if not checked.is_file():
        raise FileNotFoundError(checked)
    digest = sha256_file(checked)
    if digest != STARFIELD_BINDING["sha256"]:
        raise ValueError(f"NASA Hydra digest mismatch: {digest}")
    with Image.open(checked) as image:
        if image.size != (STARFIELD_BINDING["width"], STARFIELD_BINDING["height"]):
            raise ValueError(f"NASA Hydra dimensions changed: {image.size}")
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return {
        "id": "nasa-hydra-starfield",
        "filename": STARFIELD_BINDING["filename"],
        "license": STARFIELD_BINDING["license"],
        "sha256": digest,
        "source_url": STARFIELD_BINDING["source_url"],
        "rgb": np.moveaxis(srgb_decode(encoded), -1, 0),
    }


def _load_experiment_sources(starfield: Path) -> list[dict[str, object]]:
    by_id = {str(row["id"]): row for row in load_sources()}
    rows = [dict(by_id[source_id]) for source_id in SOURCE_IDS]
    rows.append(_load_hydra(starfield))
    return rows


def _run_baselines(
    runner: Path,
    truth: np.ndarray,
    origin_x: int,
    origin_y: int,
    directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    scalar, _ = mosaic(truth, origin_x, origin_y)
    input_path = directory / "input.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            str(runner), "run", str(input_path), str(directory),
            str(truth.shape[2]), str(truth.shape[1]), str(origin_x), str(origin_y),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(f"baseline runner failed:\n{completed.stdout}\n{completed.stderr}")
    outputs = {}
    for method in BASELINE_METHODS:
        values = np.fromfile(directory / f"{method}.f32le", dtype="<f4")
        if values.size != truth.size or not np.isfinite(values).all():
            raise RuntimeError(f"invalid baseline output: {method}")
        outputs[method] = values.reshape(truth.shape).astype(np.float64)
    timings = {}
    for line in completed.stdout.splitlines():
        match = TIMING_RE.match(line)
        if match and match.group(1) in BASELINE_METHODS:
            timings[match.group(1)] = float(match.group(2))
    return outputs, timings


def _source_oracle(source: dict[str, object], runner: Path, root: Path) -> dict[str, object]:
    source_id = str(source["id"])
    truth = np.asarray(source["rgb"], dtype=np.float64)
    height, width = truth.shape[1:]
    support_rows = []
    banks = {}
    started = time.monotonic()
    for support in SUPPORTS:
        print(f"source={source_id} support={support} training", flush=True)
        regions = _regions(width, height, support)
        observations, targets = collect_phase_samples(
            truth,
            support,
            regions["train"],
            maximum_per_phase=12000,
        )
        rgb_stats = phase_statistics(observations, targets, basis="rgb")
        validation_rows = []
        for ridge_ratio in RIDGE_RATIOS:
            bank = derive_filter_bank(
                rgb_stats, support, ridge_ratio, basis="rgb"
            )
            validation = predict_region(bank, truth, regions["validation"])
            validation_truth = truth[
                :, regions["validation"][1] : regions["validation"][3],
                regions["validation"][0] : regions["validation"][2],
            ]
            validation_rows.append((ridge_ratio, _metrics(validation, validation_truth), bank))
        ridge_ratio, validation_metric, bank = min(
            validation_rows, key=lambda row: float(row[1]["mse"])
        )
        test = predict_region(bank, truth, regions["test"])
        test_truth = truth[
            :, regions["test"][1] : regions["test"][3],
            regions["test"][0] : regions["test"][2],
        ]
        geometry = geometry_interpolate_region(truth, support, regions["test"])
        # Orthogonal RGB -> L,C1,C2 target regression is algebraically the same
        # linear estimator.  Measure the actual finite-precision discrepancy.
        lc_stats = phase_statistics(observations, targets, basis="lc")
        lc_bank = derive_filter_bank(lc_stats, support, ridge_ratio, basis="lc")
        lc_test = predict_region(lc_bank, truth, regions["test"])
        row = {
            "condition_number_maximum": float(np.max(bank.condition_numbers)),
            "filter_norm_maximum": float(np.max(bank.filter_norms)),
            "geometry": _metrics(geometry, test_truth),
            "lc_rgb_maximum_abs_difference": float(np.max(np.abs(lc_test - test))),
            "minimum_covariance_eigenvalue": float(np.min(bank.minimum_eigenvalues)),
            "ridge_ratio": ridge_ratio,
            "sample_count": int(np.sum(bank.sample_counts)),
            "support": support,
            "test": _metrics(test, test_truth),
            "test_region": list(regions["test"]),
            "train_region": list(regions["train"]),
            "validation": validation_metric,
            "validation_region": list(regions["validation"]),
        }
        support_rows.append(row)
        banks[support] = bank
        print(
            f"source={source_id} support={support} "
            f"validation={validation_metric['psnr_db']:.3f} "
            f"test={row['test']['psnr_db']:.3f}",
            flush=True,
        )
    selected = min(support_rows, key=lambda row: float(row["validation"]["mse"]))
    selected_support = int(selected["support"])
    selected_region = tuple(int(value) for value in selected["test_region"])
    x0, y0, x1, y1 = selected_region
    evaluation_truth = truth[:, y0:y1, x0:x1]
    source_work = root / source_id
    source_work.mkdir(parents=True)
    baselines, timings = _run_baselines(
        runner, evaluation_truth, x0 % 6, y0 % 6, source_work
    )
    baseline_metrics = {
        method: _metrics(result, evaluation_truth)
        for method, result in baselines.items()
    }
    # Preserve a compact filter snapshot for reproducible interpretability.
    bank = banks[selected_support]
    strongest = []
    radius = selected_support // 2
    from .model import phase_origins
    for phase in range(18):
        phase_origin_x, phase_origin_y = phase_origins()[phase]
        for channel in range(3):
            coefficients = bank.weights[phase, channel]
            indices = np.argsort(np.abs(coefficients), kind="stable")[-6:][::-1]
            strongest.append({
                "channel": channel,
                "coefficients": [
                    {
                        "dx": int(index % selected_support) - radius,
                        "dy": int(index // selected_support) - radius,
                        "sampled_color": int(
                            MLRI_CFA[
                                (phase_origin_y + int(index // selected_support) - radius) % 6,
                                (phase_origin_x + int(index % selected_support) - radius) % 6,
                            ]
                        ),
                        "value": float(coefficients[index]),
                    }
                    for index in indices
                ],
                "phase": phase,
            })
    return {
        "baseline_metrics": baseline_metrics,
        "baseline_seconds": timings,
        "elapsed_seconds": time.monotonic() - started,
        "height": height,
        "license": source.get("license", "not stated"),
        "selected_filter_strongest_coefficients": strongest,
        "selected_support": selected_support,
        "sha256": source.get("sha256", "not available"),
        "source_id": source_id,
        "source_url": source.get("source_url"),
        "supports": support_rows,
        "width": width,
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


def run(output: Path, runner: Path, starfield: Path) -> dict[str, object]:
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    sources = _load_experiment_sources(starfield)
    with tempfile.TemporaryDirectory(prefix="xtrans-lmmse-") as temporary:
        root = Path(temporary)
        source_rows = []
        for source in sources:
            source_rows.append(_source_oracle(source, runner, root))

    selected_oracle = []
    selected_geometry = []
    baselines = {method: [] for method in BASELINE_METHODS}
    lc_differences = []
    for source in source_rows:
        support = int(source["selected_support"])
        selected = next(row for row in source["supports"] if int(row["support"]) == support)
        selected_oracle.append(selected["test"])
        selected_geometry.append(selected["geometry"])
        lc_differences.append(float(selected["lc_rgb_maximum_abs_difference"]))
        for method in BASELINE_METHODS:
            baselines[method].append(source["baseline_metrics"][method])
    pooled_oracle = _pooled(selected_oracle)
    pooled_geometry = _pooled(selected_geometry)
    pooled_baselines = {method: _pooled(rows) for method, rows in baselines.items()}
    geometry_gain = float(pooled_oracle["psnr_db"]) - float(pooled_geometry["psnr_db"])
    corrected_gap = float(pooled_oracle["psnr_db"]) - float(
        pooled_baselines["corrected-final"]["psnr_db"]
    )
    gate_pass = geometry_gain >= 1.0 and corrected_gap >= -1.0
    result = {
        "basis_result": {
            "conclusion": "orthogonal target rotation is algebraically equivalent to RGB for one unconstrained linear predictor with common regularization",
            "maximum_observed_rgb_difference": max(lc_differences),
        },
        "decision": {
            "gate": "Gate 1 - model-class oracle",
            "pass": gate_pass,
            "result": "CONTINUE" if gate_pass else "NO-GO",
            "rule": "source oracle must gain at least 1 dB over geometry and finish within 1 dB of corrected-final MLRI",
            "stopped_before_population_corpus": not gate_pass,
        },
        "experiment_format": "rawtherapee-xtrans-lmmse-source-oracle-v1",
        "formulation": {
            "cfa_phase_count": 18,
            "covariance_precision": "float64",
            "dc": "phase-specific centered covariance and affine mean",
            "inference_precision": "float64 research implementation",
            "native_sample_policy": "restored exactly after prediction",
            "observation": "all physically observed scalar CFA values in row-major square support",
            "ridge_ratios": list(RIDGE_RATIOS),
            "supports": list(SUPPORTS),
        },
        "literature": list(PAPERS),
        "pooled": {
            "baselines": pooled_baselines,
            "geometry": pooled_geometry,
            "source_oracle": pooled_oracle,
            "source_oracle_minus_corrected_final_db": corrected_gap,
            "source_oracle_minus_geometry_db": geometry_gain,
        },
        "source_count": len(source_rows),
        "sources": source_rows,
        "split_policy": "per source: left training strip, disjoint center validation strip, distant right test strip; support-sized gaps; test metrics exclude 12-pixel boundary",
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "source-oracle.json"
    destination.write_bytes(canonical_json_bytes(_json_safe(result)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.runner, arguments.starfield)
    print(json.dumps({
        "decision": result["decision"],
        "pooled": result["pooled"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

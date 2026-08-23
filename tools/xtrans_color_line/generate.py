#!/usr/bin/env python3
"""Run the X-Trans two-color / local color-line feasibility experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

for _name in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_mlri_green_fusion.dataset import dataset_manifest, load_cases
from tools.xtrans_mlri_internal.dataset import mosaic
from tools.xtrans_oracle.generate import srgb_encode

from .analysis import (
    MARGIN,
    line_model_samples,
    oracle_gate,
    pooled_reconstruction_metrics,
    reconstruction_metrics,
    summarize_samples,
)
from .model import (
    ColorLineFit,
    fit_color_line,
    fit_color_line_huber,
    project_truth_to_line,
    reconstruct_from_line,
)


PATCH_SIZES = (3, 5, 7, 11, 15)
INITIALIZER_SIZES = (3, 7, 15)
BASELINE_METHODS = (
    "markesteijn",
    "corrected-final",
    "ulri-slow0",
    "ulri-slow3",
)
TIMING_RE = re.compile(r"^(\S+)\t([0-9.eE+-]+)$")
MAP_SOURCES = (
    "hubble",
    "nasa-hydra-starfield",
    "brick",
    "page",
    "white-impulse",
    "control-saturated_red_gray",
)


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
        if math.isnan(float(value)):
            raise ValueError("NaN cannot be serialized")
        if math.isinf(float(value)):
            return "infinity" if value > 0 else "-infinity"
        return float(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_baselines(
    runner: Path,
    truth: np.ndarray,
    origin_x: int,
    origin_y: int,
    directory: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    scalar, cfa = mosaic(truth, origin_x, origin_y)
    input_path = directory / "input.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = dict(os.environ)
    # This corpus measures reconstruction, not throughput.  Markesteijn's small-patch
    # test path is both faster and byte-stable with a single OpenMP worker.
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            str(runner),
            "run",
            str(input_path),
            str(directory),
            str(truth.shape[2]),
            str(truth.shape[1]),
            str(origin_x),
            str(origin_y),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"baseline runner failed:\n{completed.stdout}\n{completed.stderr}"
        )
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
    return outputs, scalar, cfa, timings


def _interior(cfa: np.ndarray) -> np.ndarray:
    result = np.zeros(cfa.shape, dtype=bool)
    result[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    return result


def _line_diagnostics(
    fit: ColorLineFit,
    reconstruction,
    truth: np.ndarray,
    scalar: np.ndarray,
    cfa: np.ndarray,
) -> dict[str, object]:
    interior = _interior(cfa)
    projected = project_truth_to_line(fit, truth)
    model_error = projected - truth
    position_error = reconstruction.rgb - projected
    projected_coordinate = reconstruction.projected_truth_coordinate
    if projected_coordinate is None:
        raise RuntimeError("missing projected coordinate")
    valid = interior & reconstruction.identifiable
    coordinate_error = reconstruction.direct_coordinate - projected_coordinate
    separation = fit.maximum_projection - fit.minimum_projection
    normalized_error = np.divide(
        coordinate_error,
        separation,
        out=np.zeros_like(coordinate_error),
        where=np.abs(separation) >= 1e-12,
    )
    channel_rows = {}
    for channel in range(3):
        selected = interior & (cfa == channel)
        channel_rows[str(channel)] = {
            "degenerate_fraction": float(np.mean(~reconstruction.identifiable[selected])),
            "spatial_fallback_unidentifiable_fraction": float(
                np.mean(
                    ~reconstruction.spatial_support_identifiable[
                        selected & ~reconstruction.identifiable
                    ]
                )
            )
            if np.any(selected & ~reconstruction.identifiable)
            else 0.0,
        }
    return {
        "alpha_error_rms_identifiable": float(
            np.sqrt(np.mean(normalized_error[valid] ** 2))
        )
        if np.any(valid)
        else 0.0,
        "cfa_position_error_rms": float(
            np.sqrt(np.mean(position_error[:, interior] ** 2))
        ),
        "channels": channel_rows,
        "line_model_error_rms": float(
            np.sqrt(np.mean(model_error[:, interior] ** 2))
        ),
        "native_sample_maximum_abs": float(
            max(
                np.max(
                    np.abs(
                        reconstruction.rgb[channel, cfa == channel]
                        - scalar[cfa == channel]
                    )
                )
                for channel in range(3)
            )
        ),
    }


def _sample_confidence(
    fit: ColorLineFit,
    truth: np.ndarray,
    initializer: np.ndarray,
    reconstruction: np.ndarray,
    cfa: np.ndarray,
    maximum: int = 4096,
) -> dict[str, np.ndarray]:
    interior = _interior(cfa)
    locations = np.flatnonzero(interior)
    if locations.size > maximum:
        locations = locations[
            np.linspace(0, locations.size - 1, maximum, dtype=np.int64)
        ]
    eigenvalues = fit.eigenvalues.reshape(-1, 3)[locations]
    total = np.sum(eigenvalues, axis=1)
    ratio = np.divide(
        eigenvalues[:, 0] + eigenvalues[:, 1],
        total,
        out=np.zeros_like(total),
        where=total > 1e-15,
    )
    target = truth.reshape(3, -1)[:, locations]
    initial = initializer.reshape(3, -1)[:, locations]
    result = reconstruction.reshape(3, -1)[:, locations]
    return {
        "endpoint_separation": (
            fit.maximum_projection - fit.minimum_projection
        ).ravel()[locations],
        "improvement_mse": np.mean((initial - target) ** 2, axis=0)
        - np.mean((result - target) ** 2, axis=0),
        "line_ratio": ratio,
        "reconstruction_mse": np.mean((result - target) ** 2, axis=0),
    }


def _groups(cases: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    natural = [case for case in cases if case["kind"] == "natural"]
    edge_ids = {
        "control-vertical_red_green",
        "control-horizontal_red_green",
        "control-diagonal_red_green",
        "control-saturated_red_gray",
        "control-saturated_blue_gray",
        "line-intersection",
        "thin-red-line",
    }
    pathological_ids = {
        "control-frequency_sweep",
        "control-periodic_chromatic_texture",
        "control-saturated_quadrants",
    }
    return {
        "all": list(cases),
        "natural": natural,
        "natural-sparse": [
            case
            for case in natural
            if case["source_id"] in ("hubble", "nasa-hydra-starfield")
        ],
        "natural-coherent": [
            case
            for case in natural
            if case["source_id"] in ("brick", "grass", "gravel", "page")
        ],
        "synthetic-sparse": [
            case
            for case in cases
            if case["kind"] in ("synthetic-failure", "synthetic-transition")
        ],
        "edges": [case for case in cases if case["source_id"] in edge_ids],
        "pathological": [
            case for case in cases if case["source_id"] in pathological_ids
        ],
    }


def _confidence_summary(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, object]:
    ratio = np.concatenate([np.asarray(row["line_ratio"]) for row in rows])
    error = np.concatenate([np.asarray(row["reconstruction_mse"]) for row in rows])
    improvement = np.concatenate([np.asarray(row["improvement_mse"]) for row in rows])
    separation = np.concatenate(
        [np.asarray(row["endpoint_separation"]) for row in rows]
    )
    order = np.argsort(ratio, kind="stable")
    bins = []
    for indices in np.array_split(order, 10):
        bins.append(
            {
                "line_ratio_mean": float(np.mean(ratio[indices])),
                "reconstruction_rms": float(np.sqrt(np.mean(error[indices]))),
                "mean_improvement_mse": float(np.mean(improvement[indices])),
                "fraction_improved": float(np.mean(improvement[indices] > 0.0)),
            }
        )
    return {
        "calibration_bins": bins,
        "endpoint_separation_error_spearman": float(
            stats.spearmanr(separation, error).statistic
        ),
        "line_ratio_error_spearman": float(stats.spearmanr(ratio, error).statistic),
        "line_ratio_improvement_spearman": float(
            stats.spearmanr(ratio, improvement).statistic
        ),
        "sample_count": int(ratio.size),
    }


def _case_record(case: Mapping[str, object], methods: Sequence[str]) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "degeneracy": case["degeneracy"],
        "family": case["family"],
        "kind": case["kind"],
        "methods": {
            method: reconstruction_metrics(
                np.asarray(case["truth"]),
                np.asarray(case["outputs"][method]),
                np.asarray(case["cfa"]),
            )
            for method in methods
        },
        "origin": [case["origin_x"], case["origin_y"]],
        "source_id": case["source_id"],
        "split": case["split"],
    }


def _source_rows(
    cases: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> list[dict[str, object]]:
    rows = []
    for source in sorted({str(case["source_id"]) for case in cases}):
        selected = [case for case in cases if case["source_id"] == source]
        rows.append(
            {
                "kind": selected[0]["kind"],
                "line_model": {
                    str(size): summarize_samples(
                        [case["model_samples"][str(size)] for case in selected]
                    )
                    for size in PATCH_SIZES
                },
                "methods": {
                    method: pooled_reconstruction_metrics(selected, method)
                    for method in methods
                },
                "source_id": source,
                "split": selected[0]["split"],
            }
        )
    return rows


def _decomposition_summary(
    cases: Sequence[Mapping[str, object]], patch_size: int
) -> dict[str, object]:
    rows = [case["degeneracy"][str(patch_size)] for case in cases]
    return {
        "alpha_error_rms_identifiable": float(
            np.sqrt(np.mean([row["alpha_error_rms_identifiable"] ** 2 for row in rows]))
        ),
        "cfa_position_error_rms": float(
            np.sqrt(np.mean([row["cfa_position_error_rms"] ** 2 for row in rows]))
        ),
        "channels": {
            str(channel): {
                "degenerate_fraction": float(
                    np.mean(
                        [
                            row["channels"][str(channel)]["degenerate_fraction"]
                            for row in rows
                        ]
                    )
                ),
                "spatial_fallback_unidentifiable_fraction": float(
                    np.mean(
                        [
                            row["channels"][str(channel)][
                                "spatial_fallback_unidentifiable_fraction"
                            ]
                            for row in rows
                        ]
                    )
                ),
            }
            for channel in range(3)
        },
        "line_model_error_rms": float(
            np.sqrt(np.mean([row["line_model_error_rms"] ** 2 for row in rows]))
        ),
        "native_sample_maximum_abs": float(
            max(row["native_sample_maximum_abs"] for row in rows)
        ),
    }


def _rgb_image(values: np.ndarray, scale: int = 3) -> Image.Image:
    encoded = np.moveaxis(srgb_encode(np.asarray(values)), 0, -1)
    image = Image.fromarray(np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8), "RGB")
    return image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )


def _heat_image(values: np.ndarray, scale: int = 3) -> Image.Image:
    checked = np.asarray(values, dtype=np.float64)
    mapped = np.clip(np.log10(np.maximum(checked, 1e-8)) / 8.0 + 1.0, 0.0, 1.0)
    rgb = np.stack((mapped, np.sqrt(mapped), 1.0 - mapped), axis=-1)
    image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8), "RGB")
    return image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )


def _write_map(
    case: Mapping[str, object],
    selected_size: int,
    selected_initializer: str,
    output: Path,
) -> str:
    truth = np.asarray(case["truth"])
    panels = [
        ("truth", _rgb_image(truth)),
        ("Markesteijn", _rgb_image(case["outputs"]["markesteijn"])),
        ("corrected MLRI", _rgb_image(case["outputs"]["corrected-final"])),
        (
            f"true line {selected_size}x{selected_size}",
            _rgb_image(case["outputs"][f"oracle-line-{selected_size}"]),
        ),
        (
            f"observable {selected_initializer}",
            _rgb_image(case["outputs"][selected_initializer]),
        ),
        (
            "true rank-1 residual ratio",
            _heat_image(case["selected_true_line_ratio"]),
        ),
    ]
    width = panels[0][1].width
    height = panels[0][1].height
    canvas = Image.new("RGB", (3 * width, 2 * (height + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        x = (index % 3) * width
        y = (index // 3) * (height + 24)
        draw.text((x + 4, y + 4), title, fill="black")
        canvas.paste(image, (x, y + 24))
    name = f"map-{case['source_id']}.png"
    canvas.save(output / name, format="PNG", compress_level=9)
    return name


def _noise_study(case: Mapping[str, object], patch_size: int) -> dict[str, object]:
    truth = np.asarray(case["truth"], dtype=np.float64)
    scalar = np.asarray(case["scalar"], dtype=np.float64)
    cfa = np.asarray(case["cfa"])
    clean_fit = fit_color_line(truth, patch_size)
    rows = []
    seed = int(hashlib.sha256(str(case["case_id"]).encode()).hexdigest()[:8], 16)
    for sigma in (0.001, 0.005):
        generator = np.random.default_rng(seed ^ int(sigma * 1_000_000))
        noisy_rgb = truth + generator.normal(0.0, sigma, size=truth.shape)
        noisy_scalar = scalar + generator.normal(0.0, sigma, size=scalar.shape)
        noisy_fit = fit_color_line(noisy_rgb, patch_size)
        alignment = np.abs(np.sum(clean_fit.direction * noisy_fit.direction, axis=-1))
        reconstruction = reconstruct_from_line(
            noisy_fit, noisy_scalar, cfa, truth=truth
        )
        rows.append(
            {
                "direction_one_minus_abs_dot_mean": float(np.mean(1.0 - alignment)),
                "endpoint_separation_bias_mean": float(
                    np.mean(
                        (noisy_fit.maximum_projection - noisy_fit.minimum_projection)
                        - (
                            clean_fit.maximum_projection
                            - clean_fit.minimum_projection
                        )
                    )
                ),
                "reconstruction": reconstruction_metrics(
                    truth, reconstruction.rgb, cfa
                ),
                "sigma": sigma,
            }
        )
    return {"case_id": case["case_id"], "rows": rows}


def _sparse_point_summary(
    case: Mapping[str, object], patch_size: int, selected_initializer: str
) -> dict[str, object]:
    truth = np.asarray(case["truth"], dtype=np.float64)
    cfa = np.asarray(case["cfa"])
    fit = case["true_fits"][patch_size]
    luma = np.mean(truth, axis=0)
    interior = _interior(cfa)
    candidates = np.where(interior, luma, -np.inf)
    y, x = np.unravel_index(np.argmax(candidates), candidates.shape)
    low = fit.minimum_projection[y, x]
    high = fit.maximum_projection[y, x]
    endpoint_low = fit.mean[y, x] + low * fit.direction[y, x]
    endpoint_high = fit.mean[y, x] + high * fit.direction[y, x]
    if np.mean(endpoint_low) <= np.mean(endpoint_high):
        background, bright = endpoint_low, endpoint_high
        background_name, bright_name = "low", "high"
    else:
        background, bright = endpoint_high, endpoint_low
        background_name, bright_name = "high", "low"
    truth_rgb = truth[:, y, x]
    direction = fit.direction[y, x]
    truth_coordinate = float(np.dot(truth_rgb - fit.mean[y, x], direction))
    alpha = (
        (truth_coordinate - low) / (high - low)
        if abs(high - low) >= 1e-15
        else 0.0
    )
    methods = (
        "markesteijn",
        "corrected-final",
        "ulri-slow0",
        "ulri-slow3",
        f"oracle-line-{patch_size}",
        selected_initializer,
    )
    return {
        "alpha_of_brightest_pixel": float(alpha),
        "background_endpoint_label": background_name,
        "background_endpoint_rgb": background,
        "bright_endpoint_label": bright_name,
        "bright_endpoint_rgb": bright,
        "brightest_pixel_rgb": truth_rgb,
        "brightest_pixel_xy": [int(x), int(y)],
        "endpoint_separation": float(high - low),
        "line_explained_variance": float(
            fit.eigenvalues[y, x, 2]
            / max(float(np.sum(fit.eigenvalues[y, x])), 1e-15)
        ),
        "method_rgb_errors_at_brightest_pixel": {
            method: float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(case["outputs"][method])[:, y, x]
                            - truth_rgb
                        )
                        ** 2
                    )
                )
            )
            for method in methods
        },
        "sampled_channel": int(cfa[y, x]),
        "source_id": case["source_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".xtrans-color-line-", dir=output.parent))
    work = Path(tempfile.mkdtemp(prefix="xtrans-color-line-work-"))
    started = time.perf_counter()
    try:
        cases = load_cases(args.starfield.resolve())
        for index, case in enumerate(cases):
            case_work = work / f"case-{index:03d}"
            case_work.mkdir()
            truth = np.asarray(case["truth"], dtype=np.float64)
            baseline, scalar, cfa, timings = _run_baselines(
                runner,
                truth,
                int(case["origin_x"]),
                int(case["origin_y"]),
                case_work,
            )
            outputs = dict(baseline)
            model_samples = {}
            degeneracy = {}
            true_fits = {}
            for patch_size in PATCH_SIZES:
                fit = fit_color_line(
                    truth, patch_size, compute_two_means=patch_size == 7
                )
                true_fits[patch_size] = fit
                reconstruction = reconstruct_from_line(
                    fit, scalar, cfa, truth=truth
                )
                outputs[f"oracle-line-{patch_size}"] = reconstruction.rgb
                model_samples[str(patch_size)] = line_model_samples(fit, cfa)
                degeneracy[str(patch_size)] = _line_diagnostics(
                    fit, reconstruction, truth, scalar, cfa
                )
            fit7 = true_fits[7]
            for name, parameters in (
                ("oracle-line-7-unbounded", {"bounded": False}),
                ("oracle-line-7-two-means", {"endpoint": "two-means"}),
                (
                    "oracle-line-7-centroid-fallback",
                    {"degenerate_fallback": "centroid"},
                ),
                (
                    "oracle-line-7-plane-fallback",
                    {"degenerate_fallback": "plane"},
                ),
            ):
                outputs[name] = reconstruct_from_line(
                    fit7, scalar, cfa, truth=truth, **parameters
                ).rgb

            confidence = {}
            for initializer in BASELINE_METHODS:
                for patch_size in INITIALIZER_SIZES:
                    method = f"{initializer}-line-{patch_size}"
                    fit = fit_color_line(outputs[initializer], patch_size)
                    reconstruction = reconstruct_from_line(
                        fit, scalar, cfa, truth=truth
                    )
                    outputs[method] = reconstruction.rgb
                    confidence[method] = _sample_confidence(
                        fit, truth, outputs[initializer], reconstruction.rgb, cfa
                    )
                method = f"{initializer}-line-7-leave-center"
                fit = fit_color_line(
                    outputs[initializer], 7, center_exclusion_radius=0
                )
                reconstruction = reconstruct_from_line(
                    fit, scalar, cfa, truth=truth
                )
                outputs[method] = reconstruction.rgb
                confidence[method] = _sample_confidence(
                    fit, truth, outputs[initializer], reconstruction.rgb, cfa
                )
                method = f"{initializer}-line-7-huber"
                fit = fit_color_line_huber(outputs[initializer], 7)
                reconstruction = reconstruct_from_line(
                    fit, scalar, cfa, truth=truth
                )
                outputs[method] = reconstruction.rgb
                confidence[method] = _sample_confidence(
                    fit, truth, outputs[initializer], reconstruction.rgb, cfa
                )

            case["baseline_timings"] = timings
            case["cfa"] = cfa
            case["confidence"] = confidence
            case["degeneracy"] = degeneracy
            case["model_samples"] = model_samples
            case["outputs"] = outputs
            case["scalar"] = scalar
            case["true_fits"] = true_fits
            shutil.rmtree(case_work)
            print(f"[{index + 1:02d}/{len(cases)}] {case['case_id']}", flush=True)

        train = [case for case in cases if case["split"] == "train"]
        train_size_metrics = {
            size: pooled_reconstruction_metrics(train, f"oracle-line-{size}")
            for size in PATCH_SIZES
        }
        selected_size = max(
            PATCH_SIZES, key=lambda size: train_size_metrics[size]["psnr_db"]
        )
        initializer_methods = [
            f"{initializer}-line-{size}"
            for initializer in BASELINE_METHODS
            for size in INITIALIZER_SIZES
        ] + [
            f"{initializer}-line-7-leave-center"
            for initializer in BASELINE_METHODS
        ] + [
            f"{initializer}-line-7-huber"
            for initializer in BASELINE_METHODS
        ]
        train_initializer_metrics = {
            method: pooled_reconstruction_metrics(train, method)
            for method in initializer_methods
        }
        selected_initializer = max(
            initializer_methods,
            key=lambda method: train_initializer_metrics[method]["psnr_db"],
        )

        gate_methods = []
        for support in (3, 7, 15):
            method = f"oracle-gate-{support}"
            gate_methods.append(method)
            for case in cases:
                gated, labels = oracle_gate(
                    np.asarray(case["truth"]),
                    np.asarray(case["outputs"]["corrected-final"]),
                    np.asarray(case["outputs"][f"oracle-line-{selected_size}"]),
                    np.asarray(case["cfa"]),
                    support,
                )
                case["outputs"][method] = gated
                case.setdefault("gate_line_fraction", {})[str(support)] = float(
                    np.mean(labels[_interior(np.asarray(case["cfa"]))])
                )

        methods = list(BASELINE_METHODS)
        methods += [f"oracle-line-{size}" for size in PATCH_SIZES]
        methods += [
            "oracle-line-7-unbounded",
            "oracle-line-7-two-means",
            "oracle-line-7-centroid-fallback",
            "oracle-line-7-plane-fallback",
        ]
        methods += initializer_methods + gate_methods
        groups = _groups(cases)
        aggregate = {
            name: {
                "decomposition": {
                    str(size): _decomposition_summary(selected, size)
                    for size in PATCH_SIZES
                },
                "line_model": {
                    str(size): summarize_samples(
                        [case["model_samples"][str(size)] for case in selected]
                    )
                    for size in PATCH_SIZES
                },
                "methods": {
                    method: pooled_reconstruction_metrics(selected, method)
                    for method in methods
                },
            }
            for name, selected in groups.items()
            if selected
        }

        confidence = {
            group: {
                method: _confidence_summary(
                    [case["confidence"][method] for case in selected]
                )
                for method in initializer_methods
            }
            for group, selected in groups.items()
            if selected
        }
        for case in cases:
            fit = case["true_fits"][selected_size]
            eigenvalues = fit.eigenvalues
            total = np.sum(eigenvalues, axis=-1)
            case["selected_true_line_ratio"] = np.divide(
                eigenvalues[..., 0] + eigenvalues[..., 1],
                total,
                out=np.zeros_like(total),
                where=total > 1e-15,
            )

        maps = []
        for source in MAP_SOURCES:
            case = next(
                (row for row in cases if row["source_id"] == source), None
            )
            if case is not None:
                maps.append(
                    {
                        "file": _write_map(
                            case, selected_size, selected_initializer, temporary
                        ),
                        "source_id": source,
                    }
                )

        phase_cases = [
            case for case in cases if str(case["case_id"]).startswith("cfa-impulse-")
        ]
        phase_metrics = {
            method: {
                "psnr_range_db": float(
                    max(
                        reconstruction_metrics(
                            case["truth"], case["outputs"][method], case["cfa"]
                        )["psnr_db"]
                        for case in phase_cases
                    )
                    - min(
                        reconstruction_metrics(
                            case["truth"], case["outputs"][method], case["cfa"]
                        )["psnr_db"]
                        for case in phase_cases
                    )
                ),
                "rms_values": [
                    reconstruction_metrics(
                        case["truth"], case["outputs"][method], case["cfa"]
                    )["rms"]
                    for case in phase_cases
                ],
            }
            for method in (
                f"oracle-line-{selected_size}",
                selected_initializer,
                "corrected-final",
            )
        }
        phase_metrics["scene_line_model"] = {
            "rank1_residual_ratio_mean_range": float(
                max(
                    np.mean(
                        case["model_samples"][str(selected_size)][
                            "rank1_residual_ratio"
                        ]
                    )
                    for case in phase_cases
                )
                - min(
                    np.mean(
                        case["model_samples"][str(selected_size)][
                            "rank1_residual_ratio"
                        ]
                    )
                    for case in phase_cases
                )
            ),
            "rank1_residual_ratio_means": [
                float(
                    np.mean(
                        case["model_samples"][str(selected_size)][
                            "rank1_residual_ratio"
                        ]
                    )
                )
                for case in phase_cases
            ],
        }
        noise_sources = (
            "hubble",
            "nasa-hydra-starfield",
            "brick",
            "control-saturated_red_gray",
        )
        noise = [
            _noise_study(
                next(case for case in cases if case["source_id"] == source),
                selected_size,
            )
            for source in noise_sources
        ]
        sparse_point_sources = (
            "hubble",
            "nasa-hydra-starfield",
            "white-impulse",
            "red-dot-2",
            "green-dot-3",
            "dim-blue-dot",
            "tiny-specular",
        )
        sparse_points = [
            _sparse_point_summary(
                next(case for case in cases if case["source_id"] == source),
                selected_size,
                selected_initializer,
            )
            for source in sparse_point_sources
        ]
        results = {
            "aggregate": aggregate,
            "candidate_contract": {
                "baseline_methods": list(BASELINE_METHODS),
                "degenerate_channel_delta": 1e-4,
                "initializer_sizes": list(INITIALIZER_SIZES),
                "patch_sizes": list(PATCH_SIZES),
                "selected_initializer_from_train": selected_initializer,
                "selected_true_line_size_from_train": selected_size,
            },
            "cases": [_case_record(case, methods) for case in cases],
            "cfa_only_identifiability": {
                "equations_per_patch": "N scalar observations",
                "unknowns_per_patch": (
                    "N alpha values plus six endpoint values, before gauge removal"
                ),
                "result": (
                    "underdetermined without spatial alpha or endpoint information; "
                    "no unconstrained solver is reported as a demosaicer"
                ),
            },
            "confidence": confidence,
            "format": "rawtherapee-xtrans-color-line-feasibility-results-v1",
            "gate_line_fraction": {
                group: {
                    str(support): float(
                        np.mean(
                            [case["gate_line_fraction"][str(support)] for case in selected]
                        )
                    )
                    for support in (3, 7, 15)
                }
                for group, selected in groups.items()
                if selected
            },
            "maps": maps,
            "noise_study": noise,
            "phase_study": phase_metrics,
            "sparse_point_study": sparse_points,
            "source_rows": _source_rows(cases, methods),
            "train_initializer_selection": train_initializer_metrics,
            "train_true_line_size_selection": train_size_metrics,
        }
        dataset = dataset_manifest(cases)
        dataset["format"] = "rawtherapee-xtrans-color-line-feasibility-dataset-v1"
        (temporary / "dataset.json").write_bytes(
            canonical_json_bytes(_json_safe(dataset))
        )
        (temporary / "results.json").write_bytes(
            canonical_json_bytes(_json_safe(results))
        )
        manifest = {
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
            "format": "rawtherapee-xtrans-color-line-feasibility-artifacts-v1",
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        print(f"wrote {output}")
        print(f"manifest SHA-256: {_sha256(output / 'manifest.json')}")
        print(f"elapsed seconds: {time.perf_counter() - started:.3f}")
        print(f"selected true-line size: {selected_size}")
        print(f"selected initializer: {selected_initializer}")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

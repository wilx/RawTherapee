#!/usr/bin/env python3
"""Run the ARI X-Trans missing-green feasibility experiment."""

from __future__ import annotations

import argparse
from collections import Counter
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
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_mlri_green_fusion.dataset import dataset_manifest, load_cases
from tools.xtrans_mlri_internal.dataset import mosaic
from tools.xtrans_ari.analysis import (
    MARGIN,
    calibration,
    green_metrics,
    label_spatial_metrics,
    oracle_reconstruction,
    pooled_green_metrics,
    ranking_metrics,
)
from tools.xtrans_ari.xtrans_reference import (
    BRANCHES,
    DIRECTIONS,
    ITERATIONS,
    green_candidate_bank,
)


BASELINE_METHODS = (
    "markesteijn", "corrected-final",
    "ulri-slow0", "ulri-slow1", "ulri-slow2", "ulri-slow3",
)
ADAPTIVE_METHODS = ("ri-adaptive", "mlri-adaptive", "ari")
ORACLE_WINDOWS = (1, 3, 7, 15)
TIMING_RE = re.compile(r"^(\S+)\t([0-9.eE+-]+)$")
MAP_SOURCES = (
    "hubble", "nasa-hydra-starfield", "brick", "page",
    "white-impulse", "transition-coherence-continuous-line",
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "infinity" if value > 0 else "-infinity"
        if math.isnan(float(value)):
            raise ValueError("NaN cannot be serialized")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
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
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, np.ndarray]:
    scalar, cfa = mosaic(truth, origin_x, origin_y)
    input_path = directory / "input.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run([
        str(runner), "run", str(input_path), str(directory),
        str(truth.shape[2]), str(truth.shape[1]), str(origin_x), str(origin_y),
    ], check=False, capture_output=True, text=True, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"baseline runner failed:\n{completed.stdout}\n{completed.stderr}"
        )
    outputs = {}
    pixels = truth.shape[1] * truth.shape[2]
    for method in BASELINE_METHODS:
        values = np.fromfile(directory / f"{method}.f32le", dtype="<f4")
        if values.size != truth.size or not np.isfinite(values).all():
            raise RuntimeError(f"invalid baseline output: {method}")
        outputs[method] = values.reshape(3, truth.shape[1], truth.shape[2]).astype(np.float64)[1]
    timings = {}
    for line in completed.stdout.splitlines():
        match = TIMING_RE.match(line)
        if match:
            timings[match.group(1)] = float(match.group(2))
    if set(timings) != set(BASELINE_METHODS):
        raise RuntimeError(f"baseline timing contract differs: {completed.stdout}")
    if scalar.size != pixels:
        raise RuntimeError("mosaic size differs")
    return outputs, timings, scalar, cfa


def _candidate_views(bank: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ri": bank[0].reshape(-1, *bank.shape[-2:]),
        "mlri": bank[1].reshape(-1, *bank.shape[-2:]),
        "combined": bank.reshape(-1, *bank.shape[-2:]),
    }


def _groups(cases: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    natural = [case for case in cases if case["kind"] == "natural"]
    return {
        "all": list(cases),
        "natural": natural,
        "natural-sparse": [
            case for case in natural
            if case["source_id"] in ("hubble", "nasa-hydra-starfield")
        ],
        "natural-coherent": [
            case for case in natural
            if case["source_id"] in ("brick", "grass", "gravel", "page")
        ],
        "synthetic-sparse": [
            case for case in cases
            if case["kind"] in ("synthetic-failure", "synthetic-transition")
        ],
        "synthetic-coherent": [
            case for case in cases if case["kind"] == "synthetic-success"
        ],
    }


def _weighted_average(rows: Sequence[Mapping[str, float]], name: str) -> float:
    weights = np.asarray([row["pixel_count"] for row in rows], dtype=np.float64)
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _aggregate_ranking(
    cases: Sequence[Mapping[str, object]], bank: str
) -> dict[str, float]:
    rows = [case["ranking"][bank] for case in cases]
    return {
        name: _weighted_average(rows, name)
        for name in (
            "candidate_regret_mse", "pairwise_auc", "pooled_spearman",
            "top1_accuracy", "top2_inclusion", "within_pixel_spearman",
        )
    } | {"pixel_count": int(sum(row["pixel_count"] for row in rows))}


def _diversity(case: Mapping[str, object]) -> dict[str, float]:
    bank = np.asarray(case["bank"])
    truth = np.asarray(case["truth"])[1]
    cfa = np.asarray(case["cfa"])
    interior = np.zeros_like(cfa, dtype=bool)
    interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    missing = interior & (cfa != 1)
    ri = bank[0].reshape(22, *bank.shape[-2:])[:, missing]
    mlri = bank[1].reshape(22, *bank.shape[-2:])[:, missing]
    target = truth[missing][None]
    values = np.corrcoef(ri.ravel(), mlri.ravel())[0, 1]
    errors = np.corrcoef((ri - target).ravel(), (mlri - target).ravel())[0, 1]
    ri_best = np.min((ri - target) ** 2, axis=0)
    mlri_best = np.min((mlri - target) ** 2, axis=0)
    return {
        "candidate_value_correlation": float(values),
        "error_correlation": float(errors),
        "mlri_oracle_wins_fraction": float(np.mean(mlri_best < ri_best)),
        "ri_oracle_mse": float(np.mean(ri_best)),
        "ri_oracle_wins_fraction": float(np.mean(ri_best < mlri_best)),
        "combined_oracle_marginal_mse": float(
            np.mean(mlri_best) - np.mean(np.minimum(ri_best, mlri_best))
        ),
    }


def _label_map(case: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bank = np.asarray(case["bank"])
    criteria = np.asarray(case["criteria"])
    truth = np.asarray(case["truth"])[1]
    # Published ARI retains one iteration for each branch/direction, then
    # weights those four candidates.  Map the strongest final weight.
    minima = np.min(criteria, axis=2)
    choice = np.argmin(minima.reshape(4, *minima.shape[-2:]), axis=0)
    selected_iteration = np.asarray(case["selected_iterations"]).reshape(
        4, *minima.shape[-2:]
    )
    final_iteration = np.take_along_axis(
        selected_iteration, choice[None], axis=0
    )[0]
    errors = (bank.reshape(44, *bank.shape[-2:]) - truth[None]) ** 2
    oracle = np.argmin(errors, axis=0)
    return choice, final_iteration, oracle


def _selection_summary(case: Mapping[str, object]) -> dict[str, object]:
    cfa = np.asarray(case["cfa"])
    interior = np.zeros_like(cfa, dtype=bool)
    interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    missing = interior & (cfa != 1)
    branch_direction, final_iteration, oracle = _label_map(case)
    selected = np.asarray(case["selected_iterations"])
    published = {}
    for branch in range(2):
        for direction in range(2):
            counts = np.bincount(
                selected[branch, direction][missing], minlength=ITERATIONS
            )
            published[f"{BRANCHES[branch]}-{DIRECTIONS[direction]}"] = counts.tolist()
    oracle_values = oracle[missing]
    oracle_branch = oracle_values // (2 * ITERATIONS)
    oracle_direction = (oracle_values % (2 * ITERATIONS)) // ITERATIONS
    oracle_iteration = oracle_values % ITERATIONS
    return {
        "oracle_branch_counts": np.bincount(oracle_branch, minlength=2).tolist(),
        "oracle_direction_counts": np.bincount(oracle_direction, minlength=2).tolist(),
        "oracle_iteration_counts": np.bincount(
            oracle_iteration, minlength=ITERATIONS
        ).tolist(),
        "published_iteration_counts": published,
        "strongest_branch_direction_counts": np.bincount(
            branch_direction[missing], minlength=4
        ).tolist(),
        "strongest_iteration_counts": np.bincount(
            final_iteration[missing], minlength=ITERATIONS
        ).tolist(),
    }


def _representative_candidate_summary(
    case: Mapping[str, object]
) -> dict[str, object]:
    cfa = np.asarray(case["cfa"])
    bank = np.asarray(case["bank"]).reshape(44, *cfa.shape)
    criteria = np.asarray(case["criteria"]).reshape(44, *cfa.shape)
    truth = np.asarray(case["truth"])[1]
    interior = np.zeros_like(cfa, dtype=bool)
    interior[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    missing = interior & (cfa != 1)
    criterion_mean = np.mean(criteria[:, missing], axis=1)
    error_rms = np.sqrt(np.mean((bank[:, missing] - truth[missing]) ** 2, axis=1))

    def decode(index: int) -> dict[str, object]:
        return {
            "branch": BRANCHES[index // (2 * ITERATIONS)],
            "criterion_mean": float(criterion_mean[index]),
            "direction": DIRECTIONS[(index % (2 * ITERATIONS)) // ITERATIONS],
            "iteration": int(index % ITERATIONS),
            "true_green_rms": float(error_rms[index]),
        }

    return {
        "lowest_mean_criterion": decode(int(np.argmin(criterion_mean))),
        "lowest_true_error": decode(int(np.argmin(error_rms))),
    }


def _map_image(labels: np.ndarray, count: int, scale: int = 3) -> Image.Image:
    palette = np.asarray([
        (35, 55, 150), (45, 150, 70), (210, 130, 30), (175, 45, 75),
        (80, 170, 180), (145, 85, 185), (210, 190, 50), (80, 80, 80),
        (40, 120, 210), (100, 190, 80), (235, 95, 30),
    ], dtype=np.uint8)
    colors = palette[np.asarray(labels, dtype=np.int64) % min(count, len(palette))]
    return Image.fromarray(colors, mode="RGB").resize(
        (labels.shape[1] * scale, labels.shape[0] * scale), Image.Resampling.NEAREST
    )


def _write_maps(case: Mapping[str, object], output: Path) -> str:
    branch, iteration, oracle = _label_map(case)
    selected_iterations = np.asarray(case["selected_iterations"])
    panels = [
        ("RI horizontal iteration", selected_iterations[0, 0], ITERATIONS),
        ("RI vertical iteration", selected_iterations[0, 1], ITERATIONS),
        ("MLRI horizontal iteration", selected_iterations[1, 0], ITERATIONS),
        ("MLRI vertical iteration", selected_iterations[1, 1], ITERATIONS),
        ("strongest ARI branch/direction", branch, 4),
        ("strongest ARI iteration", iteration, ITERATIONS),
        ("pixel oracle candidate", oracle, 44),
    ]
    rendered = [_map_image(values, count) for _, values, count in panels]
    width = max(image.width for image in rendered)
    row_height = rendered[0].height + 24
    canvas = Image.new("RGB", (width, row_height * len(rendered)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, ((title, _, _), image) in enumerate(zip(panels, rendered)):
        y = index * row_height
        draw.text((4, y + 4), title, fill="black")
        canvas.paste(image, (0, y + 24))
    name = f"map-{case['source_id']}.png"
    canvas.save(output / name, optimize=True)
    return name


def _case_record(case: Mapping[str, object]) -> dict[str, object]:
    methods = {
        name: green_metrics(
            np.asarray(case["truth"])[1], output, np.asarray(case["cfa"])
        )
        for name, output in case["green_outputs"].items()
    }
    methods.update(case["oracle_metrics"])
    return {
        "case_id": case["case_id"],
        "diversity": case["diversity"],
        "family": case["family"],
        "kind": case["kind"],
        "methods": methods,
        "origin": [case["origin_x"], case["origin_y"]],
        "oracle_spatial": case["oracle_spatial"],
        "ranking": case["ranking"],
        "selection": case["selection"],
        "source_id": case["source_id"],
        "split": case["split"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--reference-golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")
    reference_manifest = args.reference_golden.resolve() / "manifest.json"
    if not reference_manifest.is_file():
        raise SystemExit("Bayer reference manifest not found")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".xtrans-ari-", dir=output.parent))
    work = Path(tempfile.mkdtemp(prefix="xtrans-ari-work-"))
    try:
        cases = load_cases(args.starfield.resolve())
        for index, case in enumerate(cases):
            case_work = work / f"case-{index}"
            case_work.mkdir()
            truth = np.asarray(case["truth"], dtype=np.float64)
            baseline, baseline_timings, scalar, cfa = _run_baselines(
                runner, truth, int(case["origin_x"]), int(case["origin_y"]), case_work
            )
            start = time.perf_counter()
            bank = green_candidate_bank(scalar, cfa)
            total = time.perf_counter() - start
            views = _candidate_views(bank.candidates)
            criteria = _candidate_views(bank.criteria)
            outputs = dict(baseline)
            outputs.update({
                "ri-adaptive": bank.ri_adaptive,
                "mlri-adaptive": bank.mlri_adaptive,
                "ari": bank.adaptive,
            })
            observed_green = scalar * (cfa == 1)
            for iteration in range(ITERATIONS):
                candidate = bank.candidates[:, :, iteration].reshape(
                    4, *scalar.shape
                )
                score = bank.criteria[:, :, iteration].reshape(4, *scalar.shape)
                weights = 1.0 / (score + 1e-10)
                fixed = np.sum(weights * candidate, axis=0) / (
                    np.sum(weights, axis=0) + 1e-32
                )
                fixed = fixed * (cfa != 1) + observed_green
                outputs[f"ari-fixed-{iteration}"] = np.clip(fixed, 0.0, 1.0)
            oracle_metrics = {}
            oracle_spatial = {}
            for branch, candidates in views.items():
                pixel_labels = None
                oracle_spatial[branch] = {}
                for window in ORACLE_WINDOWS:
                    reconstruction, labels = oracle_reconstruction(
                        truth[1], candidates, cfa, window
                    )
                    if pixel_labels is None:
                        pixel_labels = labels
                    name = f"{branch}-oracle-{window}x{window}"
                    outputs[name] = reconstruction
                    oracle_metrics[name] = green_metrics(truth[1], reconstruction, cfa)
                    spatial = {
                        "candidate": label_spatial_metrics(
                            labels, cfa, reference=pixel_labels
                        )
                    }
                    if branch == "combined":
                        spatial["branch"] = label_spatial_metrics(
                            labels,
                            cfa,
                            class_divisor=2 * ITERATIONS,
                            reference=pixel_labels,
                        )
                    elif branch == "mlri":
                        spatial["direction"] = label_spatial_metrics(
                            labels,
                            cfa,
                            class_divisor=ITERATIONS,
                            reference=pixel_labels,
                        )
                    oracle_spatial[branch][f"{window}x{window}"] = spatial
            case["mosaic"] = scalar
            case["cfa"] = cfa
            case["bank"] = bank.candidates
            case["criteria"] = bank.criteria
            case["selected_iterations"] = bank.selected_iterations
            case["green_outputs"] = outputs
            case["oracle_metrics"] = oracle_metrics
            case["oracle_spatial"] = oracle_spatial
            case["ranking"] = {
                branch: ranking_metrics(
                    truth[1], views[branch], criteria[branch], cfa
                )
                for branch in ("ri", "mlri", "combined")
            }
            case["diversity"] = _diversity(case)
            case["selection"] = _selection_summary(case)
            case["timings"] = {
                **baseline_timings,
                "ari_criterion": bank.criterion_seconds,
                "ari_mlri_branch": bank.branch_seconds[1],
                "ari_ri_branch": bank.branch_seconds[0],
                "ari_total": total,
                "candidate_count": 44,
            }
            shutil.rmtree(case_work)
            print(f"[{index + 1:02d}/{len(cases)}] {case['case_id']} {total:.3f}s", flush=True)

        groups = _groups(cases)
        all_method_names = list(BASELINE_METHODS + ADAPTIVE_METHODS)
        all_method_names.extend(f"ari-fixed-{iteration}" for iteration in range(ITERATIONS))
        all_method_names.extend(
            f"{branch}-oracle-{window}x{window}"
            for branch in ("ri", "mlri", "combined")
            for window in ORACLE_WINDOWS
        )
        aggregate = {
            group: {
                "methods": {
                    method: pooled_green_metrics(group_cases, method)
                    for method in all_method_names
                },
                "ranking": {
                    branch: _aggregate_ranking(group_cases, branch)
                    for branch in ("ri", "mlri", "combined")
                },
            }
            for group, group_cases in groups.items() if group_cases
        }
        for group, row in aggregate.items():
            current = row["methods"]["corrected-final"]["rms"] ** 2
            adaptive = row["methods"]["ari"]["rms"] ** 2
            row["oracle_gap_recovery"] = {}
            for window in (1, 7, 15):
                oracle = row["methods"][f"combined-oracle-{window}x{window}"]["rms"] ** 2
                denominator = current - oracle
                row["oracle_gap_recovery"][f"{window}x{window}"] = (
                    (current - adaptive) / denominator if denominator > 0.0 else 0.0
                )

        natural_table = {}
        for source in (
            "astronaut", "brick", "grass", "gravel", "hubble",
            "nasa-hydra-starfield", "page",
        ):
            selected = [case for case in cases if case["source_id"] == source]
            natural_table[source] = {
                method: pooled_green_metrics(selected, method)["psnr_db"]
                for method in (
                    "markesteijn", "corrected-final", "ulri-slow0", "ulri-slow3",
                    "ri-adaptive", "mlri-adaptive", "ari",
                    "combined-oracle-7x7",
                )
            }
            fixed_rows = [
                pooled_green_metrics(selected, f"ari-fixed-{iteration}")["psnr_db"]
                for iteration in range(ITERATIONS)
            ]
            best_fixed = int(np.argmax(fixed_rows))
            natural_table[source]["best_fixed_ari"] = fixed_rows[best_fixed]
            natural_table[source]["best_fixed_ari_iteration"] = best_fixed

        maps = []
        seen = set()
        for source in MAP_SOURCES:
            case = next((row for row in cases if row["source_id"] == source), None)
            if case is not None and source not in seen:
                maps.append({"file": _write_maps(case, temporary), "source_id": source})
                seen.add(source)

        representative_calibration = {}
        representative_candidates = {}
        for source in ("hubble", "nasa-hydra-starfield", "brick", "page"):
            case = next(row for row in cases if row["source_id"] == source)
            representative_calibration[source] = calibration(
                np.asarray(case["truth"])[1],
                np.asarray(case["bank"]).reshape(44, *np.asarray(case["cfa"]).shape),
                np.asarray(case["criteria"]).reshape(44, *np.asarray(case["cfa"]).shape),
                np.asarray(case["cfa"]),
            )
            representative_candidates[source] = _representative_candidate_summary(case)

        results = {
            "aggregate": aggregate,
            "bayer_reference_manifest_sha256": _sha256(reference_manifest),
            "candidate_contract": {
                "branches": list(BRANCHES),
                "candidate_count": 44,
                "directions": list(DIRECTIONS),
                "iterations": ITERATIONS,
                "oracle_windows": list(ORACLE_WINDOWS),
            },
            "cases": [_case_record(case) for case in cases],
            "format": "rawtherapee-xtrans-ari-feasibility-results-v1",
            "maps": maps,
            "natural_source_green_psnr_db": natural_table,
            "reference_license": "research purpose only; all rights reserved",
            "representative_calibration": representative_calibration,
            "representative_candidates": representative_candidates,
        }
        dataset = dataset_manifest(cases)
        dataset["format"] = "rawtherapee-xtrans-ari-feasibility-dataset-v1"
        dataset["bayer_reference_manifest_sha256"] = _sha256(reference_manifest)
        (temporary / "dataset.json").write_bytes(canonical_json_bytes(_json_safe(dataset)))
        (temporary / "results.json").write_bytes(canonical_json_bytes(_json_safe(results)))
        manifest = {
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(temporary.iterdir()) if path.is_file()
            },
            "format": "rawtherapee-xtrans-ari-feasibility-artifacts-v1",
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        print(f"wrote {output}")
        print(f"manifest SHA-256: {_sha256(output / 'manifest.json')}")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

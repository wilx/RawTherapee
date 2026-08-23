#!/usr/bin/env python3
"""Run Gate 1: true endpoints with Bennett-style neighborhood alpha."""

from __future__ import annotations

import argparse
import hashlib
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
from tools.xtrans_color_line.analysis import MARGIN
from tools.xtrans_color_line.model import fit_color_line, reconstruct_from_line
from tools.xtrans_mlri_green_fusion.dataset import dataset_manifest, load_cases
from tools.xtrans_mlri_internal.dataset import mosaic, origin_cells

from .bayer_reference import full_bayer_reference
from .model import (
    bennett_alpha,
    endpoints_from_fit,
    oracle_neighbor_alpha_from_rgb,
    oracle_plane_alpha_from_rgb,
    projected_truth_alpha,
    reconstruct,
)


SUPPORT_RADII = (1, 2, 3, 5, 7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    return result


def _method_metrics(
    truth: np.ndarray,
    projected: np.ndarray,
    truth_alpha: np.ndarray,
    result: np.ndarray,
    alpha: np.ndarray,
    cfa: np.ndarray,
) -> dict[str, object]:
    selected = _mask(cfa.shape)
    difference = result[:, selected] - truth[:, selected]
    position = result[:, selected] - projected[:, selected]
    alpha_error = alpha[selected] - truth_alpha[selected]
    missing = np.stack(
        [selected & (cfa != channel) for channel in range(3)], axis=0
    )
    missing_difference = result[missing] - truth[missing]
    mse = float(np.mean(difference * difference))
    missing_mse = float(np.mean(missing_difference * missing_difference))
    absolute_alpha = np.abs(alpha_error)
    return {
        "alpha_mae": float(np.mean(absolute_alpha)),
        "alpha_p95": float(np.quantile(absolute_alpha, 0.95)),
        "alpha_p99": float(np.quantile(absolute_alpha, 0.99)),
        "alpha_rms": float(np.sqrt(np.mean(alpha_error * alpha_error))),
        "maximum_abs": float(np.max(np.abs(difference))),
        "missing_channel_psnr_db": float(-10.0 * math.log10(max(missing_mse, 1e-30))),
        "position_rms": float(np.sqrt(np.mean(position * position))),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "rgb_rms": float(np.sqrt(mse)),
    }


def _merge(rows: list[dict[str, object]]) -> dict[str, object]:
    # Equal-case RMS reproduces the aggregation used in the prior feasibility
    # report.  Per-case PSNR is not averaged directly.
    keys_rms = ("alpha_rms", "position_rms", "rgb_rms")
    result: dict[str, object] = {"case_count": len(rows)}
    for key in keys_rms:
        value = float(np.sqrt(np.mean([float(row[key]) ** 2 for row in rows])))
        result[key] = value
        if key == "rgb_rms":
            result["psnr_db"] = float(-20.0 * math.log10(max(value, 1e-15)))
    for key in ("alpha_mae", "alpha_p95", "alpha_p99"):
        result[f"mean_case_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    result["maximum_abs"] = float(max(float(row["maximum_abs"]) for row in rows))
    return result


def _groups(cases: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    natural = [case for case in cases if case["kind"] == "natural"]
    return {
        "all": cases,
        "natural": natural,
        "natural-sparse": [
            case for case in natural
            if case["source_id"] in ("hubble", "nasa-hydra-starfield")
        ],
        "natural-chromatic": [
            case for case in natural
            if case["source_id"] in ("astronaut", "hubble", "nasa-hydra-starfield")
        ],
        "natural-grayscale-controls": [
            case for case in natural
            if case["source_id"] in ("brick", "grass", "gravel", "page")
        ],
        "synthetic": [case for case in cases if case["kind"] != "natural"],
    }


def _bayer_reference() -> dict[str, object]:
    size = 49
    y, x = np.mgrid[:size, :size]
    bayer = np.where(
        y % 2 == 0,
        np.where(x % 2 == 0, 0, 1),
        np.where(x % 2 == 0, 1, 2),
    )
    first_color = np.asarray((0.07, 0.23, 0.68))
    second_color = np.asarray((0.91, 0.62, 0.11))
    first = np.broadcast_to(first_color, (size, size, 3)).copy()
    second = np.broadcast_to(second_color, (size, size, 3)).copy()
    scenes = {
        "constant": np.full((size, size), 0.37),
        "vertical-edge": (x >= size // 2).astype(np.float64),
        "horizontal-edge": (y >= size // 2).astype(np.float64),
        "diagonal-edge": (x + y >= size - 1).astype(np.float64),
        "soft-transition": np.clip((x - 16.0) / 16.0, 0.0, 1.0),
    }
    rows = {}
    selected = _mask((size, size))
    for name, alpha in scenes.items():
        truth_hwc = first + alpha[..., None] * (second - first)
        scalar = np.take_along_axis(
            truth_hwc, bayer[..., None], axis=2
        )[..., 0]
        result = bennett_alpha(
            scalar, bayer, first, second,
            support_radius=2, eta=1.0,
        )
        reconstructed = reconstruct(
            first, second, result.alpha, scalar=scalar, cfa=bayer
        )
        error = (
            reconstructed[:, selected]
            - np.moveaxis(truth_hwc, -1, 0)[:, selected]
        )
        full = full_bayer_reference(scalar, bayer, eta=1.0)
        truth_chw = np.moveaxis(truth_hwc, -1, 0)
        bootstrap_error = full.bootstrap[:, selected] - truth_chw[:, selected]
        full_error = full.rgb[:, selected] - truth_chw[:, selected]
        rows[name] = {
            "full_reproduction": {
                "bootstrap_maximum_abs_rgb": float(
                    np.max(np.abs(bootstrap_error))
                ),
                "bootstrap_rgb_rms": float(
                    np.sqrt(np.mean(bootstrap_error * bootstrap_error))
                ),
                "maximum_abs_rgb": float(np.max(np.abs(full_error))),
                "rgb_rms": float(np.sqrt(np.mean(full_error * full_error))),
            },
            "oracle_endpoint_likelihood": {
                "alpha_rms": float(
                    np.sqrt(
                        np.mean((result.alpha[selected] - alpha[selected]) ** 2)
                    )
                ),
                "maximum_abs_rgb": float(np.max(np.abs(error))),
                "rgb_rms": float(np.sqrt(np.mean(error * error))),
            },
        }
    return {
        "cfa": "RGGB, red at even/even",
        "endpoint_source": "exact synthetic two-color scene",
        "published_likelihood": "equations 5-11, 5x5 support, lambda 6, normalized 2/255 channel cutoff",
        "prior": "eta=1 flat control because the paper does not publish eta",
        "reproduction_inferences": {
            "center_weight": "1/(1+distance), because literal inverse distance is singular at the center",
            "cluster_initialization": "minimum/maximum mean-RGB intensity",
            "cluster_iterations": 8,
            "cluster_variance": "Euclidean RGB distance standard deviation",
        },
        "scenes": rows,
    }


def _phase_study() -> dict[str, object]:
    size = 72
    y, x = np.mgrid[:size, :size]
    first_color = np.asarray((0.04, 0.11, 0.20))
    second_color = np.asarray((0.96, 0.54, 0.08))
    scenes = {
        "impulse": ((x == size // 2) & (y == size // 2)).astype(np.float64),
        "saturated-point": ((x == size // 2 + 1) & (y == size // 2 - 1)).astype(np.float64),
        "two-color-edge": (x >= size // 2).astype(np.float64),
        "tiny-star": np.exp(-((x - size / 2.0) ** 2 + (y - size / 2.0) ** 2) / 1.4),
    }
    result = {}
    for scene_name, scene_alpha in scenes.items():
        truth = first_color[:, None, None] * (1.0 - scene_alpha) + second_color[:, None, None] * scene_alpha
        rows = []
        for origin_x, origin_y in origin_cells():
            case = {
                "case_id": f"phase-{scene_name}-{origin_x}-{origin_y}",
                "kind": "phase",
                "origin_x": origin_x,
                "origin_y": origin_y,
                "source_id": scene_name,
                "split": "test",
                "truth": truth,
            }
            row = _run_case(case)
            rows.append({
                "origin": row["origin"],
                "a0_position_rms": row["methods"]["a0-single-sample"]["position_rms"],
                "a1_3x3_position_rms": row["methods"]["a1-bayesian-flat-3x3"]["position_rms"],
                "a1_5x5_position_rms": row["methods"]["a1-bayesian-flat-5x5"]["position_rms"],
                "a1_low_threshold_position_rms": row["methods"]["a1-low-threshold-3x3"]["position_rms"],
            })
        result[scene_name] = {
            "phase_count": len(rows),
            "methods": {
                method: {
                    "minimum_position_rms": float(min(float(row[method]) for row in rows)),
                    "maximum_position_rms": float(max(float(row[method]) for row in rows)),
                    "range_position_rms": float(max(float(row[method]) for row in rows) - min(float(row[method]) for row in rows)),
                }
                for method in (
                    "a0_position_rms",
                    "a1_3x3_position_rms",
                    "a1_5x5_position_rms",
                    "a1_low_threshold_position_rms",
                )
            },
            "rows": rows,
        }
    return result


def _run_case(case: dict[str, object]) -> dict[str, object]:
    truth = np.asarray(case["truth"], dtype=np.float64)
    scalar, cfa = mosaic(truth, int(case["origin_x"]), int(case["origin_y"]))
    fit = fit_color_line(truth, 3)
    first, second = endpoints_from_fit(fit)
    truth_alpha = projected_truth_alpha(truth, first, second)
    projected = np.moveaxis(
        first + truth_alpha[..., None] * (second - first), -1, 0
    )

    previous = reconstruct_from_line(fit, scalar, cfa, truth=truth)
    separation = fit.maximum_projection - fit.minimum_projection
    alpha0 = np.divide(
        previous.used_coordinate - fit.minimum_projection,
        separation,
        out=np.full(separation.shape, 0.5),
        where=np.abs(separation) > 1e-20,
    )
    alpha0 = np.clip(alpha0, 0.0, 1.0)
    methods = {
        "a0-single-sample": _method_metrics(
            truth, projected, truth_alpha, previous.rgb, alpha0, cfa
        )
    }
    posterior = {}
    for radius in SUPPORT_RADII:
        bayesian = bennett_alpha(
            scalar, cfa, first, second,
            support_radius=radius, eta=1.0,
        )
        rgb = reconstruct(first, second, bayesian.alpha, scalar=scalar, cfa=cfa)
        name = f"a1-bayesian-flat-{2 * radius + 1}x{2 * radius + 1}"
        methods[name] = _method_metrics(
            truth, projected, truth_alpha, rgb, bayesian.alpha, cfa
        )
        selected = _mask(cfa.shape) & bayesian.identifiable
        posterior[name] = {
            "identifiable_fraction": float(np.mean(bayesian.identifiable[_mask(cfa.shape)])),
            "mean_variance": float(np.mean(bayesian.posterior_variance[selected])) if np.any(selected) else "infinity",
            "mean_log_likelihood_margin": float(np.mean(bayesian.log_likelihood_margin[selected])) if np.any(selected) else 0.0,
        }
    low_threshold = bennett_alpha(
        scalar,
        cfa,
        first,
        second,
        support_radius=1,
        eta=1.0,
        channel_threshold=1e-4,
    )
    low_threshold_rgb = reconstruct(
        first,
        second,
        low_threshold.alpha,
        scalar=scalar,
        cfa=cfa,
    )
    methods["a1-low-threshold-3x3"] = _method_metrics(
        truth,
        projected,
        truth_alpha,
        low_threshold_rgb,
        low_threshold.alpha,
        cfa,
    )
    # The paper does not publish eta.  Isolate its sensitivity at the faithful
    # 5x5 support without allowing the sweep to choose the Gate-1 result.
    for eta in (0.5, 0.1, 0.01):
        bayesian = bennett_alpha(
            scalar, cfa, first, second,
            support_radius=2, eta=eta,
        )
        rgb = reconstruct(first, second, bayesian.alpha, scalar=scalar, cfa=cfa)
        name = f"a1-bayesian-prior-eta-{eta:g}"
        methods[name] = _method_metrics(
            truth, projected, truth_alpha, rgb, bayesian.alpha, cfa
        )
    for radius in SUPPORT_RADII:
        neighbor = oracle_neighbor_alpha_from_rgb(
            truth, first, second, support_radius=radius
        )
        plane = oracle_plane_alpha_from_rgb(
            truth, first, second, support_radius=radius
        )
        for label, alpha in (("a2-oracle-neighbor", neighbor), ("a3-oracle-plane", plane)):
            rgb = reconstruct(first, second, alpha, scalar=scalar, cfa=cfa)
            methods[f"{label}-{2 * radius + 1}x{2 * radius + 1}"] = _method_metrics(
                truth, projected, truth_alpha, rgb, alpha, cfa
            )
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "methods": methods,
        "origin": [case["origin_x"], case["origin_y"]],
        "posterior": posterior,
        "source_id": case["source_id"],
        "split": case["split"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output exists: {args.output}; use --force")
    started = time.perf_counter()
    cases = load_cases(args.starfield.resolve())
    rows = []
    for index, case in enumerate(cases):
        rows.append(_run_case(case))
        if (index + 1) % 10 == 0:
            print(f"completed {index + 1}/{len(cases)}", flush=True)
    by_id = {row["case_id"]: row for row in rows}
    method_names = sorted(rows[0]["methods"])
    aggregates = {}
    for group, selected_cases in _groups(cases).items():
        selected_rows = [by_id[case["case_id"]] for case in selected_cases]
        aggregates[group] = {
            method: _merge([row["methods"][method] for row in selected_rows])
            for method in method_names
        }
    source_aggregates = {}
    for source_id in sorted({str(case["source_id"]) for case in cases}):
        selected_rows = [row for row in rows if row["source_id"] == source_id]
        source_aggregates[source_id] = {
            method: _merge([row["methods"][method] for row in selected_rows])
            for method in method_names
        }
    baseline = aggregates["natural"]["a0-single-sample"]["position_rms"]
    candidates = {
        method: values
        for method, values in aggregates["natural"].items()
        if method.startswith(("a1-bayesian-flat", "a1-low-threshold"))
    }
    selected_method = min(candidates, key=lambda name: candidates[name]["position_rms"])
    selected_position = candidates[selected_method]["position_rms"]
    reduction = 1.0 - selected_position / baseline
    sparse_baseline = aggregates["natural-sparse"]["a0-single-sample"]["position_rms"]
    sparse_position = aggregates["natural-sparse"][selected_method]["position_rms"]
    gate = {
        "decision": "PASS" if reduction >= 0.30 and sparse_position < sparse_baseline else "FAIL",
        "natural_position_reduction_fraction": reduction,
        "required_natural_reduction_fraction": 0.30,
        "selected_method": selected_method,
        "sparse_position_reduction_fraction": 1.0 - sparse_position / sparse_baseline,
    }
    result = {
        "aggregates": aggregates,
        "bayer_reference": _bayer_reference(),
        "case_count": len(rows),
        "cases": rows,
        "dataset": dataset_manifest(cases),
        "format": "rawtherapee-xtrans-bayesian-color-line-stage-a-v1",
        "gate_1": gate,
        "methodology": {
            "line_endpoints": "ground-truth 3x3 local PCA extrema, bounded",
            "a0": "previous one-CFA-sample line coordinate with its neighbor fallback and native-sample forcing",
            "a1": "Bennett equations 5-11 over actual X-Trans samples; flat-prior result selects Gate 1",
            "a1_low_threshold_control": "3x3 flat-prior likelihood with the preceding line experiment's 1e-4 channel cutoff",
            "a2": "weighted neighboring ground-truth projected alphas, target excluded",
            "a3": "weighted spatial alpha plane from ground-truth projected alphas, target excluded",
            "boundary": "numpy reflect",
            "lambda": 6.0,
            "normalized_channel_threshold": 2.0 / 255.0,
            "noise_sigma": 2.0 / 255.0,
            "sample_reinjection": True,
        },
        "reference": {
            "doi": "10.1007/11744023_40",
            "local_pdf": {
                "filename": "11744023_40.pdf",
                "sha256": "604f54015c22d362444dbe3063a62610c7878ee4bd84adc0f2de8b5238cfdeef",
                "size": 1037995,
            },
            "microsoft_pdf": {
                "sha256": "7706f6bdcc16025966dd7f3bf9451810916efdc0f59812ef217e968adda45489",
                "size": 1144774,
            },
            "pdf_text_identity": "pdftotext -layout outputs are byte-identical",
            "reference_code": "no author implementation located",
        },
        "phase_study": _phase_study(),
        "source_aggregates": source_aggregates,
    }
    data = canonical_json_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(gate, sort_keys=True))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print(f"sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

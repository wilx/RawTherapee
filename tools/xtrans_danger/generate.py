#!/usr/bin/env python3
"""Run the sparse/impulsive MLRI danger-gate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Sequence

# Freeze numerical reductions before NumPy/SciPy are imported.
for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import CANONICAL_XTRANS, canonical_json_bytes
from tools.xtrans_hybrid.generate import _json_safe
from tools.xtrans_oracle.generate import srgb_encode
from .analysis import (
    BLOCK_SIZE,
    FROZEN_ALPHA,
    FROZEN_RESULTS_SHA256,
    MARGIN,
    attach_frozen_blend,
    build_danger_table,
    calibration,
    fit_models,
    load_frozen_blender,
    method_metrics,
    safety_metrics,
    select_operating_point,
    threshold_curve,
)
from .dataset import (
    load_natural_cases,
    mosaic_with_cfa,
    safety_dataset_manifest,
    synthetic_cases,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_cfa(cell: np.ndarray) -> str:
    values = np.asarray(cell, dtype=np.uint8)
    if values.shape != (6, 6):
        raise ValueError("CFA cell must be 6x6")
    return "".join(str(int(value)) for value in values.ravel())


def _run_pair(
    runner: Path,
    truth: np.ndarray,
    cfa_cell: np.ndarray,
    work: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scalar, cfa = mosaic_with_cfa(truth, cfa_cell)
    input_path = work / "mosaic.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            str(runner), "run-pair-cfa", str(input_path), str(work),
            str(truth.shape[2]), str(truth.shape[1]), _encode_cfa(cfa_cell),
        ],
        env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native pair runner failed:\n{completed.stdout}\n{completed.stderr}"
        )
    pixels = truth.shape[1] * truth.shape[2]
    outputs = {}
    for name in ("markesteijn", "mlri-final"):
        values = np.fromfile(work / f"{name}.f32le", dtype="<f4")
        if values.size != 3 * pixels:
            raise RuntimeError(f"incorrect output size for {name}")
        outputs[name] = values.astype(np.float64).reshape(
            3, truth.shape[1], truth.shape[2]
        )
    return outputs, scalar, cfa


def _reconstruct_cases(
    cases: list[dict[str, object]], runner: Path, temporary: Path,
) -> None:
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['crop_id']}", flush=True)
        work = temporary / str(index)
        work.mkdir()
        cell = np.asarray(case.get("cfa_cell", CANONICAL_XTRANS), dtype=np.uint8)
        outputs, scalar, cfa = _run_pair(runner, np.asarray(case["truth"]), cell, work)
        case["mosaic"] = scalar
        case["cfa"] = cfa
        case["markesteijn"] = outputs["markesteijn"]
        case["mlri"] = outputs["mlri-final"]


def _split(table, split: str):
    return table.subset(table.splits == split)


def _comparisons(table) -> dict[str, object]:
    return {
        "markesteijn": method_metrics(table, table.mark_sse),
        "mlri": method_metrics(table, table.mlri_sse),
        f"fixed_alpha_{FROZEN_ALPHA:.3f}": method_metrics(table, table.fixed_sse),
        "practical_blender": method_metrics(table, table.blend_sse),
        "oracle_safety_gate": safety_metrics(
            table, table.blend_sse > table.mark_sse, 0.0
        ),
    }


def _select_candidate(train, validation):
    positive = train.regression[train.regression > 0]
    if positive.size < 10:
        raise RuntimeError("training corpus has too few blender regressions")
    severe_tau = float(np.percentile(positive, 75))
    target_taus = sorted(set([
        0.0,
        float(np.percentile(positive, 50)),
        severe_tau,
        float(np.percentile(positive, 90)),
    ]))
    rows = []
    runtime = []
    for tau in target_taus:
        for feature_groups in (("core",), ("impulse",), ("core", "impulse")):
            for model in fit_models(train, validation, tau, feature_groups):
                risk = model.risk(validation.features)
                curve = threshold_curve(validation, risk, severe_tau)
                point = select_operating_point(curve)
                row = {
                    "feature_groups": list(feature_groups),
                    "model": model.name,
                    "target_tau_mse": tau,
                    "validation_operating_point": point,
                    "validation_calibration": calibration(validation, risk, severe_tau),
                }
                rows.append(row)
                runtime.append((model, tau, point, curve, feature_groups))

    def rank(item):
        model, tau, point, _, feature_groups = item
        recall = float(point["severe_recall"])
        retained = float(point["aggregate_gain_retained_fraction"])
        feasible = recall >= 0.90 and retained >= 0.85
        return (
            0 if feasible else 1,
            float(point["fallback_fraction"]),
            -recall,
            -retained,
            float(point["p99_paired_rms_regression"]),
            model.name,
            tau,
            feature_groups,
        )

    selected = min(runtime, key=rank)
    return severe_tau, target_taus, rows, selected


def _per_source(table, risk: np.ndarray, threshold: float, severe_tau: float):
    rows = []
    for source in sorted(set(table.source_ids)):
        mask = table.source_ids == source
        subset = table.subset(mask)
        selected_risk = risk[mask]
        rows.append({
            "source": source,
            "comparison": _comparisons(subset),
            "selected_gate": safety_metrics(
                subset, selected_risk >= threshold, severe_tau
            ),
            "risk": {
                "mean": float(np.mean(selected_risk)),
                "p95": float(np.percentile(selected_risk, 95)),
                "maximum": float(np.max(selected_risk)),
            },
        })
    return rows


def _transition_rows(table, risk: np.ndarray, threshold: float, cases):
    metadata = {str(case["source_id"]): case.get("transition") for case in cases}
    rows = []
    for source, transition in sorted(metadata.items()):
        if transition is None:
            continue
        mask = table.source_ids == source
        if not mask.any():
            continue
        rows.append({
            "source": source,
            "transition": transition,
            "fallback_fraction": float(np.mean(risk[mask] >= threshold)),
            "mean_risk": float(np.mean(risk[mask])),
            "actual_regression_fraction": float(np.mean(table.regression[mask] > 0)),
            "mean_regression_mse": float(np.mean(table.regression[mask])),
        })
    return rows


def _font():
    return ImageFont.load_default()


def _cost_plot(curves: Sequence[tuple[str, Sequence[Mapping[str, object]]]], path: Path) -> None:
    image = Image.new("RGB", (1260, 840), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    colors = ((36, 108, 180), (220, 115, 25))
    metrics = (
        ("Severe-regression recall", lambda row: float(row["severe_recall"])),
        ("Aggregate PSNR (dB)", lambda row: float(row["gated"]["psnr_db"])),
        ("p95 patch RMS", lambda row: float(row["gated"]["patch_rms"]["p95"])),
        ("p99 patch RMS", lambda row: float(row["gated"]["patch_rms"]["p99"])),
        ("Worst paired RMS regression", lambda row: float(row["maximum_paired_rms_regression"])),
        ("Blender gain retained", lambda row: float(row["aggregate_gain_retained_fraction"])),
    )
    draw.text((12, 8), "Danger-threshold cost curves; horizontal axis is Markesteijn fallback fraction", fill=(20, 20, 20), font=_font())
    for curve_index, (name, _) in enumerate(curves):
        draw.text((710 + 220 * curve_index, 8), name, fill=colors[curve_index], font=_font())
    for panel, (title, accessor) in enumerate(metrics):
        column, row_index = panel % 3, panel // 3
        left = 55 + column * 410
        top = 55 + row_index * 385
        right, bottom = left + 365, top + 315
        draw.rectangle((left, top, right, bottom), outline=(25, 25, 25))
        all_values = [accessor(item) for _, curve in curves for item in curve]
        minimum, maximum = min(all_values), max(all_values)
        if maximum - minimum < 1e-12:
            maximum = minimum + 1.0
        margin = 0.05 * (maximum - minimum)
        minimum -= margin
        maximum += margin
        for curve_index, (_, curve) in enumerate(curves):
            points = []
            for item in sorted(curve, key=lambda value: float(value["fallback_fraction"])):
                x = left + float(item["fallback_fraction"]) * (right - left)
                y = bottom - (accessor(item) - minimum) / (maximum - minimum) * (bottom - top)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[curve_index], width=2)
        draw.text((left, top - 18), title, fill=(20, 20, 20), font=_font())
        draw.text((left, bottom + 8), "0%", fill=(20, 20, 20), font=_font())
        draw.text((right - 28, bottom + 8), "100%", fill=(20, 20, 20), font=_font())
        draw.text((right + 4, top), f"{maximum:.3g}", fill=(60, 60, 60), font=_font())
        draw.text((right + 4, bottom - 10), f"{minimum:.3g}", fill=(60, 60, 60), font=_font())
    image.save(path, format="PNG", compress_level=9)


def _diagnostic_map(case, rows, risk: np.ndarray, threshold: float, path: Path) -> None:
    truth = np.asarray(case["truth"])
    mark = np.asarray(case["markesteijn"])
    blend = np.asarray(case["blend"])
    height, width = truth.shape[1:]
    risk_field = np.zeros((height, width))
    danger_field = np.zeros((height, width), dtype=bool)
    for value, bound in zip(risk, rows.bounds):
        y, y2, x, x2 = bound
        risk_field[y:y2, x:x2] = value
        danger_field[y:y2, x:x2] = value >= threshold
    gated = blend.copy()
    gated[:, danger_field] = mark[:, danger_field]
    panels = (
        ("truth", srgb_encode(truth)),
        ("Markesteijn", srgb_encode(mark)),
        ("ungated blender", srgb_encode(blend)),
        ("danger risk", np.repeat(risk_field[None], 3, axis=0)),
        ("gated", srgb_encode(gated)),
    )
    zoom = 2
    output = Image.new("RGB", (len(panels) * width * zoom, height * zoom + 24), "white")
    draw = ImageDraw.Draw(output)
    for index, (label, values) in enumerate(panels):
        encoded = np.moveaxis(np.clip(values, 0, 1), 0, -1)
        panel = Image.fromarray(np.rint(encoded * 255).astype(np.uint8), "RGB")
        panel = panel.resize((width * zoom, height * zoom), Image.Resampling.NEAREST)
        output.paste(panel, (index * width * zoom, 24))
        draw.text((index * width * zoom + 4, 5), label, fill=(20, 20, 20), font=_font())
    output.save(path, format="PNG", compress_level=9)


def _decision(natural_metrics, synthetic_metrics, per_source) -> tuple[str, dict[str, object]]:
    source_map = {row["source"]: row for row in per_source}
    sparse_sources = [source_map[name] for name in ("hubble", "nasa-hydra-starfield")]
    sparse_recall = [float(row["selected_gate"]["severe_recall"]) for row in sparse_sources]
    recall = float(natural_metrics["severe_recall"])
    retained = float(natural_metrics["aggregate_gain_retained_fraction"])
    fallback = float(natural_metrics["fallback_fraction"])
    ungated_p99 = float(natural_metrics["ungated_blender"]["patch_rms"]["p99"])
    gated_p99 = float(natural_metrics["gated"]["patch_rms"]["p99"])
    tail_reduced = gated_p99 <= 0.8 * ungated_p99
    criteria = {
        "overall_severe_recall_at_least_0_90": recall >= 0.90,
        "both_sparse_sources_severe_recall_at_least_0_80": min(sparse_recall) >= 0.80,
        "synthetic_severe_recall_at_least_0_80": float(synthetic_metrics["severe_recall"]) >= 0.80,
        "p99_patch_rms_reduced_at_least_20_percent": tail_reduced,
        "blender_gain_retained_at_least_0_80": retained >= 0.80,
        "fallback_fraction_at_most_0_35": fallback <= 0.35,
        "sparse_source_recalls": sparse_recall,
    }
    required = [value for value in criteria.values() if isinstance(value, bool)]
    if all(required):
        decision = "GO"
    elif recall >= 0.75 and tail_reduced and min(sparse_recall) >= 0.60:
        decision = "PARTIAL"
    else:
        decision = "NO-GO"
    return decision, criteria


def generate(
    output: Path,
    runner: Path,
    starfield: Path,
    frozen_results: Path,
    *,
    force: bool,
) -> Path:
    output = output.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output.iterdir():
            if path.is_file():
                path.unlink()

    natural, starfield_binding = load_natural_cases(starfield)
    synthetic = synthetic_cases()
    cases = [*natural, *synthetic]
    dataset = safety_dataset_manifest(natural, synthetic, starfield_binding)
    dataset_bytes = canonical_json_bytes(dataset)
    (output / "dataset.json").write_bytes(dataset_bytes)

    with tempfile.TemporaryDirectory(prefix="rt-xtrans-danger-") as temporary:
        _reconstruct_cases(cases, runner.resolve(), Path(temporary))

    blender = load_frozen_blender(frozen_results)
    attach_frozen_blend(cases, blender)
    table = build_danger_table(cases)
    train, validation, test = (_split(table, name) for name in ("train", "validation", "test"))
    test_natural = test.subset(test.kinds == "natural")
    test_synthetic = test.subset(test.kinds == "synthetic")
    severe_tau, target_taus, candidate_rows, selected = _select_candidate(train, validation)
    model, target_tau, point, validation_curve, selected_feature_groups = selected
    threshold = float(point["threshold"])

    evaluation_tables = {
        "train": train,
        "validation": validation,
        "test_combined": test,
        "test_natural": test_natural,
        "test_synthetic": test_synthetic,
    }
    risks = {
        split: model.risk(split_table.features)
        for split, split_table in evaluation_tables.items()
    }
    selected_metrics = {
        split: safety_metrics(split_table, risks[split] >= threshold, severe_tau)
        for split, split_table in evaluation_tables.items()
    }
    test_natural_curve = threshold_curve(test_natural, risks["test_natural"], severe_tau)
    test_synthetic_curve = threshold_curve(test_synthetic, risks["test_synthetic"], severe_tau)
    per_source = _per_source(test, risks["test_combined"], threshold, severe_tau)
    transition_rows = _transition_rows(test, risks["test_combined"], threshold, cases)
    decision, criteria = _decision(
        selected_metrics["test_natural"], selected_metrics["test_synthetic"],
        per_source,
    )

    results = {
        "format": "rawtherapee-xtrans-danger-results-v1",
        "experiment": "asymmetric sparse-structure danger gate before frozen 7x7 Markesteijn-MLRI blender",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "frozen_blender": {
            "results_sha256": FROZEN_RESULTS_SHA256,
            "block_size": BLOCK_SIZE,
            "spatial_variant": "per_pixel_box_7",
            "retrained": False,
        },
        "leakage_policy": {
            "feature_inputs": ["mosaic", "CFA", "Markesteijn", "MLRI"],
            "ground_truth_used_only_for": ["danger labels", "severity", "evaluation"],
            "untouched_test_sources": ["astronaut", "brick", "hubble", "page", "nasa-hydra-starfield"],
            "test_data_used_for_selection": False,
        },
        "patch_counts": {
            split: int(split_table.features.shape[0])
            for split, split_table in evaluation_tables.items()
        },
        "danger_target": {
            "definition": "(practical blender SSE - Markesteijn SSE) / RGB sample count > tau",
            "target_taus_mse": target_taus,
            "fixed_severe_tau_mse_from_training_positive_p75": severe_tau,
            "severity_counts": {
                split: {
                    "safe_or_better": int(np.sum(split_table.regression <= 0)),
                    "moderate_positive": int(np.sum((split_table.regression > 0) & (split_table.regression <= severe_tau))),
                    "severe": int(np.sum(split_table.regression > severe_tau)),
                }
                for split, split_table in (
                    ("train", train), ("validation", validation), ("test_combined", test),
                    ("test_natural", test_natural), ("test_synthetic", test_synthetic),
                )
            },
        },
        "feature_contract": {
            "count": len(table.feature_names),
            "groups": {name: list(values) for name, values in table.feature_groups.items()},
            "names": list(table.feature_names),
        },
        "candidate_validation_rows": candidate_rows,
        "selected": {
            "model": model.serialize(table.feature_names),
            "model_name": model.name,
            "feature_groups": list(selected_feature_groups),
            "target_tau_mse": target_tau,
            "threshold": threshold,
            "selection": "minimum validation fallback subject to >=90% severe recall and >=85% retained gain; deterministic tie breaks",
            "metrics": selected_metrics,
            "calibration": {
                split: calibration(split_table, risks[split], severe_tau)
                for split, split_table in evaluation_tables.items()
            },
        },
        "baselines": {
            split: _comparisons(split_table)
            for split, split_table in evaluation_tables.items()
        },
        "cost_curves": {
            "validation": validation_curve,
            "test_natural_untouched_until_final_evaluation": test_natural_curve,
            "test_synthetic_untouched_until_final_evaluation": test_synthetic_curve,
        },
        "held_out_per_source": per_source,
        "coherence_transitions": transition_rows,
        "decision": decision,
        "decision_criteria": criteria,
    }
    result_bytes = canonical_json_bytes(_json_safe(results))
    (output / "results.json").write_bytes(result_bytes)
    _cost_plot(
        (("validation", validation_curve), ("held-out natural", test_natural_curve)),
        output / "danger-cost-curve.png",
    )

    # Diagnostic maps are generated only after the operating point is frozen.
    for source in ("hubble", "nasa-hydra-starfield", "astronaut", "brick", "page"):
        matching = [case for case in cases if case["source_id"] == source]
        if not matching:
            continue
        case = matching[0]
        mask = test.crop_ids == case["crop_id"]
        rows = test.subset(mask)
        _diagnostic_map(
            case, rows, model.risk(rows.features), threshold,
            output / f"danger-map-{source}.png",
        )

    assets = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        assets.append({
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": _sha256(path),
        })
    manifest = {
        "assets": assets,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "format": "rawtherapee-xtrans-danger-corpus-v1",
        "frozen_blender_results_sha256": FROZEN_RESULTS_SHA256,
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({
        "decision": decision,
        "manifest_sha256": _sha256(manifest_path),
        "model": model.name,
        "threshold": threshold,
    }, sort_keys=True))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--output", type=Path, default=root / "devnotes/images/xtrans-danger")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument(
        "--frozen-results", type=Path,
        default=root / "devnotes/images/xtrans-hybrid/results.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(
        args.output, args.runner, args.starfield, args.frozen_results,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

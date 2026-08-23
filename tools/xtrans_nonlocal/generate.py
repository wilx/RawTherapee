#!/usr/bin/env python3
"""Run the X-Trans nonlocal patch-recurrence upper-bound experiment."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Sequence

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_danger.analysis import load_frozen_blender
from tools.xtrans_hybrid.generate import _json_safe
from tools.xtrans_oracle.generate import srgb_encode
from .analysis import (
    PATCH_RADIUS,
    PATCH_SIZE,
    analyze_target,
    cfa_for_context,
    matching_metrics,
    mosaic,
    pooled_metrics,
    target_centers,
)
from .dataset import CONTEXT_HALO, TARGET_SIZE, dataset_manifest, load_cases


FROZEN_BLENDER = Path("devnotes/images/xtrans-hybrid/results.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_cfa(cell: np.ndarray) -> str:
    return "".join(str(int(value)) for value in np.asarray(cell).ravel())


def _run_pair(
    runner: Path, truth: np.ndarray, cell: np.ndarray, work: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cfa = np.tile(cell, ((truth.shape[1] + 5) // 6, (truth.shape[2] + 5) // 6))[
        :truth.shape[1], :truth.shape[2]
    ]
    scalar = mosaic(truth, cfa)
    input_path = work / "mosaic.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            str(runner), "run-pair-cfa", str(input_path), str(work),
            str(truth.shape[2]), str(truth.shape[1]), _encode_cfa(cell),
        ],
        env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"native runner failed:\n{completed.stdout}\n{completed.stderr}")
    outputs = {}
    for name in ("markesteijn", "mlri-final"):
        values = np.fromfile(work / f"{name}.f32le", dtype="<f4")
        if values.size != truth.size:
            raise RuntimeError(f"incorrect native output size for {name}")
        outputs[name] = values.astype(np.float64).reshape(truth.shape)
    return outputs, scalar, cfa


def _prepare_case(case: dict[str, object], runner: Path, work: Path, blender) -> None:
    context = np.asarray(case["context"], dtype=np.float64)
    context_cfa = cfa_for_context(*context.shape[1:], case["context_global_origin"])
    context_scalar = mosaic(context, context_cfa)
    x0, y0 = case["target_origin"]
    truth = context[:, y0:y0 + TARGET_SIZE, x0:x0 + TARGET_SIZE]
    target_cfa = context_cfa[y0:y0 + TARGET_SIZE, x0:x0 + TARGET_SIZE]
    outputs, scalar, cfa = _run_pair(runner, truth, target_cfa[:6, :6], work)
    blend_case = {
        "mosaic": scalar,
        "cfa": cfa,
        "markesteijn": outputs["markesteijn"],
        "mlri": outputs["mlri-final"],
    }
    alpha = blender.alpha_field(blend_case)
    blend = outputs["markesteijn"] + alpha[None] * (
        outputs["mlri-final"] - outputs["markesteijn"]
    )
    baselines = {}
    for name, values in (
        ("markesteijn", outputs["markesteijn"]),
        ("mlri", outputs["mlri-final"]),
        ("practical_blender", blend),
    ):
        canvas = np.zeros_like(context)
        canvas[:, y0:y0 + TARGET_SIZE, x0:x0 + TARGET_SIZE] = values
        baselines[name] = canvas
    case["cfa"] = context_cfa
    case["mosaic"] = context_scalar
    case["baselines"] = baselines


def _subset(rows: Sequence[Mapping[str, object]], predicate) -> list[Mapping[str, object]]:
    return [row for row in rows if predicate(row)]


def _headroom(
    rows: Sequence[Mapping[str, object]], phase: str, donor_count: str = "best1",
) -> dict[str, float]:
    baseline = np.asarray([row["errors"]["practical_blender"]["mse"] for row in rows])
    oracle = np.asarray([
        row["errors"][f"{phase}_oracle_{donor_count}_donor_upper_bound"]["mse"]
        for row in rows
    ])
    observable = np.asarray([
        row["errors"][f"{phase}_observable_{donor_count}_donor_upper_bound"]["mse"]
        for row in rows
    ])
    base, ora, obs = float(np.mean(baseline)), float(np.mean(oracle)), float(np.mean(observable))
    available = base - ora
    return {
        "baseline_mean_mse": base,
        "donor_count": donor_count,
        "oracle_mean_mse": ora,
        "observable_selection_mean_mse": obs,
        "oracle_gain_db": math.inf if ora <= 0 else 10 * math.log10(base / ora),
        "observable_gain_db": math.inf if obs <= 0 else 10 * math.log10(base / obs),
        "recoverable_fraction_of_oracle_mse_reduction": (
            (base - obs) / available if available > 0 else 0.0
        ),
    }


def _group(rows: Sequence[Mapping[str, object]], key: str) -> list[dict[str, object]]:
    result = []
    for value in sorted({str(row[key]) for row in rows}):
        selected = [row for row in rows if row[key] == value]
        result.append({
            key: value,
            "patch_count": len(selected),
            "same_phase_matching": matching_metrics(selected, "same_phase"),
            "cross_phase_matching": matching_metrics(selected, "cross_phase"),
            "same_phase_headroom_best1": _headroom(selected, "same_phase", "best1"),
            "same_phase_headroom_top8": _headroom(selected, "same_phase", "top8"),
            "same_phase_headroom_top32": _headroom(selected, "same_phase", "top32"),
            "cross_phase_headroom_best1": _headroom(selected, "cross_phase", "best1"),
            "cross_phase_headroom_top8": _headroom(selected, "cross_phase", "top8"),
            "cross_phase_headroom_top32": _headroom(selected, "cross_phase", "top32"),
            "quality": pooled_metrics(selected),
        })
    return result


def _compact_row(row: Mapping[str, object]) -> dict[str, object]:
    """Keep per-patch audit data without duplicating aggregate error summaries."""

    return {
        key: row[key]
        for key in (
            "center_yx", "cross_phase", "patch_index", "same_phase",
            "scene_class", "source_id", "split", "texture_class",
        )
    } | {
        "error_mse": {
            name: metrics["mse"] for name, metrics in row["errors"].items()
        }
    }


def _font():
    return ImageFont.load_default()


def _scatter(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    image = Image.new("RGB", (900, 720), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 85, 55, 850, 660
    draw.rectangle((left, top, right, bottom), outline=(25, 25, 25))
    colors = {
        "repeated texture": (31, 119, 180),
        "repeated text and edges": (31, 119, 180),
        "sparse colored point field": (214, 39, 40),
        "frequency sweep": (148, 103, 189),
    }
    values = []
    for row in rows:
        x = max(1e-8, float(row["same_phase"]["best_oracle_rgb_mse"]))
        y = max(1e-8, float(row["same_phase"]["best_cfa_selected_rgb_mse"]))
        values.append((x, y, colors.get(str(row["scene_class"]), (80, 130, 80))))
    minimum, maximum = -8.0, 0.0
    for x, y, color in values:
        px = left + (math.log10(x) - minimum) / (maximum - minimum) * (right - left)
        py = bottom - (math.log10(y) - minimum) / (maximum - minimum) * (bottom - top)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    draw.line((left, bottom, right, top), fill=(100, 100, 100), width=1)
    draw.text((14, 12), "Same-phase neighbor purity: CFA-selected RGB error vs RGB-oracle error", fill=(20, 20, 20), font=_font())
    draw.text((330, 680), "best oracle-neighbor RGB MSE (log10)", fill=(20, 20, 20), font=_font())
    draw.text((8, 330), "CFA-selected RGB MSE", fill=(20, 20, 20), font=_font())
    image.save(path, format="PNG", compress_level=9)


def _diagnostic(
    case: Mapping[str, object], row: Mapping[str, object], reconstructions: Mapping[str, np.ndarray], path: Path,
) -> None:
    center = tuple(row["center_yx"])
    y, x = center
    truth = np.asarray(case["context"])[:, y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1]
    baselines = case["baselines"]
    panels = (
        ("truth", truth),
        ("Markesteijn", np.asarray(baselines["markesteijn"])[:, y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1]),
        ("local blender", np.asarray(baselines["practical_blender"])[:, y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1]),
        ("cross CFA NLM", reconstructions["cross_phase_observable_direct_nlm"]),
        ("cross low-rank", reconstructions["cross_phase_observable_low_rank"]),
        ("same oracle best1", reconstructions["same_phase_oracle_best1_donor_upper_bound"]),
    )
    scale = 20
    output = Image.new("RGB", (len(panels) * PATCH_SIZE * scale, PATCH_SIZE * scale + 24), "white")
    draw = ImageDraw.Draw(output)
    for index, (label, values) in enumerate(panels):
        encoded = np.moveaxis(srgb_encode(values), 0, -1)
        panel = Image.fromarray(np.rint(encoded * 255).astype(np.uint8), "RGB")
        panel = panel.resize((PATCH_SIZE * scale, PATCH_SIZE * scale), Image.Resampling.NEAREST)
        output.paste(panel, (index * PATCH_SIZE * scale, 24))
        draw.text((index * PATCH_SIZE * scale + 3, 5), label, fill=(20, 20, 20), font=_font())
    output.save(path, format="PNG", compress_level=9)


def _decision(results: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    analysis_rows = results["natural_analysis_rows"]
    safety_rows = results["safety_rows"]
    matching = matching_metrics(analysis_rows, "same_phase")
    headroom = _headroom(analysis_rows, "same_phase", "best1")
    quality = pooled_metrics(analysis_rows)
    local_mse = quality["practical_blender"]["mean_patch_mse"]
    candidate_names = (
        "cross_phase_observable_direct_nlm", "cross_phase_observable_low_rank",
    )
    selected = min(candidate_names, key=lambda name: quality[name]["mean_patch_mse"])
    candidate_mse = quality[selected]["mean_patch_mse"]
    safety_quality = pooled_metrics(safety_rows)
    safety_ratio = (
        safety_quality[selected]["mean_patch_mse"]
        / safety_quality["markesteijn"]["mean_patch_mse"]
    )
    repeated = [
        row for row in analysis_rows
        if row["scene_class"] in ("repeated texture", "repeated text and edges")
    ]
    criteria = {
        "same_phase_oracle_gain_at_least_0_5_db": headroom["oracle_gain_db"] >= 0.5,
        "observable_selection_recovers_at_least_30_percent_oracle_reduction": headroom["recoverable_fraction_of_oracle_mse_reduction"] >= 0.30,
        "at_least_25_percent_repeated_patches_have_rgb_rms_below_0_05": matching_metrics(repeated, "same_phase")["useful_recurrence_fraction_rgb_rms_below_0_05"] >= 0.25,
        "implementable_cross_phase_method_gains_at_least_0_1_db": 10 * math.log10(local_mse / candidate_mse) >= 0.1,
        "sparse_point_safety_mse_no_more_than_10_percent_above_markesteijn": safety_ratio <= 1.10,
        "selected_implementable_method": selected,
        "selected_analysis_gain_db": 10 * math.log10(local_mse / candidate_mse),
        "selected_safety_to_markesteijn_mse_ratio": safety_ratio,
    }
    required = [value for value in criteria.values() if isinstance(value, bool)]
    return ("GO" if all(required) else "NO-GO"), criteria


def generate(output: Path, runner: Path, starfield: Path, *, force: bool) -> Path:
    output = output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(manifest_path)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output.iterdir():
            if path.is_file():
                path.unlink()

    cases, starfield_binding = load_cases(starfield)
    dataset = dataset_manifest(cases, starfield_binding)
    (output / "dataset.json").write_bytes(canonical_json_bytes(_json_safe(dataset)))
    blender = load_frozen_blender(FROZEN_BLENDER)
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-nonlocal-") as temporary:
        root = Path(temporary)
        for index, case in enumerate(cases, start=1):
            print(f"[native {index}/{len(cases)}] {case['id']}", flush=True)
            work = root / str(index)
            work.mkdir()
            _prepare_case(case, runner.resolve(), work, blender)

    rows: list[dict[str, object]] = []
    diagnostics: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for case_index, case in enumerate(cases, start=1):
        centers = target_centers(case["target_origin"])
        for patch_index, center in enumerate(centers):
            print(
                f"[match {case_index}/{len(cases)} {patch_index + 1}/{len(centers)}] {case['id']}",
                flush=True,
            )
            row, reconstructions = analyze_target(
                np.asarray(case["context"]), np.asarray(case["mosaic"]), np.asarray(case["cfa"]),
                center, case["baselines"],
            )
            row.update({
                "source_id": case["id"],
                "scene_class": case["scene_class"],
                "split": case["split"],
                "patch_index": patch_index,
            })
            rows.append(row)
            diagnostics[(str(case["id"]), patch_index)] = reconstructions

    natural_analysis_rows = [
        row for row in rows if row["split"] == "analysis" and row["source_id"] not in (
            "frequency-sweep", "saturated-edge",
        )
    ]
    diagnostic_rows = [
        row for row in rows if row["source_id"] in ("frequency-sweep", "saturated-edge")
    ]
    safety_rows = [row for row in rows if row["split"] == "safety"]
    results: dict[str, object] = {
        "format": "rawtherapee-xtrans-nonlocal-results-v1",
        "protocol": {
            "patch_size": 7,
            "target_step": 24,
            "search_radius": 72,
            "neighbors": 32,
            "same_phase_distance": "MSE over all 49 physically observed, color-aligned samples",
            "cross_phase_distance": "MSE over offsets whose physically observed colors agree; minimum 8",
            "low_rank": 4,
            "low_rank_iterations": 10,
            "sample_reinjection": True,
            "same_phase_identifiability": "same mask leaves 98 of 147 RGB rows unobserved in every group; direct donation and unconstrained completion are impossible",
            "ground_truth_policy": "used only for oracle neighbor selection, donor upper bounds, texture labels, and evaluation",
        },
        "patch_count": len(rows),
        "natural_analysis_patch_count": len(natural_analysis_rows),
        "diagnostic_patch_count": len(diagnostic_rows),
        "safety_patch_count": len(safety_rows),
        "overall": {
            "quality": pooled_metrics(rows),
            "same_phase_matching": matching_metrics(rows, "same_phase"),
            "cross_phase_matching": matching_metrics(rows, "cross_phase"),
            "same_phase_headroom_best1": _headroom(rows, "same_phase", "best1"),
            "same_phase_headroom_top8": _headroom(rows, "same_phase", "top8"),
            "same_phase_headroom_top32": _headroom(rows, "same_phase", "top32"),
            "cross_phase_headroom_best1": _headroom(rows, "cross_phase", "best1"),
            "cross_phase_headroom_top8": _headroom(rows, "cross_phase", "top8"),
            "cross_phase_headroom_top32": _headroom(rows, "cross_phase", "top32"),
        },
        "natural_analysis": {
            "quality": pooled_metrics(natural_analysis_rows),
            "same_phase_matching": matching_metrics(natural_analysis_rows, "same_phase"),
            "cross_phase_matching": matching_metrics(natural_analysis_rows, "cross_phase"),
            "same_phase_headroom_best1": _headroom(natural_analysis_rows, "same_phase", "best1"),
            "cross_phase_headroom_best1": _headroom(natural_analysis_rows, "cross_phase", "best1"),
        },
        "synthetic_diagnostics": {
            "quality": pooled_metrics(diagnostic_rows),
            "same_phase_matching": matching_metrics(diagnostic_rows, "same_phase"),
            "cross_phase_matching": matching_metrics(diagnostic_rows, "cross_phase"),
        },
        "sparse_point_safety": {
            "quality": pooled_metrics(safety_rows),
            "same_phase_matching": matching_metrics(safety_rows, "same_phase"),
            "cross_phase_matching": matching_metrics(safety_rows, "cross_phase"),
        },
        "by_source": _group(rows, "source_id"),
        "by_scene_class": _group(rows, "scene_class"),
        "by_texture_class": _group(rows, "texture_class"),
        "patches": [_compact_row(row) for row in rows],
        # Kept transiently for the decision function, removed before writing.
        "natural_analysis_rows": natural_analysis_rows,
        "safety_rows": safety_rows,
    }
    decision, criteria = _decision(results)
    results["decision"] = decision
    results["decision_criteria"] = criteria
    del results["natural_analysis_rows"], results["safety_rows"]
    results_path = output / "results.json"
    results_path.write_bytes(canonical_json_bytes(_json_safe(results)))

    _scatter(rows, output / "neighbor-purity.png")
    for source_id in ("brick", "hubble", "nasa-hydra-starfield", "frequency-sweep"):
        candidates = [row for row in rows if row["source_id"] == source_id]
        selected = max(
            candidates,
            key=lambda row: row["errors"]["practical_blender"]["mse"]
            - row["errors"]["cross_phase_observable_low_rank"]["mse"],
        )
        case = next(value for value in cases if value["id"] == source_id)
        reconstructions = diagnostics[(source_id, int(selected["patch_index"]))]
        _diagnostic(case, selected, reconstructions, output / f"diagnostic-{source_id}.png")

    files = {}
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "format": "rawtherapee-xtrans-nonlocal-manifest-v1",
        "dataset_sha256": _sha256(output / "dataset.json"),
        "results_sha256": _sha256(results_path),
        "decision": decision,
        "files": files,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(generate(arguments.output, arguments.runner, arguments.starfield, force=arguments.force))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the X-Trans demosaicer complementarity/oracle corpus."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import CANONICAL_XTRANS, canonical_json_bytes
from tools.xtrans_sparse_alias.analysis import blind_image_recovery
from tools.xtrans_sparse_alias.generate import load_natural_sources
from .analysis import (
    METHODS,
    PATCH_SIZES,
    block_oracle,
    check_rgb,
    convex_blend_oracle,
    independent_component_oracle,
    label_consistency,
    method_metrics,
    normalized_correlation,
    opponent,
    patch_rms_values,
    percentiles,
    radial_frequency_summary,
    winner_statistics,
)


MARGIN = 16
COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
)
METHOD_DESCRIPTIONS = {
    "markesteijn": "RawTherapee three-pass/CIE-Lab Markesteijn",
    "mlri-final": "corrected blue guides, direct final reconstruction",
    "triangulated-rgb": "independent fixed X-Trans triangulation",
    "triangulated-chroma": "green then triangulated R-G/B-G",
    "global-b": "global quadratic green/color-difference solve, 50 iterations",
    "sparse-alias": "replica-aware global Fourier OMP with sampled-channel DC means",
}


def synthetic_scenes(size: int = 96) -> dict[str, dict[str, object]]:
    if size % 6:
        raise ValueError("synthetic size must be divisible by six")
    y, x = np.mgrid[:size, :size].astype(np.float64)
    xn = x / (size - 1)
    yn = y / (size - 1)
    vertical = x >= size // 2
    horizontal = y >= size // 2
    diagonal = x + y >= size

    gradient = np.stack((0.1 + 0.7 * xn, 0.15 + 0.65 * yn, 0.2 + 0.5 * (xn + yn) / 2))
    luma = 0.15 + 0.55 * xn + 0.15 * np.sin(2 * np.pi * yn)
    c1 = 0.06 * np.sin(2 * np.pi * (xn + 0.3 * yn))
    c2 = 0.04 * np.cos(2 * np.pi * (0.4 * xn - yn))
    correlated = np.stack((luma + c1 + c2, luma - c2, luma - c1 + c2))
    correlated = np.clip(correlated, 0, 1)

    def red_green(mask: np.ndarray) -> np.ndarray:
        return np.stack((0.2 + 0.7 * mask, 0.2 + 0.7 * ~mask, np.full(mask.shape, 0.2)))

    red_gray = np.stack((0.5 + 0.5 * vertical, 0.5 - 0.5 * vertical, 0.5 - 0.5 * vertical))
    blue_gray = np.stack((0.5 - 0.5 * vertical, 0.5 - 0.5 * vertical, 0.5 + 0.5 * vertical))
    quadrant = np.zeros((3, size, size), dtype=np.float64)
    quadrant[0, :, : size // 2] = 1
    quadrant[1, : size // 2, size // 2 :] = 1
    quadrant[2, size // 2 :, size // 2 :] = 1
    impulse = np.full((3, size, size), 0.1)
    impulse[:, size // 2, size // 2] = (1.0, 0.8, 0.3)
    phase = 2 * np.pi * (x * (0.01 + 0.44 * yn) + 0.17 * y)
    sweep = np.stack(
        (
            0.5 + 0.45 * np.sin(phase),
            0.5 + 0.45 * np.sin(phase + 2 * np.pi / 3),
            0.5 + 0.45 * np.sin(phase + 4 * np.pi / 3),
        )
    )
    high = 0.5 + 0.45 * np.cos(2 * np.pi * (0.43 * x + 0.37 * y))
    monochrome = np.stack((high, high, high))
    chroma_wave = np.sin(2 * np.pi * (x / 4 + y / 6))
    periodic_chroma = np.stack(
        (0.5 + 0.42 * chroma_wave, 0.5 - 0.35 * chroma_wave, 0.5 + 0.1 * chroma_wave)
    )
    low = np.sin(2 * np.pi * x / 24)
    high_right = np.sin(2 * np.pi * (0.42 * x + 0.31 * y))
    mixed = np.where(x < size // 2, low, high_right)
    support_transition = np.stack((0.5 + 0.35 * mixed, 0.5 - 0.25 * mixed, 0.5 + 0.12 * mixed))

    rows = (
        ("smooth_gradient", gradient, "low-frequency gradient"),
        ("correlated_field", correlated, "smooth chroma"),
        ("vertical_red_green", red_green(vertical), "axis-aligned chromatic edge"),
        ("horizontal_red_green", red_green(horizontal), "axis-aligned chromatic edge"),
        ("diagonal_red_green", red_green(diagonal), "diagonal chromatic edge"),
        ("saturated_red_gray", red_gray, "saturated chromatic edge"),
        ("saturated_blue_gray", blue_gray, "saturated chromatic edge"),
        ("saturated_quadrants", quadrant, "corner/intersection"),
        ("isolated_impulse", impulse, "isolated point/impulse"),
        ("frequency_sweep", sweep, "near-Nyquist texture"),
        ("high_frequency_monochrome", monochrome, "near-Nyquist monochrome"),
        ("periodic_chromatic_texture", periodic_chroma, "periodic chromatic texture"),
        ("support_transition", support_transition, "mixed structure"),
    )
    return {
        name: {"truth": np.asarray(rgb, dtype=np.float64), "category": category}
        for name, rgb, category in rows
    }


def srgb_decode(values: np.ndarray) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    return np.where(
        checked <= 0.04045,
        checked / 12.92,
        ((checked + 0.055) / 1.055) ** 2.4,
    )


def srgb_encode(values: np.ndarray) -> np.ndarray:
    checked = np.clip(np.asarray(values, dtype=np.float64), 0, 1)
    return np.where(
        checked <= 0.0031308,
        checked * 12.92,
        1.055 * checked ** (1 / 2.4) - 0.055,
    )


def natural_scenes(size: int = 192) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    scenes = {}
    identities = []
    for source in load_natural_sources():
        image = srgb_decode(np.asarray(source.pop("rgb"), dtype=np.float64))
        height, width = image.shape[:2]
        if min(height, width) < size:
            raise RuntimeError(f"natural source {source['name']} is too small")
        y = (height - size) // 2
        x = (width - size) // 2
        crop = np.moveaxis(image[y : y + size, x : x + size], -1, 0)
        key = "natural-" + str(source["filename"]).rsplit(".", 1)[0].replace("_", "-")
        scenes[key] = {
            "truth": crop,
            "category": source["category"],
            "crop": [x, y, size, size],
            "source": source["name"],
        }
        identity = dict(source)
        identity["linear_crop"] = [x, y, size, size]
        identities.append(identity)
    return scenes, identities


def tiled_cfa(height: int, width: int) -> np.ndarray:
    return np.tile(
        CANONICAL_XTRANS,
        ((height + 5) // 6, (width + 5) // 6),
    )[:height, :width]


def mosaic(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = check_rgb(truth)
    cfa = tiled_cfa(truth.shape[1], truth.shape[2])
    sampled = np.take_along_axis(np.moveaxis(truth, 0, -1), cfa[..., None], axis=2)[..., 0]
    return sampled, cfa


def _run_native(
    runner: Path, truth: np.ndarray, work: Path
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    scalar, cfa = mosaic(truth)
    input_path = work / "mosaic.f32le"
    np.asarray(scalar, dtype="<f4").tofile(input_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            str(runner), "run", str(input_path), str(work),
            str(truth.shape[2]), str(truth.shape[1]),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"native oracle runner failed:\n{completed.stdout}\n{completed.stderr}")
    timings = {}
    for line in completed.stdout.splitlines():
        name, seconds = line.split("\t")
        timings[name] = float(seconds)
    outputs = {}
    pixels = truth.shape[1] * truth.shape[2]
    for name in METHODS[:-1]:
        values = np.fromfile(work / f"{name}.f32le", dtype="<f4")
        if values.size != 3 * pixels:
            raise RuntimeError(f"incorrect native output size for {name}")
        outputs[name] = values.astype(np.float64).reshape(3, truth.shape[1], truth.shape[2])

    means = np.asarray([truth[channel][cfa == channel].mean() for channel in range(3)])
    started = time.perf_counter()
    alias, alias_metrics = blind_image_recovery(
        truth, max_atoms=12, mode="replica", channel_means=means
    )
    timings["sparse-alias"] = time.perf_counter() - started
    if alias_metrics["channel_means_source"] != "caller supplied":
        raise RuntimeError("sparse alias recovery used ground-truth DC means")
    outputs["sparse-alias"] = alias
    return outputs, timings


def _interior_error(case: Mapping[str, object], method: str) -> np.ndarray:
    truth = case["truth"]
    output = case["outputs"][method]
    return output[:, MARGIN:-MARGIN, MARGIN:-MARGIN] - truth[:, MARGIN:-MARGIN, MARGIN:-MARGIN]


def _psnr_from_sse(sse: float, count: int) -> float:
    mse = sse / count
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def pooled_method_quality(cases: Sequence[Mapping[str, object]], method: str) -> dict[str, float]:
    sse = 0.0
    count = 0
    for case in cases:
        error = _interior_error(case, method)
        sse += float(np.sum(error * error))
        count += error.size
    return {"mse": sse / count, "psnr_db": _psnr_from_sse(sse, count)}


def pooled_oracle_quality(
    cases: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    block_size: int,
    *,
    target: str = "rgb",
    independent: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    sse = 0.0
    count = 0
    target_sse = 0.0
    target_count = 0
    labels = {}
    for case in cases:
        if independent:
            selected, component_labels = independent_component_oracle(
                case["truth"], case["outputs"], methods, block_size, margin=MARGIN
            )
            labels[str(case["name"])] = component_labels[0]
        else:
            selected, label = block_oracle(
                case["truth"], case["outputs"], methods, block_size,
                margin=MARGIN, target=target,
            )
            labels[str(case["name"])] = label
        error = selected[:, MARGIN:-MARGIN, MARGIN:-MARGIN] - case["truth"][:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        sse += float(np.sum(error * error))
        count += error.size
        if target != "rgb" and not independent:
            components = {"luminance": (0,), "chroma": (1, 2), "c1": (1,), "c2": (2,)}[target]
            component_error = opponent(selected)[:, MARGIN:-MARGIN, MARGIN:-MARGIN] - opponent(case["truth"])[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
            selected_error = component_error[list(components)]
            target_sse += float(np.sum(selected_error * selected_error))
            target_count += selected_error.size
    quality = {"mse": sse / count, "psnr_db": _psnr_from_sse(sse, count)}
    if target_count:
        quality["selection_target_mse"] = target_sse / target_count
        quality["selection_target_psnr_db"] = _psnr_from_sse(target_sse, target_count)
    return quality, labels


def pooled_blend_quality(
    cases: Sequence[Mapping[str, object]], pair: Sequence[str], block_size: int
) -> dict[str, float]:
    sse = 0.0
    count = 0
    for case in cases:
        blended, _ = convex_blend_oracle(
            case["truth"], case["outputs"][pair[0]], case["outputs"][pair[1]],
            block_size, margin=MARGIN,
        )
        error = blended[:, MARGIN:-MARGIN, MARGIN:-MARGIN] - case["truth"][:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        sse += float(np.sum(error * error))
        count += error.size
    return {"mse": sse / count, "psnr_db": _psnr_from_sse(sse, count)}


def aggregate_correlations(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    vectors = {name: {target: [] for target in ("RGB", "L", "C1", "C2")} for name in METHODS}
    for case in cases:
        for name in METHODS:
            error = _interior_error(case, name)
            components = opponent(error)
            vectors[name]["RGB"].append(error.ravel())
            for index, target in enumerate(("L", "C1", "C2")):
                vectors[name][target].append(components[index].ravel())
    merged = {
        name: {target: np.concatenate(rows) for target, rows in targets.items()}
        for name, targets in vectors.items()
    }
    rows = []
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            row = {"left": left, "right": right}
            for target in ("RGB", "L", "C1", "C2"):
                row[target] = {
                    "centered": normalized_correlation(
                        merged[left][target], merged[right][target], centered=True
                    ),
                    "uncentered": normalized_correlation(
                        merged[left][target], merged[right][target], centered=False
                    ),
                }
            rows.append(row)
    return rows


def pairwise_oracles(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    quality = {name: pooled_method_quality(cases, name) for name in METHODS}
    correlations = {
        (row["left"], row["right"]): row for row in aggregate_correlations(cases)
    }
    rows = []
    for pair in itertools.combinations(METHODS, 2):
        better = max(pair, key=lambda name: quality[name]["psnr_db"])
        entry = {
            "pair": list(pair),
            "better_member": better,
            "better_member_psnr_db": quality[better]["psnr_db"],
            "correlation": correlations[pair],
            "patches": {},
        }
        for block_size in PATCH_SIZES:
            oracle, _ = pooled_oracle_quality(cases, pair, block_size)
            gain = oracle["psnr_db"] - quality[better]["psnr_db"]
            error_reduction = (quality[better]["mse"] - oracle["mse"]) / quality[better]["mse"]
            entry["patches"][str(block_size)] = {
                "error_reduction_fraction": error_reduction,
                "gain_db": gain,
                "psnr_db": oracle["psnr_db"],
            }
        rows.append(entry)
    return rows


def best_subsets(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for size in (2, 3, len(METHODS)):
        candidates = (tuple(METHODS),) if size == len(METHODS) else itertools.combinations(METHODS, size)
        candidate_rows = []
        for subset in candidates:
            patch_rows = {}
            for block_size in PATCH_SIZES:
                quality, _ = pooled_oracle_quality(cases, subset, block_size)
                patch_rows[str(block_size)] = quality
            candidate_rows.append({"methods": list(subset), "patches": patch_rows})
        best = max(candidate_rows, key=lambda row: row["patches"]["15"]["psnr_db"])
        best["set_size"] = size
        rows.append(best)
    return rows


def selection_targets(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = {}
    for target in ("rgb", "luminance", "chroma", "c1", "c2"):
        rows[target] = {}
        for block_size in (1, 7, 15):
            quality, _ = pooled_oracle_quality(cases, METHODS, block_size, target=target)
            rows[target][str(block_size)] = quality
    rows["independent_l_c1_c2"] = {}
    for block_size in (1, 7, 15):
        quality, _ = pooled_oracle_quality(cases, METHODS, block_size, independent=True)
        rows["independent_l_c1_c2"][str(block_size)] = quality
    return rows


def blend_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for pair in itertools.combinations(METHODS, 2):
        entry = {"pair": list(pair), "patches": {}}
        for block_size in (1, 7, 15):
            hard, _ = pooled_oracle_quality(cases, pair, block_size)
            blend = pooled_blend_quality(cases, pair, block_size)
            entry["patches"][str(block_size)] = {
                "blend_psnr_db": blend["psnr_db"],
                "blend_over_hard_db": blend["psnr_db"] - hard["psnr_db"],
                "hard_psnr_db": hard["psnr_db"],
            }
        rows.append(entry)
    return rows


def contribution_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for block_size in (1, 7, 15):
        full, _ = pooled_oracle_quality(cases, METHODS, block_size)
        for removed in METHODS:
            subset = tuple(name for name in METHODS if name != removed)
            reduced, _ = pooled_oracle_quality(cases, subset, block_size)
            rows.append({
                "block_size": block_size,
                "loss_without_db": full["psnr_db"] - reduced["psnr_db"],
                "removed": removed,
            })
    return rows


def population_oracle_rows(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    method_quality = {name: pooled_method_quality(cases, name) for name in METHODS}
    best_name = max(METHODS, key=lambda name: method_quality[name]["psnr_db"])
    rows = {"best_individual": best_name, "patches": {}}
    for block_size in PATCH_SIZES:
        quality, _ = pooled_oracle_quality(cases, METHODS, block_size)
        rows["patches"][str(block_size)] = {
            **quality,
            "gain_over_best_db": quality["psnr_db"] - method_quality[best_name]["psnr_db"],
            "gain_over_markesteijn_db": (
                quality["psnr_db"] - method_quality["markesteijn"]["psnr_db"]
            ),
        }
    return rows


def structural_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        qualities = {
            name: method_metrics(case["truth"], case["outputs"][name], case["cfa"], margin=MARGIN)["psnr_db"]
            for name in METHODS
        }
        ordered = sorted(METHODS, key=lambda name: qualities[name], reverse=True)
        mark_mlri_7, _ = pooled_oracle_quality((case,), ("markesteijn", "mlri-final"), 7)
        all_7, _ = pooled_oracle_quality((case,), METHODS, 7)
        all_15, _ = pooled_oracle_quality((case,), METHODS, 15)
        rows.append({
            "all_oracle_gain_15_db": all_15["psnr_db"] - qualities[ordered[0]],
            "all_oracle_gain_7_db": all_7["psnr_db"] - qualities[ordered[0]],
            "best": ordered[0],
            "best_psnr_db": qualities[ordered[0]],
            "category": case["category"],
            "markesteijn_mlri_gain_7_db": mark_mlri_7["psnr_db"] - max(
                qualities["markesteijn"], qualities["mlri-final"]
            ),
            "scene": case["name"],
            "second": ordered[1],
            "second_psnr_db": qualities[ordered[1]],
        })
    return rows


def natural_patch_rows(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = {name: [] for name in METHODS}
    values["all-oracle-7"] = []
    for case in cases:
        for name in METHODS:
            values[name].extend(patch_rms_values(_interior_error(case, name), 7))
        selected, _ = block_oracle(
            case["truth"], case["outputs"], METHODS, 7, margin=MARGIN
        )
        error = selected[:, MARGIN:-MARGIN, MARGIN:-MARGIN] - case["truth"][:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        values["all-oracle-7"].extend(patch_rms_values(error, 7))
    return {name: percentiles(rows) for name, rows in values.items()}


def frequency_rows(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result = {}
    for name in METHODS:
        per_case = [radial_frequency_summary(_interior_error(case, name)) for case in cases]
        result[name] = {
            key: float(np.mean([row[key] for row in per_case]))
            for key in per_case[0]
        }
    return result


def winner_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        labels = {}
        for block_size in PATCH_SIZES:
            _, labels[block_size] = block_oracle(
                case["truth"], case["outputs"], METHODS, block_size, margin=MARGIN
            )
            stats = winner_statistics(labels[block_size], len(METHODS), margin=MARGIN)
            stats.update({"block_size": block_size, "scene": case["name"]})
            rows.append(stats)
        for left, right in ((3, 7), (7, 15), (15, 31)):
            rows.append({
                "comparison": f"{left}-to-{right}",
                "label_consistency": label_consistency(labels[left], labels[right], margin=MARGIN),
                "scene": case["name"],
            })
    return rows


def focused_winner_rows(
    cases: Sequence[Mapping[str, object]], pair: Sequence[str]
) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        labels = {}
        for block_size in (1, 7, 15):
            _, labels[block_size] = block_oracle(
                case["truth"], case["outputs"], pair, block_size, margin=MARGIN
            )
            stats = winner_statistics(labels[block_size], len(pair), margin=MARGIN)
            stats.update({
                "block_size": block_size,
                "pair": list(pair),
                "scene": case["name"],
            })
            rows.append(stats)
        rows.append({
            "comparison": "7-to-15",
            "label_consistency": label_consistency(labels[7], labels[15], margin=MARGIN),
            "pair": list(pair),
            "scene": case["name"],
        })
    return rows


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def winner_image(labels: np.ndarray, methods: Sequence[str], scale: int = 4) -> Image.Image:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for index in range(len(methods)):
        rgb[labels == index] = COLORS[index]
    rgb[labels < 0] = (245, 245, 245)
    image = Image.fromarray(rgb, "RGB").resize(
        (labels.shape[1] * scale, labels.shape[0] * scale), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (image.width, image.height + 30), (255, 255, 255))
    canvas.paste(image, (0, 30))
    draw = ImageDraw.Draw(canvas)
    x = 5
    for index, name in enumerate(methods):
        draw.rectangle((x, 8, x + 9, 17), fill=COLORS[index])
        draw.text((x + 13, 5), name, fill=(20, 20, 20), font=_font())
        x += 18 + 6 * len(name)
        if x > canvas.width - 120:
            break
    return canvas


def correlation_image(rows: Sequence[Mapping[str, object]]) -> Image.Image:
    labels = {
        "markesteijn": "markesteijn",
        "mlri-final": "mlri-final",
        "triangulated-rgb": "tri-rgb",
        "triangulated-chroma": "tri-chroma",
        "global-b": "global-b",
        "sparse-alias": "sparse-alias",
    }
    matrix = np.eye(len(METHODS), dtype=np.float64)
    for row in rows:
        left = METHODS.index(row["left"])
        right = METHODS.index(row["right"])
        matrix[left, right] = matrix[right, left] = row["RGB"]["centered"]
    cell = 90
    image = Image.new("RGB", ((len(METHODS) + 1) * cell, (len(METHODS) + 1) * cell), "white")
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(METHODS):
        draw.text(((index + 1) * cell + 3, 8), labels[name], fill="black", font=_font())
        draw.text((3, (index + 1) * cell + 8), labels[name], fill="black", font=_font())
    for y in range(len(METHODS)):
        for x in range(len(METHODS)):
            value = matrix[y, x]
            red = int(round(255 * max(0, value)))
            blue = int(round(255 * max(0, -value)))
            green = int(round(245 * (1 - min(1, abs(value)))))
            box = ((x + 1) * cell, (y + 1) * cell, (x + 2) * cell - 1, (y + 2) * cell - 1)
            draw.rectangle(box, fill=(red, green, blue), outline=(70, 70, 70))
            draw.text((box[0] + 25, box[1] + 35), f"{value:.3f}", fill="black", font=_font())
    return image


def curve_image(curves: Mapping[str, Sequence[float]]) -> Image.Image:
    image = Image.new("RGB", (850, 500), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 45, 810, 430
    draw.rectangle((left, top, right, bottom), outline=(40, 40, 40))
    maximum = max(max(values) for values in curves.values())
    maximum = max(maximum, 0.01)
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        draw.line((left, y, right, y), fill=(215, 215, 215))
        draw.text((8, y - 5), f"{maximum * tick / 5:.2f} dB", fill="black", font=_font())
    for curve_index, (name, values) in enumerate(curves.items()):
        points = []
        for index, value in enumerate(values):
            x = left + index * (right - left) / (len(PATCH_SIZES) - 1)
            y = bottom - value / maximum * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=COLORS[curve_index], width=3)
        draw.text((left + curve_index * 245, 455), name, fill=COLORS[curve_index], font=_font())
    for index, size in enumerate(PATCH_SIZES):
        x = left + index * (right - left) / (len(PATCH_SIZES) - 1)
        draw.text((x - 8, bottom + 8), f"{size}x{size}", fill="black", font=_font())
    return image


def comparison_crop(case: Mapping[str, object], x: int, y: int, size: int = 31) -> Image.Image:
    methods = ("ground-truth", "markesteijn", "mlri-final", "all-oracle-7")
    oracle, _ = block_oracle(case["truth"], case["outputs"], METHODS, 7, margin=MARGIN)
    images = {
        "ground-truth": case["truth"],
        "markesteijn": case["outputs"]["markesteijn"],
        "mlri-final": case["outputs"]["mlri-final"],
        "all-oracle-7": oracle,
    }
    scale = 5
    tile_size = size * scale
    canvas = Image.new("RGB", (tile_size * len(methods), tile_size + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(methods):
        crop = images[name][:, y : y + size, x : x + size]
        encoded = np.moveaxis(srgb_encode(crop), 0, -1)
        tile = Image.fromarray(np.round(encoded * 255).astype(np.uint8), "RGB").resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        canvas.paste(tile, (index * tile_size, 24))
        draw.text((index * tile_size + 4, 5), name, fill="black", font=_font())
    return canvas


def error_comparison_image(case: Mapping[str, object]) -> Image.Image:
    """Show truth and signed Markesteijn/MLRI error using one common scale."""

    truth = np.asarray(case["truth"], dtype=np.float64)
    errors = {
        name: np.asarray(case["outputs"][name], dtype=np.float64) - truth
        for name in ("markesteijn", "mlri-final")
    }
    scale = max(
        1e-6,
        float(np.percentile(np.abs(np.stack(tuple(errors.values()))), 99.5)),
    )
    panels = (
        ("ground truth", np.moveaxis(srgb_encode(truth), 0, -1)),
        (
            f"Markesteijn signed error (+/-{scale:.4f})",
            np.moveaxis(np.clip(0.5 + errors["markesteijn"] / (2 * scale), 0, 1), 0, -1),
        ),
        (
            f"MLRI signed error (+/-{scale:.4f})",
            np.moveaxis(np.clip(0.5 + errors["mlri-final"] / (2 * scale), 0, 1), 0, -1),
        ),
    )
    zoom = 2
    height, width = truth.shape[1:]
    canvas = Image.new("RGB", (width * zoom * len(panels), height * zoom + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, values) in enumerate(panels):
        image = Image.fromarray(np.round(values * 255).astype(np.uint8), "RGB").resize(
            (width * zoom, height * zoom), Image.Resampling.NEAREST
        )
        canvas.paste(image, (index * width * zoom, 24))
        draw.text((index * width * zoom + 4, 5), label, fill="black", font=_font())
    return canvas


def generate(output_dir: Path, runner: Path, *, force: bool) -> tuple[Path, str]:
    output_dir = output_dir.resolve()
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    manifest = output_dir / "complementarity.json"
    if manifest.exists() and not force:
        raise FileExistsError(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = synthetic_scenes()
    natural, source_identities = natural_scenes()
    all_scenes = {**synthetic, **natural}
    cases = []
    with tempfile.TemporaryDirectory(prefix="rt-xtrans-oracle-") as temporary:
        root = Path(temporary)
        for index, (name, metadata) in enumerate(all_scenes.items(), start=1):
            work = root / name
            work.mkdir()
            print(f"[{index}/{len(all_scenes)}] {name}", flush=True)
            outputs, timings = _run_native(runner, metadata["truth"], work)
            _, cfa = mosaic(metadata["truth"])
            cases.append({
                **metadata,
                "cfa": cfa,
                "name": name,
                "outputs": outputs,
                "timings": timings,
                "type": "synthetic" if name in synthetic else "natural",
            })

    analysis_started = time.perf_counter()
    synthetic_cases = [case for case in cases if case["type"] == "synthetic"]
    natural_cases = [case for case in cases if case["type"] == "natural"]
    per_case_metrics = []
    for case in cases:
        for name in METHODS:
            per_case_metrics.append({
                "dataset": case["type"],
                "method": name,
                "metrics": method_metrics(
                    case["truth"], case["outputs"][name], case["cfa"], margin=MARGIN
                ),
                "scene": case["name"],
            })

    aggregate_quality = {
        population: {name: pooled_method_quality(selected, name) for name in METHODS}
        for population, selected in (
            ("all", cases), ("synthetic", synthetic_cases), ("natural", natural_cases)
        )
    }
    pair_all = pairwise_oracles(cases)
    pair_natural = pairwise_oracles(natural_cases)
    subsets = best_subsets(cases)
    natural_subsets = best_subsets(natural_cases)
    all_method = next(row for row in subsets if row["set_size"] == len(METHODS))
    natural_all_method = next(
        row for row in natural_subsets if row["set_size"] == len(METHODS)
    )
    natural_best_pair = next(row for row in natural_subsets if row["set_size"] == 2)
    natural_best_three = next(row for row in natural_subsets if row["set_size"] == 3)
    mark_mlri = next(row for row in pair_all if row["pair"] == ["markesteijn", "mlri-final"])
    mark_alias = next(row for row in pair_all if row["pair"] == ["markesteijn", "sparse-alias"])

    natural_best_psnr = aggregate_quality["natural"]["markesteijn"]["psnr_db"]
    curves = {
        "all methods": [
            natural_all_method["patches"][str(size)]["psnr_db"] - natural_best_psnr
            for size in PATCH_SIZES
        ],
        "best three": [
            natural_best_three["patches"][str(size)]["psnr_db"] - natural_best_psnr
            for size in PATCH_SIZES
        ],
        "best pair": [
            natural_best_pair["patches"][str(size)]["psnr_db"] - natural_best_psnr
            for size in PATCH_SIZES
        ],
    }

    images = {
        "error-correlation.png": correlation_image(
            aggregate_correlations(natural_cases)
        ),
        "patch-headroom.png": curve_image(curves),
    }
    representative = (
        "diagonal_red_green", "periodic_chromatic_texture", "natural-astronaut"
    )
    case_lookup = {case["name"]: case for case in cases}
    for scene in ("diagonal_red_green", "natural-astronaut"):
        images[f"error-mark-mlri-{scene}.png"] = error_comparison_image(
            case_lookup[scene]
        )
    for scene in representative:
        for block_size in (1, 7, 15):
            _, labels = block_oracle(
                case_lookup[scene]["truth"], case_lookup[scene]["outputs"],
                METHODS, block_size, margin=MARGIN,
            )
            images[f"winner-{scene}-{block_size}.png"] = winner_image(labels, METHODS)
    for pair_name, pair, scenes in (
        ("mark-mlri", ("markesteijn", "mlri-final"), ("diagonal_red_green", "natural-astronaut")),
        ("mark-alias", ("markesteijn", "sparse-alias"), ("periodic_chromatic_texture", "support_transition")),
    ):
        for scene in scenes:
            for block_size in (1, 7, 15):
                _, labels = block_oracle(
                    case_lookup[scene]["truth"], case_lookup[scene]["outputs"],
                    pair, block_size, margin=MARGIN,
                )
                images[f"winner-{pair_name}-{scene}-{block_size}.png"] = winner_image(labels, pair)

    # Worst natural 31x31 Markesteijn blocks, restricted to full blocks and
    # deduplicated by source. These are diagnostic crops, not training data.
    for case in natural_cases:
        error = _interior_error(case, "markesteijn")
        scores = []
        for y in range(0, error.shape[1] - 30, 31):
            for x in range(0, error.shape[2] - 30, 31):
                scores.append((float(np.mean(error[:, y : y + 31, x : x + 31] ** 2)), x, y))
        _, x, y = max(scores)
        images[f"worst-{case['name']}.png"] = comparison_crop(
            case, x + MARGIN, y + MARGIN, 31
        )

    for path in output_dir.glob("*.png"):
        if force:
            path.unlink()
    for filename, image in sorted(images.items()):
        image.save(output_dir / filename, format="PNG", optimize=False, compress_level=9)
    plot_rows = []
    for filename in sorted(images):
        path = output_dir / filename
        plot_rows.append({
            "bytes": path.stat().st_size,
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    payload = {
        "aggregate_quality": aggregate_quality,
        "all_method_oracle": all_method,
        "best_subsets": {"all": subsets, "natural": natural_subsets},
        "blend_oracles": {
            "all": blend_rows(cases),
            "natural": blend_rows(natural_cases),
        },
        "cfa": np.asarray(CANONICAL_XTRANS).tolist(),
        "comparison": {
            "border_margin": MARGIN,
            "domain": "camera-linear normalized RGB",
            "native_runner": "actual C++ implementations, float32 scalar mosaic",
            "patch_grid": "nonoverlapping, anchored at evaluated-interior top-left",
            "patch_sizes": list(PATCH_SIZES),
            "sparse_alias_dc": "per-channel means estimated from physically sampled CFA values",
            "timing": "local measured timing is reported in the research note, not the canonical corpus",
        },
        "contributions": {
            "all": contribution_rows(cases),
            "natural": contribution_rows(natural_cases),
        },
        "error_correlations": {
            "all": aggregate_correlations(cases),
            "natural": aggregate_correlations(natural_cases),
        },
        "format": "rawtherapee-xtrans-demosaicer-complementarity-v1",
        "frequency_error": frequency_rows(synthetic_cases),
        "focused_winner_coherence": [
            *focused_winner_rows(cases, ("markesteijn", "mlri-final")),
            *focused_winner_rows(cases, ("markesteijn", "sparse-alias")),
        ],
        "markesteijn_alias_focus": mark_alias,
        "markesteijn_mlri_focus": mark_mlri,
        "methods": [{"id": name, "description": METHOD_DESCRIPTIONS[name]} for name in METHODS],
        "natural_patch_error": natural_patch_rows(natural_cases),
        "natural_sources": source_identities,
        "pairwise_oracles": {"all": pair_all, "natural": pair_natural},
        "per_case_metrics": per_case_metrics,
        "plots": plot_rows,
        "selection_targets": {
            "all": selection_targets(cases),
            "natural": selection_targets(natural_cases),
        },
        "population_oracles": {
            "all": population_oracle_rows(cases),
            "natural": population_oracle_rows(natural_cases),
            "synthetic": population_oracle_rows(synthetic_cases),
        },
        "structural_breakdown": structural_rows(synthetic_cases),
        "synthetic_scenes": [
            {"category": row["category"], "name": name, "shape": list(row["truth"].shape)}
            for name, row in synthetic.items()
        ],
        "winner_coherence": winner_rows(cases),
    }
    safe_payload = json_safe(payload)
    encoded = canonical_json_bytes(safe_payload)
    manifest.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    print(f"oracle analysis: {time.perf_counter() - analysis_started:.6f} seconds")
    return manifest, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest, digest = generate(args.output, args.runner, force=args.force)
    print(f"manifest: {manifest}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

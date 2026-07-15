#!/usr/bin/env python3
"""Compare the Phase 9 linear, gamma-2.2, and Markesteijn RAF exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import tifffile


FORMAT = "rawtherapee-xtrans-gamma22-comparison-v1"
EXPECTED_RAW_SHA256 = "26106d7da2ba9a87caebffd4bd5e5fb0371cc8a2b6139ec724b59ef4112842c0"
EXPECTED_RTNN_SHA256 = "b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2"
EXPECTED_SHAPE = (5178, 7752, 3)
PHASE_REDUCTION_MINIMUM = 0.75
PHASE_RMS_ABSOLUTE_FLOOR = 0.0015
COLOR_COMMON_LIMIT = 0.005
COLOR_RANGE_LIMIT = 0.005
FALLBACK_MARKER = "falling back to 3-pass (Markesteijn)"
COMPLETION_RE = re.compile(
    r"DemosaicNet X-Trans completed:.*?method=(\S+).*?artifact=([0-9a-f]{64}).*?"
    r"input_tile=(\d+x\d+).*?output_core=(\d+x\d+).*?tiles=(\d+).*?workers=(\d+).*?"
    r"workspace_per_worker=(\d+).*?workspace_total=(\d+).*?elapsed_us=(\d+)"
)


class ComparisonError(ValueError):
    """The comparison inputs do not satisfy the reviewed test contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def parse_crop(text: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be X,Y,W,H") from error
    if len(result) != 4 or min(result) < 0 or result[2] <= 0 or result[3] <= 0:
        raise argparse.ArgumentTypeError("crop must contain nonnegative X,Y and positive W,H")
    return result  # type: ignore[return-value]


def tiff_metadata(path: Path) -> dict[str, Any]:
    try:
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) != 1:
                raise ComparisonError(f"{path.name} must contain exactly one TIFF page")
            page = tiff.pages[0]
            icc_tag = page.tags.get(34675)
            icc = bytes(icc_tag.value) if icc_tag else b""
            return {
                "shape": list(page.shape),
                "dtype": page.dtype.name,
                "bits_per_sample": int(page.bitspersample),
                "photometric": int(page.photometric),
                "icc_profile_bytes": len(icc),
                "icc_profile_sha256": hashlib.sha256(icc).hexdigest(),
            }
    except (OSError, tifffile.TiffFileError) as error:
        raise ComparisonError(f"cannot read {path.name}: {error}") from error


def validate_metadata(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(metadata.values()))
    for name, value in metadata.items():
        if value != first:
            raise ComparisonError(f"{name} TIFF metadata differs from the other outputs")
    if tuple(first["shape"]) != EXPECTED_SHAPE:
        raise ComparisonError(f"output shape is {first['shape']}, expected {list(EXPECTED_SHAPE)}")
    if first["dtype"] != "uint16" or first["bits_per_sample"] != 16:
        raise ComparisonError("outputs must contain 16-bit unsigned samples")
    if first["photometric"] != int(tifffile.PHOTOMETRIC.RGB):
        raise ComparisonError("outputs must use RGB TIFF photometric interpretation")
    if first["icc_profile_bytes"] == 0:
        raise ComparisonError("outputs must contain the same nonempty ICC profile")
    return first


def parse_execution(log_text: str, timing_text: str) -> dict[str, Any]:
    if FALLBACK_MARKER in log_text:
        raise ComparisonError("gamma export fell back to Markesteijn")
    matches = COMPLETION_RE.findall(log_text)
    if len(matches) != 1:
        raise ComparisonError("gamma log must contain exactly one completion diagnostic")
    match = matches[0]
    if match[0] != "demosaicnet-xtrans-gamma22":
        raise ComparisonError(f"completion method is {match[0]}, expected gamma-2.2")
    if match[1] != EXPECTED_RTNN_SHA256:
        raise ComparisonError("completion diagnostic has the wrong RTNN identity")
    timing = {}
    for line in timing_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            timing[key.strip()] = value.strip()
    try:
        elapsed_seconds = float(timing["elapsed_seconds"])
        max_rss_kb = int(timing["max_rss_kb"])
    except (KeyError, ValueError) as error:
        raise ComparisonError("timing file lacks valid elapsed_seconds or max_rss_kb") from error
    return {
        "active_workers": int(match[5]),
        "artifact_sha256": match[1],
        "elapsed_seconds": elapsed_seconds,
        "engine_elapsed_us": int(match[8]),
        "input_tile": match[2],
        "max_rss_kb": max_rss_kb,
        "output_core": match[3],
        "tiles": int(match[4]),
        "workspace_per_worker_bytes": int(match[6]),
        "workspace_total_bytes": int(match[7]),
    }


def full_channel_mean(image: np.ndarray, row_chunk: int = 256) -> np.ndarray:
    total = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, image.shape[0], row_chunk):
        chunk = np.asarray(image[start : start + row_chunk], dtype=np.float64)
        total += chunk.sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    return total / (count * 65535.0)


def aligned_blocks(image: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = crop
    x0 = ((x + 5) // 6) * 6
    y0 = ((y + 5) // 6) * 6
    x1 = ((x + width) // 6) * 6
    y1 = ((y + height) // 6) * 6
    if x1 <= x0 or y1 <= y0:
        raise ComparisonError("crop does not contain a complete globally aligned 6x6 block")
    region = np.asarray(image[y0:y1, x0:x1], dtype=np.float64) / 65535.0
    block_rows = region.shape[0] // 6
    block_columns = region.shape[1] // 6
    return region.reshape(block_rows, 6, block_columns, 6, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 6, 6, 3)


def low_detail_selection(mark_blocks: np.ndarray) -> np.ndarray:
    centered = mark_blocks - mark_blocks.mean(axis=(1, 2), keepdims=True)
    detail = np.mean(centered * centered, axis=(1, 2, 3), dtype=np.float64)
    count = (len(detail) + 1) // 2
    selected = np.argsort(detail, kind="stable")[:count]
    mask = np.zeros(len(detail), dtype=bool)
    mask[selected] = True
    return mask


def phase_statistics(blocks: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    residual = blocks[selected] - blocks[selected].mean(axis=(1, 2), keepdims=True)
    matrix = np.empty((3, 3, 3), dtype=np.float64)
    for phase_y in range(3):
        for phase_x in range(3):
            matrix[phase_y, phase_x] = residual[:, phase_y::3, phase_x::3].mean(
                axis=(0, 1, 2), dtype=np.float64
            )
    rms = np.sqrt(np.mean(matrix * matrix, axis=(0, 1), dtype=np.float64))
    spread = np.ptp(matrix, axis=(0, 1))
    return {
        "bias_matrix_rgb": matrix.tolist(),
        "rms_rgb": rms.tolist(),
        "spread_rgb": spread.tolist(),
    }


def analyze_arrays(
    images: dict[str, np.ndarray], crop: tuple[int, int, int, int]
) -> tuple[dict[str, Any], np.ndarray]:
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1 or any(len(shape) != 3 or shape[2] != 3 for shape in shapes):
        raise ComparisonError("comparison arrays must have identical HxWx3 shapes")
    shape = next(iter(shapes))
    if x + width > shape[1] or y + height > shape[0]:
        raise ComparisonError("crop lies outside the output images")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods = {}
    crops = {}
    for name, image in images.items():
        cropped = np.asarray(image[y : y + height, x : x + width])
        crops[name] = cropped
        methods[name] = {
            "crop_mean_rgb": (cropped.mean(axis=(0, 1), dtype=np.float64) / 65535.0).tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }

    gamma_delta = np.asarray(methods["gamma22"]["crop_mean_rgb"]) - np.asarray(
        methods["markesteijn"]["crop_mean_rgb"]
    )
    linear_delta = np.asarray(methods["linear"]["crop_mean_rgb"]) - np.asarray(
        methods["markesteijn"]["crop_mean_rgb"]
    )
    gamma_rms = np.asarray(methods["gamma22"]["phase"]["rms_rgb"])
    linear_rms = np.asarray(methods["linear"]["phase"]["rms_rgb"])
    mark_rms = np.asarray(methods["markesteijn"]["phase"]["rms_rgb"])
    reduction = np.full(3, -1.0, dtype=np.float64)
    nonzero_linear = linear_rms > np.finfo(np.float64).eps * 16
    reduction[nonzero_linear] = 1.0 - gamma_rms[nonzero_linear] / linear_rms[nonzero_linear]
    reduction[~nonzero_linear & (gamma_rms <= np.finfo(np.float64).eps * 16)] = 1.0
    limits = np.maximum(2.0 * mark_rms, PHASE_RMS_ABSOLUTE_FLOOR)
    phase_channels_pass = (reduction >= PHASE_REDUCTION_MINIMUM) & (gamma_rms <= limits)
    common = float(gamma_delta.mean())
    color_range = float(np.ptp(gamma_delta))
    objective = {
        "color": {
            "common_luminance_delta": common,
            "common_luminance_limit": COLOR_COMMON_LIMIT,
            "gamma_minus_markesteijn_rgb": gamma_delta.tolist(),
            "linear_minus_markesteijn_rgb": linear_delta.tolist(),
            "pass": abs(common) <= COLOR_COMMON_LIMIT and color_range <= COLOR_RANGE_LIMIT,
            "rgb_delta_range": color_range,
            "rgb_delta_range_limit": COLOR_RANGE_LIMIT,
        },
        "phase": {
            "gamma_reduction_from_linear_rgb": reduction.tolist(),
            "gamma_rms_limits_rgb": limits.tolist(),
            "pass": bool(np.all(phase_channels_pass)),
            "pass_rgb": phase_channels_pass.tolist(),
            "required_reduction": PHASE_REDUCTION_MINIMUM,
        },
    }
    objective["pass"] = bool(objective["color"]["pass"] and objective["phase"]["pass"])
    comparison = np.concatenate((crops["linear"], crops["gamma22"], crops["markesteijn"]), axis=1)
    return {
        "aligned_block_count": int(len(selected)),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "objective": objective,
    }, comparison


def write_comparisons(output_directory: Path, comparison: np.ndarray) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for scale in (1, 5):
        image = comparison if scale == 1 else np.repeat(np.repeat(comparison, scale, axis=0), scale, axis=1)
        path = output_directory / f"DSCF0771-linear-gamma22-markesteijn-{scale * 100}.tif"
        tifffile.imwrite(path, image, photometric="rgb", metadata=None)
        outputs[f"comparison_{scale * 100}_percent"] = {
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": file_sha256(path),
            "shape": list(image.shape),
        }
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--gamma22", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gamma-log", type=Path, required=True)
    parser.add_argument("--gamma-time", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--visual-verdict", choices=("pending", "pass", "fail"), default="pending")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    raw_digest = file_sha256(args.raw)
    model_digest = file_sha256(args.model)
    if raw_digest != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW SHA-256 does not match the reviewed DSCF0771.RAF")
    if model_digest != EXPECTED_RTNN_SHA256:
        raise ComparisonError("RTNN SHA-256 does not match the reviewed Gharbi artifact")

    paths = {"linear": args.linear, "gamma22": args.gamma22, "markesteijn": args.markesteijn}
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis, comparison = analyze_arrays(images, args.crop)
    generated = write_comparisons(args.output_dir, comparison)
    execution = parse_execution(
        args.gamma_log.read_text(encoding="utf-8"), args.gamma_time.read_text(encoding="utf-8")
    )
    objective_pass = bool(analysis["objective"]["pass"])
    visual_pass = args.visual_verdict == "pass"
    if args.visual_verdict == "pending":
        decision = "pending-visual-review" if objective_pass else "close-drop-in-experiment"
    else:
        decision = (
            "gamma-quality-passed-runtime-optimization-only"
            if objective_pass and visual_pass
            else "close-drop-in-experiment"
        )

    report = {
        "comparison_order": ["linear", "gamma22", "markesteijn"],
        "crop": {"height": args.crop[3], "width": args.crop[2], "x": args.crop[0], "y": args.crop[1]},
        "decision": decision,
        "execution": execution,
        "format": FORMAT,
        "generated": generated,
        "identity": {"raw_sha256": raw_digest, "rtnn_sha256": model_digest},
        "inputs": {
            name: {"filename": path.name, "sha256": file_sha256(path)} for name, path in paths.items()
        },
        "metadata": common_metadata,
        **analysis,
        "visual": {"note": args.visual_note, "verdict": args.visual_verdict},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json(report))
    print(canonical_json(report).decode("utf-8"), end="")
    return 0 if objective_pass or args.visual_verdict == "pending" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

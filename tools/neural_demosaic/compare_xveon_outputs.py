#!/usr/bin/env python3
"""Compare X-veon with the preserved Gharbi and Markesteijn DSCF0771 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np
from PIL import Image
import tifffile

from .compare_gamma22_outputs import (
    COLOR_COMMON_LIMIT,
    COLOR_RANGE_LIMIT,
    ComparisonError,
    EXPECTED_RAW_SHA256,
    PHASE_RMS_ABSOLUTE_FLOOR,
    aligned_blocks,
    canonical_json,
    file_sha256,
    full_channel_mean,
    low_detail_selection,
    parse_crop,
    phase_statistics,
    tiff_metadata,
    validate_metadata,
)


FORMAT = "rawtherapee-xtrans-xveon-comparison-v1"
MODEL_SHA256 = "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500"
FALLBACK_MARKER = "falling back to 3-pass (Markesteijn)"
COMPLETION_RE = re.compile(
    r"X-veon X-Trans completed:.*?method=(\S+).*?artifact=([0-9a-f]{64}).*?"
    r"ort=(\S+).*?provider=(\S+).*?tile=(\S+).*?overlap=(\d+).*?stride=(\d+).*?"
    r"tiles=(\d+).*?thread_policy=(\S+).*?working_buffer_estimate=(\d+).*?elapsed_us=(\d+)"
)


def parse_execution(log_text: str, timing_text: str) -> dict[str, Any]:
    if FALLBACK_MARKER in log_text:
        raise ComparisonError("X-veon export fell back to Markesteijn")
    matches = COMPLETION_RE.findall(log_text)
    if len(matches) != 1:
        raise ComparisonError("X-veon log must contain exactly one completion diagnostic")
    match = matches[0]
    if match[0] != "xveon-xtrans-onnx" or match[1] != MODEL_SHA256:
        raise ComparisonError("X-veon completion identity differs")
    if match[2] != "1.27.0" or match[3] != "CPUExecutionProvider":
        raise ComparisonError("X-veon ONNX Runtime identity differs")
    timing = dict(
        line.split("=", 1) for line in timing_text.splitlines() if "=" in line
    )
    try:
        elapsed = float(timing.get("elapsed_seconds", timing.get("real_seconds", "")))
        rss = int(timing["max_rss_kb"])
    except (KeyError, ValueError) as error:
        raise ComparisonError("timing lacks elapsed_seconds and max_rss_kb") from error
    return {
        "artifact_sha256": match[1],
        "elapsed_seconds": elapsed,
        "engine_elapsed_us": int(match[10]),
        "max_rss_kb": rss,
        "onnxruntime_version": match[2],
        "overlap": int(match[5]),
        "provider": match[3],
        "stride": int(match[6]),
        "thread_policy": match[8],
        "tile": match[4],
        "tiles": int(match[7]),
        "working_buffer_estimate_bytes": int(match[9]),
    }


def analyze_arrays(images: dict[str, np.ndarray], crop: tuple[int, int, int, int]) -> dict[str, Any]:
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1 or x + width > next(iter(shapes))[1] or y + height > next(iter(shapes))[0]:
        raise ComparisonError("comparison shapes differ or crop lies outside them")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods: dict[str, Any] = {}
    for name, image in images.items():
        cropped = np.asarray(image[y : y + height, x : x + width])
        methods[name] = {
            "crop_mean_rgb": (cropped.mean(axis=(0, 1), dtype=np.float64) / 65535.0).tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }
    xveon_delta = np.asarray(methods["xveon"]["crop_mean_rgb"]) - np.asarray(methods["markesteijn"]["crop_mean_rgb"])
    xveon_rms = np.asarray(methods["xveon"]["phase"]["rms_rgb"])
    mark_rms = np.asarray(methods["markesteijn"]["phase"]["rms_rgb"])
    limits = np.maximum(2 * mark_rms, PHASE_RMS_ABSOLUTE_FLOOR)
    common = float(xveon_delta.mean())
    color_range = float(np.ptp(xveon_delta))

    # The rendered TIFF no longer contains the raw scalar mosaic. This is the
    # closest non-invasive real-RAF check: disagreement from Markesteijn at
    # the globally aligned X-Trans observed-channel locations.
    crop_xveon = np.asarray(images["xveon"][y : y + height, x : x + width], np.float64) / 65535
    crop_mark = np.asarray(images["markesteijn"][y : y + height, x : x + width], np.float64) / 65535
    canonical = np.asarray(((1,2,1,1,0,1),(0,1,0,2,1,2),(1,2,1,1,0,1),
                            (1,0,1,1,2,1),(2,1,2,0,1,0),(1,0,1,1,2,1)), np.uint8)
    rows, columns = np.indices((height, width))
    channels = canonical[(rows + y) % 6, (columns + x) % 6]
    observed_delta = np.take_along_axis(crop_xveon - crop_mark, channels[..., None], axis=2)[..., 0]
    phase_pass = bool(np.all(xveon_rms <= limits))
    color_pass = abs(common) <= COLOR_COMMON_LIMIT and color_range <= COLOR_RANGE_LIMIT
    return {
        "aligned_block_count": int(len(blocks["markesteijn"])),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "objective": {
            "color": {
                "common_luminance_delta": common,
                "common_luminance_limit": COLOR_COMMON_LIMIT,
                "pass": color_pass,
                "rgb_delta_range": color_range,
                "rgb_delta_range_limit": COLOR_RANGE_LIMIT,
                "xveon_minus_markesteijn_rgb": xveon_delta.tolist(),
            },
            "pass": phase_pass and color_pass,
            "phase": {
                "limits_rgb": limits.tolist(),
                "pass": phase_pass,
                "pass_rgb": (xveon_rms <= limits).tolist(),
            },
        },
        "observed_sample_delta_vs_markesteijn_rms": float(
            np.sqrt(np.mean(observed_delta * observed_delta, dtype=np.float64))
        ),
    }


def save_png_assets(xveon: Path, output_dir: Path, icc: bytes) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(xveon) as image:
        rgb = image.convert("RGB")
        full = rgb.resize((2584, 1726), Image.Resampling.LANCZOS)
        crop = rgb.crop((3510, 1930, 3650, 2090)).resize((700, 800), Image.Resampling.NEAREST)
    paths = {
        "full_third": output_dir / "DSCF0771-xveon-full-third.png",
        "earring_500": output_dir / "DSCF0771-xveon-earring-500.png",
    }
    full.save(paths["full_third"], format="PNG", compress_level=9, icc_profile=icc)
    crop.save(paths["earring_500"], format="PNG", compress_level=9, icc_profile=icc)
    return {
        name: {"bytes": path.stat().st_size, "filename": path.name, "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("linear", "gamma22", "xveon", "markesteijn"):
        parser.add_argument(f"--{method}", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--xveon-log", type=Path, action="append", required=True)
    parser.add_argument("--xveon-time", type=Path, action="append", required=True)
    parser.add_argument("--markesteijn-seconds", type=float, required=True)
    parser.add_argument("--markesteijn-rss-kb", type=int, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--visual-verdict", choices=("pending", "pass", "fail"), default="pending")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()
    if file_sha256(args.raw) != EXPECTED_RAW_SHA256 or args.model.stat().st_size != 15_536_134 or file_sha256(args.model) != MODEL_SHA256:
        raise ComparisonError("RAW or X-veon model identity differs")
    if len(args.xveon_log) != len(args.xveon_time) or len(args.xveon_log) != 3:
        raise ComparisonError("exactly three paired measured X-veon logs/timings are required")
    paths = {name: getattr(args, name) for name in ("linear", "gamma22", "xveon", "markesteijn")}
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze_arrays(images, args.crop)
    executions = [parse_execution(log.read_text(), timing.read_text()) for log, timing in zip(args.xveon_log, args.xveon_time)]
    seconds = statistics.median(item["elapsed_seconds"] for item in executions)
    peak_rss = max(item["max_rss_kb"] for item in executions)
    runtime_pass = seconds <= 10 * args.markesteijn_seconds
    memory_pass = peak_rss <= args.markesteijn_rss_kb + 4 * 1024 * 1024
    with tifffile.TiffFile(args.xveon) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_png_assets(args.xveon, args.asset_dir, icc)
    visual_pass = args.visual_verdict == "pass"
    gate_pass = bool(analysis["objective"]["pass"] and runtime_pass and memory_pass and visual_pass)
    report = {
        **analysis,
        "assets": assets,
        "comparison_order": ["linear", "gamma22", "xveon", "markesteijn"],
        "crop": {"x": args.crop[0], "y": args.crop[1], "width": args.crop[2], "height": args.crop[3]},
        "execution": {"measured_runs": executions, "median_seconds": seconds, "peak_rss_kb": peak_rss},
        "format": FORMAT,
        "gate": {
            "memory_pass": memory_pass,
            "pass": gate_pass,
            "runtime_pass": runtime_pass,
            "status": "pass-hidden-license-blocked" if gate_pass else ("pending-visual-review" if args.visual_verdict == "pending" else "fail-hidden"),
        },
        "identity": {"raw_sha256": file_sha256(args.raw), "xveon_onnx_sha256": file_sha256(args.model)},
        "inputs": {name: {"sha256": file_sha256(path)} for name, path in paths.items()},
        "metadata": common_metadata,
        "visual": {"note": args.visual_note, "verdict": args.visual_verdict},
    }
    args.report.write_bytes(canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate_pass or args.visual_verdict == "pending" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

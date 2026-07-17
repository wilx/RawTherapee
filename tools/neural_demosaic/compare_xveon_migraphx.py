"""Compare full-resolution X-veon ONNX Runtime and MIGraphX TIFF exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import tifffile

from .compare_gamma22_outputs import (
    ComparisonError,
    aligned_blocks,
    canonical_json,
    file_sha256,
    low_detail_selection,
    parse_crop,
    phase_statistics,
    tiff_metadata,
    validate_metadata,
)

FORMAT = "rawtherapee-xtrans-xveon-migraphx-comparison-v1"
MODEL_SHA256 = "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500"
COMPLETION_RE = re.compile(
    r"X-veon X-Trans completed:.*?artifact=([0-9a-f]{64}).*?runtime=(\S+).*?"
    r"provider=(\S+).*?compile_source=(\S+).*?compile_us=(\d+).*?"
    r"last_inference_us=(\d+).*?tiles=(\d+).*?working_buffer_estimate=(\d+).*?elapsed_us=(\d+)"
)


def analyze_arrays(
    cpu: np.ndarray,
    gpu: np.ndarray,
    markesteijn: np.ndarray,
    crop: tuple[int, int, int, int],
) -> dict[str, Any]:
    if cpu.shape != gpu.shape or cpu.shape != markesteijn.shape or cpu.dtype != np.uint16 or gpu.dtype != np.uint16:
        raise ComparisonError("CPU, GPU, and Markesteijn arrays must be matching uint16 RGB images")
    if cpu.ndim != 3 or cpu.shape[2] < 3:
        raise ComparisonError("comparison images must have at least three channels")
    cpu = cpu[..., :3]
    gpu = gpu[..., :3]
    markesteijn = markesteijn[..., :3]
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > cpu.shape[1] or y + height > cpu.shape[0]:
        raise ComparisonError("crop lies outside comparison images")

    difference = gpu.astype(np.float64) - cpu.astype(np.float64)
    normalized = difference / 65535.0
    absolute = np.abs(normalized)
    channel_mse = np.mean(normalized * normalized, axis=(0, 1), dtype=np.float64)
    mse = float(np.mean(channel_mse))
    psnr_rgb = [math.inf if value == 0 else -10 * math.log10(float(value)) for value in channel_mse]
    cpsnr = math.inf if mse == 0 else -10 * math.log10(mse)

    blocks = {
        "cpu": aligned_blocks(cpu, crop),
        "gpu": aligned_blocks(gpu, crop),
        "markesteijn": aligned_blocks(markesteijn, crop),
    }
    selected = low_detail_selection(blocks["markesteijn"])
    phases = {name: phase_statistics(value, selected) for name, value in blocks.items()}
    phase_increase = np.asarray(phases["gpu"]["rms_rgb"]) - np.asarray(phases["cpu"]["rms_rgb"])

    canonical = np.asarray(((1,2,1,1,0,1),(0,1,0,2,1,2),(1,2,1,1,0,1),
                            (1,0,1,1,2,1),(2,1,2,0,1,0),(1,0,1,1,2,1)), np.uint8)
    rows, columns = np.indices((height, width))
    channels = canonical[(rows + y) % 6, (columns + x) % 6]
    crop_delta = normalized[y:y + height, x:x + width]
    observed = np.take_along_axis(crop_delta, channels[..., None], axis=2)[..., 0]

    return {
        "all_rgb_samples_equal": bool(np.array_equal(cpu, gpu)),
        "channel_mean_delta_rgb": np.mean(normalized, axis=(0, 1), dtype=np.float64).tolist(),
        "cpsnr_db": cpsnr,
        "maximum_absolute_error": float(np.max(absolute)),
        "mean_absolute_error": float(np.mean(absolute, dtype=np.float64)),
        "observed_sample_rms": float(np.sqrt(np.mean(observed * observed, dtype=np.float64))),
        "p99_absolute_error": float(np.quantile(absolute, 0.99, method="higher")),
        "phase": {
            "cpu": phases["cpu"],
            "gpu": phases["gpu"],
            "increase_rgb": phase_increase.tolist(),
            "pass": bool(np.all(phase_increase <= 0.00025)),
        },
        "psnr_rgb_db": psnr_rgb,
        "rms_absolute_error": float(np.sqrt(np.mean(normalized * normalized, dtype=np.float64))),
        "ssim": float(structural_similarity(cpu, gpu, data_range=65535, channel_axis=2)),
    }


def parse_execution(log: str, timing: str) -> dict[str, Any]:
    if "falling back to 3-pass (Markesteijn)" in log:
        raise ComparisonError("MIGraphX export fell back to Markesteijn")
    matches = COMPLETION_RE.findall(log)
    if len(matches) != 1:
        raise ComparisonError("MIGraphX log lacks one completion diagnostic")
    match = matches[0]
    if match[0] != MODEL_SHA256 or match[1] != "2.15.0-20250912-17-200-gde19b73ad" or match[2] != "MIGraphX-gpu":
        raise ComparisonError("MIGraphX completion identity differs")
    measured = dict(line.split("=", 1) for line in timing.splitlines() if "=" in line)
    return {
        "compile_source": match[3],
        "compile_us": int(match[4]),
        "elapsed_seconds": float(measured["elapsed_seconds"]),
        "engine_elapsed_us": int(match[8]),
        "last_inference_us": int(match[5]),
        "max_rss_kb": int(measured["max_rss_kb"]),
        "tiles": int(match[6]),
        "working_buffer_estimate_bytes": int(match[7]),
    }


def save_assets(gpu_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(gpu_path) as source:
        icc = source.info.get("icc_profile")
        rgb = source.convert("RGB")
        images = {
            "full_third": rgb.resize((2584, 1726), Image.Resampling.LANCZOS),
            "earring_500": rgb.crop((3510, 1930, 3650, 2090)).resize((700, 800), Image.Resampling.NEAREST),
        }
    result = {}
    for name, image in images.items():
        path = output_dir / f"DSCF0771-xveon-migraphx-{name.replace('_', '-')}.png"
        image.save(path, format="PNG", compress_level=9, icc_profile=icc)
        result[name] = {"filename": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--gpu-repeat", type=Path, action="append", default=[])
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, action="append", required=True)
    parser.add_argument("--gpu-time", type=Path, action="append", required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.cpu, args.gpu, args.markesteijn, *args.gpu_repeat]
    metadata = {str(index): tiff_metadata(path) for index, path in enumerate(paths)}
    common_metadata = validate_metadata(metadata)
    arrays = [tifffile.memmap(path) for path in paths]
    analysis = analyze_arrays(arrays[0], arrays[1], arrays[2], args.crop)
    repeated_equal = all(np.array_equal(arrays[1], image) for image in arrays[3:])
    if len(args.gpu_log) != 3 or len(args.gpu_time) != 3:
        raise ComparisonError("exactly three measured GPU logs and timings are required")
    executions = [parse_execution(log.read_text(), timing.read_text())
                  for log, timing in zip(args.gpu_log, args.gpu_time)]
    median = statistics.median(item["elapsed_seconds"] for item in executions)
    peak_rss = max(item["max_rss_kb"] for item in executions)
    gates = {
        "cpsnr_at_least_60_db": analysis["cpsnr_db"] >= 60,
        "mean_delta_at_most_0_0005": max(abs(value) for value in analysis["channel_mean_delta_rgb"]) <= 0.0005,
        "phase_increase_at_most_0_00025": analysis["phase"]["pass"],
        "repeated_pixel_determinism": repeated_equal,
        "runtime_at_least_2x_cpu": median <= 45.60 / 2,
        "runtime_at_most_10x_markesteijn": median <= 3.94 * 10,
        "rss_at_most_4_gib_above_markesteijn": peak_rss <= 1_803_460 + 4 * 1024 * 1024,
        "ssim_at_least_0_999": analysis["ssim"] >= 0.999,
    }
    report = {
        "analysis": analysis,
        "assets": save_assets(args.gpu, args.asset_dir),
        "crop": {"x": args.crop[0], "y": args.crop[1], "width": args.crop[2], "height": args.crop[3]},
        "execution": {"measured_runs": executions, "median_seconds": median, "peak_rss_kb": peak_rss},
        "format": FORMAT,
        "gate": {**gates, "pass": all(gates.values())},
        "inputs": {path.name: file_sha256(path) for path in paths},
        "metadata": common_metadata,
    }
    args.report.write_bytes(canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

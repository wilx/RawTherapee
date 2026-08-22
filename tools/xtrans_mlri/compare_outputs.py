#!/usr/bin/env python3
"""Measure and preserve the DSCF0771 MLRI/Markesteijn comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
import tifffile

from tools.neural_demosaic.compare_gamma22_outputs import (
    ComparisonError,
    EXPECTED_RAW_SHA256,
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


FORMAT = "rawtherapee-xtrans-mlri-comparison-v1"
METHOD = "mlri-xtrans-2pass"
FALLBACK_MARKER = "falling back to 3-pass (Markesteijn)"
COMPLETION_RE = re.compile(
    r"MLRI X-Trans completed: method=(\S+) passes=2 sigma=2,1 epsilon=0\.01 "
    r"core=(\d+)x(\d+) halo=(\d+) boundary=zero tiles=(\d+) workers=(\d+) "
    r"workspace_per_worker_estimate=(\d+) elapsed_us=(\d+)"
)
DETACHED_TRACE_RE = re.compile(
    r'write\(2, "MLRI X-Trans completed: method=mlri-xtrans-2pass '
    r'passes=2 sigma=2,1 epsilon=0\.01 core=384x384 halo=228 boundary=zero '
    r'tiles=(\d+) w", 128\) = -1 EPIPE \(Broken pipe\)'
)


def parse_timing(text: str) -> dict[str, float | int]:
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    try:
        return {
            "elapsed_seconds": float(values["elapsed_seconds"]),
            "max_rss_kb": int(values["max_rss_kb"]),
        }
    except (KeyError, ValueError) as error:
        raise ComparisonError("timing lacks elapsed_seconds or max_rss_kb") from error


def parse_execution(
    log_text: str,
    timing_text: str,
    expected_method: str = METHOD,
) -> dict[str, Any]:
    if FALLBACK_MARKER in log_text:
        raise ComparisonError("MLRI export fell back to Markesteijn")
    matches = COMPLETION_RE.findall(log_text)
    if len(matches) == 1:
        match = matches[0]
        if match[0] != expected_method or match[1:4] != ("384", "384", "228"):
            raise ComparisonError("MLRI method or production tile geometry differs")
        return {
            **parse_timing(timing_text),
            "completion_capture": "complete stderr diagnostic",
            "core": [int(match[1]), int(match[2])],
            "halo": int(match[3]),
            "tiles": int(match[4]),
            "workers": int(match[5]),
            "workspace_per_worker_estimate_bytes": int(match[6]),
            "engine_elapsed_us": int(match[7]),
        }
    if matches:
        raise ComparisonError("MLRI log contains duplicate completion diagnostics")

    # The reviewed DSCF0771 run outlived its original tool session.  A syscall
    # trace authenticated the fixed completion prefix, but stderr's abandoned
    # pipe returned EPIPE after exactly 128 bytes and lost only the dynamic
    # workers/workspace/engine-time suffix.  Accept that exact trace form while
    # keeping ordinary invocations strict.  The two workers and workspace are
    # fixed production-contract values; wall time and peak RSS still come from
    # /usr/bin/time.  Do not invent an engine-only elapsed value.
    if expected_method != METHOD:
        raise ComparisonError("MLRI log must contain exactly one completion diagnostic")
    detached = DETACHED_TRACE_RE.findall(log_text)
    if len(detached) != 1 or detached[0] != "294":
        raise ComparisonError("MLRI log must contain exactly one completion diagnostic")
    return {
        **parse_timing(timing_text),
        "completion_capture": "strace prefix; dynamic stderr suffix lost to EPIPE",
        "core": [384, 384],
        "halo": 228,
        "tiles": 294,
        "workers": 2,
        "workspace_per_worker_estimate_bytes": 508032000,
        "engine_elapsed_us": None,
    }


def method_delta(
    images: dict[str, np.ndarray],
    methods: dict[str, Any],
    left: str,
    right: str,
    crop: tuple[int, int, int, int],
) -> dict[str, Any]:
    x, y, width, height = crop
    delta = np.asarray(methods[left]["crop_mean_rgb"]) - np.asarray(
        methods[right]["crop_mean_rgb"]
    )
    left_crop = np.asarray(images[left][y:y + height, x:x + width], np.float64) / 65535.0
    right_crop = np.asarray(images[right][y:y + height, x:x + width], np.float64) / 65535.0
    canonical = np.asarray(
        ((1,2,1,1,0,1),(0,1,0,2,1,2),(1,2,1,1,0,1),
         (1,0,1,1,2,1),(2,1,2,0,1,0),(1,0,1,1,2,1)),
        np.uint8,
    )
    rows, columns = np.indices((height, width))
    channels = canonical[(rows + y) % 6, (columns + x) % 6]
    observed = np.take_along_axis(
        left_crop - right_crop, channels[..., None], axis=2
    )[..., 0]
    return {
        "common_luminance_delta": float(delta.mean()),
        "crop_mean_rgb_delta": delta.tolist(),
        "rgb_delta_range": float(np.ptp(delta)),
        "observed_sample_delta_rms": float(
            np.sqrt(np.mean(observed * observed, dtype=np.float64))
        ),
    }


def analyze_arrays(
    images: dict[str, np.ndarray], crop: tuple[int, int, int, int]
) -> dict[str, Any]:
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1 or x + width > next(iter(shapes))[1] or y + height > next(iter(shapes))[0]:
        raise ComparisonError("comparison shapes differ or crop lies outside them")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods: dict[str, Any] = {}
    for name, image in images.items():
        cropped = np.asarray(image[y:y + height, x:x + width])
        methods[name] = {
            "crop_mean_rgb": (cropped.mean(axis=(0, 1), dtype=np.float64) / 65535.0).tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }

    return {
        "aligned_block_count": int(len(blocks["markesteijn"])),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "mlri_minus_markesteijn": method_delta(
            images, methods, "mlri", "markesteijn", crop
        ),
    }


def save_assets(
    source: Path,
    output_dir: Path,
    icc: bytes,
    method: str = METHOD,
    filename_token: str = "mlri",
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        images = {
            "full-frame-third": (
                rgb.resize((2584, 1726), Image.Resampling.LANCZOS),
                output_dir / f"DSCF0771-{filename_token}-full-third.png",
            ),
            "earring-500-percent": (
                rgb.crop((3510, 1930, 3650, 2090)).resize(
                    (700, 800), Image.Resampling.NEAREST
                ),
                output_dir / f"DSCF0771-{filename_token}-earring-500.png",
            ),
        }
    result = []
    for role, (image, path) in images.items():
        image.save(path, format="PNG", compress_level=9, icc_profile=icc)
        result.append({
            "bytes": path.stat().st_size,
            "filename": path.name,
            "method": method,
            "role": role,
            "sha256": file_sha256(path),
            "shape": [image.height, image.width, 3],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlri", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mlri-log", type=Path, required=True)
    parser.add_argument("--mlri-time", type=Path, required=True)
    parser.add_argument("--markesteijn-time", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-verdict", choices=("pending", "pass", "fail"), default="pending")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    if file_sha256(args.raw) != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW identity differs from DSCF0771")
    paths = {"mlri": args.mlri, "markesteijn": args.markesteijn}
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze_arrays(images, args.crop)
    execution = parse_execution(
        args.mlri_log.read_text(encoding="utf-8"),
        args.mlri_time.read_text(encoding="utf-8"),
    )
    mark_timing = parse_timing(args.markesteijn_time.read_text(encoding="utf-8"))
    with tifffile.TiffFile(args.mlri) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(args.mlri, args.asset_dir, icc)
    manifest = {
        **analysis,
        "assets": assets,
        "comparison_order": [METHOD, "3-pass (Markesteijn)"],
        "crop": {"x": args.crop[0], "y": args.crop[1], "width": args.crop[2], "height": args.crop[3]},
        "execution": {"markesteijn": mark_timing, "mlri": execution},
        "format": FORMAT,
        "generation": {
            "earring": {
                "filter": "nearest-neighbour", "output_depth_bits": 8,
                "output_shape": [800, 700, 3], "scale": 5,
                "source_rectangle": {"x": 3510, "y": 1930, "width": 140, "height": 160},
            },
            "full_frame": {
                "filter": "Lanczos", "output_depth_bits": 8,
                "output_shape": [1726, 2584, 3], "source_shape": [5178, 7752, 3],
            },
            "png": {"compression_level": 9, "lossless_at_output_depth": True},
        },
        "identity": {"raw_filename": args.raw.name, "raw_sha256": file_sha256(args.raw)},
        "metadata": common_metadata,
        "source_tiffs": [
            {"bytes": path.stat().st_size, "method": METHOD if name == "mlri" else "3-pass (Markesteijn)",
             "sha256": file_sha256(path)}
            for name, path in paths.items()
        ],
        "visual": {"note": args.visual_note, "verdict": args.visual_verdict},
    }
    args.manifest.write_bytes(canonical_json(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

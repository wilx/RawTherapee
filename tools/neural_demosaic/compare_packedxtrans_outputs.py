#!/usr/bin/env python3
"""Compare PackedXTransNet with X-veon and Markesteijn on DSCF0771."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


FORMAT = "rawtherapee-xtrans-packedxtrans-comparison-v1"
MODEL_SHA256 = "ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c"
MODEL_BYTES = 1_673_648
CANONICAL_CFA = np.asarray(
    ((0, 2, 1, 2, 0, 1), (1, 1, 0, 1, 1, 2), (1, 1, 2, 1, 1, 0),
     (2, 0, 1, 0, 2, 1), (1, 1, 2, 1, 1, 0), (1, 1, 0, 1, 1, 2)),
    np.uint8,
)


def analyze_arrays(images: dict[str, np.ndarray], crop: tuple[int, int, int, int]) -> dict[str, Any]:
    if set(images) != {"packedxtransnet", "xveon", "markesteijn"}:
        raise ComparisonError("comparison requires PackedXTransNet, X-veon, and Markesteijn")
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1 or x < 0 or y < 0 or x + width > next(iter(shapes))[1] or y + height > next(iter(shapes))[0]:
        raise ComparisonError("comparison shapes differ or crop lies outside them")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods: dict[str, Any] = {}
    for name, image in images.items():
        cropped = np.asarray(image[y:y + height, x:x + width])
        methods[name] = {
            "crop_mean_rgb": (cropped.mean(axis=(0, 1), dtype=np.float64) / 65535).tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }
    packed_delta = np.asarray(methods["packedxtransnet"]["crop_mean_rgb"]) - np.asarray(methods["markesteijn"]["crop_mean_rgb"])
    packed_rms = np.asarray(methods["packedxtransnet"]["phase"]["rms_rgb"])
    mark_rms = np.asarray(methods["markesteijn"]["phase"]["rms_rgb"])
    limits = np.maximum(2 * mark_rms, PHASE_RMS_ABSOLUTE_FLOOR)
    common = float(packed_delta.mean())
    color_range = float(np.ptp(packed_delta))
    packed_crop = np.asarray(images["packedxtransnet"][y:y + height, x:x + width], np.float64) / 65535
    mark_crop = np.asarray(images["markesteijn"][y:y + height, x:x + width], np.float64) / 65535
    rows, columns = np.indices((height, width))
    channels = CANONICAL_CFA[(rows + y) % 6, (columns + x) % 6]
    observed = np.take_along_axis(packed_crop - mark_crop, channels[..., None], axis=2)[..., 0]
    phase_pass = bool(np.all(packed_rms <= limits))
    color_pass = abs(common) <= COLOR_COMMON_LIMIT and color_range <= COLOR_RANGE_LIMIT
    return {
        "aligned_block_count": int(len(blocks["markesteijn"])),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "objective": {
            "color": {
                "common_luminance_delta": common,
                "limit": COLOR_COMMON_LIMIT,
                "pass": color_pass,
                "rgb_delta_range": color_range,
                "rgb_delta_range_limit": COLOR_RANGE_LIMIT,
                "packedxtransnet_minus_markesteijn_rgb": packed_delta.tolist(),
            },
            "pass": phase_pass and color_pass,
            "phase": {
                "limits_rgb": limits.tolist(),
                "pass": phase_pass,
                "pass_rgb": (packed_rms <= limits).tolist(),
            },
        },
        "observed_sample_delta_vs_markesteijn_rms": float(np.sqrt(np.mean(observed * observed))),
    }


def save_assets(source: Path, output_dir: Path, icc: bytes) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        earring = rgb.crop((3510, 1930, 3650, 2090)).resize((700, 800), Image.Resampling.NEAREST)
    paths = {
        "earring_500": output_dir / "DSCF0771-packedxtransnet-earring-500.png",
    }
    earring.save(paths["earring_500"], format="PNG", compress_level=9, icc_profile=icc)
    return {name: {"bytes": path.stat().st_size, "filename": path.name, "sha256": file_sha256(path)}
            for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("packedxtransnet", "xveon", "markesteijn"):
        parser.add_argument(f"--{method}", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--visual-verdict", choices=("pass", "fail", "pending"), default="pending")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()
    if file_sha256(args.raw) != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW identity differs")
    if args.model.stat().st_size != MODEL_BYTES or file_sha256(args.model) != MODEL_SHA256:
        raise ComparisonError("PackedXTransNet ONNX identity differs")
    paths = {name: getattr(args, name) for name in ("packedxtransnet", "xveon", "markesteijn")}
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    analysis = analyze_arrays({name: tifffile.memmap(path) for name, path in paths.items()}, args.crop)
    with tifffile.TiffFile(args.packedxtransnet) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(args.packedxtransnet, args.asset_dir, icc)
    visual_pass = args.visual_verdict == "pass"
    report = {
        **analysis,
        "assets": assets,
        "comparison_order": ["packedxtransnet", "xveon", "markesteijn"],
        "crop": {"x": args.crop[0], "y": args.crop[1], "width": args.crop[2], "height": args.crop[3]},
        "format": FORMAT,
        "gate": {
            "objective_pass": analysis["objective"]["pass"],
            "visual_pass": visual_pass,
            "pass": bool(analysis["objective"]["pass"] and visual_pass),
        },
        "identity": {"raw_sha256": file_sha256(args.raw), "packedxtrans_onnx_sha256": file_sha256(args.model)},
        "inputs": {name: {"sha256": file_sha256(path)} for name, path in paths.items()},
        "metadata": common_metadata,
        "visual": {"note": args.visual_note, "verdict": args.visual_verdict},
    }
    args.report.write_bytes(canonical_json(report))
    manifest = {
        "format": "rawtherapee-xtrans-packedxtrans-assets-v1",
        "generation": {
            "earring": "source rectangle (3510,1930,140,160), nearest-neighbour 5x",
            "full": "2584x1726 Lanczos downsample from 7752x5178 TIFF",
        },
        "identity": report["identity"],
        "images": assets,
        "source_tiff_sha256": file_sha256(args.packedxtransnet),
    }
    (args.asset_dir / "packedxtransnet-manifest.json").write_bytes(canonical_json(manifest))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if args.visual_verdict != "fail" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

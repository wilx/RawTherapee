#!/usr/bin/env python3
"""Compare the two triangulation experiments with Markesteijn on DSCF0771."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


FORMAT = "rawtherapee-xtrans-triangulation-comparison-v1"
METHODS = {
    "independent": "xtrans-triangulated-rgb",
    "chroma": "xtrans-triangulated-chroma",
    "markesteijn": "3-pass (Markesteijn)",
}
DSCF0771_CFA = np.asarray(
    ((1, 1, 0, 1, 1, 2), (1, 1, 2, 1, 1, 0), (2, 0, 1, 0, 2, 1),
     (1, 1, 2, 1, 1, 0), (1, 1, 0, 1, 1, 2), (0, 2, 1, 2, 0, 1)),
    np.uint8,
)


def method_delta(
    images: dict[str, np.ndarray],
    means: dict[str, np.ndarray],
    left: str,
    right: str,
    crop: tuple[int, int, int, int],
) -> dict[str, Any]:
    x, y, width, height = crop
    delta = means[left] - means[right]
    left_crop = np.asarray(images[left][y:y + height, x:x + width], np.float64)
    right_crop = np.asarray(images[right][y:y + height, x:x + width], np.float64)
    normalized_delta = (left_crop - right_crop) / 65535.0
    rows, columns = np.indices((height, width))
    channels = DSCF0771_CFA[(rows + y) % 6, (columns + x) % 6]
    observed = np.take_along_axis(normalized_delta, channels[..., None], axis=2)[..., 0]
    absolute = np.abs(normalized_delta)
    return {
        "common_luminance_delta": float(delta.mean()),
        "crop_mean_rgb_delta": delta.tolist(),
        "maximum_absolute_pixel_delta": float(absolute.max()),
        "observed_sample_delta_rms": float(np.sqrt(np.mean(observed * observed))),
        "rgb_delta_range": float(np.ptp(delta)),
        "rms_pixel_delta": float(np.sqrt(np.mean(normalized_delta * normalized_delta))),
    }


def analyze(
    images: dict[str, np.ndarray], crop: tuple[int, int, int, int]
) -> dict[str, Any]:
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1:
        raise ComparisonError("TIFF shapes differ")
    shape = next(iter(shapes))
    if x < 0 or y < 0 or x + width > shape[1] or y + height > shape[0]:
        raise ComparisonError("crop lies outside the TIFFs")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods: dict[str, Any] = {}
    means: dict[str, np.ndarray] = {}
    for name, image in images.items():
        cropped = np.asarray(image[y:y + height, x:x + width])
        means[name] = cropped.mean(axis=(0, 1), dtype=np.float64) / 65535.0
        methods[name] = {
            "crop_mean_rgb": means[name].tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }
    return {
        "aligned_block_count": int(len(blocks["markesteijn"])),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "independent_minus_markesteijn": method_delta(
            images, means, "independent", "markesteijn", crop),
        "chroma_minus_markesteijn": method_delta(
            images, means, "chroma", "markesteijn", crop),
        "chroma_minus_independent": method_delta(
            images, means, "chroma", "independent", crop),
    }


def save_assets(
    paths: dict[str, Path], output_directory: Path, icc: bytes
) -> list[dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    generated: dict[str, dict[str, Image.Image]] = {}
    for name, source in paths.items():
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            generated[name] = {
                "full-third": rgb.resize((2584, 1726), Image.Resampling.LANCZOS),
                "earring-500": rgb.crop((3510, 1930, 3650, 2090)).resize(
                    (700, 800), Image.Resampling.NEAREST
                ),
            }
        for role, asset in generated[name].items():
            filename = f"DSCF0771-triangulation-{name}-{role}.png"
            destination = output_directory / filename
            asset.save(destination, format="PNG", compress_level=9, icc_profile=icc)
            assets.append({
                "bytes": destination.stat().st_size,
                "filename": filename,
                "method": METHODS[name],
                "role": role,
                "sha256": file_sha256(destination),
                "shape": [asset.height, asset.width, 3],
            })

    for role in ("full-third", "earring-500"):
        comparison = Image.new(
            "RGB",
            (sum(generated[name][role].width for name in METHODS),
             generated["independent"][role].height),
        )
        offset = 0
        for name in METHODS:
            comparison.paste(generated[name][role], (offset, 0))
            offset += generated[name][role].width
        filename = f"DSCF0771-triangulation-comparison-{role}.png"
        destination = output_directory / filename
        comparison.save(destination, format="PNG", compress_level=9, icc_profile=icc)
        assets.append({
            "bytes": destination.stat().st_size,
            "filename": filename,
            "method": "independent | chroma | Markesteijn",
            "role": f"comparison-{role}",
            "sha256": file_sha256(destination),
            "shape": [comparison.height, comparison.width, 3],
        })
    return assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--independent-seconds", type=float, required=True)
    parser.add_argument("--chroma-seconds", type=float, required=True)
    parser.add_argument("--markesteijn-seconds", type=float, required=True)
    parser.add_argument("--independent-engine-us", type=int, required=True)
    parser.add_argument("--chroma-engine-us", type=int, required=True)
    parser.add_argument("--independent-rss-kb", type=int, required=True)
    parser.add_argument("--chroma-rss-kb", type=int, required=True)
    parser.add_argument("--markesteijn-rss-kb", type=int, required=True)
    args = parser.parse_args()

    if file_sha256(args.raw) != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW identity differs from reviewed DSCF0771.RAF")
    paths = {
        "independent": args.independent,
        "chroma": args.chroma,
        "markesteijn": args.markesteijn,
    }
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze(images, args.crop)
    with tifffile.TiffFile(args.independent) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(paths, args.asset_dir, icc)
    manifest = {
        **analysis,
        "assets": assets,
        "comparison_order": [METHODS[name] for name in METHODS],
        "crop": {
            "x": args.crop[0], "y": args.crop[1],
            "width": args.crop[2], "height": args.crop[3],
        },
        "execution": {
            "independent": {
                "elapsed_seconds": args.independent_seconds,
                "engine_elapsed_us": args.independent_engine_us,
                "max_rss_kb": args.independent_rss_kb,
                "workers": 16,
            },
            "chroma": {
                "elapsed_seconds": args.chroma_seconds,
                "engine_elapsed_us": args.chroma_engine_us,
                "max_rss_kb": args.chroma_rss_kb,
                "workers": 16,
            },
            "markesteijn": {
                "elapsed_seconds": args.markesteijn_seconds,
                "max_rss_kb": args.markesteijn_rss_kb,
                "workers": 16,
            },
        },
        "format": FORMAT,
        "generation": {
            "earring": {
                "filter": "nearest-neighbour", "scale": 5,
                "source_rectangle": {"x": 3510, "y": 1930, "width": 140, "height": 160},
            },
            "full_frame": {"filter": "Lanczos", "output_shape": [1726, 2584, 3]},
            "output_depth_bits": 8,
            "png_compression_level": 9,
        },
        "identity": {"raw_filename": args.raw.name, "raw_sha256": file_sha256(args.raw)},
        "metadata": common_metadata,
        "source_tiffs": [
            {
                "bytes": path.stat().st_size,
                "method": METHODS[name],
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        ],
    }
    args.manifest.write_bytes(canonical_json(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        raise SystemExit(f"error: {error}") from error

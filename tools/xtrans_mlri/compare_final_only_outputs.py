#!/usr/bin/env python3
"""Compare corrected two-pass MLRI with and without its final chroma blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
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
from tools.xtrans_mlri.compare_corrected_outputs import pixel_difference_summary
from tools.xtrans_mlri.compare_outputs import (
    FALLBACK_MARKER,
    method_delta,
    parse_timing,
    save_assets,
)


FORMAT = "rawtherapee-xtrans-mlri-corrected-final-only-comparison-v1"
METHOD = "mlri-xtrans-2pass-corrected-final-only"
CORRECTED_METHOD = "mlri-xtrans-2pass-corrected"
MARK_METHOD = "3-pass (Markesteijn)"
BLUE_OUTLIER_PROBE = (3510, 1990, 26, 41)
BLUE_OUTLIER_COUNT = 10
COMPLETION_RE = re.compile(
    r"MLRI X-Trans completed: method=(\S+) passes=2 sigma=2,1 epsilon=0\.01 "
    r"blue_diagonal_guides=corrected final=direct core=(\d+)x(\d+) halo=(\d+) "
    r"boundary=zero tiles=(\d+) workers=(\d+) "
    r"workspace_per_worker_estimate=(\d+) elapsed_us=(\d+)"
)


def parse_final_only_execution(log_text: str, timing_text: str) -> dict[str, Any]:
    if FALLBACK_MARKER in log_text:
        raise ComparisonError("corrected final-only MLRI export fell back")
    matches = COMPLETION_RE.findall(log_text)
    if len(matches) != 1:
        raise ComparisonError("final-only log must contain one completion diagnostic")
    match = matches[0]
    if match[0] != METHOD or match[1:4] != ("384", "384", "228"):
        raise ComparisonError("final-only method or tile geometry differs")
    return {
        **parse_timing(timing_text),
        "blue_diagonal_guides": "corrected",
        "core": [int(match[1]), int(match[2])],
        "engine_elapsed_us": int(match[7]),
        "final_reconstruction": "direct",
        "halo": int(match[3]),
        "passes": 2,
        "tiles": int(match[4]),
        "workers": int(match[5]),
        "workspace_per_worker_estimate_bytes": int(match[6]),
    }


def analyze_final_only_arrays(
    images: dict[str, np.ndarray], crop: tuple[int, int, int, int]
) -> dict[str, Any]:
    x, y, width, height = crop
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1 or x + width > next(iter(shapes))[1] \
            or y + height > next(iter(shapes))[0]:
        raise ComparisonError("comparison shapes differ or crop lies outside them")
    blocks = {name: aligned_blocks(image, crop) for name, image in images.items()}
    selected = low_detail_selection(blocks["markesteijn"])
    methods: dict[str, Any] = {}
    for name, image in images.items():
        cropped = np.asarray(image[y:y + height, x:x + width])
        methods[name] = {
            "crop_mean_rgb": (
                cropped.mean(axis=(0, 1), dtype=np.float64) / 65535.0
            ).tolist(),
            "full_mean_rgb": full_channel_mean(image).tolist(),
            "phase": phase_statistics(blocks[name], selected),
        }
    return {
        "aligned_block_count": int(len(blocks["markesteijn"])),
        "low_detail_block_count": int(np.count_nonzero(selected)),
        "methods": methods,
        "final_only_minus_corrected_blend": method_delta(
            images, methods, "final_only", "corrected", crop
        ),
        "final_only_minus_markesteijn": method_delta(
            images, methods, "final_only", "markesteijn", crop
        ),
    }


def local_blue_excess(
    image: np.ndarray, x: int, y: int, radius: int = 2
) -> float:
    """Return a pixel's B-G excess over its surrounding local median."""
    if radius < 1 or x < radius or y < radius \
            or x + radius >= image.shape[1] \
            or y + radius >= image.shape[0]:
        raise ComparisonError("blue-outlier neighborhood lies outside the image")
    neighborhood = np.asarray(
        image[y - radius:y + radius + 1, x - radius:x + radius + 1],
        dtype=np.float64,
    )
    blue_green = (neighborhood[..., 2] - neighborhood[..., 1]) / 65535.0
    center = radius * (2 * radius + 1) + radius
    local_median = np.median(np.delete(blue_green.ravel(), center))
    return float(blue_green[radius, radius] - local_median)


def blue_outlier_probe(
    images: dict[str, np.ndarray],
    region: tuple[int, int, int, int] = BLUE_OUTLIER_PROBE,
    count: int = BLUE_OUTLIER_COUNT,
) -> dict[str, Any]:
    """Measure the worst corrected-MLRI blue excursions in the red object."""
    required = {"final_only", "corrected", "markesteijn"}
    if set(images) != required or count < 1:
        raise ComparisonError("blue-outlier probe requires the reviewed methods")
    x, y, width, height = region
    shape = images["corrected"].shape
    if width < 1 or height < 1 or x < 2 or y < 2 \
            or x + width + 2 > shape[1] or y + height + 2 > shape[0]:
        raise ComparisonError("blue-outlier probe region lies outside the image")
    candidates = [
        (local_blue_excess(images["corrected"], column, row), column, row)
        for row in range(y, y + height)
        for column in range(x, x + width)
    ]
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    selected = candidates[:min(count, len(candidates))]
    methods: dict[str, Any] = {}
    for name, image in images.items():
        values = [
            local_blue_excess(image, column, row)
            for _, column, row in selected
        ]
        methods[name] = {
            "local_blue_excess": values,
            "maximum": float(max(values)),
            "mean": float(np.mean(values, dtype=np.float64)),
        }
    return {
        "definition": (
            "pixel B-G minus median B-G of the other 24 pixels in its 5x5 "
            "neighborhood, normalized by 65535"
        ),
        "region": {"x": x, "y": y, "width": width, "height": height},
        "selection": (
            "ten greatest corrected-MLRI local blue excess values; "
            "descending value with y,x tie order"
        ),
        "coordinates": [
            {"x": column, "y": row} for _, column, row in selected
        ],
        "methods": methods,
    }


def source_digest(manifest: dict[str, Any], method: str) -> str:
    values = [
        item["sha256"] for item in manifest["source_tiffs"]
        if item["method"] == method
    ]
    if len(values) != 1:
        raise ComparisonError(f"baseline manifest lacks one {method} source")
    return values[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-only", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--final-only-log", type=Path, required=True)
    parser.add_argument("--final-only-time", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-verdict", choices=("pending", "pass", "fail"),
                        default="pending")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    if file_sha256(args.raw) != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW identity differs from DSCF0771")
    try:
        baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError("baseline manifest is not valid JSON") from error
    if baseline.get("format") != "rawtherapee-xtrans-mlri-corrected-comparison-v1":
        raise ComparisonError("corrected baseline manifest format differs")
    if file_sha256(args.corrected) != source_digest(baseline, CORRECTED_METHOD):
        raise ComparisonError("corrected TIFF differs from reviewed baseline")
    if file_sha256(args.markesteijn) != source_digest(baseline, MARK_METHOD):
        raise ComparisonError("Markesteijn TIFF differs from reviewed baseline")

    paths = {
        "final_only": args.final_only,
        "corrected": args.corrected,
        "markesteijn": args.markesteijn,
    }
    common_metadata = validate_metadata({
        name: tiff_metadata(path) for name, path in paths.items()
    })
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze_final_only_arrays(images, args.crop)
    analysis["blue_outlier_probe"] = blue_outlier_probe(images)
    analysis["pixel_differences"] = {
        "final_only_minus_corrected_blend": {
            "crop": pixel_difference_summary(
                images["final_only"], images["corrected"], args.crop
            ),
            "full_frame": pixel_difference_summary(
                images["final_only"], images["corrected"]
            ),
        },
        "final_only_minus_markesteijn": {
            "crop": pixel_difference_summary(
                images["final_only"], images["markesteijn"], args.crop
            ),
            "full_frame": pixel_difference_summary(
                images["final_only"], images["markesteijn"]
            ),
        },
    }
    execution = parse_final_only_execution(
        args.final_only_log.read_text(encoding="utf-8"),
        args.final_only_time.read_text(encoding="utf-8"),
    )
    with tifffile.TiffFile(args.final_only) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(
        args.final_only, args.asset_dir, icc,
        method=METHOD, filename_token="mlri-corrected-final-only"
    )
    manifest = {
        **analysis,
        "assets": assets,
        "baseline_manifest_sha256": file_sha256(args.baseline_manifest),
        "comparison_order": [METHOD, CORRECTED_METHOD, MARK_METHOD],
        "controlled_change": {
            "blue_diagonal_guides": "corrected",
            "changed_operation": "select final green-guided red/blue directly",
            "green_passes": 2,
            "other_parameter_changes": 0,
            "removed_operation": "sqrt(G/255) provisional/final chroma blend",
        },
        "crop": {
            "x": args.crop[0], "y": args.crop[1],
            "width": args.crop[2], "height": args.crop[3]
        },
        "execution": {
            "final_only": execution,
            "corrected": baseline["execution"]["corrected"],
            "markesteijn": baseline["execution"]["markesteijn"],
        },
        "format": FORMAT,
        "generation": baseline["generation"],
        "identity": baseline["identity"],
        "metadata": common_metadata,
        "source_tiffs": [
            {
                "bytes": path.stat().st_size,
                "method": (
                    METHOD if name == "final_only" else
                    CORRECTED_METHOD if name == "corrected" else MARK_METHOD
                ),
                "sha256": file_sha256(path),
            }
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

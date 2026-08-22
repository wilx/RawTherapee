#!/usr/bin/env python3
"""Compare the corrected MLRI blue-guide variant with reviewed baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from tools.neural_demosaic.compare_gamma22_outputs import (
    ComparisonError,
    EXPECTED_RAW_SHA256,
    canonical_json,
    file_sha256,
    parse_crop,
    tiff_metadata,
    validate_metadata,
)
from tools.xtrans_mlri.compare_outputs import (
    METHOD,
    analyze_arrays,
    method_delta,
    parse_execution,
    save_assets,
)


FORMAT = "rawtherapee-xtrans-mlri-corrected-comparison-v1"
CORRECTED_METHOD = "mlri-xtrans-2pass-corrected"


def pixel_difference_summary(
    left: np.ndarray,
    right: np.ndarray,
    bounds: tuple[int, int, int, int] | None = None,
) -> dict[str, list[float] | list[int]]:
    """Measure RGB differences without materializing a full-frame float copy."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ComparisonError("pixel-difference image shapes differ")
    if bounds is None:
        x, y, width, height = 0, 0, left.shape[1], left.shape[0]
    else:
        x, y, width, height = bounds
    if min(x, y, width, height) < 0 or x + width > left.shape[1] \
            or y + height > left.shape[0] or width == 0 or height == 0:
        raise ComparisonError("pixel-difference bounds are invalid")

    count = width * height
    squared_sum = np.zeros(3, dtype=np.float64)
    maximum = np.zeros(3, dtype=np.int64)
    histograms = np.zeros((3, 65536), dtype=np.int64)
    for row in range(y, y + height, 128):
        stop = min(row + 128, y + height)
        delta = np.abs(
            np.asarray(left[row:stop, x:x + width], dtype=np.int32)
            - np.asarray(right[row:stop, x:x + width], dtype=np.int32)
        )
        squared_sum += np.square(delta, dtype=np.float64).sum(axis=(0, 1))
        maximum = np.maximum(maximum, delta.max(axis=(0, 1)))
        for channel in range(3):
            histograms[channel] += np.bincount(
                delta[..., channel].ravel(), minlength=65536
            )

    rms = np.sqrt(squared_sum / count)
    rank_99 = (count * 99 + 99) // 100
    percentile_99 = np.asarray([
        np.searchsorted(np.cumsum(histograms[channel]), rank_99)
        for channel in range(3)
    ], dtype=np.int64)
    return {
        "max_abs_code_rgb": maximum.tolist(),
        "max_abs_normalized_rgb": (maximum / 65535.0).tolist(),
        "p99_abs_code_rgb": percentile_99.tolist(),
        "p99_abs_normalized_rgb": (percentile_99 / 65535.0).tolist(),
        "rms_code_rgb": rms.tolist(),
        "rms_normalized_rgb": (rms / 65535.0).tolist(),
    }


def _source_digest(manifest: dict, method: str) -> str:
    matches = [
        item["sha256"] for item in manifest["source_tiffs"]
        if item["method"] == method
    ]
    if len(matches) != 1:
        raise ComparisonError(f"baseline manifest lacks one {method} source identity")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--faithful", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--corrected-log", type=Path, required=True)
    parser.add_argument("--corrected-time", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, default=(3450, 1750, 700, 500))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--visual-verdict", choices=("pending", "pass", "fail"), default="pending"
    )
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    if file_sha256(args.raw) != EXPECTED_RAW_SHA256:
        raise ComparisonError("RAW identity differs from DSCF0771")
    baseline_bytes = args.baseline_manifest.read_bytes()
    try:
        baseline = json.loads(baseline_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError("baseline manifest is not valid JSON") from error
    if baseline.get("format") != "rawtherapee-xtrans-mlri-comparison-v1":
        raise ComparisonError("baseline manifest format differs")
    if baseline.get("identity", {}).get("raw_sha256") != EXPECTED_RAW_SHA256:
        raise ComparisonError("baseline manifest RAW identity differs")
    if file_sha256(args.faithful) != _source_digest(baseline, METHOD):
        raise ComparisonError("faithful TIFF differs from the reviewed baseline")
    if file_sha256(args.markesteijn) != _source_digest(
        baseline, "3-pass (Markesteijn)"
    ):
        raise ComparisonError("Markesteijn TIFF differs from the reviewed baseline")

    paths = {
        "corrected": args.corrected,
        "mlri": args.faithful,
        "markesteijn": args.markesteijn,
    }
    metadata = {name: tiff_metadata(path) for name, path in paths.items()}
    common_metadata = validate_metadata(metadata)
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze_arrays(images, args.crop)
    methods = analysis["methods"]
    analysis["corrected_minus_faithful"] = method_delta(
        images, methods, "corrected", "mlri", args.crop
    )
    analysis["corrected_minus_markesteijn"] = method_delta(
        images, methods, "corrected", "markesteijn", args.crop
    )
    analysis["pixel_differences"] = {
        "corrected_minus_faithful": {
            "crop": pixel_difference_summary(
                images["corrected"], images["mlri"], args.crop
            ),
            "full_frame": pixel_difference_summary(
                images["corrected"], images["mlri"]
            ),
        },
        "corrected_minus_markesteijn": {
            "crop": pixel_difference_summary(
                images["corrected"], images["markesteijn"], args.crop
            ),
            "full_frame": pixel_difference_summary(
                images["corrected"], images["markesteijn"]
            ),
        },
    }

    corrected_execution = parse_execution(
        args.corrected_log.read_text(encoding="utf-8"),
        args.corrected_time.read_text(encoding="utf-8"),
        CORRECTED_METHOD,
    )
    with tifffile.TiffFile(args.corrected) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(
        args.corrected,
        args.asset_dir,
        icc,
        method=CORRECTED_METHOD,
        filename_token="mlri-corrected",
    )
    manifest = {
        **analysis,
        "assets": assets,
        "baseline_manifest_sha256": file_sha256(args.baseline_manifest),
        "comparison_order": [
            CORRECTED_METHOD,
            METHOD,
            "3-pass (Markesteijn)",
        ],
        "correction": {
            "changed_expressions": 4,
            "description": (
                "Use blue rather than red directional guides for blue "
                "diagonal and anti-diagonal Laplacians in both MLRI stages."
            ),
            "other_parameter_changes": 0,
        },
        "crop": {
            "x": args.crop[0],
            "y": args.crop[1],
            "width": args.crop[2],
            "height": args.crop[3],
        },
        "execution": {
            "corrected": corrected_execution,
            "faithful": baseline["execution"]["mlri"],
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
                    CORRECTED_METHOD if name == "corrected"
                    else METHOD if name == "mlri"
                    else "3-pass (Markesteijn)"
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

#!/usr/bin/env python3
"""Measure and preserve the controlled 2014/2016 MLRI paper-core comparison."""

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
from tools.xtrans_mlri.compare_outputs import (
    FALLBACK_MARKER,
    method_delta,
    parse_timing,
    save_assets,
)


FORMAT = "rawtherapee-xtrans-mlri-paper-core-comparison-v1"
METHOD_2014 = "mlri-xtrans-paper-core-2014"
METHOD_2016 = "mlri-xtrans-paper-core-2016"
COMPLETION_RE = re.compile(
    r"MLRI X-Trans completed: method=(\S+) passes=1 sigma=2 epsilon=0\.01 "
    r"coefficient_average=(uniform|residual-weighted) final=direct "
    r"core=(\d+)x(\d+) halo=(\d+) boundary=zero tiles=(\d+) workers=(\d+) "
    r"workspace_per_worker_estimate=(\d+) elapsed_us=(\d+)"
)


def parse_paper_execution(
    log_text: str, timing_text: str, expected_method: str
) -> dict[str, Any]:
    if FALLBACK_MARKER in log_text:
        raise ComparisonError("paper-core MLRI export fell back to Markesteijn")
    matches = COMPLETION_RE.findall(log_text)
    if len(matches) != 1:
        raise ComparisonError("paper-core log must contain one completion diagnostic")
    match = matches[0]
    expected_average = "uniform" if expected_method == METHOD_2014 else "residual-weighted"
    if match[0] != expected_method or match[1] != expected_average:
        raise ComparisonError("paper-core method or coefficient average differs")
    if match[2:5] != ("384", "384", "228"):
        raise ComparisonError("paper-core production tile geometry differs")
    return {
        **parse_timing(timing_text),
        "coefficient_average": match[1],
        "core": [int(match[2]), int(match[3])],
        "engine_elapsed_us": int(match[8]),
        "final_reconstruction": "direct",
        "halo": int(match[4]),
        "passes": 1,
        "tiles": int(match[5]),
        "workers": int(match[6]),
        "workspace_per_worker_estimate_bytes": int(match[7]),
    }


def analyze_paper_arrays(
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
        "paper_2014_minus_2016": method_delta(
            images, methods, "paper2014", "paper2016", crop
        ),
        "paper_2014_minus_markesteijn": method_delta(
            images, methods, "paper2014", "markesteijn", crop
        ),
        "paper_2016_minus_markesteijn": method_delta(
            images, methods, "paper2016", "markesteijn", crop
        ),
        "paper_2014_minus_corrected_two_pass": method_delta(
            images, methods, "paper2014", "corrected", crop
        ),
        "paper_2016_minus_corrected_two_pass": method_delta(
            images, methods, "paper2016", "corrected", crop
        ),
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
    parser.add_argument("--paper-2014", type=Path, required=True)
    parser.add_argument("--paper-2016", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--markesteijn", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--paper-2014-log", type=Path, required=True)
    parser.add_argument("--paper-2014-time", type=Path, required=True)
    parser.add_argument("--paper-2016-log", type=Path, required=True)
    parser.add_argument("--paper-2016-time", type=Path, required=True)
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
    if file_sha256(args.corrected) != source_digest(
            baseline, "mlri-xtrans-2pass-corrected"):
        raise ComparisonError("corrected TIFF differs from reviewed baseline")
    if file_sha256(args.markesteijn) != source_digest(
            baseline, "3-pass (Markesteijn)"):
        raise ComparisonError("Markesteijn TIFF differs from reviewed baseline")

    paths = {
        "paper2014": args.paper_2014,
        "paper2016": args.paper_2016,
        "corrected": args.corrected,
        "markesteijn": args.markesteijn,
    }
    common_metadata = validate_metadata({
        name: tiff_metadata(path) for name, path in paths.items()
    })
    images = {name: tifffile.memmap(path) for name, path in paths.items()}
    analysis = analyze_paper_arrays(images, args.crop)
    execution = {
        "paper2014": parse_paper_execution(
            args.paper_2014_log.read_text(encoding="utf-8"),
            args.paper_2014_time.read_text(encoding="utf-8"), METHOD_2014
        ),
        "paper2016": parse_paper_execution(
            args.paper_2016_log.read_text(encoding="utf-8"),
            args.paper_2016_time.read_text(encoding="utf-8"), METHOD_2016
        ),
        "corrected": baseline["execution"]["corrected"],
        "markesteijn": baseline["execution"]["markesteijn"],
    }
    with tifffile.TiffFile(args.paper_2014) as tiff:
        tag = tiff.pages[0].tags.get(34675)
        icc = bytes(tag.value) if tag else b""
    assets = save_assets(
        args.paper_2014, args.asset_dir, icc,
        method=METHOD_2014, filename_token="mlri-paper-2014"
    )
    assets += save_assets(
        args.paper_2016, args.asset_dir, icc,
        method=METHOD_2016, filename_token="mlri-paper-2016"
    )
    manifest = {
        **analysis,
        "assets": assets,
        "baseline_manifest_sha256": file_sha256(args.baseline_manifest),
        "comparison_order": [
            METHOD_2014, METHOD_2016,
            "mlri-xtrans-2pass-corrected", "3-pass (Markesteijn)"
        ],
        "crop": {
            "x": args.crop[0], "y": args.crop[1],
            "width": args.crop[2], "height": args.crop[3]
        },
        "execution": execution,
        "format": FORMAT,
        "generation": baseline["generation"],
        "identity": baseline["identity"],
        "metadata": common_metadata,
        "source_tiffs": [
            {
                "bytes": path.stat().st_size,
                "method": (
                    METHOD_2014 if name == "paper2014" else
                    METHOD_2016 if name == "paper2016" else
                    "mlri-xtrans-2pass-corrected" if name == "corrected" else
                    "3-pass (Markesteijn)"
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

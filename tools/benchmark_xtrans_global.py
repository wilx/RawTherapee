#!/usr/bin/env python3
"""Ground-truth benchmark for the hidden global X-Trans experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from tools.benchmark_xtrans_rafinazari import (
    CANONICAL_XTRANS,
    dependencies,
    metrics,
    mosaic,
    run_rawtherapee,
    synthetic_images,
    write_dng,
)


METHODS = {
    "global-a": "xtrans-global-spectral-rgb",
    "global-b": "xtrans-global-spectral-diff",
    "global-c": "xtrans-global-spectral-edge",
    "triangulated-rgb": "xtrans-triangulated-rgb",
    "triangulated-chroma": "xtrans-triangulated-chroma",
    "markesteijn": "3-pass (best)",
}


def write_profile(path: Path, method: str) -> None:
    path.write_text(
        "\n".join((
            "[Version]", "AppVersion=5.12", "Version=352", "",
            "[RAW X-Trans]", f"Method={method}", "CcSteps=0", "Border=0", "",
            "[Exposure]", "Auto=false", "HistogramMatching=false", "",
            "[Sharpening]", "Enabled=false", "", "[Capture Sharpening]",
            "Enabled=false", "", "[Impulse Denoising]", "Enabled=false", "",
            "[Directional Pyramid Denoising]", "Enabled=false", "",
        )),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rawtherapee-cli", type=Path,
        default=Path("build/dev/bin/rawtherapee-cli"),
    )
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--p", type=int, choices=(1, 2), default=1)
    parser.add_argument("--lambda-rgb", type=float, default=0.02)
    parser.add_argument("--lambda-green", type=float, default=0.01)
    parser.add_argument("--lambda-chroma", type=float, default=0.10)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    np, tifffile, structural_similarity = dependencies()
    cli = args.rawtherapee_cli.resolve()
    if not cli.is_file():
        raise SystemExit(f"RawTherapee CLI not found: {cli}")
    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="rt-xtrans-global-")
        work = Path(temporary.name)

    profiles = {name: work / f"{name}.pp3" for name in METHODS}
    for name, method in METHODS.items():
        write_profile(profiles[name], method)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "16",
        "RT_XTRANS_GLOBAL_ITERATIONS": str(args.iterations),
        "RT_XTRANS_GLOBAL_LAMBDA_RGB": str(args.lambda_rgb),
        "RT_XTRANS_GLOBAL_LAMBDA_G": str(args.lambda_green),
        "RT_XTRANS_GLOBAL_LAMBDA_C": str(args.lambda_chroma),
        "RT_XTRANS_GLOBAL_P": str(args.p),
        "XDG_CONFIG_HOME": str(work / "config"),
    })

    records = []
    for scene, reference in synthetic_images(np, args.size).items():
        source = work / f"{scene}.dng"
        write_dng(
            tifffile, source, mosaic(np, reference, CANONICAL_XTRANS),
            CANONICAL_XTRANS,
        )
        for name, profile in profiles.items():
            output = work / f"{scene}-{name}.tif"
            elapsed, rss = run_rawtherapee(
                cli, profile, source, output, environment,
            )
            with tifffile.TiffFile(output) as tiff:
                actual = tiff.asarray()
            psnr, ssim = metrics(
                np, structural_similarity, reference, actual,
            )
            records.append({
                "elapsed_seconds": elapsed,
                "max_rss_kb": rss,
                "method": name,
                "psnr_db": psnr if math.isfinite(psnr) else "infinity",
                "scene": scene,
                "ssim": ssim,
            })

    report = {
        "format": "rawtherapee-xtrans-global-ground-truth-v1",
        "parameters": {
            "iterations": args.iterations,
            "lambda_chroma": args.lambda_chroma,
            "lambda_green": args.lambda_green,
            "lambda_rgb": args.lambda_rgb,
            "p": args.p,
            "size": args.size,
        },
        "records": records,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.keep:
        (work / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"kept benchmark at {work}")
    if temporary is not None and args.keep:
        temporary.cleanup = lambda: None  # type: ignore[method-assign]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

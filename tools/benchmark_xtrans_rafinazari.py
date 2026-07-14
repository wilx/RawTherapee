#!/usr/bin/env python3
"""Developer benchmark for the experimental Rafinazari X-Trans demosaicer.

This script intentionally has optional development-only dependencies:

    python3 -m pip install numpy tifffile scikit-image

It creates neutral synthetic 6x6-CFA DNGs, processes each input with the
Rafinazari and three-pass Markesteijn methods, and reports PSNR, SSIM, elapsed
time, and peak child RSS.  The generated files live in a temporary directory
unless --work-dir is specified.

The --variant option also exercises the fixed and restricted weighting choices
discussed on dissertation p. 44. A build configured with WITH_BENCHMARK=ON is
required only for the ambiguous literal equation; the other development
variants are available in ordinary builds of the experimental demosaicer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


CANONICAL_XTRANS = (
    (1, 2, 1, 1, 0, 1),
    (0, 1, 0, 2, 1, 2),
    (1, 2, 1, 1, 0, 1),
    (1, 0, 1, 1, 2, 1),
    (2, 1, 2, 0, 1, 0),
    (1, 0, 1, 1, 2, 1),
)


def dependencies():
    try:
        import numpy as np
        import tifffile
    except ImportError as error:
        raise SystemExit(
            "Missing benchmark dependency. Install numpy and tifffile in a "
            "development environment.\n" + str(error)
        ) from error
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        def structural_similarity(first, second, channel_axis, data_range):
            del channel_axis
            c1 = (0.01 * data_range) ** 2
            c2 = (0.03 * data_range) ** 2
            values = []
            for channel in range(first.shape[2]):
                x = first[..., channel].astype(np.float64)
                y = second[..., channel].astype(np.float64)
                mean_x = x.mean()
                mean_y = y.mean()
                variance_x = x.var()
                variance_y = y.var()
                covariance = ((x - mean_x) * (y - mean_y)).mean()
                values.append(
                    ((2 * mean_x * mean_y + c1) * (2 * covariance + c2))
                    / ((mean_x * mean_x + mean_y * mean_y + c1)
                       * (variance_x + variance_y + c2))
                )
            return float(np.mean(values))
    return np, tifffile, structural_similarity


def synthetic_images(np, size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = x / max(1, size - 1)
    yn = y / max(1, size - 1)

    constant = np.empty((size, size, 3), dtype=np.float32)
    constant[...] = (0.18, 0.42, 0.73)

    gradient = np.stack((xn, yn, 0.5 * xn + 0.5 * yn), axis=-1)

    edges = np.zeros((size, size, 3), dtype=np.float32)
    edges[:, : size // 2, 0] = 1.0
    edges[: size // 2, size // 2 :, 1] = 1.0
    edges[size // 2 :, size // 2 :, 2] = 1.0

    impulse = np.full((size, size, 3), 0.1, dtype=np.float32)
    impulse[size // 2, size // 2] = 1.0

    sweep = np.empty((size, size, 3), dtype=np.float32)
    phase = 2.0 * math.pi * (x * (0.01 + 0.44 * yn) + 0.17 * y)
    sweep[..., 0] = 0.5 + 0.45 * np.sin(phase)
    sweep[..., 1] = 0.5 + 0.45 * np.sin(phase + 2.0 * math.pi / 3.0)
    sweep[..., 2] = 0.5 + 0.45 * np.sin(phase + 4.0 * math.pi / 3.0)

    black = np.zeros((size, size, 3), dtype=np.float32)
    return {
        "constant": constant,
        "gradient": gradient,
        "saturated-edges": edges,
        "impulse": impulse,
        "frequency-sweep": sweep,
        "black": black,
    }


def pattern_variants(np):
    canonical = np.asarray(CANONICAL_XTRANS, dtype=np.uint8)
    orientations = []
    for turns in range(4):
        rotated = np.rot90(canonical, turns)
        orientations.extend((rotated, np.fliplr(rotated)))
    variants = {}
    for oriented in orientations:
        for y in range(6):
            for x in range(6):
                shifted = np.roll(oriented, (y, x), axis=(0, 1))
                key = tuple(int(value) for value in shifted.flat)
                variants[key] = shifted
    return list(variants.values())


def mosaic(np, rgb, pattern):
    result = np.empty(rgb.shape[:2], dtype=np.uint16)
    pattern = np.asarray(pattern, dtype=np.uint8)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            result[y, x] = round(float(rgb[y, x, pattern[y % 6, x % 6]]) * 65535.0)
    return result


def write_dng(tifffile, path, data, pattern):
    # ColorMatrix1 is XYZ(D50) -> linear sRGB. It makes the synthetic camera
    # channels usable by RawTherapee's otherwise neutral color pipeline.
    color_matrix = (
        3_240_454, 1_000_000, -1_537_139, 1_000_000, -498_531, 1_000_000,
        -969_266, 1_000_000, 1_876_011, 1_000_000, 41_556, 1_000_000,
        55_643, 1_000_000, -204_026, 1_000_000, 1_057_225, 1_000_000,
    )
    tags = [
        (271, "s", 0, "Synthetic", False),
        (272, "s", 0, "Rafinazari X-Trans benchmark", False),
        (33421, "H", 2, (6, 6), False),
        (33422, "B", 36, tuple(value for row in pattern for value in row), False),
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 1, 0, 0), False),
        (50710, "B", 3, (0, 1, 2), False),
        (50711, "H", 1, 1, False),
        (50714, "H", 1, 0, False),
        (50717, "I", 1, 65535, False),
        (50721, "2i", 9, color_matrix, False),
        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), False),
    ]
    tifffile.imwrite(
        path,
        data,
        photometric=32803,
        metadata=None,
        extratags=tags,
    )


def write_profile(path, method):
    path.write_text(
        "\n".join(
            (
                "[RAW X-Trans]",
                f"Method={method}",
                "RafinazariSigma=2.32",
                "RafinazariNearRadius=7",
                "RafinazariMiddleRadius=7",
                "RafinazariFarRadius=7",
                "RafinazariEnergySigma=2.32",
                "RafinazariEnergyRadius=7",
                "RafinazariEnergyBoxRadius=2",
                "RafinazariEnergyFloor=0",
                "CcSteps=0",
                "Border=0",
                "",
                "[Exposure]",
                "Auto=false",
                "HistogramMatching=false",
                "",
                "[Sharpening]",
                "Enabled=false",
                "",
            )
        ),
        encoding="utf-8",
    )


def run_rawtherapee(cli, profile, source, output, environment):
    metrics_file = output.with_suffix(".time")
    command = [
        str(cli), "-q", "-Y", "-t", "-b16", "-p", str(profile),
        "-o", str(output), "-c", str(source),
    ]
    if Path("/usr/bin/time").exists():
        command = [
            "/usr/bin/time", "-f", "max_rss_kb=%M", "-o", str(metrics_file),
            *command,
        ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"RawTherapee failed for {source.name}:\n{completed.stdout}\n{completed.stderr}"
        )
    max_rss = None
    if metrics_file.exists():
        max_rss = int(metrics_file.read_text(encoding="utf-8").split("=", 1)[1])
    return elapsed, max_rss


def center_crop(np, image, height, width):
    y = max(0, (image.shape[0] - height) // 2)
    x = max(0, (image.shape[1] - width) // 2)
    return np.asarray(image[y : y + height, x : x + width, :3], dtype=np.float32)


def srgb_encode(np, linear):
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def metrics(np, structural_similarity, reference, actual):
    actual = actual / 65535.0 if actual.dtype.kind in "ui" else actual.astype(np.float32)
    height = min(reference.shape[0], actual.shape[0])
    width = min(reference.shape[1], actual.shape[1])
    reference = center_crop(np, srgb_encode(np, reference), height, width)
    actual = center_crop(np, actual, height, width)
    margin = min(32, max(0, min(height, width) // 8))
    if margin:
        reference = reference[margin:-margin, margin:-margin]
        actual = actual[margin:-margin, margin:-margin]
    error = np.mean((reference - actual) ** 2, dtype=np.float64)
    psnr = math.inf if error == 0 else 10.0 * math.log10(1.0 / error)
    ssim = structural_similarity(reference, actual, channel_axis=2, data_range=1.0)
    return psnr, float(ssim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawtherapee-cli", type=Path, default=Path("build/dev/bin/rawtherapee-cli"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--literal", action="store_true", help="benchmark literal weighting in a WITH_BENCHMARK build")
    parser.add_argument(
        "--variant",
        choices=("adaptive-all", "fixed-c2", "far-balanced", "q1213-only", "far-adaptive"),
        default="adaptive-all",
        help="Rafinazari weighting experiment (default: adaptive-all)",
    )
    parser.add_argument(
        "--validate-patterns",
        action="store_true",
        help="also smoke-test every unique rotation/reflection/phase of the 6x6 CFA",
    )
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
        temporary = tempfile.TemporaryDirectory(prefix="rt-rafinazari-")
        work = Path(temporary.name)

    profiles = {
        "rafinazari": work / "rafinazari.pp3",
        "markesteijn-3-pass": work / "markesteijn.pp3",
    }
    write_profile(profiles["rafinazari"], "rafinazari-adaptive")
    write_profile(profiles["markesteijn-3-pass"], "3-pass (best)")

    environment = os.environ.copy()
    environment["XDG_CONFIG_HOME"] = str(work / "config")
    selected_variant = "literal" if args.literal else args.variant
    environment["RT_RAFINAZARI_VARIANT"] = selected_variant

    report = []
    for name, reference in synthetic_images(np, args.size).items():
        source = work / f"{name}.dng"
        write_dng(tifffile, source, mosaic(np, reference, CANONICAL_XTRANS), CANONICAL_XTRANS)
        for method, profile in profiles.items():
            output = work / f"{name}-{method}.tif"
            elapsed, max_rss = run_rawtherapee(cli, profile, source, output, environment)
            image = tifffile.imread(output)
            psnr, ssim = metrics(np, structural_similarity, reference, image)
            reported_method = (
                f"rafinazari-{selected_variant}" if method == "rafinazari" else method
            )
            row = {
                "image": name,
                "method": reported_method,
                "psnr_db": round(psnr, 4),
                "ssim": round(ssim, 6),
                "seconds": round(elapsed, 4),
                "max_rss_kb": max_rss,
            }
            report.append(row)
            print(json.dumps(row, sort_keys=True))

    if args.validate_patterns:
        reference = synthetic_images(np, 72)["constant"]
        patterns = pattern_variants(np)
        for index, pattern in enumerate(patterns):
            source = work / f"pattern-{index:03d}.dng"
            output = work / f"pattern-{index:03d}.tif"
            write_dng(tifffile, source, mosaic(np, reference, pattern), pattern)
            run_rawtherapee(cli, profiles["rafinazari"], source, output, environment)
            image = tifffile.imread(output)
            if not np.isfinite(image).all():
                raise RuntimeError(f"non-finite output for CFA pattern {index}")
        print(f"validated CFA transforms: {len(patterns)}")

    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {work / 'report.json'}")

    if temporary and args.keep:
        kept = Path.cwd() / "rafinazari-benchmark-output"
        if kept.exists():
            raise SystemExit(f"refusing to overwrite existing directory: {kept}")
        import shutil
        shutil.copytree(work, kept)
        print(f"kept output: {kept}")


if __name__ == "__main__":
    main()

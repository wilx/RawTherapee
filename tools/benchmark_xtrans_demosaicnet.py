#!/usr/bin/env python3
"""Developer quality/performance benchmark for hidden X-Trans methods.

Optional dependencies::

    python3 -m pip install numpy tifffile pillow scikit-image

The script commits no inputs or outputs. Synthetic and external RGB images are
mosaicked into neutral X-Trans DNGs; real RAFs are exported for timing and
side-by-side crop review. A loud fallback marker is always a failed row even
when rawtherapee-cli itself exits successfully.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from typing import Iterable


CANONICAL_XTRANS = (
    (1, 2, 1, 1, 0, 1),
    (0, 1, 0, 2, 1, 2),
    (1, 2, 1, 1, 0, 1),
    (1, 0, 1, 1, 2, 1),
    (2, 1, 2, 0, 1, 0),
    (1, 0, 1, 1, 2, 1),
)
METHODS = {
    "markesteijn": "3-pass (best)",
}
FALLBACK_MARKER = "falling back to 3-pass (Markesteijn)"
COMPLETION_RE = re.compile(
    r"DemosaicNet X-Trans completed:.*?artifact=([0-9a-f]{64}).*?workers=(\d+).*?"
    r"workspace_per_worker=(\d+).*?workspace_total=(\d+).*?elapsed_us=(\d+)"
)
XVEON_COMPLETION_RE = re.compile(
    r"X-veon X-Trans completed:.*?artifact=([0-9a-f]{64}).*?(?:ort|runtime)=(\S+).*?provider=(\S+).*?"
    r"tiles=(\d+).*?working_buffer_estimate=(\d+).*?elapsed_us=(\d+)"
)
PACKED_COMPLETION_RE = re.compile(
    r"PackedXTransNet X-Trans completed:.*?artifact=([0-9a-f]{64}).*?runtime=(\S+).*?provider=(\S+).*?"
    r"precision=(\S+).*?compile_source=(\S+).*?margin=(\d+).*?stride=(\d+).*?tiles=(\d+).*?"
    r"working_buffer_estimate=(\d+).*?elapsed_us=(\d+)"
)
MLRI_COMPLETION_RE = re.compile(
    r"MLRI X-Trans completed:.*?passes=2 sigma=2,1 epsilon=0.01.*?"
    r"core=(\d+)x(\d+) halo=(\d+).*?tiles=(\d+).*?workers=(\d+).*?"
    r"workspace_per_worker_estimate=(\d+).*?elapsed_us=(\d+)"
)
MLRI_PAPER_CORE_COMPLETION_RE = re.compile(
    r"MLRI X-Trans completed:.*?passes=1 sigma=2 epsilon=0.01 "
    r"coefficient_average=(uniform|residual-weighted) final=direct.*?"
    r"core=(\d+)x(\d+) halo=(\d+).*?tiles=(\d+).*?workers=(\d+).*?"
    r"workspace_per_worker_estimate=(\d+).*?elapsed_us=(\d+)"
)


def dependencies():
    try:
        import numpy as np
        import tifffile
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Install benchmark dependencies: numpy tifffile pillow scikit-image\n"
            + str(error)
        ) from error
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None
    return np, tifffile, Image, structural_similarity


def synthetic_images(np, size: int):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = x / max(1, size - 1)
    yn = y / max(1, size - 1)
    constant = np.empty((size, size, 3), np.float32)
    constant[...] = (0.18, 0.42, 0.73)
    gradient = np.stack((xn, yn, 0.25 + 0.5 * xn), axis=-1)
    impulses = np.full((size, size, 3), 0.08, np.float32)
    impulses[size // 2, size // 2] = (1, 1, 1)
    sweep = np.empty_like(constant)
    phase = 2 * math.pi * (x * (0.01 + 0.44 * yn) + 0.17 * y)
    for channel in range(3):
        sweep[..., channel] = 0.5 + 0.45 * np.sin(phase + channel * 2 * math.pi / 3)
    edges = np.zeros_like(constant)
    edges[:, : size // 2, 0] = 1
    edges[: size // 2, size // 2 :, 1] = 1
    edges[size // 2 :, size // 2 :, 2] = 1
    asymmetric = np.stack(
        (xn, np.mod(3 * xn + 5 * yn, 1), np.where(x > 2 * y, 0.9, 0.1)), axis=-1
    ).astype(np.float32)
    one_pixel_lines = np.full_like(constant, 0.08)
    line_mask = (np.mod(x, 11) == 0) | (np.mod(y, 13) == 0)
    one_pixel_lines[line_mask] = 0.92
    diagonal_lines = np.full_like(constant, 0.1)
    diagonal_lines[(np.mod(x - y, 11) == 0) | (np.mod(x + y, 17) == 0)] = 0.9
    radius = np.sqrt((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
    circles_scalar = 0.5 + 0.45 * np.sign(np.sin(radius * math.pi / 2.5))
    concentric_circles = np.repeat(circles_scalar[..., None], 3, axis=2).astype(np.float32)
    centered_x = x - (size - 1) / 2
    centered_y = y - (size - 1) / 2
    zone_scalar = 0.5 + 0.45 * np.sin(
        math.pi * (centered_x * centered_x + centered_y * centered_y) / max(1, size)
    )
    zone_plate = np.repeat(zone_scalar[..., None], 3, axis=2).astype(np.float32)
    grating_scalar = 0.5 + 0.45 * np.sin(2 * math.pi * (0.37 * x + 0.29 * y))
    sinusoidal_grating = np.repeat(grating_scalar[..., None], 3, axis=2).astype(np.float32)
    checker_scalar = np.where(np.mod(x + y, 2) == 0, 0.95, 0.05)
    fine_checkerboard = np.repeat(checker_scalar[..., None], 3, axis=2).astype(np.float32)
    red_green = np.zeros_like(constant)
    red_green[:, : size // 2, 0] = 0.95
    red_green[:, size // 2 :, 1] = 0.95
    blue_green = np.zeros_like(constant)
    blue_green[:, : size // 2, 2] = 0.95
    blue_green[:, size // 2 :, 1] = 0.95
    nyquist_chromatic = np.zeros_like(constant)
    nyquist_chromatic[..., 0] = np.where(np.mod(x + y, 2) == 0, 0.95, 0.05)
    nyquist_chromatic[..., 2] = np.where(np.mod(x + y, 2) == 0, 0.05, 0.95)
    nyquist_achromatic_scalar = 0.5 + 0.45 * np.sin(2 * math.pi * 0.45 * x)
    nyquist_achromatic = np.repeat(
        nyquist_achromatic_scalar[..., None], 3, axis=2
    ).astype(np.float32)
    text_like = np.full_like(constant, 0.05)
    glyph = ((np.mod(x, 16) == 2) | (np.mod(x, 16) == 9) |
             (np.mod(y, 18) == 3) | ((np.mod(y, 18) == 10) & (np.mod(x, 16) < 10)))
    text_like[glyph] = 0.95
    colored_text_like = np.full_like(constant, 0.05)
    colored_text_like[..., 0][glyph] = 0.95
    colored_text_like[..., 1][np.roll(glyph, 5, axis=1)] = 0.9
    colored_text_like[..., 2][np.roll(glyph, 7, axis=0)] = 0.85
    return {
        "constant": constant,
        "gradient": gradient,
        "impulses": impulses,
        "frequency-sweep": sweep,
        "saturated-edges": edges,
        "black": np.zeros_like(constant),
        "asymmetric-orientation": asymmetric,
        "one-pixel-lines": one_pixel_lines,
        "diagonal-lines": diagonal_lines,
        "concentric-circles": concentric_circles,
        "zone-plate": zone_plate,
        "sinusoidal-grating": sinusoidal_grating,
        "fine-checkerboard": fine_checkerboard,
        "red-green-transition": red_green,
        "blue-green-transition": blue_green,
        "near-nyquist-achromatic": nyquist_achromatic,
        "near-nyquist-chromatic": nyquist_chromatic,
        "black-white-text-like": text_like,
        "colored-text-like": colored_text_like,
    }


def mosaic(np, rgb, pattern=CANONICAL_XTRANS):
    pattern = np.asarray(pattern, dtype=np.uint8)
    rows, columns = np.indices(rgb.shape[:2])
    channels = pattern[rows % 6, columns % 6]
    return np.rint(
        np.clip(np.take_along_axis(rgb, channels[..., None], axis=2)[..., 0], 0, 1)
        * 65535
    ).astype(np.uint16)


def write_dng(tifffile, path: Path, data, pattern=CANONICAL_XTRANS):
    # XYZ(D50) -> linear sRGB, represented as DNG SRATIONAL values.
    matrix = (
        3_240_454, 1_000_000, -1_537_139, 1_000_000, -498_531, 1_000_000,
        -969_266, 1_000_000, 1_876_011, 1_000_000, 41_556, 1_000_000,
        55_643, 1_000_000, -204_026, 1_000_000, 1_057_225, 1_000_000,
    )
    tags = [
        (271, "s", 0, "Synthetic", False),
        (272, "s", 0, "RawTherapee X-Trans demosaic benchmark", False),
        (33421, "H", 2, (6, 6), False),
        (33422, "B", 36, tuple(value for row in pattern for value in row), False),
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 1, 0, 0), False),
        (50710, "B", 3, (0, 1, 2), False),
        (50711, "H", 1, 1, False),
        (50714, "H", 1, 0, False),
        (50717, "I", 1, 65535, False),
        (50721, "2i", 9, matrix, False),
        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), False),
    ]
    tifffile.imwrite(path, data, photometric=32803, metadata=None, extratags=tags)


def write_profile(path: Path, method: str):
    path.write_text(
        "\n".join(
            (
                "[Version]", "AppVersion=5.12", "Version=352", "",
                "[RAW X-Trans]", f"Method={method}", "CcSteps=0", "Border=0", "",
                "[Exposure]", "Auto=false", "HistogramMatching=false", "",
                "[Sharpening]", "Enabled=false", "",
                "[Capture Sharpening]", "Enabled=false", "",
                "[Impulse Denoising]", "Enabled=false", "",
                "[Directional Pyramid Denoising]", "Enabled=false", "",
            )
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RunResult:
    seconds: float
    max_rss_kb: int | None
    fallback: bool
    completion: dict | None
    stderr: str


def run_cli(cli: Path, profile: Path, source: Path, output: Path, environment) -> RunResult:
    time_file = output.parent / f"{output.name}.{time.time_ns()}.time"
    command = [str(cli), "-q", "-Y", "-t", "-b16", "-p", str(profile), "-o", str(output), "-c", str(source)]
    if Path("/usr/bin/time").is_file():
        command = ["/usr/bin/time", "-f", "max_rss_kb=%M", "-o", str(time_file), *command]
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
    seconds = time.perf_counter() - started
    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"rawtherapee-cli failed ({completed.returncode}):\n{combined}")
    match = COMPLETION_RE.search(combined)
    completion = None
    if match:
        completion = {
            "artifact_sha256": match.group(1),
            "active_workers": int(match.group(2)),
            "workspace_per_worker_bytes": int(match.group(3)),
            "workspace_total_bytes": int(match.group(4)),
            "engine_elapsed_us": int(match.group(5)),
        }
    xveon_match = XVEON_COMPLETION_RE.search(combined)
    if xveon_match:
        completion = {
            "artifact_sha256": xveon_match.group(1),
            "onnxruntime_version": xveon_match.group(2),
            "provider": xveon_match.group(3),
            "tiles": int(xveon_match.group(4)),
            "working_buffer_estimate_bytes": int(xveon_match.group(5)),
            "engine_elapsed_us": int(xveon_match.group(6)),
        }
    packed_match = PACKED_COMPLETION_RE.search(combined)
    if packed_match:
        completion = {
            "artifact_sha256": packed_match.group(1),
            "runtime_version": packed_match.group(2),
            "provider": packed_match.group(3),
            "precision": packed_match.group(4),
            "compile_source": packed_match.group(5),
            "margin": int(packed_match.group(6)),
            "stride": int(packed_match.group(7)),
            "tiles": int(packed_match.group(8)),
            "working_buffer_estimate_bytes": int(packed_match.group(9)),
            "engine_elapsed_us": int(packed_match.group(10)),
        }
    mlri_match = MLRI_COMPLETION_RE.search(combined)
    if mlri_match:
        completion = {
            "core_width": int(mlri_match.group(1)),
            "core_height": int(mlri_match.group(2)),
            "halo": int(mlri_match.group(3)),
            "tiles": int(mlri_match.group(4)),
            "active_workers": int(mlri_match.group(5)),
            "workspace_per_worker_estimate_bytes": int(mlri_match.group(6)),
            "engine_elapsed_us": int(mlri_match.group(7)),
        }
    mlri_paper_match = MLRI_PAPER_CORE_COMPLETION_RE.search(combined)
    if mlri_paper_match:
        completion = {
            "coefficient_average": mlri_paper_match.group(1),
            "core_width": int(mlri_paper_match.group(2)),
            "core_height": int(mlri_paper_match.group(3)),
            "halo": int(mlri_paper_match.group(4)),
            "tiles": int(mlri_paper_match.group(5)),
            "active_workers": int(mlri_paper_match.group(6)),
            "workspace_per_worker_estimate_bytes": int(mlri_paper_match.group(7)),
            "engine_elapsed_us": int(mlri_paper_match.group(8)),
            "final_reconstruction": "direct",
            "passes": 1,
        }
    rss = None
    if time_file.is_file():
        rss = int(time_file.read_text(encoding="utf-8").strip().split("=", 1)[1])
    return RunResult(seconds, rss, FALLBACK_MARKER in combined, completion, completed.stderr)


def srgb_encode(np, linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055)


def quality_metrics(np, structural_similarity, reference, actual, pattern):
    actual = actual[..., :3].astype(np.float32) / 65535.0
    height = min(reference.shape[0], actual.shape[0])
    width = min(reference.shape[1], actual.shape[1])
    reference = srgb_encode(np, reference[:height, :width])
    actual = actual[:height, :width]
    margin = min(24, min(height, width) // 8)
    if margin:
        reference = reference[margin:-margin, margin:-margin]
        actual = actual[margin:-margin, margin:-margin]
    channel_mse = np.mean((reference - actual) ** 2, axis=(0, 1), dtype=np.float64)
    channel_psnr = [math.inf if value == 0 else 10 * math.log10(1 / value) for value in channel_mse]
    cpsnr = float(sum(channel_psnr) / 3)
    mse = float(np.mean((reference - actual) ** 2, dtype=np.float64))
    psnr = math.inf if mse == 0 else 10 * math.log10(1 / mse)
    ssim = None
    if structural_similarity is not None:
        ssim = float(structural_similarity(reference, actual, channel_axis=2, data_range=1.0))
    cfa = np.asarray(pattern, dtype=np.uint8)
    rows, columns = np.indices(reference.shape[:2])
    observed = cfa[(rows + margin) % 6, (columns + margin) % 6]
    error = np.take_along_axis(actual - reference, observed[..., None], axis=2)[..., 0]
    return {
        "cpsnr_db": cpsnr if math.isfinite(cpsnr) else None,
        "psnr_db": psnr if math.isfinite(psnr) else None,
        "ssim": ssim,
        "observed_sample_rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
    }


def write_difference_assets(np, tifffile, Image, work: Path, scene: str,
                            method: str, reference, actual) -> dict[str, dict[str, object]]:
    """Write quantitative float error and an explicitly amplified visual map."""
    rendered = np.asarray(actual[..., :3], dtype=np.float32) / 65535.0
    height = min(reference.shape[0], rendered.shape[0])
    width = min(reference.shape[1], rendered.shape[1])
    expected = srgb_encode(np, reference[:height, :width])
    error = np.abs(rendered[:height, :width] - expected).astype(np.float32)
    float_path = work / f"{scene}-{method}-absolute-error.f32.tif"
    visual_path = work / f"{scene}-{method}-absolute-error-8x.png"
    tifffile.imwrite(float_path, error, photometric="rgb", metadata=None)
    visible = np.rint(np.clip(error * 8.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(visible, mode="RGB").save(visual_path, compress_level=9)
    return {
        "absolute_error_float32": {
            "filename": float_path.name,
            "sha256": file_sha256(float_path),
        },
        "absolute_error_visual_8x": {
            "filename": visual_path.name,
            "sha256": file_sha256(visual_path),
        },
    }


def parse_crop(text: str):
    try:
        name, values = text.split(":", 1)
        x, y, width, height = map(int, values.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be NAME:X,Y,W,H") from error
    if min(x, y) < 0 or min(width, height) <= 0:
        raise argparse.ArgumentTypeError("crop coordinates and dimensions are invalid")
    return name, (x, y, width, height)


def external_images(np, Image, directory: Path) -> Iterable[tuple[str, object, dict]]:
    paths = [directory / f"{index:06d}.png" for index in range(1, 11)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "ground-truth directory does not contain the pinned Caffe scenes: "
            + ", ".join(missing)
        )
    for path in paths:
        with Image.open(path) as image:
            yield (
                f"upstream-{path.stem}",
                np.asarray(image.convert("RGB"), np.float32) / 255.0,
                {
                    "kind": "upstream-caffe-test-image",
                    "path_within_upstream": f"data/test_images/{path.name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "upstream_revision": "ba4e77c50e46619b85cb871d740c6a39f12c0f25",
                },
            )


def assignment(text: str):
    if "=" not in text:
        raise argparse.ArgumentTypeError("value must be NAME=TEXT")
    return tuple(text.split("=", 1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_metadata(path: Path) -> dict[str, str]:
    result = {}
    if not Path("/usr/bin/exiv2").is_file():
        return result
    for label, key in (
        ("make", "Exif.Image.Make"),
        ("model", "Exif.Image.Model"),
        ("iso", "Exif.Photo.ISOSpeedRatings"),
    ):
        completed = subprocess.run(
            ["/usr/bin/exiv2", "-g", key, "-Pv", str(path)],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            result[label] = completed.stdout.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawtherapee-cli", type=Path, default=Path("build/dev/rtgui/rawtherapee-cli"))
    parser.add_argument("--model", type=Path,
                        help="external Gharbi RTNN; enables the linear and gamma methods")
    parser.add_argument("--mlri", action="store_true",
                        help="enable the hidden two-pass algorithmic MLRI method")
    parser.add_argument("--mlri-corrected", action="store_true",
                        help="enable MLRI with corrected blue diagonal guides")
    parser.add_argument("--mlri-paper-core-2014", action="store_true",
                        help="enable one-pass MLRI with uniform 2014 coefficient averaging")
    parser.add_argument("--mlri-paper-core-2016", action="store_true",
                        help="enable one-pass MLRI with weighted 2016 coefficient averaging")
    parser.add_argument("--xveon-model", type=Path,
                        help="external pinned xtrans.onnx; enables the hidden X-veon method")
    parser.add_argument("--packed-model", type=Path,
                        help="external converted PackedXTransNet ONNX; enables its hidden method")
    parser.add_argument("--packed-backend", choices=("onnxruntime-cpu", "migraphx"),
                        default="onnxruntime-cpu")
    parser.add_argument("--packed-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--raf", type=Path, action="append", default=[])
    parser.add_argument("--raf-source", type=assignment, action="append", default=[],
                        help="record provenance as RAF_NAME=URL")
    parser.add_argument("--crop", type=parse_crop, action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--real-runs", type=int, default=3,
                        help="measured real-RAF runs after one warm-up (default: 3)")
    args = parser.parse_args()
    np, tifffile, Image, structural_similarity = dependencies()
    cli = args.rawtherapee_cli.resolve()
    model = args.model.resolve() if args.model else None
    xveon_model = args.xveon_model.resolve() if args.xveon_model else None
    packed_model = args.packed_model.resolve() if args.packed_model else None
    if (not cli.is_file() or (model and not model.is_file()) or
            (xveon_model and not xveon_model.is_file()) or
            (packed_model and not packed_model.is_file())):
        raise SystemExit("rawtherapee-cli or RTNN model does not exist")
    methods = dict(METHODS)
    if model:
        methods["linear"] = "demosaicnet-xtrans-linear"
        methods["gamma22"] = "demosaicnet-xtrans-gamma22"
    if args.mlri:
        methods["mlri"] = "mlri-xtrans-2pass"
    if args.mlri_corrected:
        methods["mlri-corrected"] = "mlri-xtrans-2pass-corrected"
    if args.mlri_paper_core_2014:
        methods["mlri-paper-2014"] = "mlri-xtrans-paper-core-2014"
    if args.mlri_paper_core_2016:
        methods["mlri-paper-2016"] = "mlri-xtrans-paper-core-2016"
    if len(methods) == 1 and not xveon_model and not packed_model:
        raise SystemExit("select at least one experimental method")
    if xveon_model:
        methods["xveon"] = "xveon-xtrans-onnx"
    if packed_model:
        methods["packedxtransnet"] = "packedxtransnet-onnx"
    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="rt-demosaicnet-xtrans-")
        work = Path(temporary.name)
    profiles = {}
    for key, method in methods.items():
        profiles[key] = work / f"{key}.pp3"
        write_profile(profiles[key], method)
    environment = os.environ.copy()
    if model:
        environment["RT_DEMOSAICNET_XTRANS_MODEL"] = str(model)
    if xveon_model:
        environment["RT_XVEON_XTRANS_MODEL"] = str(xveon_model)
    if packed_model:
        environment["RT_PACKED_XTRANS_MODEL"] = str(packed_model)
        environment["RT_PACKED_XTRANS_BACKEND"] = args.packed_backend
        environment["RT_PACKED_XTRANS_PRECISION"] = args.packed_precision
    environment["XDG_CONFIG_HOME"] = str(work / "config")
    report = {"synthetic_and_ground_truth": [], "real_raf": [], "gate": {}}

    scenes = [] if args.skip_synthetic else [
        (name, image, {"kind": "generated-analytical"})
        for name, image in synthetic_images(np, args.size).items()
    ]
    if args.ground_truth_dir:
        scenes.extend(external_images(np, Image, args.ground_truth_dir.resolve()))
    for scene_name, reference, source_identity in scenes:
        source = work / f"{scene_name}.dng"
        write_dng(tifffile, source, mosaic(np, reference))
        for method, profile in profiles.items():
            output = work / f"{scene_name}-{method}.tif"
            execution = run_cli(cli, profile, source, output, environment)
            image = tifffile.imread(output)
            row = {
                "scene": scene_name,
                "method": method,
                "source": source_identity,
                **quality_metrics(np, structural_similarity, reference, image, CANONICAL_XTRANS),
                "difference_assets": write_difference_assets(
                    np, tifffile, Image, work, scene_name, method, reference, image
                ),
                "seconds": execution.seconds,
                "max_rss_kb": execution.max_rss_kb,
                "fallback": execution.fallback,
                "completion": execution.completion,
            }
            report["synthetic_and_ground_truth"].append(row)
            print(json.dumps(row, sort_keys=True))

    crops = dict(args.crop)
    raf_sources = dict(args.raf_source)
    for raf in args.raf:
        raf = raf.resolve()
        raf_identity = {
            "camera": camera_metadata(raf),
            "sha256": file_sha256(raf),
            "source_url": raf_sources.get(raf.name) or raf_sources.get(raf.stem),
        }
        outputs = {}
        rows = []
        for method, profile in profiles.items():
            output = work / f"{raf.stem}-{method}.tif"
            # Each CLI invocation is a new process; the warm-up primarily
            # removes cold filesystem effects and matches the agreed gate.
            run_cli(cli, profile, raf, output, environment)
            executions = [
                run_cli(cli, profile, raf, output, environment)
                for _ in range(max(1, args.real_runs))
            ]
            execution = executions[-1]
            outputs[method] = tifffile.imread(output)[..., :3]
            row = {
                "file": raf.name,
                "method": method,
                "source": raf_identity,
                "seconds": statistics.median(item.seconds for item in executions),
                "seconds_runs": [item.seconds for item in executions],
                "max_rss_kb": max(
                    (item.max_rss_kb for item in executions if item.max_rss_kb is not None),
                    default=None,
                ),
                "fallback": any(item.fallback for item in executions),
                "completion": execution.completion,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        crop = crops.get(raf.name) or crops.get(raf.stem)
        if crop:
            x, y, width, height = crop
            panels = [outputs[key][y:y + height, x:x + width] for key in methods]
            tifffile.imwrite(work / f"{raf.stem}-comparison-crop.tif", np.concatenate(panels, axis=1))
        report["real_raf"].extend(rows)

    neural_rows = [row for row in report["synthetic_and_ground_truth"] if row["method"] != "markesteijn"]
    report["gate"]["no_fallback"] = not any(row["fallback"] for row in neural_rows)
    upstream = [row for row in report["synthetic_and_ground_truth"] if row["scene"].startswith("upstream-")]
    for method in ("linear", "gamma22"):
        values = [row["cpsnr_db"] for row in upstream if row["method"] == method]
        report["gate"][f"{method}_upstream_mean_cpsnr_db"] = sum(values) / len(values) if values else None
    if len(upstream) == 10 * len(methods):
        means = {
            method: statistics.mean(row["cpsnr_db"] for row in upstream if row["method"] == method)
            for method in methods
        }
        by_scene = {
            name: {row["method"]: row for row in upstream if row["scene"] == name}
            for name in {row["scene"] for row in upstream}
        }
        if "linear" in methods and "gamma22" in methods:
            selected = "gamma22" if means["gamma22"] >= means["linear"] - 0.1 else "linear"
            report["gate"]["selected_wrapper"] = selected
            report["gate"]["selected_within_0_5_db_of_markesteijn"] = (
                means[selected] >= means["markesteijn"] - 0.5
            )
            report["gate"]["selected_upstream_wins"] = sum(
                rows[selected]["cpsnr_db"] > rows["markesteijn"]["cpsnr_db"]
                for rows in by_scene.values()
            )
        if "xveon" in methods:
            report["gate"]["xveon_upstream_mean_cpsnr_db"] = means["xveon"]
            report["gate"]["xveon_upstream_wins"] = sum(
                rows["xveon"]["cpsnr_db"] > rows["markesteijn"]["cpsnr_db"]
                for rows in by_scene.values()
            )
        if "packedxtransnet" in methods:
            report["gate"]["packedxtransnet_upstream_mean_cpsnr_db"] = means["packedxtransnet"]
            report["gate"]["packedxtransnet_upstream_wins"] = sum(
                rows["packedxtransnet"]["cpsnr_db"] > rows["markesteijn"]["cpsnr_db"]
                for rows in by_scene.values()
            )
        if "mlri" in methods:
            report["gate"]["mlri_upstream_mean_cpsnr_db"] = means["mlri"]
            report["gate"]["mlri_upstream_wins"] = sum(
                rows["mlri"]["cpsnr_db"] > rows["markesteijn"]["cpsnr_db"]
                for rows in by_scene.values()
            )
        if "mlri-corrected" in methods:
            report["gate"]["mlri_corrected_upstream_mean_cpsnr_db"] = means["mlri-corrected"]
            report["gate"]["mlri_corrected_upstream_wins"] = sum(
                rows["mlri-corrected"]["cpsnr_db"] > rows["markesteijn"]["cpsnr_db"]
                for rows in by_scene.values()
            )
    for filename in {row["file"] for row in report["real_raf"]}:
        rows = {row["method"]: row for row in report["real_raf"] if row["file"] == filename}
        if set(rows) == set(methods):
            for method in tuple(name for name in methods if name != "markesteijn"):
                rows[method]["time_ratio_to_markesteijn"] = (
                    rows[method]["seconds"] / rows["markesteijn"]["seconds"]
                )
                if rows[method]["max_rss_kb"] is not None and rows["markesteijn"]["max_rss_kb"] is not None:
                    rows[method]["rss_delta_to_markesteijn_kb"] = (
                        rows[method]["max_rss_kb"] - rows["markesteijn"]["max_rss_kb"]
                    )
    report["gate"]["phase10_ready"] = False
    report["gate"]["note"] = "Final Phase 10 decision requires the complete ten-image and six-camera matrix."
    report_path = args.report or work / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report: {report_path}")
    return 1 if not report["gate"]["no_fallback"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

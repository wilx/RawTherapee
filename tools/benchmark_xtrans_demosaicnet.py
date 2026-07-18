#!/usr/bin/env python3
"""Developer quality/performance gate for hidden Phase 9 X-Trans methods.

Optional dependencies::

    python3 -m pip install numpy tifffile pillow scikit-image

The script commits no inputs or outputs. Synthetic and external RGB images are
mosaicked into neutral X-Trans DNGs; real RAFs are exported for timing and
side-by-side crop review. A neural fallback marker is always a failed row even
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
    "linear": "demosaicnet-xtrans-linear",
    "gamma22": "demosaicnet-xtrans-gamma22",
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
    return {
        "constant": constant,
        "gradient": gradient,
        "impulses": impulses,
        "frequency-sweep": sweep,
        "saturated-edges": edges,
        "black": np.zeros_like(constant),
        "asymmetric-orientation": asymmetric,
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
        (272, "s", 0, "DemosaicNet X-Trans Phase 9", False),
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
    parser.add_argument("--model", type=Path, required=True)
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
    model = args.model.resolve()
    xveon_model = args.xveon_model.resolve() if args.xveon_model else None
    packed_model = args.packed_model.resolve() if args.packed_model else None
    if (not cli.is_file() or not model.is_file() or
            (xveon_model and not xveon_model.is_file()) or
            (packed_model and not packed_model.is_file())):
        raise SystemExit("rawtherapee-cli or RTNN model does not exist")
    methods = dict(METHODS)
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
        selected = "gamma22" if means["gamma22"] >= means["linear"] - 0.1 else "linear"
        report["gate"]["selected_wrapper"] = selected
        report["gate"]["selected_within_0_5_db_of_markesteijn"] = (
            means[selected] >= means["markesteijn"] - 0.5
        )
        by_scene = {
            name: {row["method"]: row for row in upstream if row["scene"] == name}
            for name in {row["scene"] for row in upstream}
        }
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

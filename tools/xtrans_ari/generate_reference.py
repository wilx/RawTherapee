#!/usr/bin/env python3
"""Generate and verify deterministic Bayer ARI reference fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from tools.xtrans_ari.bayer_reference import green_ari, mosaic_bayer


FORMAT = "rawtherapee-ari-bayer-reference-v1"
ARCHIVE_BYTES = 720_117
ARCHIVE_SHA256 = "eecb94b92a2f1f4fb1bbdf41de93982697fbba09baa80cf8227104203dbd9272"
PAPER_2017 = {
    "bytes": 1_243_059,
    "doi": "10.3390/s17122787",
    "sha256": "32c11929c42a7550b57b50c74b0cecbfb12f0f985af77dcfa60acfb5bedd2a16",
}
PAPER_2015 = {
    "bytes": 2_568_311,
    "doi": "10.1109/ICIP.2015.7351687",
    "sha256": "f763278c460ae0d40fdd8829513c44462e5748c4cf34d94de094ff3741a7cde5",
}
REFERENCE_FILES = {
    "blue_interpolation.m": "16e7f0f82b2e0734a8c093054a32044f67fb87e67eae455ebccc064da7bf3f07",
    "boxfilter.m": "002b1925acad4e472772b08f1eea34d25a1ca0d1995619c11026a7d06dfd4844",
    "clip.m": "c2fe7a4a7153d0faf84e04e94e74e5bf91e1b8ae6349372ef39f06a50d97cbe4",
    "demosaic_ARI.m": "71b5980873db641bb12790f25a7d1b1c5f1ab7e7f3a9a33f88ed4def645d7a5c",
    "green_interpolation.m": "aa670e39bffafd83c5015fcde535b7274ddea0f466a8981e6077897c5ca84b2c",
    "guidedfilter.m": "c08847c29eb417f5ff539d33543503e9cbce4ca08ecbc428631d83835c7ca062",
    "guidedfilter_MLRI.m": "77dccdf4da3cca389ae5661f09f484ce1b3d6b5a7f00e593e7c2f4212a729caf",
    "guidedfilter_MLRI_diagonal.m": "d2630fd0952f2059396570274f3c2db8ae97853ce04fb5279119ea7b1e762366",
    "guidedfilter_diagonal.m": "f084e0e5e5979739e37c32aec01488efe434d635b3943f4d5cada9270a0edb97",
    "mosaic_bayer.m": "271e4de3f2a6f8ba07dd0b99de9f237c182cec128ab3f650322143ab569556f1",
    "readme.txt": "ddd1265692e490fff4dcd524a9840d2f21f5fa61a0c5989ce76045ad895c83d5",
    "red_blue_interpolation_first.m": "debcb308661ce5cc6b10db0188a821ab3533ae69abdd3c6d9c04f6f2a62a2d2c",
    "red_blue_interpolation_second.m": "3d55df2878135b75d3ef5312a3870b19ef74b5c4ba79a2739c403e76e946aeab",
}
OCTAVE_COMPATIBILITY_FILES = (
    "guidedfilter.m",
    "guidedfilter_MLRI.m",
    "guidedfilter_diagonal.m",
    "guidedfilter_MLRI_diagonal.m",
)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:size, :size]
    return x / (size - 1), y / (size - 1)


def scenes(size: int = 48) -> dict[str, np.ndarray]:
    x, y = _grid(size)
    result = {}
    for name, color in (
        ("constant-gray", (0.42, 0.42, 0.42)),
        ("constant-red", (1.0, 0.0, 0.0)),
        ("constant-green", (0.0, 1.0, 0.0)),
        ("constant-blue", (0.0, 0.0, 1.0)),
    ):
        result[name] = np.broadcast_to(
            np.asarray(color)[:, None, None], (3, size, size)
        ).copy()
    result["horizontal-gradient"] = np.stack((x, 0.1 + 0.8 * x, 1.0 - x))
    result["vertical-gradient"] = np.stack((y, 0.1 + 0.8 * y, 1.0 - y))
    horizontal = np.where(y < 0.5, 0.12, 0.88)
    vertical = np.where(x < 0.5, 0.12, 0.88)
    diagonal = np.where(x + y < 1.0, 0.12, 0.88)
    result["horizontal-edge"] = np.stack((horizontal, horizontal, horizontal))
    result["vertical-edge"] = np.stack((vertical, vertical, vertical))
    result["diagonal-edge"] = np.stack((diagonal, diagonal, diagonal))
    impulse = np.full((3, size, size), 0.05)
    impulse[:, size // 2, size // 2] = 1.0
    result["isolated-impulse"] = impulse
    point = np.full((3, size, size), 0.04)
    point[:, size // 2, size // 2] = (1.0, 0.0, 0.0)
    result["saturated-red-point"] = point
    mono = 0.5 + 0.42 * np.sin(2.0 * np.pi * (0.29 * np.arange(size)[None, :] + 0.23 * np.arange(size)[:, None]))
    result["periodic-monochrome"] = np.stack((mono, mono, mono))
    result["periodic-chromatic"] = np.stack((
        0.5 + 0.42 * np.sin(2.0 * np.pi * (0.31 * x + 0.19 * y) * size),
        0.5 + 0.42 * np.sin(2.0 * np.pi * (0.27 * x - 0.21 * y) * size + 1.1),
        0.5 + 0.42 * np.sin(2.0 * np.pi * (0.23 * x + 0.29 * y) * size + 2.2),
    ))
    return result


def write_f32(path: Path, values: np.ndarray) -> None:
    checked = np.asarray(values, dtype="<f4")
    if not np.isfinite(checked).all():
        raise ValueError("non-finite reference data")
    path.write_bytes(checked.tobytes(order="C"))


def authenticate(root: Path, archive: Path, paper2017: Path, paper2015: Path) -> None:
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise SystemExit("ARI reference archive identity mismatch")
    for path, identity in ((paper2017, PAPER_2017), (paper2015, PAPER_2015)):
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise SystemExit(f"ARI paper identity mismatch: {path}")
    for name, expected in REFERENCE_FILES.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"ARI reference identity mismatch: {name}")


def prepare_octave_reference(root: Path, destination: Path) -> None:
    """Copy and minimally stabilize mathematically nonnegative fit variances."""
    shutil.copytree(root, destination)
    original = "dif = dif.^0.5;"
    replacement = "dif = max(dif,0).^0.5; % Octave real-roundoff compatibility"
    for name in OCTAVE_COMPATIBILITY_FILES:
        path = destination / name
        text = path.read_text()
        if text.count(original) != 1:
            raise SystemExit(f"unexpected ARI compatibility patch context: {name}")
        path.write_text(text.replace(original, replacement))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--paper-2017", type=Path, required=True)
    parser.add_argument("--paper-2015", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.reference_root.resolve()
    archive = args.archive.resolve()
    authenticate(root, archive, args.paper_2017.resolve(), args.paper_2015.resolve())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ari-bayer-reference-", dir=output.parent))
    script = Path(__file__).with_name("reference") / "run_reference.m"
    entries = []
    maximum = 0.0
    squared = 0.0
    count = 0
    interior_maximum = 0.0
    interior_squared = 0.0
    interior_count = 0
    try:
        octave_root = temporary / "octave-reference"
        prepare_octave_reference(root, octave_root)
        for pattern in ("grbg", "rggb", "gbrg", "bggr"):
            for name, truth64 in scenes().items():
                # All scenes are tested on GRBG; other patterns use a compact
                # subset sufficient to freeze phase/orientation behavior.
                if pattern != "grbg" and name not in ("horizontal-gradient", "saturated-red-point"):
                    continue
                truth = np.asarray(truth64, dtype=np.float32)
                mosaic, masks = mosaic_bayer(truth, pattern)
                case = f"{pattern}-{name}"
                truth_path = temporary / f"{case}-truth.f32le"
                mosaic_path = temporary / f"{case}-mosaic.f32le"
                output_path = temporary / f"{case}-official.f32le"
                official_green_path = temporary / f"{case}-official-green.f32le"
                green_path = temporary / f"{case}-independent-green.f32le"
                write_f32(truth_path, truth)
                write_f32(mosaic_path, mosaic)
                environment = dict(os.environ)
                environment["ARI_REFERENCE_ARCHIVE"] = str(archive)
                common = [
                    "octave", "--quiet", str(script), str(octave_root), str(truth_path),
                    str(official_green_path), str(truth.shape[2]), str(truth.shape[1]), pattern,
                    "green",
                    REFERENCE_FILES["demosaic_ARI.m"], REFERENCE_FILES["readme.txt"],
                    ARCHIVE_SHA256,
                ]
                official_green = None
                official_green_identity = None
                subprocess.run(common, check=True, env=environment)
                official_green = np.fromfile(
                    official_green_path, dtype="<f4"
                ).reshape(truth.shape[1:])
                official_green_identity = {
                    "bytes": official_green_path.stat().st_size,
                    "file": official_green_path.name,
                    "sha256": sha256(official_green_path),
                }
                full_output = None
                full_command = list(common)
                full_command[5] = str(output_path)
                full_command[9] = "full"
                subprocess.run(full_command, check=True, env=environment)
                full_output = {
                    "bytes": output_path.stat().st_size,
                    "file": output_path.name,
                    "sha256": sha256(output_path),
                }
                trace = green_ari(mosaic, masks, pattern)
                write_f32(green_path, trace.adaptive)
                if official_green is not None:
                    difference = trace.adaptive - official_green
                    maximum = max(maximum, float(np.max(np.abs(difference))))
                    squared += float(np.sum(difference**2))
                    count += difference.size
                    interior = difference[16:-16, 16:-16]
                    interior_maximum = max(interior_maximum, float(np.max(np.abs(interior))))
                    interior_squared += float(np.sum(interior**2))
                    interior_count += interior.size
                entries.append({
                    "height": truth.shape[1],
                    "independent_green": {
                        "bytes": green_path.stat().st_size,
                        "file": green_path.name,
                        "sha256": sha256(green_path),
                    },
                    "mosaic": {
                        "bytes": mosaic_path.stat().st_size,
                        "file": mosaic_path.name,
                        "sha256": sha256(mosaic_path),
                    },
                    "name": name,
                    "official_green": official_green_identity,
                    "official_output": full_output,
                    "pattern": pattern,
                    "truth": {
                        "bytes": truth_path.stat().st_size,
                        "file": truth_path.name,
                        "sha256": sha256(truth_path),
                    },
                    "width": truth.shape[2],
                })
        # The authenticated upstream MATLAB tree is research-only input.  It
        # must never become part of the independently generated golden corpus.
        shutil.rmtree(octave_root)
        parity = {
            "interior_margin": 16,
            "interior_maximum_abs": interior_maximum,
            "interior_rms": float(np.sqrt(interior_squared / interior_count)),
            "maximum_abs": maximum,
            "rms": float(np.sqrt(squared / count)),
        }
        manifest = {
            "archive": {"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256, "version": "1.0"},
            "arithmetic": {
                "input": "normalized little-endian channel-major float32 linear RGB",
                "official_internal_domain": "double 0..255",
                "output": "normalized little-endian channel-major float32 RGB",
                "green_iterations": 11,
                "red_blue_iterations_per_stage": 2,
            },
            "cases": entries,
            "format": FORMAT,
            "independent_green_parity": parity,
            "license": "research purpose only; all rights reserved",
            "octave_compatibility": {
                "files": list(OCTAVE_COMPATIBILITY_FILES),
                "replacement": "dif = max(dif,0).^0.5",
                "scope": (
                    "temporary authenticated source copy only; clamps negative "
                    "floating-point roundoff before a mathematically nonnegative RMS"
                ),
            },
            "octave_unmodified_limitation": (
                "The unmodified reference can form a complex sqrt from a tiny "
                "negative regression roundoff, and Octave imfilter rejects complex "
                "intermediates. The documented compatibility clamp enforces the "
                "nonnegative least-squares residual invariant."
            ),
            "papers": {"icip_2015": PAPER_2015, "sensors_2017": PAPER_2017},
            "reference_files": REFERENCE_FILES,
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        print(f"wrote {output}")
        print(f"manifest SHA-256: {sha256(output / 'manifest.json')}")
        print(json.dumps(parity, sort_keys=True))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

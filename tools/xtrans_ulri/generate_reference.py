#!/usr/bin/env python3
"""Generate a deterministic corpus from the official ULRI X-Trans source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


FORMAT = "rawtherapee-xtrans-ulri-reference-v1"
REFERENCE_FILES = {
    "function_demosaic_x_trans.m": "bc8a2a557a326af2ea72f7b77b17d55d6785570fa36197f732288a2d54024b69",
    "guidedfilter_MLRI_wei.m": "c26f06bef57e6ce99fca8369d8dc0ad414674978175623d4a0e5a9b0db59d0fd",
    "imgconv2.m": "dd7b68011c2fc7e9a20c5688bbc92e4d51b8ab387575b6d73826043bba95e388",
    "clip.m": "f66bed703a4d95336832815d83f2d15fbc33004043920ad08847753fd2488022",
    "license.txt": "8da7c29a607b5d6450bb2dbf7b5c0a7a93cbe5bb69a8d35d0fb4de5d3940a7c4",
}
ARCHIVE_SHA256 = "0dab03107479153a68fe6feeef738e3fb9c058031634dc006e251f2fd7a81347"
ARCHIVE_BYTES = 8_214_511
CFA = np.asarray(
    (
        (1, 0, 1, 1, 2, 1),
        (2, 1, 2, 0, 1, 0),
        (1, 0, 1, 1, 2, 1),
        (1, 2, 1, 1, 0, 1),
        (0, 1, 0, 2, 1, 2),
        (1, 2, 1, 1, 0, 1),
    ),
    dtype=np.uint8,
)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:height, :width]
    return x.astype(np.float64) / (width - 1), y.astype(np.float64) / (height - 1)


def scenes() -> dict[str, np.ndarray]:
    height = width = 48
    x, y = _grid(height, width)
    result: dict[str, np.ndarray] = {}
    for name, color in (
        ("constant-gray", (0.4, 0.4, 0.4)),
        ("constant-red", (1.0, 0.0, 0.0)),
        ("constant-green", (0.0, 1.0, 0.0)),
        ("constant-blue", (0.0, 0.0, 1.0)),
    ):
        result[name] = np.broadcast_to(
            np.asarray(color, dtype=np.float64)[:, None, None], (3, height, width)
        ).copy()
    result["horizontal-gradient"] = np.stack((x, 0.15 + 0.7 * x, 1.0 - x))
    result["vertical-gradient"] = np.stack((y, 0.1 + 0.8 * y, 1.0 - y))
    diagonal = (x + y) * 0.5
    result["diagonal-gradient"] = np.stack(
        (diagonal, 0.2 + 0.6 * diagonal, 1.0 - diagonal)
    )
    wave = 0.5 + 0.32 * np.sin(2 * np.pi * (x * 1.25 + y * 0.5))
    result["low-frequency-sinusoid"] = np.stack(
        (wave, 0.2 + 0.6 * wave, 0.85 - 0.55 * wave)
    )
    impulse = np.full((3, height, width), 0.08, dtype=np.float64)
    impulse[:, height // 2, width // 2] = 1.0
    result["white-impulse"] = impulse
    red_point = np.full((3, height, width), 0.05, dtype=np.float64)
    red_point[0, height // 2, width // 2 + 1] = 1.0
    result["isolated-red-point"] = red_point
    edge = np.empty((3, height, width), dtype=np.float64)
    edge[:, :, : width // 2] = np.asarray((0.95, 0.1, 0.1))[:, None, None]
    edge[:, :, width // 2 :] = np.asarray((0.1, 0.15, 0.95))[:, None, None]
    result["vertical-chromatic-edge"] = edge
    return result


def mosaic(truth: np.ndarray) -> np.ndarray:
    height, width = truth.shape[1:]
    y, x = np.mgrid[:height, :width]
    phases = CFA[y % 6, x % 6]
    return np.take_along_axis(np.moveaxis(truth, 0, -1), phases[..., None], axis=2)[..., 0]


def write_float32(path: Path, values: np.ndarray) -> None:
    checked = np.asarray(values, dtype="<f4")
    if not np.isfinite(checked).all():
        raise ValueError("reference corpus contains a non-finite value")
    path.write_bytes(checked.tobytes(order="C"))


def authenticate(reference_root: Path, archive: Path | None) -> None:
    for name, expected in REFERENCE_FILES.items():
        path = reference_root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"ULRI reference identity mismatch: {name}")
    if archive is not None:
        if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
            raise SystemExit("ULRI archive identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("reference_golden"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.reference_root.resolve()
    archive = args.archive.resolve() if args.archive else None
    authenticate(root, archive)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")

    script = Path(__file__).with_name("reference") / "run_reference.m"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ulri-reference-", dir=output.parent))
    try:
        entries = []
        for name, truth64 in scenes().items():
            truth = np.asarray(truth64, dtype=np.float32)
            scalar = np.asarray(mosaic(truth), dtype=np.float32)
            scalar = np.round(scalar * 65535.0).astype(np.float32) / 65535.0
            height, width = scalar.shape
            truth_path = temporary / f"{name}-truth.f32le"
            input_path = temporary / f"{name}-mosaic.f32le"
            prefix = temporary / name
            write_float32(truth_path, truth)
            write_float32(input_path, scalar)
            subprocess.run(
                [
                    "octave", "--quiet", str(script),
                    str(root / "function_demosaic_x_trans.m"), str(input_path),
                    str(prefix), str(width), str(height),
                    REFERENCE_FILES["function_demosaic_x_trans.m"], "3",
                ],
                check=True,
            )
            outputs = []
            for slow in range(4):
                path = temporary / f"{name}-slow{slow}.f32le"
                expected_bytes = 3 * height * width * 4
                if path.stat().st_size != expected_bytes:
                    raise SystemExit(f"incorrect Octave output size: {path}")
                values = np.fromfile(path, dtype="<f4")
                if not np.isfinite(values).all():
                    raise SystemExit(f"non-finite Octave output: {path}")
                outputs.append({
                    "bytes": expected_bytes,
                    "file": path.name,
                    "passes": 1 if slow == 0 else slow + 1,
                    "sha256": sha256(path),
                    "slow": slow,
                })
            entries.append({
                "height": height,
                "mosaic": {
                    "bytes": input_path.stat().st_size,
                    "file": input_path.name,
                    "sha256": sha256(input_path),
                },
                "name": name,
                "outputs": outputs,
                "truth": {
                    "bytes": truth_path.stat().st_size,
                    "file": truth_path.name,
                    "sha256": sha256(truth_path),
                },
                "width": width,
            })
        manifest = {
            "arithmetic": {
                "boundary": "zero extension through conv2(..., same)",
                "epsilon": 0.01,
                "input": "normalized little-endian row-major float32 scalar mosaic",
                "internal_domain": "uint16 input converted to single precision 0..255",
                "output": "normalized little-endian channel-major float32 RGB",
                "sigma": 2.0,
                "slow_contract": "slow=0 is one pass/direct final; slow=N is N+1 passes plus sqrt(G/255) blend",
            },
            "archive": {
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
                "version": "1.0.0",
            },
            "cases": entries,
            "cfa": CFA.tolist(),
            "format": FORMAT,
            "license": "BSD-3-Clause",
            "octave_packages": ["image"],
            "reference_files": REFERENCE_FILES,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        print(f"wrote {output}")
        print(f"manifest SHA-256: {hashlib.sha256(manifest_bytes).hexdigest()}")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

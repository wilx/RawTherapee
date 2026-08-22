#!/usr/bin/env python3
"""Generate the deterministic Octave oracle corpus for X-Trans MLRI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import struct
import subprocess
import tempfile


REFERENCE_SHA256 = "055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c"
FORMAT = "rawtherapee-xtrans-mlri-golden-v1"
CFA = (
    (1, 0, 1, 1, 2, 1),
    (2, 1, 2, 0, 1, 0),
    (1, 0, 1, 1, 2, 1),
    (1, 2, 1, 1, 0, 1),
    (0, 1, 0, 2, 1, 2),
    (1, 2, 1, 1, 0, 1),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases() -> list[tuple[str, int, int, list[float]]]:
    result: list[tuple[str, int, int, list[float]]] = []

    width = height = 48
    flat_values = (12000.0, 24000.0, 36000.0)
    result.append(("flat", width, height, [
        flat_values[CFA[y % 6][x % 6]] for y in range(height) for x in range(width)
    ]))

    width, height = 54, 48
    gradient = []
    for y in range(height):
        for x in range(width):
            red = 3000.0 + 42000.0 * x / (width - 1) + 3500.0 * y / (height - 1)
            green = 7000.0 + 33000.0 * y / (height - 1) + 2200.0 * x / (width - 1)
            blue = 41000.0 - 26000.0 * x / (width - 1) + 6500.0 * y / (height - 1)
            gradient.append(float(round((red, green, blue)[CFA[y % 6][x % 6]])))
    result.append(("gradient", width, height, gradient))

    width = height = 48
    impulse = [6000.0 + 1500.0 * CFA[y % 6][x % 6]
               for y in range(height) for x in range(width)]
    impulse[24 * width + 25] = 65535.0
    result.append(("impulse", width, height, impulse))

    saturated = []
    for y in range(height):
        for x in range(width):
            phase = (x * 5 + y * 7 + (x // 3) * 11) & 3
            saturated.append((0.0, 65535.0, 4096.0, 61440.0)[phase])
    result.append(("saturated", width, height, saturated))

    width, height = 37, 35
    boundary = []
    for y in range(height):
        for x in range(width):
            value = 9000.0 + ((x * 7919 + y * 104729 + x * y * 17) % 43000)
            if x in (0, width - 1) or y in (0, height - 1):
                value = 65535.0 if (x + y) & 1 else 0.0
            boundary.append(value)
    result.append(("boundary", width, height, boundary))
    return result


def write_floats(path: pathlib.Path, values: list[float]) -> None:
    with path.open("wb") as stream:
        for value in values:
            if not math.isfinite(value):
                raise ValueError("golden input contains a non-finite value")
            stream.write(struct.pack("<f", value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True,
                        help="path to authenticated function_demosaic_x_trans.m")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("golden"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    if sha256(source) != REFERENCE_SHA256:
        raise SystemExit("X-Trans MATLAB reference SHA-256 mismatch")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")

    script = pathlib.Path(__file__).with_name("reference") / "run_reference.m"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(
        prefix=".rawtherapee-mlri-golden-", dir=output.parent))
    try:
        entries = []
        for name, width, height, values in cases():
            input_path = temporary / f"{name}-input.f32"
            output_path = temporary / f"{name}-output.f32"
            write_floats(input_path, values)
            subprocess.run([
                "octave", "--quiet", str(script), str(source), str(input_path),
                str(output_path), str(width), str(height), REFERENCE_SHA256,
            ], check=True)
            expected_bytes = width * height * 3 * 4
            if output_path.stat().st_size != expected_bytes:
                raise SystemExit(f"incorrect Octave output size for {name}")
            entries.append({
                "height": height,
                "input_bytes": input_path.stat().st_size,
                "input_file": input_path.name,
                "input_sha256": sha256(input_path),
                "name": name,
                "origin": [0, 0],
                "output_bytes": output_path.stat().st_size,
                "output_file": output_path.name,
                "output_sha256": sha256(output_path),
                "width": width,
            })

        manifest = {
            "arithmetic": {
                "boundary": "zero",
                "epsilon": 0.01,
                "input": "little-endian row-major float32 in the 0..65535 domain",
                "oracle_input_type": "uint16",
                "output": "uint16 oracle values stored as little-endian channel-major row-major float32",
                "passes": 2,
                "sigma_by_pass": [2.0, 1.0],
            },
            "cases": entries,
            "format": FORMAT,
            "octave_packages": ["image"],
            "reference_sha256": REFERENCE_SHA256,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
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

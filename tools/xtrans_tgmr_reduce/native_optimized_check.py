#!/usr/bin/env python3
"""Exercise the optimized native TGMR kernel on frozen evaluation samples."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tools.xtrans_gmm.dataset import (
    TEST_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
)
from tools.xtrans_gmr.experiment import _synthetic_samples, load_model
from tools.xtrans_tgmr.experiment import _synthetic_scenes

from .native_check import _samples
from .native_model import native_float32_predict, prepare_native_model


def write_observations(path: Path, native, samples) -> None:
    """Write the bounded, little-endian native-check observation format."""
    with path.open("wb") as stream:
        stream.write(b"XTGROBS1")
        stream.write(struct.pack("<I", len(samples)))
        for sample in samples:
            phase_index = int(sample.phase)
            phase = native.phases[phase_index]
            observed = np.asarray(
                sample.vector[phase.observed_indices], dtype="<f4"
            )
            if observed.shape != (49,) or not np.isfinite(observed).all():
                raise RuntimeError("invalid native-check observation")
            stream.write(struct.pack("<I", phase_index))
            stream.write(observed.tobytes(order="C"))


def _full_groups(bsds_root: Path, starfield: Path):
    _, splits = corpus_manifest(bsds_root)
    groups = {
        "bsds-test": evaluation_samples(
            splits["test"], 7, "bsds-test", TEST_GRID_SIDE
        ),
        "external": external_samples(7),
    }
    groups.update(established_samples(starfield, 7))
    groups.update({
        f"synthetic:{name}": _synthetic_samples(scene)
        for name, scene in _synthetic_scenes().items()
    })
    return groups


def run_check(
    executable: Path,
    native_artifact: Path,
    model_path: Path,
    groups,
    threads: int,
    chunk: int,
) -> dict[str, object]:
    native = prepare_native_model(load_model(model_path))
    samples = [sample for values in groups.values() for sample in values]
    expected, _ = native_float32_predict(native, samples)
    with tempfile.TemporaryDirectory(prefix="xtrans-tgmr-native-check-") as directory:
        root = Path(directory)
        observations = root / "observations.bin"
        output = root / "output.f32"
        write_observations(observations, native, samples)
        completed = subprocess.run(
            [
                str(executable),
                str(native_artifact),
                "o4-avx-f2",
                str(len(samples)),
                "1",
                str(threads),
                "1",
                "--chunk",
                str(chunk),
                "--input",
                str(observations),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        benchmark = json.loads(completed.stdout)
        actual = np.fromfile(output, dtype="<f4").reshape(-1, 3)
    difference = np.asarray(actual, dtype=np.float64) - expected
    rows = {}
    offset = 0
    for name, values in groups.items():
        count = len(values)
        group_difference = difference[offset : offset + count]
        rows[name] = {
            "count": count,
            "maximum_abs": float(np.max(np.abs(group_difference))),
            "rms": float(np.sqrt(np.mean(group_difference * group_difference))),
        }
        offset += count
    native_exact = True
    for sample, rgb in zip(samples, actual, strict=True):
        center_index = int(sample.indices[np.asarray(sample.indices) % 49 == 24][0])
        sampled_channel = center_index // 49
        native_exact &= bool(
            np.asarray(rgb[sampled_channel], dtype=np.float32)
            == np.asarray(sample.vector[center_index], dtype=np.float32)
        )
    result = {
        "format": "rawtherapee-xtrans-tgmr-native-parity-v1",
        "benchmark": benchmark,
        "finite": bool(np.isfinite(actual).all()),
        "groups": rows,
        "maximum_abs": float(np.max(np.abs(difference))),
        "native_center_exact": native_exact,
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "sample_count": len(samples),
    }
    result["pass"] = bool(
        result["finite"]
        and result["native_center_exact"]
        and result["rms"] <= 1e-6
        and result["maximum_abs"] <= 1e-5
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pixels", type=int, default=18 * 64)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield",
        type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    arguments = parser.parse_args()
    native = prepare_native_model(load_model(arguments.model))
    if arguments.full:
        groups = _full_groups(arguments.bsds_root, arguments.starfield)
    else:
        groups = {"deterministic-probe": _samples(native, arguments.pixels)}
    result = run_check(
        arguments.executable,
        arguments.native_artifact,
        arguments.model,
        groups,
        arguments.threads,
        arguments.chunk,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

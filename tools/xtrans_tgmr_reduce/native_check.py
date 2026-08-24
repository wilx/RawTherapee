#!/usr/bin/env python3
"""Compare a native benchmark output dump with the float32 Python contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.xtrans_gmr.experiment import load_model
from tools.xtrans_sparse_dictionary.dataset import PatchSample

from .native_model import native_float32_predict, prepare_native_model


def _mix(value: int) -> int:
    value &= 0xffffffff
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xffffffff
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xffffffff
    return (value ^ (value >> 16)) & 0xffffffff


def _samples(native, count: int) -> list[PatchSample]:
    result = []
    for pixel in range(count):
        phase_index = pixel % 18
        phase = native.phases[phase_index]
        values = np.asarray([
            0.05 + 0.9 * (
                _mix(pixel ^ (index * 0x9E3779B9) ^ (phase_index * 0x85EBCA6B))
                & 0x00FFFFFF
            ) / 0x01000000
            for index in range(49)
        ], dtype=np.float32)
        vector = np.zeros(147, dtype=np.float64)
        vector[phase.observed_indices] = values
        result.append(PatchSample(
            source_id="native-probe", group="probe", vector=vector,
            indices=phase.observed_indices.astype(np.int64), phase=phase_index,
            x=0, y=0,
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--pixels", type=int, required=True)
    arguments = parser.parse_args()
    native = prepare_native_model(load_model(arguments.model))
    reference, _ = native_float32_predict(native, _samples(native, arguments.pixels))
    actual = np.fromfile(arguments.dump, dtype="<f4")
    if actual.size != arguments.pixels * 3:
        raise RuntimeError("native dump size does not match probe")
    actual = actual.reshape(arguments.pixels, 3)
    difference = np.asarray(actual, dtype=np.float64) - reference
    result = {
        "maximum_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "finite": bool(np.isfinite(actual).all()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["maximum_abs"] <= 2e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())

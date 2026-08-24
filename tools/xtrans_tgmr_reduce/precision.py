#!/usr/bin/env python3
"""Measure selected-model float32 error against its float64 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmm.dataset import (
    TEST_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
)
from tools.xtrans_gmr.experiment import _synthetic_samples, load_model
from tools.xtrans_tgmr.experiment import _synthetic_scenes

from .final import _shortlist_batch
from .native_model import native_float32_predict, prepare_native_model


def _summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - reference
    absolute = np.abs(difference).reshape(-1)
    return {
        "maximum_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "p99_abs": float(np.quantile(absolute, 0.99)),
    }


def run(model_path: Path, bsds_root: Path, starfield: Path) -> dict[str, object]:
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
    model = load_model(model_path)
    native = prepare_native_model(model)
    rows = {}
    all_reference = []
    all_candidate = []
    for name, samples in groups.items():
        reference, _ = _shortlist_batch(model, samples, 3, 8)
        candidate, _ = native_float32_predict(native, samples)
        rows[name] = _summary(reference.mmse_rgb, candidate)
        all_reference.append(reference.mmse_rgb)
        all_candidate.append(candidate)
        print(f"precision group={name}", flush=True)
    result = {
        "format": "rawtherapee-xtrans-tgmr-reduction-precision-v1",
        "contract": {
            "components": 32,
            "coarse_support": 3,
            "shortlist": 8,
            "reference": "float64",
            "candidate": "float32-parameters-and-inference",
        },
        "groups": rows,
        "pooled": _summary(
            np.concatenate(all_reference), np.concatenate(all_candidate)
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    arguments = parser.parse_args()
    result = run(arguments.model, arguments.bsds_root, arguments.starfield)
    arguments.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

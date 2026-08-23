#!/usr/bin/env python3
"""Regenerate detailed metrics for the validation-selected population bank."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes

from .dataset import discover, load_rgb
from .experiment import _metrics, _pooled
from .model import predict_region
from .population import _train_bank


def run(
    bsds_root: Path,
    population_path: Path,
    output: Path,
) -> dict[str, object]:
    population = json.loads(population_path.read_text(encoding="utf-8"))
    support = int(population["selected"]["support"])
    dc_mode = str(population["selected"]["dc_mode"])
    splits = discover(bsds_root)
    bank, training = _train_bank(
        splits["train"], splits["val"], support, dc_mode
    )
    old_rows = {
        str(row["filename"]): row for row in population["test_sources"]
    }
    rows = []
    unbounded_metrics = []
    clipped_metrics = []
    absolute_errors = []
    for source in splits["test"]:
        truth = load_rgb(source)
        result = predict_region(bank, truth, (0, 0, truth.shape[2], truth.shape[1]))
        clipped = np.clip(result, 0.0, 1.0)
        unbounded = _metrics(result, truth)
        bounded = _metrics(clipped, truth)
        unbounded_metrics.append(unbounded)
        clipped_metrics.append(bounded)
        difference = np.abs(result - truth)
        selected = np.zeros(truth.shape[1:], dtype=bool)
        selected[12:-12, 12:-12] = True
        absolute_errors.append(difference[:, selected].ravel())
        rows.append({
            "baselines": {
                method: old_rows[str(source["filename"])]["methods"][method]
                for method in ("markesteijn", "corrected-final", "ulri-slow0", "ulri-slow3")
            },
            "clipped": bounded,
            "filename": source["filename"],
            "sha256": source["sha256"],
            "unclipped": unbounded,
        })
    absolute = np.concatenate(absolute_errors)
    result = {
        "dc_mode": dc_mode,
        "format": "rawtherapee-xtrans-lmmse-selected-details-v1",
        "native_sample_maximum_abs": 0.0,
        "pooled": {
            "clipped": _pooled(clipped_metrics),
            "unclipped": {
                **_pooled(unbounded_metrics),
                "p50_abs": float(np.quantile(absolute, 0.50)),
                "p90_abs": float(np.quantile(absolute, 0.90)),
                "p95_abs": float(np.quantile(absolute, 0.95)),
                "p99_abs": float(np.quantile(absolute, 0.99)),
            },
        },
        "preclip_maximum": float(max(
            np.max(predict_region(
                bank, load_rgb(source),
                (0, 0, int(source["width"]), int(source["height"])),
            ))
            for source in splits["test"]
        )),
        "preclip_minimum": float(min(
            np.min(predict_region(
                bank, load_rgb(source),
                (0, 0, int(source["width"]), int(source["height"])),
            ))
            for source in splits["test"]
        )),
        "ridge_ratio": bank.ridge_ratio,
        "sources": rows,
        "support": support,
        "training": training,
    }
    output.write_bytes(canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsds-root", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.bsds_root, arguments.population, arguments.output)
    print(json.dumps({
        "pooled": result["pooled"],
        "preclip_maximum": result["preclip_maximum"],
        "preclip_minimum": result["preclip_minimum"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

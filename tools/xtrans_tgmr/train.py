#!/usr/bin/env python3
"""Continue a frozen phase GMR checkpoint with fixed-nu Student-t ECM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmr.experiment import _logical_digest, load_model, save_model
from tools.xtrans_gmr.model import model_summary

from .model import refine_fixed_student_t_mixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees-of-freedom", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    arguments = parser.parse_args()
    manifest = arguments.output.with_suffix(arguments.output.suffix + ".json")
    if arguments.output.exists() or manifest.exists():
        raise SystemExit("refusing to replace Student-t checkpoint or manifest")
    vectors = np.load(arguments.vectors, mmap_mode="r")
    initial = load_model(arguments.initial)
    started = time.perf_counter()
    result = refine_fixed_student_t_mixture(
        vectors,
        initial,
        degrees_of_freedom=arguments.degrees_of_freedom,
        iterations=arguments.iterations,
    )
    elapsed = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(arguments.output.name + ".tmp.npz")
    save_model(temporary, result.model)
    os.replace(temporary, arguments.output)
    payload = {
        "format": "rawtherapee-xtrans-fixed-student-t-ecm-checkpoint-v1",
        "degrees_of_freedom": result.degrees_of_freedom,
        "ecm_iterations_this_run": result.ecm_iterations,
        "elapsed_seconds": elapsed,
        "initial_logical_sha256": _logical_digest(initial),
        "latent_weight_quantiles_0_1_5_50_95_99_100": result.latent_weight_quantiles,
        "mean_training_log_likelihood": result.mean_log_likelihood,
        "model": {
            **model_summary(result.model),
            "logical_sha256": _logical_digest(result.model),
        },
    }
    temporary_manifest = manifest.with_name(manifest.name + ".tmp")
    temporary_manifest.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary_manifest, manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

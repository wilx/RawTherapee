#!/usr/bin/env python3
"""Reproduce the architecture of SCICO's public Bayer BM3D-PnP example.

The forward operator, crop, noise level, initializer family, rho, denoiser
strength, and iteration count match the documented example.  SCICO's JAX RNG
and older ``bm3d_rgb`` implementation are not bit-reproducible through the
authenticated BM3D 4.0.3 package, so this is an equation/interface control,
not a claim of byte-identical SCICO output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.xtrans_alias.analysis import canonical_json_bytes

from .denoiser import BM3DDenoiser, authenticate_bm3d
from .operator import bayer_bggr_operator
from .solver import pnp_admm


SCICO_COMMIT = "b082d7b5ee37b5d3dcbb3a7136f4c8628042b8ea"
SCICO_DATA_COMMIT = "0f5afcfcabb99b4af10fe5d94edceed5369ad5ac"
KODIM23_SHA256 = "34d8f029d2eae3ed37b65e2f94fd2c319923ebfae0fa8bff5f001e2565b7d80f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(output: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    difference = np.asarray(output, dtype=np.float64) - truth
    mse = float(np.mean(difference * difference))
    return {
        "maximum_abs": float(np.max(np.abs(difference))),
        "mse": mse,
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
    }


def run(image_path: Path) -> dict[str, object]:
    if _sha256(image_path) != KODIM23_SHA256:
        raise ValueError("SCICO kodim23 digest mismatch")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    truth_hwc = np.ascontiguousarray(rgb[160:416, 60:316])
    truth = np.moveaxis(truth_hwc, -1, 0)
    operator = bayer_bggr_operator(256, 256)
    clean = operator.forward(truth)
    rng = np.random.default_rng(0)
    noisy = clean + 0.02 * rng.standard_normal(clean.shape)

    # SCICO uses the Menon 2007 BGGR initializer followed by color BM3D at
    # three times the simulated noise sigma.
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

    menon = demosaicing_CFA_Bayer_Menon2007(noisy, pattern="BGGR")
    denoiser = BM3DDenoiser("linear-luma")
    initial = denoiser(np.moveaxis(menon, -1, 0), 0.06)
    result = pnp_admm(
        operator,
        noisy,
        initial,
        denoiser,
        sigma=0.061,
        rho=0.18,
        iterations=12,
        exact_projection=False,
        capture_iterations=(1, 2, 3, 5, 10, 12),
    )
    return {
        "bm3d": authenticate_bm3d(),
        "data": {
            "crop_hwc": [160, 416, 60, 316],
            "kodim23_sha256": KODIM23_SHA256,
            "scico_commit": SCICO_COMMIT,
            "scico_data_commit": SCICO_DATA_COMMIT,
        },
        "format": "rawtherapee-xtrans-pnp-bayer-reference-v1",
        "interpretation": (
            "SCICO architecture reproduction; modern authenticated BM3D and NumPy RNG "
            "prevent bit-identical comparison with the published notebook"
        ),
        "metrics": {
            "initial": _metric(initial, truth),
            "iterations": {
                str(iteration): _metric(result.snapshots[iteration], truth)
                for iteration in (1, 2, 3, 5, 10, 12)
            },
        },
        "parameters": {
            "bayer_pattern": "BGGR",
            "denoiser_color_mode": "orthonormal linear-luma first channel",
            "denoiser_sigma": 0.061,
            "initial_bm3d_sigma": 0.06,
            "measurement_noise_sigma": 0.02,
            "rho": 0.18,
            "iterations": 12,
            "projection": "soft quadratic data fidelity",
        },
        "records": [record.__dict__ for record in result.records],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("/tmp/scico-kodim23.png"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = canonical_json_bytes(run(arguments.image))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)
    print(json.dumps(json.loads(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Authenticated external BM3D adapter used only by the research CLI."""

from __future__ import annotations

import hashlib
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import time

import numpy as np


BM3D_VERSION = "4.0.3"
BM4D_VERSION = "4.2.5"
BM3D_WHEEL_SHA256 = "fc4dfc0de0cd810fcb6ad198e1d0c6f99cf19d41f2ec69ff867674cfb9f2a775"
BM4D_WHEEL_SHA256 = "601338bbd54bcbd971d70b5a6961583a9ccaa564c982389cf3f82bd0a6d5bad5"

# Installed payload authentication is necessary because the experiment invokes
# external compiled code.  Wheel hashes remain documented separately because
# installed distributions no longer retain the original wheel container.
EXPECTED_FILES = {
    "bm3d": {
        "bm3d/__init__.py": "974f655466702259b8a49d5bf44ec7c734b0677c40781350476ad747bdf9ba44",
        "bm3d/profiles.py": "5f25cb08dbed7f1e70e88e3ffb09aef838cb745aaef4ed01a83768178832be0e",
    },
    "bm4d": {
        "bm4d/__init__.py": "7e5cae1b2146e27ef49280255e7f1a280c82fc41828cad657bd7a399d6b572b0",
        "bm4d/bm4d_ctypes.py": "9ef0a3d31e7b69e959c8bc572ce7d2bcefb5d0fadb0103c2686be906b3d33eb4",
        "bm4d/libbm4d.so": "70d7e1b2e955ded5c3e85c465da7d90525d76d6e583be9249ca14f83da2fed30",
        "bm4d/param_matching_data.mat": "5f3974cbba676c80f268b1ccefa4a763fd808f353b38ca1f2a3ac6688ccd7593",
    },
}


LINEAR_LUMA_BASIS = np.asarray(
    (
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
        (1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0)),
        (1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)),
    ),
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_bm3d() -> dict[str, object]:
    """Authenticate the exact external Python wrapper and Linux binary."""

    identities = {}
    for package, expected_version in (("bm3d", BM3D_VERSION), ("bm4d", BM4D_VERSION)):
        try:
            installed = distribution(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"required external package {package} is not installed") from error
        if installed.version != expected_version:
            raise RuntimeError(
                f"{package} version mismatch: {installed.version}, expected {expected_version}"
            )
        files = {}
        for relative, expected_digest in EXPECTED_FILES[package].items():
            path = Path(installed.locate_file(relative))
            digest = _sha256(path)
            if digest != expected_digest:
                raise RuntimeError(f"{package} payload digest mismatch: {relative}: {digest}")
            files[relative] = digest
        identities[package] = {
            "files": files,
            "version": installed.version,
            "wheel_sha256": BM3D_WHEEL_SHA256 if package == "bm3d" else BM4D_WHEEL_SHA256,
        }
    return {
        "algorithm": "BM3D 2020 correlated-noise wrapper, all stages",
        "color_behavior": "joint multichannel filtering; block matching on channel zero",
        "license": "Tampere University non-commercial research license",
        "packages": identities,
        "profile": "np",
    }


class BM3DDenoiser:
    """Convert CHW linear RGB to the external package's HWC convention."""

    def __init__(self, color_mode: str = "rgb"):
        if color_mode not in ("rgb", "linear-luma"):
            raise ValueError("color mode must be rgb or linear-luma")
        self.color_mode = color_mode
        self.identity = authenticate_bm3d()
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, chw: np.ndarray, sigma: float) -> np.ndarray:
        checked = np.asarray(chw, dtype=np.float64)
        if checked.ndim != 3 or checked.shape[0] != 3 or not np.isfinite(checked).all():
            raise ValueError("BM3D input must be finite CHW RGB")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("BM3D strength must be finite and nonnegative")
        from bm3d import bm3d

        values = checked
        if self.color_mode == "linear-luma":
            values = np.einsum("ab,bhw->ahw", LINEAR_LUMA_BASIS, checked)
        started = time.perf_counter()
        output = bm3d(np.moveaxis(values, 0, -1), float(sigma), profile="np")
        self.seconds += time.perf_counter() - started
        self.calls += 1
        result = np.moveaxis(np.asarray(output, dtype=np.float64), -1, 0)
        if self.color_mode == "linear-luma":
            result = np.einsum("ba,bhw->ahw", LINEAR_LUMA_BASIS, result)
        if result.shape != checked.shape or not np.isfinite(result).all():
            raise RuntimeError("external BM3D returned invalid data")
        return np.ascontiguousarray(result)


__all__ = (
    "BM3DDenoiser",
    "BM3D_VERSION",
    "BM4D_VERSION",
    "authenticate_bm3d",
)

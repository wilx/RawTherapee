"""Exact scalar-CFA forward and adjoint operators.

The operator is intentionally pattern-agnostic.  It never converts X-Trans to
a Bayer-like grid: each scalar observation selects the actual R, G, or B
component at that pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.xtrans_mlri_internal.dataset import cfa_for_origin


@dataclass(frozen=True)
class CFAOperator:
    """A fixed H x W scalar-CFA sampling operator."""

    cfa: np.ndarray

    def __post_init__(self) -> None:
        checked = np.asarray(self.cfa)
        if checked.ndim != 2 or checked.size == 0:
            raise ValueError("CFA must be a nonempty two-dimensional array")
        if checked.dtype.kind not in "iu" or np.any((checked < 0) | (checked > 2)):
            raise ValueError("CFA values must be integer RGB indices")
        object.__setattr__(self, "cfa", np.ascontiguousarray(checked, dtype=np.uint8))

    @property
    def shape(self) -> tuple[int, int]:
        return self.cfa.shape

    @property
    def mask(self) -> np.ndarray:
        """Return the diagonal of A^T A in CHW layout."""

        return np.moveaxis(np.eye(3, dtype=np.float64)[self.cfa], -1, 0)

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        checked = np.asarray(rgb, dtype=np.float64)
        if checked.shape != (3,) + self.shape or not np.isfinite(checked).all():
            raise ValueError("RGB input must be finite CHW data matching the CFA")
        hwc = np.moveaxis(checked, 0, -1)
        return np.take_along_axis(hwc, self.cfa[..., None], axis=2)[..., 0]

    def adjoint(self, scalar: np.ndarray) -> np.ndarray:
        checked = np.asarray(scalar, dtype=np.float64)
        if checked.shape != self.shape or not np.isfinite(checked).all():
            raise ValueError("scalar input must be finite data matching the CFA")
        return self.mask * checked[None, ...]

    def project(self, rgb: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        """Project an RGB estimate onto the noiseless affine set A x = y."""

        checked = np.asarray(rgb, dtype=np.float64)
        if checked.shape != (3,) + self.shape or not np.isfinite(checked).all():
            raise ValueError("RGB input must be finite CHW data matching the CFA")
        observed = self.adjoint(scalar)
        mask = self.mask
        return checked * (1.0 - mask) + observed

    def soft_update(
        self, consensus: np.ndarray, scalar: np.ndarray, rho: float
    ) -> np.ndarray:
        """Solve the diagonal PnP-ADMM data subproblem exactly.

        For v = z - u, the minimizer of

            1/2 ||A x - y||^2 + rho/2 ||x - v||^2

        is v on unmeasured components and (y + rho v)/(1 + rho) on
        measured components.
        """

        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho must be finite and positive")
        checked = np.asarray(consensus, dtype=np.float64)
        if checked.shape != (3,) + self.shape or not np.isfinite(checked).all():
            raise ValueError("consensus must be finite CHW data matching the CFA")
        mask = self.mask
        observed = self.adjoint(scalar)
        return checked * (1.0 - mask) + mask * (observed + rho * checked) / (1.0 + rho)


def xtrans_operator(
    height: int, width: int, origin_x: int = 0, origin_y: int = 0
) -> CFAOperator:
    return CFAOperator(cfa_for_origin(height, width, origin_x, origin_y))


def bayer_bggr_operator(height: int, width: int) -> CFAOperator:
    """Return the BGGR phase used by SCICO's public PnP Bayer example."""

    if height <= 0 or width <= 0:
        raise ValueError("dimensions must be positive")
    cfa = np.empty((height, width), dtype=np.uint8)
    cfa[0::2, 0::2] = 2
    cfa[0::2, 1::2] = 1
    cfa[1::2, 0::2] = 1
    cfa[1::2, 1::2] = 0
    return CFAOperator(cfa)


__all__ = ("CFAOperator", "bayer_bggr_operator", "xtrans_operator")

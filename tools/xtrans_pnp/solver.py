"""Small, deterministic PnP solvers for the offline feasibility study.

These routines compute a fixed point/consensus iteration.  They deliberately
do not claim to minimize an explicit energy because the external BM3D denoiser
is not established here as the proximal map of a convex regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable

import numpy as np

from .operator import CFAOperator


Denoiser = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    seconds: float
    cfa_rms: float
    cfa_maximum: float
    primal_rms: float
    dual_rms: float
    estimate_rms_change: float


@dataclass(frozen=True)
class PnPResult:
    image: np.ndarray
    records: tuple[IterationRecord, ...]
    snapshots: dict[int, np.ndarray]


def _checked_initial(operator: CFAOperator, initial: np.ndarray) -> np.ndarray:
    checked = np.asarray(initial, dtype=np.float64)
    if checked.shape != (3,) + operator.shape or not np.isfinite(checked).all():
        raise ValueError("initial estimate must be finite CHW data matching the CFA")
    return np.ascontiguousarray(checked)


def _checked_denoised(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    checked = np.asarray(value, dtype=np.float64)
    if checked.shape != shape or not np.isfinite(checked).all():
        raise ValueError("denoiser returned invalid data")
    return np.ascontiguousarray(checked)


def _record(
    iteration: int,
    seconds: float,
    operator: CFAOperator,
    scalar: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    previous_z: np.ndarray,
) -> IterationRecord:
    cfa_difference = operator.forward(z) - scalar
    return IterationRecord(
        iteration=iteration,
        seconds=seconds,
        cfa_rms=float(np.sqrt(np.mean(cfa_difference * cfa_difference))),
        cfa_maximum=float(np.max(np.abs(cfa_difference))),
        primal_rms=float(np.sqrt(np.mean((x - z) ** 2))),
        dual_rms=float(np.sqrt(np.mean((z - previous_z) ** 2))),
        estimate_rms_change=float(np.sqrt(np.mean((z - previous_z) ** 2))),
    )


def pnp_admm(
    operator: CFAOperator,
    scalar: np.ndarray,
    initial: np.ndarray,
    denoiser: Denoiser,
    sigma: float,
    rho: float,
    iterations: int,
    *,
    exact_projection: bool,
    capture_iterations: Iterable[int] = (),
    continuation: float = 1.0,
) -> PnPResult:
    """Run PnP-ADMM with the exact diagonal X-Trans data update."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(continuation) or not (0.0 < continuation <= 1.0):
        raise ValueError("continuation must lie in (0, 1]")
    measured = np.asarray(scalar, dtype=np.float64)
    if measured.shape != operator.shape or not np.isfinite(measured).all():
        raise ValueError("measurements must be finite and match the CFA")
    z = _checked_initial(operator, initial).copy()
    if exact_projection:
        z = operator.project(z, measured)
    u = np.zeros_like(z)
    capture = set(int(value) for value in capture_iterations)
    records: list[IterationRecord] = []
    snapshots: dict[int, np.ndarray] = {}
    for index in range(1, iterations + 1):
        started = time.perf_counter()
        consensus = z - u
        x = (
            operator.project(consensus, measured)
            if exact_projection
            else operator.soft_update(consensus, measured, rho)
        )
        denoised = _checked_denoised(
            denoiser(x + u, sigma * continuation ** (index - 1)), z.shape
        )
        if exact_projection:
            denoised = operator.project(denoised, measured)
        previous_z = z
        z = denoised
        u = u + x - z
        if not np.isfinite(u).all():
            raise ValueError("PnP dual variable became non-finite")
        records.append(
            _record(index, time.perf_counter() - started, operator, measured, x, z, previous_z)
        )
        if index in capture:
            snapshots[index] = z.copy()
    return PnPResult(z, tuple(records), snapshots)


def pnp_pgm(
    operator: CFAOperator,
    scalar: np.ndarray,
    initial: np.ndarray,
    denoiser: Denoiser,
    sigma: float,
    step: float,
    iterations: int,
    *,
    exact_projection: bool,
    capture_iterations: Iterable[int] = (),
) -> PnPResult:
    """Run the simple PnP proximal-gradient control."""

    if iterations < 1 or not np.isfinite(step) or not (0.0 < step <= 2.0):
        raise ValueError("PGM requires positive iterations and step in (0, 2]")
    measured = np.asarray(scalar, dtype=np.float64)
    x = _checked_initial(operator, initial).copy()
    if exact_projection:
        x = operator.project(x, measured)
    capture = set(int(value) for value in capture_iterations)
    records: list[IterationRecord] = []
    snapshots: dict[int, np.ndarray] = {}
    for index in range(1, iterations + 1):
        started = time.perf_counter()
        previous = x
        gradient = operator.adjoint(operator.forward(x) - measured)
        x = _checked_denoised(denoiser(x - step * gradient, sigma), x.shape)
        if exact_projection:
            x = operator.project(x, measured)
        records.append(
            _record(index, time.perf_counter() - started, operator, measured, x, x, previous)
        )
        if index in capture:
            snapshots[index] = x.copy()
    return PnPResult(x, tuple(records), snapshots)


__all__ = ("IterationRecord", "PnPResult", "pnp_admm", "pnp_pgm")

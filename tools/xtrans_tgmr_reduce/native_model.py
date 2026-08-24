"""Float32 preparation, serialization, and reference inference for TGMR32-S9-q8."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_gmr.model import PhaseConditionedGMR, prepare_gmr_cache

from .model import _coarse_positions


MAGIC = b"XTGRRED1"
VERSION = 1
PATCH_SIZE = 7
PHASES = 18
COMPONENTS = 32
SUPPORT = 3
SHORTLIST = 8
NU = np.float32(3.0)
TEMPERATURE = np.float32(4.0)
TAU = 3e-4


@dataclass(frozen=True)
class NativePhase:
    observed_indices: np.ndarray
    sampled_center_channel: int
    target_channels: np.ndarray
    log_weights: np.ndarray
    means_observed: np.ndarray
    means_target: np.ndarray
    full_cholesky: np.ndarray
    full_log_determinants: np.ndarray
    target_gains: np.ndarray
    coarse_positions: np.ndarray
    coarse_cholesky: np.ndarray
    coarse_log_determinants: np.ndarray


@dataclass(frozen=True)
class NativeModel:
    phases: tuple[NativePhase, ...]


def prepare_native_model(model: PhaseConditionedGMR) -> NativeModel:
    """Quantize the validation-selected dense model and precompute S9 caches."""

    if (
        model.patch_size != PATCH_SIZE
        or model.component_count != COMPONENTS
        or model.dc_mode != "observed-rgb"
    ):
        raise ValueError("native prototype accepts only the frozen K32/7x7 model")
    cache = prepare_gmr_cache(model, TAU)
    phases = []
    for phase_index, (source, cached) in enumerate(zip(model.phases, cache.phases)):
        coarse_positions = _coarse_positions(cached, PATCH_SIZE, SUPPORT)
        coarse_factors = []
        coarse_logdet = []
        for component in range(COMPONENTS):
            covariance = source.covariances[component][
                np.ix_(coarse_positions, coarse_positions)
            ].copy()
            covariance.flat[:: coarse_positions.size + 1] += TAU * TAU
            factor = np.linalg.cholesky(covariance)
            coarse_factors.append(factor)
            coarse_logdet.append(2.0 * np.sum(np.log(np.diag(factor))))
        phases.append(NativePhase(
            observed_indices=np.asarray(cached.observed_indices, dtype=np.uint32),
            sampled_center_channel=cached.sampled_center_channel,
            target_channels=np.asarray(cached.target_channels, dtype=np.uint32),
            log_weights=np.asarray(np.log(source.weights), dtype=np.float32),
            means_observed=np.asarray(cached.means_observed, dtype=np.float32),
            means_target=np.asarray(cached.means_target, dtype=np.float32),
            # scipy.linalg.cho_factor leaves the unused triangle undefined.
            # Canonicalize it both for deterministic serialization and because
            # the native forward solve consumes an explicitly lower matrix.
            full_cholesky=np.asarray(
                [np.tril(factor) for factor in cached.cholesky], dtype=np.float32
            ),
            full_log_determinants=np.asarray(
                cached.log_determinants, dtype=np.float32
            ),
            target_gains=np.asarray(cached.target_gains, dtype=np.float32),
            coarse_positions=np.asarray(coarse_positions, dtype=np.uint32),
            coarse_cholesky=np.asarray(coarse_factors, dtype=np.float32),
            coarse_log_determinants=np.asarray(coarse_logdet, dtype=np.float32),
        ))
    return NativeModel(tuple(phases))


def _forward_quadratic(cholesky: np.ndarray, residual: np.ndarray) -> np.float32:
    solved = np.linalg.solve(cholesky, residual)
    return np.float32(np.dot(solved, solved))


def _logpdf(quadratic: np.float32, logdet: np.float32, dimension: int) -> np.float32:
    common = (
        math.lgamma(0.5 * (float(NU) + dimension))
        - math.lgamma(0.5 * float(NU))
        - 0.5 * dimension * math.log(float(NU) * math.pi)
    )
    return np.float32(
        common
        - 0.5 * float(logdet)
        - 0.5 * (float(NU) + dimension)
        * math.log1p(float(quadratic) / float(NU))
    )


def native_float32_predict(
    model: NativeModel, samples: list[PatchSample]
) -> tuple[np.ndarray, np.ndarray]:
    """Reference the exact arithmetic contract used by the C++ prototype."""

    output = np.empty((len(samples), 3), dtype=np.float32)
    shortlist_ids = np.empty((len(samples), SHORTLIST), dtype=np.int32)
    area = PATCH_SIZE * PATCH_SIZE
    center = area // 2
    for sample_index, sample in enumerate(samples):
        phase = model.phases[sample.phase]
        vector = np.asarray(sample.vector, dtype=np.float32)
        colors = phase.observed_indices // area
        color_means = np.asarray([
            np.mean(vector[phase.observed_indices[colors == channel]], dtype=np.float32)
            for channel in range(3)
        ], dtype=np.float32)
        dc = np.mean(color_means, dtype=np.float32)
        observed = vector[phase.observed_indices] - dc
        coarse = observed[phase.coarse_positions]
        scores = np.empty(COMPONENTS, dtype=np.float32)
        for component in range(COMPONENTS):
            residual = coarse - phase.means_observed[
                component, phase.coarse_positions
            ]
            quadratic = _forward_quadratic(
                phase.coarse_cholesky[component], residual
            )
            scores[component] = (
                phase.log_weights[component]
                + _logpdf(
                    quadratic, phase.coarse_log_determinants[component], SUPPORT ** 2
                )
            )
        selected = np.argsort(-scores, kind="stable")[:SHORTLIST]
        shortlist_ids[sample_index] = selected
        predictions = np.empty((SHORTLIST, 2), dtype=np.float32)
        full_scores = np.empty(SHORTLIST, dtype=np.float32)
        for slot, component in enumerate(selected):
            residual = observed - phase.means_observed[component]
            quadratic = _forward_quadratic(
                phase.full_cholesky[component], residual
            )
            full_scores[slot] = (
                phase.log_weights[component]
                + _logpdf(
                    quadratic, phase.full_log_determinants[component], area
                )
            ) / TEMPERATURE
            predictions[slot] = (
                phase.means_target[component]
                + phase.target_gains[component] @ residual
                + dc
            )
        shifted = full_scores - np.max(full_scores)
        weights = np.exp(shifted).astype(np.float32)
        weights /= np.sum(weights, dtype=np.float32)
        rgb = np.empty(3, dtype=np.float32)
        rgb[phase.target_channels] = np.sum(
            predictions * weights[:, None], axis=0, dtype=np.float32
        )
        measured_index = phase.observed_indices[
            np.flatnonzero(phase.observed_indices % area == center)[0]
        ]
        rgb[phase.sampled_center_channel] = vector[measured_index]
        output[sample_index] = rgb
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite native-reference output")
    return output, shortlist_ids


def _array_bytes(value: np.ndarray, dtype: str) -> bytes:
    return np.asarray(value, dtype=np.dtype(dtype).newbyteorder("<"), order="C").tobytes()


def serialize_native_model(model: NativeModel) -> bytes:
    header = struct.pack(
        "<8s7I", MAGIC, VERSION, PATCH_SIZE, PHASES, COMPONENTS,
        SUPPORT, SHORTLIST, 0,
    )
    rows = [header]
    for phase in model.phases:
        rows.extend((
            _array_bytes(phase.observed_indices, "u4"),
            struct.pack("<I2I", phase.sampled_center_channel, *phase.target_channels),
            _array_bytes(phase.log_weights, "f4"),
            _array_bytes(phase.means_observed, "f4"),
            _array_bytes(phase.means_target, "f4"),
            _array_bytes(phase.full_cholesky, "f4"),
            _array_bytes(phase.full_log_determinants, "f4"),
            _array_bytes(phase.target_gains, "f4"),
            _array_bytes(phase.coarse_positions, "u4"),
            _array_bytes(phase.coarse_cholesky, "f4"),
            _array_bytes(phase.coarse_log_determinants, "f4"),
        ))
    return b"".join(rows)


def write_native_model(path: Path, model: NativeModel) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(serialize_native_model(model))


__all__ = (
    "NativeModel",
    "native_float32_predict",
    "prepare_native_model",
    "serialize_native_model",
    "write_native_model",
)

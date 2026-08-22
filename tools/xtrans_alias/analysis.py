"""Exact and empirical analysis of the periodic X-Trans sampling operator.

Frequencies use cycles/pixel in the half-open interval [-0.5, 0.5).  The
periodic X-Trans cell has lattice generators (6, 0) and (3, 3), so its
reciprocal alias group contains 18 carrier offsets.  All Fourier transforms
use NumPy's negative-exponent convention and are normalized by sample count.

This module characterizes sampling only.  It contains no interpolation or
reconstruction algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CANONICAL_XTRANS = np.asarray(
    (
        (1, 2, 1, 1, 0, 1),
        (0, 1, 0, 2, 1, 2),
        (1, 2, 1, 1, 0, 1),
        (1, 0, 1, 1, 2, 1),
        (2, 1, 2, 0, 1, 0),
        (1, 0, 1, 1, 2, 1),
    ),
    dtype=np.uint8,
)

COLOR_NAMES = ("red", "green", "blue")

# The 18 reciprocal-lattice representatives on the 6x6 DFT grid.  The other
# 18 bins are absent because the cell is also periodic under translation by
# (3, 3).  Five of these 18 allowed carriers have zero coefficient for every
# color, leaving the 13 nonzero components reported by Rafinazari and Dubois.
CARRIER_INDICES = tuple(
    (kx, ky)
    for ky in range(6)
    for kx in range(6)
    if (kx + ky) % 2 == 0
)

DIRECTIONS: Mapping[str, np.ndarray] = {
    "luma": np.asarray((1.0, 1.0, 1.0), dtype=np.float64) / math.sqrt(3.0),
    "c1_r_minus_b": np.asarray((1.0, 0.0, -1.0), dtype=np.float64)
    / math.sqrt(2.0),
    "c2_green_opponent": np.asarray((1.0, -2.0, 1.0), dtype=np.float64)
    / math.sqrt(6.0),
    "red": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    "green": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    "blue": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    "red_green": np.asarray((1.0, -1.0, 0.0), dtype=np.float64)
    / math.sqrt(2.0),
    "blue_green": np.asarray((0.0, -1.0, 1.0), dtype=np.float64)
    / math.sqrt(2.0),
    "red_blue": np.asarray((1.0, 0.0, -1.0), dtype=np.float64)
    / math.sqrt(2.0),
}

ORTHONORMAL_DIRECTION_NAMES = (
    "luma",
    "c1_r_minus_b",
    "c2_green_opponent",
)

PHASES = (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)


@dataclass(frozen=True)
class PhaseConfusion:
    values: np.ndarray
    strongest: dict[str, object]
    minimum: float
    maximum: float
    mean: float


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centered_frequency(index: int, size: int) -> float:
    wrapped = index if index < (size + 1) // 2 else index - size
    return wrapped / size


def mask_spectra(cfa: np.ndarray = CANONICAL_XTRANS) -> np.ndarray:
    """Return normalized spectra with shape (3, 6, 6)."""

    checked = np.asarray(cfa, dtype=np.uint8)
    if checked.shape != (6, 6):
        raise ValueError("X-Trans CFA must be 6x6")
    if set(int(value) for value in checked.flat) != {0, 1, 2}:
        raise ValueError("X-Trans CFA must contain red, green, and blue")
    return np.stack(
        [np.fft.fft2(checked == color) / 36.0 for color in range(3)], axis=0
    )


def exact_sixth_root_coefficient(value: complex) -> dict[str, int]:
    """Represent a mask coefficient as (a + b sqrt(3) i) / 36."""

    real = int(round(float(value.real) * 36.0))
    imag_sqrt3 = int(round(float(value.imag) * 36.0 / math.sqrt(3.0)))
    reconstructed = (real + 1j * imag_sqrt3 * math.sqrt(3.0)) / 36.0
    if abs(reconstructed - value) > 2e-14:
        raise ValueError("coefficient is not on the expected sixth-root lattice")
    return {
        "denominator": 36,
        "imag_sqrt3_numerator": imag_sqrt3,
        "real_numerator": real,
    }


def exact_carrier_table(cfa: np.ndarray = CANONICAL_XTRANS) -> list[dict[str, object]]:
    spectra = mask_spectra(cfa)
    rows: list[dict[str, object]] = []
    for kx, ky in CARRIER_INDICES:
        coefficients = {
            COLOR_NAMES[color]: exact_sixth_root_coefficient(
                spectra[color, ky, kx]
            )
            for color in range(3)
        }
        rows.append(
            {
                "coefficient": coefficients,
                "frequency": [centered_frequency(kx, 6), centered_frequency(ky, 6)],
                "index": [kx, ky],
                "magnitude": [
                    float(abs(spectra[color, ky, kx])) for color in range(3)
                ],
                "nonzero": any(
                    abs(spectra[color, ky, kx]) > 1e-14 for color in range(3)
                ),
            }
        )
    return rows


def carrier_signature(
    carrier: tuple[int, int],
    color_direction: Sequence[float],
    spectra: np.ndarray | None = None,
) -> np.ndarray:
    """Observed 18-bin signature of one complex RGB exponential.

    ``carrier`` is the source-frequency offset within one alias family.  A
    continuous base frequency translates every output bin equally and does not
    alter the signature or any translation-invariant confusion metric.
    """

    if spectra is None:
        spectra = mask_spectra()
    direction = np.asarray(color_direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("color direction must have three values")
    sx, sy = carrier
    return np.asarray(
        [
            np.dot(spectra[:, (oy - sy) % 6, (ox - sx) % 6], direction)
            for ox, oy in CARRIER_INDICES
        ],
        dtype=np.complex128,
    )


def normalized_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("similarity is undefined for a zero observation")
    return float(abs(np.vdot(left, right)) / (left_norm * right_norm))


def complex_confusion_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_name in ORTHONORMAL_DIRECTION_NAMES:
        source = carrier_signature((0, 0), DIRECTIONS[source_name])
        for competitor_name in ORTHONORMAL_DIRECTION_NAMES:
            strongest = (-1.0, (0, 0))
            for carrier in CARRIER_INDICES:
                if source_name == competitor_name and carrier == (0, 0):
                    continue
                candidate = carrier_signature(carrier, DIRECTIONS[competitor_name])
                score = normalized_similarity(source, candidate)
                if score > strongest[0]:
                    strongest = (score, carrier)
            rows.append(
                {
                    "competitor": competitor_name,
                    "competitor_carrier_index": list(strongest[1]),
                    "competitor_frequency_displacement": [
                        centered_frequency(strongest[1][0], 6),
                        centered_frequency(strongest[1][1], 6),
                    ],
                    "similarity": strongest[0],
                    "source": source_name,
                }
            )
    return rows


def bayer_comparison() -> dict[str, object]:
    """Return the same compact linear diagnostics for a 2x2 RGGB Bayer CFA."""

    cfa = np.asarray(((0, 1), (1, 2)), dtype=np.uint8)
    carriers = tuple((kx, ky) for ky in range(2) for kx in range(2))
    spectra = np.stack(
        [np.fft.fft2(cfa == color) / 4.0 for color in range(3)], axis=0
    )

    def signature(carrier: tuple[int, int], direction: np.ndarray) -> np.ndarray:
        sx, sy = carrier
        return np.asarray(
            [
                np.dot(spectra[:, (oy - sy) % 2, (ox - sx) % 2], direction)
                for ox, oy in carriers
            ],
            dtype=np.complex128,
        )

    identity = np.eye(3, dtype=np.float64)
    full = np.column_stack(
        [signature(carrier, identity[color]) for carrier in carriers for color in range(3)]
    )
    known = np.column_stack([signature((0, 0), identity[color]) for color in range(3)])
    confusion = []
    for source_name in ORTHONORMAL_DIRECTION_NAMES:
        source = signature((0, 0), DIRECTIONS[source_name])
        for competitor_name in ORTHONORMAL_DIRECTION_NAMES:
            strongest = (-1.0, (0, 0))
            for carrier in carriers:
                if source_name == competitor_name and carrier == (0, 0):
                    continue
                score = normalized_similarity(
                    source, signature(carrier, DIRECTIONS[competitor_name])
                )
                if score > strongest[0]:
                    strongest = (score, carrier)
            confusion.append(
                {
                    "competitor": competitor_name,
                    "competitor_carrier_index": list(strongest[1]),
                    "competitor_frequency_displacement": [
                        centered_frequency(strongest[1][0], 2),
                        centered_frequency(strongest[1][1], 2),
                    ],
                    "similarity": strongest[0],
                    "source": source_name,
                }
            )
    return {
        "cfa": cfa.tolist(),
        "complex_confusion": confusion,
        "full_rgb_operator": singular_summary(full),
        "known_frequency_rgb_operator": singular_summary(known),
    }


def full_alias_matrix() -> np.ndarray:
    columns = []
    identity = np.eye(3, dtype=np.float64)
    for carrier in CARRIER_INDICES:
        for color in range(3):
            columns.append(carrier_signature(carrier, identity[color]))
    return np.column_stack(columns)


def known_frequency_color_matrix() -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)
    return np.column_stack(
        [carrier_signature((0, 0), identity[color]) for color in range(3)]
    )


def singular_summary(matrix: np.ndarray) -> dict[str, object]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    nullity = int(matrix.shape[1] - rank)
    return {
        "columns": int(matrix.shape[1]),
        "condition_nonzero": float(singular[0] / singular[rank - 1]),
        "nullity": nullity,
        "rank": rank,
        "rows": int(matrix.shape[0]),
        "singular_values": [float(value) for value in singular],
    }


def sparse_luma_null_example() -> dict[str, object]:
    """Find a small chroma superposition that exactly mimics one luma mode."""

    target = carrier_signature((0, 0), DIRECTIONS["luma"])
    names = ("c1_r_minus_b", "c2_green_opponent")
    descriptors = [
        (carrier, name) for carrier in CARRIER_INDICES for name in names
    ]
    candidates = np.column_stack(
        [carrier_signature(carrier, DIRECTIONS[name]) for carrier, name in descriptors]
    )
    target_norm = float(np.linalg.norm(target))
    residual = target.copy()
    selected: list[int] = []
    coefficients = np.empty(0, dtype=np.complex128)
    for _ in range(candidates.shape[0]):
        correlations = np.abs(candidates.conj().T @ residual) / np.linalg.norm(
            candidates, axis=0
        )
        correlations[selected] = -1.0
        selected.append(int(np.argmax(correlations)))
        coefficients = np.linalg.lstsq(
            candidates[:, selected], target, rcond=None
        )[0]
        residual = target - candidates[:, selected] @ coefficients
        if np.linalg.norm(residual) <= 1e-12 * target_norm:
            break
    terms = []
    for selected_index, coefficient in zip(selected, coefficients):
        carrier, name = descriptors[selected_index]
        terms.append(
            {
                "carrier_index": list(carrier),
                "coefficient_imag": float(coefficient.imag),
                "coefficient_magnitude": float(abs(coefficient)),
                "coefficient_real": float(coefficient.real),
                "direction": name,
                "frequency_displacement": [
                    centered_frequency(carrier[0], 6),
                    centered_frequency(carrier[1], 6),
                ],
            }
        )
    return {
        "relative_residual": float(np.linalg.norm(residual) / target_norm),
        "target": {"carrier_index": [0, 0], "direction": "luma"},
        "terms": terms,
    }


def direct_complex_replica_check(
    size: int = 96,
    source_bin: tuple[int, int] = (7, 11),
) -> dict[str, object]:
    if size % 6:
        raise ValueError("complex replica image size must be divisible by six")
    y, x = np.mgrid[:size, :size]
    phase = 2.0 * math.pi * (
        source_bin[0] * x / size + source_bin[1] * y / size
    )
    spectra = mask_spectra()
    cases = {}
    for name in ORTHONORMAL_DIRECTION_NAMES:
        direction = DIRECTIONS[name]
        rgb = direction[:, None, None] * np.exp(1j * phase)[None, :, :]
        observed = np.take_along_axis(
            np.moveaxis(rgb, 0, -1),
            np.tile(CANONICAL_XTRANS, (size // 6, size // 6))[..., None],
            axis=2,
        )[..., 0]
        actual = np.fft.fft2(observed) / (size * size)
        predicted = np.zeros_like(actual)
        peaks = []
        for kx, ky in CARRIER_INDICES:
            ox = (source_bin[0] + kx * size // 6) % size
            oy = (source_bin[1] + ky * size // 6) % size
            value = np.dot(spectra[:, ky, kx], direction)
            predicted[oy, ox] = value
            if abs(value) > 1e-13:
                peaks.append(
                    {
                        "frequency": [
                            centered_frequency(ox, size),
                            centered_frequency(oy, size),
                        ],
                        "magnitude": float(abs(value)),
                        "phase": float(np.angle(value)),
                        "relative_carrier_index": [kx, ky],
                    }
                )
        cases[name] = {
            "maximum_complex_error": float(np.max(np.abs(actual - predicted))),
            "peak_count": len(peaks),
            "peaks": peaks,
        }
    return {
        "cases": cases,
        "image_size": [size, size],
        "source_bin": list(source_bin),
        "source_frequency": [
            centered_frequency(source_bin[0], size),
            centered_frequency(source_bin[1], size),
        ],
    }


class _RealInnerProducts:
    def __init__(self, size: int):
        if size % 6:
            raise ValueError("real-sinusoid grid must be divisible by six")
        self.size = size
        tiled = np.tile(CANONICAL_XTRANS, (size // 6, size // 6))
        self.transforms = {}
        for left_name in ORTHONORMAL_DIRECTION_NAMES:
            for right_name in ORTHONORMAL_DIRECTION_NAMES:
                weights = DIRECTIONS[left_name][tiled] * DIRECTIONS[right_name][tiled]
                self.transforms[(left_name, right_name)] = np.fft.fft2(weights)

    def _weighted_sum(
        self, left_name: str, right_name: str, kx: int, ky: int
    ) -> complex:
        transform = self.transforms[(left_name, right_name)]
        return complex(transform[(-ky) % self.size, (-kx) % self.size])

    def inner(
        self,
        left_name: str,
        left_frequency: tuple[int, int],
        left_phase: float,
        right_name: str,
        right_frequency: tuple[int, int],
        right_phase: float,
    ) -> float:
        difference = self._weighted_sum(
            left_name,
            right_name,
            left_frequency[0] - right_frequency[0],
            left_frequency[1] - right_frequency[1],
        )
        total = self._weighted_sum(
            left_name,
            right_name,
            left_frequency[0] + right_frequency[0],
            left_frequency[1] + right_frequency[1],
        )
        value = 0.5 * np.real(
            np.exp(1j * (left_phase - right_phase)) * difference
            + np.exp(1j * (left_phase + right_phase)) * total
        )
        return float(value)

    def norm(self, name: str, frequency: tuple[int, int], phase: float) -> float:
        squared = self.inner(name, frequency, phase, name, frequency, phase)
        return math.sqrt(max(0.0, squared))


def phase_sensitive_confusion(
    source_name: str,
    competitor_names: Sequence[str],
    size: int = 24,
) -> PhaseConfusion:
    """Worst real-cosine ambiguity across the requested four phases.

    Only frequencies related by a carrier through their sum or difference can
    have a nonzero full-period inner product, so the candidate set is exact,
    not a heuristic truncation.
    """

    products = _RealInnerProducts(size)
    carrier_step = size // 6
    carrier_bins = [
        (kx * carrier_step, ky * carrier_step) for kx, ky in CARRIER_INDICES
    ]
    values = np.zeros((size, size), dtype=np.float64)
    strongest: dict[str, object] = {"similarity": -1.0}
    for source_y in range(size):
        for source_x in range(size):
            candidate_frequencies = set()
            for carrier_x, carrier_y in carrier_bins:
                candidate_frequencies.add(
                    (
                        (source_x + carrier_x) % size,
                        (source_y + carrier_y) % size,
                    )
                )
                candidate_frequencies.add(
                    (
                        (-source_x + carrier_x) % size,
                        (-source_y + carrier_y) % size,
                    )
                )
            best = -1.0
            best_descriptor: dict[str, object] = {}
            for source_phase_index, source_phase in enumerate(PHASES):
                source_norm = products.norm(
                    source_name, (source_x, source_y), source_phase
                )
                if source_norm <= 1e-12:
                    continue
                for competitor_name in competitor_names:
                    for competitor_frequency in sorted(candidate_frequencies):
                        for competitor_phase_index, competitor_phase in enumerate(PHASES):
                            competitor_norm = products.norm(
                                competitor_name,
                                competitor_frequency,
                                competitor_phase,
                            )
                            if competitor_norm <= 1e-12:
                                continue
                            score = abs(
                                products.inner(
                                    source_name,
                                    (source_x, source_y),
                                    source_phase,
                                    competitor_name,
                                    competitor_frequency,
                                    competitor_phase,
                                )
                            ) / (source_norm * competitor_norm)
                            if score > best:
                                best = score
                                best_descriptor = {
                                    "competitor": competitor_name,
                                    "competitor_bin": list(competitor_frequency),
                                    "competitor_frequency": [
                                        centered_frequency(competitor_frequency[0], size),
                                        centered_frequency(competitor_frequency[1], size),
                                    ],
                                    "competitor_phase_index": competitor_phase_index,
                                    "source_bin": [source_x, source_y],
                                    "source_frequency": [
                                        centered_frequency(source_x, size),
                                        centered_frequency(source_y, size),
                                    ],
                                    "source_phase_index": source_phase_index,
                                }
            if best < 0.0:
                raise RuntimeError("no nonzero phase comparison was available")
            values[source_y, source_x] = best
            if best > float(strongest["similarity"]):
                strongest = {"similarity": best, **best_descriptor}
    return PhaseConfusion(
        values=values,
        strongest=strongest,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
    )


def cfa_variants() -> list[np.ndarray]:
    variants: dict[tuple[int, ...], np.ndarray] = {}
    for turns in range(4):
        rotated = np.rot90(CANONICAL_XTRANS, turns)
        for oriented in (rotated, np.fliplr(rotated)):
            for oy in range(6):
                for ox in range(6):
                    shifted = np.roll(oriented, (oy, ox), axis=(0, 1))
                    variants[tuple(int(value) for value in shifted.flat)] = shifted
    return [variants[key] for key in sorted(variants)]


def variant_invariance() -> dict[str, object]:
    reference = mask_spectra()
    reference_magnitudes = [
        sorted(float(abs(reference[color, ky, kx])) for kx, ky in CARRIER_INDICES)
        for color in range(3)
    ]
    maximum_magnitude_error = 0.0
    maximum_singular_error = 0.0
    reference_singular = np.linalg.svd(full_alias_matrix(), compute_uv=False)
    variants = cfa_variants()
    for variant in variants:
        spectra = mask_spectra(variant)
        for color in range(3):
            magnitudes = sorted(
                float(abs(spectra[color, ky, kx]))
                for kx, ky in CARRIER_INDICES
            )
            maximum_magnitude_error = max(
                maximum_magnitude_error,
                max(abs(a - b) for a, b in zip(magnitudes, reference_magnitudes[color])),
            )
        columns = []
        for carrier in CARRIER_INDICES:
            for color in range(3):
                columns.append(carrier_signature(carrier, np.eye(3)[color], spectra))
        singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
        maximum_singular_error = max(
            maximum_singular_error,
            float(np.max(np.abs(singular - reference_singular))),
        )
    return {
        "maximum_mask_magnitude_error": maximum_magnitude_error,
        "maximum_singular_value_error": maximum_singular_error,
        "variant_count": len(variants),
    }


def _alias_families(size: int) -> list[tuple[tuple[int, int], ...]]:
    if size % 6:
        raise ValueError("alias-family size must be divisible by six")
    step = size // 6
    carriers = [(kx * step, ky * step) for kx, ky in CARRIER_INDICES]
    families: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for ky in range(size):
        for kx in range(size):
            family = tuple(
                sorted(((kx + hx) % size, (ky + hy) % size) for hx, hy in carriers)
            )
            families.setdefault(family[0], family)
    return [families[key] for key in sorted(families)]


def alias_families(size: int) -> list[tuple[tuple[int, int], ...]]:
    """Return the validated 18-frequency partition for an FFT grid.

    This public wrapper lets later sampling experiments reuse the exact family
    definition without copying the reciprocal-lattice implementation.
    """

    return _alias_families(size)


def active_family_analysis(
    rgb: np.ndarray,
    relative_threshold: float = 1e-8,
) -> dict[str, object]:
    """Measure source energy residing in overloaded alias families.

    This analysis uses exact ground-truth RGB coefficients.  A family is
    marked rank deficient only when its active source columns outnumber or
    linearly exceed the observable rank; it does not infer ambiguity from a
    demosaicer's output.
    """

    checked = np.asarray(rgb, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("RGB input must have shape (3, height, width)")
    if checked.shape[1] != checked.shape[2] or checked.shape[1] % 6:
        raise ValueError("edge-analysis inputs must be square and divisible by six")
    # Constant color offsets are directly represented at DC and would dominate
    # the energy percentages without describing edge/frequency confusion.
    checked = checked - np.mean(checked, axis=(1, 2), keepdims=True)
    size = checked.shape[1]
    basis = np.stack([DIRECTIONS[name] for name in ORTHONORMAL_DIRECTION_NAMES])
    coefficients = np.einsum(
        "dc,cyx->dyx", basis, np.fft.fft2(checked, axes=(-2, -1)) / (size * size)
    )
    threshold = float(np.max(np.abs(coefficients))) * relative_threshold
    total_energy = float(np.sum(np.abs(coefficients) ** 2))
    deficient_energy = 0.0
    poorly_conditioned_energy = 0.0
    maximum_active = 0
    maximum_finite_condition = 0.0
    active_family_count = 0
    step = size // 6
    for family in _alias_families(size):
        base_x, base_y = family[0]
        active = []
        family_energy = 0.0
        for kx, ky in family:
            relative = (((kx - base_x) % size) // step, ((ky - base_y) % size) // step)
            if relative not in CARRIER_INDICES:
                raise AssertionError("invalid reciprocal-lattice family")
            for direction_index, direction_name in enumerate(ORTHONORMAL_DIRECTION_NAMES):
                coefficient = coefficients[direction_index, ky, kx]
                if abs(coefficient) > threshold:
                    active.append((relative, direction_name))
                    family_energy += float(abs(coefficient) ** 2)
        if not active:
            continue
        active_family_count += 1
        matrix = np.column_stack(
            [carrier_signature(carrier, DIRECTIONS[name]) for carrier, name in active]
        )
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        maximum_active = max(maximum_active, len(active))
        if rank < len(active):
            deficient_energy += family_energy
            poorly_conditioned_energy += family_energy
        else:
            condition = float(singular[0] / singular[-1])
            maximum_finite_condition = max(maximum_finite_condition, condition)
            if condition > 10.0:
                poorly_conditioned_energy += family_energy
    direction_energy = np.sum(np.abs(coefficients) ** 2, axis=(1, 2))
    return {
        "active_family_count": active_family_count,
        "direction_energy_fraction": {
            name: float(direction_energy[index] / total_energy)
            for index, name in enumerate(ORTHONORMAL_DIRECTION_NAMES)
        },
        "maximum_active_columns": maximum_active,
        "maximum_finite_condition": maximum_finite_condition,
        "poorly_conditioned_energy_fraction": poorly_conditioned_energy / total_energy,
        "rank_deficient_energy_fraction": deficient_energy / total_energy,
        "relative_activity_threshold": relative_threshold,
        "rgb_channel_means_removed": True,
    }


def synthetic_edge_cases(size: int = 96) -> dict[str, np.ndarray]:
    y, x = np.mgrid[:size, :size]
    vertical = x >= size // 2
    horizontal = y >= size // 2
    diagonal = x + y >= size
    antidiagonal = x >= y
    result = {}
    for name, mask in (
        ("vertical_luma", vertical),
        ("horizontal_luma", horizontal),
        ("diagonal_luma", diagonal),
        ("antidiagonal_luma", antidiagonal),
    ):
        value = 0.2 + 0.6 * mask
        result[name] = np.stack((value, value, value))
    result["vertical_red_green"] = np.stack(
        (0.2 + 0.7 * vertical, 0.2 + 0.7 * ~vertical, np.full_like(x, 0.2))
    )
    result["horizontal_red_green"] = np.stack(
        (0.2 + 0.7 * horizontal, 0.2 + 0.7 * ~horizontal, np.full_like(x, 0.2))
    )
    result["diagonal_red_green"] = np.stack(
        (0.2 + 0.7 * diagonal, 0.2 + 0.7 * ~diagonal, np.full_like(x, 0.2))
    )
    result["antidiagonal_red_green"] = np.stack(
        (
            0.2 + 0.7 * antidiagonal,
            0.2 + 0.7 * ~antidiagonal,
            np.full_like(x, 0.2),
        )
    )
    result["saturated_red_gray"] = np.stack(
        (0.5 + 0.5 * vertical, 0.5 - 0.5 * vertical, 0.5 - 0.5 * vertical)
    )
    result["saturated_blue_gray"] = np.stack(
        (0.5 - 0.5 * vertical, 0.5 - 0.5 * vertical, 0.5 + 0.5 * vertical)
    )
    # Reproduce the analytical DNG scene used for the A/B/C PSNR table.
    benchmark = np.zeros((3, size, size), dtype=np.float64)
    benchmark[0, :, : size // 2] = 1.0
    benchmark[1, : size // 2, size // 2 :] = 1.0
    benchmark[2, size // 2 :, size // 2 :] = 1.0
    result["abc_saturated_edges"] = benchmark
    # Reproduce the phase-opposed chirp from benchmark_xtrans_rafinazari.py.
    phase = 2.0 * math.pi * (
        x * (0.01 + 0.44 * y / max(1, size - 1)) + 0.17 * y
    )
    result["abc_frequency_sweep"] = np.stack(
        (
            0.5 + 0.45 * np.sin(phase),
            0.5 + 0.45 * np.sin(phase + 2.0 * math.pi / 3.0),
            0.5 + 0.45 * np.sin(phase + 4.0 * math.pi / 3.0),
        )
    )
    return result


def edge_case_analysis() -> dict[str, object]:
    return {
        name: active_family_analysis(rgb)
        for name, rgb in synthetic_edge_cases().items()
    }


def multifrequency_linearity(size: int = 96) -> dict[str, float]:
    y, x = np.mgrid[:size, :size]
    tiled_cfa = np.tile(CANONICAL_XTRANS, (size // 6, size // 6))

    def component(name: str, kx: int, ky: int, phase: float) -> np.ndarray:
        wave = np.cos(2.0 * math.pi * (kx * x + ky * y) / size + phase)
        rgb = DIRECTIONS[name][:, None, None] * wave[None, :, :]
        return np.take_along_axis(
            np.moveaxis(rgb, 0, -1), tiled_cfa[..., None], axis=2
        )[..., 0]

    first = component("c1_r_minus_b", 7, 11, 0.31)
    second = component("c2_green_opponent", 19, -5, -0.47)
    combined_direct = first + second
    combined_fft = np.fft.fft2(combined_direct)
    separate_fft = np.fft.fft2(first) + np.fft.fft2(second)
    return {
        "fft_maximum_error": float(np.max(np.abs(combined_fft - separate_fft))),
        "sample_maximum_error": float(np.max(np.abs(combined_direct - (first + second)))),
    }


def build_characterization() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    carrier_rows = exact_carrier_table()
    full = singular_summary(full_alias_matrix())
    full["zero_singular_values_in_domain"] = full["nullity"]
    known = singular_summary(known_frequency_color_matrix())
    luma_chroma = phase_sensitive_confusion(
        "luma", ("c1_r_minus_b", "c2_green_opponent")
    )
    chroma_chroma = phase_sensitive_confusion(
        "c1_r_minus_b", ("c2_green_opponent",)
    )
    edge_cases = edge_case_analysis()
    payload: dict[str, object] = {
        "alias_family": {
            "carrier_count": 18,
            "full_rgb_operator": full,
            "known_frequency_rgb_operator": known,
            "sparse_exact_null_example": sparse_luma_null_example(),
        },
        "cfa": {
            "cell": CANONICAL_XTRANS.tolist(),
            "counts_rgb": [
                int(np.count_nonzero(CANONICAL_XTRANS == color))
                for color in range(3)
            ],
            "lattice_generators": [[6, 0], [3, 3]],
        },
        "bayer_context": bayer_comparison(),
        "complex_confusion": complex_confusion_table(),
        "edge_cases": edge_cases,
        "format": "rawtherapee-xtrans-alias-characterization-v1",
        "frequency_convention": {
            "domain": "[-0.5,0.5) cycles/pixel on each axis",
            "fft": "negative-exponent DFT normalized by sample count",
            "phase_origin": "upper-left pixel center x=0,y=0",
        },
        "mask_carriers": carrier_rows,
        "multifrequency_linearity": multifrequency_linearity(),
        "paper_validation": {
            "citation": "Rafinazari and Dubois, IEEE ICIP 2014, DOI 10.1109/ICIP.2014.7025132",
            "nonzero_component_count": sum(row["nonzero"] for row in carrier_rows),
            "paper_nonzero_component_count": 13,
            "zero_carrier_indices": [
                row["index"] for row in carrier_rows if not row["nonzero"]
            ],
        },
        "phase_sensitive_real_confusion": {
            "chroma_to_chroma": {
                "maximum": chroma_chroma.maximum,
                "mean": chroma_chroma.mean,
                "minimum": chroma_chroma.minimum,
                "strongest": chroma_chroma.strongest,
            },
            "grid_size": 24,
            "luma_to_chroma": {
                "maximum": luma_chroma.maximum,
                "mean": luma_chroma.mean,
                "minimum": luma_chroma.minimum,
                "strongest": luma_chroma.strongest,
            },
            "phases": list(PHASES),
        },
        "single_frequency_replica_check": direct_complex_replica_check(),
        "variant_invariance": variant_invariance(),
    }
    maps = {
        "chroma_chroma": chroma_chroma.values,
        "luma_chroma": luma_chroma.values,
    }
    return payload, maps

"""Sparse, confidence-aware recovery of the validated X-Trans alias model.

The routines in this module operate on exact 18-bin Fourier alias families.
They are research diagnostics, not a demosaicing implementation.  All source
columns are expressed in the orthonormal L/C1/C2 basis established by
``tools.xtrans_alias.analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from tools.xtrans_alias.analysis import (
    CANONICAL_XTRANS,
    CARRIER_INDICES,
    DIRECTIONS,
    ORTHONORMAL_DIRECTION_NAMES,
    alias_families,
    carrier_signature,
    centered_frequency,
)


MODEL_FORMAT = "rawtherapee-xtrans-sparse-alias-recovery-v1"
BASIS_NAMES = ORTHONORMAL_DIRECTION_NAMES
BASIS = np.stack([DIRECTIONS[name] for name in BASIS_NAMES])
DESCRIPTORS = tuple(
    (carrier, name) for carrier in CARRIER_INDICES for name in BASIS_NAMES
)


@dataclass(frozen=True)
class LeastSquaresResult:
    coefficients: np.ndarray
    condition: float
    rank: int
    relative_residual: float
    residual: np.ndarray
    sigma_min: float
    singular_values: np.ndarray


@dataclass(frozen=True)
class PursuitResult:
    coefficients: np.ndarray
    initial_score_gap: float
    relative_residual: float
    residual_history: tuple[float, ...]
    support: tuple[int, ...]


@dataclass(frozen=True)
class GroupPursuitResult:
    coefficients: np.ndarray
    relative_residual: float
    residual_history: tuple[float, ...]
    support: tuple[int, ...]


def alias_dictionary() -> np.ndarray:
    """Return the 18x54 L/C1/C2 constellation dictionary."""

    return np.column_stack(
        [carrier_signature(carrier, DIRECTIONS[name]) for carrier, name in DESCRIPTORS]
    )


def descriptor(index: int) -> dict[str, object]:
    carrier, name = DESCRIPTORS[index]
    return {
        "carrier_index": list(carrier),
        "direction": name,
        "frequency_displacement": [
            centered_frequency(carrier[0], 6),
            centered_frequency(carrier[1], 6),
        ],
        "index": index,
    }


def _checked_support(support: Iterable[int], column_count: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in support)
    if len(result) != len(set(result)):
        raise ValueError("support contains duplicate columns")
    if any(value < 0 or value >= column_count for value in result):
        raise ValueError("support column is out of range")
    return result


def least_squares_support(
    observation: np.ndarray,
    support: Iterable[int],
    dictionary: np.ndarray | None = None,
) -> LeastSquaresResult:
    """Fit one proposed support and report its numerical observability."""

    atoms = alias_dictionary() if dictionary is None else np.asarray(dictionary)
    y = np.asarray(observation, dtype=np.complex128)
    if atoms.ndim != 2 or y.shape != (atoms.shape[0],):
        raise ValueError("observation and dictionary dimensions do not match")
    checked = _checked_support(support, atoms.shape[1])
    norm_y = float(np.linalg.norm(y))
    if not checked:
        residual = y.copy()
        return LeastSquaresResult(
            coefficients=np.empty(0, dtype=np.complex128),
            condition=1.0,
            rank=0,
            relative_residual=0.0 if norm_y == 0.0 else 1.0,
            residual=residual,
            sigma_min=0.0,
            singular_values=np.empty(0, dtype=np.float64),
        )
    selected = atoms[:, checked]
    singular = np.linalg.svd(selected, compute_uv=False)
    tolerance = max(selected.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    coefficients = np.linalg.lstsq(selected, y, rcond=None)[0]
    residual = y - selected @ coefficients
    relative = float(np.linalg.norm(residual) / norm_y) if norm_y else 0.0
    condition = math.inf if rank < len(checked) else float(singular[0] / singular[-1])
    return LeastSquaresResult(
        coefficients=coefficients,
        condition=condition,
        rank=rank,
        relative_residual=relative,
        residual=residual,
        sigma_min=float(singular[-1]),
        singular_values=singular,
    )


def _candidate_scores(
    atoms: np.ndarray,
    residual: np.ndarray,
    mode: str,
) -> np.ndarray:
    norm_residual = float(np.linalg.norm(residual))
    if norm_residual == 0.0:
        return np.zeros(atoms.shape[1], dtype=np.float64)
    norms = np.linalg.norm(atoms, axis=0)
    if mode == "replica":
        return np.abs(atoms.conj().T @ residual) / (norms * norm_residual)
    if mode == "single-peak":
        peaks = np.argmax(np.abs(atoms), axis=0)
        column_indices = np.arange(atoms.shape[1])
        numerators = np.abs(
            np.conj(atoms[peaks, column_indices]) * residual[peaks]
        )
        denominators = np.abs(atoms[peaks, column_indices]) * norm_residual
        return np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators != 0.0,
        )
    raise ValueError(f"unknown pursuit score mode: {mode}")


def orthogonal_matching_pursuit(
    observation: np.ndarray,
    max_atoms: int,
    *,
    dictionary: np.ndarray | None = None,
    mode: str = "replica",
    residual_threshold: float = 0.0,
    improvement_threshold: float = 0.0,
    forbidden: Iterable[int] = (),
) -> PursuitResult:
    """Run OMP with either full-constellation or single-peak scoring."""

    atoms = alias_dictionary() if dictionary is None else np.asarray(dictionary)
    y = np.asarray(observation, dtype=np.complex128)
    if atoms.ndim != 2 or y.shape != (atoms.shape[0],):
        raise ValueError("observation and dictionary dimensions do not match")
    if max_atoms < 0 or max_atoms > atoms.shape[1]:
        raise ValueError("invalid maximum atom count")
    blocked = set(_checked_support(forbidden, atoms.shape[1]))
    selected: list[int] = []
    residual = y.copy()
    norm_y = float(np.linalg.norm(y))
    relative = 0.0 if norm_y == 0.0 else 1.0
    history = [relative]
    coefficients = np.empty(0, dtype=np.complex128)
    first_gap = 0.0
    for iteration in range(max_atoms):
        scores = _candidate_scores(atoms, residual, mode)
        if selected:
            scores[selected] = -1.0
        if blocked:
            scores[list(blocked)] = -1.0
        ordering = np.argsort(scores, kind="stable")
        best = int(ordering[-1])
        if scores[best] < 0.0:
            break
        if iteration == 0:
            second = float(scores[ordering[-2]]) if len(ordering) > 1 else 0.0
            first_gap = float(scores[best] - second)
        selected.append(best)
        fit = least_squares_support(y, selected, atoms)
        new_relative = fit.relative_residual
        improvement = history[-1] - new_relative
        if new_relative > history[-1] + 2e-14:
            raise ArithmeticError("OMP residual increased after refitting")
        residual = fit.residual
        coefficients = fit.coefficients
        history.append(new_relative)
        relative = new_relative
        if relative <= residual_threshold:
            break
        if improvement <= improvement_threshold:
            break
    return PursuitResult(
        coefficients=coefficients,
        initial_score_gap=first_gap,
        relative_residual=relative,
        residual_history=tuple(history),
        support=tuple(selected),
    )


def support_metrics(truth: Sequence[int], recovered: Sequence[int]) -> dict[str, float]:
    expected = set(truth)
    actual = set(recovered)
    intersection = len(expected & actual)
    precision = intersection / len(actual) if actual else float(not expected)
    recall = intersection / len(expected) if expected else float(not actual)
    return {
        "exact": float(expected == actual),
        "precision": precision,
        "recall": recall,
    }


def coefficient_relative_error(
    truth_support: Sequence[int],
    truth_coefficients: np.ndarray,
    recovered_support: Sequence[int],
    recovered_coefficients: np.ndarray,
    column_count: int = 54,
) -> float:
    truth = np.zeros(column_count, dtype=np.complex128)
    recovered = np.zeros(column_count, dtype=np.complex128)
    truth[list(truth_support)] = truth_coefficients
    recovered[list(recovered_support)] = recovered_coefficients
    denominator = float(np.linalg.norm(truth))
    return float(np.linalg.norm(recovered - truth) / denominator) if denominator else 0.0


def _random_coefficients(rng: np.random.Generator, count: int) -> np.ndarray:
    amplitudes = rng.uniform(0.2, 1.0, count)
    phases = rng.uniform(-math.pi, math.pi, count)
    return amplitudes * np.exp(1j * phases)


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def oracle_support_study(
    *,
    seed: int = 0x4F524143,
    random_trials: int = 500,
) -> list[dict[str, object]]:
    """Enumerate small supports and sample larger supports deterministically."""

    rng = np.random.default_rng(seed)
    atoms = alias_dictionary()
    support_sizes = tuple(range(1, 13)) + (15, 18, 21, 24)
    rows = []
    for support_size in support_sizes:
        if support_size == 1:
            supports: Iterable[tuple[int, ...]] = ((index,) for index in range(54))
            trial_count = 54
        elif support_size == 2:
            supports = itertools.combinations(range(54), 2)
            trial_count = math.comb(54, 2)
        else:
            supports = (
                tuple(sorted(rng.choice(54, support_size, replace=False).tolist()))
                for _ in range(random_trials)
            )
            trial_count = random_trials
        full_rank = 0
        conditions = []
        residuals = []
        coefficient_errors = []
        recoverable_energy = 0.0
        total_energy = 0.0
        for support in supports:
            coefficients = _random_coefficients(rng, support_size)
            observation = atoms[:, support] @ coefficients
            fit = least_squares_support(observation, support, atoms)
            error = float(
                np.linalg.norm(fit.coefficients - coefficients)
                / np.linalg.norm(coefficients)
            )
            energy = float(np.sum(np.abs(coefficients) ** 2))
            total_energy += energy
            if fit.rank == support_size:
                full_rank += 1
                recoverable_energy += energy
                conditions.append(fit.condition)
            residuals.append(fit.relative_residual)
            coefficient_errors.append(error)
        rows.append(
            {
                "coefficient_error_median": _percentile(coefficient_errors, 50),
                "coefficient_error_p99": _percentile(coefficient_errors, 99),
                "condition_median_full_rank": _percentile(conditions, 50) if conditions else None,
                "condition_p90_full_rank": _percentile(conditions, 90) if conditions else None,
                "condition_p99_full_rank": _percentile(conditions, 99) if conditions else None,
                "full_rank_fraction": full_rank / trial_count,
                "oracle_recoverable_energy_fraction": recoverable_energy / total_energy,
                "relative_residual_maximum": max(residuals),
                "support_size": support_size,
                "trials": trial_count,
            }
        )
    return rows


def _aggregate_trials(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def blind_support_study(
    *,
    seed: int = 0x4F4D5031,
    trials_per_size: int = 300,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    atoms = alias_dictionary()
    output = []
    for support_size in range(1, 9):
        results: dict[str, list[dict[str, float]]] = {
            "replica": [],
            "single_peak": [],
        }
        oracle_full_rank = 0
        for _ in range(trials_per_size):
            support = tuple(sorted(rng.choice(54, support_size, replace=False).tolist()))
            coefficients = _random_coefficients(rng, support_size)
            observation = atoms[:, support] @ coefficients
            if least_squares_support(observation, support, atoms).rank == support_size:
                oracle_full_rank += 1
            for label, mode in (("replica", "replica"), ("single_peak", "single-peak")):
                recovered = orthogonal_matching_pursuit(
                    observation, support_size, dictionary=atoms, mode=mode
                )
                metrics = support_metrics(support, recovered.support)
                metrics["coefficient_error"] = coefficient_relative_error(
                    support,
                    coefficients,
                    recovered.support,
                    recovered.coefficients,
                )
                metrics["iterations"] = float(len(recovered.support))
                metrics["relative_residual"] = recovered.relative_residual
                results[label].append(metrics)
        output.append(
            {
                "oracle_full_rank_fraction": oracle_full_rank / trials_per_size,
                "replica_aware": _aggregate_trials(results["replica"]),
                "single_peak": _aggregate_trials(results["single_peak"]),
                "support_size": support_size,
                "trials": trials_per_size,
            }
        )
    return output


def two_component_breakdown(
    *,
    seed: int = 0x50414952,
    trials_per_case: int = 160,
) -> dict[str, object]:
    """Measure amplitude, phase, color, and carrier-separation effects."""

    rng = np.random.default_rng(seed)
    atoms = alias_dictionary()

    def trial(first: int, second: int, ratio: float, phase: float) -> float:
        coefficients = np.asarray((1.0 + 0.0j, ratio * np.exp(1j * phase)))
        observation = atoms[:, (first, second)] @ coefficients
        recovered = orthogonal_matching_pursuit(observation, 2, dictionary=atoms)
        return support_metrics((first, second), recovered.support)["exact"]

    amplitude_rows = []
    for ratio in (1.0, 0.5, 0.2, 0.1, 0.05):
        values = []
        for _ in range(trials_per_case):
            first, second = rng.choice(54, 2, replace=False)
            values.append(trial(int(first), int(second), ratio, rng.uniform(-math.pi, math.pi)))
        amplitude_rows.append({"amplitude_ratio": ratio, "exact_support_rate": float(np.mean(values))})

    phase_rows = []
    for phase in np.linspace(0.0, math.pi, 9):
        values = []
        for _ in range(trials_per_case):
            first, second = rng.choice(54, 2, replace=False)
            values.append(trial(int(first), int(second), 1.0, float(phase)))
        phase_rows.append({"exact_support_rate": float(np.mean(values)), "relative_phase": float(phase)})

    composition: dict[str, list[float]] = {
        "luma_luma": [],
        "chroma_chroma": [],
        "luma_chroma": [],
    }
    separation: dict[str, list[float]] = {}
    for _ in range(trials_per_case * 8):
        first, second = (int(value) for value in rng.choice(54, 2, replace=False))
        first_carrier, first_name = DESCRIPTORS[first]
        second_carrier, second_name = DESCRIPTORS[second]
        key = (
            "luma_luma"
            if first_name == second_name == "luma"
            else "luma_chroma"
            if "luma" in (first_name, second_name)
            else "chroma_chroma"
        )
        value = trial(first, second, rng.uniform(0.2, 1.0), rng.uniform(-math.pi, math.pi))
        composition[key].append(value)
        dx = min((first_carrier[0] - second_carrier[0]) % 6, (second_carrier[0] - first_carrier[0]) % 6)
        dy = min((first_carrier[1] - second_carrier[1]) % 6, (second_carrier[1] - first_carrier[1]) % 6)
        distance = f"{math.sqrt(dx * dx + dy * dy):.3f}"
        separation.setdefault(distance, []).append(value)
    return {
        "amplitude_ratio": amplitude_rows,
        "carrier_separation": [
            {"distance_on_6x6_torus": float(key), "exact_support_rate": float(np.mean(values)), "trials": len(values)}
            for key, values in sorted(separation.items(), key=lambda item: float(item[0]))
        ],
        "color_composition": {
            key: {"exact_support_rate": float(np.mean(values)), "trials": len(values)}
            for key, values in composition.items()
        },
        "relative_phase": phase_rows,
    }


def noise_study(
    *,
    seed: int = 0x4E4F4953,
    trials_per_case: int = 180,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    atoms = alias_dictionary()
    rows = []
    for support_size in (1, 2, 3, 4, 5):
        for snr_db in (math.inf, 60.0, 40.0, 30.0, 20.0):
            exact = []
            precision = []
            recall = []
            for _ in range(trials_per_case):
                support = tuple(sorted(rng.choice(54, support_size, replace=False).tolist()))
                coefficients = _random_coefficients(rng, support_size)
                clean = atoms[:, support] @ coefficients
                if math.isfinite(snr_db):
                    noise = rng.normal(size=18) + 1j * rng.normal(size=18)
                    noise *= np.linalg.norm(clean) / (
                        np.linalg.norm(noise) * 10.0 ** (snr_db / 20.0)
                    )
                    observation = clean + noise
                else:
                    observation = clean
                recovered = orthogonal_matching_pursuit(observation, support_size, dictionary=atoms)
                metrics = support_metrics(support, recovered.support)
                exact.append(metrics["exact"])
                precision.append(metrics["precision"])
                recall.append(metrics["recall"])
            rows.append(
                {
                    "exact_support_rate": float(np.mean(exact)),
                    "precision": float(np.mean(precision)),
                    "recall": float(np.mean(recall)),
                    "snr_db": "infinite" if not math.isfinite(snr_db) else snr_db,
                    "support_size": support_size,
                    "trials": trials_per_case,
                }
            )
    return rows


def competing_support(
    observation: np.ndarray,
    best_support: Sequence[int],
    max_atoms: int,
    *,
    dictionary: np.ndarray | None = None,
) -> PursuitResult:
    """Find the best OMP interpretation forced to differ materially."""

    atoms = alias_dictionary() if dictionary is None else np.asarray(dictionary)
    if not best_support:
        return orthogonal_matching_pursuit(observation, max_atoms, dictionary=atoms)
    alternatives = [
        orthogonal_matching_pursuit(
            observation,
            max_atoms,
            dictionary=atoms,
            forbidden=(excluded,),
        )
        for excluded in best_support
    ]
    return min(alternatives, key=lambda result: result.relative_residual)


def confidence_study(
    *,
    seed: int = 0x434F4E46,
    trials_per_size: int = 100,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    atoms = alias_dictionary()
    populations: dict[str, list[float]] = {"correct": [], "incorrect": [], "deficient": []}
    records = []
    for support_size in range(1, 6):
        for _ in range(trials_per_size):
            support = tuple(sorted(rng.choice(54, support_size, replace=False).tolist()))
            coefficients = _random_coefficients(rng, support_size)
            observation = atoms[:, support] @ coefficients
            oracle = least_squares_support(observation, support, atoms)
            best = orthogonal_matching_pursuit(observation, support_size, dictionary=atoms)
            alternative = competing_support(
                observation, best.support, support_size, dictionary=atoms
            )
            gap = max(0.0, alternative.relative_residual - best.relative_residual)
            label = (
                "deficient"
                if oracle.rank < support_size
                else "correct"
                if set(best.support) == set(support)
                else "incorrect"
            )
            populations[label].append(gap)
            records.append((gap, label == "correct"))

    # The known five-column dependency is too rare to be represented reliably
    # by random K=5 draws, so exercise its complete support explicitly.
    target_index = DESCRIPTORS.index(((0, 0), "luma"))
    known_deficient_support = (
        target_index,
        DESCRIPTORS.index(((5, 3), "c1_r_minus_b")),
        DESCRIPTORS.index(((0, 0), "c2_green_opponent")),
        DESCRIPTORS.index(((1, 3), "c1_r_minus_b")),
        DESCRIPTORS.index(((3, 3), "c1_r_minus_b")),
    )
    for _ in range(50):
        coefficients = _random_coefficients(rng, len(known_deficient_support))
        observation = atoms[:, known_deficient_support] @ coefficients
        best = orthogonal_matching_pursuit(
            observation, len(known_deficient_support), dictionary=atoms
        )
        alternative = competing_support(
            observation, best.support, len(known_deficient_support), dictionary=atoms
        )
        gap = max(0.0, alternative.relative_residual - best.relative_residual)
        populations["deficient"].append(gap)
        records.append((gap, False))

    candidate_thresholds = sorted(set(gap for gap, _ in records))
    best_threshold = 0.0
    best_balanced = -1.0
    best_sensitivity = 0.0
    best_specificity = 0.0
    for threshold in candidate_thresholds:
        positives = [gap >= threshold for gap, correct in records if correct]
        negatives = [gap < threshold for gap, correct in records if not correct]
        sensitivity = float(np.mean(positives)) if positives else 0.0
        specificity = float(np.mean(negatives)) if negatives else 0.0
        balanced = 0.5 * (sensitivity + specificity)
        if balanced > best_balanced:
            best_threshold = threshold
            best_balanced = balanced
            best_sensitivity = sensitivity
            best_specificity = specificity

    stability_rows = []
    for _ in range(200):
        support_size = int(rng.integers(1, 6))
        support = tuple(sorted(rng.choice(54, support_size, replace=False).tolist()))
        coefficients = _random_coefficients(rng, support_size)
        clean = atoms[:, support] @ coefficients
        best = orthogonal_matching_pursuit(clean, support_size, dictionary=atoms)
        alternative = competing_support(clean, best.support, support_size, dictionary=atoms)
        gap = max(0.0, alternative.relative_residual - best.relative_residual)
        stable = []
        for _ in range(8):
            noise = rng.normal(size=18) + 1j * rng.normal(size=18)
            noise *= np.linalg.norm(clean) / (np.linalg.norm(noise) * 100.0)
            perturbed = orthogonal_matching_pursuit(
                clean + noise, support_size, dictionary=atoms
            )
            stable.append(float(set(perturbed.support) == set(best.support)))
        stability_rows.append((gap >= best_threshold, float(np.mean(stable))))

    null_target = atoms[:, DESCRIPTORS.index(((0, 0), "luma"))]
    null_competitor = orthogonal_matching_pursuit(
        null_target,
        4,
        dictionary=atoms,
        forbidden=(target_index,),
        residual_threshold=1e-12,
    )
    return {
        "derived_gap_classifier": {
            "balanced_accuracy": best_balanced,
            "sensitivity_correct": best_sensitivity,
            "specificity_incorrect_or_deficient": best_specificity,
            "threshold": best_threshold,
        },
        "known_null_competitor": {
            "alternative_support": [descriptor(index) for index in null_competitor.support],
            "relative_residual": null_competitor.relative_residual,
            "target": descriptor(target_index),
        },
        "noise_stability_at_40db": {
            "classified_identifiable_mean_support_stability": float(
                np.mean([value for identifiable, value in stability_rows if identifiable])
            ),
            "classified_uncertain_mean_support_stability": float(
                np.mean([value for identifiable, value in stability_rows if not identifiable])
            ),
            "perturbations_per_case": 8,
            "trials": len(stability_rows),
        },
        "populations": {
            key: {
                "count": len(values),
                "gap_median": _percentile(values, 50) if values else None,
                "gap_p10": _percentile(values, 10) if values else None,
                "gap_p90": _percentile(values, 90) if values else None,
            }
            for key, values in populations.items()
        },
    }


def _canonical_real_frequency(kx: int, ky: int, size: int) -> tuple[int, int]:
    positive = (kx % size, ky % size)
    negative = ((-kx) % size, (-ky) % size)
    return min(positive, negative)


def real_group_dictionary(size: int = 24) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Build real cosine/sine groups in sampled mosaic space.

    Positive/negative frequencies are represented once.  Degenerate sine
    columns at DC/Nyquist points are removed from the group metadata.
    """

    if size % 6:
        raise ValueError("real group grid must be divisible by six")
    y, x = np.mgrid[:size, :size]
    tiled_cfa = np.tile(CANONICAL_XTRANS, (size // 6, size // 6))
    frequencies = sorted(
        {
            _canonical_real_frequency(kx, ky, size)
            for ky in range(size)
            for kx in range(size)
        }
    )
    columns = []
    groups = []
    for kx, ky in frequencies:
        phase = 2.0 * math.pi * (kx * x + ky * y) / size
        for direction_name in BASIS_NAMES:
            direction = DIRECTIONS[direction_name]
            group_columns = []
            quadratures = []
            for quadrature_name, wave in (("cos", np.cos(phase)), ("sin", np.sin(phase))):
                rgb = direction[:, None, None] * wave[None, :, :]
                sampled = np.take_along_axis(
                    np.moveaxis(rgb, 0, -1), tiled_cfa[..., None], axis=2
                )[..., 0].reshape(-1)
                if np.linalg.norm(sampled) > 1e-12:
                    group_columns.append(sampled)
                    quadratures.append(quadrature_name)
            start = len(columns)
            columns.extend(group_columns)
            groups.append(
                {
                    "column_indices": tuple(range(start, len(columns))),
                    "direction": direction_name,
                    "frequency_bin": (kx, ky),
                    "quadratures": tuple(quadratures),
                }
            )
    return np.column_stack(columns), tuple(groups)


def group_orthogonal_matching_pursuit(
    observation: np.ndarray,
    dictionary: np.ndarray,
    groups: Sequence[Mapping[str, object]],
    max_groups: int,
) -> GroupPursuitResult:
    y = np.asarray(observation, dtype=np.float64)
    atoms = np.asarray(dictionary, dtype=np.float64)
    if y.shape != (atoms.shape[0],):
        raise ValueError("group observation and dictionary dimensions do not match")
    selected_groups: list[int] = []
    selected_columns: list[int] = []
    residual = y.copy()
    norm_y = float(np.linalg.norm(y))
    history = [0.0 if norm_y == 0.0 else 1.0]
    coefficients = np.empty(0, dtype=np.float64)
    for _ in range(max_groups):
        best_group = -1
        best_reduction = -1.0
        for group_index, group in enumerate(groups):
            if group_index in selected_groups:
                continue
            columns = tuple(int(value) for value in group["column_indices"])
            candidate = atoms[:, columns]
            projected = candidate @ np.linalg.lstsq(candidate, residual, rcond=None)[0]
            reduction = float(np.vdot(projected, projected).real)
            if reduction > best_reduction:
                best_group = group_index
                best_reduction = reduction
        if best_group < 0:
            break
        selected_groups.append(best_group)
        selected_columns.extend(int(value) for value in groups[best_group]["column_indices"])
        selected = atoms[:, selected_columns]
        coefficients = np.linalg.lstsq(selected, y, rcond=None)[0]
        residual = y - selected @ coefficients
        relative = float(np.linalg.norm(residual) / norm_y) if norm_y else 0.0
        if relative > history[-1] + 2e-14:
            raise ArithmeticError("group OMP residual increased")
        history.append(relative)
    return GroupPursuitResult(
        coefficients=coefficients,
        relative_residual=history[-1],
        residual_history=tuple(history),
        support=tuple(selected_groups),
    )


def grouped_real_study(
    *,
    seed: int = 0x47524F55,
    size: int = 24,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    atoms, groups = real_group_dictionary(size)
    group_lookup = {
        (tuple(group["frequency_bin"]), str(group["direction"])): index
        for index, group in enumerate(groups)
    }
    orthonormal_atoms = np.zeros_like(atoms)
    for group in groups:
        columns = tuple(int(value) for value in group["column_indices"])
        orthonormal_atoms[:, columns] = np.linalg.qr(atoms[:, columns], mode="reduced")[0]
    step = size // 6

    def alias_candidates(group_index: int) -> tuple[int, ...]:
        source = groups[group_index]
        sx, sy = (int(value) for value in source["frequency_bin"])
        frequencies = {
            _canonical_real_frequency(
                sign * sx + carrier_x * step,
                sign * sy + carrier_y * step,
                size,
            )
            for sign in (-1, 1)
            for carrier_x, carrier_y in CARRIER_INDICES
        }
        return tuple(
            sorted(
                group_lookup[(frequency, name)]
                for frequency in frequencies
                for name in BASIS_NAMES
            )
        )

    def recover_one_group(observation: np.ndarray, candidates: Sequence[int]) -> tuple[int, float]:
        norm = float(np.linalg.norm(observation))
        candidate_columns = [
            int(column)
            for candidate_index in candidates
            for column in groups[candidate_index]["column_indices"]
        ]
        correlations = orthonormal_atoms[:, candidate_columns].T @ observation
        offset = 0
        best_index = -1
        best_energy = -1.0
        for candidate_index in candidates:
            columns = tuple(int(value) for value in groups[candidate_index]["column_indices"])
            energy = float(np.sum(correlations[offset : offset + len(columns)] ** 2))
            offset += len(columns)
            if energy > best_energy:
                best_index = candidate_index
                best_energy = energy
        best_columns = tuple(int(value) for value in groups[best_index]["column_indices"])
        selected = atoms[:, best_columns]
        fit = np.linalg.lstsq(selected, observation, rcond=None)[0]
        relative = float(np.linalg.norm(observation - selected @ fit) / norm) if norm else 0.0
        return best_index, relative

    def recover_ordinary(
        observation: np.ndarray, candidates: Sequence[int], atom_count: int
    ) -> tuple[set[int], float]:
        candidate_columns = [
            int(column)
            for group_index in candidates
            for column in groups[group_index]["column_indices"]
        ]
        local = atoms[:, candidate_columns].astype(np.complex128)
        result = orthogonal_matching_pursuit(
            observation.astype(np.complex128), atom_count, dictionary=local
        )
        selected_global_columns = [candidate_columns[index] for index in result.support]
        selected_groups = {
            next(
                index
                for index in candidates
                if column in groups[index]["column_indices"]
            )
            for column in selected_global_columns
        }
        return selected_groups, result.relative_residual
    half_carrier_sources = {
        (8, 4),
        (4, 8),
        (8, 8),
        (4, 12),
        (12, 4),
    }
    generic_sources = {(1, 2), (3, 5), (5, 7), (7, 1), (9, 5)}
    phase_values = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)

    def matching_groups(frequencies: set[tuple[int, int]]) -> list[int]:
        canonical = {_canonical_real_frequency(x, y, size) for x, y in frequencies}
        return [
            index
            for index, group in enumerate(groups)
            if tuple(group["frequency_bin"]) in canonical
        ]

    output = {}
    for class_name, frequencies in (
        ("generic", generic_sources),
        ("half_carrier", half_carrier_sources),
    ):
        grouped_success = []
        ordinary_success = []
        grouped_residual = []
        ordinary_residual = []
        for group_index in matching_groups(frequencies):
            columns = tuple(int(value) for value in groups[group_index]["column_indices"])
            for phase in phase_values:
                coefficients = np.asarray(
                    [math.cos(phase), -math.sin(phase)][: len(columns)], dtype=np.float64
                )
                observation = atoms[:, columns] @ coefficients
                candidates = alias_candidates(group_index)
                grouped_index, grouped_error = recover_one_group(observation, candidates)
                ordinary_groups, ordinary_error = recover_ordinary(
                    observation, candidates, len(columns)
                )
                grouped_success.append(float(grouped_index == group_index))
                ordinary_success.append(float(ordinary_groups == {group_index}))
                grouped_residual.append(grouped_error)
                ordinary_residual.append(ordinary_error)
        output[class_name] = {
            "grouped_exact_rate": float(np.mean(grouped_success)),
            "grouped_residual_maximum": max(grouped_residual),
            "ordinary_exact_rate": float(np.mean(ordinary_success)),
            "ordinary_residual_maximum": max(ordinary_residual),
            "phases_per_group": len(phase_values),
            "tested_groups": len(grouped_success) // len(phase_values),
        }

    noise_rows = []
    eligible = [index for index, group in enumerate(groups) if len(group["column_indices"]) == 2]
    for snr_db in (40.0, 30.0, 20.0):
        successes = []
        for _ in range(200):
            group_index = int(rng.choice(eligible))
            columns = tuple(int(value) for value in groups[group_index]["column_indices"])
            phase = rng.uniform(-math.pi, math.pi)
            clean = atoms[:, columns] @ np.asarray((math.cos(phase), -math.sin(phase)))
            noise = rng.normal(size=clean.shape)
            noise *= np.linalg.norm(clean) / (np.linalg.norm(noise) * 10.0 ** (snr_db / 20.0))
            recovered, _ = recover_one_group(clean + noise, alias_candidates(group_index))
            successes.append(float(recovered == group_index))
        noise_rows.append({"exact_group_rate": float(np.mean(successes)), "snr_db": snr_db, "trials": 200})
    return {
        "grid_size": size,
        "group_count": len(groups),
        "noise": noise_rows,
        "phase_sweep": output,
    }


def _family_coordinates(size: int, family: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    base_x, base_y = family[0]
    step = size // 6
    coordinates = [
        ((base_x + carrier_x * step) % size, (base_y + carrier_y * step) % size)
        for carrier_x, carrier_y in CARRIER_INDICES
    ]
    if set(coordinates) != set(family):
        raise AssertionError("alias-family coordinate mapping changed")
    return coordinates


def family_coordinates(
    size: int,
    family: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Expose the validated carrier-to-family ordering to later experiments."""

    return tuple(_family_coordinates(size, family))


def oracle_image_recovery(
    rgb: np.ndarray,
    *,
    relative_threshold: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover one exact synthetic image with its true Fourier support."""

    checked = np.asarray(rgb, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("RGB image must have shape (3,height,width)")
    if checked.shape[1] != checked.shape[2] or checked.shape[1] % 6:
        raise ValueError("oracle image must be square and divisible by six")
    size = checked.shape[1]
    means = np.mean(checked, axis=(1, 2), keepdims=True)
    centered = checked - means
    basis_coefficients = np.einsum(
        "dc,cyx->dyx", BASIS, np.fft.fft2(centered, axes=(-2, -1)) / (size * size)
    )
    tiled_cfa = np.tile(CANONICAL_XTRANS, (size // 6, size // 6))
    sampled = np.take_along_axis(
        np.moveaxis(centered, 0, -1), tiled_cfa[..., None], axis=2
    )[..., 0]
    observed = np.fft.fft2(sampled) / (size * size)
    recovered_coefficients = np.zeros_like(basis_coefficients)
    threshold = float(np.max(np.abs(basis_coefficients))) * relative_threshold
    atoms = alias_dictionary()
    total_energy = float(np.sum(np.abs(basis_coefficients) ** 2))
    full_rank_energy = 0.0
    deficient_energy = 0.0
    active_families = 0
    deficient_families = 0
    maximum_support = 0
    maximum_observation_error = 0.0
    for family in alias_families(size):
        coordinates = _family_coordinates(size, family)
        truth = np.asarray(
            [
                basis_coefficients[direction, y, x]
                for x, y in coordinates
                for direction in range(3)
            ],
            dtype=np.complex128,
        )
        support = tuple(int(index) for index in np.flatnonzero(np.abs(truth) > threshold))
        if not support:
            continue
        active_families += 1
        maximum_support = max(maximum_support, len(support))
        observation = np.asarray([observed[y, x] for x, y in coordinates])
        predicted = atoms @ truth
        maximum_observation_error = max(
            maximum_observation_error,
            float(np.max(np.abs(predicted - observation))),
        )
        fit = least_squares_support(observation, support, atoms)
        recovered = np.zeros(54, dtype=np.complex128)
        recovered[list(support)] = fit.coefficients
        for carrier_index, (x, y) in enumerate(coordinates):
            recovered_coefficients[:, y, x] = recovered[
                carrier_index * 3 : carrier_index * 3 + 3
            ]
        family_energy = float(np.sum(np.abs(truth) ** 2))
        if fit.rank == len(support):
            full_rank_energy += family_energy
        else:
            deficient_energy += family_energy
            deficient_families += 1
    recovered_rgb_fft = np.einsum("dc,dyx->cyx", BASIS, recovered_coefficients)
    reconstructed = np.fft.ifft2(recovered_rgb_fft * (size * size), axes=(-2, -1)).real + means
    error = reconstructed - checked
    mse = float(np.mean(error * error))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return reconstructed, {
        "active_family_count": active_families,
        "deficient_energy_fraction": deficient_energy / total_energy if total_energy else 0.0,
        "deficient_family_count": deficient_families,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "maximum_active_support": maximum_support,
        "maximum_sampling_equation_error": maximum_observation_error,
        "oracle_recoverable_energy_fraction": full_rank_energy / total_energy if total_energy else 1.0,
        "psnr_db": "infinite" if not math.isfinite(psnr) else psnr,
        "relative_activity_threshold": relative_threshold,
    }


def blind_image_recovery(
    rgb: np.ndarray,
    *,
    max_atoms: int = 12,
    mode: str = "replica",
    residual_threshold: float = 1e-10,
    channel_means: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Crude global Fourier reconstruction using blind family-wise OMP.

    The original diagnostic defaults to exact RGB means because its purpose
    was frequency-support recovery after DC removal.  Image-space comparison
    callers can instead provide three means estimated from the physically
    sampled CFA values, avoiding ground-truth DC information.
    """

    checked = np.asarray(rgb, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("RGB image must have shape (3,height,width)")
    if checked.shape[1] != checked.shape[2] or checked.shape[1] % 6:
        raise ValueError("blind image must be square and divisible by six")
    size = checked.shape[1]
    if channel_means is None:
        means = np.mean(checked, axis=(1, 2), keepdims=True)
    else:
        means = np.asarray(channel_means, dtype=np.float64)
        if means.size != 3 or not np.isfinite(means).all():
            raise ValueError("channel_means must contain three finite values")
        means = means.reshape(3, 1, 1)
    centered = checked - means
    tiled_cfa = np.tile(CANONICAL_XTRANS, (size // 6, size // 6))
    sampled = np.take_along_axis(
        np.moveaxis(centered, 0, -1), tiled_cfa[..., None], axis=2
    )[..., 0]
    observed = np.fft.fft2(sampled) / (size * size)
    recovered_coefficients = np.zeros((3, size, size), dtype=np.complex128)
    atoms = alias_dictionary()
    observation_threshold = float(np.max(np.abs(observed))) * 1e-12
    active_families = 0
    selected_total = 0
    residuals = []
    competing_gaps = []
    for family in alias_families(size):
        coordinates = _family_coordinates(size, family)
        observation = np.asarray([observed[y, x] for x, y in coordinates])
        if np.linalg.norm(observation) <= observation_threshold:
            continue
        active_families += 1
        recovered = orthogonal_matching_pursuit(
            observation,
            max_atoms,
            dictionary=atoms,
            mode=mode,
            residual_threshold=residual_threshold,
        )
        selected_total += len(recovered.support)
        residuals.append(recovered.relative_residual)
        if mode == "replica" and len(competing_gaps) < 64:
            alternative = competing_support(
                observation,
                recovered.support,
                len(recovered.support),
                dictionary=atoms,
            )
            competing_gaps.append(
                max(0.0, alternative.relative_residual - recovered.relative_residual)
            )
        coefficients = np.zeros(54, dtype=np.complex128)
        coefficients[list(recovered.support)] = recovered.coefficients
        for carrier_index, (x, y) in enumerate(coordinates):
            recovered_coefficients[:, y, x] = coefficients[
                carrier_index * 3 : carrier_index * 3 + 3
            ]
    recovered_rgb_fft = np.einsum("dc,dyx->cyx", BASIS, recovered_coefficients)
    reconstructed = np.fft.ifft2(recovered_rgb_fft * (size * size), axes=(-2, -1)).real + means
    error = reconstructed - checked
    mse = float(np.mean(error * error))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    metrics = {
        "active_family_count": active_families,
        "average_selected_atoms": selected_total / active_families if active_families else 0.0,
        "competing_gap_median_first_64_families": (
            _percentile(competing_gaps, 50) if competing_gaps else None
        ),
        "competing_gap_p10_first_64_families": (
            _percentile(competing_gaps, 10) if competing_gaps else None
        ),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "maximum_atoms": max_atoms,
        "mean_relative_observation_residual": float(np.mean(residuals)) if residuals else 0.0,
        "psnr_db": "infinite" if not math.isfinite(psnr) else psnr,
        "score_mode": mode,
    }
    if channel_means is not None:
        metrics["channel_means_source"] = "caller supplied"
    return reconstructed, metrics


def synthetic_oracle_study(size: int = 96) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    from tools.xtrans_alias.analysis import synthetic_edge_cases

    scenes = synthetic_edge_cases(size)
    selected = (
        "vertical_red_green",
        "diagonal_red_green",
        "saturated_red_gray",
        "abc_saturated_edges",
        "abc_frequency_sweep",
    )
    output = {}
    images = {}
    for name in selected:
        reconstructed, metrics = oracle_image_recovery(scenes[name])
        blind, blind_metrics = blind_image_recovery(scenes[name], mode="replica")
        peak, peak_metrics = blind_image_recovery(scenes[name], mode="single-peak")
        output[name] = {
            "blind_replica": blind_metrics,
            "blind_single_peak": peak_metrics,
            "oracle": metrics,
        }
        images[f"{name}_ground_truth"] = scenes[name]
        images[f"{name}_oracle"] = reconstructed
        images[f"{name}_blind_replica"] = blind
    return output, images


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    checked = np.asarray(rgb, dtype=np.float64)
    return np.where(
        checked <= 0.04045,
        checked / 12.92,
        ((checked + 0.055) / 1.055) ** 2.4,
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = percentile / 100.0 * cumulative[-1]
    return float(sorted_values[min(int(np.searchsorted(cumulative, target)), len(values) - 1)])


def natural_sparsity_study(
    sources: Sequence[Mapping[str, object]],
    *,
    window_sizes: Sequence[int] = (24, 48, 96),
    maximum_windows_per_size: int = 16,
) -> dict[str, object]:
    """Measure alias-family occupancy in independently supplied RGB images."""

    thresholds_db = (-20, -30, -40)
    top_counts = (1, 2, 3, 5)
    accumulators: dict[str, dict[str, list[float]]] = {
        name: {"effective": [], "weight": [], **{f"top_{count}": [] for count in top_counts}}
        for name in (*BASIS_NAMES, "mixed")
    }
    significant: dict[str, dict[int, list[int]]] = {
        name: {threshold: [] for threshold in thresholds_db}
        for name in (*BASIS_NAMES, "mixed")
    }
    category_accumulators: dict[str, dict[str, dict[str, list[float]]]] = {}
    band_accumulators: dict[str, dict[str, dict[str, list[float]]]] = {
        band: {
            name: {"effective": [], "weight": [], "top_1": [], "top_3": [], "top_5": []}
            for name in (*BASIS_NAMES, "mixed")
        }
        for band in ("low", "middle", "high")
    }
    recovery_samples: list[tuple[np.ndarray, float]] = []

    def compact_accumulator() -> dict[str, dict[str, list[float]]]:
        return {
            name: {"effective": [], "weight": [], "top_1": [], "top_3": [], "top_5": []}
            for name in (*BASIS_NAMES, "mixed")
        }

    def update_compact(
        destination: dict[str, dict[str, list[float]]],
        name: str,
        effective: float,
        total: float,
        ordered: np.ndarray,
    ) -> None:
        destination[name]["effective"].append(effective)
        destination[name]["weight"].append(total)
        for count in (1, 3, 5):
            destination[name][f"top_{count}"].append(float(np.sum(ordered[:count]) / total))

    per_source = []
    total_windows = 0
    total_families = 0
    for source in sources:
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("natural source RGB must have shape (height,width,3)")
        linear = srgb_to_linear(rgb)
        source_windows = 0
        category = str(source["category"])
        category_accumulators.setdefault(category, compact_accumulator())
        for size in window_sizes:
            if min(linear.shape[:2]) < size:
                continue
            stride = size // 2
            positions = [
                (y, x)
                for y in range(0, linear.shape[0] - size + 1, stride)
                for x in range(0, linear.shape[1] - size + 1, stride)
            ]
            if len(positions) > maximum_windows_per_size:
                indices = np.linspace(0, len(positions) - 1, maximum_windows_per_size, dtype=int)
                positions = [positions[int(index)] for index in indices]
            hann = np.outer(np.hanning(size), np.hanning(size))
            for y, x in positions:
                patch = np.moveaxis(linear[y : y + size, x : x + size], -1, 0)
                patch = patch - np.mean(patch, axis=(1, 2), keepdims=True)
                coefficients = np.einsum(
                    "dc,cyx->dyx",
                    BASIS,
                    np.fft.fft2(patch * hann[None, :, :], axes=(-2, -1)),
                )
                for family in alias_families(size):
                    coordinates = _family_coordinates(size, family)
                    direction_energies = [
                        np.asarray([abs(coefficients[d, yy, xx]) ** 2 for xx, yy in coordinates])
                        for d in range(3)
                    ]
                    vectors = {
                        BASIS_NAMES[d]: direction_energies[d] for d in range(3)
                    }
                    vectors["mixed"] = np.concatenate(direction_energies)
                    frequency_radius = np.asarray(
                        [
                            math.hypot(centered_frequency(xx, size), centered_frequency(yy, size))
                            for xx, yy in coordinates
                        ]
                    )
                    for name, energies in vectors.items():
                        total = float(np.sum(energies))
                        if total <= 1e-24:
                            continue
                        ordered = np.sort(energies)[::-1]
                        effective = total * total / float(np.sum(energies * energies))
                        accumulators[name]["effective"].append(effective)
                        accumulators[name]["weight"].append(total)
                        for count in top_counts:
                            accumulators[name][f"top_{count}"].append(
                                float(np.sum(ordered[:count]) / total)
                            )
                        update_compact(
                            category_accumulators[category], name, effective, total, ordered
                        )
                        radii = (
                            np.tile(frequency_radius, 3)
                            if name == "mixed"
                            else frequency_radius
                        )
                        dominant_radius = float(radii[int(np.argmax(energies))])
                        band = "low" if dominant_radius < 0.125 else "middle" if dominant_radius < 0.25 else "high"
                        update_compact(band_accumulators[band], name, effective, total, ordered)
                        maximum = float(ordered[0])
                        for threshold in thresholds_db:
                            significant[name][threshold].append(
                                int(np.count_nonzero(energies >= maximum * 10.0 ** (threshold / 10.0)))
                            )
                    truth = np.concatenate(
                        [
                            np.asarray([coefficients[d, yy, xx] for xx, yy in coordinates])
                            for d in range(3)
                        ]
                    )
                    # Convert direction-major natural coefficients to the
                    # carrier-major dictionary order.
                    truth = truth.reshape(3, 18).T.reshape(54)
                    recovery_samples.append((truth, float(np.sum(np.abs(truth) ** 2))))
                    total_families += 1
                total_windows += 1
                source_windows += 1
        per_source.append(
            {
                "category": source["category"],
                "height": int(rgb.shape[0]),
                "name": source["name"],
                "sha256": source["sha256"],
                "width": int(rgb.shape[1]),
                "windows": source_windows,
            }
        )

    summaries = {}
    for name, values in accumulators.items():
        effective = np.asarray(values["effective"], dtype=np.float64)
        weights = np.asarray(values["weight"], dtype=np.float64)
        summaries[name] = {
            "active_family_samples": len(effective),
            "effective_sparsity": {
                "median": _percentile(effective, 50),
                "p90": _percentile(effective, 90),
                "p99": _percentile(effective, 99),
                "energy_weighted_median": _weighted_percentile(effective, weights, 50),
                "energy_weighted_p90": _weighted_percentile(effective, weights, 90),
            },
            "significant_count_histograms": {
                str(threshold): {
                    str(count): int(quantity)
                    for count, quantity in enumerate(np.bincount(significant[name][threshold]))
                    if quantity
                }
                for threshold in thresholds_db
            },
            "top_energy_fraction": {
                str(count): float(
                    np.average(np.asarray(values[f"top_{count}"]), weights=weights)
                )
                for count in top_counts
            },
        }

    def summarize_compact(
        source: dict[str, dict[str, list[float]]]
    ) -> dict[str, object]:
        output = {}
        for name, values in source.items():
            if not values["effective"]:
                continue
            effective = np.asarray(values["effective"], dtype=np.float64)
            weights = np.asarray(values["weight"], dtype=np.float64)
            output[name] = {
                "family_samples": len(effective),
                "effective_sparsity_energy_weighted_median": _weighted_percentile(
                    effective, weights, 50
                ),
                "top_energy_fraction": {
                    str(count): float(
                        np.average(np.asarray(values[f"top_{count}"]), weights=weights)
                    )
                    for count in (1, 3, 5)
                },
            }
        return output

    atoms = alias_dictionary()
    sample_indices = np.linspace(
        0, len(recovery_samples) - 1, min(4096, len(recovery_samples)), dtype=int
    )
    natural_recovery = []
    for support_size in (1, 2, 3, 5):
        metrics = {
            "weight": [],
            "oracle_error": [],
            "replica_error": [],
            "peak_error": [],
            "replica_observation_residual": [],
            "peak_observation_residual": [],
            "replica_top_support_recall": [],
            "peak_top_support_recall": [],
        }
        for sample_index in sample_indices:
            truth, weight = recovery_samples[int(sample_index)]
            observation = atoms @ truth
            top_support = tuple(
                sorted(np.argsort(np.abs(truth), kind="stable")[-support_size:].tolist())
            )
            oracle = least_squares_support(observation, top_support, atoms)
            replica = orthogonal_matching_pursuit(
                observation, support_size, dictionary=atoms, mode="replica"
            )
            peak = orthogonal_matching_pursuit(
                observation, support_size, dictionary=atoms, mode="single-peak"
            )
            metrics["weight"].append(weight)
            metrics["oracle_error"].append(
                coefficient_relative_error(
                    range(54), truth, top_support, oracle.coefficients
                )
            )
            metrics["replica_error"].append(
                coefficient_relative_error(
                    range(54), truth, replica.support, replica.coefficients
                )
            )
            metrics["peak_error"].append(
                coefficient_relative_error(
                    range(54), truth, peak.support, peak.coefficients
                )
            )
            metrics["replica_observation_residual"].append(replica.relative_residual)
            metrics["peak_observation_residual"].append(peak.relative_residual)
            metrics["replica_top_support_recall"].append(
                support_metrics(top_support, replica.support)["recall"]
            )
            metrics["peak_top_support_recall"].append(
                support_metrics(top_support, peak.support)["recall"]
            )
        weights = np.asarray(metrics.pop("weight"), dtype=np.float64)
        natural_recovery.append(
            {
                "energy_weighted_mean": {
                    key: float(np.average(np.asarray(values), weights=weights))
                    for key, values in metrics.items()
                },
                "sampled_families": len(sample_indices),
                "support_size": support_size,
            }
        )
    return {
        "analysis": {
            "channel_means_removed": True,
            "color_transfer": "IEC 61966-2-1 sRGB inverse transfer to linear RGB",
            "hann_window": "separable symmetric Hann",
            "maximum_windows_per_image_and_size": maximum_windows_per_size,
            "significant_thresholds_db_energy_relative_to_family_maximum": list(thresholds_db),
            "window_overlap": "50 percent before deterministic subsampling",
            "window_sizes": list(window_sizes),
        },
        "family_samples": total_families,
        "by_dominant_frequency_band": {
            band: summarize_compact(values) for band, values in band_accumulators.items()
        },
        "by_source_category": {
            category: summarize_compact(values)
            for category, values in category_accumulators.items()
        },
        "fixed_cardinality_recovery": natural_recovery,
        "sources": per_source,
        "summaries": summaries,
        "windows": total_windows,
    }


def build_recovery_characterization(
    natural_sources: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    oracle = oracle_support_study()
    blind = blind_support_study()
    pairs = two_component_breakdown()
    noise = noise_study()
    confidence = confidence_study()
    grouped = grouped_real_study()
    synthetic, images = synthetic_oracle_study()
    natural = natural_sparsity_study(natural_sources)
    operator = alias_dictionary()
    previous_rgb = np.column_stack(
        [
            carrier_signature(carrier, np.eye(3)[color])
            for carrier in CARRIER_INDICES
            for color in range(3)
        ]
    )
    basis_to_rgb = np.kron(np.eye(18), BASIS.T)
    operator_match = float(np.max(np.abs(operator - previous_rgb @ basis_to_rgb)))
    payload = {
        "alias_operator": {
            "basis": {name: DIRECTIONS[name].tolist() for name in BASIS_NAMES},
            "columns": 54,
            "descriptor_order": [descriptor(index) for index in range(54)],
            "maximum_reuse_transform_error": operator_match,
            "rows": 18,
        },
        "blind_support": blind,
        "confidence": confidence,
        "format": MODEL_FORMAT,
        "grouped_real": grouped,
        "natural_sparsity": natural,
        "noise": noise,
        "oracle_support": oracle,
        "synthetic_oracle_images": synthetic,
        "two_component_breakdown": pairs,
    }
    return payload, images

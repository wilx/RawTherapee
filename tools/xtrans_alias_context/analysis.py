"""Windowed classification of competing X-Trans alias-source hypotheses.

This module deliberately operates on source supports and their complex Fourier
coefficients.  It never regularizes reconstructed RGB or chroma samples.  The
experiment reuses the previously validated 18x54 alias operator and asks if
translation-predicted phase and support continuity can select between locally
plausible sparse explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from tools.xtrans_alias.analysis import (
    CANONICAL_XTRANS,
    CARRIER_INDICES,
    DIRECTIONS,
    alias_families,
    carrier_signature,
    centered_frequency,
    mask_spectra,
    synthetic_edge_cases,
)
from tools.xtrans_sparse_alias.analysis import (
    BASIS,
    BASIS_NAMES,
    DESCRIPTORS,
    alias_dictionary,
    coefficient_relative_error,
    family_coordinates,
    least_squares_support,
    orthogonal_matching_pursuit,
    srgb_to_linear,
    support_metrics,
)


MODEL_FORMAT = "rawtherapee-xtrans-windowed-alias-context-v1"


@dataclass(frozen=True)
class Hypothesis:
    """One materially distinct fixed-cardinality source interpretation."""

    coefficients: np.ndarray
    condition: float
    energy: float
    normalized_residual: float
    sigma_min: float
    singular_values: tuple[float, ...]
    support: tuple[int, ...]


@dataclass(frozen=True)
class WindowFamily:
    """One alias family observed in one translated analysis window."""

    candidates: tuple[Hypothesis, ...]
    coordinates: tuple[tuple[int, int], ...]
    family_index: int
    observation: np.ndarray
    origin: tuple[int, int]
    size: int
    truth: np.ndarray
    truth_support: tuple[int, ...]


@dataclass(frozen=True)
class Compatibility:
    amplitude: float
    color: float
    frequency: float
    phase: float
    support: float

    def weighted(self, weights: Mapping[str, float]) -> float:
        total = float(sum(weights.values()))
        if total <= 0.0:
            return 0.0
        return float(
            sum(float(weights[name]) * float(getattr(self, name)) for name in weights)
            / total
        )


CONTEXT_WEIGHTS: dict[str, dict[str, float]] = {
    "local_only": {"support": 0.0, "phase": 0.0, "color": 0.0, "frequency": 0.0, "amplitude": 0.0},
    "support_only": {"support": 1.0, "phase": 0.0, "color": 0.0, "frequency": 0.0, "amplitude": 0.0},
    "phase_only": {"support": 0.0, "phase": 1.0, "color": 0.0, "frequency": 0.0, "amplitude": 0.0},
    "color_only": {"support": 0.0, "phase": 0.0, "color": 1.0, "frequency": 0.0, "amplitude": 0.0},
    "support_phase": {"support": 1.0, "phase": 1.5, "color": 0.0, "frequency": 0.0, "amplitude": 0.0},
    "full": {"support": 1.0, "phase": 1.5, "color": 0.4, "frequency": 0.3, "amplitude": 0.1},
}


def dictionary_for_origin(origin: tuple[int, int]) -> np.ndarray:
    """Return the exact alias dictionary for the CFA phase at ``origin``."""

    ox, oy = origin
    shifted = np.asarray(
        [
            [CANONICAL_XTRANS[(y + oy) % 6, (x + ox) % 6] for x in range(6)]
            for y in range(6)
        ],
        dtype=np.int8,
    )
    spectra = mask_spectra(shifted)
    return np.column_stack(
        [carrier_signature(carrier, DIRECTIONS[name], spectra) for carrier, name in DESCRIPTORS]
    )


def support_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a = set(left)
    b = set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _hypothesis(
    observation: np.ndarray,
    support: Sequence[int],
    dictionary: np.ndarray,
) -> Hypothesis:
    checked = tuple(sorted(int(value) for value in support))
    fit = least_squares_support(observation, checked, dictionary)
    return Hypothesis(
        coefficients=fit.coefficients,
        condition=fit.condition,
        energy=float(np.sum(np.abs(fit.coefficients) ** 2)),
        normalized_residual=fit.relative_residual,
        sigma_min=fit.sigma_min,
        singular_values=tuple(float(value) for value in fit.singular_values),
        support=checked,
    )


def beam_hypotheses(
    observation: np.ndarray,
    maximum_support: int,
    proposal_count: int,
    *,
    beam_width: int = 8,
    dictionary: np.ndarray | None = None,
    branch_factor: int = 8,
    maximum_jaccard: float = 0.8,
) -> tuple[Hypothesis, ...]:
    """Generate distinct supports using deterministic limited OMP beam search."""

    atoms = alias_dictionary() if dictionary is None else np.asarray(dictionary)
    y = np.asarray(observation, dtype=np.complex128)
    if y.shape != (atoms.shape[0],):
        raise ValueError("observation and dictionary dimensions do not match")
    if not 1 <= maximum_support <= min(18, atoms.shape[1]):
        raise ValueError("maximum support must be in 1..18")
    if not 1 <= proposal_count <= 32 or not 1 <= beam_width <= 64:
        raise ValueError("proposal count or beam width is out of range")

    partial: dict[tuple[int, ...], Hypothesis] = {
        (): Hypothesis(np.empty(0, dtype=np.complex128), 1.0, 0.0, 1.0, 0.0, (), ())
    }
    for _depth in range(maximum_support):
        expanded: dict[tuple[int, ...], Hypothesis] = {}
        for support, current in partial.items():
            residual = y if not support else (
                y - atoms[:, support] @ current.coefficients
            )
            norms = np.linalg.norm(atoms, axis=0)
            scores = np.abs(atoms.conj().T @ residual) / np.maximum(norms, 1e-300)
            if support:
                scores[list(support)] = -1.0
            order = np.argsort(scores, kind="stable")[-branch_factor:][::-1]
            for candidate in order:
                selected = tuple(sorted((*support, int(candidate))))
                value = _hypothesis(y, selected, atoms)
                old = expanded.get(selected)
                if old is None or value.normalized_residual < old.normalized_residual:
                    expanded[selected] = value
        partial = dict(
            sorted(
                expanded.items(),
                key=lambda item: (
                    item[1].normalized_residual,
                    not math.isfinite(item[1].condition),
                    item[1].condition,
                    item[0],
                ),
            )[:beam_width]
        )

    # Forced-exclusion OMP gives a cheap second source of alternatives when a
    # narrow beam converged on near-identical branches.
    best_omp = orthogonal_matching_pursuit(y, maximum_support, dictionary=atoms)
    pool: dict[tuple[int, ...], Hypothesis] = dict(partial)
    base_support = tuple(sorted(best_omp.support))
    if len(base_support) == maximum_support:
        pool[base_support] = _hypothesis(y, base_support, atoms)
    for excluded in base_support:
        alternate = orthogonal_matching_pursuit(
            y, maximum_support, dictionary=atoms, forbidden=(excluded,)
        )
        selected = tuple(sorted(alternate.support))
        if len(selected) == maximum_support:
            pool[selected] = _hypothesis(y, selected, atoms)

    ordered = sorted(
        pool.values(),
        key=lambda value: (
            value.normalized_residual,
            not math.isfinite(value.condition),
            value.condition,
            value.support,
        ),
    )
    diverse: list[Hypothesis] = []
    for candidate in ordered:
        if all(support_jaccard(candidate.support, kept.support) <= maximum_jaccard for kept in diverse):
            diverse.append(candidate)
            if len(diverse) == proposal_count:
                break
    if len(diverse) < proposal_count:
        kept_supports = {candidate.support for candidate in diverse}
        for candidate in ordered:
            if candidate.support not in kept_supports:
                diverse.append(candidate)
                kept_supports.add(candidate.support)
                if len(diverse) == proposal_count:
                    break
    return tuple(diverse)


def _periodic_hann(size: int) -> np.ndarray:
    one = np.hanning(size + 1)[:-1]
    return np.sqrt(np.outer(one, one))


def extract_window_spectra(
    rgb: np.ndarray,
    origin: tuple[int, int],
    size: int,
) -> tuple[tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]], ...]:
    """Extract exact observed and ground-truth alias families for one window."""

    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[0] != 3 or size % 6:
        raise ValueError("RGB must be CHW and window size must be divisible by six")
    ox, oy = origin
    if ox < 0 or oy < 0 or ox + size > image.shape[2] or oy + size > image.shape[1]:
        raise ValueError("window lies outside the RGB image")
    patch = image[:, oy : oy + size, ox : ox + size]
    centered = patch - np.mean(patch, axis=(1, 2), keepdims=True)
    window = _periodic_hann(size)
    tapered = centered * window[None, :, :]
    basis_fft = np.einsum(
        "dc,cyx->dyx", BASIS, np.fft.fft2(tapered, axes=(-2, -1)) / (size * size)
    )
    local_cfa = np.asarray(
        [
            [CANONICAL_XTRANS[(y + oy) % 6, (x + ox) % 6] for x in range(size)]
            for y in range(size)
        ],
        dtype=np.int8,
    )
    sampled = np.take_along_axis(
        np.moveaxis(tapered, 0, -1), local_cfa[..., None], axis=2
    )[..., 0]
    observed = np.fft.fft2(sampled) / (size * size)
    output = []
    for family in alias_families(size):
        coordinates = family_coordinates(size, family)
        observation = np.asarray([observed[y, x] for x, y in coordinates])
        truth = np.asarray(
            [basis_fft[direction, y, x] for x, y in coordinates for direction in range(3)]
        )
        output.append((observation, truth, coordinates))
    return tuple(output)


def make_window_family(
    spectra: tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]],
    family_index: int,
    origin: tuple[int, int],
    size: int,
    *,
    maximum_support: int = 5,
    proposal_count: int = 8,
    beam_width: int = 8,
) -> WindowFamily:
    observation, truth, coordinates = spectra
    truth_support = tuple(
        sorted(np.argsort(np.abs(truth), kind="stable")[-maximum_support:].tolist())
    )
    candidates = beam_hypotheses(
        observation,
        maximum_support,
        proposal_count,
        beam_width=beam_width,
        dictionary=dictionary_for_origin(origin),
    )
    return WindowFamily(
        candidates=candidates,
        coordinates=coordinates,
        family_index=family_index,
        observation=observation,
        origin=origin,
        size=size,
        truth=truth,
        truth_support=truth_support,
    )


def _atom_frequency(record: WindowFamily, atom: int) -> tuple[float, float]:
    x, y = record.coordinates[atom // 3]
    return centered_frequency(x, record.size), centered_frequency(y, record.size)


def _energy_map(hypothesis: Hypothesis) -> dict[int, float]:
    return {
        atom: float(abs(coefficient) ** 2)
        for atom, coefficient in zip(hypothesis.support, hypothesis.coefficients)
    }


def compatibility(
    left_record: WindowFamily,
    left: Hypothesis,
    right_record: WindowFamily,
    right: Hypothesis,
) -> Compatibility:
    """Return individually interpretable source-hypothesis disagreement terms."""

    if left_record.size != right_record.size or left_record.family_index != right_record.family_index:
        raise ValueError("compatibility requires matching window grids and alias families")
    left_energy = _energy_map(left)
    right_energy = _energy_map(right)
    all_atoms = set(left_energy) | set(right_energy)
    intersection = sum(min(left_energy.get(atom, 0.0), right_energy.get(atom, 0.0)) for atom in all_atoms)
    union = sum(max(left_energy.get(atom, 0.0), right_energy.get(atom, 0.0)) for atom in all_atoms)
    support_error = 0.0 if union == 0.0 else 1.0 - intersection / union

    dx = right_record.origin[0] - left_record.origin[0]
    dy = right_record.origin[1] - left_record.origin[1]
    left_coefficients = dict(zip(left.support, left.coefficients))
    right_coefficients = dict(zip(right.support, right.coefficients))
    common = sorted(set(left.support) & set(right.support))
    phase_errors = []
    phase_weights = []
    amplitude_errors = []
    for atom in common:
        first = left_coefficients[atom]
        second = right_coefficients[atom]
        weight = min(abs(first), abs(second)) ** 2
        if weight <= 1e-24:
            continue
        fx, fy = _atom_frequency(left_record, atom)
        predicted = 2.0 * math.pi * (fx * dx + fy * dy)
        difference = float(np.angle(second) - np.angle(first) - predicted)
        wrapped = (difference + math.pi) % (2.0 * math.pi) - math.pi
        phase_errors.append(abs(wrapped) / math.pi)
        phase_weights.append(weight)
        amplitude_errors.append(min(1.0, abs(math.log((abs(second) + 1e-15) / (abs(first) + 1e-15))) / math.log(4.0)))
    usable_phase = phase_errors and sum(phase_weights) > 1e-24
    phase_error = float(np.average(phase_errors, weights=phase_weights)) if usable_phase else 1.0
    amplitude_error = float(np.average(amplitude_errors, weights=phase_weights)) if usable_phase else 1.0

    def color_distribution(hypothesis: Hypothesis) -> np.ndarray:
        values = np.zeros(3, dtype=np.float64)
        for atom, coefficient in zip(hypothesis.support, hypothesis.coefficients):
            values[atom % 3] += abs(coefficient) ** 2
        total = float(np.sum(values))
        return values / total if total else values

    color_error = float(np.sum(np.abs(color_distribution(left) - color_distribution(right))) / 2.0)

    frequency_errors = []
    frequency_weights = []
    for atom, energy in left_energy.items():
        fx, fy = _atom_frequency(left_record, atom)
        same_color = [other for other in right.support if other % 3 == atom % 3]
        if not same_color:
            frequency_errors.append(1.0)
        else:
            distances = []
            for other in same_color:
                gx, gy = _atom_frequency(right_record, other)
                distances.append(math.hypot(fx - gx, fy - gy) / math.sqrt(0.5))
            frequency_errors.append(min(1.0, min(distances)))
        frequency_weights.append(energy)
    frequency_error = (
        float(np.average(frequency_errors, weights=frequency_weights))
        if frequency_weights and sum(frequency_weights) > 1e-24 else 0.0
    )
    return Compatibility(
        amplitude=amplitude_error,
        color=color_error,
        frequency=frequency_error,
        phase=phase_error,
        support=support_error,
    )


def local_costs(candidates: Sequence[Hypothesis]) -> np.ndarray:
    residuals = np.asarray([value.normalized_residual for value in candidates])
    spread = float(np.max(residuals) - np.min(residuals))
    if spread <= 1e-12:
        return np.zeros_like(residuals)
    return (residuals - np.min(residuals)) / spread


def rerank(
    center: WindowFamily,
    neighbors: Sequence[WindowFamily],
    weights: Mapping[str, float],
    *,
    context_strength: float = 0.5,
) -> tuple[int, np.ndarray, dict[str, float]]:
    """Select a center hypothesis using best compatible neighbor proposals."""

    costs = local_costs(center.candidates)
    term_sums = {name: np.zeros(len(center.candidates)) for name in Compatibility.__dataclass_fields__}
    if sum(weights.values()) > 0.0 and neighbors:
        for neighbor in neighbors:
            for center_index, candidate in enumerate(center.candidates):
                alternatives = [compatibility(center, candidate, neighbor, value) for value in neighbor.candidates]
                best = min(alternatives, key=lambda value: value.weighted(weights))
                costs[center_index] += context_strength * best.weighted(weights) / len(neighbors)
                for name in term_sums:
                    term_sums[name][center_index] += float(getattr(best, name)) / len(neighbors)
    order = np.argsort(costs, kind="stable")
    selected = int(order[0])
    margin = float(costs[order[1]] - costs[order[0]]) if len(order) > 1 else math.inf
    terms = {name: float(values[selected]) for name, values in term_sums.items()}
    terms["contextual_margin"] = margin
    return selected, costs, terms


def _neighbor_indices(records: Sequence[WindowFamily]) -> dict[int, list[int]]:
    origins = {record.origin: index for index, record in enumerate(records)}
    xs = sorted({record.origin[0] for record in records})
    ys = sorted({record.origin[1] for record in records})
    dx = min((right - left for left, right in zip(xs, xs[1:])), default=0)
    dy = min((bottom - top for top, bottom in zip(ys, ys[1:])), default=0)
    output: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        neighbors = []
        for offset in ((-dx, 0), (dx, 0), (0, -dy), (0, dy)):
            if offset == (0, 0):
                continue
            other = origins.get((record.origin[0] + offset[0], record.origin[1] + offset[1]))
            if other is not None:
                neighbors.append(other)
        output[index] = neighbors
    return output


def select_track(
    records: Sequence[WindowFamily],
    mode: str,
    *,
    iterative: bool = False,
    iterations: int = 5,
) -> tuple[list[int], list[dict[str, float]]]:
    weights = CONTEXT_WEIGHTS[mode]
    graph = _neighbor_indices(records)
    selections = [0 for _ in records]
    diagnostics: list[dict[str, float]] = []
    if not iterative or mode == "local_only":
        for index, record in enumerate(records):
            selected, _costs, terms = rerank(
                record, [records[value] for value in graph[index]], weights
            )
            selections[index] = selected
            diagnostics.append(terms)
        return selections, diagnostics

    for _ in range(iterations):
        changed = False
        updated = list(selections)
        for index, record in enumerate(records):
            costs = local_costs(record.candidates)
            for candidate_index, candidate in enumerate(record.candidates):
                if graph[index]:
                    costs[candidate_index] += 0.5 * float(
                        np.mean(
                            [
                                compatibility(
                                    record,
                                    candidate,
                                    records[other],
                                    records[other].candidates[selections[other]],
                                ).weighted(weights)
                                for other in graph[index]
                            ]
                        )
                    )
            updated[index] = int(np.argmin(costs))
            changed |= updated[index] != selections[index]
        selections = updated
        if not changed:
            break
    for index, record in enumerate(records):
        neighbors = graph[index]
        chosen = record.candidates[selections[index]]
        terms = {name: 0.0 for name in Compatibility.__dataclass_fields__}
        if neighbors:
            values = [
                compatibility(record, chosen, records[other], records[other].candidates[selections[other]])
                for other in neighbors
            ]
            for name in terms:
                terms[name] = float(np.mean([getattr(value, name) for value in values]))
        terms["contextual_margin"] = 0.0
        diagnostics.append(terms)
    return selections, diagnostics


def candidate_error(record: WindowFamily, hypothesis: Hypothesis) -> float:
    return coefficient_relative_error(
        range(54), record.truth, hypothesis.support, hypothesis.coefficients
    )


def _aggregate_selection(
    tracks: Sequence[Sequence[WindowFamily]],
    mode: str,
    *,
    iterative: bool = False,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    exact = []
    precision = []
    recall = []
    errors = []
    residuals = []
    records_out = []
    for track_index, track in enumerate(tracks):
        selected, diagnostics = select_track(track, mode, iterative=iterative)
        for window_index, (record, choice, terms) in enumerate(zip(track, selected, diagnostics)):
            hypothesis = record.candidates[choice]
            metrics = support_metrics(record.truth_support, hypothesis.support)
            error = candidate_error(record, hypothesis)
            exact.append(metrics["exact"])
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])
            errors.append(error)
            residuals.append(hypothesis.normalized_residual)
            records_out.append(
                {
                    "candidate_index": choice,
                    "coefficient_error": error,
                    "correct": bool(metrics["exact"]),
                    "diagnostics": terms,
                    "family_index": record.family_index,
                    "origin": list(record.origin),
                    "track_index": track_index,
                    "window_index": window_index,
                }
            )
    return {
        "coefficient_error_mean": float(np.mean(errors)),
        "exact_support_rate": float(np.mean(exact)),
        "mean_observation_residual": float(np.mean(residuals)),
        "support_precision": float(np.mean(precision)),
        "support_recall": float(np.mean(recall)),
    }, records_out


def _scene_tracks(
    rgb: np.ndarray,
    *,
    window_size: int = 48,
    shift: int = 12,
    maximum_support: int = 5,
    proposal_count: int = 8,
    beam_width: int = 8,
    family_count: int = 5,
) -> tuple[list[list[WindowFamily]], list[tuple[int, int]]]:
    height, width = rgb.shape[1:]
    center_x = max(0, (width - window_size) // 2)
    center_y = max(0, (height - window_size) // 2)
    xs = sorted({max(0, min(width - window_size, center_x + offset)) for offset in (-shift, 0, shift)})
    ys = sorted({max(0, min(height - window_size, center_y + offset)) for offset in (-shift, 0, shift)})
    origins = [(x, y) for y in ys for x in xs]
    spectra = {origin: extract_window_spectra(rgb, origin, window_size) for origin in origins}
    family_energies = []
    for family_index in range(len(next(iter(spectra.values())))):
        energy = float(np.mean([np.sum(np.abs(value[family_index][1]) ** 2) for value in spectra.values()]))
        family_energies.append((energy, family_index))
    selected_families = [index for _energy, index in sorted(family_energies, reverse=True)[:family_count]]
    tracks = []
    for family_index in selected_families:
        tracks.append(
            [
                make_window_family(
                    spectra[origin][family_index],
                    family_index,
                    origin,
                    window_size,
                    maximum_support=maximum_support,
                    proposal_count=proposal_count,
                    beam_width=beam_width,
                )
                for origin in origins
            ]
        )
    return tracks, origins


def _proposal_summary(tracks: Sequence[Sequence[WindowFamily]]) -> dict[str, object]:
    records = [record for track in tracks for record in track]
    result = {}
    for count in (1, 3, 5, 8):
        result[str(count)] = {
            "truth_in_top_n": float(
                np.mean(
                    [
                        any(set(candidate.support) == set(record.truth_support) for candidate in record.candidates[:count])
                        for record in records
                    ]
                )
            ),
            "mean_distinct_supports": float(np.mean([min(count, len(record.candidates)) for record in records])),
        }
    residual_spreads = [
        record.candidates[min(1, len(record.candidates) - 1)].normalized_residual
        - record.candidates[0].normalized_residual
        for record in records
    ]
    jaccards = [
        support_jaccard(record.candidates[0].support, record.candidates[1].support)
        for record in records
        if len(record.candidates) > 1
    ]
    result["diversity"] = {
        "best_second_residual_gap_median": float(np.median(residual_spreads)),
        "best_second_support_jaccard_mean": float(np.mean(jaccards)),
        "near_equal_fraction_gap_below_0_01": float(np.mean(np.asarray(residual_spreads) < 0.01)),
    }
    return result


def _half_carrier_scene(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    phase = 2.0 * math.pi * (x / 3.0 + y / 6.0)
    return 0.5 + 0.35 * DIRECTIONS["c1_r_minus_b"][:, None, None] * np.cos(phase)[None, :, :]


def _support_transition_scene(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    wave = np.cos(2.0 * math.pi * (x / 12.0 + y / 24.0))[None, :, :]
    left = DIRECTIONS["c1_r_minus_b"][:, None, None] * wave
    right = DIRECTIONS["c2_green_opponent"][:, None, None] * wave
    return 0.5 + 0.28 * np.where((x < size // 2)[None, :, :], left, right)


def synthetic_context_study() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scenes = synthetic_edge_cases(96)
    selected = {
        name: scenes[name]
        for name in (
            "vertical_red_green",
            "saturated_red_gray",
            "diagonal_red_green",
            "abc_saturated_edges",
            "abc_frequency_sweep",
        )
    }
    selected["half_carrier"] = _half_carrier_scene()
    selected["support_transition"] = _support_transition_scene()
    result: dict[str, object] = {}
    confidence_samples = []
    confidence_grids = {}
    for name, rgb in selected.items():
        tracks, origins = _scene_tracks(rgb)
        modes = {}
        detailed = {}
        for mode in CONTEXT_WEIGHTS:
            metrics, rows = _aggregate_selection(tracks, mode)
            modes[mode] = metrics
            detailed[mode] = rows
        iterative_metrics, _iterative_rows = _aggregate_selection(tracks, "full", iterative=True)
        proposals = _proposal_summary(tracks)
        local_rows = detailed["local_only"]
        full_rows = detailed["full"]
        ambiguous_improved = []
        scene_confidences = []
        for local, contextual, track in zip(
            local_rows, full_rows, [track for track in tracks for _record in track]
        ):
            record = track[local["window_index"]]
            gap = (
                record.candidates[1].normalized_residual - record.candidates[0].normalized_residual
                if len(record.candidates) > 1 else math.inf
            )
            if gap < 0.01:
                ambiguous_improved.append(contextual["coefficient_error"] < local["coefficient_error"] - 1e-12)
            margin = float(contextual["diagnostics"]["contextual_margin"])
            agreement = 1.0 - float(contextual["diagnostics"]["support"])
            phase_agreement = 1.0 - float(contextual["diagnostics"]["phase"])
            chosen_record = record.candidates[int(contextual["candidate_index"])]
            condition = chosen_record.condition
            condition_score = 0.0 if not math.isfinite(condition) else 1.0 / (1.0 + math.log1p(condition))
            confidence = max(0.0, margin) + 0.5 * agreement + 0.75 * phase_agreement + 0.25 * condition_score
            confidence_samples.append(
                {
                    "coefficient_error": contextual["coefficient_error"],
                    "confidence": confidence,
                    "correct": contextual["correct"],
                    "scene": name,
                }
            )
            scene_confidences.append(confidence)
        confidence_grids[name] = _window_confidence_grid(origins, full_rows)
        result[name] = {
            "ambiguous_context_improvement_fraction": float(np.mean(ambiguous_improved)) if ambiguous_improved else None,
            "context_modes": modes,
            "iterative_full": iterative_metrics,
            "proposals": proposals,
            "contextual_confidence_mean": float(np.mean(scene_confidences)),
            "windows": len(origins),
            "families_per_window": len(tracks),
        }
        if name == "abc_saturated_edges":
            region_tracks, _region_origins = _scene_tracks(rgb, shift=24)
            region_local_rows = _aggregate_selection(region_tracks, "local_only")[1]
            region_context_rows = _aggregate_selection(region_tracks, "full")[1]
            center = 48
            edge_local, edge_context, corner_local, corner_context = [], [], [], []
            for local, contextual in zip(region_local_rows, region_context_rows):
                x, y = local["origin"]
                cross_x = x < center < x + 48
                cross_y = y < center < y + 48
                if not (cross_x or cross_y):
                    continue
                target_local, target_context = (
                    (corner_local, corner_context) if cross_x and cross_y else (edge_local, edge_context)
                )
                target_local.append(float(local["coefficient_error"]))
                target_context.append(float(contextual["coefficient_error"]))
            result[name]["region_breakdown"] = {
                "corner": {
                    "local_error": float(np.mean(corner_local)) if corner_local else None,
                    "context_error": float(np.mean(corner_context)) if corner_context else None,
                    "families": len(corner_local),
                },
                "edge": {
                    "local_error": float(np.mean(edge_local)) if edge_local else None,
                    "context_error": float(np.mean(edge_context)) if edge_context else None,
                    "families": len(edge_local),
                },
            }
        if name == "support_transition":
            result[name]["transition_localization"] = _transition_localization(
                tracks, detailed["full"]
            )
    return {
        "confidence_calibration": confidence_calibration(confidence_samples),
        "scenes": result,
    }, confidence_grids


def _window_confidence_grid(
    origins: Sequence[tuple[int, int]],
    rows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    xs = sorted({origin[0] for origin in origins})
    ys = sorted({origin[1] for origin in origins})
    values: dict[tuple[int, int], list[float]] = {origin: [] for origin in origins}
    for row in rows:
        diagnostics = row["diagnostics"]
        score = (
            max(0.0, float(diagnostics["contextual_margin"]))
            + 0.5 * (1.0 - float(diagnostics["support"]))
            + 0.75 * (1.0 - float(diagnostics["phase"]))
        )
        values[tuple(row["origin"])].append(score)
    return np.asarray(
        [[float(np.mean(values[(x, y)])) for x in xs] for y in ys], dtype=np.float64
    )


def _transition_localization(
    tracks: Sequence[Sequence[WindowFamily]],
    selected_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows_by_track: dict[int, list[Mapping[str, object]]] = {}
    for row in selected_rows:
        rows_by_track.setdefault(int(row["track_index"]), []).append(row)
    errors = []
    detected = 0
    evaluated = 0
    for track_index, track in enumerate(tracks):
        lookup = {record.origin: record for record in track}
        selected = {tuple(row["origin"]): int(row["candidate_index"]) for row in rows_by_track[track_index]}
        for y in sorted({origin[1] for origin in lookup}):
            origins = sorted((origin for origin in lookup if origin[1] == y), key=lambda value: value[0])
            if len(origins) < 3:
                continue
            truth_signatures = [int(np.argmax(np.abs(lookup[origin].truth))) for origin in origins]
            chosen_signatures = [
                lookup[origin].candidates[selected[origin]].support[
                    int(np.argmax(np.abs(lookup[origin].candidates[selected[origin]].coefficients)))
                ]
                for origin in origins
            ]
            truth_changes = [index for index in range(1, len(origins)) if truth_signatures[index] != truth_signatures[index - 1]]
            chosen_changes = [index for index in range(1, len(origins)) if chosen_signatures[index] != chosen_signatures[index - 1]]
            if not truth_changes:
                continue
            evaluated += 1
            if chosen_changes:
                detected += 1
                errors.append(min(abs(value - truth_changes[0]) for value in chosen_changes) * 12.0)
            else:
                errors.append(24.0)
    return {
        "detected_fraction": detected / evaluated if evaluated else None,
        "evaluated_rows": evaluated,
        "mean_absolute_error_pixels": float(np.mean(errors)) if errors else None,
    }


def confidence_calibration(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ordered = sorted(samples, key=lambda value: float(value["confidence"]), reverse=True)
    curve = []
    for retained in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, int(round(len(ordered) * retained)))
        subset = ordered[:count]
        curve.append(
            {
                "coefficient_error_mean": float(np.mean([float(value["coefficient_error"]) for value in subset])),
                "exact_support_rate": float(np.mean([bool(value["correct"]) for value in subset])),
                "false_confidence_rate": float(np.mean([not bool(value["correct"]) for value in subset])),
                "retained_fraction": retained,
            }
        )
    errors = [float(row["coefficient_error_mean"]) for row in curve]
    return {
        "curve": curve,
        "error_monotone_with_retained_fraction": all(right >= left - 1e-12 for left, right in zip(errors, errors[1:])),
        "samples": len(samples),
    }


def null_context_study() -> dict[str, object]:
    """Test whether the known exact null remains coherent after translation."""

    support = (0, 2, 28, 31, 34)
    target = np.zeros(5, dtype=np.complex128)
    target[0] = 1.0
    base = dictionary_for_origin((0, 0))[:, support]
    alternative = np.linalg.lstsq(base[:, 1:], base[:, 0], rcond=None)[0]
    competitor = np.concatenate(([0.0 + 0.0j], alternative))
    frequencies = np.asarray(
        [
            _support_frequency_for_base(atom)
            for atom in support
        ]
    )

    def stacked(origins: Sequence[tuple[int, int]]) -> tuple[np.ndarray, float, float]:
        blocks = []
        truth_observation = []
        competitor_observation = []
        for origin in origins:
            phases = np.exp(2j * math.pi * (frequencies[:, 0] * origin[0] + frequencies[:, 1] * origin[1]))
            block = dictionary_for_origin(origin)[:, support] * phases[None, :]
            blocks.append(block)
            truth_observation.append(block @ target)
            competitor_observation.append(block @ competitor)
        matrix = np.vstack(blocks)
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        relative = float(
            np.linalg.norm(np.concatenate(truth_observation) - np.concatenate(competitor_observation))
            / np.linalg.norm(np.concatenate(truth_observation))
        )
        sigma_min = float(singular[-1])
        return matrix, relative, sigma_min

    cases = {}
    for name, origins in (
        ("single", ((0, 0),)),
        ("aligned_6", ((0, 0), (6, 0), (12, 0), (0, 6))),
        ("mixed_non_aligned", ((0, 0), (1, 0), (0, 1), (7, 5))),
    ):
        matrix, relative, sigma_min = stacked(origins)
        cases[name] = {
            "nullity": len(support) - int(np.linalg.matrix_rank(matrix)),
            "rank": int(np.linalg.matrix_rank(matrix)),
            "smallest_singular_value": sigma_min,
            "truth_competitor_relative_observation_difference": relative,
            "windows": len(origins),
        }
    return {
        "cases": cases,
        "support": list(support),
        "interpretation": "stacking assumes one globally coherent sinusoidal coefficient per support atom",
    }


def grouped_half_carrier_context_control() -> dict[str, object]:
    """Verify that context's phase term preserves the grouped half-carrier trajectory."""

    size = 24
    target = (8, 4)
    family_index, coordinates = next(
        (index, family_coordinates(size, family))
        for index, family in enumerate(alias_families(size))
        if target in family
    )
    carrier_index = coordinates.index(target)
    atom = carrier_index * 3 + 1
    phase_errors = []
    wrong_errors = []
    for phase in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
        coefficient = np.exp(1j * phase)
        left = _single_hypothesis_record(size, family_index, coordinates, atom, (0, 0), coefficient)
        fx, fy = centered_frequency(target[0], size), centered_frequency(target[1], size)
        translated_phase = phase + 2.0 * math.pi * (fx * 6 + fy * 12)
        right = _single_hypothesis_record(
            size, family_index, coordinates, atom, (6, 12), np.exp(1j * translated_phase)
        )
        wrong = _single_hypothesis_record(
            size, family_index, coordinates, atom, (6, 12), np.exp(1j * (translated_phase + math.pi / 2.0))
        )
        phase_errors.append(compatibility(left, left.candidates[0], right, right.candidates[0]).phase)
        wrong_errors.append(compatibility(left, left.candidates[0], wrong, wrong.candidates[0]).phase)
    return {
        "frequency_cycles_per_pixel": [1.0 / 3.0, 1.0 / 6.0],
        "phases": 72,
        "true_trajectory_phase_error_maximum": max(phase_errors),
        "quarter_cycle_wrong_trajectory_phase_error_median": float(np.median(wrong_errors)),
    }


def _single_hypothesis_record(
    size: int,
    family_index: int,
    coordinates: tuple[tuple[int, int], ...],
    atom: int,
    origin: tuple[int, int],
    coefficient: complex,
) -> WindowFamily:
    hypothesis = Hypothesis(
        coefficients=np.asarray((coefficient,)),
        condition=1.0,
        energy=float(abs(coefficient) ** 2),
        normalized_residual=0.0,
        sigma_min=1.0,
        singular_values=(1.0,),
        support=(atom,),
    )
    truth = np.zeros(54, dtype=np.complex128)
    truth[atom] = coefficient
    return WindowFamily(
        candidates=(hypothesis,), coordinates=coordinates, family_index=family_index,
        observation=np.zeros(18, dtype=np.complex128), origin=origin, size=size,
        truth=truth, truth_support=(atom,),
    )


def _support_frequency_for_base(atom: int) -> tuple[float, float]:
    carrier = CARRIER_INDICES[atom // 3]
    return centered_frequency(carrier[0], 6), centered_frequency(carrier[1], 6)


def parameter_ablation() -> dict[str, object]:
    """Small controlled grid; it intentionally does not explode all combinations."""

    scene = synthetic_edge_cases(192)["saturated_red_gray"]
    output: dict[str, object] = {"window_shift": [], "support_and_beam": []}
    for window_size in (24, 48, 96):
        for shift in (6, 12, 24, 7):
            if shift >= window_size:
                continue
            tracks, _origins = _scene_tracks(
                scene,
                window_size=window_size,
                shift=shift,
                maximum_support=4,
                proposal_count=5,
                beam_width=4,
                family_count=2,
            )
            local, _ = _aggregate_selection(tracks, "local_only")
            context, _ = _aggregate_selection(tracks, "full")
            output["window_shift"].append(
                {
                    "context_error": context["coefficient_error_mean"],
                    "context_exact_rate": context["exact_support_rate"],
                    "local_error": local["coefficient_error_mean"],
                    "local_exact_rate": local["exact_support_rate"],
                    "shift": shift,
                    "window_size": window_size,
                    "windows": sum(len(track) for track in tracks),
                }
            )
    base_spectra = extract_window_spectra(scene, (72, 72), 48)
    energies = sorted(
        ((float(np.sum(np.abs(value[1]) ** 2)), index) for index, value in enumerate(base_spectra)),
        reverse=True,
    )[:4]
    for maximum_support in (3, 4, 5):
        for beam_width in (4, 8, 16):
            inclusions = {1: [], 3: [], 5: [], 8: []}
            for _energy, family_index in energies:
                record = make_window_family(
                    base_spectra[family_index], family_index, (72, 72), 48,
                    maximum_support=maximum_support, proposal_count=8, beam_width=beam_width,
                )
                for count in inclusions:
                    inclusions[count].append(
                        any(set(value.support) == set(record.truth_support) for value in record.candidates[:count])
                    )
            output["support_and_beam"].append(
                {
                    "beam_width": beam_width,
                    "maximum_support": maximum_support,
                    "truth_in_top_n": {str(count): float(np.mean(values)) for count, values in inclusions.items()},
                }
            )
    return output


def noise_and_propagation_study() -> dict[str, object]:
    scene = synthetic_edge_cases(96)["saturated_red_gray"]
    tracks, _origins = _scene_tracks(scene, family_count=3)
    rng = np.random.default_rng(0x434F4E54)
    noise_rows = []
    for snr_db in (math.inf, 40.0, 30.0, 20.0):
        noisy_tracks = []
        for track in tracks:
            noisy_track = []
            for record in track:
                observation = record.observation.copy()
                if math.isfinite(snr_db):
                    noise = rng.normal(size=observation.shape) + 1j * rng.normal(size=observation.shape)
                    noise *= np.linalg.norm(observation) / (np.linalg.norm(noise) * 10.0 ** (snr_db / 20.0))
                    observation += noise
                spectra = (observation, record.truth, record.coordinates)
                noisy_track.append(
                    make_window_family(
                        spectra, record.family_index, record.origin, record.size,
                        maximum_support=5, proposal_count=8, beam_width=8,
                    )
                )
            noisy_tracks.append(noisy_track)
        local, _ = _aggregate_selection(noisy_tracks, "local_only")
        one_pass, _ = _aggregate_selection(noisy_tracks, "full")
        iterative, _ = _aggregate_selection(noisy_tracks, "full", iterative=True)
        noise_rows.append(
            {
                "iterative_coefficient_error": iterative["coefficient_error_mean"],
                "iterative_exact_rate": iterative["exact_support_rate"],
                "local_coefficient_error": local["coefficient_error_mean"],
                "local_exact_rate": local["exact_support_rate"],
                "one_pass_coefficient_error": one_pass["coefficient_error_mean"],
                "one_pass_exact_rate": one_pass["exact_support_rate"],
                "snr_db": "infinite" if not math.isfinite(snr_db) else snr_db,
            }
        )

    corrupted_tracks = [list(track) for track in tracks]
    center = len(corrupted_tracks[0]) // 2
    for track_index, track in enumerate(corrupted_tracks):
        record = track[center]
        corrupt = record.observation + 2.5 * np.linalg.norm(record.observation) * (
            rng.normal(size=18) + 1j * rng.normal(size=18)
        ) / math.sqrt(36.0)
        track[center] = make_window_family(
            (corrupt, record.truth, record.coordinates), record.family_index,
            record.origin, record.size, maximum_support=5, proposal_count=8, beam_width=8,
        )
    clean_one, _ = _aggregate_selection(tracks, "full")
    corrupt_one, corrupt_one_rows = _aggregate_selection(corrupted_tracks, "full")
    corrupt_iter, corrupt_iter_rows = _aggregate_selection(corrupted_tracks, "full", iterative=True)
    neighbor_changes_one = _neighbor_selection_changes(tracks, corrupted_tracks, corrupt_one_rows, center)
    neighbor_changes_iter = _neighbor_selection_changes(tracks, corrupted_tracks, corrupt_iter_rows, center)
    return {
        "error_propagation": {
            "clean_one_pass_exact_rate": clean_one["exact_support_rate"],
            "corrupt_iterative_exact_rate": corrupt_iter["exact_support_rate"],
            "corrupt_one_pass_exact_rate": corrupt_one["exact_support_rate"],
            "neighbor_selection_change_fraction_iterative": neighbor_changes_iter,
            "neighbor_selection_change_fraction_one_pass": neighbor_changes_one,
        },
        "noise": noise_rows,
    }


def _neighbor_selection_changes(
    clean_tracks: Sequence[Sequence[WindowFamily]],
    corrupted_tracks: Sequence[Sequence[WindowFamily]],
    rows: Sequence[Mapping[str, object]],
    center: int,
) -> float:
    changes = []
    flat_index = 0
    for clean, corrupt in zip(clean_tracks, corrupted_tracks):
        clean_selection, _ = select_track(clean, "full")
        for index, _record in enumerate(corrupt):
            choice = int(rows[flat_index]["candidate_index"])
            if index != center:
                changes.append(
                    set(corrupt[index].candidates[choice].support)
                    != set(clean[index].candidates[clean_selection[index]].support)
                )
            flat_index += 1
    return float(np.mean(changes)) if changes else 0.0


def _structure_class(source: Mapping[str, object], patch: np.ndarray) -> str:
    luminance = np.mean(patch, axis=0)
    gx = np.diff(luminance, axis=1)
    gy = np.diff(luminance, axis=0)
    gradient = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    variation = float(np.var(luminance))
    chroma = float(np.mean(np.std(patch, axis=0)))
    category = str(source["category"])
    if variation < 2e-5 and gradient < 0.002:
        return "smooth"
    if chroma > 0.12 and gradient > 0.02:
        return "saturated_chromatic_edge"
    gx_energy = float(np.mean(gx * gx))
    gy_energy = float(np.mean(gy * gy))
    if max(gx_energy, gy_energy) > 4.0 * max(min(gx_energy, gy_energy), 1e-12):
        return "single_edge"
    if "fine_colored" in category:
        return "irregular_texture"
    if "portrait" in category:
        return "corner"
    if "texture" in category:
        return "foliage_like"
    return "periodic_texture" if variation > 0.01 else "irregular_texture"


def natural_context_study(sources: Sequence[Mapping[str, object]]) -> dict[str, object]:
    source_rows = []
    all_local = []
    all_context = []
    all_records: list[WindowFamily] = []
    all_tracks: list[list[WindowFamily]] = []
    all_structures = []
    mode_rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in CONTEXT_WEIGHTS}
    for source in sources:
        rgb = np.moveaxis(srgb_to_linear(np.asarray(source["rgb"], dtype=np.float64)), -1, 0)
        tracks, origins = _scene_tracks(rgb, family_count=4)
        all_tracks.extend(tracks)
        per_mode = {}
        per_mode_rows = {}
        for mode in CONTEXT_WEIGHTS:
            metrics, rows = _aggregate_selection(tracks, mode)
            per_mode[mode] = metrics
            per_mode_rows[mode] = rows
            mode_rows[mode].extend(rows)
        local_metrics = per_mode["local_only"]
        context_metrics = per_mode["full"]
        local_rows = per_mode_rows["local_only"]
        context_rows = per_mode_rows["full"]
        all_local.extend(local_rows)
        all_context.extend(context_rows)
        all_records.extend(record for track in tracks for record in track)
        structure_by_origin = {
            origin: _structure_class(
                source,
                rgb[:, origin[1] : origin[1] + 48, origin[0] : origin[0] + 48],
            )
            for origin in origins
        }
        all_structures.extend(
            structure_by_origin[tuple(row["origin"])] for row in local_rows
        )
        source_rows.append(
            {
                "category": source["category"],
                "context_modes": per_mode,
                "name": source["name"],
                "sha256": source["sha256"],
            }
        )

    ambiguous = []
    improved = []
    regressed = []
    gaps = []
    hypothesis_counts = []
    structure_values: dict[str, dict[str, list[float]]] = {}
    confidence_samples = []
    for record, local, contextual, structure in zip(all_records, all_local, all_context, all_structures):
        gap = record.candidates[1].normalized_residual - record.candidates[0].normalized_residual
        is_ambiguous = gap < 0.01
        ambiguous.append(is_ambiguous)
        gaps.append(gap)
        hypothesis_counts.append(sum(candidate.normalized_residual <= record.candidates[0].normalized_residual + 0.01 for candidate in record.candidates))
        difference = float(local["coefficient_error"] - contextual["coefficient_error"])
        if is_ambiguous:
            improved.append(difference > 1e-12)
            regressed.append(difference < -1e-12)
        bucket = structure_values.setdefault(structure, {"difference": [], "local": [], "context": []})
        bucket["difference"].append(difference)
        bucket["local"].append(float(local["coefficient_error"]))
        bucket["context"].append(float(contextual["coefficient_error"]))
        diagnostics = contextual["diagnostics"]
        confidence_samples.append(
            {
                "coefficient_error": contextual["coefficient_error"],
                "confidence": max(0.0, float(diagnostics["contextual_margin"]))
                    + 0.5 * (1.0 - float(diagnostics["support"]))
                    + 0.75 * (1.0 - float(diagnostics["phase"])),
                "correct": contextual["correct"],
            }
        )
    local_mean = float(np.mean([float(value["coefficient_error"]) for value in all_local]))
    context_mean = float(np.mean([float(value["coefficient_error"]) for value in all_context]))
    oracle_errors = []
    for record in all_records:
        fit = least_squares_support(record.observation, record.truth_support, dictionary_for_origin(record.origin))
        oracle_errors.append(coefficient_relative_error(range(54), record.truth, record.truth_support, fit.coefficients))
    oracle_mean = float(np.mean(oracle_errors))
    denominator = local_mean - oracle_mean
    aggregate_modes = {}
    for mode, rows in mode_rows.items():
        aggregate_modes[mode] = {
            "coefficient_error_mean": float(np.mean([float(row["coefficient_error"]) for row in rows])),
            "exact_support_rate": float(np.mean([bool(row["correct"]) for row in rows])),
        }
    return {
        "ambiguous_family_fraction": float(np.mean(ambiguous)),
        "ambiguous_improved_fraction": float(np.mean(improved)) if improved else None,
        "ambiguous_regressed_fraction": float(np.mean(regressed)) if regressed else None,
        "confidence_calibration": confidence_calibration(confidence_samples),
        "context_coefficient_error_mean": context_mean,
        "context_modes": aggregate_modes,
        "fraction_blind_to_oracle_gap_closed": (local_mean - context_mean) / denominator if denominator > 1e-15 else None,
        "local_coefficient_error_mean": local_mean,
        "mean_plausible_hypotheses_within_0_01_residual": float(np.mean(hypothesis_counts)),
        "oracle_top_support_error_mean": oracle_mean,
        "proposals": _proposal_summary(all_tracks),
        "sources": source_rows,
        "structures": {
            name: {
                "context_error": float(np.mean(values["context"])),
                "context_improvement": float(np.mean(values["difference"])),
                "families": len(values["difference"]),
                "local_error": float(np.mean(values["local"])),
            }
            for name, values in sorted(structure_values.items())
        },
    }


def build_context_characterization(
    natural_sources: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    synthetic, confidence_grids = synthetic_context_study()
    payload = {
        "ablation": parameter_ablation(),
        "alias_operator_binding": {
            "columns": 54,
            "rows": 18,
            "sparse_recovery_format": "rawtherapee-xtrans-sparse-alias-recovery-v1",
        },
        "context_weights": CONTEXT_WEIGHTS,
        "format": MODEL_FORMAT,
        "grouped_half_carrier_context_control": grouped_half_carrier_context_control(),
        "natural": natural_context_study(natural_sources),
        "noise_and_error_propagation": noise_and_propagation_study(),
        "null_context": null_context_study(),
        "synthetic": synthetic,
        "window_contract": {
            "neighbor_connectivity": 4,
            "primary_proposal_count": 8,
            "primary_shift": 12,
            "primary_window_size": 48,
            "taper": "separable square-root periodic Hann",
        },
    }
    return payload, confidence_grids

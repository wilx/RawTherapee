"""Deterministic classical sparse coding for small RGB/X-Trans patches.

The implementation is deliberately self contained.  Dictionary learning uses
alternating FISTA sparse coding and a regularized method-of-optimal-directions
(MOD) update; inference uses ordinary matching pursuit (OMP).  Neither stage
uses a neural model or an image-content classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import dct

from tools.xtrans_mlri_internal.dataset import cfa_for_origin, origin_cells


OPPONENT = np.asarray(
    (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, -1.0),
        (1.0, -2.0, 1.0),
    ),
    dtype=np.float64,
)
OPPONENT /= np.linalg.norm(OPPONENT, axis=1, keepdims=True)


@dataclass(frozen=True)
class SparseResult:
    coefficients: np.ndarray
    support: np.ndarray
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class DictionaryTraining:
    dictionary: np.ndarray
    objectives: tuple[float, ...]
    effective_nonzero: tuple[float, ...]


def normalize_atoms(dictionary: np.ndarray) -> np.ndarray:
    checked = np.asarray(dictionary, dtype=np.float64).copy()
    if checked.ndim != 2 or not np.isfinite(checked).all():
        raise ValueError("dictionary must be a finite matrix")
    norms = np.linalg.norm(checked, axis=0)
    if np.any(norms <= 1e-12):
        raise ValueError("dictionary contains a degenerate atom")
    checked /= norms
    return checked


def dct_dictionary(
    patch_size: int, *, color_basis: str = "opponent",
) -> np.ndarray:
    """Return a complete orthonormal spatial-DCT/color dictionary."""

    if patch_size < 3 or patch_size % 2 != 1:
        raise ValueError("patch size must be odd and at least three")
    spatial = dct(np.eye(patch_size), type=2, norm="ortho", axis=0)
    spatial_2d = np.asarray(
        [np.outer(spatial[:, y], spatial[:, x]).ravel()
         for y in range(patch_size) for x in range(patch_size)],
        dtype=np.float64,
    ).T
    if color_basis == "opponent":
        colors = OPPONENT
    elif color_basis == "rgb":
        colors = np.eye(3, dtype=np.float64)
    else:
        raise ValueError("color basis must be rgb or opponent")
    # CHW vector order: a color vector multiplied by one spatial atom.
    atoms = [np.kron(color, spatial_2d[:, index])
             for color in colors
             for index in range(spatial_2d.shape[1])]
    return normalize_atoms(np.stack(atoms, axis=1))


def observation_indices(
    patch_size: int, origin_x: int, origin_y: int,
) -> np.ndarray:
    cfa = cfa_for_origin(patch_size, patch_size, origin_x, origin_y)
    spatial = np.arange(patch_size * patch_size, dtype=np.int64)
    return cfa.ravel().astype(np.int64) * (patch_size * patch_size) + spatial


def phase_index(origin_x: int, origin_y: int) -> int:
    key = cfa_for_origin(6, 6, origin_x, origin_y).tobytes()
    keys = {
        cfa_for_origin(6, 6, x, y).tobytes(): index
        for index, (x, y) in enumerate(origin_cells())
    }
    return keys[key]


def observable_dc(vector: np.ndarray, indices: np.ndarray) -> float:
    return float(np.mean(np.asarray(vector, dtype=np.float64)[indices]))


def remove_observable_dc(
    vector: np.ndarray, indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    checked = np.asarray(vector, dtype=np.float64)
    dc = observable_dc(checked, indices)
    return checked - dc, dc


def restore_observed(
    reconstruction: np.ndarray, truth_vector: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    result = np.asarray(reconstruction, dtype=np.float64).copy()
    result[indices] = np.asarray(truth_vector, dtype=np.float64)[indices]
    return result


def omp(
    dictionary: np.ndarray,
    signal: np.ndarray,
    sparsity: int,
    *,
    ridge: float = 1e-10,
) -> SparseResult:
    """Orthogonal matching pursuit with stable tie-breaking."""

    d = np.asarray(dictionary, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if d.ndim != 2 or y.shape != (d.shape[0],) or not np.isfinite(d).all() or not np.isfinite(y).all():
        raise ValueError("invalid OMP input")
    if sparsity < 1 or sparsity > min(d.shape):
        raise ValueError("invalid OMP sparsity")
    norms = np.linalg.norm(d, axis=0)
    selectable = norms > 1e-12
    scaled = np.zeros_like(d)
    scaled[:, selectable] = d[:, selectable] / norms[selectable]
    support: list[int] = []
    residual = y.copy()
    history = [float(np.dot(residual, residual))]
    coefficients = np.empty(0, dtype=np.float64)
    for _ in range(sparsity):
        scores = np.abs(scaled.T @ residual)
        scores[~selectable] = -1.0
        if support:
            scores[np.asarray(support, dtype=np.int64)] = -1.0
        selected = int(np.argmax(scores))
        if scores[selected] <= 1e-14:
            break
        support.append(selected)
        active = d[:, support]
        gram = active.T @ active + ridge * np.eye(len(support))
        coefficients = np.linalg.solve(gram, active.T @ y)
        residual = y - active @ coefficients
        value = float(np.dot(residual, residual))
        history.append(min(value, history[-1]))
    full = np.zeros(d.shape[1], dtype=np.float64)
    if support:
        full[np.asarray(support, dtype=np.int64)] = coefficients
    return SparseResult(full, np.asarray(support, dtype=np.int64), tuple(history))


def oracle_support_reconstruction(
    dictionary: np.ndarray,
    truth_vector: np.ndarray,
    indices: np.ndarray,
    support: np.ndarray,
    *,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float | int]]:
    d = np.asarray(dictionary, dtype=np.float64)
    truth = np.asarray(truth_vector, dtype=np.float64)
    selected = np.asarray(support, dtype=np.int64)
    if selected.size == 0:
        reconstruction = np.zeros_like(truth)
        return reconstruction, {"condition_number": 1.0, "rank": 0, "sigma_min": 0.0}
    sensing = d[np.asarray(indices, dtype=np.int64)][:, selected]
    singular = np.linalg.svd(sensing, compute_uv=False)
    rank = int(np.linalg.matrix_rank(sensing))
    condition = float(singular[0] / max(singular[-1], np.finfo(np.float64).tiny))
    gram = sensing.T @ sensing + ridge * np.eye(selected.size)
    beta = np.linalg.solve(gram, sensing.T @ truth[indices])
    return d[:, selected] @ beta, {
        "condition_number": condition,
        "rank": rank,
        "sigma_min": float(singular[-1]),
    }


def blind_cfa_reconstruction(
    dictionary: np.ndarray,
    truth_vector: np.ndarray,
    indices: np.ndarray,
    sparsity: int,
) -> tuple[np.ndarray, SparseResult]:
    d = np.asarray(dictionary, dtype=np.float64)
    observed = d[np.asarray(indices, dtype=np.int64)]
    result = omp(observed, np.asarray(truth_vector, dtype=np.float64)[indices], sparsity)
    return d @ result.coefficients, result


def lasso_cfa_reconstruction(
    dictionary: np.ndarray,
    truth_vector: np.ndarray,
    indices: np.ndarray,
    regularization: float,
    *,
    elastic_ridge: float = 0.0,
    iterations: int = 100,
) -> tuple[np.ndarray, SparseResult]:
    """CFA-domain FISTA LASSO/elastic-net control for one patch."""

    d = np.asarray(dictionary, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    sensing = d[selected]
    signal = np.asarray(truth_vector, dtype=np.float64)[selected]
    if regularization <= 0.0 or elastic_ridge < 0.0 or iterations < 1:
        raise ValueError("invalid LASSO configuration")
    lipschitz = float(np.linalg.norm(sensing, ord=2) ** 2 + elastic_ridge)
    step = 1.0 / max(lipschitz, np.finfo(np.float64).tiny)
    coefficients = np.zeros(d.shape[1], dtype=np.float64)
    accelerated = coefficients.copy()
    momentum = 1.0
    history = []
    for _ in range(iterations):
        residual = sensing @ accelerated - signal
        gradient = sensing.T @ residual + elastic_ridge * accelerated
        updated = _soft_threshold(
            accelerated - step * gradient, regularization * step,
        )
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        coefficients = updated
        momentum = next_momentum
        current_residual = sensing @ coefficients - signal
        objective = (
            0.5 * float(np.dot(current_residual, current_residual))
            + regularization * float(np.sum(np.abs(coefficients)))
            + 0.5 * elastic_ridge * float(np.dot(coefficients, coefficients))
        )
        history.append(objective)
    support = np.nonzero(np.abs(coefficients) > 1e-6)[0].astype(np.int64)
    return d @ coefficients, SparseResult(
        coefficients, support, tuple(history),
    )


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def fista_codes(
    dictionary: np.ndarray,
    samples: np.ndarray,
    regularization: float,
    *,
    iterations: int,
) -> np.ndarray:
    """Solve all L1 sparse codes together; samples are dimension by count."""

    d = np.asarray(dictionary, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    if regularization <= 0.0 or iterations < 1 or x.ndim != 2 or x.shape[0] != d.shape[0]:
        raise ValueError("invalid FISTA configuration")
    lipschitz = float(np.linalg.norm(d, ord=2) ** 2)
    step = 1.0 / max(lipschitz, np.finfo(np.float64).tiny)
    coefficients = np.zeros((d.shape[1], x.shape[1]), dtype=np.float64)
    accelerated = coefficients.copy()
    momentum = 1.0
    for _ in range(iterations):
        gradient = d.T @ (d @ accelerated - x)
        updated = _soft_threshold(accelerated - step * gradient, regularization * step)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        coefficients = updated
        momentum = next_momentum
    return coefficients


def train_mod_dictionary(
    samples: np.ndarray,
    atom_count: int,
    *,
    regularization: float = 0.015,
    outer_iterations: int = 6,
    coding_iterations: int = 30,
    ridge: float = 1e-5,
    seed: int = 0x58445247,
) -> DictionaryTraining:
    """Train a deterministic overcomplete dictionary using L1/MOD alternation."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or atom_count < x.shape[0] or x.shape[1] < atom_count:
        raise ValueError("invalid training matrix or atom count")
    base = dct_dictionary(int(round(math.sqrt(x.shape[0] / 3.0))))
    rng = np.random.default_rng(seed)
    atoms = [base[:, index] for index in range(base.shape[1])]
    order = rng.permutation(x.shape[1])
    for sample_index in order:
        if len(atoms) >= atom_count:
            break
        candidate = x[:, sample_index].copy()
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-8:
            atoms.append(candidate / norm)
    if len(atoms) != atom_count:
        raise RuntimeError("could not initialize all dictionary atoms")
    dictionary = np.stack(atoms, axis=1)
    objectives: list[float] = []
    effective: list[float] = []
    for _ in range(outer_iterations):
        coefficients = fista_codes(
            dictionary, x, regularization, iterations=coding_iterations,
        )
        gram = coefficients @ coefficients.T + ridge * np.eye(atom_count)
        updated = np.linalg.solve(gram, coefficients @ x.T).T
        norms = np.linalg.norm(updated, axis=0)
        weak = norms <= 1e-10
        if np.any(weak):
            residual = x - dictionary @ coefficients
            ranking = np.argsort(np.sum(residual * residual, axis=0), kind="stable")[::-1]
            for atom, sample_index in zip(np.nonzero(weak)[0], ranking):
                updated[:, atom] = residual[:, sample_index]
        dictionary = normalize_atoms(updated)
        coefficients = fista_codes(
            dictionary, x, regularization, iterations=coding_iterations,
        )
        residual = dictionary @ coefficients - x
        objective = 0.5 * float(np.sum(residual * residual)) + regularization * float(np.sum(np.abs(coefficients)))
        objectives.append(objective / x.shape[1])
        effective.append(float(np.mean(np.sum(np.abs(coefficients) > 1e-5, axis=0))))
    return DictionaryTraining(dictionary, tuple(objectives), tuple(effective))


def dictionary_coherence(dictionary: np.ndarray) -> float:
    d = normalize_atoms(dictionary)
    gram = np.abs(d.T @ d)
    np.fill_diagonal(gram, 0.0)
    return float(np.max(gram))


def masked_coherence(dictionary: np.ndarray, indices: np.ndarray) -> float:
    projected = np.asarray(dictionary, dtype=np.float64)[indices]
    norms = np.linalg.norm(projected, axis=0)
    valid = norms > 1e-12
    projected = projected[:, valid] / norms[valid]
    gram = np.abs(projected.T @ projected)
    np.fill_diagonal(gram, 0.0)
    return float(np.max(gram))


__all__ = (
    "DictionaryTraining", "SparseResult", "blind_cfa_reconstruction",
    "dct_dictionary", "dictionary_coherence", "fista_codes",
    "lasso_cfa_reconstruction", "masked_coherence", "normalize_atoms", "observable_dc", "omp",
    "oracle_support_reconstruction", "phase_index", "remove_observable_dc",
    "restore_observed", "train_mod_dictionary", "observation_indices",
)

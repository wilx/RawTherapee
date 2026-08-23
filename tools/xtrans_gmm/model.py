"""Joint RGB Gaussian mixtures conditioned on physical X-Trans samples.

This is an independent research implementation of the local conditional
Gaussian step described by Sandeep and Jacob, IEEE SPL 2019.  Unlike their
headline algorithm, the primary experiment also evaluates the exact posterior
mean over components.  It does not implement overlapping-patch aggregation or
the paper's image-adaptive GMM update unless the local model passes its gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.mixture import GaussianMixture

from tools.xtrans_sparse_dictionary.dataset import PatchSample
from tools.xtrans_sparse_dictionary.model import observation_indices
from tools.xtrans_mlri_internal.dataset import origin_cells


LOG_2PI = math.log(2.0 * math.pi)
DC_MODES = ("absolute", "observed-scalar", "observed-rgb")


@dataclass(frozen=True)
class JointColorGMM:
    patch_size: int
    component_count: int
    dc_mode: str
    covariance_floor: float
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    converged: bool
    iterations: int
    lower_bound: float
    effective_counts: np.ndarray
    training_sample_count: int
    seed: int


@dataclass(frozen=True)
class PhaseConditional:
    observed_indices: np.ndarray
    sampled_center_channel: int
    means_observed: np.ndarray
    means_center: np.ndarray
    cholesky: tuple[np.ndarray, ...]
    log_determinants: np.ndarray
    center_gains: np.ndarray


@dataclass(frozen=True)
class ConditionalCache:
    model: JointColorGMM
    tau: float
    phases: tuple[PhaseConditional, ...]


@dataclass(frozen=True)
class ConditionalBatch:
    map_rgb: np.ndarray
    mmse_rgb: np.ndarray
    component_rgb: np.ndarray
    responsibilities: np.ndarray
    log_responsibilities: np.ndarray
    map_components: np.ndarray
    entropy: np.ndarray


def _check_patch_size(patch_size: int) -> int:
    if patch_size < 3 or patch_size % 2 != 1:
        raise ValueError("patch size must be odd and at least three")
    return patch_size


def center_indices(patch_size: int) -> np.ndarray:
    checked = _check_patch_size(patch_size)
    area = checked * checked
    center = area // 2
    return np.asarray([channel * area + center for channel in range(3)], dtype=np.int64)


def observable_dc(vector: np.ndarray, indices: np.ndarray, mode: str) -> float:
    checked = np.asarray(vector, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    if mode == "absolute":
        return 0.0
    if mode == "observed-scalar":
        return float(np.mean(checked[selected]))
    if mode == "observed-rgb":
        area = checked.size // 3
        colors = selected // area
        means = [float(np.mean(checked[selected[colors == color]])) for color in range(3)]
        if any(not math.isfinite(value) for value in means):
            raise ValueError("patch does not observe every RGB channel")
        return float(np.mean(means))
    raise ValueError(f"unknown DC mode: {mode}")


def normalize_vector(vector: np.ndarray, indices: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    checked = np.asarray(vector, dtype=np.float64)
    if checked.ndim != 1 or not np.isfinite(checked).all():
        raise ValueError("patch vector must be finite and one-dimensional")
    dc = observable_dc(checked, indices, mode)
    return checked - dc, dc


def sample_matrix(samples: Iterable[PatchSample], mode: str) -> np.ndarray:
    rows = [normalize_vector(sample.vector, sample.indices, mode)[0] for sample in samples]
    if len(rows) < 2:
        raise ValueError("at least two patches are required")
    matrix = np.ascontiguousarray(np.stack(rows), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite training matrix")
    return matrix


def fit_joint_gmm(
    matrix: np.ndarray,
    patch_size: int,
    component_count: int,
    *,
    dc_mode: str,
    covariance_floor: float = 1e-6,
    seed: int = 0x58474D4D,
    maximum_iterations: int = 40,
    tolerance: float = 1e-3,
    covariance_type: str = "full",
) -> JointColorGMM:
    """Fit deterministic EM through scikit-learn.

    The research candidate always uses full covariance.  ``diagonal`` exists
    only for the required joint-covariance ablation and is expanded into the
    same immutable matrix representation used by conditional inference.
    """

    checked_size = _check_patch_size(patch_size)
    x = np.asarray(matrix, dtype=np.float64)
    dimension = 3 * checked_size * checked_size
    if (
        x.ndim != 2
        or x.shape[1] != dimension
        or x.shape[0] < component_count
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid GMM training matrix")
    if (
        component_count < 1
        or covariance_floor <= 0.0
        or dc_mode not in DC_MODES
        or covariance_type not in ("full", "diagonal")
    ):
        raise ValueError("invalid GMM configuration")
    estimator = GaussianMixture(
        n_components=component_count,
        covariance_type="diag" if covariance_type == "diagonal" else "full",
        tol=tolerance,
        reg_covar=covariance_floor,
        max_iter=maximum_iterations,
        n_init=1,
        init_params="k-means++",
        random_state=seed,
    )
    estimator.fit(x)
    responsibilities = estimator.predict_proba(x)
    effective = np.sum(responsibilities, axis=0)
    weights = np.ascontiguousarray(estimator.weights_, dtype=np.float64)
    means = np.ascontiguousarray(estimator.means_, dtype=np.float64)
    fitted_covariances = np.asarray(estimator.covariances_, dtype=np.float64)
    if covariance_type == "diagonal":
        covariances = np.zeros(
            (component_count, dimension, dimension), dtype=np.float64
        )
        diagonal = np.arange(dimension)
        covariances[:, diagonal, diagonal] = fitted_covariances
    else:
        covariances = np.ascontiguousarray(fitted_covariances)
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(means).all()
        or not np.isfinite(covariances).all()
        or np.any(weights <= 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-10
    ):
        raise RuntimeError("invalid fitted GMM")
    return JointColorGMM(
        patch_size=checked_size,
        component_count=component_count,
        dc_mode=dc_mode,
        covariance_floor=covariance_floor,
        weights=weights,
        means=means,
        covariances=covariances,
        converged=bool(estimator.converged_),
        iterations=int(estimator.n_iter_),
        lower_bound=float(estimator.lower_bound_),
        effective_counts=np.ascontiguousarray(effective, dtype=np.float64),
        training_sample_count=x.shape[0],
        seed=seed,
    )


def refine_joint_gmm(
    matrix: np.ndarray,
    model: JointColorGMM,
    *,
    additional_iterations: int = 70,
    tolerance: float = 1e-4,
) -> JointColorGMM:
    """Continue EM from a screened model without repeating initialization."""

    x = np.asarray(matrix, dtype=np.float64)
    if x.shape != (model.training_sample_count, model.means.shape[1]):
        raise ValueError("refinement matrix does not match the screened model")
    precisions = np.stack([
        np.linalg.inv(covariance) for covariance in model.covariances
    ])
    estimator = GaussianMixture(
        n_components=model.component_count,
        covariance_type="full",
        tol=tolerance,
        reg_covar=model.covariance_floor,
        max_iter=additional_iterations,
        n_init=1,
        init_params="random",
        random_state=model.seed,
        weights_init=model.weights,
        means_init=model.means,
        precisions_init=precisions,
    )
    estimator.fit(x)
    effective = np.sum(estimator.predict_proba(x), axis=0)
    return JointColorGMM(
        patch_size=model.patch_size,
        component_count=model.component_count,
        dc_mode=model.dc_mode,
        covariance_floor=model.covariance_floor,
        weights=np.ascontiguousarray(estimator.weights_, dtype=np.float64),
        means=np.ascontiguousarray(estimator.means_, dtype=np.float64),
        covariances=np.ascontiguousarray(estimator.covariances_, dtype=np.float64),
        converged=bool(estimator.converged_),
        iterations=model.iterations + int(estimator.n_iter_),
        lower_bound=float(estimator.lower_bound_),
        effective_counts=np.ascontiguousarray(effective, dtype=np.float64),
        training_sample_count=model.training_sample_count,
        seed=model.seed,
    )


def _phase_origins() -> tuple[tuple[int, int], ...]:
    return origin_cells()


def prepare_conditional_cache(model: JointColorGMM, tau: float) -> ConditionalCache:
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    target = center_indices(model.patch_size)
    rows = []
    for origin_x, origin_y in _phase_origins():
        observed = observation_indices(model.patch_size, origin_x, origin_y)
        area = model.patch_size * model.patch_size
        center_spatial = area // 2
        sampled = int(
            observed[np.nonzero(observed % area == center_spatial)[0][0]] // area
        )
        means_observed = model.means[:, observed]
        means_center = model.means[:, target]
        factors = []
        log_determinants = np.empty(model.component_count, dtype=np.float64)
        gains = np.empty(
            (model.component_count, 3, observed.size), dtype=np.float64
        )
        for component in range(model.component_count):
            covariance = model.covariances[component]
            system = covariance[np.ix_(observed, observed)].copy()
            system.flat[:: system.shape[0] + 1] += tau * tau
            factor = cho_factor(system, lower=True, check_finite=True)
            diagonal = np.diag(factor[0])
            if np.any(diagonal <= 0.0):
                raise ValueError("conditional covariance is not positive definite")
            cross = covariance[np.ix_(target, observed)]
            gains[component] = cho_solve(factor, cross.T, check_finite=True).T
            log_determinants[component] = 2.0 * float(np.sum(np.log(diagonal)))
            factors.append(np.ascontiguousarray(factor[0]))
        rows.append(PhaseConditional(
            observed_indices=np.ascontiguousarray(observed),
            sampled_center_channel=sampled,
            means_observed=np.ascontiguousarray(means_observed),
            means_center=np.ascontiguousarray(means_center),
            cholesky=tuple(factors),
            log_determinants=log_determinants,
            center_gains=gains,
        ))
    return ConditionalCache(model=model, tau=tau, phases=tuple(rows))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def conditional_predict(
    cache: ConditionalCache,
    samples: list[PatchSample],
    *,
    temperature: float = 1.0,
) -> ConditionalBatch:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    count = len(samples)
    components = cache.model.component_count
    component_rgb = np.empty((count, components, 3), dtype=np.float64)
    log_weights = np.empty((count, components), dtype=np.float64)
    center_truth = np.empty((count, 3), dtype=np.float64)
    for phase_index, phase in enumerate(cache.phases):
        selected = [index for index, sample in enumerate(samples) if sample.phase == phase_index]
        if not selected:
            continue
        vectors = []
        dc_values = []
        for index in selected:
            normalized, dc = normalize_vector(
                samples[index].vector, samples[index].indices, cache.model.dc_mode
            )
            vectors.append(normalized)
            dc_values.append(dc)
            center_truth[index] = samples[index].vector[center_indices(cache.model.patch_size)]
        matrix = np.stack(vectors)
        y = matrix[:, phase.observed_indices]
        dcs = np.asarray(dc_values, dtype=np.float64)
        for component in range(components):
            residual = y - phase.means_observed[component]
            factor = (phase.cholesky[component], True)
            solved = cho_solve(factor, residual.T, check_finite=False).T
            quadratic = np.sum(residual * solved, axis=1)
            log_weights[selected, component] = (
                math.log(float(cache.model.weights[component]))
                - 0.5 * (
                    quadratic
                    + phase.log_determinants[component]
                    + y.shape[1] * LOG_2PI
                )
            )
            component_rgb[selected, component] = (
                phase.means_center[component]
                + residual @ phase.center_gains[component].T
                + dcs[:, None]
            )
        component_rgb[selected, :, phase.sampled_center_channel] = center_truth[
            selected, phase.sampled_center_channel
        ][:, None]
    scaled = log_weights / temperature
    normalizer = _logsumexp(scaled, axis=1)
    log_responsibilities = scaled - normalizer[:, None]
    responsibilities = np.exp(log_responsibilities)
    map_components = np.argmax(log_weights, axis=1)
    map_rgb = component_rgb[np.arange(count), map_components]
    mmse_rgb = np.sum(component_rgb * responsibilities[:, :, None], axis=1)
    entropy = -np.sum(responsibilities * log_responsibilities, axis=1)
    if not all(np.isfinite(value).all() for value in (
        component_rgb, responsibilities, map_rgb, mmse_rgb, entropy
    )):
        raise RuntimeError("non-finite conditional prediction")
    return ConditionalBatch(
        map_rgb=map_rgb,
        mmse_rgb=mmse_rgb,
        component_rgb=component_rgb,
        responsibilities=responsibilities,
        log_responsibilities=log_responsibilities,
        map_components=map_components,
        entropy=entropy,
    )


def full_rgb_log_likelihood(model: JointColorGMM, samples: list[PatchSample]) -> np.ndarray:
    result = np.empty((len(samples), model.component_count), dtype=np.float64)
    matrix = sample_matrix(samples, model.dc_mode)
    dimension = matrix.shape[1]
    for component in range(model.component_count):
        factor = cho_factor(model.covariances[component], lower=True, check_finite=True)
        residual = matrix - model.means[component]
        solved = cho_solve(factor, residual.T, check_finite=False).T
        quadratic = np.sum(residual * solved, axis=1)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
        result[:, component] = (
            math.log(float(model.weights[component]))
            - 0.5 * (quadratic + log_determinant + dimension * LOG_2PI)
        )
    return result


def model_summary(model: JointColorGMM) -> dict[str, object]:
    condition_numbers = []
    effective_ranks = []
    for covariance in model.covariances:
        eigenvalues = np.linalg.eigvalsh(covariance)
        condition_numbers.append(float(eigenvalues[-1] / eigenvalues[0]))
        effective_ranks.append(int(np.sum(eigenvalues > eigenvalues[-1] * 1e-6)))
    dimension = model.means.shape[1]
    parameter_count = (
        model.component_count * (dimension + dimension * (dimension + 1) // 2)
        + model.component_count - 1
    )
    return {
        "component_count": model.component_count,
        "converged": model.converged,
        "covariance_condition_maximum": max(condition_numbers),
        "covariance_floor": model.covariance_floor,
        "dc_mode": model.dc_mode,
        "effective_component_counts": model.effective_counts.tolist(),
        "effective_rank_minimum": min(effective_ranks),
        "iterations": model.iterations,
        "lower_bound": model.lower_bound,
        "maximum_component_weight": float(np.max(model.weights)),
        "minimum_component_weight": float(np.min(model.weights)),
        "parameter_count": parameter_count,
        "patch_size": model.patch_size,
        "seed": model.seed,
        "training_sample_count": model.training_sample_count,
    }


__all__ = (
    "ConditionalBatch", "ConditionalCache", "DC_MODES", "JointColorGMM",
    "center_indices", "conditional_predict", "fit_joint_gmm",
    "full_rgb_log_likelihood", "model_summary", "normalize_vector",
    "observable_dc", "prepare_conditional_cache", "sample_matrix",
    "refine_joint_gmm",
)

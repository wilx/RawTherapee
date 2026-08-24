"""Task-focused phase-conditioned Gaussian-mixture regression.

The model deliberately represents only the variables needed by center-pixel
X-Trans demosaicing: the 49 physically observed values in a 7x7 patch and the
two missing center components.  Each of the 18 distinct X-Trans translations
has an independent density p_p(y, t).  Inference uses the exact Gaussian
conditional mean and observed-data posterior; there is no content classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.mixture import GaussianMixture

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_mlri_internal.dataset import origin_cells
from tools.xtrans_sparse_dictionary.model import observation_indices


LOG_2PI = math.log(2.0 * math.pi)
DC_MODES = ("absolute", "observed-rgb")


@dataclass(frozen=True)
class PhaseMixture:
    origin_x: int
    origin_y: int
    observed_indices: np.ndarray
    sampled_center_channel: int
    target_channels: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    converged: bool
    iterations: int
    lower_bound: float
    effective_counts: np.ndarray


@dataclass(frozen=True)
class PhaseConditionedGMR:
    patch_size: int
    component_count: int
    dc_mode: str
    covariance_floor: float
    training_sample_count: int
    seed: int
    phases: tuple[PhaseMixture, ...]
    shrinkage: str = "none"


@dataclass(frozen=True)
class CachedPhase:
    observed_indices: np.ndarray
    sampled_center_channel: int
    target_channels: np.ndarray
    means_observed: np.ndarray
    means_target: np.ndarray
    cholesky: tuple[np.ndarray, ...]
    log_determinants: np.ndarray
    target_gains: np.ndarray
    conditional_risks: np.ndarray


@dataclass(frozen=True)
class GMRCache:
    model: PhaseConditionedGMR
    tau: float
    phases: tuple[CachedPhase, ...]


@dataclass(frozen=True)
class GMRBatch:
    map_rgb: np.ndarray
    mmse_rgb: np.ndarray
    component_rgb: np.ndarray
    responsibilities: np.ndarray
    log_responsibilities: np.ndarray
    map_components: np.ndarray
    joint_map_components: np.ndarray
    entropy: np.ndarray
    conditional_risks: np.ndarray


def _check_patch_size(patch_size: int) -> int:
    if patch_size < 3 or patch_size % 2 != 1:
        raise ValueError("patch size must be odd and at least three")
    return patch_size


def phase_contract(patch_size: int, phase: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Return observed indices, measured center channel and missing order.

    Missing target channels are always ascending RGB channel indices.  Thus an
    R center predicts [G,B], G predicts [R,B], and B predicts [R,G].
    """

    size = _check_patch_size(patch_size)
    origins = origin_cells()
    if phase < 0 or phase >= len(origins):
        raise ValueError("invalid X-Trans phase")
    observed = observation_indices(size, *origins[phase])
    area = size * size
    center = area // 2
    matches = observed[np.flatnonzero(observed % area == center)]
    if matches.size != 1:
        raise RuntimeError("phase does not contain exactly one center sample")
    sampled = int(matches[0] // area)
    targets = np.asarray([channel for channel in range(3) if channel != sampled], dtype=np.int64)
    return np.ascontiguousarray(observed), sampled, targets


def _dc_values(vectors: np.ndarray, observed: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.zeros(vectors.shape[0], dtype=np.float64)
    if mode != "observed-rgb":
        raise ValueError(f"unknown DC mode: {mode}")
    area = vectors.shape[1] // 3
    colors = observed // area
    means = np.stack([
        np.mean(vectors[:, observed[colors == channel]], axis=1)
        for channel in range(3)
    ], axis=1)
    return np.mean(means, axis=1)


def phase_training_matrix(
    vectors: np.ndarray,
    patch_size: int,
    phase: int,
    dc_mode: str,
) -> np.ndarray:
    checked = np.asarray(vectors, dtype=np.float64)
    size = _check_patch_size(patch_size)
    if checked.ndim != 2 or checked.shape[1] != 3 * size * size or not np.isfinite(checked).all():
        raise ValueError("invalid RGB patch matrix")
    observed, _, targets = phase_contract(size, phase)
    area = size * size
    center = area // 2
    target_indices = targets * area + center
    dc = _dc_values(checked, observed, dc_mode)
    result = np.concatenate((checked[:, observed], checked[:, target_indices]), axis=1)
    result -= dc[:, None]
    return np.ascontiguousarray(result, dtype=np.float64)


def _fit_one_phase(
    matrix: np.ndarray,
    phase: int,
    patch_size: int,
    component_count: int,
    *,
    covariance_floor: float,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> PhaseMixture:
    estimator = GaussianMixture(
        n_components=component_count,
        covariance_type="full",
        tol=tolerance,
        reg_covar=covariance_floor,
        max_iter=maximum_iterations,
        n_init=1,
        init_params="k-means++",
        random_state=seed + phase * 104729,
    )
    estimator.fit(matrix)
    effective = np.sum(estimator.predict_proba(matrix), axis=0)
    observed, sampled, targets = phase_contract(patch_size, phase)
    return PhaseMixture(
        origin_x=origin_cells()[phase][0],
        origin_y=origin_cells()[phase][1],
        observed_indices=observed,
        sampled_center_channel=sampled,
        target_channels=targets,
        weights=np.ascontiguousarray(estimator.weights_, dtype=np.float64),
        means=np.ascontiguousarray(estimator.means_, dtype=np.float64),
        covariances=np.ascontiguousarray(estimator.covariances_, dtype=np.float64),
        converged=bool(estimator.converged_),
        iterations=int(estimator.n_iter_),
        lower_bound=float(estimator.lower_bound_),
        effective_counts=np.ascontiguousarray(effective, dtype=np.float64),
    )


def fit_phase_conditioned_gmr(
    vectors: np.ndarray,
    patch_size: int,
    component_count: int,
    *,
    dc_mode: str = "observed-rgb",
    covariance_floor: float = 1e-6,
    seed: int = 0x58474D52,
    maximum_iterations: int = 30,
    tolerance: float = 1e-3,
    phases: Iterable[int] | None = None,
) -> PhaseConditionedGMR:
    checked = np.asarray(vectors, dtype=np.float64)
    size = _check_patch_size(patch_size)
    if (
        checked.ndim != 2
        or checked.shape[1] != 3 * size * size
        or checked.shape[0] < component_count
        or not np.isfinite(checked).all()
        or component_count < 1
        or covariance_floor <= 0.0
        or dc_mode not in DC_MODES
        or maximum_iterations < 1
    ):
        raise ValueError("invalid GMR training configuration")
    selected = tuple(range(18)) if phases is None else tuple(phases)
    if selected != tuple(range(18)):
        raise ValueError("a production research model must contain all 18 phases")
    rows = []
    for phase in selected:
        matrix = phase_training_matrix(checked, size, phase, dc_mode)
        rows.append(_fit_one_phase(
            matrix, phase, size, component_count,
            covariance_floor=covariance_floor,
            seed=seed,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        ))
    return PhaseConditionedGMR(
        patch_size=size,
        component_count=component_count,
        dc_mode=dc_mode,
        covariance_floor=covariance_floor,
        training_sample_count=checked.shape[0],
        seed=seed,
        phases=tuple(rows),
    )


def refine_phase_conditioned_gmr(
    vectors: np.ndarray,
    model: PhaseConditionedGMR,
    *,
    additional_iterations: int,
    tolerance: float = 0.0,
) -> PhaseConditionedGMR:
    """Continue every phase from one common saved EM checkpoint.

    This is preferable to separately refitting nominal 30- and 50-iteration
    models: it follows one actual EM trajectory and avoids repeating the first
    twenty iterations for each diagnostic checkpoint.
    """

    checked = np.asarray(vectors, dtype=np.float64)
    if (
        checked.shape != (
            model.training_sample_count,
            3 * model.patch_size * model.patch_size,
        )
        or not np.isfinite(checked).all()
        or additional_iterations < 1
    ):
        raise ValueError("invalid GMR refinement input")
    rows = []
    for phase_index, phase in enumerate(model.phases):
        matrix = phase_training_matrix(
            checked, model.patch_size, phase_index, model.dc_mode
        )
        precision_rows = []
        identity = np.eye(phase.covariances.shape[1], dtype=np.float64)
        for covariance in phase.covariances:
            factor = cho_factor(covariance, lower=True, check_finite=True)
            precision = cho_solve(factor, identity, check_finite=True)
            # LAPACK solves may differ by a few ULPs across the diagonal.
            # sklearn validates precisions with a symmetric eigensolver, so
            # restore the mathematical symmetry explicitly before handoff.
            precision_rows.append(0.5 * (precision + precision.T))
        precisions = np.stack(precision_rows)
        estimator = GaussianMixture(
            n_components=model.component_count,
            covariance_type="full",
            tol=tolerance,
            reg_covar=model.covariance_floor,
            max_iter=additional_iterations,
            n_init=1,
            init_params="random",
            random_state=model.seed + phase_index * 104729,
            weights_init=phase.weights,
            means_init=phase.means,
            precisions_init=precisions,
        )
        estimator.fit(matrix)
        effective = np.sum(estimator.predict_proba(matrix), axis=0)
        rows.append(PhaseMixture(
            **{
                **phase.__dict__,
                "weights": np.ascontiguousarray(estimator.weights_, dtype=np.float64),
                "means": np.ascontiguousarray(estimator.means_, dtype=np.float64),
                "covariances": np.ascontiguousarray(estimator.covariances_, dtype=np.float64),
                "converged": bool(estimator.converged_),
                "iterations": phase.iterations + int(estimator.n_iter_),
                "lower_bound": float(estimator.lower_bound_),
                "effective_counts": np.ascontiguousarray(effective, dtype=np.float64),
            }
        ))
    return PhaseConditionedGMR(**{**model.__dict__, "phases": tuple(rows)})


def shrink_covariances(
    model: PhaseConditionedGMR,
    global_model: PhaseConditionedGMR,
    *,
    alpha: float | None = None,
    kappa: float | None = None,
) -> PhaseConditionedGMR:
    if global_model.component_count != 1 or global_model.patch_size != model.patch_size:
        raise ValueError("global backoff model must be matching K=1 GMR")
    if (alpha is None) == (kappa is None):
        raise ValueError("specify exactly one shrinkage policy")
    if alpha is not None and (alpha < 0.0 or alpha > 1.0 or not math.isfinite(alpha)):
        raise ValueError("invalid uniform shrinkage")
    if kappa is not None and (kappa <= 0.0 or not math.isfinite(kappa)):
        raise ValueError("invalid occupancy shrinkage")
    phases = []
    labels = []
    for phase, global_phase in zip(model.phases, global_model.phases):
        if alpha is not None:
            amounts = np.full(model.component_count, alpha, dtype=np.float64)
            label = f"uniform-{alpha:g}"
        else:
            amounts = kappa / (phase.effective_counts + kappa)
            label = f"occupancy-{kappa:g}"
        covariances = np.empty_like(phase.covariances)
        for component, amount in enumerate(amounts):
            covariances[component] = (
                (1.0 - amount) * phase.covariances[component]
                + amount * global_phase.covariances[0]
            )
        phases.append(PhaseMixture(
            **{**phase.__dict__, "covariances": np.ascontiguousarray(covariances)}
        ))
        labels.append(label)
    return PhaseConditionedGMR(
        **{**model.__dict__, "phases": tuple(phases), "shrinkage": labels[0]}
    )


def prepare_gmr_cache(model: PhaseConditionedGMR, tau: float) -> GMRCache:
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    rows = []
    observed_dimension = model.patch_size * model.patch_size
    for phase in model.phases:
        means_y = phase.means[:, :observed_dimension]
        means_t = phase.means[:, observed_dimension:]
        factors = []
        log_determinants = np.empty(model.component_count, dtype=np.float64)
        gains = np.empty((model.component_count, 2, observed_dimension), dtype=np.float64)
        risks = np.empty(model.component_count, dtype=np.float64)
        for component in range(model.component_count):
            covariance = phase.covariances[component]
            system = covariance[:observed_dimension, :observed_dimension].copy()
            system.flat[:: observed_dimension + 1] += tau * tau
            factor = cho_factor(system, lower=True, check_finite=True)
            diagonal = np.diag(factor[0])
            if np.any(diagonal <= 0.0):
                raise ValueError("conditional covariance is not positive definite")
            cross = covariance[observed_dimension:, :observed_dimension]
            gain = cho_solve(factor, cross.T, check_finite=True).T
            conditional = covariance[observed_dimension:, observed_dimension:] - gain @ cross.T
            gains[component] = gain
            risks[component] = max(float(np.trace(conditional)), 0.0)
            log_determinants[component] = 2.0 * float(np.sum(np.log(diagonal)))
            factors.append(np.ascontiguousarray(factor[0]))
        rows.append(CachedPhase(
            observed_indices=phase.observed_indices,
            sampled_center_channel=phase.sampled_center_channel,
            target_channels=phase.target_channels,
            means_observed=np.ascontiguousarray(means_y),
            means_target=np.ascontiguousarray(means_t),
            cholesky=tuple(factors),
            log_determinants=log_determinants,
            target_gains=gains,
            conditional_risks=risks,
        ))
    return GMRCache(model=model, tau=tau, phases=tuple(rows))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def conditional_predict(
    cache: GMRCache,
    samples: list[PatchSample],
    *,
    temperature: float = 1.0,
    risk_beta: float = 0.0,
    compute_joint_map: bool = True,
) -> GMRBatch:
    if temperature <= 0.0 or not math.isfinite(temperature) or risk_beta < 0.0:
        raise ValueError("invalid posterior control")
    count = len(samples)
    components = cache.model.component_count
    component_rgb = np.empty((count, components, 3), dtype=np.float64)
    log_weights = np.empty((count, components), dtype=np.float64)
    joint_log_weights = np.empty((count, components), dtype=np.float64) if compute_joint_map else None
    risk_rows = np.empty((count, components), dtype=np.float64)
    measured_channels = np.empty(count, dtype=np.int64)
    measured_values = np.empty(count, dtype=np.float64)
    area = cache.model.patch_size * cache.model.patch_size
    center = area // 2
    center_indices = np.asarray([channel * area + center for channel in range(3)])
    for phase_index, phase in enumerate(cache.phases):
        selected = np.asarray(
            [index for index, sample in enumerate(samples) if sample.phase == phase_index],
            dtype=np.int64,
        )
        if selected.size == 0:
            continue
        vectors = np.stack([samples[index].vector for index in selected]).astype(np.float64)
        expected = phase.observed_indices
        for index in selected:
            if not np.array_equal(samples[index].indices, expected):
                raise ValueError("sample observation indices do not match its phase")
        dc = _dc_values(vectors, expected, cache.model.dc_mode)
        y = vectors[:, expected] - dc[:, None]
        target_indices = phase.target_channels * area + center
        true_t = vectors[:, target_indices] - dc[:, None]
        measured = vectors[:, center_indices[phase.sampled_center_channel]]
        measured_channels[selected] = phase.sampled_center_channel
        measured_values[selected] = measured
        component_rgb[selected, :, :] = measured[:, None, None]
        risk_rows[selected] = phase.conditional_risks[None, :]
        for component in range(components):
            residual = y - phase.means_observed[component]
            factor = (phase.cholesky[component], True)
            solved = cho_solve(factor, residual.T, check_finite=False).T
            quadratic = np.sum(residual * solved, axis=1)
            log_weights[selected, component] = (
                math.log(float(cache.model.phases[phase_index].weights[component]))
                - 0.5 * (quadratic + phase.log_determinants[component] + y.shape[1] * LOG_2PI)
            )
            target_prediction = (
                phase.means_target[component]
                + residual @ phase.target_gains[component].T
                + dc[:, None]
            )
            component_rgb[selected[:, None], component, phase.target_channels[None, :]] = target_prediction
            if joint_log_weights is not None:
                joint_residual = np.concatenate((residual, true_t - phase.means_target[component]), axis=1)
                covariance = cache.model.phases[phase_index].covariances[component]
                joint_factor = cho_factor(covariance, lower=True, check_finite=False)
                joint_solved = cho_solve(joint_factor, joint_residual.T, check_finite=False).T
                joint_quadratic = np.sum(joint_residual * joint_solved, axis=1)
                joint_logdet = 2.0 * float(np.sum(np.log(np.diag(joint_factor[0]))))
                joint_log_weights[selected, component] = (
                    math.log(float(cache.model.phases[phase_index].weights[component]))
                    - 0.5 * (joint_quadratic + joint_logdet + (y.shape[1] + 2) * LOG_2PI)
                )
    scaled = log_weights / temperature - risk_beta * risk_rows
    normalizer = _logsumexp(scaled, axis=1)
    log_responsibilities = scaled - normalizer[:, None]
    responsibilities = np.exp(log_responsibilities)
    map_components = np.argmax(log_weights, axis=1)
    map_rgb = component_rgb[np.arange(count), map_components]
    mmse_rgb = np.sum(component_rgb * responsibilities[:, :, None], axis=1)
    # Weighted sums can perturb an otherwise identical measured value by one
    # ULP.  The physical center CFA sample is a hard constraint, so restore it
    # after posterior averaging rather than merely relying on sum(gamma)=1.
    mmse_rgb[np.arange(count), measured_channels] = measured_values
    entropy = -np.sum(responsibilities * log_responsibilities, axis=1)
    if not all(np.isfinite(value).all() for value in (
        component_rgb, responsibilities, map_rgb, mmse_rgb, entropy, risk_rows,
    )):
        raise RuntimeError("non-finite GMR prediction")
    return GMRBatch(
        map_rgb=map_rgb,
        mmse_rgb=mmse_rgb,
        component_rgb=component_rgb,
        responsibilities=responsibilities,
        log_responsibilities=log_responsibilities,
        map_components=map_components,
        joint_map_components=(
            np.argmax(joint_log_weights, axis=1)
            if joint_log_weights is not None
            else np.full(count, -1, dtype=np.int64)
        ),
        entropy=entropy,
        conditional_risks=risk_rows,
    )


def model_summary(model: PhaseConditionedGMR) -> dict[str, object]:
    conditions = []
    ranks = []
    occupancies = []
    likelihoods = []
    iterations = []
    for phase in model.phases:
        occupancies.extend(phase.effective_counts.tolist())
        likelihoods.append(phase.lower_bound)
        iterations.append(phase.iterations)
        for covariance in phase.covariances:
            eigenvalues = np.linalg.eigvalsh(covariance)
            conditions.append(float(eigenvalues[-1] / eigenvalues[0]))
            ranks.append(int(np.sum(eigenvalues > eigenvalues[-1] * 1e-6)))
    dimension = model.patch_size * model.patch_size + 2
    per_component = dimension + dimension * (dimension + 1) // 2
    parameters = 18 * (
        model.component_count * per_component + model.component_count - 1
    )
    return {
        "component_count": model.component_count,
        "converged_phase_count": int(sum(phase.converged for phase in model.phases)),
        "covariance_condition_maximum": max(conditions),
        "covariance_floor": model.covariance_floor,
        "dc_mode": model.dc_mode,
        "dimension": dimension,
        "effective_count_maximum": max(occupancies),
        "effective_count_median": float(np.median(occupancies)),
        "effective_count_minimum": min(occupancies),
        "effective_rank_minimum": min(ranks),
        "iteration_maximum": max(iterations),
        "iteration_minimum": min(iterations),
        "mean_training_lower_bound": float(np.mean(likelihoods)),
        "parameter_count": parameters,
        "patch_size": model.patch_size,
        "phase_count": len(model.phases),
        "seed": model.seed,
        "shrinkage": model.shrinkage,
        "training_sample_count_per_phase": model.training_sample_count,
    }


__all__ = (
    "GMRBatch", "GMRCache", "PhaseConditionedGMR", "PhaseMixture",
    "conditional_predict", "fit_phase_conditioned_gmr", "model_summary",
    "phase_contract", "phase_training_matrix", "prepare_gmr_cache",
    "refine_phase_conditioned_gmr", "shrink_covariances",
)

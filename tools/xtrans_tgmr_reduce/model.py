"""Reduction primitives for the validated phase-conditioned Student-t GMR.

The module deliberately separates three questions:

* component removal from a frozen full-covariance model;
* post-hoc factor-plus-diagonal covariance approximation;
* exact-posterior truncation and an observable coarse-likelihood shortlist.

None of these operations changes the CFA contract, Student-t degree of
freedom, DC transform, posterior temperature, or measured-sample constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_solve

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_gmr.model import (
    _dc_values,
    _logsumexp,
    GMRCache,
    PhaseConditionedGMR,
    PhaseMixture,
)
from tools.xtrans_tgmr.model import StudentTBatch, student_t_logpdf


@dataclass(frozen=True)
class FactorPhase:
    """One phase's post-hoc factor-plus-diagonal component parameters."""

    observed_indices: np.ndarray
    sampled_center_channel: int
    target_channels: np.ndarray
    weights: np.ndarray
    means_observed: np.ndarray
    means_target: np.ndarray
    factors_observed: np.ndarray
    factors_target: np.ndarray
    uniqueness_observed: np.ndarray
    uniqueness_target: np.ndarray


@dataclass(frozen=True)
class FactorModel:
    """Immutable post-hoc factor approximation of a reviewed t-GMR."""

    patch_size: int
    component_count: int
    rank: int
    dc_mode: str
    uniqueness_floor: float
    phases: tuple[FactorPhase, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CachedFactorPhase:
    observed_indices: np.ndarray
    sampled_center_channel: int
    target_channels: np.ndarray
    weights: np.ndarray
    means_observed: np.ndarray
    means_target: np.ndarray
    factors_observed: np.ndarray
    factors_target: np.ndarray
    inverse_diagonal: np.ndarray
    middle_cholesky: tuple[np.ndarray, ...]
    log_determinants: np.ndarray
    conditional_risks: np.ndarray


@dataclass(frozen=True)
class FactorCache:
    model: FactorModel
    tau: float
    dtype: np.dtype
    phases: tuple[CachedFactorPhase, ...]


@dataclass(frozen=True)
class ShortlistBatch:
    """A reduced posterior prediction and shortlist diagnostics."""

    mmse_rgb: np.ndarray
    responsibilities: np.ndarray
    selected_components: np.ndarray
    retained_exact_mass: np.ndarray
    exact_top_included: np.ndarray


def prune_components(
    model: PhaseConditionedGMR,
    retained: int,
    *,
    policy: str,
) -> PhaseConditionedGMR:
    """Remove frozen components phase-by-phase without retraining."""

    if retained < 1 or retained > model.component_count:
        raise ValueError("invalid retained component count")
    if policy not in ("weight", "training-responsibility"):
        raise ValueError("unknown pruning policy")
    phases = []
    for phase in model.phases:
        score = phase.weights if policy == "weight" else phase.effective_counts
        # Stable sorting makes tied training statistics deterministic.
        selected = np.argsort(-score, kind="stable")[:retained]
        weights = phase.weights[selected].copy()
        weights /= np.sum(weights)
        phases.append(PhaseMixture(
            **{
                **phase.__dict__,
                "weights": np.ascontiguousarray(weights),
                "means": np.ascontiguousarray(phase.means[selected]),
                "covariances": np.ascontiguousarray(phase.covariances[selected]),
                "effective_counts": np.ascontiguousarray(
                    phase.effective_counts[selected]
                ),
            }
        ))
    return PhaseConditionedGMR(
        **{
            **model.__dict__,
            "component_count": retained,
            "phases": tuple(phases),
            "shrinkage": f"posthoc-{policy}-k{retained}",
        }
    )


def _factorize_covariance(
    covariance: np.ndarray,
    rank: int,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    selected_values = np.maximum(eigenvalues[-rank:], 0.0)
    factor = eigenvectors[:, -rank:] * np.sqrt(selected_values)[None, :]
    low_rank = factor @ factor.T
    uniqueness = np.maximum(np.diag(covariance - low_rank), floor)
    approximate = low_rank + np.diag(uniqueness)
    return (
        np.ascontiguousarray(factor),
        np.ascontiguousarray(uniqueness),
        np.ascontiguousarray(approximate),
    )


def approximate_factor_model(
    model: PhaseConditionedGMR,
    rank: int,
    *,
    uniqueness_floor: float = 1e-8,
) -> FactorModel:
    """Approximate every full covariance by ``Lambda Lambda.T + diag(Psi)``."""

    dimension = model.patch_size * model.patch_size + 2
    observed_dimension = dimension - 2
    if rank < 1 or rank >= dimension or uniqueness_floor <= 0.0:
        raise ValueError("invalid factor approximation")
    phases = []
    full_errors = []
    diagonal_errors = []
    observed_errors = []
    for phase_index, source in enumerate(model.phases):
        factors = np.empty((model.component_count, dimension, rank), dtype=np.float64)
        uniqueness = np.empty((model.component_count, dimension), dtype=np.float64)
        for component, covariance in enumerate(source.covariances):
            factor, diagonal, approximate = _factorize_covariance(
                covariance, rank, uniqueness_floor
            )
            factors[component] = factor
            uniqueness[component] = diagonal
            full_errors.append(
                np.linalg.norm(approximate - covariance)
                / max(np.linalg.norm(covariance), np.finfo(np.float64).tiny)
            )
            diagonal_errors.append(
                np.linalg.norm(np.diag(approximate) - np.diag(covariance))
                / max(np.linalg.norm(np.diag(covariance)), np.finfo(np.float64).tiny)
            )
            observed_full = covariance[:observed_dimension, :observed_dimension]
            observed_approximate = approximate[
                :observed_dimension, :observed_dimension
            ]
            observed_errors.append(
                np.linalg.norm(observed_approximate - observed_full)
                / max(np.linalg.norm(observed_full), np.finfo(np.float64).tiny)
            )
        phases.append(FactorPhase(
            observed_indices=source.observed_indices,
            sampled_center_channel=source.sampled_center_channel,
            target_channels=source.target_channels,
            weights=source.weights,
            means_observed=np.ascontiguousarray(source.means[:, :observed_dimension]),
            means_target=np.ascontiguousarray(source.means[:, observed_dimension:]),
            factors_observed=np.ascontiguousarray(factors[:, :observed_dimension]),
            factors_target=np.ascontiguousarray(factors[:, observed_dimension:]),
            uniqueness_observed=np.ascontiguousarray(
                uniqueness[:, :observed_dimension]
            ),
            uniqueness_target=np.ascontiguousarray(
                uniqueness[:, observed_dimension:]
            ),
        ))
    return FactorModel(
        patch_size=model.patch_size,
        component_count=model.component_count,
        rank=rank,
        dc_mode=model.dc_mode,
        uniqueness_floor=uniqueness_floor,
        phases=tuple(phases),
        diagnostics={
            "relative_frobenius_mean": float(np.mean(full_errors)),
            "relative_frobenius_p95": float(np.quantile(full_errors, 0.95)),
            "relative_diagonal_mean": float(np.mean(diagonal_errors)),
            "relative_observed_frobenius_mean": float(np.mean(observed_errors)),
            "relative_observed_frobenius_p95": float(
                np.quantile(observed_errors, 0.95)
            ),
        },
    )


def prepare_factor_cache(
    model: FactorModel,
    tau: float,
    *,
    dtype: np.dtype | type = np.float64,
) -> FactorCache:
    """Precompute determinant, Woodbury, and conditional-risk constants."""

    selected_dtype = np.dtype(dtype)
    if selected_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("factor inference supports only float32 or float64")
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    phases = []
    rank = model.rank
    for source in model.phases:
        inverse_diagonal = np.empty_like(
            source.uniqueness_observed, dtype=selected_dtype
        )
        log_determinants = np.empty(model.component_count, dtype=selected_dtype)
        risks = np.empty(model.component_count, dtype=selected_dtype)
        cholesky = []
        factors_y = np.asarray(source.factors_observed, dtype=selected_dtype)
        factors_t = np.asarray(source.factors_target, dtype=selected_dtype)
        uniqueness_y = np.asarray(source.uniqueness_observed, dtype=selected_dtype)
        uniqueness_t = np.asarray(source.uniqueness_target, dtype=selected_dtype)
        for component in range(model.component_count):
            diagonal = uniqueness_y[component] + selected_dtype.type(tau * tau)
            inverse = selected_dtype.type(1.0) / diagonal
            middle = np.eye(rank, dtype=selected_dtype) + (
                factors_y[component].T * inverse[None, :]
            ) @ factors_y[component]
            factor = np.linalg.cholesky(middle)
            inverse_diagonal[component] = inverse
            cholesky.append(np.ascontiguousarray(factor))
            log_determinants[component] = (
                np.sum(np.log(diagonal))
                + selected_dtype.type(2.0) * np.sum(np.log(np.diag(factor)))
            )
            # The two-dimensional conditional covariance is prepared once.
            # Cross = Lt Ly.T and S^-1 Cross.T is obtained through Woodbury.
            cross = factors_t[component] @ factors_y[component].T
            weighted = cross.T * inverse[:, None]
            projected = factors_y[component].T @ weighted
            solved = cho_solve((factor, True), projected, check_finite=False)
            system_solution = weighted - (
                inverse[:, None] * factors_y[component]
            ) @ solved
            covariance_t = (
                factors_t[component] @ factors_t[component].T
                + np.diag(uniqueness_t[component])
            )
            conditional = covariance_t - cross @ system_solution
            risks[component] = max(float(np.trace(conditional)), 0.0)
        phases.append(CachedFactorPhase(
            observed_indices=source.observed_indices,
            sampled_center_channel=source.sampled_center_channel,
            target_channels=source.target_channels,
            weights=np.asarray(source.weights, dtype=selected_dtype),
            means_observed=np.asarray(source.means_observed, dtype=selected_dtype),
            means_target=np.asarray(source.means_target, dtype=selected_dtype),
            factors_observed=factors_y,
            factors_target=factors_t,
            inverse_diagonal=inverse_diagonal,
            middle_cholesky=tuple(cholesky),
            log_determinants=log_determinants,
            conditional_risks=risks,
        ))
    return FactorCache(
        model=model, tau=tau, dtype=selected_dtype, phases=tuple(phases)
    )


def factor_student_t_predict(
    cache: FactorCache,
    samples: list[PatchSample],
    *,
    degrees_of_freedom: float,
    temperature: float,
) -> StudentTBatch:
    """Evaluate factor likelihoods and predictors using Woodbury identities."""

    if degrees_of_freedom <= 2.0 or not math.isfinite(degrees_of_freedom):
        raise ValueError("degrees of freedom must exceed two")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    count = len(samples)
    components = cache.model.component_count
    area = cache.model.patch_size * cache.model.patch_size
    center = area // 2
    center_indices = np.asarray([channel * area + center for channel in range(3)])
    dtype = cache.dtype
    component_rgb = np.empty((count, components, 3), dtype=dtype)
    log_weights = np.empty((count, components), dtype=dtype)
    mahalanobis = np.empty((count, components), dtype=dtype)
    within_risk = np.empty((count, components), dtype=dtype)
    measured_channels = np.empty(count, dtype=np.int64)
    measured_values = np.empty(count, dtype=dtype)
    for phase_index, phase in enumerate(cache.phases):
        selected = np.asarray(
            [index for index, sample in enumerate(samples) if sample.phase == phase_index],
            dtype=np.int64,
        )
        if selected.size == 0:
            continue
        vectors = np.stack([samples[index].vector for index in selected]).astype(dtype)
        for index in selected:
            if not np.array_equal(samples[index].indices, phase.observed_indices):
                raise ValueError("sample observation indices do not match its phase")
        if cache.model.dc_mode == "absolute":
            dc = np.zeros(vectors.shape[0], dtype=dtype)
        elif cache.model.dc_mode == "observed-rgb":
            colors = phase.observed_indices // area
            color_means = np.stack([
                np.mean(
                    vectors[:, phase.observed_indices[colors == channel]], axis=1
                )
                for channel in range(3)
            ], axis=1)
            dc = np.mean(color_means, axis=1, dtype=dtype)
        else:
            raise ValueError("unknown factor-model DC mode")
        observed = vectors[:, phase.observed_indices] - dc[:, None]
        measured = vectors[:, center_indices[phase.sampled_center_channel]]
        measured_channels[selected] = phase.sampled_center_channel
        measured_values[selected] = measured
        component_rgb[selected, :, :] = measured[:, None, None]
        for component in range(components):
            residual = observed - phase.means_observed[component]
            weighted = residual * phase.inverse_diagonal[component]
            projected = weighted @ phase.factors_observed[component]
            solved = cho_solve(
                (phase.middle_cholesky[component], True),
                projected.T,
                check_finite=False,
            ).T
            quadratic = (
                np.sum(residual * weighted, axis=1)
                - np.sum(projected * solved, axis=1)
            )
            quadratic = np.maximum(quadratic, dtype.type(0.0))
            dimension = observed.shape[1]
            nu = degrees_of_freedom
            logpdf = (
                math.lgamma(0.5 * (nu + dimension))
                - math.lgamma(0.5 * nu)
                - 0.5 * (
                    dimension * math.log(nu * math.pi)
                    + phase.log_determinants[component]
                )
                - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
            )
            log_weights[selected, component] = (
                np.log(phase.weights[component]) + logpdf
            )
            mahalanobis[selected, component] = quadratic
            prediction = (
                phase.means_target[component]
                + solved @ phase.factors_target[component].T
                + dc[:, None]
            )
            component_rgb[
                selected[:, None], component, phase.target_channels[None, :]
            ] = prediction
            within_risk[selected, component] = (
                phase.conditional_risks[component]
                * (nu + quadratic)
                / (nu + dimension - 2.0)
            )
    scaled = log_weights / dtype.type(temperature)
    normalizer = _logsumexp(scaled, axis=1)
    log_responsibilities = scaled - normalizer[:, None]
    responsibilities = np.exp(log_responsibilities)
    map_components = np.argmax(log_weights, axis=1)
    map_rgb = component_rgb[np.arange(count), map_components]
    mmse_rgb = np.sum(component_rgb * responsibilities[:, :, None], axis=1)
    mmse_rgb[np.arange(count), measured_channels] = measured_values
    deviation = component_rgb - mmse_rgb[:, None, :]
    deviation[np.arange(count), :, measured_channels] = dtype.type(0.0)
    between = np.sum(deviation * deviation, axis=2)
    predictive_risk = np.sum(responsibilities * (within_risk + between), axis=1)
    entropy = -np.sum(responsibilities * log_responsibilities, axis=1)
    ordered = np.sort(responsibilities, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if components > 1 else np.zeros(count, dtype=dtype)
    result = StudentTBatch(
        map_rgb=np.asarray(map_rgb),
        mmse_rgb=np.asarray(mmse_rgb),
        component_rgb=np.asarray(component_rgb),
        responsibilities=np.asarray(responsibilities),
        log_responsibilities=np.asarray(log_responsibilities),
        map_components=map_components,
        entropy=np.asarray(entropy),
        maximum_responsibility=np.asarray(maximum),
        effective_component_count=np.asarray(np.exp(entropy)),
        top_two_ratio=np.asarray(
            maximum / np.maximum(second, np.finfo(dtype).tiny)
        ),
        predictive_risk=np.asarray(predictive_risk),
        observed_mahalanobis=np.asarray(mahalanobis),
    )
    if not all(np.isfinite(value).all() for value in (
        result.mmse_rgb,
        result.component_rgb,
        result.responsibilities,
        result.predictive_risk,
    )):
        raise RuntimeError("non-finite factor Student-t prediction")
    return result


def _restore_measured(
    output: np.ndarray,
    samples: list[PatchSample],
    patch_size: int,
) -> None:
    area = patch_size * patch_size
    center = area // 2
    for index, sample in enumerate(samples):
        matches = sample.indices[sample.indices % area == center]
        if matches.size != 1:
            raise ValueError("sample has invalid measured center")
        channel = int(matches[0] // area)
        output[index, channel] = sample.vector[channel * area + center]


def truncate_exact_posterior(
    batch: StudentTBatch,
    samples: list[PatchSample],
    patch_size: int,
    retained: int,
) -> ShortlistBatch:
    """Renormalize the exact full posterior over its largest entries."""

    count, components = batch.responsibilities.shape
    if retained < 1 or retained > components:
        raise ValueError("invalid posterior truncation")
    order = np.argsort(-batch.responsibilities, axis=1, kind="stable")
    selected = order[:, :retained]
    chosen = np.take_along_axis(batch.responsibilities, selected, axis=1)
    mass = np.sum(chosen, axis=1)
    normalized = chosen / mass[:, None]
    predictions = batch.component_rgb[np.arange(count)[:, None], selected]
    output = np.sum(predictions * normalized[:, :, None], axis=1)
    _restore_measured(output, samples, patch_size)
    sparse = np.zeros_like(batch.responsibilities)
    np.put_along_axis(sparse, selected, normalized, axis=1)
    return ShortlistBatch(
        mmse_rgb=output,
        responsibilities=sparse,
        selected_components=selected,
        retained_exact_mass=mass,
        exact_top_included=np.ones(count, dtype=bool),
    )


def _coarse_positions(phase, patch_size: int, support: int) -> np.ndarray:
    if support not in (3, 5) or support > patch_size:
        raise ValueError("coarse support must be 3 or 5")
    area = patch_size * patch_size
    radius = support // 2
    center = patch_size // 2
    spatial = phase.observed_indices % area
    y = spatial // patch_size
    x = spatial % patch_size
    return np.flatnonzero(
        (np.abs(y - center) <= radius) & (np.abs(x - center) <= radius)
    )


def shortlist_student_t_predict(
    full_cache: GMRCache,
    full_batch: StudentTBatch,
    samples: list[PatchSample],
    *,
    degrees_of_freedom: float,
    support: int,
    retained: int,
    component_order: np.ndarray | None = None,
) -> ShortlistBatch:
    """Select components by a physical S9/S25 Student-t marginal.

    The function intentionally consumes an exact full batch so this research
    implementation can isolate shortlist error. A real implementation would
    evaluate the full 49-D likelihood/predictor only for the returned IDs.
    """

    count = len(samples)
    components = full_cache.model.component_count
    if retained < 1 or retained > components:
        raise ValueError("invalid shortlist size")
    order = (
        coarse_component_order(
            full_cache,
            samples,
            degrees_of_freedom=degrees_of_freedom,
            support=support,
        )
        if component_order is None
        else np.asarray(component_order, dtype=np.int64)
    )
    if order.shape != (count, components):
        raise ValueError("invalid precomputed coarse component order")
    shortlist = order[:, :retained]
    exact_top = np.argmax(full_batch.responsibilities, axis=1)
    included = np.any(shortlist == exact_top[:, None], axis=1)
    exact = np.take_along_axis(full_batch.responsibilities, shortlist, axis=1)
    mass = np.sum(exact, axis=1)
    normalized = exact / mass[:, None]
    predictions = full_batch.component_rgb[np.arange(count)[:, None], shortlist]
    output = np.sum(predictions * normalized[:, :, None], axis=1)
    _restore_measured(output, samples, full_cache.model.patch_size)
    sparse = np.zeros_like(full_batch.responsibilities)
    np.put_along_axis(sparse, shortlist, normalized, axis=1)
    return ShortlistBatch(
        mmse_rgb=output,
        responsibilities=sparse,
        selected_components=shortlist,
        retained_exact_mass=mass,
        exact_top_included=included,
    )


def coarse_component_order(
    full_cache: GMRCache,
    samples: list[PatchSample],
    *,
    degrees_of_freedom: float,
    support: int,
) -> np.ndarray:
    """Return deterministic component order from an S9/S25 marginal."""

    count = len(samples)
    components = full_cache.model.component_count
    coarse_scores = np.empty((count, components), dtype=np.float64)
    for phase_index, phase in enumerate(full_cache.phases):
        selected_rows = np.asarray(
            [index for index, sample in enumerate(samples) if sample.phase == phase_index],
            dtype=np.int64,
        )
        if selected_rows.size == 0:
            continue
        local = _coarse_positions(phase, full_cache.model.patch_size, support)
        vectors = np.stack([samples[index].vector for index in selected_rows]).astype(
            np.float64
        )
        dc = _dc_values(vectors, phase.observed_indices, full_cache.model.dc_mode)
        observed = vectors[:, phase.observed_indices] - dc[:, None]
        for component in range(components):
            covariance = full_cache.model.phases[phase_index].covariances[
                component
            ][np.ix_(local, local)].copy()
            covariance.flat[:: local.size + 1] += full_cache.tau * full_cache.tau
            factor = np.linalg.cholesky(covariance)
            logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
            logpdf, _ = student_t_logpdf(
                observed[:, local] - phase.means_observed[component, local],
                factor,
                logdet,
                degrees_of_freedom,
            )
            coarse_scores[selected_rows, component] = (
                math.log(float(full_cache.model.phases[phase_index].weights[component]))
                + logpdf
            )
    return np.argsort(-coarse_scores, axis=1, kind="stable")


def factor_parameter_count(model: FactorModel) -> int:
    dimension = model.patch_size * model.patch_size + 2
    per_component = dimension + dimension * model.rank + dimension
    return 18 * (
        model.component_count * per_component + model.component_count - 1
    )


def factor_mac_estimate(model: FactorModel) -> int:
    """Conservative per-pixel dense multiply/add-scale count."""

    observed = model.patch_size * model.patch_size
    rank = model.rank
    # residual/diagonal quadratic, projection, two triangular solves, and two
    # target predictions. Constants and transcendental operations are separate.
    per_component = 2 * observed + observed * rank + 2 * rank * rank + 2 * rank
    return model.component_count * per_component


__all__ = (
    "FactorCache",
    "FactorModel",
    "ShortlistBatch",
    "approximate_factor_model",
    "coarse_component_order",
    "factor_mac_estimate",
    "factor_parameter_count",
    "factor_student_t_predict",
    "prepare_factor_cache",
    "prune_components",
    "shortlist_student_t_predict",
    "truncate_exact_posterior",
)

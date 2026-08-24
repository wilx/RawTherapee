"""Student-t responsibility inference over frozen phase GMR experts.

For a partitioned multivariate Student-t component, the conditional location
has the same affine form as a Gaussian conditional.  Its degrees of freedom
increase by the observed dimension and its scale is multiplied by
``(nu + delta_y) / (nu + d_y)``.  Consequently Stage A can isolate the effect
of heavy-tailed responsibilities without changing any expert prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_gmr.model import (
    GMRBatch,
    GMRCache,
    _dc_values,
    _logsumexp,
    conditional_predict,
    PhaseConditionedGMR,
    PhaseMixture,
    phase_training_matrix,
)


@dataclass(frozen=True)
class StudentTBatch:
    """Posterior predictions and uncertainty diagnostics."""

    map_rgb: np.ndarray
    mmse_rgb: np.ndarray
    component_rgb: np.ndarray
    responsibilities: np.ndarray
    log_responsibilities: np.ndarray
    map_components: np.ndarray
    entropy: np.ndarray
    maximum_responsibility: np.ndarray
    effective_component_count: np.ndarray
    top_two_ratio: np.ndarray
    predictive_risk: np.ndarray
    observed_mahalanobis: np.ndarray


@dataclass(frozen=True)
class StudentTTrainingResult:
    """One deterministic fixed-nu ECM checkpoint."""

    model: PhaseConditionedGMR
    degrees_of_freedom: float
    ecm_iterations: int
    mean_log_likelihood: float
    latent_weight_quantiles: tuple[float, ...]


def student_t_logpdf(
    residual: np.ndarray,
    cholesky: np.ndarray,
    log_determinant: float,
    degrees_of_freedom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return multivariate Student-t log density and Mahalanobis distance.

    ``cholesky`` is the lower Cholesky factor of the Student-t scale matrix.
    The Gaussian limit is intentionally evaluated by the ordinary Gaussian
    expression, giving an exact reference path instead of relying on a very
    large finite degree of freedom.
    """

    checked = np.asarray(residual, dtype=np.float64)
    if checked.ndim != 2 or cholesky.shape != (checked.shape[1], checked.shape[1]):
        raise ValueError("invalid Student-t residual or scale factor")
    if not np.isfinite(checked).all() or not math.isfinite(log_determinant):
        raise ValueError("non-finite Student-t input")
    solved = cho_solve((cholesky, True), checked.T, check_finite=False).T
    quadratic = np.sum(checked * solved, axis=1)
    dimension = checked.shape[1]
    if math.isinf(degrees_of_freedom):
        logpdf = -0.5 * (
            quadratic + log_determinant + dimension * math.log(2.0 * math.pi)
        )
    else:
        if degrees_of_freedom <= 0.0 or not math.isfinite(degrees_of_freedom):
            raise ValueError("degrees of freedom must be positive")
        nu = degrees_of_freedom
        logpdf = (
            gammaln(0.5 * (nu + dimension))
            - gammaln(0.5 * nu)
            - 0.5 * (dimension * math.log(nu * math.pi) + log_determinant)
            - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
        )
    return np.ascontiguousarray(logpdf), np.ascontiguousarray(quadratic)


def _from_gaussian(batch: GMRBatch) -> StudentTBatch:
    responsibilities = batch.responsibilities
    ordered = np.sort(responsibilities, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if ordered.shape[1] > 1 else np.zeros(len(batch.mmse_rgb))
    between = np.sum(
        (batch.component_rgb - batch.mmse_rgb[:, None, :]) ** 2, axis=2
    )
    return StudentTBatch(
        map_rgb=batch.map_rgb,
        mmse_rgb=batch.mmse_rgb,
        component_rgb=batch.component_rgb,
        responsibilities=responsibilities,
        log_responsibilities=batch.log_responsibilities,
        map_components=batch.map_components,
        entropy=batch.entropy,
        maximum_responsibility=maximum,
        effective_component_count=np.exp(batch.entropy),
        top_two_ratio=maximum / np.maximum(second, np.finfo(np.float64).tiny),
        predictive_risk=np.sum(
            responsibilities * (batch.conditional_risks + between), axis=1
        ),
        observed_mahalanobis=np.full_like(responsibilities, np.nan),
    )


def conditional_student_t_predict(
    cache: GMRCache,
    samples: list[PatchSample],
    *,
    degrees_of_freedom: float,
    temperature: float = 1.0,
) -> StudentTBatch:
    """Predict with Student-t responsibilities and frozen Gaussian experts.

    With ``degrees_of_freedom=math.inf`` this dispatches to the frozen
    Gaussian implementation and therefore must reproduce it bit-for-bit.
    For finite degrees of freedom, the returned predictive risk is the trace
    of the exact conditional Student-t covariance plus between-component
    target-mean variance.
    """

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if math.isinf(degrees_of_freedom):
        return _from_gaussian(
            conditional_predict(
                cache, samples, temperature=temperature, compute_joint_map=False
            )
        )
    if degrees_of_freedom <= 0.0 or not math.isfinite(degrees_of_freedom):
        raise ValueError("degrees of freedom must be positive")

    count = len(samples)
    components = cache.model.component_count
    observed_dimension = cache.model.patch_size * cache.model.patch_size
    area = observed_dimension
    center = area // 2
    center_indices = np.asarray([channel * area + center for channel in range(3)])
    component_rgb = np.empty((count, components, 3), dtype=np.float64)
    log_weights = np.empty((count, components), dtype=np.float64)
    mahalanobis = np.empty((count, components), dtype=np.float64)
    within_risk = np.empty((count, components), dtype=np.float64)
    measured_channels = np.empty(count, dtype=np.int64)
    measured_values = np.empty(count, dtype=np.float64)

    for phase_index, phase in enumerate(cache.phases):
        selected = np.asarray(
            [index for index, sample in enumerate(samples) if sample.phase == phase_index],
            dtype=np.int64,
        )
        if selected.size == 0:
            continue
        vectors = np.stack([samples[index].vector for index in selected]).astype(np.float64)
        for index in selected:
            if not np.array_equal(samples[index].indices, phase.observed_indices):
                raise ValueError("sample observation indices do not match its phase")
        dc = _dc_values(vectors, phase.observed_indices, cache.model.dc_mode)
        observed = vectors[:, phase.observed_indices] - dc[:, None]
        measured = vectors[:, center_indices[phase.sampled_center_channel]]
        measured_channels[selected] = phase.sampled_center_channel
        measured_values[selected] = measured
        component_rgb[selected, :, :] = measured[:, None, None]
        for component in range(components):
            residual = observed - phase.means_observed[component]
            logpdf, quadratic = student_t_logpdf(
                residual,
                phase.cholesky[component],
                float(phase.log_determinants[component]),
                degrees_of_freedom,
            )
            log_weights[selected, component] = (
                math.log(float(cache.model.phases[phase_index].weights[component]))
                + logpdf
            )
            mahalanobis[selected, component] = quadratic
            prediction = (
                phase.means_target[component]
                + residual @ phase.target_gains[component].T
                + dc[:, None]
            )
            component_rgb[
                selected[:, None], component, phase.target_channels[None, :]
            ] = prediction
            # t | y has df nu+d_y and scale multiplied by
            # (nu+delta)/(nu+d_y).  Covariance adds the familiar
            # df/(df-2), leaving (nu+delta)/(nu+d_y-2).
            within_risk[selected, component] = (
                phase.conditional_risks[component]
                * (degrees_of_freedom + quadratic)
                / (degrees_of_freedom + observed_dimension - 2.0)
            )

    scaled = log_weights / temperature
    log_normalizer = _logsumexp(scaled, axis=1)
    log_responsibilities = scaled - log_normalizer[:, None]
    responsibilities = np.exp(log_responsibilities)
    map_components = np.argmax(log_weights, axis=1)
    map_rgb = component_rgb[np.arange(count), map_components]
    mmse_rgb = np.sum(component_rgb * responsibilities[:, :, None], axis=1)
    mmse_rgb[np.arange(count), measured_channels] = measured_values
    target_deviation = component_rgb - mmse_rgb[:, None, :]
    # Exclude the identical measured channel: its variance is a hard zero.
    target_deviation[np.arange(count), :, measured_channels] = 0.0
    between = np.sum(target_deviation * target_deviation, axis=2)
    predictive_risk = np.sum(responsibilities * (within_risk + between), axis=1)
    entropy = -np.sum(responsibilities * log_responsibilities, axis=1)
    ordered = np.sort(responsibilities, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if components > 1 else np.zeros(count)
    result = StudentTBatch(
        map_rgb=map_rgb,
        mmse_rgb=mmse_rgb,
        component_rgb=component_rgb,
        responsibilities=responsibilities,
        log_responsibilities=log_responsibilities,
        map_components=map_components,
        entropy=entropy,
        maximum_responsibility=maximum,
        effective_component_count=np.exp(entropy),
        top_two_ratio=maximum / np.maximum(second, np.finfo(np.float64).tiny),
        predictive_risk=predictive_risk,
        observed_mahalanobis=mahalanobis,
    )
    if not all(np.isfinite(value).all() for value in (
        result.map_rgb,
        result.mmse_rgb,
        result.component_rgb,
        result.responsibilities,
        result.entropy,
        result.predictive_risk,
        result.observed_mahalanobis,
    )):
        raise RuntimeError("non-finite Student-t prediction")
    return result


def _student_t_e_step(
    matrix: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    degrees_of_freedom: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate responsibilities and latent precision expectations."""

    count, dimension = matrix.shape
    components = weights.size
    log_weights = np.empty((count, components), dtype=np.float64)
    quadratic = np.empty_like(log_weights)
    for component in range(components):
        factor = np.linalg.cholesky(scales[component])
        logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
        logpdf, distance = student_t_logpdf(
            matrix - means[component], factor, logdet, degrees_of_freedom
        )
        log_weights[:, component] = math.log(float(weights[component])) + logpdf
        quadratic[:, component] = distance
    normalizer = _logsumexp(log_weights, axis=1)
    responsibilities = np.exp(log_weights - normalizer[:, None])
    latent = (degrees_of_freedom + dimension) / (degrees_of_freedom + quadratic)
    return responsibilities, latent, float(np.mean(normalizer))


def refine_fixed_student_t_mixture(
    vectors: np.ndarray,
    initial: PhaseConditionedGMR,
    *,
    degrees_of_freedom: float,
    iterations: int,
) -> StudentTTrainingResult:
    """Run fixed-nu Student-t mixture ECM from a reviewed GMR checkpoint.

    The M-step follows the scale-mixture representation.  Means use
    responsibility times latent-precision weights; component scale matrices
    divide by the ordinary effective component count, as required by the
    Student-t mixture likelihood.
    """

    checked = np.asarray(vectors, dtype=np.float64)
    if (
        checked.shape != (
            initial.training_sample_count,
            3 * initial.patch_size * initial.patch_size,
        )
        or not np.isfinite(checked).all()
        or degrees_of_freedom <= 2.0
        or not math.isfinite(degrees_of_freedom)
        or iterations < 1
    ):
        raise ValueError("invalid Student-t ECM configuration")
    phases = []
    phase_likelihoods = []
    latent_samples = []
    for phase_index, source in enumerate(initial.phases):
        matrix = phase_training_matrix(
            checked, initial.patch_size, phase_index, initial.dc_mode
        )
        weights = source.weights.copy()
        means = source.means.copy()
        scales = source.covariances.copy()
        responsibilities = latent = None
        likelihood = -math.inf
        for _ in range(iterations):
            responsibilities, latent, likelihood = _student_t_e_step(
                matrix, weights, means, scales, degrees_of_freedom
            )
            effective = np.sum(responsibilities, axis=0)
            if np.any(effective <= 1e-8):
                raise RuntimeError("Student-t component became empty")
            robust = responsibilities * latent
            robust_sum = np.sum(robust, axis=0)
            means = (robust.T @ matrix) / robust_sum[:, None]
            updated = np.empty_like(scales)
            for component in range(initial.component_count):
                centered = matrix - means[component]
                weighted = centered * robust[:, component, None]
                updated[component] = centered.T @ weighted / effective[component]
                updated[component].flat[:: updated.shape[1] + 1] += initial.covariance_floor
                updated[component] = 0.5 * (
                    updated[component] + updated[component].T
                )
                np.linalg.cholesky(updated[component])
            scales = updated
            weights = effective / matrix.shape[0]
        assert responsibilities is not None and latent is not None
        # Recompute final likelihood/effective counts for the published state.
        responsibilities, latent, likelihood = _student_t_e_step(
            matrix, weights, means, scales, degrees_of_freedom
        )
        effective = np.sum(responsibilities, axis=0)
        latent_samples.append(latent.reshape(-1)[:: max(1, latent.size // 10000)])
        phase_likelihoods.append(likelihood)
        phases.append(PhaseMixture(
            **{
                **source.__dict__,
                "weights": np.ascontiguousarray(weights),
                "means": np.ascontiguousarray(means),
                "covariances": np.ascontiguousarray(scales),
                "converged": False,
                "iterations": source.iterations + iterations,
                "lower_bound": likelihood,
                "effective_counts": np.ascontiguousarray(effective),
            }
        ))
    sampled_latent = np.concatenate(latent_samples)
    model = PhaseConditionedGMR(
        **{**initial.__dict__, "phases": tuple(phases), "shrinkage": f"student-t-nu{degrees_of_freedom:g}"}
    )
    return StudentTTrainingResult(
        model=model,
        degrees_of_freedom=degrees_of_freedom,
        ecm_iterations=iterations,
        mean_log_likelihood=float(np.mean(phase_likelihoods)),
        latent_weight_quantiles=tuple(
            float(value) for value in np.quantile(
                sampled_latent, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
            )
        ),
    )


def estimate_shared_degrees_of_freedom(
    vectors: np.ndarray,
    model: PhaseConditionedGMR,
    *,
    current_degrees_of_freedom: float,
) -> float:
    """Perform the shared-nu ECM scalar update over all phases/components.

    The E-step expectation is pooled over all 18 phase populations.  The
    returned value is bounded away from the Gaussian singular limit and from
    the undefined-covariance region below two degrees of freedom.
    """

    checked = np.asarray(vectors, dtype=np.float64)
    if (
        checked.shape != (
            model.training_sample_count,
            3 * model.patch_size * model.patch_size,
        )
        or not np.isfinite(checked).all()
        or current_degrees_of_freedom <= 2.0
        or not math.isfinite(current_degrees_of_freedom)
    ):
        raise ValueError("invalid shared-nu estimation input")
    total = 0.0
    population = 0
    dimension = model.patch_size * model.patch_size + 2
    nu = current_degrees_of_freedom
    for phase_index, phase in enumerate(model.phases):
        matrix = phase_training_matrix(
            checked, model.patch_size, phase_index, model.dc_mode
        )
        responsibilities, latent, _ = _student_t_e_step(
            matrix, phase.weights, phase.means, phase.covariances, nu
        )
        # E[log U | z,k] for the Gamma latent scale representation.
        # Recover delta from u=(nu+d)/(nu+delta).
        delta = (nu + dimension) / latent - nu
        expected_log = digamma(0.5 * (nu + dimension)) - np.log(
            0.5 * (nu + delta)
        )
        total += float(np.sum(responsibilities * (expected_log - latent)))
        population += matrix.shape[0]
    average = total / population

    def objective(candidate: float) -> float:
        return (
            math.log(0.5 * candidate)
            - float(digamma(0.5 * candidate))
            + 1.0
            + average
        )

    lower = 2.000001
    upper = 200.0
    left = objective(lower)
    right = objective(upper)
    if left * right > 0.0:
        return lower if abs(left) < abs(right) else upper
    return float(brentq(objective, lower, upper, xtol=1e-10, rtol=1e-12))


def refine_learned_student_t_mixture(
    vectors: np.ndarray,
    initial: PhaseConditionedGMR,
    *,
    initial_degrees_of_freedom: float,
    iterations: int,
) -> StudentTTrainingResult:
    """Alternate shared-nu and model conditional maximizations.

    This bounded generalized-ECM diagnostic recomputes the E-step after the
    scalar nu update rather than implementing 1,152 independent tail
    parameters.  Fixed-nu ECM remains the primary faithful training path.
    """

    checked = np.asarray(vectors, dtype=np.float64)
    if (
        checked.shape != (
            initial.training_sample_count,
            3 * initial.patch_size * initial.patch_size,
        )
        or not np.isfinite(checked).all()
        or initial_degrees_of_freedom <= 2.0
        or not math.isfinite(initial_degrees_of_freedom)
        or iterations < 1
    ):
        raise ValueError("invalid learned-nu Student-t ECM configuration")
    model = initial
    nu = initial_degrees_of_freedom
    for _ in range(iterations):
        nu = estimate_shared_degrees_of_freedom(
            checked, model, current_degrees_of_freedom=nu
        )
        trained = refine_fixed_student_t_mixture(
            checked, model, degrees_of_freedom=nu, iterations=1
        )
        model = trained.model
    return StudentTTrainingResult(
        model=PhaseConditionedGMR(
            **{**model.__dict__, "shrinkage": "student-t-learned-shared-nu"}
        ),
        degrees_of_freedom=nu,
        ecm_iterations=iterations,
        mean_log_likelihood=trained.mean_log_likelihood,
        latent_weight_quantiles=trained.latent_weight_quantiles,
    )


__all__ = (
    "StudentTBatch", "StudentTTrainingResult",
    "conditional_student_t_predict",
    "estimate_shared_degrees_of_freedom", "refine_fixed_student_t_mixture",
    "refine_learned_student_t_mixture",
    "student_t_logpdf",
)

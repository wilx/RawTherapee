import math

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import multivariate_t

from tools.xtrans_gmr.model import (
    conditional_predict,
    fit_phase_conditioned_gmr,
    prepare_gmr_cache,
)
from tools.xtrans_gmr.tests.test_model import _manual_model, _samples
from tools.xtrans_tgmr.model import (
    conditional_student_t_predict,
    estimate_shared_degrees_of_freedom,
    refine_fixed_student_t_mixture,
    refine_learned_student_t_mixture,
    student_t_logpdf,
)


def test_logpdf_matches_scipy_multivariate_t() -> None:
    scale = np.asarray([[1.2, 0.3], [0.3, 0.8]])
    residual = np.asarray([[0.0, 0.0], [0.4, -0.7], [2.0, 1.0]])
    factor = cholesky(scale, lower=True)
    logdet = 2.0 * np.sum(np.log(np.diag(factor)))
    actual, quadratic = student_t_logpdf(residual, factor, logdet, 5.0)
    expected = multivariate_t.logpdf(residual, loc=np.zeros(2), shape=scale, df=5.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(quadratic, np.einsum("ni,ij,nj->n", residual, np.linalg.inv(scale), residual))


def test_gaussian_limit_is_bit_identical() -> None:
    vectors = np.random.default_rng(11).normal(size=(96, 147))
    model = fit_phase_conditioned_gmr(
        vectors, 7, 2, seed=13, maximum_iterations=2, tolerance=0.0
    )
    cache = prepare_gmr_cache(model, 3e-4)
    samples = _samples(7)
    ordinary = conditional_predict(cache, samples, temperature=4.0, compute_joint_map=False)
    student = conditional_student_t_predict(
        cache, samples, degrees_of_freedom=math.inf, temperature=4.0
    )
    assert np.array_equal(student.mmse_rgb, ordinary.mmse_rgb)
    assert np.array_equal(student.map_rgb, ordinary.map_rgb)
    assert np.array_equal(student.responsibilities, ordinary.responsibilities)


def test_finite_student_t_restores_observed_sample_and_is_finite() -> None:
    vectors = np.random.default_rng(17).normal(size=(96, 147))
    model = fit_phase_conditioned_gmr(
        vectors, 7, 2, seed=19, maximum_iterations=2, tolerance=0.0
    )
    samples = _samples(7)
    result = conditional_student_t_predict(
        prepare_gmr_cache(model, 3e-4),
        samples,
        degrees_of_freedom=5.0,
        temperature=4.0,
    )
    area = 49
    center = 24
    for index, sample in enumerate(samples):
        sampled = int(sample.indices[np.flatnonzero(sample.indices % area == center)][0] // area)
        assert result.mmse_rgb[index, sampled] == sample.vector[sampled * area + center]
    np.testing.assert_allclose(np.sum(result.responsibilities, axis=1), 1.0, atol=2e-15)
    assert np.isfinite(result.predictive_risk).all()
    assert np.all(result.predictive_risk >= 0.0)


def test_conditional_student_t_predictive_covariance_is_exact() -> None:
    model = _manual_model(3, 1)
    tau = 1e-3
    nu = 5.0
    samples = _samples(3)
    cache = prepare_gmr_cache(model, tau)
    result = conditional_student_t_predict(
        cache, samples, degrees_of_freedom=nu
    )
    sample = samples[0]
    phase = cache.phases[sample.phase]
    observed = sample.vector[phase.observed_indices]
    residual = observed - phase.means_observed[0]
    delta = float(residual @ np.linalg.solve(
        model.phases[sample.phase].covariances[:1, :9, :9][0]
        + tau * tau * np.eye(9), residual
    ))
    expected = phase.conditional_risks[0] * (nu + delta) / (nu + 9 - 2)
    np.testing.assert_allclose(result.predictive_risk[0], expected, rtol=2e-12)


def test_invalid_student_t_controls_are_rejected() -> None:
    factor = np.eye(2)
    residual = np.zeros((1, 2))
    for value in (0.0, -1.0, math.nan):
        try:
            student_t_logpdf(residual, factor, 0.0, value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid degrees of freedom accepted")


def test_fixed_student_t_ecm_is_finite_and_deterministic() -> None:
    vectors = np.random.default_rng(23).normal(size=(48, 27))
    initial = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=29,
        maximum_iterations=2, tolerance=0.0,
    )
    first = refine_fixed_student_t_mixture(
        vectors, initial, degrees_of_freedom=5.0, iterations=1
    )
    second = refine_fixed_student_t_mixture(
        vectors, initial, degrees_of_freedom=5.0, iterations=1
    )
    assert math.isfinite(first.mean_log_likelihood)
    assert len(first.latent_weight_quantiles) == 7
    for left, right in zip(first.model.phases, second.model.phases):
        np.testing.assert_array_equal(left.weights, right.weights)
        np.testing.assert_array_equal(left.means, right.means)
        np.testing.assert_array_equal(left.covariances, right.covariances)
        assert np.all(left.effective_counts > 0.0)


def test_shared_degrees_of_freedom_update_is_bounded_and_deterministic() -> None:
    vectors = np.random.default_rng(31).standard_t(5.0, size=(48, 27))
    initial = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=37,
        maximum_iterations=2, tolerance=0.0,
    )
    first = estimate_shared_degrees_of_freedom(
        vectors, initial, current_degrees_of_freedom=10.0
    )
    second = estimate_shared_degrees_of_freedom(
        vectors, initial, current_degrees_of_freedom=10.0
    )
    assert 2.0 < first <= 200.0
    assert first == second


def test_learned_shared_nu_ecm_updates_model() -> None:
    vectors = np.random.default_rng(41).standard_t(6.0, size=(48, 27))
    initial = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=43,
        maximum_iterations=2, tolerance=0.0,
    )
    result = refine_learned_student_t_mixture(
        vectors, initial, initial_degrees_of_freedom=10.0, iterations=1
    )
    assert 2.0 < result.degrees_of_freedom <= 200.0
    assert result.model.shrinkage == "student-t-learned-shared-nu"
    assert any(
        not np.array_equal(before.covariances, after.covariances)
        for before, after in zip(initial.phases, result.model.phases)
    )

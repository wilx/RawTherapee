import numpy as np
import pytest

from tools.xtrans_gmr.model import (
    PhaseConditionedGMR,
    PhaseMixture,
    prepare_gmr_cache,
)
from tools.xtrans_gmr.tests.test_model import _manual_model, _samples
from tools.xtrans_tgmr.model import conditional_student_t_predict
from tools.xtrans_tgmr_reduce.model import (
    approximate_factor_model,
    factor_mac_estimate,
    factor_parameter_count,
    factor_student_t_predict,
    prepare_factor_cache,
    prune_components,
    shortlist_student_t_predict,
    truncate_exact_posterior,
)
from tools.xtrans_tgmr_reduce.native_model import (
    native_float32_predict,
    prepare_native_model,
    serialize_native_model,
)


def _dense_factor_model(source, factor_model) -> PhaseConditionedGMR:
    phases = []
    for original, factor in zip(source.phases, factor_model.phases):
        factors = np.concatenate(
            (factor.factors_observed, factor.factors_target), axis=1
        )
        uniqueness = np.concatenate(
            (factor.uniqueness_observed, factor.uniqueness_target), axis=1
        )
        covariance = np.stack([
            row @ row.T + np.diag(diagonal)
            for row, diagonal in zip(factors, uniqueness)
        ])
        phases.append(PhaseMixture(
            **{**original.__dict__, "covariances": covariance}
        ))
    return PhaseConditionedGMR(**{**source.__dict__, "phases": tuple(phases)})


def test_factor_woodbury_matches_dense_approximate_covariance() -> None:
    source = _manual_model(3, 2)
    factor = approximate_factor_model(source, 3)
    dense = _dense_factor_model(source, factor)
    samples = _samples(3)
    expected = conditional_student_t_predict(
        prepare_gmr_cache(dense, 3e-4),
        samples,
        degrees_of_freedom=3.0,
        temperature=4.0,
    )
    actual = factor_student_t_predict(
        prepare_factor_cache(factor, 3e-4),
        samples,
        degrees_of_freedom=3.0,
        temperature=4.0,
    )
    np.testing.assert_allclose(actual.mmse_rgb, expected.mmse_rgb, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        actual.responsibilities, expected.responsibilities, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        actual.predictive_risk, expected.predictive_risk, rtol=2e-10, atol=2e-10
    )


def test_factor_float32_is_finite_and_preserves_native_sample() -> None:
    source = _manual_model(3, 2)
    factor = approximate_factor_model(source, 2)
    samples = _samples(3)
    reference = factor_student_t_predict(
        prepare_factor_cache(factor, 3e-4), samples,
        degrees_of_freedom=3.0, temperature=4.0,
    )
    reduced = factor_student_t_predict(
        prepare_factor_cache(factor, 3e-4, dtype=np.float32), samples,
        degrees_of_freedom=3.0, temperature=4.0,
    )
    np.testing.assert_allclose(reduced.mmse_rgb, reference.mmse_rgb, rtol=2e-5, atol=2e-6)
    assert np.isfinite(reduced.mmse_rgb).all()
    area = 9
    center = 4
    for index, sample in enumerate(samples):
        measured = sample.indices[sample.indices % area == center]
        channel = int(measured[0] // area)
        assert reduced.mmse_rgb[index, channel] == np.float32(
            sample.vector[channel * area + center]
        )


def test_posthoc_pruning_is_phase_local_and_normalized() -> None:
    source = _manual_model(3, 3)
    phases = []
    for index, phase in enumerate(source.phases):
        weights = np.asarray((0.1, 0.7, 0.2))
        counts = np.asarray((30.0, 10.0, 20.0))
        phases.append(PhaseMixture(
            **{**phase.__dict__, "weights": weights, "effective_counts": counts}
        ))
    source = PhaseConditionedGMR(**{**source.__dict__, "phases": tuple(phases)})
    by_weight = prune_components(source, 2, policy="weight")
    by_count = prune_components(source, 2, policy="training-responsibility")
    assert by_weight.component_count == 2
    assert by_count.component_count == 2
    np.testing.assert_allclose(by_weight.phases[0].weights, (0.7 / 0.9, 0.2 / 0.9))
    np.testing.assert_allclose(by_count.phases[0].effective_counts, (30.0, 20.0))
    for model in (by_weight, by_count):
        for phase in model.phases:
            assert np.sum(phase.weights) == 1.0


def test_exact_posterior_truncation_retains_reported_mass() -> None:
    source = _manual_model(3, 3)
    samples = _samples(3)
    full = conditional_student_t_predict(
        prepare_gmr_cache(source, 3e-4), samples,
        degrees_of_freedom=3.0, temperature=4.0,
    )
    top_one = truncate_exact_posterior(full, samples, 3, 1)
    expected_mass = np.max(full.responsibilities, axis=1)
    np.testing.assert_allclose(top_one.retained_exact_mass, expected_mass)
    assert np.all(np.count_nonzero(top_one.responsibilities, axis=1) == 1)
    all_components = truncate_exact_posterior(full, samples, 3, 3)
    np.testing.assert_allclose(all_components.mmse_rgb, full.mmse_rgb, atol=2e-15)
    np.testing.assert_allclose(all_components.retained_exact_mass, 1.0, atol=2e-15)


def test_coarse_shortlist_full_size_reproduces_exact_posterior() -> None:
    source = _manual_model(5, 2)
    samples = _samples(5)
    cache = prepare_gmr_cache(source, 3e-4)
    full = conditional_student_t_predict(
        cache, samples, degrees_of_freedom=3.0, temperature=4.0
    )
    result = shortlist_student_t_predict(
        cache, full, samples,
        degrees_of_freedom=3.0, support=5, retained=2,
    )
    np.testing.assert_allclose(result.mmse_rgb, full.mmse_rgb, atol=2e-15)
    np.testing.assert_allclose(result.retained_exact_mass, 1.0, atol=2e-15)
    assert np.all(result.exact_top_included)


def test_factor_cost_and_parameter_counts_decrease_with_rank() -> None:
    source = _manual_model(7, 4)
    rank4 = approximate_factor_model(source, 4)
    rank8 = approximate_factor_model(source, 8)
    assert factor_mac_estimate(rank4) < factor_mac_estimate(rank8)
    assert factor_parameter_count(rank4) < factor_parameter_count(rank8)


def test_native_float32_model_is_deterministic_and_close_to_float64() -> None:
    source = _manual_model(7, 32)
    source = PhaseConditionedGMR(
        **{**source.__dict__, "dc_mode": "observed-rgb"}
    )
    samples = _samples(7)
    cache = prepare_gmr_cache(source, 3e-4)
    full = conditional_student_t_predict(
        cache, samples, degrees_of_freedom=3.0, temperature=4.0
    )
    expected = shortlist_student_t_predict(
        cache, full, samples,
        degrees_of_freedom=3.0, support=3, retained=8,
    )
    native = prepare_native_model(source)
    actual, selected = native_float32_predict(native, samples)
    np.testing.assert_allclose(actual, expected.mmse_rgb, rtol=5e-5, atol=5e-6)
    assert selected.shape == (len(samples), 8)
    assert serialize_native_model(native) == serialize_native_model(
        prepare_native_model(source)
    )


def test_fixed_student_t_coarse_ranking_specialization_is_exact() -> None:
    rng = np.random.default_rng(0x53395138)
    component_constant = rng.normal(size=32).astype(np.float32)
    quadratic = np.exp(rng.uniform(-8.0, 8.0, size=32)).astype(np.float32)
    logarithmic = component_constant - np.float32(6.0) * np.log1p(
        quadratic / np.float32(3.0)
    )
    scale = np.exp(
        (component_constant - np.max(component_constant)) / np.float32(6.0)
    )
    specialized = scale / (np.float32(1.0) + quadratic / np.float32(3.0))
    expected = np.lexsort((np.arange(32), -logarithmic))[:8]
    actual = np.lexsort((np.arange(32), -specialized))[:8]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", [1, 7, 0x54474D52])
def test_fixed_student_t_full_weight_specialization_matches_softmax(seed: int) -> None:
    rng = np.random.default_rng(seed)
    component_constant = rng.normal(size=8).astype(np.float32)
    quadratic = np.exp(rng.uniform(-7.0, 7.0, size=8)).astype(np.float32)
    logarithmic = component_constant - np.float32(6.5) * np.log1p(
        quadratic / np.float32(3.0)
    )
    expected = np.exp(logarithmic - np.max(logarithmic))
    expected /= np.sum(expected)
    scale = np.exp(component_constant - np.max(component_constant))
    s = np.float32(1.0) + quadratic / np.float32(3.0)
    specialized = scale / (s**6 * np.sqrt(s))
    specialized /= np.sum(specialized)
    np.testing.assert_allclose(specialized, expected, rtol=2e-6, atol=2e-7)

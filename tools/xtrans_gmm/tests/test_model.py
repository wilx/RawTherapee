from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from tools.xtrans_gmm.dataset import grid_samples
from tools.xtrans_gmm.model import (
    JointColorGMM,
    center_indices,
    conditional_predict,
    fit_joint_gmm,
    normalize_vector,
    prepare_conditional_cache,
)
from tools.xtrans_sparse_dictionary.model import observation_indices, phase_index


def _scene(height: int = 24, width: int = 24) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack((
        0.1 + 0.7 * x / max(width - 1, 1),
        0.2 + 0.6 * y / max(height - 1, 1),
        0.15 + 0.5 * (x + y) / max(width + height - 2, 1),
    ))


def _single_component_model(patch_size: int = 3) -> JointColorGMM:
    dimension = 3 * patch_size * patch_size
    rng = np.random.default_rng(37)
    basis = rng.normal(size=(dimension, dimension))
    covariance = basis @ basis.T / dimension + 0.05 * np.eye(dimension)
    return JointColorGMM(
        patch_size=patch_size,
        component_count=1,
        dc_mode="absolute",
        covariance_floor=1e-6,
        weights=np.ones(1),
        means=np.linspace(0.1, 0.3, dimension)[None],
        covariances=covariance[None],
        converged=True,
        iterations=1,
        lower_bound=0.0,
        effective_counts=np.ones(1),
        training_sample_count=10,
        seed=1,
    )


def test_k1_matches_direct_conditional_gaussian() -> None:
    model = _single_component_model()
    samples = grid_samples(_scene(), "scene", "test", 3, 3)
    cache = prepare_conditional_cache(model, 1e-3)
    result = conditional_predict(cache, samples)
    targets = center_indices(3)
    for index, sample in enumerate(samples):
        phase = cache.phases[sample.phase]
        observed = phase.observed_indices
        system = model.covariances[0][np.ix_(observed, observed)].copy()
        system.flat[:: system.shape[0] + 1] += 1e-6
        cross = model.covariances[0][np.ix_(targets, observed)]
        direct = model.means[0, targets] + cross @ np.linalg.solve(
            system, sample.vector[observed] - model.means[0, observed]
        )
        direct[phase.sampled_center_channel] = sample.vector[
            targets[phase.sampled_center_channel]
        ]
        np.testing.assert_allclose(result.mmse_rgb[index], direct, atol=2e-12)


def test_all_18_phases_restore_center_sample() -> None:
    model = _single_component_model(5)
    cache = prepare_conditional_cache(model, 1e-3)
    rgb = _scene(18, 18)
    samples = []
    for y in range(6):
        for x in range(6):
            patch = rgb[:, y:y + 5, x:x + 5]
            indices = observation_indices(5, x, y)
            from tools.xtrans_sparse_dictionary.dataset import PatchSample
            samples.append(PatchSample(
                source_id=f"{x}-{y}", group="phase", vector=patch.reshape(-1),
                indices=indices, phase=phase_index(x, y), x=x, y=y,
            ))
    assert len({sample.phase for sample in samples}) == 18
    result = conditional_predict(cache, samples)
    target = center_indices(5)
    for sample, value in zip(samples, result.mmse_rgb):
        phase = cache.phases[sample.phase]
        channel = phase.sampled_center_channel
        assert value[channel] == sample.vector[target[channel]]


def test_soft_responsibilities_are_normalized_and_finite() -> None:
    base = _single_component_model()
    model = JointColorGMM(
        **{
            **base.__dict__,
            "component_count": 2,
            "weights": np.asarray((0.4, 0.6)),
            "means": np.concatenate((base.means, base.means + 0.04), axis=0),
            "covariances": np.concatenate((base.covariances, base.covariances * 1.2), axis=0),
            "effective_counts": np.asarray((4.0, 6.0)),
        }
    )
    result = conditional_predict(
        prepare_conditional_cache(model, 1e-3),
        grid_samples(_scene(), "scene", "test", 3, 4),
        temperature=2.0,
    )
    np.testing.assert_allclose(np.sum(result.responsibilities, axis=1), 1.0)
    assert np.isfinite(result.mmse_rgb).all()
    assert np.all(result.entropy >= 0.0)


def test_log_sum_exp_survives_extreme_component_ratio() -> None:
    base = _single_component_model()
    model = JointColorGMM(
        **{
            **base.__dict__,
            "component_count": 2,
            "weights": np.asarray((1.0 - 1e-14, 1e-14)),
            "means": np.concatenate((base.means, base.means + 100.0), axis=0),
            "covariances": np.concatenate((base.covariances, base.covariances), axis=0),
            "effective_counts": np.asarray((10.0, 1e-12)),
        }
    )
    result = conditional_predict(
        prepare_conditional_cache(model, 1e-3),
        grid_samples(_scene(), "scene", "test", 3, 2),
    )
    assert np.isfinite(result.log_responsibilities).all()
    np.testing.assert_allclose(result.responsibilities[:, 0], 1.0)


def test_deterministic_full_covariance_em() -> None:
    rng = np.random.default_rng(91)
    matrix = np.concatenate((
        rng.normal(-0.2, 0.1, size=(60, 27)),
        rng.normal(0.3, 0.08, size=(60, 27)),
    ))
    first = fit_joint_gmm(
        matrix, 3, 2, dc_mode="absolute", seed=42, maximum_iterations=20,
    )
    second = fit_joint_gmm(
        matrix, 3, 2, dc_mode="absolute", seed=42, maximum_iterations=20,
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.covariances, second.covariances)
    for covariance in first.covariances:
        assert np.linalg.eigvalsh(covariance)[0] > 0.0


def test_diagonal_control_is_deterministic_and_has_no_cross_covariance() -> None:
    rng = np.random.default_rng(117)
    matrix = np.concatenate((
        rng.normal(-0.1, 0.12, size=(50, 27)),
        rng.normal(0.25, 0.09, size=(50, 27)),
    ))
    first = fit_joint_gmm(
        matrix, 3, 2, dc_mode="absolute", seed=43,
        maximum_iterations=20, covariance_type="diagonal",
    )
    second = fit_joint_gmm(
        matrix, 3, 2, dc_mode="absolute", seed=43,
        maximum_iterations=20, covariance_type="diagonal",
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.covariances, second.covariances)
    diagonal = np.arange(27)
    without_diagonal = first.covariances.copy()
    without_diagonal[:, diagonal, diagonal] = 0.0
    assert np.count_nonzero(without_diagonal) == 0
    assert np.all(first.covariances[:, diagonal, diagonal] > 0.0)


@pytest.mark.parametrize("mode", ("absolute", "observed-scalar", "observed-rgb"))
def test_observable_dc_modes(mode: str) -> None:
    sample = grid_samples(_scene(), "scene", "test", 5, 1)[0]
    normalized, dc = normalize_vector(sample.vector, sample.indices, mode)
    np.testing.assert_allclose(normalized + dc, sample.vector)
    if mode == "absolute":
        assert dc == 0.0


def test_invalid_tau_and_temperature_are_rejected() -> None:
    model = _single_component_model()
    with pytest.raises(ValueError):
        prepare_conditional_cache(model, 0.0)
    cache = prepare_conditional_cache(model, 1e-3)
    with pytest.raises(ValueError):
        conditional_predict(cache, grid_samples(_scene(), "x", "y", 3, 1), temperature=0.0)

from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_gmm.dataset import PatchSample
from tools.xtrans_gmr.model import (
    PhaseConditionedGMR,
    PhaseMixture,
    conditional_predict,
    fit_phase_conditioned_gmr,
    phase_contract,
    phase_training_matrix,
    prepare_gmr_cache,
    refine_phase_conditioned_gmr,
    shrink_covariances,
)
from tools.xtrans_mlri_internal.dataset import origin_cells
from tools.xtrans_sparse_dictionary.model import observation_indices, phase_index


def _scene(size: int = 16) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    return np.stack((
        0.1 + 0.7 * x / (size - 1),
        0.2 + 0.6 * y / (size - 1),
        0.15 + 0.5 * (x + y) / (2 * size - 2),
    ))


def _samples(patch_size: int) -> list[PatchSample]:
    rgb = _scene(18)
    rows = []
    for phase, (origin_x, origin_y) in enumerate(origin_cells()):
        patch = np.ascontiguousarray(rgb[:, origin_y:origin_y + patch_size, origin_x:origin_x + patch_size])
        indices = observation_indices(patch_size, origin_x, origin_y)
        rows.append(PatchSample(
            source_id=f"phase-{phase}", group="test", vector=patch.reshape(-1),
            indices=indices, phase=phase_index(origin_x, origin_y),
            x=origin_x, y=origin_y,
        ))
    return rows


def _manual_model(patch_size: int = 3, components: int = 1) -> PhaseConditionedGMR:
    dimension = patch_size * patch_size + 2
    phases = []
    for phase, (origin_x, origin_y) in enumerate(origin_cells()):
        rng = np.random.default_rng(1000 + phase)
        basis = rng.normal(size=(components, dimension, dimension))
        covariance = basis @ np.swapaxes(basis, 1, 2) / dimension
        covariance += 0.05 * np.eye(dimension)[None]
        observed, sampled, targets = phase_contract(patch_size, phase)
        phases.append(PhaseMixture(
            origin_x=origin_x, origin_y=origin_y,
            observed_indices=observed, sampled_center_channel=sampled,
            target_channels=targets,
            weights=np.full(components, 1.0 / components),
            means=np.stack([
                np.linspace(0.05 + 0.02 * component, 0.3, dimension)
                for component in range(components)
            ]),
            covariances=covariance, converged=True, iterations=1,
            lower_bound=0.0, effective_counts=np.full(components, 20.0),
        ))
    return PhaseConditionedGMR(
        patch_size=patch_size, component_count=components,
        dc_mode="absolute", covariance_floor=1e-6,
        training_sample_count=40, seed=1, phases=tuple(phases),
    )


def test_phase_contract_has_18_phases_and_canonical_targets() -> None:
    contracts = [phase_contract(7, phase) for phase in range(18)]
    assert len({indices.tobytes() for indices, _, _ in contracts}) == 18
    for indices, measured, targets in contracts:
        assert indices.shape == (49,)
        assert targets.tolist() == [channel for channel in range(3) if channel != measured]


def test_phase_training_matrix_contains_49_observations_and_two_targets() -> None:
    vectors = np.stack([sample.vector for sample in _samples(3)])
    absolute = phase_training_matrix(vectors, 3, 0, "absolute")
    normalized = phase_training_matrix(vectors, 3, 0, "observed-rgb")
    assert absolute.shape == (18, 11)
    assert normalized.shape == absolute.shape
    observed, _, targets = phase_contract(3, 0)
    center = 4
    expected = np.concatenate((vectors[:, observed], vectors[:, targets * 9 + center]), axis=1)
    np.testing.assert_array_equal(absolute, expected)
    assert np.max(np.abs(normalized - absolute)) > 0.0


def test_k1_matches_independent_phase_affine_lmmse() -> None:
    model = _manual_model(3)
    tau = 1e-3
    samples = _samples(3)
    result = conditional_predict(prepare_gmr_cache(model, tau), samples)
    area = 9
    center = 4
    for index, sample in enumerate(samples):
        phase = model.phases[sample.phase]
        covariance = phase.covariances[0]
        system = covariance[:area, :area] + tau * tau * np.eye(area)
        gain = covariance[area:, :area] @ np.linalg.inv(system)
        observed = sample.vector[phase.observed_indices]
        target = phase.means[0, area:] + gain @ (observed - phase.means[0, :area])
        expected = np.empty(3)
        expected[phase.sampled_center_channel] = sample.vector[
            phase.sampled_center_channel * area + center
        ]
        expected[phase.target_channels] = target
        np.testing.assert_allclose(result.mmse_rgb[index], expected, atol=2e-12)


def test_measured_center_is_exact_and_responsibilities_normalize() -> None:
    model = _manual_model(5, 2)
    samples = _samples(5)
    result = conditional_predict(prepare_gmr_cache(model, 3e-3), samples, temperature=2.0)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0)
    assert np.isfinite(result.mmse_rgb).all()
    for index, sample in enumerate(samples):
        phase = model.phases[sample.phase]
        center_index = phase.sampled_center_channel * 25 + 12
        assert result.mmse_rgb[index, phase.sampled_center_channel] == sample.vector[center_index]


def test_global_covariance_shrinkage_is_bounded() -> None:
    model = _manual_model(3, 2)
    global_model = _manual_model(3, 1)
    shrunk = shrink_covariances(model, global_model, alpha=0.2)
    expected = 0.8 * model.phases[0].covariances[0] + 0.2 * global_model.phases[0].covariances[0]
    np.testing.assert_allclose(shrunk.phases[0].covariances[0], expected)
    assert shrunk.shrinkage == "uniform-0.2"
    occupied = shrink_covariances(model, global_model, kappa=20.0)
    np.testing.assert_allclose(
        occupied.phases[0].covariances[0],
        0.5 * model.phases[0].covariances[0] + 0.5 * global_model.phases[0].covariances[0],
    )


def test_small_full_model_fit_is_deterministic() -> None:
    rng = np.random.default_rng(91)
    vectors = rng.normal(size=(80, 27))
    first = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=44,
        maximum_iterations=3, tolerance=0.0,
    )
    second = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=44,
        maximum_iterations=3, tolerance=0.0,
    )
    for left, right in zip(first.phases, second.phases):
        np.testing.assert_array_equal(left.weights, right.weights)
        np.testing.assert_array_equal(left.means, right.means)
        np.testing.assert_array_equal(left.covariances, right.covariances)


def test_refinement_continues_every_phase_checkpoint() -> None:
    rng = np.random.default_rng(121)
    vectors = rng.normal(size=(80, 27))
    initial = fit_phase_conditioned_gmr(
        vectors, 3, 2, dc_mode="absolute", seed=45,
        maximum_iterations=2, tolerance=0.0,
    )
    refined = refine_phase_conditioned_gmr(
        vectors, initial, additional_iterations=2, tolerance=0.0,
    )
    assert all(phase.iterations == 4 for phase in refined.phases)
    assert any(
        not np.array_equal(before.means, after.means)
        for before, after in zip(initial.phases, refined.phases)
    )


def test_invalid_controls_are_rejected() -> None:
    model = _manual_model()
    with pytest.raises(ValueError):
        prepare_gmr_cache(model, 0.0)
    with pytest.raises(ValueError):
        conditional_predict(prepare_gmr_cache(model, 1e-3), _samples(3), temperature=0.0)
    with pytest.raises(ValueError):
        shrink_covariances(model, _manual_model(), alpha=0.2, kappa=10.0)

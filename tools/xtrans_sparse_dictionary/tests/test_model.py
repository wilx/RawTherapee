from __future__ import annotations

import numpy as np
import pytest

from tools.xtrans_mlri_internal.dataset import origin_cells
from tools.xtrans_sparse_dictionary.model import (
    blind_cfa_reconstruction,
    dct_dictionary,
    dictionary_coherence,
    lasso_cfa_reconstruction,
    observation_indices,
    omp,
    oracle_support_reconstruction,
    restore_observed,
    train_mod_dictionary,
)


def test_dct_is_orthonormal_and_deterministic():
    first = dct_dictionary(5)
    second = dct_dictionary(5)
    assert first.shape == (75, 75)
    assert np.array_equal(first, second)
    assert np.allclose(first.T @ first, np.eye(75), atol=1e-12)
    assert dictionary_coherence(first) < 1e-12
    rgb = dct_dictionary(5, color_basis="rgb")
    assert np.allclose(rgb.T @ rgb, np.eye(75), atol=1e-12)
    assert not np.array_equal(first, rgb)


def test_omp_exact_sparse_signal_and_monotonic_residual():
    rng = np.random.default_rng(17)
    dictionary = rng.normal(size=(24, 12))
    dictionary /= np.linalg.norm(dictionary, axis=0)
    truth = np.zeros(12)
    truth[[1, 5, 9]] = (0.75, -0.4, 0.2)
    signal = dictionary @ truth
    result = omp(dictionary, signal, 3)
    assert set(result.support) == {1, 5, 9}
    assert np.all(np.diff(result.residual_history) <= 1e-14)
    assert np.max(np.abs(dictionary @ result.coefficients - signal)) < 1e-8


@pytest.mark.parametrize("origin", origin_cells())
def test_known_support_and_native_sample_restoration_for_all_phases(origin):
    dictionary = dct_dictionary(5)
    support = np.asarray((0, 7, 31, 52), dtype=np.int64)
    coefficients = np.asarray((0.4, -0.2, 0.1, 0.3))
    truth = dictionary[:, support] @ coefficients
    indices = observation_indices(5, *origin)
    reconstruction, diagnostic = oracle_support_reconstruction(
        dictionary, truth, indices, support,
    )
    assert diagnostic["rank"] == support.size
    assert np.max(np.abs(reconstruction - truth)) < 1e-6
    corrupted = reconstruction.copy()
    corrupted[indices] += 1.0
    restored = restore_observed(corrupted, truth, indices)
    assert np.array_equal(restored[indices], truth[indices])


def test_blind_single_atom_recovery():
    dictionary = dct_dictionary(5)
    truth = dictionary[:, 9] * 0.4
    indices = observation_indices(5, 0, 0)
    reconstruction, result = blind_cfa_reconstruction(
        dictionary, truth, indices, 1,
    )
    assert result.support.tolist() == [9]
    assert np.max(np.abs(reconstruction - truth)) < 1e-8


def test_lasso_objective_converges_and_recovers_a_sparse_signal():
    rng = np.random.default_rng(113)
    dictionary = rng.normal(size=(27, 18))
    dictionary /= np.linalg.norm(dictionary, axis=0)
    truth_coefficients = np.zeros(18)
    truth_coefficients[[2, 11]] = (0.7, -0.3)
    truth = dictionary @ truth_coefficients
    indices = np.arange(27, dtype=np.int64)
    reconstruction, result = lasso_cfa_reconstruction(
        dictionary, truth, indices, 1e-5, iterations=200,
    )
    assert result.residual_history[-1] <= result.residual_history[0]
    assert np.sqrt(np.mean((reconstruction - truth) ** 2)) < 1e-4


def test_dictionary_training_is_deterministic_and_finite():
    rng = np.random.default_rng(23)
    samples = rng.normal(size=(27, 40))
    first = train_mod_dictionary(
        samples, 32, outer_iterations=2, coding_iterations=4, seed=31,
    )
    second = train_mod_dictionary(
        samples, 32, outer_iterations=2, coding_iterations=4, seed=31,
    )
    assert np.array_equal(first.dictionary, second.dictionary)
    assert np.isfinite(first.dictionary).all()
    assert np.allclose(np.linalg.norm(first.dictionary, axis=0), 1.0)
    assert len(first.objectives) == 2


def test_constant_rgb_is_finite_and_measured_samples_are_exact():
    dictionary = dct_dictionary(5)
    truth = np.full(75, 0.42)
    indices = observation_indices(5, 5, 2)
    reconstruction, _ = blind_cfa_reconstruction(dictionary, truth, indices, 4)
    restored = restore_observed(reconstruction, truth, indices)
    assert np.isfinite(restored).all()
    assert np.array_equal(restored[indices], truth[indices])

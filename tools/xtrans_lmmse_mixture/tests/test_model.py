from __future__ import annotations

import numpy as np

from tools.xtrans_lmmse.model import collect_phase_samples, derive_filter_bank, phase_statistics
from tools.xtrans_lmmse_mixture.model import (
    FeatureNormalization,
    MixtureBank,
    MixtureDefinition,
    classify_features,
    observable_features,
    oracle_select,
    predict_mixture,
    soft_weights,
)
from tools.xtrans_lmmse_mixture.synthetic import scenes


def _definition() -> MixtureDefinition:
    return MixtureDefinition(
        name="test",
        partition="set1",
        activity_feature="highpass_variance",
        activity_threshold=0.0,
        anisotropy_threshold=0.5,
        activity_width=0.2,
        anisotropy_width=0.1,
        class_names=("smooth", "edge", "texture"),
        activity_quantile=0.4,
        anisotropy_quantile=0.6,
    )


def test_observable_features_are_constant_safe_and_directional() -> None:
    constant = np.full((7, 121), 0.4, dtype=np.float64)
    values = observable_features(constant, 0, 11)
    assert np.max(np.abs(values["raw_variance"])) < 1e-30
    assert np.max(np.abs(values["highpass_variance"])) < 1e-30
    assert np.max(np.abs(values["chroma_variation"])) < 1e-30
    assert np.max(np.abs(values["anisotropy"])) < 1e-30

    horizontal_edge = np.zeros((1, 11, 11), dtype=np.float64)
    horizontal_edge[:, 6:, :] = 1.0
    edge_features = observable_features(horizontal_edge.reshape(1, -1), 0, 11)
    assert edge_features["anisotropy"][0] > 0.5
    assert edge_features["vertical_energy"][0] > edge_features["horizontal_energy"][0]


def test_explicit_rule_and_soft_weights_are_bounded() -> None:
    definition = _definition()
    features = {
        "highpass_variance": np.asarray((-1.0, 1.0, 1.0)),
        "anisotropy": np.asarray((0.9, 0.9, 0.1)),
    }
    assert classify_features(features, definition).tolist() == [0, 1, 2]
    weights = soft_weights(features, definition)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)


def test_mixture_restores_samples_and_oracle_selects_best() -> None:
    y, x = np.mgrid[:35, :37]
    truth = np.stack(
        (
            0.15 + 0.6 * x / 36.0,
            0.10 + 0.7 * y / 34.0,
            0.20 + 0.5 * (x + y) / 70.0,
        )
    )
    observations, targets = collect_phase_samples(
        truth, 11, (5, 5, 32, 30), maximum_per_phase=None
    )
    statistics = phase_statistics(observations, targets, dc_mode="m2", support=11)
    bank = derive_filter_bank(statistics, 11, 1e-3, dc_mode="m2")
    normalization = FeatureNormalization(
        raw_variance_median=np.ones(18),
        highpass_variance_median=np.ones(18),
        chroma_variation_median=np.ones(18),
    )
    mixture = MixtureBank(
        definition=_definition(),
        normalization=normalization,
        banks=(bank, bank, bank),
        fallback_to_global=np.zeros((3, 18), dtype=bool),
    )
    output, classes = predict_mixture(mixture, truth, (0, 0, 37, 35))
    assert output.shape == truth.shape
    assert classes.shape == truth.shape[1:]
    assert np.isfinite(output).all()

    candidates = (truth + 0.1, truth.copy(), truth - 0.2)
    oracle, selected = oracle_select(candidates, truth, 7)
    np.testing.assert_array_equal(oracle, truth)
    assert set(np.unique(selected)) == {1}


def test_synthetic_suite_is_deterministic_and_bounded() -> None:
    first = scenes()
    second = scenes()
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) >= 20
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["rgb"], right["rgb"])
        assert np.min(left["rgb"]) >= 0.0
        assert np.max(left["rgb"]) <= 1.0

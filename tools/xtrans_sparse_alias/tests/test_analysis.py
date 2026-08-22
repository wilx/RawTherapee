from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.xtrans_alias.analysis import (
    CARRIER_INDICES,
    DIRECTIONS,
    canonical_json_bytes,
    carrier_signature,
    cfa_variants,
    mask_spectra,
)
from tools.xtrans_sparse_alias.analysis import (
    BASIS,
    BASIS_NAMES,
    DESCRIPTORS,
    alias_dictionary,
    confidence_study,
    grouped_real_study,
    least_squares_support,
    natural_sparsity_study,
    oracle_image_recovery,
    orthogonal_matching_pursuit,
    support_metrics,
    blind_image_recovery,
)
from tools.xtrans_sparse_alias.generate import generate


EXPECTED_MANIFEST_SHA256 = "f550c51edf78f34d7e1ed006ff02526c607e1f251e3c5e936db93b441e4455f1"


def test_dictionary_reuses_the_validated_operator_and_variants():
    actual = alias_dictionary()
    rgb = np.column_stack(
        [
            carrier_signature(carrier, np.eye(3)[color])
            for carrier in CARRIER_INDICES
            for color in range(3)
        ]
    )
    assert np.max(np.abs(actual - rgb @ np.kron(np.eye(18), BASIS.T))) < 2e-16
    reference_singular = np.linalg.svd(actual, compute_uv=False)
    for cfa in cfa_variants():
        spectra = mask_spectra(cfa)
        variant = np.column_stack(
            [
                carrier_signature(carrier, DIRECTIONS[name], spectra)
                for carrier, name in DESCRIPTORS
            ]
        )
        assert np.max(np.abs(np.linalg.svd(variant, compute_uv=False) - reference_singular)) < 2e-15


def test_oracle_one_component_is_exact_for_all_atoms():
    atoms = alias_dictionary()
    for index in range(54):
        coefficient = np.asarray([0.37 - 0.21j + index * 0.001])
        observation = atoms[:, (index,)] @ coefficient
        recovered = least_squares_support(observation, (index,), atoms)
        assert recovered.rank == 1
        assert recovered.relative_residual < 1e-14
        assert np.max(np.abs(recovered.coefficients - coefficient)) < 1e-14


def test_omp_refits_monotonically_and_uses_complete_constellations():
    atoms = alias_dictionary()
    support = (2, 19, 47)
    coefficients = np.asarray((1.0 + 0.2j, -0.55 + 0.35j, 0.25 - 0.4j))
    observation = atoms[:, support] @ coefficients
    recovered = orthogonal_matching_pursuit(observation, 3, dictionary=atoms)
    assert support_metrics(support, recovered.support)["exact"] == 1.0
    assert recovered.relative_residual < 1e-14
    assert all(
        right <= left + 2e-14
        for left, right in zip(recovered.residual_history, recovered.residual_history[1:])
    )
    peak = orthogonal_matching_pursuit(
        observation, 3, dictionary=atoms, mode="single-peak"
    )
    assert peak.relative_residual > recovered.relative_residual + 0.1


def test_known_null_competitor_is_detected():
    result = confidence_study(trials_per_size=8)
    assert result["known_null_competitor"]["relative_residual"] < 1e-14
    assert result["populations"]["deficient"]["count"] == 50
    assert result["populations"]["deficient"]["gap_p90"] < 1e-14


def test_grouped_real_recovery_handles_half_carrier_phase():
    result = grouped_real_study()
    assert result["phase_sweep"]["generic"]["grouped_exact_rate"] == 1.0
    assert result["phase_sweep"]["half_carrier"]["grouped_exact_rate"] > 0.99
    assert (
        result["phase_sweep"]["half_carrier"]["grouped_exact_rate"]
        > result["phase_sweep"]["half_carrier"]["ordinary_exact_rate"] + 0.15
    )


def test_oracle_image_separates_observable_and_deficient_edges():
    from tools.xtrans_alias.analysis import synthetic_edge_cases

    scenes = synthetic_edge_cases()
    axis, axis_metrics = oracle_image_recovery(scenes["saturated_red_gray"])
    diagonal, diagonal_metrics = oracle_image_recovery(scenes["diagonal_red_green"])
    sweep, sweep_metrics = oracle_image_recovery(scenes["abc_frequency_sweep"])
    assert np.max(np.abs(axis - scenes["saturated_red_gray"])) < 1e-14
    assert axis_metrics["oracle_recoverable_energy_fraction"] == pytest.approx(1.0)
    assert diagonal_metrics["deficient_energy_fraction"] > 0.33
    assert sweep_metrics["deficient_energy_fraction"] == 1.0
    assert np.max(np.abs(diagonal - scenes["diagonal_red_green"])) > 0.1
    assert np.max(np.abs(sweep - scenes["abc_frequency_sweep"])) > 0.3


def test_blind_image_recovery_accepts_measured_channel_means():
    from tools.xtrans_alias.analysis import CANONICAL_XTRANS

    y, x = np.mgrid[:24, :24]
    rgb = np.stack((0.2 + x / 100, 0.3 + y / 120, 0.4 + (x + y) / 180))
    tiled = np.tile(CANONICAL_XTRANS, (4, 4))
    means = np.asarray([rgb[color][tiled == color].mean() for color in range(3)])
    reconstructed, metrics = blind_image_recovery(
        rgb, max_atoms=2, channel_means=means
    )
    assert reconstructed.shape == rgb.shape
    assert np.isfinite(reconstructed).all()
    assert metrics["channel_means_source"] == "caller supplied"


def test_natural_sparsity_contract_on_small_rgb_input():
    y, x = np.mgrid[:24, :24]
    rgb = np.stack(
        (
            0.5 + 0.3 * np.sin(2.0 * np.pi * x / 24.0),
            0.4 + 0.2 * np.cos(2.0 * np.pi * y / 24.0),
            0.3 + 0.1 * np.sin(2.0 * np.pi * (x + y) / 24.0),
        ),
        axis=-1,
    )
    result = natural_sparsity_study(
        [{"category": "procedural", "name": "test", "rgb": rgb, "sha256": "0" * 64}],
        window_sizes=(24,),
        maximum_windows_per_size=1,
    )
    assert result["windows"] == 1
    assert result["family_samples"] == 32
    assert result["fixed_cardinality_recovery"][0]["support_size"] == 1
    assert set(result["summaries"]) == {*BASIS_NAMES, "mixed"}


def test_generator_is_deterministic_and_refuses_replacement(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest, first_digest = generate(first, force=False)
    second_manifest, second_digest = generate(second, force=False)
    assert first_digest == second_digest == EXPECTED_MANIFEST_SHA256
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    for first_file in sorted(first.iterdir()):
        assert first_file.read_bytes() == (second / first_file.name).read_bytes()
    with pytest.raises(FileExistsError):
        generate(first, force=False)


def test_tracked_corpus_is_canonical():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes" / "images" / "xtrans-sparse-alias"
    manifest = corpus / "recovery-characterization.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.read_bytes() == canonical_json_bytes(payload)
    for plot in payload["plots"]:
        path = corpus / plot["filename"]
        assert path.stat().st_size == plot["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plot["sha256"]

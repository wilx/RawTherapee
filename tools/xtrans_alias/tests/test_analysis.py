from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.xtrans_alias.analysis import (
    CANONICAL_XTRANS,
    CARRIER_INDICES,
    DIRECTIONS,
    build_characterization,
    canonical_json_bytes,
    carrier_signature,
    exact_carrier_table,
    full_alias_matrix,
    mask_spectra,
    multifrequency_linearity,
    normalized_similarity,
    singular_summary,
)
from tools.xtrans_alias.generate import generate


EXPECTED_MANIFEST_SHA256 = "5f96d3a329516dd0396a0258553a8b83176a2ea3e3dec3efcc4ac34044709dd5"


@pytest.fixture(scope="module")
def characterization():
    return build_characterization()


def test_exact_mask_spectrum_and_periodicity():
    assert np.array_equal(CANONICAL_XTRANS, np.roll(CANONICAL_XTRANS, (3, 3), axis=(0, 1)))
    assert [np.count_nonzero(CANONICAL_XTRANS == color) for color in range(3)] == [8, 20, 8]
    table = exact_carrier_table()
    assert len(table) == 18
    assert sum(row["nonzero"] for row in table) == 13
    assert [row["index"] for row in table if not row["nonzero"]] == [
        [1, 1], [5, 1], [3, 3], [1, 5], [5, 5]
    ]
    spectra = mask_spectra()
    for ky in range(6):
        for kx in range(6):
            if (kx + ky) % 2:
                assert np.max(np.abs(spectra[:, ky, kx])) < 1e-14


def test_shifted_replica_theory(characterization):
    check = characterization[0]["single_frequency_replica_check"]
    assert check["image_size"] == [96, 96]
    assert check["cases"]["luma"]["peak_count"] == 1
    assert check["cases"]["c1_r_minus_b"]["peak_count"] == 4
    assert check["cases"]["c2_green_opponent"]["peak_count"] == 9
    for case in check["cases"].values():
        assert case["maximum_complex_error"] < 2e-14


def test_alias_family_rank_and_explicit_null(characterization):
    summary = singular_summary(full_alias_matrix())
    assert summary["rows"] == 18
    assert summary["columns"] == 54
    assert summary["rank"] == 18
    assert summary["nullity"] == 36
    assert np.allclose(summary["singular_values"], 1.0, atol=7e-16, rtol=0.0)
    null_example = characterization[0]["alias_family"]["sparse_exact_null_example"]
    assert len(null_example["terms"]) == 4
    assert null_example["relative_residual"] < 1e-14


def test_known_frequency_is_well_conditioned(characterization):
    summary = characterization[0]["alias_family"]["known_frequency_rgb_operator"]
    assert summary["rank"] == 3
    assert summary["nullity"] == 0
    assert summary["singular_values"] == pytest.approx(
        [np.sqrt(5.0 / 9.0), np.sqrt(2.0 / 9.0), np.sqrt(2.0 / 9.0)],
        abs=2e-15,
    )
    assert summary["condition_nonzero"] == pytest.approx(np.sqrt(5.0 / 2.0), abs=2e-15)


def test_confusion_metric_and_phase_hotspots(characterization):
    signature = carrier_signature((0, 0), DIRECTIONS["luma"])
    assert normalized_similarity(signature, signature) == pytest.approx(1.0)
    red = carrier_signature((0, 0), DIRECTIONS["red"])
    green = carrier_signature((0, 0), DIRECTIONS["green"])
    assert normalized_similarity(red, green) < 1e-15
    real = characterization[0]["phase_sensitive_real_confusion"]
    assert real["luma_to_chroma"]["maximum"] == pytest.approx(0.8068982213550734)
    assert real["luma_to_chroma"]["minimum"] == pytest.approx(0.48296291314453416)
    assert real["chroma_to_chroma"]["maximum"] == pytest.approx(0.5705632040475361)


def test_variants_edges_and_linearity(characterization):
    payload = characterization[0]
    variants = payload["variant_invariance"]
    assert variants["variant_count"] == 18
    assert variants["maximum_mask_magnitude_error"] < 3e-16
    assert variants["maximum_singular_value_error"] < 1e-14
    edges = payload["edge_cases"]
    assert edges["vertical_luma"]["rank_deficient_energy_fraction"] == 0.0
    assert edges["horizontal_red_green"]["rank_deficient_energy_fraction"] == 0.0
    assert edges["diagonal_red_green"]["rank_deficient_energy_fraction"] > 0.33
    assert edges["diagonal_red_green"]["rank_deficient_energy_fraction"] == pytest.approx(
        edges["antidiagonal_red_green"]["rank_deficient_energy_fraction"]
    )
    assert edges["abc_saturated_edges"]["rank_deficient_energy_fraction"] == pytest.approx(0.20)
    assert edges["abc_frequency_sweep"]["rank_deficient_energy_fraction"] > 0.999999
    linearity = multifrequency_linearity()
    assert linearity["sample_maximum_error"] == 0.0
    assert linearity["fft_maximum_error"] < 5e-13


def test_bayer_context_preserves_the_sampling_distinction(characterization):
    bayer = characterization[0]["bayer_context"]
    assert bayer["full_rgb_operator"]["rank"] == 4
    assert bayer["full_rgb_operator"]["nullity"] == 8
    assert bayer["known_frequency_rgb_operator"]["condition_nonzero"] == pytest.approx(
        np.sqrt(2.0)
    )
    luma_c2 = next(
        row
        for row in bayer["complex_confusion"]
        if row["source"] == "luma" and row["competitor"] == "c2_green_opponent"
    )
    assert luma_c2["similarity"] == pytest.approx(np.sqrt(0.9))


def test_generator_is_deterministic_and_refuses_replacement(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest, first_digest = generate(first, force=False)
    second_manifest, second_digest = generate(second, force=False)
    assert first_digest == second_digest == EXPECTED_MANIFEST_SHA256
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    for first_file in sorted(first.iterdir()):
        second_file = second / first_file.name
        assert second_file.read_bytes() == first_file.read_bytes()
    with pytest.raises(FileExistsError):
        generate(first, force=False)


def test_tracked_corpus_is_canonical():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes" / "images" / "xtrans-alias"
    manifest_path = corpus / "characterization.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.read_bytes() == canonical_json_bytes(payload)
    for plot in payload["plots"]:
        path = corpus / plot["filename"]
        assert path.stat().st_size == plot["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plot["sha256"]

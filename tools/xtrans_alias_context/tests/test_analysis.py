from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.xtrans_alias.analysis import alias_families, canonical_json_bytes, synthetic_edge_cases
from tools.xtrans_alias_context.analysis import (
    Compatibility,
    Hypothesis,
    WindowFamily,
    beam_hypotheses,
    compatibility,
    confidence_calibration,
    dictionary_for_origin,
    extract_window_spectra,
    null_context_study,
    support_jaccard,
)
from tools.xtrans_sparse_alias.analysis import alias_dictionary, family_coordinates


EXPECTED_MANIFEST_SHA256 = "fb2e4a0497cb88c7121a1bff81d31f7c613ba2b609995a5fd6c97eb1d2a3d765"


def test_phase_zero_dictionary_reuses_exact_sparse_operator():
    assert np.max(np.abs(dictionary_for_origin((0, 0)) - alias_dictionary())) == 0.0


def test_shifted_dictionary_matches_direct_window_sampling():
    rng = np.random.default_rng(0x434641)
    rgb = rng.random((3, 72, 72))
    for origin in ((0, 0), (6, 6), (7, 5)):
        spectra = extract_window_spectra(rgb, origin, 48)
        atoms = dictionary_for_origin(origin)
        assert max(np.max(np.abs(atoms @ truth - observed)) for observed, truth, _coordinates in spectra) < 5e-17


def test_beam_proposals_are_distinct_fixed_cardinality_and_deterministic():
    atoms = alias_dictionary()
    support = (2, 19, 47, 50)
    coefficients = np.asarray((1.0 + 0.2j, -0.55 + 0.35j, 0.25 - 0.4j, 0.11 + 0.09j))
    observation = atoms[:, support] @ coefficients
    first = beam_hypotheses(observation, 4, 8, beam_width=8)
    second = beam_hypotheses(observation, 4, 8, beam_width=8)
    assert len(first) == 8
    assert all(len(value.support) == 4 for value in first)
    assert len({value.support for value in first}) == len(first)
    assert [value.support for value in first] == [value.support for value in second]
    assert np.allclose(
        [value.normalized_residual for value in first],
        [value.normalized_residual for value in second],
        rtol=0.0,
        atol=0.0,
    )


def _record(origin: tuple[int, int], coefficient: complex) -> WindowFamily:
    size = 24
    coordinates = family_coordinates(size, alias_families(size)[3])
    hypothesis = Hypothesis(
        coefficients=np.asarray((coefficient,)),
        condition=1.0,
        energy=float(abs(coefficient) ** 2),
        normalized_residual=0.0,
        sigma_min=1.0,
        singular_values=(1.0,),
        support=(6,),
    )
    truth = np.zeros(54, dtype=np.complex128)
    truth[6] = coefficient
    return WindowFamily(
        candidates=(hypothesis,),
        coordinates=coordinates,
        family_index=3,
        observation=np.zeros(18, dtype=np.complex128),
        origin=origin,
        size=size,
        truth=truth,
        truth_support=(6,),
    )


def test_phase_compatibility_uses_predicted_fourier_translation():
    left = _record((0, 0), 1.0 + 0.0j)
    x, y = left.coordinates[2]
    fx = x / left.size if x <= left.size // 2 else (x - left.size) / left.size
    fy = y / left.size if y <= left.size // 2 else (y - left.size) / left.size
    delta = (6, 12)
    predicted = 2.0 * math.pi * (fx * delta[0] + fy * delta[1])
    right = _record(delta, np.exp(1j * predicted))
    terms = compatibility(left, left.candidates[0], right, right.candidates[0])
    assert terms.phase < 1e-14
    assert terms.support < 1e-14
    wrong = _record(delta, np.exp(1j * (predicted + math.pi / 2.0)))
    assert compatibility(left, left.candidates[0], wrong, wrong.candidates[0]).phase == 0.5


def test_support_jaccard_distinguishes_material_alternatives():
    assert support_jaccard((1, 2, 3), (1, 2, 3)) == 1.0
    assert support_jaccard((1, 2, 3), (1, 4, 5)) == 0.2
    assert support_jaccard((), ()) == 1.0


def test_known_null_remains_exact_across_aligned_and_nonaligned_windows():
    result = null_context_study()["cases"]
    assert result["single"]["rank"] == 4
    assert result["aligned_6"]["rank"] == 4
    assert result["mixed_non_aligned"]["rank"] == 4
    assert result["aligned_6"]["truth_competitor_relative_observation_difference"] < 2e-14
    assert result["mixed_non_aligned"]["truth_competitor_relative_observation_difference"] < 2e-14


def test_confidence_curve_reports_monotonicity():
    samples = [
        {"confidence": 4.0, "coefficient_error": 0.01, "correct": True},
        {"confidence": 3.0, "coefficient_error": 0.02, "correct": True},
        {"confidence": 2.0, "coefficient_error": 0.2, "correct": False},
        {"confidence": 1.0, "coefficient_error": 0.4, "correct": False},
    ]
    result = confidence_calibration(samples)
    assert result["error_monotone_with_retained_fraction"]
    assert result["curve"][0]["coefficient_error_mean"] < result["curve"][-1]["coefficient_error_mean"]


def test_easy_scene_window_contract_is_finite():
    rgb = synthetic_edge_cases(96)["vertical_red_green"]
    spectra = extract_window_spectra(rgb, (24, 24), 48)
    energies = [float(np.sum(np.abs(value[1]) ** 2)) for value in spectra]
    family = int(np.argmax(energies))
    proposals = beam_hypotheses(
        spectra[family][0], 3, 5, beam_width=4, dictionary=dictionary_for_origin((24, 24))
    )
    assert len(proposals) == 5
    assert all(np.isfinite(value.normalized_residual) for value in proposals)


def test_tracked_corpus_is_canonical():
    root = Path(__file__).resolve().parents[3]
    corpus = root / "devnotes" / "images" / "xtrans-alias-context"
    manifest = corpus / "context-characterization.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.read_bytes() == canonical_json_bytes(payload)
    for plot in payload["plots"]:
        path = corpus / plot["filename"]
        assert path.stat().st_size == plot["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plot["sha256"]

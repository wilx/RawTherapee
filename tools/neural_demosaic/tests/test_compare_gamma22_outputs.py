from __future__ import annotations

import numpy as np
import pytest

from tools.neural_demosaic.compare_gamma22_outputs import (
    COLOR_COMMON_LIMIT,
    ComparisonError,
    EXPECTED_SHAPE,
    analyze_arrays,
    parse_execution,
    validate_metadata,
)


def image(value=(0.3, 0.4, 0.5)):
    result = np.empty((96, 96, 3), dtype=np.uint16)
    result[...] = np.rint(np.asarray(value) * 65535).astype(np.uint16)
    return result


def analyze(linear, gamma, mark):
    return analyze_arrays(
        {"linear": linear, "gamma22": gamma, "markesteijn": mark},
        (12, 18, 60, 60),
    )[0]


def test_identical_outputs_have_zero_phase_and_color_difference():
    reference = image()
    result = analyze(reference, reference, reference)
    assert result["objective"]["pass"]
    assert result["objective"]["color"]["gamma_minus_markesteijn_rgb"] == [0.0, 0.0, 0.0]
    assert result["methods"]["gamma22"]["phase"]["rms_rgb"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)


def test_injected_three_by_three_bias_is_detected():
    reference = image()
    linear = reference.copy()
    gamma = reference.copy()
    for y in range(gamma.shape[0]):
        for x in range(gamma.shape[1]):
            if x % 3 == 0 and y % 3 == 0:
                gamma[y, x, 0] = min(65535, int(gamma[y, x, 0]) + 1800)
                linear[y, x, 0] = min(65535, int(linear[y, x, 0]) + 3600)
    result = analyze(linear, gamma, reference)
    assert not result["objective"]["phase"]["pass"]
    assert result["methods"]["gamma22"]["phase"]["spread_rgb"][0] > 0.02


def test_neutral_luminance_shift_is_not_a_color_cast():
    reference = image()
    shifted = image((0.302, 0.402, 0.502))
    result = analyze(reference, shifted, reference)
    color = result["objective"]["color"]
    assert color["pass"]
    assert abs(color["common_luminance_delta"] - 0.002) < 2e-5


def test_excessive_neutral_luminance_shift_fails():
    reference = image()
    shifted = image(tuple(value + COLOR_COMMON_LIMIT * 2 for value in (0.3, 0.4, 0.5)))
    assert not analyze(reference, shifted, reference)["objective"]["color"]["pass"]


def test_channel_dependent_shift_fails_color_gate():
    reference = image()
    shifted = image((0.29, 0.4, 0.51))
    result = analyze(reference, shifted, reference)
    assert not result["objective"]["color"]["pass"]
    assert result["objective"]["color"]["rgb_delta_range"] > 0.019


def test_invalid_crop_is_rejected():
    reference = image()
    with pytest.raises(ComparisonError, match="outside"):
        analyze_arrays(
            {"linear": reference, "gamma22": reference, "markesteijn": reference},
            (80, 80, 20, 20),
        )


def test_mismatched_metadata_is_rejected():
    common = {
        "shape": list(EXPECTED_SHAPE),
        "dtype": "uint16",
        "bits_per_sample": 16,
        "photometric": 2,
        "icc_profile_bytes": 100,
        "icc_profile_sha256": "1" * 64,
    }
    different = dict(common, icc_profile_sha256="2" * 64)
    with pytest.raises(ComparisonError, match="metadata differs"):
        validate_metadata({"linear": common, "gamma22": different, "markesteijn": common})


def test_execution_diagnostic_and_timing_are_authenticated():
    log = (
        "DemosaicNet X-Trans completed: method=demosaicnet-xtrans-gamma22 "
        "artifact=b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2 "
        "input_tile=192x192 output_core=168x168 tiles=1426 workers=24 "
        "workspace_per_worker=18483200 workspace_total=443596800 elapsed_us=123456\n"
    )
    result = parse_execution(log, "elapsed_seconds=601.25\nmax_rss_kb=2200000\n")
    assert result["active_workers"] == 24
    assert result["elapsed_seconds"] == 601.25
    with pytest.raises(ComparisonError, match="fell back"):
        parse_execution(log + "falling back to 3-pass (Markesteijn)", "elapsed_seconds=1\nmax_rss_kb=2\n")

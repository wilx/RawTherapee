from __future__ import annotations

import numpy as np
import pytest

from tools.neural_demosaic.compare_gamma22_outputs import ComparisonError
from tools.xtrans_mlri.compare_corrected_outputs import pixel_difference_summary
from tools.xtrans_mlri.compare_final_only_outputs import (
    METHOD as FINAL_ONLY_METHOD,
    analyze_final_only_arrays,
    blue_outlier_probe,
    local_blue_excess,
    parse_final_only_execution,
)
from tools.xtrans_mlri.compare_paper_core_outputs import (
    METHOD_2014,
    METHOD_2016,
    analyze_paper_arrays,
    parse_paper_execution,
)
from tools.xtrans_mlri.compare_outputs import (
    analyze_arrays,
    method_delta,
    parse_execution,
)


def test_identical_arrays_have_zero_delta() -> None:
    rows, columns = np.indices((48, 48))
    image = np.stack((rows * 101 + columns, rows * 31 + columns * 17,
                      rows * 7 + columns * 89), axis=2).astype(np.uint16)
    result = analyze_arrays({"mlri": image, "markesteijn": image.copy()}, (0, 0, 48, 48))
    delta = result["mlri_minus_markesteijn"]
    assert delta["crop_mean_rgb_delta"] == [0.0, 0.0, 0.0]
    assert delta["common_luminance_delta"] == 0.0
    assert delta["rgb_delta_range"] == 0.0
    assert delta["observed_sample_delta_rms"] == 0.0


def test_phase_bias_and_color_shift_are_measured() -> None:
    mark = np.full((48, 48, 3), 20000, dtype=np.uint16)
    mlri = mark.copy()
    for y in range(48):
        for x in range(48):
            mlri[y, x, 0] += (y % 3) * 60 + (x % 3) * 20
            mlri[y, x, 1] += 100
    result = analyze_arrays({"mlri": mlri, "markesteijn": mark}, (0, 0, 48, 48))
    assert result["methods"]["mlri"]["phase"]["rms_rgb"][0] > 0
    assert result["mlri_minus_markesteijn"]["crop_mean_rgb_delta"][1] == pytest.approx(100 / 65535)
    assert result["mlri_minus_markesteijn"]["rgb_delta_range"] > 0


def test_observed_delta_uses_dscf0771_cfa_phase() -> None:
    right = np.zeros((6, 6, 3), dtype=np.uint16)
    left = right.copy()
    # DSCF0771 has green at (1,0), whereas the canonical MLRI cell has blue.
    # A blue-only change at this coordinate is therefore interpolated, not an
    # observed-sample change.
    left[0, 1, 2] = 1000
    methods = {
        name: {"crop_mean_rgb": image.mean(axis=(0, 1)) / 65535.0}
        for name, image in {"left": left, "right": right}.items()
    }
    result = method_delta(
        {"left": left, "right": right}, methods,
        "left", "right", (0, 0, 6, 6)
    )
    assert result["observed_sample_delta_rms"] == 0.0


def test_execution_requires_fixed_contract_and_no_fallback() -> None:
    log = (
        "MLRI X-Trans completed: method=mlri-xtrans-2pass passes=2 sigma=2,1 "
        "epsilon=0.01 core=384x384 halo=228 boundary=zero tiles=294 workers=2 "
        "workspace_per_worker_estimate=508032000 elapsed_us=123456\n"
    )
    result = parse_execution(log, "elapsed_seconds=4.5\nmax_rss_kb=1234\n")
    assert result["tiles"] == 294
    assert result["workspace_per_worker_estimate_bytes"] == 508032000
    with pytest.raises(ComparisonError):
        parse_execution(log + "falling back to 3-pass (Markesteijn)\n", "elapsed_seconds=4.5\nmax_rss_kb=1234\n")
    with pytest.raises(ComparisonError):
        parse_execution(log.replace("halo=228", "halo=222"), "elapsed_seconds=4.5\nmax_rss_kb=1234\n")


def test_execution_accepts_only_the_reviewed_detached_trace_prefix() -> None:
    trace = (
        'write(2, "MLRI X-Trans completed: method=mlri-xtrans-2pass passes=2 '
        'sigma=2,1 epsilon=0.01 core=384x384 halo=228 boundary=zero tiles=294 w", '
        '128) = -1 EPIPE (Broken pipe)\n'
    )
    result = parse_execution(trace, "elapsed_seconds=4602.98\nmax_rss_kb=1888272\n")
    assert result["completion_capture"].startswith("strace prefix")
    assert result["engine_elapsed_us"] is None
    assert result["workers"] == 2
    with pytest.raises(ComparisonError):
        parse_execution(trace.replace("tiles=294", "tiles=293"),
                        "elapsed_seconds=4602.98\nmax_rss_kb=1888272\n")


def test_corrected_execution_requires_its_own_complete_diagnostic() -> None:
    corrected = (
        "MLRI X-Trans completed: method=mlri-xtrans-2pass-corrected passes=2 "
        "sigma=2,1 epsilon=0.01 core=384x384 halo=228 boundary=zero tiles=294 "
        "workers=2 workspace_per_worker_estimate=508032000 elapsed_us=123456\n"
    )
    result = parse_execution(
        corrected,
        "elapsed_seconds=4.5\nmax_rss_kb=1234\n",
        "mlri-xtrans-2pass-corrected",
    )
    assert result["engine_elapsed_us"] == 123456
    with pytest.raises(ComparisonError):
        parse_execution(
            corrected.replace("mlri-xtrans-2pass-corrected", "mlri-xtrans-2pass"),
            "elapsed_seconds=4.5\nmax_rss_kb=1234\n",
            "mlri-xtrans-2pass-corrected",
        )


def test_corrected_pair_delta_uses_requested_methods() -> None:
    mark = np.full((48, 48, 3), 20000, dtype=np.uint16)
    faithful = mark + np.asarray((10, 20, 30), dtype=np.uint16)
    corrected = mark + np.asarray((30, 20, 10), dtype=np.uint16)
    images = {"corrected": corrected, "mlri": faithful, "markesteijn": mark}
    result = analyze_arrays(images, (0, 0, 48, 48))
    delta = method_delta(
        images, result["methods"], "corrected", "mlri", (0, 0, 48, 48)
    )
    assert delta["crop_mean_rgb_delta"] == pytest.approx(
        [20 / 65535, 0.0, -20 / 65535]
    )
    assert delta["common_luminance_delta"] == pytest.approx(0.0)


def test_corrected_pixel_difference_summary_is_channel_specific() -> None:
    faithful = np.full((6, 7, 3), 1000, dtype=np.uint16)
    corrected = faithful.copy()
    corrected[..., 0] += 1
    corrected[..., 1] += 2
    corrected[..., 2] += 3
    result = pixel_difference_summary(corrected, faithful, (1, 1, 4, 3))
    assert result["max_abs_code_rgb"] == [1, 2, 3]
    assert result["p99_abs_code_rgb"] == [1, 2, 3]
    assert result["rms_code_rgb"] == pytest.approx([1, 2, 3])
    with pytest.raises(ComparisonError):
        pixel_difference_summary(corrected, faithful, (6, 0, 2, 2))


def test_paper_core_execution_requires_controlled_contract() -> None:
    log2014 = (
        "MLRI X-Trans completed: method=mlri-xtrans-paper-core-2014 passes=1 "
        "sigma=2 epsilon=0.01 coefficient_average=uniform final=direct "
        "core=384x384 halo=228 boundary=zero tiles=294 workers=2 "
        "workspace_per_worker_estimate=508032000 elapsed_us=123456\n"
    )
    result = parse_paper_execution(
        log2014, "elapsed_seconds=2.5\nmax_rss_kb=1234\n", METHOD_2014
    )
    assert result["passes"] == 1
    assert result["coefficient_average"] == "uniform"
    assert result["final_reconstruction"] == "direct"

    log2016 = log2014.replace(METHOD_2014, METHOD_2016).replace(
        "coefficient_average=uniform", "coefficient_average=residual-weighted"
    )
    assert parse_paper_execution(
        log2016, "elapsed_seconds=2.5\nmax_rss_kb=1234\n", METHOD_2016
    )["coefficient_average"] == "residual-weighted"
    with pytest.raises(ComparisonError):
        parse_paper_execution(
            log2014.replace("final=direct", "final=sqrt-blend"),
            "elapsed_seconds=2.5\nmax_rss_kb=1234\n", METHOD_2014
        )
    with pytest.raises(ComparisonError):
        parse_paper_execution(
            log2014 + "falling back to 3-pass (Markesteijn)\n",
            "elapsed_seconds=2.5\nmax_rss_kb=1234\n", METHOD_2014
        )


def test_final_only_execution_requires_controlled_contract() -> None:
    log = (
        f"MLRI X-Trans completed: method={FINAL_ONLY_METHOD} passes=2 sigma=2,1 "
        "epsilon=0.01 blue_diagonal_guides=corrected final=direct core=384x384 "
        "halo=228 boundary=zero tiles=294 workers=2 "
        "workspace_per_worker_estimate=508032000 elapsed_us=123456\n"
    )
    result = parse_final_only_execution(
        log, "elapsed_seconds=2.5\nmax_rss_kb=1234\n"
    )
    assert result["passes"] == 2
    assert result["final_reconstruction"] == "direct"
    assert result["blue_diagonal_guides"] == "corrected"
    with pytest.raises(ComparisonError):
        parse_final_only_execution(
            log.replace("final=direct", "final=sqrt-blend"),
            "elapsed_seconds=2.5\nmax_rss_kb=1234\n"
        )
    with pytest.raises(ComparisonError):
        parse_final_only_execution(
            log + "falling back to 3-pass (Markesteijn)\n",
            "elapsed_seconds=2.5\nmax_rss_kb=1234\n"
        )


def test_final_only_analysis_uses_requested_pairs() -> None:
    mark = np.full((48, 48, 3), 20000, dtype=np.uint16)
    corrected = mark + np.asarray((10, 20, 30), dtype=np.uint16)
    final_only = mark + np.asarray((30, 20, 10), dtype=np.uint16)
    result = analyze_final_only_arrays(
        {
            "final_only": final_only,
            "corrected": corrected,
            "markesteijn": mark,
        },
        (0, 0, 48, 48),
    )
    assert result["final_only_minus_corrected_blend"][
        "crop_mean_rgb_delta"
    ] == pytest.approx([20 / 65535, 0.0, -20 / 65535])
    assert result["final_only_minus_markesteijn"][
        "crop_mean_rgb_delta"
    ] == pytest.approx([30 / 65535, 20 / 65535, 10 / 65535])


def test_final_only_blue_outlier_probe_uses_corrected_selection() -> None:
    mark = np.full((24, 24, 3), 20000, dtype=np.uint16)
    corrected = mark.copy()
    final_only = mark.copy()
    corrected[10, 9, 2] += 3000
    corrected[10, 12, 2] += 2000
    final_only[10, 9, 2] += 300
    final_only[10, 12, 2] += 200
    result = blue_outlier_probe(
        {
            "final_only": final_only,
            "corrected": corrected,
            "markesteijn": mark,
        },
        (7, 7, 9, 7),
        2,
    )
    assert result["coordinates"] == [{"x": 9, "y": 10}, {"x": 12, "y": 10}]
    assert result["methods"]["corrected"]["mean"] == pytest.approx(2500 / 65535)
    assert result["methods"]["final_only"]["mean"] == pytest.approx(250 / 65535)
    assert result["methods"]["markesteijn"]["maximum"] == 0.0
    with pytest.raises(ComparisonError):
        blue_outlier_probe(
            {
                "final_only": final_only,
                "corrected": corrected,
                "markesteijn": mark,
            },
            (0, 0, 5, 5),
        )


def test_local_blue_excess_rejects_incomplete_neighborhood() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint16)
    with pytest.raises(ComparisonError):
        local_blue_excess(image, 1, 4)


def test_paper_core_analysis_uses_markesteijn_phase_selection() -> None:
    mark = np.full((48, 48, 3), 20000, dtype=np.uint16)
    corrected = mark + np.asarray((10, 20, 30), dtype=np.uint16)
    paper2016 = mark + np.asarray((30, 20, 10), dtype=np.uint16)
    paper2014 = paper2016.copy()
    paper2014[..., 0] += 5
    result = analyze_paper_arrays(
        {
            "paper2014": paper2014,
            "paper2016": paper2016,
            "corrected": corrected,
            "markesteijn": mark,
        },
        (0, 0, 48, 48),
    )
    assert result["paper_2014_minus_2016"]["crop_mean_rgb_delta"] == pytest.approx(
        [5 / 65535, 0.0, 0.0]
    )
    assert result["paper_2016_minus_markesteijn"]["crop_mean_rgb_delta"] == pytest.approx(
        [30 / 65535, 20 / 65535, 10 / 65535]
    )

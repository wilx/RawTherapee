from __future__ import annotations

import numpy as np
import pytest

from tools.neural_demosaic.compare_xveon_outputs import analyze_arrays, parse_execution


def image(value=(0.3, 0.4, 0.5)):
    result = np.empty((96, 96, 3), np.uint16)
    result[...] = np.rint(np.asarray(value) * 65535).astype(np.uint16)
    return result


def analyze(xveon, mark):
    return analyze_arrays({"linear": mark, "gamma22": mark, "xveon": xveon, "markesteijn": mark}, (12, 18, 60, 60))


def test_identical_xveon_passes_objective_gate():
    reference = image()
    result = analyze(reference, reference)
    assert result["objective"]["pass"]
    assert result["observed_sample_delta_vs_markesteijn_rms"] == 0


def test_phase_and_color_defects_fail():
    reference = image()
    defective = image((0.29, 0.4, 0.51))
    defective[::3, ::3, 0] += 1000
    result = analyze(defective, reference)
    assert not result["objective"]["pass"]
    assert not result["objective"]["color"]["pass"]
    assert not result["objective"]["phase"]["pass"]


def test_xveon_diagnostic_is_authenticated():
    log = (
        "X-veon X-Trans completed: method=xveon-xtrans-onnx "
        "artifact=45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500 "
        "ort=1.27.0 provider=CPUExecutionProvider tile=288x288 overlap=48 stride=240 tiles=726 "
        "thread_policy=ort-default-intra,inter-1,sequential working_buffer_estimate=180000000 elapsed_us=123\n"
    )
    result = parse_execution(log, "elapsed_seconds=31.5\nmax_rss_kb=900000\n")
    assert result["tiles"] == 726
    assert result["elapsed_seconds"] == 31.5
    with pytest.raises(ValueError, match="fell back"):
        parse_execution(log + "falling back to 3-pass (Markesteijn)", "elapsed_seconds=1\nmax_rss_kb=2\n")

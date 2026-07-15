from __future__ import annotations

import builtins
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from tools.neural_demosaic.demosaicnet_reference import DemosaicNetXTransReference
from tools.neural_demosaic.raw_wrapper import (
    CANONICAL_XTRANS,
    CanonicalView,
    RawWrapperError,
    _reflect101_indices,
    find_canonical_transform,
    run_raw_wrapper_model,
    unique_xtrans_cfas,
)
from tools.neural_demosaic.raw_wrapper_corpus import (
    RAW_WRAPPER_MANIFEST_SHA256,
    RawWrapperCorpusError,
    build_raw_wrapper_corpus,
    load_raw_wrapper_corpus,
    publish_raw_wrapper_corpus,
)


CORPUS = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "demosaicnet-xtrans-raw-wrapper-v1"
)


def _constant_output_model() -> DemosaicNetXTransReference:
    model = DemosaicNetXTransReference()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fullres_processor.output.bias.copy_(
            torch.tensor((-0.25, 0.5, 1.25), dtype=torch.float32)
        )
    model.eval()
    return model


def _reviewed_rtnn() -> Path:
    value = os.environ.get("GHARBI_XTRANS_RTNN")
    if not value:
        pytest.skip("GHARBI_XTRANS_RTNN is not set")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"GHARBI_XTRANS_RTNN does not exist: {path}")
    return path


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_all_18_cfa_matrices_have_deterministic_coordinate_round_trips() -> None:
    cfas = unique_xtrans_cfas()
    assert len(cfas) == 18
    assert cfas[0] == CANONICAL_XTRANS
    for cfa in cfas:
        transform = find_canonical_transform(cfa)
        for width, height in ((1, 1), (17, 9), (9, 17), (173, 31), (31, 173)):
            view = CanonicalView.create(transform, width, height)
            points = set()
            for y in range(height):
                for x in range(width):
                    canonical = view.actual_to_canonical(x, y)
                    assert view.canonical_to_actual(*canonical) == (x, y)
                    points.add(canonical)
            assert len(points) == width * height


def test_reflect101_does_not_repeat_edges_and_handles_one_pixel() -> None:
    assert _reflect101_indices(4, 3).tolist() == [3, 2, 1, 0, 1, 2, 3, 2, 1, 0]
    assert _reflect101_indices(1, 12).tolist() == [0] * 25


@pytest.mark.parametrize("shape", ((1, 1), (7, 13), (13, 7), (31, 173), (173, 31)))
def test_wrapper_clamps_signed_output_without_sample_reinjection(shape) -> None:
    height, width = shape
    raw = np.full((height, width), 32768.0, dtype=np.float32)
    output = run_raw_wrapper_model(
        _constant_output_model(), raw, CANONICAL_XTRANS, "linear"
    )
    assert output.shape == (3, height, width)
    assert np.array_equal(output[0], np.zeros((height, width), dtype=np.float32))
    assert np.array_equal(output[1], np.full((height, width), 0.5, dtype=np.float32))
    assert np.array_equal(output[2], np.ones((height, width), dtype=np.float32))
    # Every observed sample was 0.5, but red and blue are still the network's
    # clamped result. This freezes the deliberate no-reinjection contract.


def test_gamma22_applies_only_the_documented_inverse_after_clamping() -> None:
    output = run_raw_wrapper_model(
        _constant_output_model(),
        np.array([[-100.0, 70000.0]], dtype=np.float32),
        CANONICAL_XTRANS,
        "gamma22",
    )
    assert np.array_equal(output[0], np.zeros((1, 2), dtype=np.float32))
    assert np.allclose(output[1], np.float32(0.5) ** np.float32(2.2), rtol=0, atol=2e-8)
    assert np.array_equal(output[2], np.ones((1, 2), dtype=np.float32))


def test_wrapper_rejects_unsupported_cfa_and_nonfinite_raw() -> None:
    wrong = tuple(tuple(0 for _ in range(6)) for _ in range(6))
    with pytest.raises(RawWrapperError, match="not a supported phase"):
        run_raw_wrapper_model(_constant_output_model(), np.zeros((2, 2), np.float32), wrong, "linear")
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(RawWrapperError, match="NaN or infinity"):
            run_raw_wrapper_model(
                _constant_output_model(),
                np.array([[value]], dtype=np.float32),
                CANONICAL_XTRANS,
                "linear",
            )


def test_committed_raw_wrapper_corpus_is_canonical_and_complete() -> None:
    loaded = load_raw_wrapper_corpus(CORPUS)
    assert hashlib.sha256(loaded.manifest_bytes).hexdigest() == RAW_WRAPPER_MANIFEST_SHA256
    assert loaded.manifest["summary"] == {
        "case_count": 7,
        "float_blob_bytes": 311376,
        "float_blob_count": 14,
    }
    assert len(loaded.blobs) == 14


def test_raw_wrapper_corpus_rejects_changes(tmp_path: Path) -> None:
    changed = tmp_path / "changed"
    shutil.copytree(CORPUS, changed)
    blob = next((changed / "cases").glob("*.rgb.f32le"))
    payload = bytearray(blob.read_bytes())
    payload[0] ^= 1
    blob.write_bytes(payload)
    with pytest.raises(RawWrapperCorpusError, match="digest differs"):
        load_raw_wrapper_corpus(changed)


def test_real_rtnn_regenerates_raw_wrapper_corpus_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rtnn = _reviewed_rtnn()

    def forbidden_torch_load(*args, **kwargs):
        raise AssertionError("raw-wrapper export must not call torch.load")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "demosaicnet" or name.startswith("demosaicnet."):
            raise AssertionError("raw-wrapper export imported upstream code")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    first = build_raw_wrapper_corpus(rtnn)
    second = build_raw_wrapper_corpus(rtnn)
    assert first.manifest_bytes == second.manifest_bytes
    assert first.blobs == second.blobs
    assert hashlib.sha256(first.manifest_bytes).hexdigest() == RAW_WRAPPER_MANIFEST_SHA256
    first_dir = publish_raw_wrapper_corpus(tmp_path / "first", first)
    second_dir = publish_raw_wrapper_corpus(tmp_path / "second", second)
    assert _files(first_dir) == _files(second_dir) == _files(CORPUS)

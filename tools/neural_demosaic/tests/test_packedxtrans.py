from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.neural_demosaic.convert_packedxtrans import build_model, convert
from tools.neural_demosaic.inspect_checkpoint import canonical_manifest_bytes
from tools.neural_demosaic.inspect_packedxtrans_checkpoint import main as inspect_main
from tools.neural_demosaic.packedxtrans import (
    PackedXTransReference,
    XTRANS_PATTERN,
    expected_fixed_buffers,
    load_packed_checkpoint,
    reference_from_checkpoint,
)
from tools.neural_demosaic.schema import PACKED_XTRANS_V1


EXPECTED_ONNX_SIZE = 1_673_648
EXPECTED_ONNX_SHA256 = "ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c"
EXPECTED_MANIFEST_SHA256 = "ebd978aef293d1cf35a5d15234ef185d0785ac222c9c10f6477903e74d4c338d"


def checkpoint_path() -> Path:
    value = os.environ.get("PACKED_XTRANS_CHECKPOINT")
    if not value:
        pytest.skip("PACKED_XTRANS_CHECKPOINT is not set")
    return Path(value)


def test_schema_and_fixed_buffers() -> None:
    assert len(PACKED_XTRANS_V1.tensors) == 39
    assert PACKED_XTRANS_V1.parameter_count == 158_865
    assert PACKED_XTRANS_V1.payload_bytes == 635_460
    buffers = expected_fixed_buffers()
    assert set(buffers) == {"baseline.masks", "baseline.kern_g", "baseline.kern_rb"}
    assert sum(tensor.numel() for tensor in buffers.values()) == 182
    assert all(tensor.dtype is torch.float32 and torch.isfinite(tensor).all() for tensor in buffers.values())


def test_reference_contract() -> None:
    torch.manual_seed(0x52544E4E)
    model = PackedXTransReference().eval()
    source = torch.rand(1, 1, 36, 42)
    with torch.inference_mode():
        output = model(source)
    assert output.shape == (1, 3, 36, 42)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="divisible by six"):
        model(torch.zeros(1, 1, 35, 36))


def test_real_checkpoint_identity_and_conversion() -> None:
    path = checkpoint_path()
    validated = load_packed_checkpoint(path)
    assert validated.manifest["summary"] == {
        "dtype": "float32",
        "parameter_count": 158_865,
        "payload_bytes": 635_460,
        "tensor_count": 39,
    }
    model_bytes, manifest = build_model(path)
    manifest_bytes = canonical_manifest_bytes(manifest)
    assert len(model_bytes) == EXPECTED_ONNX_SIZE
    assert hashlib.sha256(model_bytes).hexdigest() == EXPECTED_ONNX_SHA256
    assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST_SHA256


def test_real_conversion_is_reproducible_and_refuses_overwrite(tmp_path: Path) -> None:
    path = checkpoint_path()
    first = tmp_path / "first.onnx"
    second = tmp_path / "nested" / "second.onnx"
    second.parent.mkdir()
    convert(path, first)
    convert(path, second)
    assert first.read_bytes() == second.read_bytes()
    assert Path(str(first) + ".json").read_bytes() == Path(str(second) + ".json").read_bytes()
    with pytest.raises(Exception, match="already exists"):
        convert(path, first)
    convert(path, first, force=True)


def test_real_onnxruntime_matches_reference() -> None:
    ort = pytest.importorskip("onnxruntime")
    path = checkpoint_path()
    model_bytes, _ = build_model(path)
    rng = np.random.default_rng(0x52544E4E)
    rows, columns = np.indices((288, 288))
    channels = XTRANS_PATTERN[rows % 6, columns % 6]
    constant = np.choose(channels, (0.25, 0.5, 0.75)).astype(np.float32)[None, None]
    cases = {
        "zero": np.zeros((1, 1, 288, 288), np.float32),
        "constant": constant,
        "random": rng.random((1, 1, 288, 288), dtype=np.float32),
        "red-impulse": np.zeros((1, 1, 288, 288), np.float32),
        "green-impulse": np.zeros((1, 1, 288, 288), np.float32),
        "blue-impulse": np.zeros((1, 1, 288, 288), np.float32),
        "alternating-saturated": (((rows + columns) & 1).astype(np.float32))[None, None],
    }
    for channel, name, offset in ((0, "red-impulse", (144, 144)),
                                  (1, "green-impulse", (145, 144)),
                                  (2, "blue-impulse", (144, 145))):
        y, x = offset
        positions = np.argwhere(channels == channel)
        nearest = positions[np.argmin(np.sum((positions - (y, x)) ** 2, axis=1))]
        cases[name][0, 0, nearest[0], nearest[1]] = 1
    reference = reference_from_checkpoint(path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            for name, source in cases.items():
                expected = reference(torch.from_numpy(source)).numpy()
                actual = session.run(["output"], {"input": source})[0]
                assert np.isfinite(actual).all(), name
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-6, err_msg=name)
    finally:
        torch.set_num_threads(old_threads)


def test_manifest_contains_no_local_paths_or_timestamps() -> None:
    path = checkpoint_path()
    _, manifest = build_model(path)
    payload = canonical_manifest_bytes(manifest)
    assert str(path).encode() not in payload
    assert b"timestamp" not in payload.lower()
    assert payload == canonical_manifest_bytes(json.loads(payload))


def test_inspector_writes_the_reviewed_manifest(tmp_path: Path) -> None:
    path = checkpoint_path()
    output = tmp_path / "inspection.json"
    assert inspect_main([str(path), "--output", str(output)]) == 0
    manifest = json.loads(output.read_bytes())
    assert manifest["summary"]["tensor_count"] == 39
    assert manifest["summary"]["parameter_count"] == 158_865
    assert inspect_main([str(path), "--output", str(output)]) == 2

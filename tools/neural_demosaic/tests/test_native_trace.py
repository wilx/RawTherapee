from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

import pytest

from tools.neural_demosaic.native_trace import (
    NATIVE_TRACE_MANIFEST_SHA256,
    NativeTraceError,
    build_native_trace,
    load_native_trace,
    publish_native_trace,
)


ROOT = Path(__file__).parents[1] / "native_trace"


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_committed_native_trace_is_canonical_and_complete() -> None:
    loaded = load_native_trace(ROOT)
    assert hashlib.sha256(loaded.manifest_bytes).hexdigest() == NATIVE_TRACE_MANIFEST_SHA256
    assert loaded.manifest["summary"] == {
        "activation_count": 13,
        "sample_bytes": 16700,
        "sample_count": 4175,
    }
    assert len(loaded.blobs) == 13
    assert loaded.manifest["phase7_manifest_sha256"] == (
        "ed4b6ff5544ef361d3613fdf354544acd89db56059de249468ac416c238f3b92"
    )


def test_native_trace_rejects_manifest_and_payload_corruption(tmp_path: Path) -> None:
    changed_manifest = tmp_path / "manifest"
    shutil.copytree(ROOT, changed_manifest)
    manifest = changed_manifest / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(NativeTraceError, match="manifest SHA-256"):
        load_native_trace(changed_manifest)

    changed_payload = tmp_path / "payload"
    shutil.copytree(ROOT, changed_payload)
    payload = next((changed_payload / "activations").glob("*.f32le"))
    data = bytearray(payload.read_bytes())
    data[0] ^= 1
    payload.write_bytes(data)
    with pytest.raises(NativeTraceError, match="payload differs"):
        load_native_trace(changed_payload)


def test_native_trace_publication_refuses_existing_output(tmp_path: Path) -> None:
    loaded = load_native_trace(ROOT)
    output = tmp_path / "trace"
    output.mkdir()
    from tools.neural_demosaic.native_trace import NativeTrace

    trace = NativeTrace(loaded.manifest, loaded.manifest_bytes, loaded.blobs)
    with pytest.raises(NativeTraceError, match="output already exists"):
        publish_native_trace(output, trace)


def test_native_trace_regeneration_is_byte_identical(tmp_path: Path) -> None:
    rtnn = os.environ.get("GHARBI_XTRANS_RTNN")
    if not rtnn:
        pytest.skip("GHARBI_XTRANS_RTNN is not set")
    first = build_native_trace(rtnn)
    second = build_native_trace(rtnn)
    assert first.manifest_bytes == second.manifest_bytes
    assert dict(first.blobs) == dict(second.blobs)
    first_root = publish_native_trace(tmp_path / "first", first)
    second_root = publish_native_trace(tmp_path / "second", second)
    assert tree_bytes(first_root) == tree_bytes(second_root) == tree_bytes(ROOT)

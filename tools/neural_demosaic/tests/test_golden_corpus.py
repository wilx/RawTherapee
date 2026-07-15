from __future__ import annotations

import builtins
import hashlib
import os
from pathlib import Path
import shutil

import pytest
import torch

from tools.neural_demosaic.golden_corpus import (
    GOLDEN_MANIFEST_NAME,
    GOLDEN_MANIFEST_SHA256,
    GoldenCorpus,
    GoldenCorpusError,
    build_golden_corpus,
    load_golden_corpus,
    publish_golden_corpus,
)


CORPUS = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "demosaicnet-xtrans-v1"
)


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


def test_committed_golden_corpus_is_canonical_and_complete() -> None:
    loaded = load_golden_corpus(CORPUS)

    assert hashlib.sha256(loaded.manifest_bytes).hexdigest() == GOLDEN_MANIFEST_SHA256
    assert len(loaded.manifest["cases"]) == 7
    assert loaded.manifest["summary"] == {
        "case_count": 7,
        "float_blob_bytes": 114600,
        "float_blob_count": 14,
        "trace_case": "random-37x38",
    }
    assert len(loaded.blobs) == 16

    for case in loaded.manifest["cases"]:
        for role in ("input", "output"):
            blob = case[role]
            payload = loaded.blobs[blob["file"]]
            assert len(payload) == blob["byte_length"]
            assert hashlib.sha256(payload).hexdigest() == blob["sha256"]


def test_committed_trace_has_all_native_debugging_checkpoints() -> None:
    loaded = load_golden_corpus(CORPUS)
    random_case = next(
        case for case in loaded.manifest["cases"] if case["id"] == "random-37x38"
    )
    names = [entry["name"] for entry in random_case["intermediate_activations"]]
    assert names == [
        *(f"main_processor.relu{index}" for index in range(1, 12)),
        "fullres_processor.input_concat",
        "fullres_processor.post_relu",
    ]


def test_manifest_or_blob_changes_are_rejected(tmp_path: Path) -> None:
    changed_manifest = tmp_path / "changed-manifest"
    shutil.copytree(CORPUS, changed_manifest)
    manifest_path = changed_manifest / GOLDEN_MANIFEST_NAME
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(GoldenCorpusError, match="manifest SHA-256"):
        load_golden_corpus(changed_manifest)

    changed_blob = tmp_path / "changed-blob"
    shutil.copytree(CORPUS, changed_blob)
    blob_path = changed_blob / "cases" / "random-37x38.output.f32le"
    payload = bytearray(blob_path.read_bytes())
    payload[0] ^= 1
    blob_path.write_bytes(payload)
    with pytest.raises(GoldenCorpusError, match="digest differs"):
        load_golden_corpus(changed_blob)

    extra_file = tmp_path / "extra-file"
    shutil.copytree(CORPUS, extra_file)
    (extra_file / "host-details.txt").write_text("must not be present\n")
    with pytest.raises(GoldenCorpusError, match="golden files differ"):
        load_golden_corpus(extra_file)


def test_publisher_refuses_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    empty = GoldenCorpus({}, b"{}\n", {})
    with pytest.raises(GoldenCorpusError, match="output already exists"):
        publish_golden_corpus(output, empty)


def test_real_rtnn_regenerates_the_committed_corpus_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rtnn = _reviewed_rtnn()

    def forbidden_torch_load(*args, **kwargs):
        raise AssertionError("golden export must not call torch.load")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "demosaicnet" or name.startswith("demosaicnet."):
            raise AssertionError("golden export imported upstream executable code")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    first = build_golden_corpus(rtnn)
    second = build_golden_corpus(rtnn)
    assert first.manifest_bytes == second.manifest_bytes
    assert first.blobs == second.blobs
    assert hashlib.sha256(first.manifest_bytes).hexdigest() == GOLDEN_MANIFEST_SHA256

    first_dir = publish_golden_corpus(tmp_path / "first", first)
    second_dir = publish_golden_corpus(tmp_path / "second", second)
    assert _files(first_dir) == _files(second_dir) == _files(CORPUS)
    assert load_golden_corpus(first_dir).manifest == load_golden_corpus(CORPUS).manifest

"""Deterministic Phase 9 scalar-mosaic and normalized-RGB fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .golden_corpus import GOLDEN_MANIFEST_SHA256
from .native_trace import NATIVE_TRACE_MANIFEST_SHA256
from .raw_wrapper import (
    CfaTransform,
    cfa_from_transform,
    run_raw_wrapper_loaded,
)
from .rtnn_reader import read_rtnn
from .semantic_schema import canonical_json_bytes, parse_canonical_json


RAW_WRAPPER_CORPUS_FORMAT = "rawtherapee-xtrans-raw-wrapper-corpus-v1"
RAW_WRAPPER_MANIFEST_NAME = "manifest.json"
RAW_WRAPPER_MANIFEST_SHA256 = "bd415eb33ecb10c01c7ef127039e6ef69267d9398d01edbb7a339a661790b526"
EXPECTED_TORCH_VERSION = "2.12.1+cpu"


class RawWrapperCorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawWrapperCorpus:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    blobs: Mapping[str, bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _f32le(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")


def _blob(filename: str, array: np.ndarray, payload: bytes) -> dict[str, Any]:
    shape = [int(value) for value in array.shape]
    if len(payload) != math.prod(shape) * 4:
        raise RawWrapperCorpusError(f"{filename} byte count differs")
    return {
        "byte_length": len(payload),
        "element_count": math.prod(shape),
        "file": filename,
        "sha256": _sha256(payload),
        "shape": shape,
    }


def _raw_pattern(height: int, width: int, seed: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint64)[:, None]
    columns = np.arange(width, dtype=np.uint64)[None, :]
    bits = (rows * 1103515245 + columns * 2654435761 + seed) & 0xFFFF
    raw = bits.astype(np.float32)
    if height and width:
        raw[0, 0] = -1024.0
        raw[-1, -1] = 70000.0
    return raw


def _case_specs() -> tuple[tuple[str, int, int, CfaTransform, str, int], ...]:
    return (
        ("canonical-linear-43x41", 41, 43, CfaTransform(1, 0, 0, 1, 0, 0), "linear", 1),
        ("canonical-gamma22-43x41", 41, 43, CfaTransform(1, 0, 0, 1, 0, 0), "gamma22", 2),
        ("translated-linear-47x37", 37, 47, CfaTransform(1, 0, 0, 1, 2, 5), "linear", 3),
        ("rotated-gamma22-45x39", 39, 45, CfaTransform(0, -1, 1, 0, 1, 4), "gamma22", 4),
        ("reflected-linear-49x35", 35, 49, CfaTransform(-1, 0, 0, 1, 3, 2), "linear", 5),
        ("horizontal-seam-linear-173x31", 31, 173, CfaTransform(1, 0, 0, 1, 4, 1), "linear", 6),
        ("vertical-seam-gamma22-31x173", 173, 31, CfaTransform(0, 1, 1, 0, 5, 3), "gamma22", 7),
    )


def build_raw_wrapper_corpus(path: str | Path) -> RawWrapperCorpus:
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RawWrapperCorpusError(
            f"PyTorch {torch.__version__} is installed; expected {EXPECTED_TORCH_VERSION}"
        )
    loaded = read_rtnn(path)
    blobs: dict[str, bytes] = {}
    cases = []
    for case_id, height, width, transform, domain, seed in _case_specs():
        cfa = cfa_from_transform(transform)
        raw = _raw_pattern(height, width, seed)
        output = run_raw_wrapper_loaded(loaded, raw, cfa, domain)  # normalized CHW
        input_name = f"cases/{case_id}.raw.f32le"
        output_name = f"cases/{case_id}.rgb.f32le"
        input_payload = _f32le(raw)
        output_payload = _f32le(output)
        blobs[input_name] = input_payload
        blobs[output_name] = output_payload
        cases.append(
            {
                "cfa": [list(row) for row in cfa],
                "domain": domain,
                "id": case_id,
                "input": _blob(input_name, raw, input_payload),
                "output": _blob(output_name, output, output_payload),
            }
        )
    manifest = {
        "cases": cases,
        "execution": {
            "boundary": "reflect-without-edge-repetition",
            "byte_order": "little-endian",
            "input_contract": "scaled-scalar-rawData",
            "input_scale": 65535,
            "output_contract": "full-size-normalized-rgb",
            "output_layout": "CHW",
            "receptive_halo": 12,
            "scalar_type": "float32",
            "torch_version": EXPECTED_TORCH_VERSION,
        },
        "format": RAW_WRAPPER_CORPUS_FORMAT,
        "model": {
            "architecture_id": loaded.architecture_id,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "model_revision": loaded.model_revision,
            "rtnn_sha256": loaded.artifact_sha256,
            "semantic_schema_sha256": loaded.semantic_schema_sha256,
        },
        "phase_bindings": {
            "phase7_golden_manifest_sha256": GOLDEN_MANIFEST_SHA256,
            "phase8_native_trace_manifest_sha256": NATIVE_TRACE_MANIFEST_SHA256,
        },
        "summary": {
            "case_count": len(cases),
            "float_blob_bytes": sum(map(len, blobs.values())),
            "float_blob_count": len(blobs),
        },
    }
    return RawWrapperCorpus(manifest, canonical_json_bytes(manifest), blobs)


def publish_raw_wrapper_corpus(output: str | Path, corpus: RawWrapperCorpus) -> Path:
    output = Path(output)
    if output.exists():
        raise RawWrapperCorpusError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise RawWrapperCorpusError(f"output parent does not exist: {output.parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        for filename, payload in sorted(corpus.blobs.items()):
            destination = temporary / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (temporary / RAW_WRAPPER_MANIFEST_NAME).write_bytes(corpus.manifest_bytes)
        load_raw_wrapper_corpus(
            temporary,
            expected_manifest_sha256=_sha256(corpus.manifest_bytes),
        )
        os.replace(temporary, output)
        published = True
        return output
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def load_raw_wrapper_corpus(
    root: str | Path,
    *,
    expected_manifest_sha256: str | None = RAW_WRAPPER_MANIFEST_SHA256 or None,
) -> RawWrapperCorpus:
    root = Path(root)
    manifest_bytes = (root / RAW_WRAPPER_MANIFEST_NAME).read_bytes()
    if expected_manifest_sha256 and _sha256(manifest_bytes) != expected_manifest_sha256:
        raise RawWrapperCorpusError("raw-wrapper manifest SHA-256 differs")
    manifest = parse_canonical_json(manifest_bytes, "raw-wrapper manifest")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise RawWrapperCorpusError("raw-wrapper manifest is not canonical JSON")
    if manifest.get("format") != RAW_WRAPPER_CORPUS_FORMAT:
        raise RawWrapperCorpusError("raw-wrapper corpus format differs")
    filenames = {
        blob["file"]
        for case in manifest.get("cases", ())
        for blob in (case["input"], case["output"])
    }
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != RAW_WRAPPER_MANIFEST_NAME
    }
    if actual != filenames:
        raise RawWrapperCorpusError("raw-wrapper corpus files differ")
    blobs = {}
    for case in manifest["cases"]:
        for role in ("input", "output"):
            document = case[role]
            payload = (root / document["file"]).read_bytes()
            if len(payload) != document["byte_length"]:
                raise RawWrapperCorpusError("raw-wrapper blob byte length differs")
            if _sha256(payload) != document["sha256"]:
                raise RawWrapperCorpusError("raw-wrapper blob digest differs")
            blobs[document["file"]] = payload
    return RawWrapperCorpus(manifest, manifest_bytes, blobs)

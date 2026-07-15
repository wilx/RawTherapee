"""Compact numeric activation samples for portable native inference parity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from .demosaicnet_reference import (
    deterministic_reference_execution,
    reference_forward_with_activations,
    reference_from_loaded_rtnn,
    reference_inputs,
)
from .golden_corpus import (
    EXPECTED_TORCH_VERSION,
    GOLDEN_MANIFEST_SHA256,
    TRACE_CASE_ID,
    LoadedGoldenCorpus,
    load_golden_corpus,
)
from .rtnn_reader import read_rtnn
from .semantic_schema import SemanticSchemaError, canonical_json_bytes, parse_canonical_json


NATIVE_TRACE_FORMAT = "rawtherapee-neural-native-parity-trace-v1"
NATIVE_TRACE_MANIFEST_NAME = "manifest.json"
NATIVE_TRACE_MANIFEST_SHA256 = "1415ffa3c8c072740f39b2c084525483910cb09e71dcfba22061901f0cf1bf66"
MAX_MANIFEST_BYTES = 128 * 1024
DEFAULT_GOLDEN_ROOT = (
    Path(__file__).parent / "golden" / "demosaicnet-xtrans-v1"
)


class NativeTraceError(RuntimeError):
    """The native parity trace could not be generated or authenticated."""


@dataclass(frozen=True)
class NativeTrace:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    blobs: Mapping[str, bytes]


@dataclass(frozen=True)
class LoadedNativeTrace:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    blobs: Mapping[str, bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positions(height: int, width: int) -> list[list[int]]:
    return [
        [0, 0],
        [0, width - 1],
        [height // 2, width // 2],
        [height - 1, 0],
        [height - 1, width - 1],
    ]


def _samples(tensor: torch.Tensor, positions: list[list[int]]) -> bytes:
    array = tensor.detach().contiguous().cpu().numpy()
    sampled = np.stack(
        [array[0, :, row, column] for row, column in positions],
        axis=0,
    ).astype("<f4", copy=False)
    return sampled.tobytes(order="C")


def _golden_trace_case(golden: LoadedGoldenCorpus) -> dict[str, Any]:
    return next(case for case in golden.manifest["cases"] if case["id"] == TRACE_CASE_ID)


def build_native_trace(
    rtnn_path: str | Path,
    *,
    golden_root: str | Path = DEFAULT_GOLDEN_ROOT,
) -> NativeTrace:
    """Generate numeric samples after authenticating RTNN and Phase 7."""

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise NativeTraceError(
            f"PyTorch {torch.__version__} is installed; expected {EXPECTED_TORCH_VERSION}"
        )
    golden = load_golden_corpus(golden_root)
    golden_case = _golden_trace_case(golden)
    loaded_rtnn = read_rtnn(rtnn_path)
    model = reference_from_loaded_rtnn(loaded_rtnn)
    input_tensor = dict(reference_inputs())[TRACE_CASE_ID]
    input_payload = (
        input_tensor.detach().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
    )
    if _sha256(input_payload) != golden_case["input"]["sha256"]:
        raise NativeTraceError("reference trace input differs from Phase 7")

    with deterministic_reference_execution(), torch.inference_mode():
        _, activations = reference_forward_with_activations(model, input_tensor)

    expected_activations = golden_case["intermediate_activations"]
    if len(activations) != len(expected_activations):
        raise NativeTraceError("activation count differs from Phase 7")

    blobs: dict[str, bytes] = {}
    documents = []
    for activation, expected in zip(activations, expected_activations, strict=True):
        tensor = activation.tensor
        shape = [int(value) for value in tensor.shape]
        full_payload = (
            tensor.detach().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
        )
        if (
            activation.name != expected["name"]
            or shape != expected["shape"]
            or _sha256(full_payload) != expected["sha256"]
        ):
            raise NativeTraceError(f"{activation.name} differs from Phase 7")
        positions = _positions(shape[-2], shape[-1])
        payload = _samples(tensor, positions)
        filename = f"activations/{activation.name}.samples.f32le"
        blobs[filename] = payload
        documents.append(
            {
                "byte_length": len(payload),
                "channels": shape[1],
                "file": filename,
                "full_activation_sha256": expected["sha256"],
                "name": activation.name,
                "positions": positions,
                "sample_count": len(payload) // 4,
                "sample_sha256": _sha256(payload),
                "shape": shape,
            }
        )

    manifest = {
        "activations": documents,
        "case": {
            "id": TRACE_CASE_ID,
            "input_sha256": golden_case["input"]["sha256"],
        },
        "format": NATIVE_TRACE_FORMAT,
        "model": golden.manifest["model"],
        "phase7_manifest_sha256": GOLDEN_MANIFEST_SHA256,
        "sampling": {
            "channel_order": "all-channels-ascending",
            "position_order": "top-left,top-right,center,bottom-left,bottom-right",
            "scalar_type": "float32",
            "storage": "little-endian-position-major-channel-minor",
        },
        "summary": {
            "activation_count": len(documents),
            "sample_bytes": sum(len(payload) for payload in blobs.values()),
            "sample_count": sum(len(payload) // 4 for payload in blobs.values()),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    return NativeTrace(manifest, manifest_bytes, blobs)


def publish_native_trace(output: str | Path, trace: NativeTrace) -> Path:
    output_path = Path(output)
    if output_path.exists():
        raise NativeTraceError(f"output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise NativeTraceError(f"output parent does not exist: {output_path.parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    published = False
    try:
        for filename, payload in sorted(trace.blobs.items()):
            destination = temporary / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (temporary / NATIVE_TRACE_MANIFEST_NAME).write_bytes(trace.manifest_bytes)
        load_native_trace(
            temporary,
            expected_manifest_sha256=_sha256(trace.manifest_bytes),
        )
        os.replace(temporary, output_path)
        published = True
        return output_path
    except OSError as error:
        raise NativeTraceError(f"cannot publish native trace: {error}") from error
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _require_fields(document: Any, fields: set[str], label: str) -> None:
    if type(document) is not dict:
        raise NativeTraceError(f"{label} must be an object")
    actual = set(document)
    if actual != fields:
        raise NativeTraceError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def load_native_trace(
    root: str | Path,
    *,
    expected_manifest_sha256: str = NATIVE_TRACE_MANIFEST_SHA256,
    golden_root: str | Path = DEFAULT_GOLDEN_ROOT,
) -> LoadedNativeTrace:
    root_path = Path(root)
    try:
        manifest_bytes = (root_path / NATIVE_TRACE_MANIFEST_NAME).read_bytes()
    except OSError as error:
        raise NativeTraceError(f"cannot read native trace manifest: {error}") from error
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise NativeTraceError("native trace manifest exceeds its size limit")
    digest = _sha256(manifest_bytes)
    if not expected_manifest_sha256:
        raise NativeTraceError("native trace manifest digest is not pinned")
    if digest != expected_manifest_sha256:
        raise NativeTraceError(
            f"native trace manifest SHA-256 is {digest}; expected {expected_manifest_sha256}"
        )
    try:
        manifest = parse_canonical_json(manifest_bytes, "native trace manifest")
    except SemanticSchemaError as error:
        raise NativeTraceError(str(error)) from error
    _require_fields(
        manifest,
        {
            "activations",
            "case",
            "format",
            "model",
            "phase7_manifest_sha256",
            "sampling",
            "summary",
        },
        "native trace manifest",
    )
    if manifest["format"] != NATIVE_TRACE_FORMAT:
        raise NativeTraceError("native trace format differs")
    if manifest["phase7_manifest_sha256"] != GOLDEN_MANIFEST_SHA256:
        raise NativeTraceError("native trace Phase 7 binding differs")

    golden = load_golden_corpus(golden_root)
    golden_case = _golden_trace_case(golden)
    if manifest["model"] != golden.manifest["model"]:
        raise NativeTraceError("native trace model binding differs")
    expected_case = {
        "id": TRACE_CASE_ID,
        "input_sha256": golden_case["input"]["sha256"],
    }
    if manifest["case"] != expected_case:
        raise NativeTraceError("native trace case binding differs")
    expected_sampling = {
        "channel_order": "all-channels-ascending",
        "position_order": "top-left,top-right,center,bottom-left,bottom-right",
        "scalar_type": "float32",
        "storage": "little-endian-position-major-channel-minor",
    }
    if manifest["sampling"] != expected_sampling:
        raise NativeTraceError("native trace sampling contract differs")

    activations = manifest["activations"]
    expected_activations = golden_case["intermediate_activations"]
    if not isinstance(activations, list) or len(activations) != len(expected_activations):
        raise NativeTraceError("native trace activation count differs")
    blobs: dict[str, bytes] = {}
    for index, (document, expected) in enumerate(
        zip(activations, expected_activations, strict=True)
    ):
        _require_fields(
            document,
            {
                "byte_length",
                "channels",
                "file",
                "full_activation_sha256",
                "name",
                "positions",
                "sample_count",
                "sample_sha256",
                "shape",
            },
            f"native trace activation {index}",
        )
        shape = expected["shape"]
        channels = shape[1]
        positions = _positions(shape[-2], shape[-1])
        filename = f"activations/{expected['name']}.samples.f32le"
        sample_count = channels * len(positions)
        if document != {
            "byte_length": sample_count * 4,
            "channels": channels,
            "file": filename,
            "full_activation_sha256": expected["sha256"],
            "name": expected["name"],
            "positions": positions,
            "sample_count": sample_count,
            "sample_sha256": document["sample_sha256"],
            "shape": shape,
        }:
            raise NativeTraceError(f"native trace activation {index} metadata differs")
        sample_digest = document["sample_sha256"]
        if not isinstance(sample_digest, str) or len(sample_digest) != 64:
            raise NativeTraceError(f"native trace activation {index} digest differs")
        try:
            bytes.fromhex(sample_digest)
        except ValueError as error:
            raise NativeTraceError(
                f"native trace activation {index} digest differs"
            ) from error
        try:
            payload = (root_path / filename).read_bytes()
        except OSError as error:
            raise NativeTraceError(f"cannot read {filename}: {error}") from error
        if len(payload) != sample_count * 4 or _sha256(payload) != sample_digest:
            raise NativeTraceError(f"native trace activation {index} payload differs")
        values = np.frombuffer(payload, dtype="<f4")
        if not bool(np.isfinite(values).all()):
            raise NativeTraceError(f"native trace activation {index} is non-finite")
        blobs[filename] = payload

    expected_summary = {
        "activation_count": len(activations),
        "sample_bytes": sum(len(payload) for payload in blobs.values()),
        "sample_count": sum(len(payload) // 4 for payload in blobs.values()),
    }
    if manifest["summary"] != expected_summary:
        raise NativeTraceError("native trace summary differs")
    expected_files = {NATIVE_TRACE_MANIFEST_NAME, *blobs}
    actual_files = {
        str(path.relative_to(root_path))
        for path in root_path.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise NativeTraceError("native trace file set differs")
    return LoadedNativeTrace(manifest, manifest_bytes, blobs)

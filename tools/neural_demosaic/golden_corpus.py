"""Deterministic golden inference corpus for the reviewed X-Trans model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from .demosaicnet_reference import (
    CANONICAL_XTRANS,
    deterministic_reference_execution,
    reference_forward_with_activations,
    reference_from_loaded_rtnn,
    reference_inputs,
)
from .rtnn_reader import LoadedRTNN, read_rtnn, reviewed_binding
from .semantic_schema import (
    SemanticSchemaError,
    canonical_json_bytes,
    parse_canonical_json,
)


GOLDEN_CORPUS_FORMAT = "rawtherapee-neural-golden-corpus-v1"
GOLDEN_MANIFEST_NAME = "manifest.json"
GOLDEN_MANIFEST_SHA256 = "ed4b6ff5544ef361d3613fdf354544acd89db56059de249468ac416c238f3b92"
EXPECTED_TORCH_VERSION = "2.12.1+cpu"
TRACE_CASE_ID = "random-37x38"
MAX_MANIFEST_BYTES = 256 * 1024

UPSTREAM_LICENSE_TEXT = """MIT License

Deep Joint Demosaicking and Denoising
Siggraph Asia 2016
Michael Gharbi, Gaurav Chaurasia, Sylvain Paris, Fredo Durand

Copyright (c) 2016 Michael Gharbi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

ATTRIBUTION_TEXT = """# DemosaicNet X-Trans golden corpus

These deterministic inference inputs and outputs were generated from the
reviewed Gharbi DemosaicNet X-Trans RTNN artifact. They are small numerical
test fixtures for implementing RawTherapee's native inference engine; they do
not contain the checkpoint or RTNN model weights.

Upstream project: https://github.com/mgharbi/demosaicnet
Upstream revision: 959e9d1630976b421d5af5e35b2e2a01f5630e5c
Paper: Deep Joint Demosaicking and Denoising, SIGGRAPH Asia 2016
License: MIT; see `UPSTREAM-LICENSE.txt`.

The tensors are exact network inputs and outputs. No raw normalization, gamma
wrapper, CFA orientation transform, clipping, sample reinjection, or
postprocessing is represented by this corpus.
"""

EXPECTED_CASE_SHAPES = {
    "zero-even-32x32": (1, 3, 32, 32),
    "constant-odd-31x35": (1, 3, 31, 35),
    "random-37x38": (1, 3, 37, 38),
    "impulse-red-36x36": (1, 3, 36, 36),
    "impulse-green-36x36": (1, 3, 36, 36),
    "impulse-blue-36x36": (1, 3, 36, 36),
    "alternating-saturated-35x36": (1, 3, 35, 36),
}


class GoldenCorpusError(RuntimeError):
    """The golden corpus could not be generated or authenticated."""


@dataclass(frozen=True)
class GoldenCorpus:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    blobs: Mapping[str, bytes]


@dataclass(frozen=True)
class LoadedGoldenCorpus:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    blobs: Mapping[str, bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = (
        tensor.detach()
        .contiguous()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
    )
    return array.tobytes(order="C")


def _blob_document(filename: str, tensor: torch.Tensor, payload: bytes) -> dict[str, Any]:
    shape = [int(value) for value in tensor.shape]
    element_count = math.prod(shape)
    if len(payload) != element_count * 4:
        raise GoldenCorpusError(f"{filename} byte length differs from its shape")
    return {
        "byte_length": len(payload),
        "element_count": element_count,
        "file": filename,
        "sha256": _sha256(payload),
        "shape": shape,
    }


def _activation_document(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    payload = _tensor_bytes(tensor)
    return {
        "byte_length": len(payload),
        "element_count": tensor.numel(),
        "name": name,
        "sha256": _sha256(payload),
        "shape": [int(value) for value in tensor.shape],
    }


def _model_document(model: LoadedRTNN) -> dict[str, Any]:
    return {
        "architecture_id": model.architecture_id,
        "architecture_symbol": model.architecture_symbol,
        "checkpoint_sha256": model.checkpoint_sha256,
        "id": model.model_id,
        "revision": model.model_revision,
        "rtnn_sha256": model.artifact_sha256,
        "semantic_schema_sha256": model.semantic_schema_sha256,
    }


def build_golden_corpus(path: str | Path) -> GoldenCorpus:
    """Authenticate RTNN and generate the complete corpus without publishing it."""

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise GoldenCorpusError(
            f"PyTorch {torch.__version__} is installed; expected {EXPECTED_TORCH_VERSION}"
        )

    loaded_rtnn = read_rtnn(path)
    model = reference_from_loaded_rtnn(loaded_rtnn)
    blobs: dict[str, bytes] = {}
    cases = []

    with deterministic_reference_execution(), torch.inference_mode():
        for case_id, input_tensor in reference_inputs():
            input_filename = f"cases/{case_id}.input.f32le"
            output_filename = f"cases/{case_id}.output.f32le"
            input_payload = _tensor_bytes(input_tensor)

            if case_id == TRACE_CASE_ID:
                output_tensor, activations = reference_forward_with_activations(
                    model,
                    input_tensor,
                )
                ordinary_output = model(input_tensor)
                if not torch.equal(output_tensor, ordinary_output):
                    raise GoldenCorpusError(
                        "traced execution differs from the ordinary reference graph"
                    )
                intermediate = [
                    _activation_document(activation.name, activation.tensor)
                    for activation in activations
                ]
            else:
                output_tensor = model(input_tensor)
                intermediate = []

            if output_tensor.shape[-2:] != (
                input_tensor.shape[-2] - 24,
                input_tensor.shape[-1] - 24,
            ):
                raise GoldenCorpusError(f"{case_id} output does not shrink by 24")
            if not bool(torch.isfinite(output_tensor).all()):
                raise GoldenCorpusError(f"{case_id} output contains a non-finite value")

            output_payload = _tensor_bytes(output_tensor)
            blobs[input_filename] = input_payload
            blobs[output_filename] = output_payload
            cases.append(
                {
                    "id": case_id,
                    "input": _blob_document(
                        input_filename,
                        input_tensor,
                        input_payload,
                    ),
                    "intermediate_activations": intermediate,
                    "output": _blob_document(
                        output_filename,
                        output_tensor,
                        output_payload,
                    ),
                }
            )

    license_bytes = UPSTREAM_LICENSE_TEXT.encode("utf-8")
    attribution_bytes = ATTRIBUTION_TEXT.encode("utf-8")
    blobs["UPSTREAM-LICENSE.txt"] = license_bytes
    blobs["ATTRIBUTION.md"] = attribution_bytes
    manifest = {
        "cases": cases,
        "execution": {
            "byte_order": "little-endian",
            "deterministic_algorithms": True,
            "device": "cpu",
            "input_contract": "canonical-sparse-xtrans-network-input",
            "mode": "inference",
            "onednn_enabled": False,
            "postprocessing": "none",
            "preprocessing": "none",
            "scalar_type": "float32",
            "tensor_layout": "NCHW",
            "threads": 1,
            "torch_version": EXPECTED_TORCH_VERSION,
        },
        "format": GOLDEN_CORPUS_FORMAT,
        "model": _model_document(loaded_rtnn),
        "provenance": {
            "attribution_file": "ATTRIBUTION.md",
            "attribution_sha256": _sha256(attribution_bytes),
            "license": "MIT",
            "license_file": "UPSTREAM-LICENSE.txt",
            "license_sha256": _sha256(license_bytes),
            "upstream_repository": "https://github.com/mgharbi/demosaicnet",
            "upstream_revision": "959e9d1630976b421d5af5e35b2e2a01f5630e5c",
        },
        "summary": {
            "case_count": len(cases),
            "float_blob_bytes": sum(
                len(payload)
                for filename, payload in blobs.items()
                if filename.endswith(".f32le")
            ),
            "float_blob_count": sum(
                filename.endswith(".f32le") for filename in blobs
            ),
            "trace_case": TRACE_CASE_ID,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    return GoldenCorpus(manifest, manifest_bytes, blobs)


def publish_golden_corpus(
    output: str | Path,
    corpus: GoldenCorpus,
) -> Path:
    """Publish a validated corpus to a new directory as one completion unit."""

    output_path = Path(output)
    if output_path.exists():
        raise GoldenCorpusError(f"output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise GoldenCorpusError(f"output parent does not exist: {output_path.parent}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    published = False
    try:
        for filename, payload in sorted(corpus.blobs.items()):
            destination = temporary / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (temporary / GOLDEN_MANIFEST_NAME).write_bytes(corpus.manifest_bytes)
        load_golden_corpus(
            temporary,
            expected_manifest_sha256=_sha256(corpus.manifest_bytes),
        )
        os.replace(temporary, output_path)
        published = True
        return output_path
    except OSError as error:
        raise GoldenCorpusError(f"cannot publish golden corpus: {error}") from error
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _require_fields(document: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(document) is not dict:
        raise GoldenCorpusError(f"{label} must be an object")
    actual = set(document)
    if actual != expected:
        raise GoldenCorpusError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _validate_hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GoldenCorpusError(f"{label} is not a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise GoldenCorpusError(f"{label} is not hexadecimal") from error
    return value


def _read_blob(
    root: Path,
    document: Mapping[str, Any],
    expected_filename: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> bytes:
    _require_fields(
        document,
        {"byte_length", "element_count", "file", "sha256", "shape"},
        label,
    )
    if document["file"] != expected_filename:
        raise GoldenCorpusError(f"{label} filename differs")
    if document["shape"] != list(expected_shape):
        raise GoldenCorpusError(f"{label} shape differs")
    element_count = math.prod(expected_shape)
    if document["element_count"] != element_count:
        raise GoldenCorpusError(f"{label} element count differs")
    if document["byte_length"] != element_count * 4:
        raise GoldenCorpusError(f"{label} byte length differs")
    expected_digest = _validate_hex_digest(document["sha256"], f"{label} digest")
    try:
        payload = (root / expected_filename).read_bytes()
    except OSError as error:
        raise GoldenCorpusError(f"cannot read {label}: {error}") from error
    if len(payload) != document["byte_length"]:
        raise GoldenCorpusError(f"{label} file length differs")
    if _sha256(payload) != expected_digest:
        raise GoldenCorpusError(f"{label} digest differs")
    values = np.frombuffer(payload, dtype="<f4")
    if not bool(np.isfinite(values).all()):
        raise GoldenCorpusError(f"{label} contains a non-finite value")
    return payload


def _validate_sparse_input(payload: bytes, shape: tuple[int, ...], label: str) -> None:
    values = np.frombuffer(payload, dtype="<f4").reshape(shape)
    _, channels, height, width = shape
    if channels != 3:
        raise GoldenCorpusError(f"{label} does not have three channels")
    for row in range(height):
        for column in range(width):
            observed = CANONICAL_XTRANS[row % 6][column % 6]
            for channel in range(3):
                if channel != observed and values[0, channel, row, column] != 0:
                    raise GoldenCorpusError(f"{label} is not a sparse X-Trans input")


def _expected_trace_shapes(height: int, width: int) -> list[tuple[str, tuple[int, ...]]]:
    result = [
        (
            f"main_processor.relu{index}",
            (1, 64, height - index * 2, width - index * 2),
        )
        for index in range(1, 12)
    ]
    result.append(("fullres_processor.input_concat", (1, 67, height - 22, width - 22)))
    result.append(("fullres_processor.post_relu", (1, 64, height - 24, width - 24)))
    return result


def _validate_trace(
    activations: Any,
    case_id: str,
    height: int,
    width: int,
) -> None:
    expected = _expected_trace_shapes(height, width) if case_id == TRACE_CASE_ID else []
    if not isinstance(activations, list) or len(activations) != len(expected):
        raise GoldenCorpusError(f"{case_id} intermediate activation list differs")
    for index, (activation, (name, shape)) in enumerate(zip(activations, expected, strict=True)):
        if not isinstance(activation, dict):
            raise GoldenCorpusError(f"{case_id} activation {index} is not an object")
        _require_fields(
            activation,
            {"byte_length", "element_count", "name", "sha256", "shape"},
            f"{case_id} activation {index}",
        )
        count = math.prod(shape)
        if (
            activation["name"] != name
            or activation["shape"] != list(shape)
            or activation["element_count"] != count
            or activation["byte_length"] != count * 4
        ):
            raise GoldenCorpusError(f"{case_id} activation {index} metadata differs")
        _validate_hex_digest(
            activation["sha256"],
            f"{case_id} activation {index} digest",
        )


def _validate_random_coverage(payload: bytes) -> None:
    shape = EXPECTED_CASE_SHAPES[TRACE_CASE_ID]
    values = np.frombuffer(payload, dtype="<f4").reshape(shape)
    height, width = shape[-2:]
    phases = set()
    for row in range(height):
        for column in range(width):
            channel = CANONICAL_XTRANS[row % 6][column % 6]
            if values[0, channel, row, column] != 0:
                phases.add((row % 6, column % 6))
    if len(phases) != 36:
        raise GoldenCorpusError("random case does not exercise all 36 CFA phases")
    observed_values = [
        values[0, CANONICAL_XTRANS[row % 6][column % 6], row, column]
        for row in range(height)
        for column in range(width)
    ]
    if not all(value != 0 for value in observed_values):
        raise GoldenCorpusError("random case does not exercise every input site")
    output_height = height - 24
    output_width = width - 24
    output_phases = {
        ((row + 12) % 6, (column + 12) % 6)
        for row in range(output_height)
        for column in range(output_width)
    }
    if len(output_phases) != 36:
        raise GoldenCorpusError("random output does not cover every CFA phase")


def load_golden_corpus(
    root: str | Path,
    *,
    expected_manifest_sha256: str = GOLDEN_MANIFEST_SHA256,
) -> LoadedGoldenCorpus:
    """Strictly authenticate the tracked corpus without an external model artifact."""

    root_path = Path(root)
    manifest_path = root_path / GOLDEN_MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise GoldenCorpusError(f"cannot read golden manifest: {error}") from error
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise GoldenCorpusError("golden manifest exceeds its size limit")
    digest = _sha256(manifest_bytes)
    if digest != expected_manifest_sha256:
        raise GoldenCorpusError(
            f"golden manifest SHA-256 is {digest}; expected {expected_manifest_sha256}"
        )
    try:
        manifest = parse_canonical_json(manifest_bytes, "golden manifest")
    except SemanticSchemaError as error:
        raise GoldenCorpusError(str(error)) from error

    _require_fields(
        manifest,
        {"cases", "execution", "format", "model", "provenance", "summary"},
        "golden manifest",
    )
    if manifest["format"] != GOLDEN_CORPUS_FORMAT:
        raise GoldenCorpusError("golden manifest format differs")

    binding = reviewed_binding()
    expected_model = {
        "architecture_id": binding.architecture_id,
        "architecture_symbol": binding.architecture_symbol,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "id": binding.model_id,
        "revision": binding.model_revision,
        "rtnn_sha256": binding.artifact_sha256,
        "semantic_schema_sha256": binding.semantic_schema_sha256,
    }
    if manifest["model"] != expected_model:
        raise GoldenCorpusError("golden model binding differs")

    expected_execution = {
        "byte_order": "little-endian",
        "deterministic_algorithms": True,
        "device": "cpu",
        "input_contract": "canonical-sparse-xtrans-network-input",
        "mode": "inference",
        "onednn_enabled": False,
        "postprocessing": "none",
        "preprocessing": "none",
        "scalar_type": "float32",
        "tensor_layout": "NCHW",
        "threads": 1,
        "torch_version": EXPECTED_TORCH_VERSION,
    }
    if manifest["execution"] != expected_execution:
        raise GoldenCorpusError("golden execution contract differs")

    provenance = manifest["provenance"]
    expected_provenance = {
        "attribution_file": "ATTRIBUTION.md",
        "attribution_sha256": _sha256(ATTRIBUTION_TEXT.encode("utf-8")),
        "license": "MIT",
        "license_file": "UPSTREAM-LICENSE.txt",
        "license_sha256": _sha256(UPSTREAM_LICENSE_TEXT.encode("utf-8")),
        "upstream_repository": "https://github.com/mgharbi/demosaicnet",
        "upstream_revision": "959e9d1630976b421d5af5e35b2e2a01f5630e5c",
    }
    if provenance != expected_provenance:
        raise GoldenCorpusError("golden provenance differs")

    blobs: dict[str, bytes] = {}
    for filename, expected_payload in (
        ("ATTRIBUTION.md", ATTRIBUTION_TEXT.encode("utf-8")),
        ("UPSTREAM-LICENSE.txt", UPSTREAM_LICENSE_TEXT.encode("utf-8")),
    ):
        try:
            actual = (root_path / filename).read_bytes()
        except OSError as error:
            raise GoldenCorpusError(f"cannot read {filename}: {error}") from error
        if actual != expected_payload:
            raise GoldenCorpusError(f"{filename} differs")
        blobs[filename] = actual

    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_CASE_SHAPES):
        raise GoldenCorpusError("golden case count differs")
    for case, (expected_id, input_shape) in zip(
        cases,
        EXPECTED_CASE_SHAPES.items(),
        strict=True,
    ):
        if not isinstance(case, dict):
            raise GoldenCorpusError("golden case is not an object")
        _require_fields(
            case,
            {"id", "input", "intermediate_activations", "output"},
            "golden case",
        )
        if case["id"] != expected_id:
            raise GoldenCorpusError("golden case ordering differs")
        output_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2] - 24,
            input_shape[3] - 24,
        )
        input_filename = f"cases/{expected_id}.input.f32le"
        output_filename = f"cases/{expected_id}.output.f32le"
        input_payload = _read_blob(
            root_path,
            case["input"],
            input_filename,
            input_shape,
            f"{expected_id} input",
        )
        output_payload = _read_blob(
            root_path,
            case["output"],
            output_filename,
            output_shape,
            f"{expected_id} output",
        )
        _validate_sparse_input(input_payload, input_shape, f"{expected_id} input")
        _validate_trace(
            case["intermediate_activations"],
            expected_id,
            input_shape[-2],
            input_shape[-1],
        )
        blobs[input_filename] = input_payload
        blobs[output_filename] = output_payload

    _validate_random_coverage(blobs[f"cases/{TRACE_CASE_ID}.input.f32le"])
    float_bytes = sum(
        len(payload) for filename, payload in blobs.items() if filename.endswith(".f32le")
    )
    expected_summary = {
        "case_count": len(EXPECTED_CASE_SHAPES),
        "float_blob_bytes": float_bytes,
        "float_blob_count": len(EXPECTED_CASE_SHAPES) * 2,
        "trace_case": TRACE_CASE_ID,
    }
    if manifest["summary"] != expected_summary:
        raise GoldenCorpusError("golden summary differs")

    expected_files = {GOLDEN_MANIFEST_NAME, *blobs}
    actual_files = {
        str(path.relative_to(root_path))
        for path in root_path.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise GoldenCorpusError(
            f"golden files differ: missing={sorted(expected_files - actual_files)}, "
            f"unknown={sorted(actual_files - expected_files)}"
        )
    return LoadedGoldenCorpus(manifest, manifest_bytes, blobs)

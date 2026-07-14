"""Deterministic RTNN v1 serialization for authenticated neural weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

from .schema import GHARBI_XTRANS_V1
from .semantic_schema import (
    SEMANTIC_SCHEMA_SHA256,
    SemanticSchema,
    SemanticTensorSpec,
    canonical_json_bytes,
)


RTNN_MAGIC = b"RTNN\r\n\x1a\n"
RTNN_FORMAT_MAJOR = 1
RTNN_FORMAT_MINOR = 0
RTNN_ENDIAN_MARKER = 0x01020304
RTNN_HEADER_SIZE = 192
RTNN_DIRECTORY_RECORD_SIZE = 96
RTNN_ALIGNMENT = 64
RTNN_ARCHITECTURE_DEMOSAICNET_XTRANS_V1 = 1
RTNN_MODEL_REVISION_GHARBI_XTRANS_V1 = 1
RTNN_SCALAR_FLOAT32 = 1
RTNN_LAYOUT_VECTOR = 1
RTNN_LAYOUT_OIHW = 2
RTNN_FLAGS = 0
RTNN_TENSOR_COUNT = 26
RTNN_EXPECTED_TENSOR_BYTES = 1_639_692
RTNN_EXPECTED_PAYLOAD_REGION_BYTES = 1_639_744
RTNN_EXPECTED_FILE_SIZE = 1_642_432
RTNN_GHARBI_XTRANS_V1_PAYLOAD_SHA256 = (
    "e0e501a3f3a4905e3c7bb1ab1f0e3acb5598da818d6406d5cf6ed030ab5af606"
)
RTNN_GHARBI_XTRANS_V1_SHA256 = (
    "b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2"
)
RTNN_GHARBI_XTRANS_V1_MANIFEST_SHA256 = (
    "f9b5d784356a455327304cfbfe302b2041a5a1a1eb2970134e3c1dfb621ec447"
)
CONVERSION_MANIFEST_FORMAT = "rawtherapee-rtnn-conversion-manifest-v1"

_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

_HEADER_PREFIX = struct.Struct("<8sHH9I6Q")
_DIRECTORY_RECORD = struct.Struct("<IHHII4IQQQ32sQ")

assert _HEADER_PREFIX.size == 96
assert _DIRECTORY_RECORD.size == RTNN_DIRECTORY_RECORD_SIZE


class RTNNError(RuntimeError):
    """The semantic tensor set cannot be represented as canonical RTNN v1."""


@dataclass(frozen=True)
class RTNNTensorRecord:
    """Metadata for one serialized tensor payload."""

    id: int
    symbol: str
    source_name: str
    layout: str
    shape: tuple[int, ...]
    element_count: int
    payload_offset: int
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class SerializedRTNN:
    """A complete RTNN artifact and the metadata needed for its manifest."""

    data: bytes
    payload_sha256: str
    payload_region_bytes: int
    tensor_payload_bytes: int
    records: tuple[RTNNTensorRecord, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def _require_uint(value: int, maximum: int, label: str) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise RTNNError(f"{label} is outside its unsigned integer range")

    return value


def _checked_add(left: int, right: int, label: str) -> int:
    _require_uint(left, _UINT64_MAX, label)
    _require_uint(right, _UINT64_MAX, label)

    if left > _UINT64_MAX - right:
        raise RTNNError(f"{label} overflows uint64")

    return left + right


def _checked_multiply(left: int, right: int, label: str) -> int:
    _require_uint(left, _UINT64_MAX, label)
    _require_uint(right, _UINT64_MAX, label)

    if left and right > _UINT64_MAX // left:
        raise RTNNError(f"{label} overflows uint64")

    return left * right


def _align_up(value: int, alignment: int = RTNN_ALIGNMENT) -> int:
    remainder = value % alignment

    if remainder == 0:
        return value

    return _checked_add(value, alignment - remainder, "aligned offset")


def _digest_bytes(value: str, label: str) -> bytes:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RTNNError(f"{label} is not a lowercase SHA-256 digest")

    return bytes.fromhex(value)


def _layout_id(layout: str) -> int:
    try:
        return {
            "vector": RTNN_LAYOUT_VECTOR,
            "OIHW": RTNN_LAYOUT_OIHW,
        }[layout]
    except KeyError as error:
        raise RTNNError(f"unsupported tensor layout: {layout}") from error


def _validate_tensor(
    tensor: SemanticTensorSpec,
    payload: bytes,
    expected_id: int,
) -> tuple[tuple[int, int, int, int], int, bytes]:
    _require_uint(tensor.id, _UINT32_MAX, "tensor ID")

    if tensor.id != expected_id:
        raise RTNNError(
            f"tensor ID {tensor.id} is out of canonical order; expected {expected_id}"
        )

    if tensor.dtype != "float32":
        raise RTNNError(f"tensor {tensor.id} has unsupported dtype {tensor.dtype}")

    source_spec = GHARBI_XTRANS_V1.tensors[expected_id - 1]
    expected_symbol = (
        f"MAIN_CONV{(expected_id + 1) // 2}_"
        f"{'WEIGHT' if expected_id % 2 else 'BIAS'}"
        if expected_id <= 22
        else {
            23: "POST_CONV_WEIGHT",
            24: "POST_CONV_BIAS",
            25: "OUTPUT_WEIGHT",
            26: "OUTPUT_BIAS",
        }[expected_id]
    )

    if tensor.symbol != expected_symbol:
        raise RTNNError(
            f"tensor {tensor.id} symbol is {tensor.symbol}; expected {expected_symbol}"
        )

    if tensor.source_name != source_spec.name:
        raise RTNNError(
            f"tensor {tensor.id} source is {tensor.source_name}; "
            f"expected {source_spec.name}"
        )

    if tensor.shape != source_spec.shape:
        raise RTNNError(
            f"tensor {tensor.id} shape is {tensor.shape}; expected {source_spec.shape}"
        )

    if tensor.rank not in (1, 4) or tensor.rank != len(tensor.shape):
        raise RTNNError(f"tensor {tensor.id} has unsupported rank {tensor.rank}")

    _require_uint(tensor.rank, _UINT16_MAX, f"tensor {tensor.id} rank")

    expected_layout = "vector" if tensor.rank == 1 else "OIHW"

    if tensor.layout != expected_layout:
        raise RTNNError(
            f"tensor {tensor.id} has layout {tensor.layout}; expected {expected_layout}"
        )

    dimensions = []
    element_count = 1

    for index, dimension in enumerate(tensor.shape):
        _require_uint(dimension, _UINT32_MAX, f"tensor {tensor.id} dimension {index}")

        if dimension == 0:
            raise RTNNError(f"tensor {tensor.id} has a zero dimension")

        element_count = _checked_multiply(
            element_count,
            dimension,
            f"tensor {tensor.id} element count",
        )
        dimensions.append(dimension)

    if element_count != tensor.element_count:
        raise RTNNError(
            f"tensor {tensor.id} element count is {tensor.element_count}; "
            f"shape requires {element_count}"
        )

    expected_bytes = _checked_multiply(
        tensor.element_count,
        4,
        f"tensor {tensor.id} byte length",
    )

    if not isinstance(payload, bytes):
        raise RTNNError(f"tensor {tensor.id} payload must be immutable bytes")

    if len(payload) != expected_bytes:
        raise RTNNError(
            f"tensor {tensor.id} payload has {len(payload)} bytes; "
            f"expected {expected_bytes}"
        )

    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise RTNNError(f"tensor {tensor.id} payload contains NaN or infinity")

    digest = hashlib.sha256(payload).hexdigest()

    if digest != tensor.sha256:
        raise RTNNError(
            f"tensor {tensor.id} SHA-256 is {digest}; expected {tensor.sha256}"
        )

    dimensions.extend([0] * (4 - len(dimensions)))
    return tuple(dimensions), _layout_id(tensor.layout), bytes.fromhex(digest)


def serialize_rtnn(
    schema: SemanticSchema,
    tensor_payloads: Sequence[bytes],
) -> SerializedRTNN:
    """Serialize one authenticated semantic tensor set as canonical RTNN v1."""

    _require_uint(schema.architecture_id, _UINT32_MAX, "architecture ID")

    if schema.architecture_id != RTNN_ARCHITECTURE_DEMOSAICNET_XTRANS_V1:
        raise RTNNError(f"unsupported architecture ID: {schema.architecture_id}")

    if schema.architecture_symbol != "DEMOSAICNET_XTRANS_V1":
        raise RTNNError(f"unsupported architecture symbol: {schema.architecture_symbol}")

    if schema.model_id != GHARBI_XTRANS_V1.model_id:
        raise RTNNError(f"unsupported model ID: {schema.model_id}")

    if schema.canonical_sha256 != SEMANTIC_SCHEMA_SHA256:
        raise RTNNError(
            f"semantic schema SHA-256 is {schema.canonical_sha256}; "
            f"expected {SEMANTIC_SCHEMA_SHA256}"
        )

    if schema.checkpoint_sha256 != GHARBI_XTRANS_V1.expected_sha256:
        raise RTNNError(
            f"checkpoint SHA-256 is {schema.checkpoint_sha256}; "
            f"expected {GHARBI_XTRANS_V1.expected_sha256}"
        )

    if len(schema.tensors) != RTNN_TENSOR_COUNT:
        raise RTNNError(
            f"semantic schema has {len(schema.tensors)} tensors; "
            f"expected {RTNN_TENSOR_COUNT}"
        )

    if len(tensor_payloads) != len(schema.tensors):
        raise RTNNError(
            f"received {len(tensor_payloads)} tensor payloads; "
            f"expected {len(schema.tensors)}"
        )

    if len({tensor.id for tensor in schema.tensors}) != len(schema.tensors):
        raise RTNNError("semantic schema contains duplicate tensor IDs")

    if len({tensor.symbol for tensor in schema.tensors}) != len(schema.tensors):
        raise RTNNError("semantic schema contains duplicate tensor symbols")

    if len({tensor.source_name for tensor in schema.tensors}) != len(schema.tensors):
        raise RTNNError("semantic schema contains duplicate source tensor names")

    directory_size = _checked_multiply(
        len(schema.tensors),
        RTNN_DIRECTORY_RECORD_SIZE,
        "directory size",
    )
    directory_offset = RTNN_HEADER_SIZE
    payload_offset = _checked_add(directory_offset, directory_size, "payload offset")

    if payload_offset % RTNN_ALIGNMENT:
        raise RTNNError("RTNN v1 directory does not end on a 64-byte boundary")

    payload_region = bytearray()
    record_values = []
    records = []
    parameter_count = 0
    tensor_payload_bytes = 0

    for expected_id, (tensor, payload) in enumerate(
        zip(schema.tensors, tensor_payloads, strict=True),
        start=1,
    ):
        dimensions, layout_id, tensor_digest = _validate_tensor(
            tensor,
            payload,
            expected_id,
        )
        aligned_offset = _align_up(len(payload_region))
        payload_region.extend(b"\0" * (aligned_offset - len(payload_region)))
        payload_region.extend(payload)
        byte_length = len(payload)
        parameter_count = _checked_add(
            parameter_count,
            tensor.element_count,
            "parameter count",
        )
        tensor_payload_bytes = _checked_add(
            tensor_payload_bytes,
            byte_length,
            "tensor payload bytes",
        )
        record_values.append(
            (
                tensor.id,
                tensor.rank,
                layout_id,
                RTNN_SCALAR_FLOAT32,
                RTNN_FLAGS,
                *dimensions,
                tensor.element_count,
                byte_length,
                aligned_offset,
                tensor_digest,
                0,
            )
        )
        records.append(
            RTNNTensorRecord(
                id=tensor.id,
                symbol=tensor.symbol,
                source_name=tensor.source_name,
                layout=tensor.layout,
                shape=tensor.shape,
                element_count=tensor.element_count,
                payload_offset=aligned_offset,
                byte_length=byte_length,
                sha256=tensor.sha256,
            )
        )

    final_payload_size = _align_up(len(payload_region))
    payload_region.extend(b"\0" * (final_payload_size - len(payload_region)))

    if parameter_count != schema.parameter_count:
        raise RTNNError(
            f"serialized parameter count is {parameter_count}; "
            f"schema requires {schema.parameter_count}"
        )

    if tensor_payload_bytes != schema.payload_bytes:
        raise RTNNError(
            f"serialized tensor payload is {tensor_payload_bytes} bytes; "
            f"schema requires {schema.payload_bytes}"
        )

    file_size = _checked_add(payload_offset, final_payload_size, "file size")
    payload_digest = hashlib.sha256(payload_region).digest()
    header = _HEADER_PREFIX.pack(
        RTNN_MAGIC,
        RTNN_FORMAT_MAJOR,
        RTNN_FORMAT_MINOR,
        RTNN_HEADER_SIZE,
        RTNN_ENDIAN_MARKER,
        RTNN_FLAGS,
        schema.architecture_id,
        RTNN_MODEL_REVISION_GHARBI_XTRANS_V1,
        RTNN_SCALAR_FLOAT32,
        len(schema.tensors),
        RTNN_DIRECTORY_RECORD_SIZE,
        0,
        directory_offset,
        directory_size,
        payload_offset,
        final_payload_size,
        tensor_payload_bytes,
        file_size,
    )
    header += _digest_bytes(schema.canonical_sha256, "semantic schema SHA-256")
    header += payload_digest
    header += _digest_bytes(schema.checkpoint_sha256, "checkpoint SHA-256")

    if len(header) != RTNN_HEADER_SIZE:
        raise RTNNError(f"internal header size is {len(header)} bytes")

    directory = b"".join(_DIRECTORY_RECORD.pack(*values) for values in record_values)
    data = header + directory + payload_region

    if len(data) != file_size:
        raise RTNNError(
            f"internal RTNN size is {len(data)} bytes; expected {file_size}"
        )

    return SerializedRTNN(
        data=data,
        payload_sha256=payload_digest.hex(),
        payload_region_bytes=final_payload_size,
        tensor_payload_bytes=tensor_payload_bytes,
        records=tuple(records),
    )


def build_conversion_manifest(
    schema: SemanticSchema,
    artifact: SerializedRTNN,
) -> dict[str, Any]:
    """Build the deterministic human-readable companion to an RTNN artifact."""

    return {
        "artifact": {
            "directory_bytes": len(artifact.records) * RTNN_DIRECTORY_RECORD_SIZE,
            "file_size": len(artifact.data),
            "format": {
                "magic_hex": RTNN_MAGIC.hex(),
                "major": RTNN_FORMAT_MAJOR,
                "minor": RTNN_FORMAT_MINOR,
            },
            "header_bytes": RTNN_HEADER_SIZE,
            "payload_region_bytes": artifact.payload_region_bytes,
            "payload_sha256": artifact.payload_sha256,
            "sha256": artifact.sha256,
            "tensor_payload_bytes": artifact.tensor_payload_bytes,
        },
        "format": CONVERSION_MANIFEST_FORMAT,
        "model": {
            "architecture": {
                "id": schema.architecture_id,
                "symbol": schema.architecture_symbol,
            },
            "id": schema.model_id,
            "revision": RTNN_MODEL_REVISION_GHARBI_XTRANS_V1,
            "semantic_schema_sha256": schema.canonical_sha256,
        },
        "source": {
            "checkpoint": {
                "path": GHARBI_XTRANS_V1.upstream_checkpoint_path,
                "sha256": schema.checkpoint_sha256,
                "size_bytes": GHARBI_XTRANS_V1.expected_file_size,
            },
            "inspection_manifest": {
                "format": schema.inspection_manifest_format,
                "sha256": schema.inspection_manifest_sha256,
            },
            "license": GHARBI_XTRANS_V1.upstream_license,
            "repository": GHARBI_XTRANS_V1.upstream_repository,
            "revision": GHARBI_XTRANS_V1.upstream_revision,
        },
        "summary": {
            "parameter_count": schema.parameter_count,
            "scalar_type": "float32",
            "tensor_count": len(artifact.records),
        },
        "tensors": [
            {
                "byte_length": record.byte_length,
                "element_count": record.element_count,
                "id": record.id,
                "layout": record.layout,
                "payload_offset": record.payload_offset,
                "sha256": record.sha256,
                "shape": list(record.shape),
                "source_name": record.source_name,
                "symbol": record.symbol,
            }
            for record in artifact.records
        ],
    }


def canonical_conversion_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Encode a conversion manifest with the shared canonical JSON rules."""

    return canonical_json_bytes(manifest)

"""Writer-independent, strict reader for the reviewed RTNN v1 artifact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import struct

from .semantic_schema import SemanticSchemaError, load_semantic_schema


# These wire values intentionally duplicate devnotes/rtnn-v1-format.md. This
# reader must not share parsing constants or helpers with the RTNN writer.
_MAGIC = b"RTNN\r\n\x1a\n"
_FORMAT_MAJOR = 1
_FORMAT_MINOR = 0
_ENDIAN_MARKER = 0x01020304
_HEADER_SIZE = 192
_RECORD_SIZE = 96
_ALIGNMENT = 64
_ARCHITECTURE_ID = 1
_MODEL_REVISION = 1
_SCALAR_FLOAT32 = 1
_LAYOUT_VECTOR = 1
_LAYOUT_OIHW = 2
_FLAGS = 0
_REVIEWED_RTNN_SHA256 = (
    "b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2"
)

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_TENSOR_COUNT = 256
MAX_RANK = 4
MAX_DIMENSION = 1_048_576
MAX_TENSOR_ELEMENTS = 16_777_216

_UINT64_MAX = (1 << 64) - 1


class RTNNErrorCode(str, Enum):
    IO = "IO"
    SIZE = "SIZE"
    MAGIC = "MAGIC"
    VERSION = "VERSION"
    ENDIAN = "ENDIAN"
    FLAGS = "FLAGS"
    ENUM = "ENUM"
    RESERVED = "RESERVED"
    LIMIT = "LIMIT"
    RANGE = "RANGE"
    ALIGNMENT = "ALIGNMENT"
    ORDER = "ORDER"
    SCHEMA = "SCHEMA"
    DIGEST = "DIGEST"
    NONFINITE = "NONFINITE"


class RTNNReadError(RuntimeError):
    """A rejected RTNN artifact with a stable machine-readable category."""

    def __init__(self, code: RTNNErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TensorBinding:
    id: int
    symbol: str
    source_name: str
    rank: int
    layout_id: int
    layout: str
    shape: tuple[int, ...]
    element_count: int
    sha256: str


@dataclass(frozen=True)
class ModelBinding:
    architecture_id: int
    architecture_symbol: str
    model_id: str
    model_revision: int
    semantic_schema_sha256: str
    checkpoint_sha256: str
    artifact_sha256: str
    parameter_count: int
    tensor_payload_bytes: int
    tensors: tuple[TensorBinding, ...]


@dataclass(frozen=True)
class LoadedTensor:
    id: int
    symbol: str
    source_name: str
    rank: int
    layout_id: int
    layout: str
    shape: tuple[int, ...]
    element_count: int
    payload_offset: int
    byte_length: int
    sha256: str
    data: bytes


@dataclass(frozen=True)
class LoadedRTNN:
    format_major: int
    format_minor: int
    architecture_id: int
    architecture_symbol: str
    model_id: str
    model_revision: int
    semantic_schema_sha256: str
    checkpoint_sha256: str
    artifact_sha256: str
    payload_sha256: str
    file_size: int
    header_bytes: int
    directory_bytes: int
    payload_region_bytes: int
    tensor_payload_bytes: int
    parameter_count: int
    tensors: tuple[LoadedTensor, ...]

    def tensor(self, tensor_id: int) -> LoadedTensor:
        if tensor_id < 1 or tensor_id > len(self.tensors):
            raise KeyError(tensor_id)

        tensor = self.tensors[tensor_id - 1]

        if tensor.id != tensor_id:
            raise KeyError(tensor_id)

        return tensor


def _fail(code: RTNNErrorCode, message: str) -> None:
    raise RTNNReadError(code, message)


def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little", signed=False)


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little", signed=False)


def _u64(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 8], "little", signed=False)


def _checked_add(left: int, right: int, label: str) -> int:
    if left < 0 or right < 0 or left > _UINT64_MAX or right > _UINT64_MAX:
        _fail(RTNNErrorCode.RANGE, f"{label} is outside uint64")

    if left > _UINT64_MAX - right:
        _fail(RTNNErrorCode.RANGE, f"{label} overflows uint64")

    return left + right


def _checked_multiply(left: int, right: int, label: str) -> int:
    if left < 0 or right < 0 or left > _UINT64_MAX or right > _UINT64_MAX:
        _fail(RTNNErrorCode.RANGE, f"{label} is outside uint64")

    if left and right > _UINT64_MAX // left:
        _fail(RTNNErrorCode.RANGE, f"{label} overflows uint64")

    return left * right


def _align_up(value: int) -> int:
    remainder = value % _ALIGNMENT
    return value if remainder == 0 else _checked_add(
        value,
        _ALIGNMENT - remainder,
        "aligned offset",
    )


@lru_cache(maxsize=1)
def reviewed_binding() -> ModelBinding:
    """Return the authenticated Gharbi X-Trans model binding."""

    try:
        schema = load_semantic_schema()
    except SemanticSchemaError as error:
        raise RTNNReadError(
            RTNNErrorCode.SCHEMA,
            f"cannot authenticate reviewed semantic schema: {error}",
        ) from error
    tensors = tuple(
        TensorBinding(
            id=tensor.id,
            symbol=tensor.symbol,
            source_name=tensor.source_name,
            rank=tensor.rank,
            layout_id=(
                _LAYOUT_VECTOR if tensor.layout == "vector" else _LAYOUT_OIHW
            ),
            layout=tensor.layout,
            shape=tensor.shape,
            element_count=tensor.element_count,
            sha256=tensor.sha256,
        )
        for tensor in schema.tensors
    )
    return ModelBinding(
        architecture_id=_ARCHITECTURE_ID,
        architecture_symbol=schema.architecture_symbol,
        model_id=schema.model_id,
        model_revision=_MODEL_REVISION,
        semantic_schema_sha256=schema.canonical_sha256,
        checkpoint_sha256=schema.checkpoint_sha256,
        artifact_sha256=_REVIEWED_RTNN_SHA256,
        parameter_count=schema.parameter_count,
        tensor_payload_bytes=schema.payload_bytes,
        tensors=tensors,
    )


def _require_zero(data: bytes, start: int, end: int, label: str) -> None:
    if any(data[start:end]):
        _fail(RTNNErrorCode.RESERVED, f"{label} is not zero")


def _parse_rtnn(data: bytes, binding: ModelBinding) -> LoadedRTNN:
    """Internal parser with injectable binding for isolated test fixtures."""

    if type(data) is not bytes:
        _fail(RTNNErrorCode.SIZE, "RTNN input must be immutable bytes")

    if len(data) > MAX_FILE_BYTES:
        _fail(RTNNErrorCode.LIMIT, "RTNN exceeds the 64 MiB file limit")

    if len(data) < _HEADER_SIZE:
        _fail(RTNNErrorCode.SIZE, "RTNN is shorter than its fixed header")

    if data[0:8] != _MAGIC:
        _fail(RTNNErrorCode.MAGIC, "RTNN magic differs")

    major = _u16(data, 8)
    minor = _u16(data, 10)

    if major != _FORMAT_MAJOR or minor != _FORMAT_MINOR:
        _fail(RTNNErrorCode.VERSION, f"unsupported RTNN version {major}.{minor}")

    header_size = _u32(data, 12)

    if header_size != _HEADER_SIZE:
        _fail(RTNNErrorCode.SIZE, f"header size is {header_size}; expected 192")

    if _u32(data, 16) != _ENDIAN_MARKER:
        _fail(RTNNErrorCode.ENDIAN, "RTNN endian marker differs")

    if _u32(data, 20) != _FLAGS:
        _fail(RTNNErrorCode.FLAGS, "RTNN header contains unknown flags")

    architecture_id = _u32(data, 24)

    if architecture_id == 0:
        _fail(RTNNErrorCode.ENUM, "architecture ID zero is invalid")

    if architecture_id != binding.architecture_id:
        _fail(RTNNErrorCode.SCHEMA, "RTNN architecture is not the reviewed model")

    model_revision = _u32(data, 28)

    if model_revision == 0:
        _fail(RTNNErrorCode.ENUM, "model revision zero is invalid")

    if model_revision != binding.model_revision:
        _fail(RTNNErrorCode.SCHEMA, "RTNN model revision is not reviewed")

    scalar_type = _u32(data, 32)

    if scalar_type != _SCALAR_FLOAT32:
        _fail(RTNNErrorCode.ENUM, "unsupported RTNN scalar type")

    tensor_count = _u32(data, 36)

    if tensor_count > MAX_TENSOR_COUNT:
        _fail(RTNNErrorCode.LIMIT, "RTNN tensor count exceeds the configured limit")

    if tensor_count != len(binding.tensors):
        _fail(RTNNErrorCode.SCHEMA, "RTNN tensor count differs from the model schema")

    record_size = _u32(data, 40)

    if record_size != _RECORD_SIZE:
        _fail(RTNNErrorCode.SIZE, "RTNN directory-record size differs")

    if _u32(data, 44) != 0:
        _fail(RTNNErrorCode.RESERVED, "RTNN header reserved field is not zero")

    directory_offset = _u64(data, 48)
    directory_size = _u64(data, 56)
    payload_offset = _u64(data, 64)
    payload_size = _u64(data, 72)
    tensor_payload_bytes = _u64(data, 80)
    declared_file_size = _u64(data, 88)
    schema_digest = data[96:128].hex()
    payload_digest = data[128:160]
    checkpoint_digest = data[160:192].hex()

    if payload_size > MAX_PAYLOAD_BYTES:
        _fail(RTNNErrorCode.LIMIT, "RTNN payload exceeds the 64 MiB limit")

    expected_directory_size = _checked_multiply(
        tensor_count,
        record_size,
        "directory size",
    )

    if directory_offset != header_size or directory_size != expected_directory_size:
        _fail(RTNNErrorCode.RANGE, "RTNN directory placement is not canonical")

    directory_end = _checked_add(directory_offset, directory_size, "directory end")

    if payload_offset != directory_end:
        _fail(RTNNErrorCode.RANGE, "RTNN payload does not immediately follow directory")

    if payload_offset % _ALIGNMENT:
        _fail(RTNNErrorCode.ALIGNMENT, "RTNN payload region is not 64-byte aligned")

    payload_end = _checked_add(payload_offset, payload_size, "payload end")

    if payload_end != declared_file_size or declared_file_size != len(data):
        _fail(RTNNErrorCode.SIZE, "RTNN declared and actual file sizes differ")

    if schema_digest != binding.semantic_schema_sha256:
        _fail(RTNNErrorCode.SCHEMA, "RTNN semantic-schema digest differs")

    if checkpoint_digest != binding.checkpoint_sha256:
        _fail(RTNNErrorCode.SCHEMA, "RTNN checkpoint digest differs")

    parsed_records = []
    parameter_count = 0
    summed_tensor_bytes = 0
    previous_end = 0

    for index, expected in enumerate(binding.tensors):
        record_offset = directory_offset + index * record_size
        tensor_id = _u32(data, record_offset)
        rank = _u16(data, record_offset + 4)
        layout_id = _u16(data, record_offset + 6)
        record_scalar = _u32(data, record_offset + 8)
        record_flags = _u32(data, record_offset + 12)
        dimensions = tuple(_u32(data, record_offset + 16 + i * 4) for i in range(4))
        element_count = _u64(data, record_offset + 32)
        byte_length = _u64(data, record_offset + 40)
        relative_offset = _u64(data, record_offset + 48)
        tensor_digest = data[record_offset + 56 : record_offset + 88].hex()
        record_reserved = _u64(data, record_offset + 88)

        if tensor_id != index + 1 or tensor_id != expected.id:
            _fail(RTNNErrorCode.ORDER, "RTNN tensor IDs are not canonical")

        if rank > MAX_RANK:
            _fail(RTNNErrorCode.LIMIT, f"tensor {tensor_id} rank exceeds the limit")

        if rank not in (1, 4):
            _fail(RTNNErrorCode.ENUM, f"tensor {tensor_id} rank is unsupported")

        if layout_id not in (_LAYOUT_VECTOR, _LAYOUT_OIHW):
            _fail(RTNNErrorCode.ENUM, f"tensor {tensor_id} layout is unsupported")

        if record_scalar != _SCALAR_FLOAT32:
            _fail(RTNNErrorCode.ENUM, f"tensor {tensor_id} scalar type is unsupported")

        if record_flags != _FLAGS:
            _fail(RTNNErrorCode.FLAGS, f"tensor {tensor_id} contains unknown flags")

        if record_reserved != 0:
            _fail(RTNNErrorCode.RESERVED, f"tensor {tensor_id} reserved field is nonzero")

        for dimension in dimensions[:rank]:
            if dimension == 0:
                _fail(RTNNErrorCode.SCHEMA, f"tensor {tensor_id} has a zero dimension")

            if dimension > MAX_DIMENSION:
                _fail(RTNNErrorCode.LIMIT, f"tensor {tensor_id} dimension exceeds the limit")

        if any(dimensions[rank:]):
            _fail(RTNNErrorCode.RESERVED, f"tensor {tensor_id} unused dimensions are nonzero")

        shape = dimensions[:rank]
        shape_count = 1

        for dimension in shape:
            shape_count = _checked_multiply(
                shape_count,
                dimension,
                f"tensor {tensor_id} element count",
            )

        if element_count > MAX_TENSOR_ELEMENTS:
            _fail(RTNNErrorCode.LIMIT, f"tensor {tensor_id} element count exceeds the limit")

        if element_count != shape_count:
            _fail(RTNNErrorCode.SCHEMA, f"tensor {tensor_id} element count differs from shape")

        expected_bytes = _checked_multiply(
            element_count,
            4,
            f"tensor {tensor_id} byte length",
        )

        if byte_length != expected_bytes:
            _fail(RTNNErrorCode.SCHEMA, f"tensor {tensor_id} byte length differs")

        if rank != expected.rank or layout_id != expected.layout_id:
            _fail(RTNNErrorCode.SCHEMA, f"tensor {tensor_id} rank or layout differs")

        if shape != expected.shape or element_count != expected.element_count:
            _fail(RTNNErrorCode.SCHEMA, f"tensor {tensor_id} shape differs")

        if tensor_digest != expected.sha256:
            _fail(RTNNErrorCode.DIGEST, f"tensor {tensor_id} recorded digest differs")

        if relative_offset % _ALIGNMENT:
            _fail(RTNNErrorCode.ALIGNMENT, f"tensor {tensor_id} is not 64-byte aligned")

        expected_offset = _align_up(previous_end)

        if relative_offset != expected_offset:
            if relative_offset < previous_end:
                _fail(RTNNErrorCode.RANGE, f"tensor {tensor_id} overlaps its predecessor")

            _fail(RTNNErrorCode.ALIGNMENT, f"tensor {tensor_id} padding is not minimal")

        tensor_end = _checked_add(relative_offset, byte_length, f"tensor {tensor_id} end")

        if tensor_end > payload_size:
            _fail(RTNNErrorCode.RANGE, f"tensor {tensor_id} exceeds the payload region")

        parameter_count = _checked_add(
            parameter_count,
            element_count,
            "model parameter count",
        )
        summed_tensor_bytes = _checked_add(
            summed_tensor_bytes,
            byte_length,
            "model tensor byte count",
        )
        parsed_records.append(
            (
                expected,
                relative_offset,
                byte_length,
                tensor_digest,
            )
        )
        previous_end = tensor_end

    canonical_payload_size = _align_up(previous_end)

    if payload_size != canonical_payload_size:
        _fail(RTNNErrorCode.ALIGNMENT, "RTNN trailing payload padding is not minimal")

    if tensor_payload_bytes != summed_tensor_bytes:
        _fail(RTNNErrorCode.SCHEMA, "RTNN tensor-byte summary differs")

    if parameter_count != binding.parameter_count:
        _fail(RTNNErrorCode.SCHEMA, "RTNN parameter count differs from the binding")

    if summed_tensor_bytes != binding.tensor_payload_bytes:
        _fail(RTNNErrorCode.SCHEMA, "RTNN tensor-byte count differs from the binding")

    payload = data[payload_offset:payload_end]

    if hashlib.sha256(payload).digest() != payload_digest:
        _fail(RTNNErrorCode.DIGEST, "RTNN payload digest differs")

    tensors = []
    previous_end = 0

    for expected, relative_offset, byte_length, tensor_digest in parsed_records:
        _require_zero(
            payload,
            previous_end,
            relative_offset,
            f"tensor {expected.id} leading padding",
        )
        tensor_data = bytes(payload[relative_offset : relative_offset + byte_length])

        if hashlib.sha256(tensor_data).hexdigest() != tensor_digest:
            _fail(RTNNErrorCode.DIGEST, f"tensor {expected.id} payload digest differs")

        for (value,) in struct.iter_unpack("<f", tensor_data):
            if not math.isfinite(value):
                _fail(RTNNErrorCode.NONFINITE, f"tensor {expected.id} is non-finite")

        tensors.append(
            LoadedTensor(
                id=expected.id,
                symbol=expected.symbol,
                source_name=expected.source_name,
                rank=expected.rank,
                layout_id=expected.layout_id,
                layout=expected.layout,
                shape=expected.shape,
                element_count=expected.element_count,
                payload_offset=relative_offset,
                byte_length=byte_length,
                sha256=tensor_digest,
                data=tensor_data,
            )
        )
        previous_end = relative_offset + byte_length

    _require_zero(payload, previous_end, len(payload), "trailing payload padding")
    artifact_digest = hashlib.sha256(data).hexdigest()

    if artifact_digest != binding.artifact_sha256:
        _fail(RTNNErrorCode.DIGEST, "RTNN complete-file digest differs")

    return LoadedRTNN(
        format_major=major,
        format_minor=minor,
        architecture_id=architecture_id,
        architecture_symbol=binding.architecture_symbol,
        model_id=binding.model_id,
        model_revision=model_revision,
        semantic_schema_sha256=schema_digest,
        checkpoint_sha256=checkpoint_digest,
        artifact_sha256=artifact_digest,
        payload_sha256=payload_digest.hex(),
        file_size=len(data),
        header_bytes=header_size,
        directory_bytes=directory_size,
        payload_region_bytes=payload_size,
        tensor_payload_bytes=summed_tensor_bytes,
        parameter_count=parameter_count,
        tensors=tuple(tensors),
    )


def parse_rtnn(data: bytes) -> LoadedRTNN:
    """Parse and authenticate the reviewed RTNN artifact from immutable bytes."""

    return _parse_rtnn(data, reviewed_binding())


def read_rtnn(path: str | Path) -> LoadedRTNN:
    """Read a bounded file and authenticate it as the reviewed RTNN artifact."""

    input_path = Path(path)

    try:
        with input_path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size

            if size > MAX_FILE_BYTES:
                _fail(RTNNErrorCode.LIMIT, "RTNN exceeds the 64 MiB file limit")

            data = stream.read(MAX_FILE_BYTES + 1)
    except RTNNReadError:
        raise
    except OSError as error:
        raise RTNNReadError(RTNNErrorCode.IO, f"cannot read RTNN: {error}") from error

    if len(data) != size:
        _fail(RTNNErrorCode.SIZE, "RTNN changed while it was being read")

    return parse_rtnn(data)


def inspection_document(model: LoadedRTNN) -> dict:
    """Derive stable inspection metadata solely from an authenticated RTNN."""

    return {
        "artifact": {
            "directory_bytes": model.directory_bytes,
            "file_size": model.file_size,
            "format": {
                "magic_hex": _MAGIC.hex(),
                "major": model.format_major,
                "minor": model.format_minor,
            },
            "header_bytes": model.header_bytes,
            "payload_region_bytes": model.payload_region_bytes,
            "payload_sha256": model.payload_sha256,
            "sha256": model.artifact_sha256,
            "tensor_payload_bytes": model.tensor_payload_bytes,
        },
        "format": "rawtherapee-rtnn-inspection-v1",
        "model": {
            "architecture": {
                "id": model.architecture_id,
                "symbol": model.architecture_symbol,
            },
            "id": model.model_id,
            "revision": model.model_revision,
            "semantic_schema_sha256": model.semantic_schema_sha256,
        },
        "summary": {
            "parameter_count": model.parameter_count,
            "scalar_type": "float32",
            "tensor_count": len(model.tensors),
        },
        "tensors": [
            {
                "byte_length": tensor.byte_length,
                "element_count": tensor.element_count,
                "id": tensor.id,
                "layout": tensor.layout,
                "payload_offset": tensor.payload_offset,
                "sha256": tensor.sha256,
                "shape": list(tensor.shape),
                "source_name": tensor.source_name,
                "symbol": tensor.symbol,
            }
            for tensor in model.tensors
        ],
    }

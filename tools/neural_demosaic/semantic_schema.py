"""Language-neutral RTNN semantic-schema loading and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .schema import (
    GHARBI_XTRANS_V1,
    GHARBI_XTRANS_V1_MANIFEST_SHA256,
    INSPECTION_MANIFEST_FORMAT,
)


SEMANTIC_SCHEMA_FORMAT = "rawtherapee-rtnn-semantic-schema-v1"
SEMANTIC_SCHEMA_SHA256 = "0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151"
SEMANTIC_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "demosaicnet-xtrans-v1.json"
)
MAX_SCHEMA_BYTES = 128 * 1024
MAX_INSPECTION_MANIFEST_BYTES = 1024 * 1024


class SemanticSchemaError(RuntimeError):
    """The semantic schema or its bound inspection manifest is invalid."""


@dataclass(frozen=True)
class SemanticTensorSpec:
    id: int
    symbol: str
    source_name: str
    dtype: str
    layout: str
    rank: int
    shape: tuple[int, ...]
    element_count: int
    sha256: str


@dataclass(frozen=True)
class SemanticSchema:
    format: str
    canonical_sha256: str
    architecture_id: int
    architecture_symbol: str
    model_id: str
    checkpoint_sha256: str
    inspection_manifest_format: str
    inspection_manifest_sha256: str
    tensors: tuple[SemanticTensorSpec, ...]
    parameter_count: int
    payload_bytes: int


EXPECTED_ARCHITECTURE = {
    "feature_channels": 64,
    "graph": {
        "execution_order": ["main", "skip", "post", "output"],
        "main": {
            "activation": "relu",
            "first_input_channels": 3,
            "kernel": [3, 3],
            "layers": 11,
            "output_channels": 64,
            "padding": 0,
            "remaining_input_channels": 64,
        },
        "output": {
            "input_channels": 64,
            "kernel": [1, 1],
            "output_channels": 3,
            "padding": 0,
        },
        "post": {
            "activation": "relu",
            "input_channels": 67,
            "kernel": [3, 3],
            "output_channels": 64,
            "padding": 0,
        },
        "skip": {
            "crop": "center",
            "input_channels": 3,
            "operation": "concatenate",
            "output_channels": 67,
            "source": "sparse_input",
        },
    },
    "id": 1,
    "input_channels": 3,
    "network_class": "XTransDemosaick",
    "output_channels": 3,
    "output_shrink": 24,
    "receptive_field_radius": 12,
    "symbol": "DEMOSAICNET_XTRANS_V1",
}

EXPECTED_BINDING = {
    "inspection_manifest_format": INSPECTION_MANIFEST_FORMAT,
    "inspection_manifest_sha256": GHARBI_XTRANS_V1_MANIFEST_SHA256,
    "model_id": GHARBI_XTRANS_V1.model_id,
    "upstream_checkpoint_sha256": GHARBI_XTRANS_V1.expected_sha256,
}

EXPECTED_SUMMARY = {
    "dtype": "float32",
    "parameter_count": GHARBI_XTRANS_V1.parameter_count,
    "payload_bytes": GHARBI_XTRANS_V1.payload_bytes,
    "tensor_count": len(GHARBI_XTRANS_V1.tensors),
}

TENSOR_FIELDS = {
    "dtype",
    "element_count",
    "id",
    "layout",
    "rank",
    "sha256",
    "shape",
    "source_name",
    "symbol",
}


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    """Encode schema or manifest JSON using the project's canonical form."""

    try:
        text = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SemanticSchemaError(f"document cannot be encoded canonically: {error}") from error

    return (text + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise SemanticSchemaError(f"duplicate JSON key: {key}")

        result[key] = value

    return result


def _reject_nonfinite_number(value: str) -> None:
    raise SemanticSchemaError(f"non-finite JSON number: {value}")


def parse_canonical_json(data: bytes, label: str) -> dict[str, Any]:
    """Parse strict canonical JSON, rejecting duplicate keys and alternate encodings."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SemanticSchemaError(f"{label} is not valid UTF-8") from error

    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    except json.JSONDecodeError as error:
        raise SemanticSchemaError(f"{label} is not valid JSON: {error}") from error

    if type(document) is not dict:
        raise SemanticSchemaError(f"{label} root must be an object")

    if canonical_json_bytes(document) != data:
        raise SemanticSchemaError(f"{label} is not in canonical JSON form")

    return document


def _read_bounded(path: Path, maximum_size: int, label: str) -> bytes:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size

            if size > maximum_size:
                raise SemanticSchemaError(
                    f"{label} is {size} bytes; maximum is {maximum_size}"
                )

            data = stream.read(maximum_size + 1)
    except OSError as error:
        raise SemanticSchemaError(f"cannot read {label}: {error}") from error

    if len(data) != size:
        raise SemanticSchemaError(f"{label} changed while it was being read")

    return data


def _require_exact(actual: Any, expected: Any, path: str) -> None:
    if type(expected) is dict:
        if type(actual) is not dict:
            raise SemanticSchemaError(f"{path} must be an object")

        actual_keys = set(actual)
        expected_keys = set(expected)

        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            additional = sorted(actual_keys - expected_keys)
            details = []

            if missing:
                details.append("missing " + ", ".join(missing))

            if additional:
                details.append("unknown " + ", ".join(additional))

            raise SemanticSchemaError(f"{path} fields differ: " + "; ".join(details))

        for key in sorted(expected):
            _require_exact(actual[key], expected[key], f"{path}.{key}")

        return

    if type(expected) is list:
        if type(actual) is not list:
            raise SemanticSchemaError(f"{path} must be an array")

        if len(actual) != len(expected):
            raise SemanticSchemaError(
                f"{path} has {len(actual)} entries; expected {len(expected)}"
            )

        for index, expected_value in enumerate(expected):
            _require_exact(actual[index], expected_value, f"{path}[{index}]")

        return

    if type(actual) is not type(expected) or actual != expected:
        raise SemanticSchemaError(f"{path} is {actual!r}; expected {expected!r}")


def _require_fields(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SemanticSchemaError(f"{path} must be an object")

    actual = set(value)

    if actual != fields:
        missing = sorted(fields - actual)
        additional = sorted(actual - fields)
        details = []

        if missing:
            details.append("missing " + ", ".join(missing))

        if additional:
            details.append("unknown " + ", ".join(additional))

        raise SemanticSchemaError(f"{path} fields differ: " + "; ".join(details))

    return value


def _require_sha256(value: Any, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticSchemaError(f"{path} must be a lowercase SHA-256 digest")

    return value


def _tensor_symbol(tensor_id: int) -> str:
    if tensor_id <= 22:
        layer = (tensor_id + 1) // 2
        kind = "WEIGHT" if tensor_id % 2 else "BIAS"
        return f"MAIN_CONV{layer}_{kind}"

    return {
        23: "POST_CONV_WEIGHT",
        24: "POST_CONV_BIAS",
        25: "OUTPUT_WEIGHT",
        26: "OUTPUT_BIAS",
    }[tensor_id]


def validate_semantic_schema_document(document: Mapping[str, Any]) -> SemanticSchema:
    """Validate the complete semantic contract independently of file encoding."""

    root = _require_fields(
        document,
        {"architecture", "binding", "format", "summary", "tensors"},
        "schema",
    )
    _require_exact(root["format"], SEMANTIC_SCHEMA_FORMAT, "schema.format")
    _require_exact(root["architecture"], EXPECTED_ARCHITECTURE, "schema.architecture")
    _require_exact(root["binding"], EXPECTED_BINDING, "schema.binding")
    _require_exact(root["summary"], EXPECTED_SUMMARY, "schema.summary")

    tensors = root["tensors"]

    if type(tensors) is not list:
        raise SemanticSchemaError("schema.tensors must be an array")

    if len(tensors) != len(GHARBI_XTRANS_V1.tensors):
        raise SemanticSchemaError(
            f"schema.tensors has {len(tensors)} entries; "
            f"expected {len(GHARBI_XTRANS_V1.tensors)}"
        )

    semantic_tensors: list[SemanticTensorSpec] = []
    ids: set[int] = set()
    symbols: set[str] = set()
    source_names: set[str] = set()

    for index, (entry_value, source_spec) in enumerate(
        zip(tensors, GHARBI_XTRANS_V1.tensors, strict=True)
    ):
        path = f"schema.tensors[{index}]"
        entry = _require_fields(entry_value, TENSOR_FIELDS, path)
        tensor_id = index + 1
        symbol = _tensor_symbol(tensor_id)
        expected_layout = source_spec.layout
        expected_shape = list(source_spec.shape)
        expected_rank = len(source_spec.shape)

        _require_exact(entry["id"], tensor_id, f"{path}.id")
        _require_exact(entry["symbol"], symbol, f"{path}.symbol")
        _require_exact(entry["source_name"], source_spec.name, f"{path}.source_name")
        _require_exact(entry["dtype"], "float32", f"{path}.dtype")
        _require_exact(entry["layout"], expected_layout, f"{path}.layout")
        _require_exact(entry["rank"], expected_rank, f"{path}.rank")
        _require_exact(entry["shape"], expected_shape, f"{path}.shape")
        _require_exact(
            entry["element_count"],
            source_spec.element_count,
            f"{path}.element_count",
        )
        tensor_digest = _require_sha256(entry["sha256"], f"{path}.sha256")

        if tensor_id in ids:
            raise SemanticSchemaError(f"duplicate semantic tensor ID: {tensor_id}")

        if symbol in symbols:
            raise SemanticSchemaError(f"duplicate semantic tensor symbol: {symbol}")

        if source_spec.name in source_names:
            raise SemanticSchemaError(f"duplicate source tensor name: {source_spec.name}")

        ids.add(tensor_id)
        symbols.add(symbol)
        source_names.add(source_spec.name)
        semantic_tensors.append(
            SemanticTensorSpec(
                id=tensor_id,
                symbol=symbol,
                source_name=source_spec.name,
                dtype="float32",
                layout=expected_layout,
                rank=expected_rank,
                shape=source_spec.shape,
                element_count=source_spec.element_count,
                sha256=tensor_digest,
            )
        )

    canonical = canonical_json_bytes(root)

    return SemanticSchema(
        format=SEMANTIC_SCHEMA_FORMAT,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        architecture_id=1,
        architecture_symbol="DEMOSAICNET_XTRANS_V1",
        model_id=GHARBI_XTRANS_V1.model_id,
        checkpoint_sha256=GHARBI_XTRANS_V1.expected_sha256,
        inspection_manifest_format=INSPECTION_MANIFEST_FORMAT,
        inspection_manifest_sha256=GHARBI_XTRANS_V1_MANIFEST_SHA256,
        tensors=tuple(semantic_tensors),
        parameter_count=GHARBI_XTRANS_V1.parameter_count,
        payload_bytes=GHARBI_XTRANS_V1.payload_bytes,
    )


def load_semantic_schema(path: str | Path = SEMANTIC_SCHEMA_PATH) -> SemanticSchema:
    """Load and authenticate the tracked canonical semantic schema."""

    schema_path = Path(path)
    data = _read_bounded(schema_path, MAX_SCHEMA_BYTES, "semantic schema")
    digest = hashlib.sha256(data).hexdigest()

    if digest != SEMANTIC_SCHEMA_SHA256:
        raise SemanticSchemaError(
            f"semantic schema SHA-256 is {digest}; expected {SEMANTIC_SCHEMA_SHA256}"
        )

    document = parse_canonical_json(data, "semantic schema")
    schema = validate_semantic_schema_document(document)

    if schema.canonical_sha256 != digest:
        raise SemanticSchemaError("semantic schema canonical digest is inconsistent")

    return schema


def _validate_inspection_manifest_structure(
    manifest: Mapping[str, Any],
    schema: SemanticSchema,
) -> None:
    root = _require_fields(
        manifest,
        {"format", "model", "source", "summary", "tensors"},
        "manifest",
    )
    _require_exact(root["format"], schema.inspection_manifest_format, "manifest.format")

    expected_model = {
        "architecture": {
            "convolution_padding": GHARBI_XTRANS_V1.convolution_padding,
            "depth": GHARBI_XTRANS_V1.architecture_depth,
            "name": GHARBI_XTRANS_V1.architecture_name,
            "width": GHARBI_XTRANS_V1.architecture_width,
        },
        "id": GHARBI_XTRANS_V1.model_id,
        "upstream": {
            "checkpoint_path": GHARBI_XTRANS_V1.upstream_checkpoint_path,
            "license": GHARBI_XTRANS_V1.upstream_license,
            "repository": GHARBI_XTRANS_V1.upstream_repository,
            "revision": GHARBI_XTRANS_V1.upstream_revision,
        },
    }
    _require_exact(root["model"], expected_model, "manifest.model")
    _require_exact(
        root["source"],
        {
            "sha256": schema.checkpoint_sha256,
            "size_bytes": GHARBI_XTRANS_V1.expected_file_size,
        },
        "manifest.source",
    )
    _require_exact(root["summary"], EXPECTED_SUMMARY, "manifest.summary")

    tensors = root["tensors"]

    if type(tensors) is not list:
        raise SemanticSchemaError("manifest.tensors must be an array")

    if len(tensors) != len(schema.tensors):
        raise SemanticSchemaError(
            f"manifest.tensors has {len(tensors)} entries; expected {len(schema.tensors)}"
        )

    manifest_tensor_fields = {
        "dtype",
        "element_count",
        "layout",
        "name",
        "payload_bytes",
        "rank",
        "sha256",
        "shape",
    }

    for index, (entry_value, semantic_tensor) in enumerate(
        zip(tensors, schema.tensors, strict=True)
    ):
        path = f"manifest.tensors[{index}]"
        entry = _require_fields(entry_value, manifest_tensor_fields, path)
        expected = {
            "dtype": semantic_tensor.dtype,
            "element_count": semantic_tensor.element_count,
            "layout": semantic_tensor.layout,
            "name": semantic_tensor.source_name,
            "payload_bytes": semantic_tensor.element_count * 4,
            "rank": semantic_tensor.rank,
            "sha256": semantic_tensor.sha256,
            "shape": list(semantic_tensor.shape),
        }
        _require_exact(entry, expected, path)


def validate_inspection_manifest(
    schema: SemanticSchema,
    manifest: Mapping[str, Any],
) -> str:
    """Cross-check an inspection manifest and return its canonical SHA-256."""

    _validate_inspection_manifest_structure(manifest, schema)
    digest = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()

    if digest != schema.inspection_manifest_sha256:
        raise SemanticSchemaError(
            f"inspection manifest SHA-256 is {digest}; "
            f"expected {schema.inspection_manifest_sha256}"
        )

    return digest


def validate_inspection_manifest_file(
    schema: SemanticSchema,
    path: str | Path,
) -> str:
    """Load a canonical Phase 1 manifest and validate its semantic binding."""

    data = _read_bounded(
        Path(path),
        MAX_INSPECTION_MANIFEST_BYTES,
        "inspection manifest",
    )
    manifest = parse_canonical_json(data, "inspection manifest")
    return validate_inspection_manifest(schema, manifest)

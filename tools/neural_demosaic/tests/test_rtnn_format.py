from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from tools.neural_demosaic.convert_checkpoint import (
    ConversionResult,
    convert_checkpoint,
    publish_conversion,
)
from tools.neural_demosaic.inspect_checkpoint import (
    canonical_manifest_bytes,
    inspect_checkpoint,
)
from tools.neural_demosaic.rtnn_format import (
    CONVERSION_MANIFEST_FORMAT,
    RTNN_ALIGNMENT,
    RTNN_ARCHITECTURE_DEMOSAICNET_XTRANS_V1,
    RTNN_DIRECTORY_RECORD_SIZE,
    RTNN_ENDIAN_MARKER,
    RTNN_EXPECTED_FILE_SIZE,
    RTNN_EXPECTED_PAYLOAD_REGION_BYTES,
    RTNN_EXPECTED_TENSOR_BYTES,
    RTNN_FLAGS,
    RTNN_FORMAT_MAJOR,
    RTNN_FORMAT_MINOR,
    RTNN_GHARBI_XTRANS_V1_MANIFEST_SHA256,
    RTNN_GHARBI_XTRANS_V1_PAYLOAD_SHA256,
    RTNN_GHARBI_XTRANS_V1_SHA256,
    RTNN_HEADER_SIZE,
    RTNN_LAYOUT_OIHW,
    RTNN_LAYOUT_VECTOR,
    RTNN_MAGIC,
    RTNN_MODEL_REVISION_GHARBI_XTRANS_V1,
    RTNN_SCALAR_FLOAT32,
    RTNN_TENSOR_COUNT,
    RTNNError,
    _checked_add,
    _checked_multiply,
    build_conversion_manifest,
    canonical_conversion_manifest_bytes,
    serialize_rtnn,
)
from tools.neural_demosaic.schema import (
    GHARBI_XTRANS_V1_MANIFEST_SHA256,
)
from tools.neural_demosaic.semantic_schema import (
    SEMANTIC_SCHEMA_PATH,
    SEMANTIC_SCHEMA_SHA256,
    SemanticSchema,
    load_semantic_schema,
)


def synthetic_binding() -> tuple[SemanticSchema, tuple[bytes, ...]]:
    schema = load_semantic_schema()
    payloads = tuple(b"\0" * (tensor.element_count * 4) for tensor in schema.tensors)
    tensors = tuple(
        replace(tensor, sha256=hashlib.sha256(payload).hexdigest())
        for tensor, payload in zip(schema.tensors, payloads, strict=True)
    )
    return replace(schema, tensors=tensors), payloads


@pytest.fixture(scope="module")
def synthetic_artifact():
    schema, payloads = synthetic_binding()
    return schema, payloads, serialize_rtnn(schema, payloads)


def unpack_header(data: bytes) -> tuple:
    return struct.unpack_from("<8sHH9I6Q", data, 0)


def unpack_record(data: bytes, index: int) -> tuple:
    offset = RTNN_HEADER_SIZE + index * RTNN_DIRECTORY_RECORD_SIZE
    return struct.unpack_from("<IHHII4IQQQ32sQ", data, offset)


def test_header_layout_and_little_endian_encoding(synthetic_artifact) -> None:
    schema, _, artifact = synthetic_artifact
    header = unpack_header(artifact.data)

    assert header == (
        RTNN_MAGIC,
        RTNN_FORMAT_MAJOR,
        RTNN_FORMAT_MINOR,
        RTNN_HEADER_SIZE,
        RTNN_ENDIAN_MARKER,
        RTNN_FLAGS,
        RTNN_ARCHITECTURE_DEMOSAICNET_XTRANS_V1,
        RTNN_MODEL_REVISION_GHARBI_XTRANS_V1,
        RTNN_SCALAR_FLOAT32,
        RTNN_TENSOR_COUNT,
        RTNN_DIRECTORY_RECORD_SIZE,
        0,
        RTNN_HEADER_SIZE,
        RTNN_TENSOR_COUNT * RTNN_DIRECTORY_RECORD_SIZE,
        2688,
        RTNN_EXPECTED_PAYLOAD_REGION_BYTES,
        RTNN_EXPECTED_TENSOR_BYTES,
        RTNN_EXPECTED_FILE_SIZE,
    )
    assert artifact.data[:8] == b"RTNN\r\n\x1a\n"
    assert artifact.data[8:12] == b"\x01\0\0\0"
    assert artifact.data[16:20] == b"\x04\x03\x02\x01"
    assert artifact.data[96:128] == bytes.fromhex(schema.canonical_sha256)
    assert artifact.data[128:160] == bytes.fromhex(artifact.payload_sha256)
    assert artifact.data[160:192] == bytes.fromhex(schema.checkpoint_sha256)


def test_directory_records_are_canonical(synthetic_artifact) -> None:
    schema, _, artifact = synthetic_artifact

    for index, (tensor, metadata) in enumerate(
        zip(schema.tensors, artifact.records, strict=True)
    ):
        record = unpack_record(artifact.data, index)
        expected_dimensions = tensor.shape + (0,) * (4 - tensor.rank)

        assert record[0] == index + 1
        assert record[1] == tensor.rank
        assert record[2] == (
            RTNN_LAYOUT_VECTOR if tensor.layout == "vector" else RTNN_LAYOUT_OIHW
        )
        assert record[3] == RTNN_SCALAR_FLOAT32
        assert record[4] == RTNN_FLAGS
        assert record[5:9] == expected_dimensions
        assert record[9] == tensor.element_count
        assert record[10] == tensor.element_count * 4
        assert record[11] == metadata.payload_offset
        assert record[12] == bytes.fromhex(tensor.sha256)
        assert record[13] == 0


def test_payload_alignment_padding_and_digests(synthetic_artifact) -> None:
    schema, payloads, artifact = synthetic_artifact
    payload_base = unpack_header(artifact.data)[14]
    payload_region = artifact.data[payload_base:]
    previous_end = 0

    assert payload_base % RTNN_ALIGNMENT == 0
    assert len(payload_region) == RTNN_EXPECTED_PAYLOAD_REGION_BYTES
    assert hashlib.sha256(payload_region).hexdigest() == artifact.payload_sha256

    for tensor, payload, record in zip(
        schema.tensors,
        payloads,
        artifact.records,
        strict=True,
    ):
        assert record.payload_offset % RTNN_ALIGNMENT == 0
        assert payload_region[previous_end:record.payload_offset] == (
            b"\0" * (record.payload_offset - previous_end)
        )
        tensor_bytes = payload_region[
            record.payload_offset : record.payload_offset + record.byte_length
        ]
        assert tensor_bytes == payload
        assert hashlib.sha256(tensor_bytes).hexdigest() == tensor.sha256
        previous_end = record.payload_offset + record.byte_length

    assert payload_region[previous_end:] == b"\0" * (len(payload_region) - previous_end)
    assert len(artifact.data) == RTNN_EXPECTED_FILE_SIZE


def test_conversion_manifest_is_canonical_and_machine_independent(
    synthetic_artifact,
) -> None:
    schema, _, artifact = synthetic_artifact
    manifest = build_conversion_manifest(schema, artifact)
    payload = canonical_conversion_manifest_bytes(manifest)

    assert manifest["format"] == CONVERSION_MANIFEST_FORMAT
    assert manifest["artifact"]["sha256"] == hashlib.sha256(artifact.data).hexdigest()
    assert manifest["artifact"]["payload_sha256"] == artifact.payload_sha256
    assert manifest["model"]["semantic_schema_sha256"] == schema.canonical_sha256
    assert manifest["source"]["inspection_manifest"]["sha256"] == (
        GHARBI_XTRANS_V1_MANIFEST_SHA256
    )
    assert len(manifest["tensors"]) == RTNN_TENSOR_COUNT
    assert payload == canonical_conversion_manifest_bytes(json.loads(payload))
    assert b"timestamp" not in payload.lower()
    assert b"python" not in payload.lower()
    assert b"/tmp/" not in payload


@pytest.mark.parametrize(
    "mutation",
    [
        lambda schema: replace(schema, architecture_id=2),
        lambda schema: replace(schema, architecture_symbol="OTHER_ARCHITECTURE"),
        lambda schema: replace(schema, model_id="other-model"),
        lambda schema: replace(schema, canonical_sha256="0" * 64),
        lambda schema: replace(schema, checkpoint_sha256="0" * 64),
        lambda schema: replace(schema, tensors=schema.tensors[:-1]),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], id=2),) + schema.tensors[1:],
        ),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], symbol="WRONG"),) + schema.tensors[1:],
        ),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], source_name="wrong.weight"),)
            + schema.tensors[1:],
        ),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], dtype="float16"),) + schema.tensors[1:],
        ),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], layout="vector"),) + schema.tensors[1:],
        ),
        lambda schema: replace(
            schema,
            tensors=(replace(schema.tensors[0], shape=(1728,), rank=1),)
            + schema.tensors[1:],
        ),
        lambda schema: replace(schema, parameter_count=schema.parameter_count - 1),
        lambda schema: replace(schema, payload_bytes=schema.payload_bytes - 4),
    ],
)
def test_schema_mutations_are_rejected(mutation) -> None:
    schema, payloads = synthetic_binding()

    with pytest.raises(RTNNError):
        serialize_rtnn(mutation(schema), payloads)


def test_payload_count_length_digest_and_nonfinite_values_are_rejected() -> None:
    schema, payloads = synthetic_binding()

    with pytest.raises(RTNNError, match="payloads"):
        serialize_rtnn(schema, payloads[:-1])

    shortened = (payloads[0][:-4],) + payloads[1:]

    with pytest.raises(RTNNError, match="payload has"):
        serialize_rtnn(schema, shortened)

    changed = (b"\x01" + payloads[0][1:],) + payloads[1:]

    with pytest.raises(RTNNError, match="SHA-256"):
        serialize_rtnn(schema, changed)

    nonfinite_payload = struct.pack("<f", float("nan")) + payloads[0][4:]
    nonfinite_tensor = replace(
        schema.tensors[0],
        sha256=hashlib.sha256(nonfinite_payload).hexdigest(),
    )
    nonfinite_schema = replace(
        schema,
        tensors=(nonfinite_tensor,) + schema.tensors[1:],
    )

    with pytest.raises(RTNNError, match="NaN or infinity"):
        serialize_rtnn(nonfinite_schema, (nonfinite_payload,) + payloads[1:])


def test_checked_arithmetic_rejects_uint64_overflow() -> None:
    maximum = (1 << 64) - 1

    with pytest.raises(RTNNError, match="overflows"):
        _checked_add(maximum, 1, "test addition")

    with pytest.raises(RTNNError, match="overflows"):
        _checked_multiply(maximum, 2, "test multiplication")


def conversion_result(synthetic_artifact) -> ConversionResult:
    schema, _, artifact = synthetic_artifact
    manifest = build_conversion_manifest(schema, artifact)
    return ConversionResult(
        artifact=artifact,
        manifest=manifest,
        manifest_bytes=canonical_conversion_manifest_bytes(manifest),
    )


def test_publication_refuses_existing_outputs_and_force_replaces(
    synthetic_artifact,
    tmp_path: Path,
) -> None:
    result = conversion_result(synthetic_artifact)
    output = tmp_path / "model.rtnn"
    manifest = Path(str(output) + ".json")
    output.write_bytes(b"old model")
    manifest.write_bytes(b"old manifest")

    with pytest.raises(RTNNError, match="already exists"):
        publish_conversion(output, result)

    assert output.read_bytes() == b"old model"
    assert manifest.read_bytes() == b"old manifest"

    publish_conversion(output, result, force=True)
    assert output.read_bytes() == result.artifact.data
    assert manifest.read_bytes() == result.manifest_bytes


def test_publication_cleans_temporary_files_after_failure(
    synthetic_artifact,
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = conversion_result(synthetic_artifact)
    output = tmp_path / "model.rtnn"

    def fail_replace(source, destination):
        raise OSError("injected publication failure")

    monkeypatch.setattr("tools.neural_demosaic.convert_checkpoint.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        publish_conversion(output, result)

    assert list(tmp_path.iterdir()) == []


def test_pinned_checkpoint_conversion_is_reproducible(tmp_path: Path) -> None:
    checkpoint = os.environ.get("GHARBI_XTRANS_CHECKPOINT")

    if not checkpoint:
        pytest.skip("GHARBI_XTRANS_CHECKPOINT is not set")

    first = convert_checkpoint(checkpoint)
    second = convert_checkpoint(checkpoint)

    assert first.artifact.data == second.artifact.data
    assert first.manifest_bytes == second.manifest_bytes
    assert len(first.artifact.records) == RTNN_TENSOR_COUNT
    assert first.artifact.tensor_payload_bytes == RTNN_EXPECTED_TENSOR_BYTES
    assert first.artifact.payload_region_bytes == RTNN_EXPECTED_PAYLOAD_REGION_BYTES
    assert len(first.artifact.data) == RTNN_EXPECTED_FILE_SIZE
    assert first.artifact.payload_sha256 == RTNN_GHARBI_XTRANS_V1_PAYLOAD_SHA256
    assert first.artifact.sha256 == RTNN_GHARBI_XTRANS_V1_SHA256
    assert hashlib.sha256(first.manifest_bytes).hexdigest() == (
        RTNN_GHARBI_XTRANS_V1_MANIFEST_SHA256
    )

    header = unpack_header(first.artifact.data)
    payload_base = header[14]

    for index, tensor in enumerate(load_semantic_schema().tensors):
        record = unpack_record(first.artifact.data, index)
        tensor_bytes = first.artifact.data[
            payload_base + record[11] : payload_base + record[11] + record[10]
        ]
        assert hashlib.sha256(tensor_bytes).hexdigest() == tensor.sha256

    phase1 = canonical_manifest_bytes(inspect_checkpoint(checkpoint))
    assert hashlib.sha256(phase1).hexdigest() == GHARBI_XTRANS_V1_MANIFEST_SHA256
    assert hashlib.sha256(SEMANTIC_SCHEMA_PATH.read_bytes()).hexdigest() == (
        SEMANTIC_SCHEMA_SHA256
    )

    first_output = tmp_path / "first" / "model.rtnn"
    second_output = tmp_path / "second" / "model.rtnn"
    first_output.parent.mkdir()
    second_output.parent.mkdir()
    publish_conversion(first_output, first)
    publish_conversion(second_output, second)
    assert first_output.read_bytes() == second_output.read_bytes()
    assert Path(str(first_output) + ".json").read_bytes() == Path(
        str(second_output) + ".json"
    ).read_bytes()

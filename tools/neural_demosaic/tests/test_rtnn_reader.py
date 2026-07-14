from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct

import pytest

from tools.neural_demosaic.convert_checkpoint import convert_checkpoint
from tools.neural_demosaic.inspect_rtnn import main as inspect_main
from tools.neural_demosaic.rtnn_reader import (
    MAX_FILE_BYTES,
    MAX_PAYLOAD_BYTES,
    MAX_TENSOR_COUNT,
    MAX_TENSOR_ELEMENTS,
    ModelBinding,
    RTNNErrorCode,
    RTNNReadError,
    _checked_add,
    _checked_multiply,
    _parse_rtnn,
    inspection_document,
    parse_rtnn,
    read_rtnn,
    reviewed_binding,
)
import tools.neural_demosaic.rtnn_reader as reader_module
from tools.neural_demosaic.semantic_schema import SemanticSchemaError


HEADER_SIZE = 192
RECORD_SIZE = 96
PAYLOAD_OFFSET = 2688
ALIGNMENT = 64


def put_uint(data: bytearray, offset: int, size: int, value: int) -> None:
    data[offset : offset + size] = value.to_bytes(size, "little", signed=False)


def get_uint(data: bytes | bytearray, offset: int, size: int) -> int:
    return int.from_bytes(data[offset : offset + size], "little", signed=False)


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class SyntheticFixture:
    data: bytes
    binding: ModelBinding


def make_synthetic_fixture() -> SyntheticFixture:
    reviewed = reviewed_binding()
    payload = bytearray()
    tensor_bindings = []
    records = []
    tensor_payload_bytes = 0

    for tensor in reviewed.tensors:
        relative_offset = align_up(len(payload))
        payload.extend(b"\0" * (relative_offset - len(payload)))
        tensor_data = b"\0" * (tensor.element_count * 4)
        digest = hashlib.sha256(tensor_data).hexdigest()
        payload.extend(tensor_data)
        tensor_bindings.append(replace(tensor, sha256=digest))
        records.append((tensor, relative_offset, tensor_data, digest))
        tensor_payload_bytes += len(tensor_data)

    payload.extend(b"\0" * (align_up(len(payload)) - len(payload)))
    schema_digest = "11" * 32
    checkpoint_digest = "22" * 32
    directory_size = len(records) * RECORD_SIZE
    file_size = PAYLOAD_OFFSET + len(payload)
    header = bytearray(HEADER_SIZE)
    header[0:8] = b"RTNN\r\n\x1a\n"
    put_uint(header, 8, 2, 1)
    put_uint(header, 10, 2, 0)
    put_uint(header, 12, 4, HEADER_SIZE)
    put_uint(header, 16, 4, 0x01020304)
    put_uint(header, 20, 4, 0)
    put_uint(header, 24, 4, reviewed.architecture_id)
    put_uint(header, 28, 4, reviewed.model_revision)
    put_uint(header, 32, 4, 1)
    put_uint(header, 36, 4, len(records))
    put_uint(header, 40, 4, RECORD_SIZE)
    put_uint(header, 44, 4, 0)
    put_uint(header, 48, 8, HEADER_SIZE)
    put_uint(header, 56, 8, directory_size)
    put_uint(header, 64, 8, PAYLOAD_OFFSET)
    put_uint(header, 72, 8, len(payload))
    put_uint(header, 80, 8, tensor_payload_bytes)
    put_uint(header, 88, 8, file_size)
    header[96:128] = bytes.fromhex(schema_digest)
    header[128:160] = hashlib.sha256(payload).digest()
    header[160:192] = bytes.fromhex(checkpoint_digest)
    directory = bytearray(directory_size)

    for index, (tensor, relative_offset, tensor_data, digest) in enumerate(records):
        offset = index * RECORD_SIZE
        put_uint(directory, offset, 4, tensor.id)
        put_uint(directory, offset + 4, 2, tensor.rank)
        put_uint(directory, offset + 6, 2, tensor.layout_id)
        put_uint(directory, offset + 8, 4, 1)
        put_uint(directory, offset + 12, 4, 0)
        dimensions = tensor.shape + (0,) * (4 - tensor.rank)

        for dimension_index, dimension in enumerate(dimensions):
            put_uint(directory, offset + 16 + 4 * dimension_index, 4, dimension)

        put_uint(directory, offset + 32, 8, tensor.element_count)
        put_uint(directory, offset + 40, 8, len(tensor_data))
        put_uint(directory, offset + 48, 8, relative_offset)
        directory[offset + 56 : offset + 88] = bytes.fromhex(digest)
        put_uint(directory, offset + 88, 8, 0)

    data = bytes(header + directory + payload)
    binding = replace(
        reviewed,
        semantic_schema_sha256=schema_digest,
        checkpoint_sha256=checkpoint_digest,
        artifact_sha256=hashlib.sha256(data).hexdigest(),
        tensors=tuple(tensor_bindings),
    )
    return SyntheticFixture(data=data, binding=binding)


@pytest.fixture(scope="module")
def synthetic_fixture() -> SyntheticFixture:
    return make_synthetic_fixture()


def assert_rejected(
    data: bytes | bytearray,
    binding: ModelBinding,
    code: RTNNErrorCode,
) -> None:
    with pytest.raises(RTNNReadError) as caught:
        _parse_rtnn(bytes(data), binding)

    assert caught.value.code is code


def refresh_payload_digest(data: bytearray) -> None:
    payload_offset = get_uint(data, 64, 8)
    payload_size = get_uint(data, 72, 8)
    data[128:160] = hashlib.sha256(
        data[payload_offset : payload_offset + payload_size]
    ).digest()


def test_reader_is_independent_from_writer_module() -> None:
    source = inspect.getsource(reader_module)
    assert "rtnn_format" not in source
    assert "convert_checkpoint" not in source


def test_synthetic_fixture_parses_to_immutable_canonical_tensors(
    synthetic_fixture: SyntheticFixture,
) -> None:
    model = _parse_rtnn(synthetic_fixture.data, synthetic_fixture.binding)

    assert model.file_size == len(synthetic_fixture.data)
    assert model.architecture_id == 1
    assert model.model_revision == 1
    assert model.parameter_count == 409_923
    assert len(model.tensors) == 26
    assert [tensor.id for tensor in model.tensors] == list(range(1, 27))
    assert all(type(tensor.data) is bytes for tensor in model.tensors)
    assert model.tensor(1) is model.tensors[0]

    with pytest.raises(KeyError):
        model.tensor(0)

    with pytest.raises(KeyError):
        model.tensor(27)


def test_public_parser_rejects_unreviewed_synthetic_binding(
    synthetic_fixture: SyntheticFixture,
) -> None:
    assert_rejected(
        synthetic_fixture.data,
        reviewed_binding(),
        RTNNErrorCode.SCHEMA,
    )

    with pytest.raises(RTNNReadError) as caught:
        parse_rtnn(synthetic_fixture.data)

    assert caught.value.code is RTNNErrorCode.SCHEMA


def test_reviewed_binding_maps_schema_authentication_failure(
    monkeypatch,
) -> None:
    def fail_schema_load():
        raise SemanticSchemaError("injected schema failure")

    reviewed_binding.cache_clear()
    monkeypatch.setattr(reader_module, "load_semantic_schema", fail_schema_load)

    try:
        with pytest.raises(RTNNReadError) as caught:
            reviewed_binding()

        assert caught.value.code is RTNNErrorCode.SCHEMA
    finally:
        reviewed_binding.cache_clear()


@pytest.mark.parametrize("length", [0, 7, 191])
def test_truncated_header_is_rejected(length: int) -> None:
    fixture = make_synthetic_fixture()
    assert_rejected(fixture.data[:length], fixture.binding, RTNNErrorCode.SIZE)


@pytest.mark.parametrize(
    ("offset", "size", "value", "code"),
    [
        (8, 2, 2, RTNNErrorCode.VERSION),
        (10, 2, 1, RTNNErrorCode.VERSION),
        (12, 4, 191, RTNNErrorCode.SIZE),
        (16, 4, 0x04030201, RTNNErrorCode.ENDIAN),
        (20, 4, 1, RTNNErrorCode.FLAGS),
        (24, 4, 0, RTNNErrorCode.ENUM),
        (24, 4, 2, RTNNErrorCode.SCHEMA),
        (28, 4, 0, RTNNErrorCode.ENUM),
        (28, 4, 2, RTNNErrorCode.SCHEMA),
        (32, 4, 0, RTNNErrorCode.ENUM),
        (36, 4, MAX_TENSOR_COUNT + 1, RTNNErrorCode.LIMIT),
        (36, 4, 25, RTNNErrorCode.SCHEMA),
        (40, 4, 95, RTNNErrorCode.SIZE),
        (44, 4, 1, RTNNErrorCode.RESERVED),
        (48, 8, 193, RTNNErrorCode.RANGE),
        (56, 8, 2495, RTNNErrorCode.RANGE),
        (64, 8, 2689, RTNNErrorCode.RANGE),
        (72, 8, MAX_PAYLOAD_BYTES + 1, RTNNErrorCode.LIMIT),
        (80, 8, 1, RTNNErrorCode.SCHEMA),
        (88, 8, 1, RTNNErrorCode.SIZE),
    ],
)
def test_header_field_mutations_are_rejected(
    synthetic_fixture: SyntheticFixture,
    offset: int,
    size: int,
    value: int,
    code: RTNNErrorCode,
) -> None:
    data = bytearray(synthetic_fixture.data)
    put_uint(data, offset, size, value)
    assert_rejected(data, synthetic_fixture.binding, code)


def test_magic_and_identity_digest_mutations_are_rejected(
    synthetic_fixture: SyntheticFixture,
) -> None:
    data = bytearray(synthetic_fixture.data)
    data[0] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.MAGIC)

    data = bytearray(synthetic_fixture.data)
    data[96] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.SCHEMA)

    data = bytearray(synthetic_fixture.data)
    data[160] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.SCHEMA)


@pytest.mark.parametrize(
    ("relative_offset", "size", "value", "code"),
    [
        (0, 4, 0, RTNNErrorCode.ORDER),
        (0, 4, 2, RTNNErrorCode.ORDER),
        (4, 2, 5, RTNNErrorCode.LIMIT),
        (4, 2, 2, RTNNErrorCode.ENUM),
        (6, 2, 0, RTNNErrorCode.ENUM),
        (8, 4, 0, RTNNErrorCode.ENUM),
        (12, 4, 1, RTNNErrorCode.FLAGS),
        (16, 4, 0, RTNNErrorCode.SCHEMA),
        (16, 4, 1_048_577, RTNNErrorCode.LIMIT),
        (32, 8, MAX_TENSOR_ELEMENTS + 1, RTNNErrorCode.LIMIT),
        (32, 8, 1, RTNNErrorCode.SCHEMA),
        (40, 8, 1, RTNNErrorCode.SCHEMA),
        (48, 8, 1, RTNNErrorCode.ALIGNMENT),
        (88, 8, 1, RTNNErrorCode.RESERVED),
    ],
)
def test_first_directory_record_mutations_are_rejected(
    synthetic_fixture: SyntheticFixture,
    relative_offset: int,
    size: int,
    value: int,
    code: RTNNErrorCode,
) -> None:
    data = bytearray(synthetic_fixture.data)
    put_uint(data, HEADER_SIZE + relative_offset, size, value)
    assert_rejected(data, synthetic_fixture.binding, code)


def test_unused_dimension_tensor_digest_and_record_order_are_checked(
    synthetic_fixture: SyntheticFixture,
) -> None:
    data = bytearray(synthetic_fixture.data)
    second_record = HEADER_SIZE + RECORD_SIZE
    put_uint(data, second_record + 20, 4, 1)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.RESERVED)

    data = bytearray(synthetic_fixture.data)
    data[HEADER_SIZE + 56] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.DIGEST)

    data = bytearray(synthetic_fixture.data)
    first = bytes(data[HEADER_SIZE : HEADER_SIZE + RECORD_SIZE])
    second = bytes(data[second_record : second_record + RECORD_SIZE])
    data[HEADER_SIZE : HEADER_SIZE + RECORD_SIZE] = second
    data[second_record : second_record + RECORD_SIZE] = first
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.ORDER)


def test_overlapping_and_nonminimal_tensor_offsets_are_rejected(
    synthetic_fixture: SyntheticFixture,
) -> None:
    second_offset_field = HEADER_SIZE + RECORD_SIZE + 48
    original_offset = get_uint(synthetic_fixture.data, second_offset_field, 8)
    data = bytearray(synthetic_fixture.data)
    put_uint(data, second_offset_field, 8, 0)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.RANGE)

    data = bytearray(synthetic_fixture.data)
    put_uint(data, second_offset_field, 8, original_offset + ALIGNMENT)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.ALIGNMENT)


def test_truncated_extended_and_nonminimal_payload_are_rejected(
    synthetic_fixture: SyntheticFixture,
) -> None:
    assert_rejected(
        synthetic_fixture.data[:-1],
        synthetic_fixture.binding,
        RTNNErrorCode.SIZE,
    )
    assert_rejected(
        synthetic_fixture.data + b"\0",
        synthetic_fixture.binding,
        RTNNErrorCode.SIZE,
    )

    data = bytearray(synthetic_fixture.data[:-1])
    put_uint(data, 72, 8, get_uint(data, 72, 8) - 1)
    put_uint(data, 88, 8, len(data))
    refresh_payload_digest(data)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.ALIGNMENT)


def test_payload_padding_and_digest_corruption_are_rejected(
    synthetic_fixture: SyntheticFixture,
) -> None:
    data = bytearray(synthetic_fixture.data)
    data[128] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.DIGEST)

    data = bytearray(synthetic_fixture.data)
    data[PAYLOAD_OFFSET] ^= 1
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.DIGEST)

    data = bytearray(synthetic_fixture.data)
    data[PAYLOAD_OFFSET] ^= 1
    refresh_payload_digest(data)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.DIGEST)

    data = bytearray(synthetic_fixture.data)
    data[-1] = 1
    refresh_payload_digest(data)
    assert_rejected(data, synthetic_fixture.binding, RTNNErrorCode.RESERVED)


def test_tensor_digest_nonfinite_and_complete_file_digest_are_independent(
    synthetic_fixture: SyntheticFixture,
) -> None:
    data = bytearray(synthetic_fixture.data)
    data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + 4] = struct.pack("<f", 1.0)
    first_length = get_uint(data, HEADER_SIZE + 40, 8)
    tensor_data = bytes(data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + first_length])
    digest = hashlib.sha256(tensor_data).hexdigest()
    data[HEADER_SIZE + 56 : HEADER_SIZE + 88] = bytes.fromhex(digest)
    refresh_payload_digest(data)
    tensor_binding = replace(synthetic_fixture.binding.tensors[0], sha256=digest)
    binding = replace(
        synthetic_fixture.binding,
        tensors=(tensor_binding,) + synthetic_fixture.binding.tensors[1:],
    )
    assert_rejected(data, binding, RTNNErrorCode.DIGEST)

    data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + 4] = struct.pack("<f", float("nan"))
    tensor_data = bytes(data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + first_length])
    digest = hashlib.sha256(tensor_data).hexdigest()
    data[HEADER_SIZE + 56 : HEADER_SIZE + 88] = bytes.fromhex(digest)
    refresh_payload_digest(data)
    tensor_binding = replace(synthetic_fixture.binding.tensors[0], sha256=digest)
    binding = replace(
        synthetic_fixture.binding,
        tensors=(tensor_binding,) + synthetic_fixture.binding.tensors[1:],
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )
    assert_rejected(data, binding, RTNNErrorCode.NONFINITE)


def test_checked_arithmetic_reports_range_errors() -> None:
    maximum = (1 << 64) - 1

    with pytest.raises(RTNNReadError) as addition:
        _checked_add(maximum, 1, "addition")

    assert addition.value.code is RTNNErrorCode.RANGE

    with pytest.raises(RTNNReadError) as multiplication:
        _checked_multiply(maximum, 2, "multiplication")

    assert multiplication.value.code is RTNNErrorCode.RANGE


def test_read_rtnn_reports_io_and_rejects_large_file_before_reading(
    tmp_path: Path,
) -> None:
    with pytest.raises(RTNNReadError) as missing:
        read_rtnn(tmp_path / "missing.rtnn")

    assert missing.value.code is RTNNErrorCode.IO

    oversized = tmp_path / "oversized.rtnn"

    with oversized.open("wb") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)

    with pytest.raises(RTNNReadError) as limit:
        read_rtnn(oversized)

    assert limit.value.code is RTNNErrorCode.LIMIT


def test_inspection_document_and_cli_are_canonical(
    synthetic_fixture: SyntheticFixture,
    monkeypatch,
    capsys,
) -> None:
    model = _parse_rtnn(synthetic_fixture.data, synthetic_fixture.binding)
    document = inspection_document(model)
    assert document["format"] == "rawtherapee-rtnn-inspection-v1"
    assert document["artifact"]["sha256"] == model.artifact_sha256
    assert document["summary"]["tensor_count"] == 26

    monkeypatch.setattr("tools.neural_demosaic.inspect_rtnn.read_rtnn", lambda path: model)
    assert inspect_main(["ignored.rtnn"]) == 0
    output = capsys.readouterr()
    parsed = json.loads(output.out)
    assert parsed == document
    assert output.out.endswith("\n")
    assert "/tmp/" not in output.out
    assert "timestamp" not in output.out.lower()
    assert output.err == ""


def test_inspection_cli_reports_stable_reader_error_code(tmp_path: Path, capsys) -> None:
    assert inspect_main([str(tmp_path / "missing.rtnn")]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "[IO]" in output.err


def test_reviewed_rtnn_matches_conversion_manifest_and_binding() -> None:
    checkpoint = os.environ.get("GHARBI_XTRANS_CHECKPOINT")

    if not checkpoint:
        pytest.skip("GHARBI_XTRANS_CHECKPOINT is not set")

    conversion = convert_checkpoint(checkpoint)
    model = parse_rtnn(conversion.artifact.data)
    inspection = inspection_document(model)
    manifest = conversion.manifest

    assert model.artifact_sha256 == reviewed_binding().artifact_sha256
    assert inspection["artifact"] == manifest["artifact"]
    assert inspection["model"] == manifest["model"]
    assert inspection["summary"] == manifest["summary"]
    assert inspection["tensors"] == manifest["tensors"]

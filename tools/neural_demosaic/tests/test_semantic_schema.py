from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.neural_demosaic.inspect_checkpoint import inspect_checkpoint
from tools.neural_demosaic.schema import (
    GHARBI_XTRANS_V1,
    GHARBI_XTRANS_V1_MANIFEST_SHA256,
)
from tools.neural_demosaic.semantic_schema import (
    SEMANTIC_SCHEMA_PATH,
    SEMANTIC_SCHEMA_SHA256,
    SemanticSchemaError,
    canonical_json_bytes,
    load_semantic_schema,
    parse_canonical_json,
    validate_inspection_manifest,
    validate_inspection_manifest_file,
    validate_semantic_schema_document,
)
from tools.neural_demosaic.validate_semantic_schema import main as validate_main


def schema_document() -> dict:
    return json.loads(SEMANTIC_SCHEMA_PATH.read_text(encoding="utf-8"))


def reference_manifest() -> dict:
    semantic_schema = load_semantic_schema()
    tensors = []

    for tensor in semantic_schema.tensors:
        tensors.append(
            {
                "dtype": tensor.dtype,
                "element_count": tensor.element_count,
                "layout": tensor.layout,
                "name": tensor.source_name,
                "payload_bytes": tensor.element_count * 4,
                "rank": tensor.rank,
                "sha256": tensor.sha256,
                "shape": list(tensor.shape),
            }
        )

    return {
        "format": semantic_schema.inspection_manifest_format,
        "model": {
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
        },
        "source": {
            "sha256": GHARBI_XTRANS_V1.expected_sha256,
            "size_bytes": GHARBI_XTRANS_V1.expected_file_size,
        },
        "summary": {
            "dtype": "float32",
            "parameter_count": GHARBI_XTRANS_V1.parameter_count,
            "payload_bytes": GHARBI_XTRANS_V1.payload_bytes,
            "tensor_count": len(GHARBI_XTRANS_V1.tensors),
        },
        "tensors": tensors,
    }


def test_tracked_schema_is_canonical_and_authenticated() -> None:
    data = SEMANTIC_SCHEMA_PATH.read_bytes()
    document = parse_canonical_json(data, "semantic schema")
    schema = load_semantic_schema()

    assert canonical_json_bytes(document) == data
    assert hashlib.sha256(data).hexdigest() == SEMANTIC_SCHEMA_SHA256
    assert schema.canonical_sha256 == SEMANTIC_SCHEMA_SHA256


def test_phase1_manifest_remains_exactly_bound() -> None:
    manifest = reference_manifest()
    payload = canonical_json_bytes(manifest)
    schema = load_semantic_schema()

    assert hashlib.sha256(payload).hexdigest() == GHARBI_XTRANS_V1_MANIFEST_SHA256
    assert validate_inspection_manifest(schema, manifest) == GHARBI_XTRANS_V1_MANIFEST_SHA256


def test_architecture_and_tensor_id_contract() -> None:
    schema = load_semantic_schema()

    assert schema.architecture_id == 1
    assert schema.architecture_symbol == "DEMOSAICNET_XTRANS_V1"
    assert [tensor.id for tensor in schema.tensors] == list(range(1, 27))
    assert len({tensor.symbol for tensor in schema.tensors}) == 26
    assert len({tensor.source_name for tensor in schema.tensors}) == 26

    for layer in range(1, 12):
        weight = schema.tensors[2 * layer - 2]
        bias = schema.tensors[2 * layer - 1]
        assert weight.id == 2 * layer - 1
        assert weight.symbol == f"MAIN_CONV{layer}_WEIGHT"
        assert bias.id == 2 * layer
        assert bias.symbol == f"MAIN_CONV{layer}_BIAS"

    assert [(tensor.id, tensor.symbol) for tensor in schema.tensors[22:]] == [
        (23, "POST_CONV_WEIGHT"),
        (24, "POST_CONV_BIAS"),
        (25, "OUTPUT_WEIGHT"),
        (26, "OUTPUT_BIAS"),
    ]
    assert sum(tensor.element_count for tensor in schema.tensors) == 409_923
    assert schema.payload_bytes == 1_639_692


def mutate_tensor(document: dict, index: int, key: str, value) -> None:
    document["tensors"][index][key] = value


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: mutate_tensor(document, 0, "id", 0),
        lambda document: mutate_tensor(document, 1, "id", 1),
        lambda document: mutate_tensor(document, 1, "id", 27),
        lambda document: document["tensors"].__setitem__(
            slice(0, 2),
            [document["tensors"][1], document["tensors"][0]],
        ),
        lambda document: mutate_tensor(
            document,
            1,
            "symbol",
            document["tensors"][0]["symbol"],
        ),
        lambda document: mutate_tensor(
            document,
            1,
            "source_name",
            document["tensors"][0]["source_name"],
        ),
        lambda document: mutate_tensor(document, 0, "source_name", "unknown.weight"),
        lambda document: mutate_tensor(document, 0, "dtype", "float16"),
        lambda document: mutate_tensor(document, 0, "layout", "HWIO"),
        lambda document: mutate_tensor(document, 0, "rank", 3),
        lambda document: mutate_tensor(document, 0, "shape", [64, 3, 9]),
        lambda document: mutate_tensor(document, 0, "element_count", 1729),
        lambda document: mutate_tensor(document, 0, "sha256", "0" * 64),
        lambda document: document["architecture"].__setitem__("id", 2),
        lambda document: document["architecture"].__setitem__("output_shrink", 22),
        lambda document: document["architecture"]["graph"]["main"].__setitem__(
            "padding", 1
        ),
        lambda document: document["binding"].__setitem__(
            "upstream_checkpoint_sha256", "0" * 64
        ),
        lambda document: document["binding"].__setitem__(
            "inspection_manifest_sha256", "0" * 64
        ),
        lambda document: document["summary"].__setitem__("parameter_count", 1),
        lambda document: document.__setitem__("unknown", True),
        lambda document: document["tensors"][0].__setitem__("unknown", True),
    ],
)
def test_semantic_schema_mutations_are_rejected(mutation) -> None:
    document = copy.deepcopy(schema_document())
    mutation(document)

    with pytest.raises(SemanticSchemaError):
        schema = validate_semantic_schema_document(document)
        validate_inspection_manifest(schema, reference_manifest())


def test_schema_parser_rejects_duplicate_keys_and_noncanonical_json() -> None:
    with pytest.raises(SemanticSchemaError, match="duplicate JSON key"):
        parse_canonical_json(b'{"format": 1, "format": 2}\n', "schema")

    noncanonical = json.dumps(schema_document(), separators=(",", ":")).encode("utf-8")

    with pytest.raises(SemanticSchemaError, match="canonical JSON form"):
        parse_canonical_json(noncanonical, "schema")


def test_schema_loader_rejects_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_bytes(SEMANTIC_SCHEMA_PATH.read_bytes() + b"\n")

    with pytest.raises(SemanticSchemaError, match="SHA-256"):
        load_semantic_schema(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["model"].__setitem__("id", "other-model"),
        lambda manifest: manifest["source"].__setitem__("sha256", "0" * 64),
        lambda manifest: manifest["summary"].__setitem__("tensor_count", 25),
        lambda manifest: manifest["tensors"].reverse(),
        lambda manifest: manifest["tensors"][0].__setitem__("name", "unknown.weight"),
        lambda manifest: manifest["tensors"][0].__setitem__("dtype", "float16"),
        lambda manifest: manifest["tensors"][0].__setitem__("layout", "HWIO"),
        lambda manifest: manifest["tensors"][0].__setitem__("shape", [64, 3, 9]),
        lambda manifest: manifest["tensors"][0].__setitem__("element_count", 1729),
        lambda manifest: manifest["tensors"][0].__setitem__("sha256", "0" * 64),
        lambda manifest: manifest.__setitem__("unknown", True),
    ],
)
def test_inspection_manifest_mutations_are_rejected(mutation) -> None:
    manifest = reference_manifest()
    mutation(manifest)

    with pytest.raises(SemanticSchemaError):
        validate_inspection_manifest(load_semantic_schema(), manifest)


def test_inspection_manifest_digest_binding_is_enforced() -> None:
    schema = replace(load_semantic_schema(), inspection_manifest_sha256="0" * 64)

    with pytest.raises(SemanticSchemaError, match="inspection manifest SHA-256"):
        validate_inspection_manifest(schema, reference_manifest())


def test_manifest_file_and_cli_validation(tmp_path: Path, capsys) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(reference_manifest()))
    schema = load_semantic_schema()

    assert validate_inspection_manifest_file(schema, path) == GHARBI_XTRANS_V1_MANIFEST_SHA256
    assert validate_main([str(path)]) == 0
    output = capsys.readouterr()
    assert "validated architecture 1 (DEMOSAICNET_XTRANS_V1)" in output.out
    assert SEMANTIC_SCHEMA_SHA256 in output.out
    assert GHARBI_XTRANS_V1_MANIFEST_SHA256 in output.out
    assert output.err == ""


def test_real_checkpoint_semantic_binding() -> None:
    checkpoint = os.environ.get("GHARBI_XTRANS_CHECKPOINT")

    if not checkpoint:
        pytest.skip("GHARBI_XTRANS_CHECKPOINT is not set")

    manifest = inspect_checkpoint(checkpoint)
    payload = canonical_json_bytes(manifest)
    assert hashlib.sha256(payload).hexdigest() == GHARBI_XTRANS_V1_MANIFEST_SHA256
    assert validate_inspection_manifest(load_semantic_schema(), manifest) == (
        GHARBI_XTRANS_V1_MANIFEST_SHA256
    )

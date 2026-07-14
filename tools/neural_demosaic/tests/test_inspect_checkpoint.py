from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tools.neural_demosaic.inspect_checkpoint import (
    CheckpointError,
    canonical_manifest_bytes,
    inspect_checkpoint,
    load_validated_checkpoint,
    write_manifest,
)
from tools.neural_demosaic.schema import (
    CheckpointSchema,
    GHARBI_XTRANS_V1,
    GHARBI_XTRANS_V1_MANIFEST_SHA256,
)


def make_state_dict() -> OrderedDict[str, torch.Tensor]:
    state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for index, spec in enumerate(GHARBI_XTRANS_V1.tensors):
        values = torch.arange(spec.element_count, dtype=torch.float32)
        state[spec.name] = (values.reshape(spec.shape) + index) / 1024.0

    return state


def schema_for_file(path: Path) -> CheckpointSchema:
    data = path.read_bytes()
    return replace(
        GHARBI_XTRANS_V1,
        expected_file_size=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
    )


def save_fixture(
    tmp_path: Path,
    state: object,
    name: str = "fixture.pth",
) -> tuple[Path, CheckpointSchema]:
    path = tmp_path / name
    torch.save(state, path)
    return path, schema_for_file(path)


@pytest.fixture
def valid_fixture(tmp_path: Path) -> tuple[Path, CheckpointSchema, OrderedDict[str, torch.Tensor]]:
    state = make_state_dict()
    path, schema = save_fixture(tmp_path, state)
    return path, schema, state


def test_valid_checkpoint_summary_and_tensor_digests(valid_fixture) -> None:
    path, schema, original = valid_fixture
    manifest = inspect_checkpoint(path, schema)

    assert manifest["summary"] == {
        "dtype": "float32",
        "parameter_count": 409_923,
        "payload_bytes": 1_639_692,
        "tensor_count": 26,
    }
    assert manifest["model"]["architecture"] == {
        "convolution_padding": 0,
        "depth": 11,
        "name": "XTransDemosaick",
        "width": 64,
    }

    assert len(manifest["tensors"]) == 26

    for entry in manifest["tensors"]:
        tensor = original[entry["name"]].detach().contiguous().numpy()
        independent_payload = np.asarray(tensor, dtype="<f4", order="C").tobytes(order="C")
        assert entry["sha256"] == hashlib.sha256(independent_payload).hexdigest()
        assert entry["payload_bytes"] == len(independent_payload)


def test_validated_checkpoint_exposes_the_same_canonical_tensor_bytes(
    valid_fixture,
) -> None:
    path, schema, original = valid_fixture
    validated = load_validated_checkpoint(path, schema)

    assert validated.manifest == inspect_checkpoint(path, schema)
    assert len(validated.tensor_payloads) == len(schema.tensors)

    for spec, payload in zip(schema.tensors, validated.tensor_payloads, strict=True):
        independent = (
            original[spec.name]
            .detach()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        assert payload == independent


def test_manifest_is_deterministic_and_has_no_local_path(valid_fixture, tmp_path: Path) -> None:
    path, schema, _ = valid_fixture
    first = canonical_manifest_bytes(inspect_checkpoint(path, schema))
    second = canonical_manifest_bytes(inspect_checkpoint(path, schema))

    assert first == second
    assert str(tmp_path).encode() not in first
    assert b"timestamp" not in first.lower()
    assert first == canonical_manifest_bytes(json.loads(first))


def test_checkpoint_is_loaded_without_importing_upstream(valid_fixture, monkeypatch) -> None:
    path, schema, _ = valid_fixture
    imported_upstream = False
    original_import = __import__

    def checked_import(name, *args, **kwargs):
        nonlocal imported_upstream

        if name == "demosaicnet" or name.startswith("demosaicnet."):
            imported_upstream = True
            raise AssertionError("upstream model code must not be imported")

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", checked_import)
    inspect_checkpoint(path, schema)
    assert not imported_upstream
    assert not any(name == "demosaicnet" or name.startswith("demosaicnet.") for name in sys.modules)


def test_loader_is_restricted_to_cpu_weights(valid_fixture, monkeypatch) -> None:
    path, schema, _ = valid_fixture
    original_load = torch.load
    call = {}

    def checked_load(source, *, map_location, weights_only):
        call["map_location"] = map_location
        call["weights_only"] = weights_only
        return original_load(
            source,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr("tools.neural_demosaic.inspect_checkpoint.torch.load", checked_load)
    inspect_checkpoint(path, schema)
    assert call == {"map_location": "cpu", "weights_only": True}


def test_wrong_hash_is_rejected_before_deserialization(valid_fixture, monkeypatch) -> None:
    path, schema, _ = valid_fixture
    wrong_schema = replace(schema, expected_sha256="0" * 64)

    def forbidden_load(*args, **kwargs):
        pytest.fail("a checkpoint with the wrong hash must not be deserialized")

    monkeypatch.setattr("tools.neural_demosaic.inspect_checkpoint.torch.load", forbidden_load)

    with pytest.raises(CheckpointError, match="SHA-256"):
        inspect_checkpoint(path, wrong_schema)


def test_wrong_size_is_rejected(valid_fixture) -> None:
    path, schema, _ = valid_fixture
    wrong_schema = replace(schema, expected_file_size=schema.expected_file_size + 1)

    with pytest.raises(CheckpointError, match="size"):
        inspect_checkpoint(path, wrong_schema)


def test_truncated_checkpoint_is_rejected(tmp_path: Path, valid_fixture) -> None:
    original_path, _, _ = valid_fixture
    truncated = tmp_path / "truncated.pth"
    truncated.write_bytes(original_path.read_bytes()[:128])
    schema = schema_for_file(truncated)

    with pytest.raises(CheckpointError, match="deserialization"):
        inspect_checkpoint(truncated, schema)


def test_non_mapping_checkpoint_is_rejected(tmp_path: Path) -> None:
    path, schema = save_fixture(tmp_path, [torch.zeros(1)])

    with pytest.raises(CheckpointError, match="expected a tensor mapping"):
        inspect_checkpoint(path, schema)


@pytest.mark.parametrize("change", ["missing", "additional"])
def test_inexact_keys_are_rejected(tmp_path: Path, change: str) -> None:
    state = make_state_dict()

    if change == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.weight"] = torch.zeros(1)

    path, schema = save_fixture(tmp_path, state)

    with pytest.raises(CheckpointError, match=change):
        inspect_checkpoint(path, schema)


def test_non_tensor_value_is_rejected(tmp_path: Path) -> None:
    state = make_state_dict()
    state[next(iter(state))] = 1.0
    path, schema = save_fixture(tmp_path, state)

    with pytest.raises(CheckpointError, match="expected torch.Tensor"):
        inspect_checkpoint(path, schema)


def test_wrong_shape_is_rejected(tmp_path: Path) -> None:
    state = make_state_dict()
    name = next(iter(state))
    state[name] = state[name].reshape(-1)
    path, schema = save_fixture(tmp_path, state)

    with pytest.raises(CheckpointError, match="shape"):
        inspect_checkpoint(path, schema)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (lambda tensor: tensor.to(torch.float64), "dtype"),
        (lambda tensor: tensor.to(torch.complex64), "complex"),
        (
            lambda tensor: torch.quantize_per_tensor(
                tensor,
                scale=0.1,
                zero_point=0,
                dtype=torch.qint8,
            ),
            "quantized",
        ),
        (lambda tensor: tensor.to_sparse(), "layout"),
    ],
)
@pytest.mark.filterwarnings("ignore:Sparse invariant checks are implicitly disabled")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
def test_unsupported_tensor_kinds_are_rejected(tmp_path: Path, replacement, message: str) -> None:
    state = make_state_dict()
    name = next(iter(state))
    state[name] = replacement(state[name])
    path, schema = save_fixture(tmp_path, state)

    with pytest.raises(CheckpointError, match=message):
        inspect_checkpoint(path, schema)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tensor_is_rejected(tmp_path: Path, non_finite: float) -> None:
    state = make_state_dict()
    name = next(iter(state))
    state[name][0, 0, 0, 0] = non_finite
    path, schema = save_fixture(tmp_path, state)

    with pytest.raises(CheckpointError, match="NaN or infinity"):
        inspect_checkpoint(path, schema)


def test_manifest_writer_does_not_replace_without_force(valid_fixture, tmp_path: Path) -> None:
    path, schema, _ = valid_fixture
    payload = canonical_manifest_bytes(inspect_checkpoint(path, schema))
    output = tmp_path / "manifest.json"
    write_manifest(output, payload)

    with pytest.raises(CheckpointError, match="already exists"):
        write_manifest(output, b"different\n")

    assert output.read_bytes() == payload
    write_manifest(output, b"replacement\n", force=True)
    assert output.read_bytes() == b"replacement\n"


def test_pinned_gharbi_checkpoint() -> None:
    checkpoint = os.environ.get("GHARBI_XTRANS_CHECKPOINT")

    if not checkpoint:
        pytest.skip("GHARBI_XTRANS_CHECKPOINT is not set")

    manifest = inspect_checkpoint(checkpoint)
    assert manifest["source"]["sha256"] == GHARBI_XTRANS_V1.expected_sha256
    assert manifest["source"]["size_bytes"] == GHARBI_XTRANS_V1.expected_file_size
    assert manifest["summary"]["tensor_count"] == 26
    assert manifest["summary"]["parameter_count"] == 409_923
    assert manifest["summary"]["payload_bytes"] == 1_639_692
    payload = canonical_manifest_bytes(manifest)
    assert hashlib.sha256(payload).hexdigest() == GHARBI_XTRANS_V1_MANIFEST_SHA256

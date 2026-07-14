#!/usr/bin/env python3
"""Safely inspect the pinned Gharbi X-Trans PyTorch checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

from .schema import (
    CheckpointSchema,
    GHARBI_XTRANS_V1,
    INSPECTION_MANIFEST_FORMAT,
    TensorSpec,
)


MANIFEST_FORMAT = INSPECTION_MANIFEST_FORMAT


class CheckpointError(RuntimeError):
    """The checkpoint is not the expected safe, pinned tensor dictionary."""


def _read_pinned_checkpoint(path: Path, schema: CheckpointSchema) -> tuple[bytes, str]:
    try:
        with path.open("rb") as stream:
            file_size = os.fstat(stream.fileno()).st_size

            if file_size != schema.expected_file_size:
                raise CheckpointError(
                    f"checkpoint size is {file_size} bytes; "
                    f"expected {schema.expected_file_size}"
                )

            data = stream.read(schema.expected_file_size + 1)
    except OSError as error:
        raise CheckpointError(f"cannot read checkpoint: {error}") from error

    if len(data) != schema.expected_file_size:
        raise CheckpointError("checkpoint changed while it was being read")

    digest = hashlib.sha256(data).hexdigest()

    if digest != schema.expected_sha256:
        raise CheckpointError(
            f"checkpoint SHA-256 is {digest}; expected {schema.expected_sha256}"
        )

    return data, digest


def _load_weights_only(data: bytes) -> Mapping[str, Any]:
    try:
        state = torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(f"weights-only deserialization failed: {error}") from error

    if not isinstance(state, Mapping):
        raise CheckpointError(
            f"checkpoint root is {type(state).__name__}; expected a tensor mapping"
        )

    return state


def _validate_keys(state: Mapping[str, Any], specs: Sequence[TensorSpec]) -> None:
    expected = {spec.name for spec in specs}
    actual = set(state.keys())

    if not all(isinstance(key, str) for key in state):
        raise CheckpointError("checkpoint contains a non-string tensor key")

    missing = sorted(expected - actual)
    additional = sorted(actual - expected)

    if missing or additional:
        details = []

        if missing:
            details.append("missing: " + ", ".join(missing))

        if additional:
            details.append("additional: " + ", ".join(additional))

        raise CheckpointError("checkpoint tensor keys differ (" + "; ".join(details) + ")")


def _canonical_tensor_bytes(tensor: torch.Tensor) -> bytes:
    # clone() guarantees an exact, offset-zero storage even if a checkpoint
    # tensor happens to be a contiguous view into a larger storage.
    normalized = tensor.detach().contiguous().clone()
    array = normalized.numpy().astype("<f4", copy=False)
    payload = array.tobytes(order="C")

    if len(payload) != normalized.numel() * normalized.element_size():
        raise CheckpointError("normalized tensor payload has an unexpected size")

    return payload


def _inspect_tensor(value: Any, spec: TensorSpec) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise CheckpointError(
            f"tensor {spec.name} is {type(value).__name__}; expected torch.Tensor"
        )

    if value.layout is not torch.strided:
        raise CheckpointError(f"tensor {spec.name} has unsupported layout {value.layout}")

    if value.is_quantized:
        raise CheckpointError(f"tensor {spec.name} is quantized")

    if value.is_complex():
        raise CheckpointError(f"tensor {spec.name} is complex")

    if value.dtype is not torch.float32:
        raise CheckpointError(
            f"tensor {spec.name} has dtype {value.dtype}; expected torch.float32"
        )

    if value.device.type != "cpu":
        raise CheckpointError(f"tensor {spec.name} remained on device {value.device}")

    shape = tuple(value.shape)

    if shape != spec.shape:
        raise CheckpointError(
            f"tensor {spec.name} has shape {shape}; expected {spec.shape}"
        )

    if not bool(torch.isfinite(value).all().item()):
        raise CheckpointError(f"tensor {spec.name} contains NaN or infinity")

    payload = _canonical_tensor_bytes(value)

    return {
        "dtype": "float32",
        "element_count": value.numel(),
        "layout": spec.layout,
        "name": spec.name,
        "payload_bytes": len(payload),
        "rank": value.ndim,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": list(shape),
    }


def inspect_checkpoint(
    path: str | Path,
    schema: CheckpointSchema = GHARBI_XTRANS_V1,
) -> dict[str, Any]:
    """Validate a checkpoint and return its deterministic inspection manifest."""

    checkpoint_path = Path(path)
    data, source_digest = _read_pinned_checkpoint(checkpoint_path, schema)
    state = _load_weights_only(data)
    _validate_keys(state, schema.tensors)
    tensors = [_inspect_tensor(state[spec.name], spec) for spec in schema.tensors]
    parameter_count = sum(tensor["element_count"] for tensor in tensors)
    payload_bytes = sum(tensor["payload_bytes"] for tensor in tensors)

    if parameter_count != schema.parameter_count:
        raise CheckpointError(
            f"checkpoint has {parameter_count} parameters; expected {schema.parameter_count}"
        )

    if payload_bytes != schema.payload_bytes:
        raise CheckpointError(
            f"checkpoint has {payload_bytes} payload bytes; expected {schema.payload_bytes}"
        )

    return {
        "format": MANIFEST_FORMAT,
        "model": {
            "architecture": {
                "convolution_padding": schema.convolution_padding,
                "depth": schema.architecture_depth,
                "name": schema.architecture_name,
                "width": schema.architecture_width,
            },
            "id": schema.model_id,
            "upstream": {
                "checkpoint_path": schema.upstream_checkpoint_path,
                "license": schema.upstream_license,
                "repository": schema.upstream_repository,
                "revision": schema.upstream_revision,
            },
        },
        "source": {
            "sha256": source_digest,
            "size_bytes": len(data),
        },
        "summary": {
            "dtype": "float32",
            "parameter_count": parameter_count,
            "payload_bytes": payload_bytes,
            "tensor_count": len(tensors),
        },
        "tensors": tensors,
    }


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Encode a manifest canonically for reproducible comparison and storage."""

    text = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def write_manifest(path: str | Path, payload: bytes, *, force: bool = False) -> None:
    """Atomically write a manifest without replacing one accidentally."""

    output_path = Path(path)

    if output_path.exists() and not force:
        raise CheckpointError(f"output already exists: {output_path}")

    if not output_path.parent.is_dir():
        raise CheckpointError(f"output directory does not exist: {output_path.parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
    )

    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        if output_path.exists() and not force:
            raise CheckpointError(f"output already exists: {output_path}")

        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inspect the pinned Gharbi X-Trans checkpoint.",
    )
    parser.add_argument("checkpoint", type=Path, help="path to xtrans.pth")
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON here instead of standard output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output file",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)

    if options.force and options.output is None:
        print("error: --force requires --output", file=sys.stderr)
        return 2

    try:
        manifest = inspect_checkpoint(options.checkpoint)
        payload = canonical_manifest_bytes(manifest)

        if options.output is None:
            sys.stdout.buffer.write(payload)
        else:
            write_manifest(options.output, payload, force=options.force)
    except CheckpointError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

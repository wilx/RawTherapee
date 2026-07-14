#!/usr/bin/env python3
"""Convert the pinned Gharbi X-Trans checkpoint to deterministic RTNN v1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .inspect_checkpoint import CheckpointError, load_validated_checkpoint
from .rtnn_format import (
    RTNNError,
    SerializedRTNN,
    build_conversion_manifest,
    canonical_conversion_manifest_bytes,
    serialize_rtnn,
)
from .semantic_schema import (
    SemanticSchemaError,
    load_semantic_schema,
    validate_inspection_manifest,
)


@dataclass(frozen=True)
class ConversionResult:
    """Complete deterministic outputs from one authenticated conversion."""

    artifact: SerializedRTNN
    manifest: dict[str, Any]
    manifest_bytes: bytes


def convert_checkpoint(path: str | Path) -> ConversionResult:
    """Authenticate and convert the pinned checkpoint without publishing files."""

    semantic_schema = load_semantic_schema()
    checkpoint = load_validated_checkpoint(path)
    validate_inspection_manifest(semantic_schema, checkpoint.manifest)
    artifact = serialize_rtnn(semantic_schema, checkpoint.tensor_payloads)
    manifest = build_conversion_manifest(semantic_schema, artifact)
    manifest_bytes = canonical_conversion_manifest_bytes(manifest)
    return ConversionResult(
        artifact=artifact,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )


def _write_temporary(output_path: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as stream:
            written = stream.write(payload)

            if written != len(payload):
                raise RTNNError(
                    f"short write for temporary output: {written} of {len(payload)} bytes"
                )

            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return temporary_path


def _validate_temporary(path: Path, payload: bytes) -> None:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RTNNError(f"cannot verify temporary output {path}: {error}") from error

    if len(data) != len(payload):
        raise RTNNError(
            f"temporary output {path} has {len(data)} bytes; expected {len(payload)}"
        )

    if hashlib.sha256(data).digest() != hashlib.sha256(payload).digest():
        raise RTNNError(f"temporary output verification failed: {path}")


def publish_conversion(
    output: str | Path,
    result: ConversionResult,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Publish the manifest first and RTNN last as the completion marker."""

    output_path = Path(output)
    manifest_path = Path(str(output_path) + ".json")

    if not output_path.parent.is_dir():
        raise RTNNError(f"output directory does not exist: {output_path.parent}")

    existing = [path for path in (output_path, manifest_path) if path.exists()]

    if existing and not force:
        raise RTNNError(f"output already exists: {existing[0]}")

    artifact_temporary: Path | None = None
    manifest_temporary: Path | None = None

    try:
        artifact_temporary = _write_temporary(output_path, result.artifact.data)
        manifest_temporary = _write_temporary(manifest_path, result.manifest_bytes)
        _validate_temporary(artifact_temporary, result.artifact.data)
        _validate_temporary(manifest_temporary, result.manifest_bytes)

        existing = [path for path in (output_path, manifest_path) if path.exists()]

        if existing and not force:
            raise RTNNError(f"output already exists: {existing[0]}")

        os.replace(manifest_temporary, manifest_path)
        manifest_temporary = None
        os.replace(artifact_temporary, output_path)
        artifact_temporary = None
    finally:
        for temporary in (artifact_temporary, manifest_temporary):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    return output_path, manifest_path


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the pinned Gharbi X-Trans checkpoint to RTNN v1.",
    )
    parser.add_argument("checkpoint", type=Path, help="path to xtrans.pth")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="write RTNN here and canonical JSON to <output>.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing RTNN and companion manifest files",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)

    try:
        result = convert_checkpoint(options.checkpoint)
        output, manifest = publish_conversion(
            options.output,
            result,
            force=options.force,
        )
    except (CheckpointError, SemanticSchemaError, RTNNError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"wrote RTNN v1: {output} "
        f"({len(result.artifact.data)} bytes, SHA-256 {result.artifact.sha256})"
    )
    print(f"wrote canonical manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

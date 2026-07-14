"""Validate a Phase 1 manifest against the tracked RTNN semantic schema."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .semantic_schema import (
    SEMANTIC_SCHEMA_SHA256,
    SemanticSchemaError,
    load_semantic_schema,
    validate_inspection_manifest_file,
)


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 1 manifest against the tracked RTNN semantic schema.",
    )
    parser.add_argument("manifest", type=Path, help="canonical Phase 1 manifest")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)

    try:
        schema = load_semantic_schema()
        manifest_digest = validate_inspection_manifest_file(schema, options.manifest)
    except SemanticSchemaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"validated architecture {schema.architecture_id} "
        f"({schema.architecture_symbol}), {len(schema.tensors)} tensors, "
        f"{schema.parameter_count} parameters"
    )
    print(f"semantic schema SHA-256: {SEMANTIC_SCHEMA_SHA256}")
    print(f"inspection manifest SHA-256: {manifest_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

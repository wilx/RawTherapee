#!/usr/bin/env python3
"""Inspect the reviewed RTNN artifact without reading companion metadata."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .rtnn_reader import RTNNReadError, inspection_document, read_rtnn
from .semantic_schema import SemanticSchemaError, canonical_json_bytes


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and inspect the reviewed RTNN v1 artifact.",
    )
    parser.add_argument("rtnn", type=Path, help="path to the converted .rtnn file")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)

    try:
        document = inspection_document(read_rtnn(options.rtnn))
        sys.stdout.buffer.write(canonical_json_bytes(document))
    except (RTNNReadError, SemanticSchemaError) as error:
        code = f" [{error.code.value}]" if isinstance(error, RTNNReadError) else ""
        print(f"error{code}: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

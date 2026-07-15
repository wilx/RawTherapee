#!/usr/bin/env python3
"""Export compact reference activations for native C++ parity tests."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
import sys

from .native_trace import NativeTraceError, build_native_trace, publish_native_trace
from .rtnn_reader import RTNNReadError
from .semantic_schema import SemanticSchemaError


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export native inference trace samples.")
    parser.add_argument("rtnn", type=Path, help="path to reviewed RTNN v1 weights")
    parser.add_argument("--output", type=Path, required=True, help="publish to this new directory")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    try:
        trace = build_native_trace(options.rtnn)
        output = publish_native_trace(options.output, trace)
    except (NativeTraceError, RTNNReadError, SemanticSchemaError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(trace.manifest['activations'])} activation samples to {output} "
        f"(manifest SHA-256 {hashlib.sha256(trace.manifest_bytes).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

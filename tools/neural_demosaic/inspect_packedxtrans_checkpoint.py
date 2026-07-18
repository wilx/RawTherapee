#!/usr/bin/env python3
"""Safely inspect the pinned PackedXTransNet checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .inspect_checkpoint import CheckpointError, canonical_manifest_bytes, write_manifest
from .packedxtrans import load_packed_checkpoint


def _arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to packed_5183_3208.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _arguments(arguments)
    if options.force and options.output is None:
        print("error: --force requires --output", file=sys.stderr)
        return 2
    try:
        payload = canonical_manifest_bytes(load_packed_checkpoint(options.checkpoint).manifest)
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

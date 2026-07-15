#!/usr/bin/env python3
"""Export deterministic golden tensors from the reviewed RTNN artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
import sys

from .golden_corpus import (
    GoldenCorpusError,
    build_golden_corpus,
    publish_golden_corpus,
)
from .rtnn_reader import RTNNReadError
from .semantic_schema import SemanticSchemaError


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the reviewed DemosaicNet X-Trans golden corpus.",
    )
    parser.add_argument("rtnn", type=Path, help="path to reviewed RTNN v1 weights")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="publish the corpus to this new directory",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    try:
        corpus = build_golden_corpus(options.rtnn)
        output = publish_golden_corpus(options.output, corpus)
    except (GoldenCorpusError, RTNNReadError, SemanticSchemaError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"wrote {len(corpus.manifest['cases'])} golden cases to {output} "
        f"(manifest SHA-256 {hashlib.sha256(corpus.manifest_bytes).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

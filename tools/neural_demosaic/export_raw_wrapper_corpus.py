#!/usr/bin/env python3
"""Export Phase 9 raw-wrapper fixtures from the strict reviewed RTNN reader."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
import sys

from .raw_wrapper_corpus import (
    RawWrapperCorpusError,
    build_raw_wrapper_corpus,
    publish_raw_wrapper_corpus,
)
from .rtnn_reader import RTNNReadError


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtnn", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        corpus = build_raw_wrapper_corpus(options.rtnn)
        output = publish_raw_wrapper_corpus(options.output, corpus)
    except (OSError, RawWrapperCorpusError, RTNNReadError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(corpus.manifest['cases'])} raw-wrapper cases to {output} "
        f"(manifest SHA-256 {hashlib.sha256(corpus.manifest_bytes).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Remove run-time-only fields from an X-Trans GMM result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import _write_json


NONDETERMINISTIC_FIELDS = frozenset((
    "baseline_seconds",
    "elapsed_seconds",
    "mean_inference_seconds_per_patch",
    "native_timings",
    "source",
))


def freeze(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: freeze(item)
            for key, item in value.items()
            if key not in NONDETERMINISTIC_FIELDS
        }
    if isinstance(value, list):
        return [freeze(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    _write_json(
        arguments.output,
        freeze(json.loads(arguments.input.read_text(encoding="utf-8"))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export the validation-selected K32/S9/q8 float32 research model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmr.experiment import _logical_digest, load_model

from .native_model import prepare_native_model, serialize_native_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError("native model output already exists")
    source = load_model(arguments.model)
    data = serialize_native_model(prepare_native_model(source))
    manifest = {
        "format": "rawtherapee-xtrans-tgmr-native-model-v1",
        "source_logical_sha256": _logical_digest(source),
        "contract": {
            "components": 32,
            "coarse_support": 3,
            "shortlist": 8,
            "degrees_of_freedom": 3,
            "tau": 0.0003,
            "temperature": 4,
            "scalar": "float32",
        },
        "binary": {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }
    temporary_manifest = arguments.output.with_suffix(".json.tmp")
    temporary_binary = arguments.output.with_suffix(".tmp")
    temporary_manifest.write_bytes(canonical_json_bytes(manifest))
    temporary_binary.write_bytes(data)
    temporary_manifest.replace(arguments.output.with_suffix(".json"))
    temporary_binary.replace(arguments.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

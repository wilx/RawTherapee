"""Compare deterministic X-veon float32 outputs from two native backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

OUTPUT_FLOATS = 3 * 288 * 288


class BackendComparisonError(ValueError):
    pass


def _read(path: Path) -> tuple[bytes, tuple[float, ...]]:
    data = path.read_bytes()
    if len(data) != OUTPUT_FLOATS * 4:
        raise BackendComparisonError(f"{path.name}: expected {OUTPUT_FLOATS * 4} bytes")
    values = struct.unpack(f"<{OUTPUT_FLOATS}f", data)
    if not all(math.isfinite(value) for value in values):
        raise BackendComparisonError(f"{path.name}: output contains NaN or infinity")
    return data, values


def compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference_bytes, reference = _read(reference_path)
    candidate_bytes, candidate = _read(candidate_path)
    absolute = sorted(abs(left - right) for left, right in zip(reference, candidate))
    squared_sum = sum(value * value for value in absolute)
    relative = [
        abs(left - right) / abs(left)
        for left, right in zip(reference, candidate)
        if abs(left) >= 1e-6
    ]

    def percentile(fraction: float) -> float:
        index = int(math.ceil(fraction * len(absolute))) - 1
        return absolute[max(0, min(index, len(absolute) - 1))]

    within = sum(
        abs(left - right) <= 5e-6 + 1e-5 * abs(left)
        for left, right in zip(reference, candidate)
    )
    return {
        "acceptance": {
            "maximum_at_most_0_005": absolute[-1] <= 0.005,
            "p99_at_most_0_001": percentile(0.99) <= 0.001,
            "rms_at_most_0_0005": math.sqrt(squared_sum / len(absolute)) <= 0.0005,
        },
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "exact_match_count": sum(left == right for left, right in zip(reference, candidate)),
        "float_count": len(absolute),
        "maximum_absolute_error": absolute[-1],
        "maximum_relative_error_abs_reference_ge_1e_6": max(relative, default=0.0),
        "mean_absolute_error": sum(absolute) / len(absolute),
        "p90_absolute_error": percentile(0.90),
        "p99_absolute_error": percentile(0.99),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "rms_absolute_error": math.sqrt(squared_sum / len(absolute)),
        "within_5e_6_plus_1e_5_abs_reference": within,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate)
    except (OSError, BackendComparisonError) as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

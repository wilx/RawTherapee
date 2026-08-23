from __future__ import annotations

from tools.xtrans_gmm.freeze import freeze


def test_freeze_removes_runtime_fields_but_preserves_provenance() -> None:
    value = {
        "elapsed_seconds": 1.5,
        "nested": {
            "baseline_seconds": {"a": 2.0},
            "mean_inference_seconds_per_patch": 0.1,
            "native_timings": {"b": 3.0},
            "source": "cache",
            "source_url": "https://example.invalid/source",
            "value": 7,
        },
    }
    assert freeze(value) == {
        "nested": {
            "source_url": "https://example.invalid/source",
            "value": 7,
        }
    }

"""Extended deterministic corpus for the green-candidate fusion experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.xtrans_mlri_internal.dataset import (
    dataset_manifest as internal_dataset_manifest,
    load_cases as load_internal_cases,
    origin_cells,
)


def _coherence_truth(level: str, size: int = 72) -> np.ndarray:
    truth = np.full((3, size, size), 0.12, dtype=np.float64)

    def point(x: int, y: int) -> None:
        if 1 <= x < size - 1 and 1 <= y < size - 1:
            truth[:, y, x] = (0.95, 0.16, 0.08)

    if level == "isolated-points":
        for x, y in ((14, 20), (51, 15), (38, 51), (19, 57), (59, 43)):
            point(x, y)
    elif level == "aligned-sparse-points":
        for coordinate in range(12, 61, 10):
            point(coordinate, coordinate)
    elif level == "dotted-line":
        for coordinate in range(8, 65, 4):
            point(coordinate, coordinate)
    elif level == "continuous-line":
        for coordinate in range(7, 65):
            point(coordinate, coordinate)
    elif level == "repeated-texture":
        for y in range(8, 65, 4):
            for x in range(8, 65, 4):
                if (x // 4 + y // 4) % 2 == 0:
                    point(x, y)
    else:
        raise ValueError(level)
    return truth


def load_cases(starfield_path: Path) -> list[dict[str, object]]:
    cases = load_internal_cases(starfield_path)
    origins = origin_cells()
    levels = (
        "isolated-points", "aligned-sparse-points", "dotted-line",
        "continuous-line", "repeated-texture",
    )
    for index, level in enumerate(levels):
        origin_x, origin_y = origins[13 + index]
        cases.append({
            "case_id": f"transition-coherence-{level}",
            "category": "coherence-transition",
            "family": "coherence-transition",
            "group": f"transition-coherence-{level}",
            "kind": "synthetic-transition",
            "origin_x": origin_x,
            "origin_y": origin_y,
            "source_id": f"transition-coherence-{level}",
            "split": "test",
            "truth": _coherence_truth(level),
        })
    return cases


def dataset_manifest(cases: list[dict[str, object]]) -> dict[str, object]:
    manifest = internal_dataset_manifest(cases)
    manifest["format"] = "rawtherapee-xtrans-mlri-green-fusion-dataset-v1"
    manifest["added_untouched_coherence_transition"] = [
        "isolated-points", "aligned-sparse-points", "dotted-line",
        "continuous-line", "repeated-texture",
    ]
    manifest["held_out_sources"] = list(manifest["held_out_sources"]) + [
        f"transition-coherence-{level}"
        for level in manifest["added_untouched_coherence_transition"]
    ]
    return manifest

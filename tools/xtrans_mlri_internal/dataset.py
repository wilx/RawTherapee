"""Deterministic linear-light corpus for the MLRI internal experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from tools.xtrans_danger.dataset import load_natural_cases, synthetic_cases
from tools.xtrans_hybrid.dataset import iter_crops, load_sources
from tools.xtrans_oracle.generate import synthetic_scenes


MLRI_CFA = np.asarray(
    (
        (1, 0, 1, 1, 2, 1),
        (2, 1, 2, 0, 1, 0),
        (1, 0, 1, 1, 2, 1),
        (1, 2, 1, 1, 0, 1),
        (0, 1, 0, 2, 1, 2),
        (1, 2, 1, 1, 0, 1),
    ),
    dtype=np.uint8,
)

NATURAL_SPLITS = {
    "astronaut": "train",
    "grass": "train",
    "gravel": "train",
    "brick": "validation",
    "hubble": "validation",
    "page": "test",
    "nasa-hydra-starfield": "test",
}


def cfa_for_origin(height: int, width: int, origin_x: int, origin_y: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return MLRI_CFA[(y + origin_y) % 6, (x + origin_x) % 6]


def origin_cells() -> tuple[tuple[int, int], ...]:
    """Return one translation origin for each of the 18 distinct CFA cells."""

    seen: set[bytes] = set()
    result: list[tuple[int, int]] = []
    for origin_y in range(6):
        for origin_x in range(6):
            cell = cfa_for_origin(6, 6, origin_x, origin_y)
            key = cell.tobytes()
            if key not in seen:
                seen.add(key)
                result.append((origin_x, origin_y))
    if len(result) != 18:
        raise RuntimeError(f"expected 18 X-Trans phase cells, found {len(result)}")
    return tuple(result)


def mosaic(truth: np.ndarray, origin_x: int, origin_y: int) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(truth, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3 or not np.isfinite(checked).all():
        raise ValueError("truth must be finite CHW RGB")
    cfa = cfa_for_origin(checked.shape[1], checked.shape[2], origin_x, origin_y)
    scalar = np.take_along_axis(np.moveaxis(checked, 0, -1), cfa[..., None], axis=2)[..., 0]
    return scalar, cfa


def _natural_cases(starfield_path: Path) -> list[dict[str, object]]:
    selected_crops: dict[str, list[dict[str, object]]] = {
        source_id: [] for source_id in NATURAL_SPLITS
    }
    for row in iter_crops(load_sources()):
        source_id = str(row["source_id"])
        if source_id in NATURAL_SPLITS:
            selected_crops[source_id].append(row)

    danger_natural, _ = load_natural_cases(starfield_path)
    selected_crops["nasa-hydra-starfield"] = [
        row for row in danger_natural
        if row["source_id"] == "nasa-hydra-starfield"
    ]

    cases: list[dict[str, object]] = []
    for source_id in NATURAL_SPLITS:
        if len(selected_crops[source_id]) != 3:
            raise RuntimeError(f"expected three deterministic crops for {source_id}")
        for row in selected_crops[source_id]:
            source = dict(row)
            x, y = source["crop"][:2]
            source.update({
                "case_id": str(source["crop_id"]),
                "family": str(source["category"]),
                "kind": "natural",
                "origin_x": int(x) % 6,
                "origin_y": int(y) % 6,
                "split": NATURAL_SPLITS[source_id],
            })
            cases.append(source)
    return cases


def _failure_cases() -> list[dict[str, object]]:
    origins = origin_cells()
    result = []
    for serial, source in enumerate(synthetic_cases(72)):
        case = dict(source)
        origin_x, origin_y = origins[int(case["cfa_variant"]) % len(origins)]
        case.update({
            "case_id": str(case["crop_id"]),
            "family": str(case["category"]),
            "kind": "synthetic-failure",
            "origin_x": origin_x,
            "origin_y": origin_y,
            "source_id": str(case["source_id"]),
        })
        # Keep the published source-level split.  The transition families and
        # the last three CFA placements remain untouched tests.
        result.append(case)
    return result


def _success_cases() -> list[dict[str, object]]:
    origins = origin_cells()
    result = []
    rows = synthetic_scenes(72)
    split_overrides = {
        "frequency_sweep": "test",
        "periodic_chromatic_texture": "test",
        "saturated_red_gray": "validation",
        "saturated_blue_gray": "validation",
    }
    for serial, (name, source) in enumerate(rows.items()):
        origin_x, origin_y = origins[(serial * 5 + 3) % len(origins)]
        result.append({
            "case_id": f"control-{name}",
            "category": source["category"],
            "family": source["category"],
            "group": f"control-{name}",
            "kind": "synthetic-success",
            "origin_x": origin_x,
            "origin_y": origin_y,
            "source_id": f"control-{name}",
            "split": split_overrides.get(name, "train"),
            "truth": np.asarray(source["truth"], dtype=np.float64),
        })
    return result


def load_cases(starfield_path: Path) -> list[dict[str, object]]:
    cases = _natural_cases(starfield_path) + _failure_cases() + _success_cases()
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate diagnostic case ID")
    return cases


def dataset_manifest(cases: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(cases)
    return {
        "format": "rawtherapee-xtrans-mlri-internal-dataset-v1",
        "case_count": len(rows),
        "cases": [
            {
                "id": row["case_id"],
                "family": row["family"],
                "kind": row["kind"],
                "origin": [row["origin_x"], row["origin_y"]],
                "shape_chw": list(np.asarray(row["truth"]).shape),
                "source_id": row["source_id"],
                "split": row["split"],
            }
            for row in rows
        ],
        "held_out_sources": ["nasa-hydra-starfield", "page"],
        "phase_cell_count": len(origin_cells()),
        "phase_policy": "the internal canonicalized MLRI domain is exercised at all 18 distinct X-Trans translation cells",
        "truth_domain": "normalized linear-light RGB",
    }

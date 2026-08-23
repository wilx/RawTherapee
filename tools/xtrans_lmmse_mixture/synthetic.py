#!/usr/bin/env python3
"""Deterministic analytical safety suite for a validation-selected MIX3 bank."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_lmmse.model import phase_origins

from .dataset import discover_expanded
from .experiment import _evaluate_cases, _json_safe
from .model import MixtureDefinition
from .training import survey_and_train_globals, train_mixtures


SIZE = 96


def _base(value: tuple[float, float, float] = (0.08, 0.08, 0.08)) -> np.ndarray:
    return np.broadcast_to(np.asarray(value)[:, None, None], (3, SIZE, SIZE)).copy()


def scenes() -> list[dict[str, object]]:
    y, x = np.mgrid[:SIZE, :SIZE]
    rows: list[dict[str, object]] = []

    gradient = np.stack((0.05 + 0.9 * x / 95, 0.1 + 0.75 * y / 95, 0.1 + 0.7 * (x + y) / 190))
    rows.append({"id": "gradient", "category": "smooth", "rgb": gradient, "all_phases": False})

    for name, color in (
        ("white", (1.0, 1.0, 1.0)),
        ("red", (1.0, 0.0, 0.0)),
        ("green", (0.0, 1.0, 0.0)),
        ("blue", (0.0, 0.0, 1.0)),
    ):
        impulse = _base()
        impulse[:, 48, 48] = color
        rows.append({"id": f"impulse-{name}", "category": "impulse", "rgb": impulse, "all_phases": True})

    for radius, color in ((1, (1.0, 0.05, 0.8)), (2, (0.0, 0.9, 1.0)), (3, (1.0, 0.7, 0.0))):
        dot = _base((0.03, 0.04, 0.05))
        selected = (x - 48) ** 2 + (y - 48) ** 2 <= radius * radius
        dot[:, selected] = np.asarray(color)[:, None]
        rows.append({"id": f"saturated-dot-r{radius}", "category": "saturated-point", "rgb": dot, "all_phases": radius == 1})

    star = _base((0.01, 0.012, 0.018))
    gaussian = np.exp(-((x - 48) ** 2 + (y - 48) ** 2) / (2.0 * 1.2 ** 2))
    star += np.asarray((0.95, 0.45, 0.8))[:, None, None] * gaussian
    rows.append({"id": "tiny-colored-star", "category": "star", "rgb": np.clip(star, 0.0, 1.0), "all_phases": True})

    for orientation in ("vertical", "horizontal", "diagonal"):
        line = _base((0.08, 0.12, 0.18))
        if orientation == "vertical":
            selected = np.abs(x - 48) <= 0
        elif orientation == "horizontal":
            selected = np.abs(y - 48) <= 0
        else:
            selected = np.abs(x - y) <= 0
        line[:, selected] = np.asarray((0.95, 0.1, 0.65))[:, None]
        rows.append({"id": f"thin-line-{orientation}", "category": "thin-line", "rgb": line, "all_phases": False})

    intersection = _base((0.04, 0.05, 0.06))
    intersection[:, np.abs(x - 48) <= 0] = np.asarray((1.0, 0.0, 0.1))[:, None]
    intersection[:, np.abs(y - 48) <= 0] = np.asarray((0.0, 0.8, 1.0))[:, None]
    rows.append({"id": "line-intersection", "category": "intersection", "rgb": intersection, "all_phases": False})

    for orientation in ("vertical", "horizontal", "diagonal"):
        if orientation == "vertical":
            side = x >= 48
        elif orientation == "horizontal":
            side = y >= 48
        else:
            side = x >= y
        edge = np.empty((3, SIZE, SIZE), dtype=np.float64)
        first = np.asarray((0.95, 0.08, 0.12))
        second = np.asarray((0.03, 0.75, 0.95))
        edge[:] = first[:, None, None]
        edge[:, side] = second[:, None]
        rows.append({"id": f"chromatic-edge-{orientation}", "category": "chromatic-edge", "rgb": edge, "all_phases": False})

    frequency = 0.01 + 0.47 * x / 95.0
    phase = 2.0 * np.pi * frequency * x
    sweep = np.stack((0.5 + 0.45 * np.sin(phase), 0.5 + 0.45 * np.sin(phase + 2.1), 0.5 + 0.45 * np.sin(phase + 4.2)))
    rows.append({"id": "frequency-sweep", "category": "frequency", "rgb": sweep, "all_phases": False})

    periodic = np.stack((
        0.1 + 0.85 * ((x + y) % 4 == 0),
        0.1 + 0.85 * ((x - y) % 5 == 0),
        0.1 + 0.85 * ((2 * x + y) % 7 == 0),
    ))
    rows.append({"id": "periodic-chromatic", "category": "periodic", "rgb": periodic, "all_phases": False})

    checker = np.asarray(((x + y) % 2), dtype=np.float64)
    monochrome = np.stack((checker, checker, checker)) * 0.9 + 0.05
    rows.append({"id": "high-frequency-monochrome", "category": "monochrome", "rgb": monochrome, "all_phases": False})

    for spacing in (16, 8, 4, 2):
        transition = _base((0.03, 0.04, 0.05))
        points = (x % spacing == 0) & (y % spacing == 0)
        transition[:, points] = np.asarray((1.0, 0.15, 0.7))[:, None]
        rows.append({"id": f"point-density-{spacing}", "category": "sparse-to-coherent", "rgb": transition, "all_phases": False})
    for thickness in (1, 2, 4, 8):
        transition = _base((0.05, 0.08, 0.1))
        selected = np.abs(x - 48) < thickness
        transition[:, selected] = np.asarray((0.9, 0.2, 0.7))[:, None]
        rows.append({"id": f"line-thickness-{thickness}", "category": "sparse-to-coherent", "rgb": transition, "all_phases": False})

    for row in rows:
        checked = np.asarray(row["rgb"], dtype=np.float64)
        if checked.shape != (3, SIZE, SIZE) or not np.isfinite(checked).all() or np.min(checked) < 0.0 or np.max(checked) > 1.0:
            raise RuntimeError(f"invalid synthetic scene {row['id']}")
    return rows


def _definition(row: dict[str, object]) -> MixtureDefinition:
    return MixtureDefinition(
        name=str(row["name"]),
        partition=str(row["partition"]),
        activity_feature=str(row["activity_feature"]),
        activity_threshold=float(row["activity_threshold"]),
        anisotropy_threshold=float(row["anisotropy_threshold"]),
        activity_width=float(row["activity_width"]),
        anisotropy_width=float(row["anisotropy_width"]),
        class_names=tuple(str(value) for value in row["class_names"]),
        activity_quantile=float(row["activity_quantile"]),
        anisotropy_quantile=float(row["anisotropy_quantile"]),
    )


def run(output: Path, natural_result: Path, bsds_root: Path, runner: Path) -> dict[str, object]:
    started = time.monotonic()
    natural = json.loads(natural_result.read_text(encoding="utf-8"))
    splits = discover_expanded(bsds_root)
    globals_by_size, normalization, _ = survey_and_train_globals(splits["train"])
    selected_definition = _definition(natural["selected"]["definition"])
    mixtures, diagnostics = train_mixtures(
        splits["train"], [selected_definition], normalization, globals_by_size[200]
    )
    mixture = mixtures[selected_definition.name]
    cases = []
    for scene in scenes():
        truth = np.asarray(scene["rgb"], dtype=np.float64)
        origins = phase_origins() if bool(scene["all_phases"]) else ((0, 0),)
        for phase_index, (origin_x, origin_y) in enumerate(origins):
            cases.append({
                "case_id": f"{scene['id']}-phase{phase_index}",
                "context_truth": truth,
                "group": str(scene["category"]),
                "origin_x": origin_x,
                "origin_y": origin_y,
                "region": (0, 0, SIZE, SIZE),
                "truth": truth,
            })
    evaluation = _evaluate_cases(
        cases, globals_by_size, mixture, runner, run_baselines=True
    )
    evaluation.pop("feature_samples", None)
    deltas = []
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in evaluation["rows"]:
        delta = float(row["methods"]["mix3-hard"]["psnr_db"]) - float(row["methods"]["gmax"]["psnr_db"])
        deltas.append(delta)
        by_category.setdefault(str(row["group"]), []).append(row)
    categories = {}
    for category, rows in by_category.items():
        values = [
            float(row["methods"]["mix3-hard"]["psnr_db"]) - float(row["methods"]["gmax"]["psnr_db"])
            for row in rows
        ]
        categories[category] = {
            "case_count": len(rows),
            "maximum_mix_minus_gmax_db": max(values),
            "minimum_mix_minus_gmax_db": min(values),
            "mean_mix_minus_gmax_db": float(np.mean(values)),
        }
    result = {
        "case_count": len(cases),
        "categories": categories,
        "decision": {
            "catastrophic_regression": min(deltas) < -2.0,
            "minimum_case_delta_db": min(deltas),
            "phase_safe": all(
                categories[name]["minimum_mix_minus_gmax_db"] >= -0.5
                for name in ("impulse", "star", "saturated-point")
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "evaluation": evaluation,
        "format": "rawtherapee-xtrans-lmmse-mixture-synthetic-v1",
        "selected_definition": natural["selected"]["definition"],
        "training": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(_json_safe(result)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-result", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.output, arguments.natural_result, arguments.bsds_root,
        arguments.runner,
    )
    print(json.dumps({
        "case_count": result["case_count"],
        "categories": result["categories"],
        "decision": result["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

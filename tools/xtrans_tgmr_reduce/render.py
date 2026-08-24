#!/usr/bin/env python3
"""Render deterministic figures for the t-GMR reduction report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=144, metadata={"Software": "RawTherapee research"})
    plt.close()


def render(screen_path: Path, final_path: Path, output: Path) -> None:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    components = screen["component_count"]
    x = [row["component_count"] for row in components]
    quality = [row["metric"]["psnr_db"] for row in components]
    retained = [100.0 * row["quality_retention"] for row in components]
    figure, left = plt.subplots(figsize=(7.0, 4.2))
    right = left.twinx()
    left.plot(x, quality, "o-", color="#2458a6", label="validation PSNR")
    right.plot(x, retained, "s--", color="#b64926", label="retained MSE gain")
    right.axhline(90.0, color="#8b2020", linewidth=1, linestyle=":")
    left.set_xlabel("components per phase")
    left.set_ylabel("validation PSNR (dB)", color="#2458a6")
    right.set_ylabel("TGMR64 gain retained (%)", color="#b64926")
    left.grid(alpha=0.25)
    figure.suptitle("Genuine Student-t component reduction")
    _save(output / "component-count.png")

    factors = screen["factor_rank"]
    x = [row["rank"] for row in factors]
    retained = [100.0 * row["quality_retention"] for row in factors]
    reduction = [163072.0 / row["estimated_mac_per_pixel"] for row in factors]
    figure, left = plt.subplots(figsize=(7.0, 4.2))
    right = left.twinx()
    left.plot(x, retained, "o-", color="#2f7f4f")
    right.plot(x, reduction, "s--", color="#7851a9")
    left.axhline(90.0, color="#8b2020", linewidth=1, linestyle=":")
    left.set_xlabel("post-hoc factor rank")
    left.set_ylabel("TGMR64 gain retained (%)", color="#2f7f4f")
    right.set_ylabel("estimated arithmetic reduction", color="#7851a9")
    left.grid(alpha=0.25)
    figure.suptitle("Factor covariance has no quality-cost knee")
    _save(output / "factor-rank.png")

    points = []
    for row in components:
        points.append((
            163072.0 * row["relative_dense_cost"],
            100.0 * row["quality_retention"],
            f"K{row['component_count']}",
            "#2458a6",
            True,
        ))
    for row in screen["factor_rank"]:
        if row["quality_retention"] < 0.0:
            continue
        points.append((
            row["estimated_mac_per_pixel"],
            100.0 * row["quality_retention"],
            f"r{row['rank']}",
            "#2f7f4f",
            True,
        ))
    selected = screen["validation_selection"]["combined"]
    for row in screen["combined_component_shortlist"]:
        label = f"K{row['component_count']}/S{row['support']**2}/q{row['retained']}"
        is_selected = (
            selected is not None
            and row["component_count"] == selected["component_count"]
            and row["support"] == selected["support"]
            and row["retained"] == selected["retained"]
        )
        points.append((
            row["estimated_mac_per_pixel"],
            100.0 * row["quality_retention"],
            label,
            "#b64926",
            is_selected,
        ))
    plt.figure(figsize=(9.0, 5.2))
    for cost, quality_value, label, color, annotate in points:
        plt.scatter(cost, quality_value, color=color, s=38 if annotate else 24)
        if annotate:
            plt.annotate(
                label, (cost, quality_value), xytext=(4, 4),
                textcoords="offset points", fontsize=8,
            )
    plt.axhline(90.0, color="#8b2020", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("estimated MAC-scale operations per pixel (log scale)")
    plt.ylabel("TGMR64 validation gain retained (%)")
    plt.title("Validation quality-cost frontier")
    plt.ylim(0.0, 106.0)
    plt.grid(alpha=0.22)
    _save(output / "pareto.png")

    selected = final["selected"]["evaluation"]
    labels = ["external worst", "Hubble bright", "Hydra bright"]
    values = [
        selected["external"]["worst_delta_vs_gmax_db"],
        selected["stars"]["hubble-bright"]["delta_vs_gmax_db"],
        selected["stars"]["nasa-hydra-starfield-bright"]["delta_vs_gmax_db"],
    ]
    synthetic = selected["synthetic"]
    for name in sorted(synthetic):
        labels.append(name)
        values.append(synthetic[name]["delta_vs_gmax_db"])
    positions = np.arange(len(labels))
    colors = ["#2f7f4f" if value >= 0.0 else "#b64926" for value in values]
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(positions, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(-0.5, color="#8b2020", linestyle=":", linewidth=1)
    plt.axhline(-2.0, color="#8b2020", linestyle="--", linewidth=1)
    plt.xticks(positions, labels, rotation=40, ha="right")
    plt.ylabel("selected candidate minus GMAX (dB)")
    plt.title("Frozen safety confirmation")
    plt.grid(axis="y", alpha=0.2)
    _save(output / "selected-safety.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.screen, arguments.final, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

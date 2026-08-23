#!/usr/bin/env python3
"""Render deterministic report figures from the canonical population result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(result_path: Path, output: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    supports = sorted(int(value) for value in result["pooled"]["lmmse_by_support"])
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.plot(
        supports,
        [result["pooled"]["lmmse_by_support"][str(value)]["psnr_db"] for value in supports],
        "o-", label="population LMMSE, unclipped",
    )
    axis.plot(
        supports,
        [result["pooled"]["lmmse_clipped_by_support"][str(value)]["psnr_db"] for value in supports],
        "o-", label="population LMMSE, clipped",
    )
    axis.plot(
        supports,
        [result["pooled"]["geometry_by_support"][str(value)]["psnr_db"] for value in supports],
        "o-", label="geometry-only",
    )
    for method, style in (("corrected-final", "--"), ("ulri-slow0", ":"), ("markesteijn", "-.")):
        axis.axhline(
            result["pooled"]["baselines"][method]["psnr_db"],
            linestyle=style,
            label=method,
        )
    axis.set_xlabel("square support (pixels)")
    axis.set_ylabel("pooled test PSNR (dB)")
    axis.set_xticks(supports)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "support-curve.png", dpi=160, metadata={"Software": "Matplotlib"})
    plt.close(figure)

    curve = result["training_size_curve"]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.plot(
        [row["source_count"] for row in curve],
        [row["test"]["psnr_db"] for row in curve],
        "o-",
    )
    axis.set_xlabel("training source images")
    axis.set_ylabel("pooled test PSNR (dB)")
    axis.grid(True, alpha=0.25)
    figure.savefig(output / "training-size-curve.png", dpi=160, metadata={"Software": "Matplotlib"})
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    render(arguments.result, arguments.output)


if __name__ == "__main__":
    main()

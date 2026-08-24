#!/usr/bin/env python3
"""Render the tracked native TGMR optimization measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def render(results_path: Path, output: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))

    variants = data["variants_512"]
    labels = [f'{row["name"]} ({row["threads"]}T)' for row in variants]
    values = [row["mp_s"] for row in variants]
    colors = ["#4472c4" if row["input"] == "matrix" else "#70ad47" for row in variants]
    plt.figure(figsize=(10, 5.5))
    plt.barh(range(len(values)), values, color=colors)
    plt.yticks(range(len(values)), labels)
    plt.gca().invert_yaxis()
    plt.axvline(6.0, color="#c00000", linestyle="--", label="strong-GO threshold")
    plt.xlabel("Throughput (MP/s), 512x512")
    plt.title("Incremental native TGMR optimization")
    plt.legend()
    _save(output / "variant-throughput.png")

    reference = data["baseline_scaling_512"]
    final = data["final_scaling_4mp"]
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        [row["threads"] for row in reference],
        [row["mp_s"] for row in reference],
        marker="o",
        label="O0 phase-only (512x512)",
    )
    plt.plot(
        [row["threads"] for row in final],
        [row["mp_s"] for row in final],
        marker="o",
        label="O5 streaming (4 MP)",
    )
    plt.axhline(6.0, color="#c00000", linestyle="--", label="strong-GO threshold")
    plt.xlabel("OpenMP threads")
    plt.ylabel("Throughput (MP/s)")
    plt.title("Thread scaling")
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "thread-scaling.png")

    large = data["large_streaming"]
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        [row["pixels"] / 1e6 for row in large],
        [row["mp_s"] for row in large],
        marker="o",
        color="#70ad47",
    )
    plt.axhline(6.0, color="#c00000", linestyle="--", label="strong-GO threshold")
    plt.xlabel("Mosaic size (megapixels)")
    plt.ylabel("Throughput (MP/s)")
    plt.title("O5 large-mosaic throughput")
    plt.ylim(0, max(8, max(row["mp_s"] for row in large) * 1.15))
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "large-mosaic-throughput.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    render(arguments.results, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render compact plots from canonical MIX3 result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, metadata={"Software": "RawTherapee research tooling"})
    plt.close()


def render(natural_path: Path, synthetic_path: Path, output: Path) -> None:
    natural = json.loads(natural_path.read_text(encoding="utf-8"))
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    scale = natural["global_training_scale"]
    counts = np.asarray((100, 150, 200))
    scores = np.asarray([scale[str(value)]["psnr_db"] for value in counts])
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(counts, scores, "o-", label="global LMMSE")
    plt.axhline(
        natural["evaluation"]["bsds_test"]["pooled"]["mix3-hard"]["psnr_db"],
        color="#d95f02", linestyle="--", label="MIX3 hard",
    )
    plt.axhline(
        natural["evaluation"]["bsds_test"]["pooled"]["mix3-soft"]["psnr_db"],
        color="#7570b3", linestyle=":", label="MIX3 soft",
    )
    plt.xlabel("BSDS training sources")
    plt.ylabel("Untouched test PSNR (dB)")
    plt.title("More data versus three covariance banks")
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "training-scale.png")

    samples = []
    for evaluation in natural["evaluation"].values():
        samples.extend(evaluation["feature_samples"])
    labels = sorted({str(row["dataset"]) for row in samples})
    plt.figure(figsize=(7.2, 5.2))
    palette = plt.get_cmap("tab10")
    for index, label in enumerate(labels):
        rows = [row for row in samples if row["dataset"] == label]
        plt.scatter(
            [row["activity"] for row in rows],
            [row["anisotropy"] for row in rows],
            s=7, alpha=0.35, label=label, color=palette(index % 10),
        )
    definition = natural["selected"]["definition"]
    plt.axvline(definition["activity_threshold"], color="black", linewidth=1)
    plt.axhline(definition["anisotropy_threshold"], color="black", linewidth=1)
    plt.xlabel("Phase-normalized log observed-chroma activity")
    plt.ylabel("Same-color directional anisotropy")
    plt.title("Held-out domains in the two-feature selector plane")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    _save(output / "feature-distribution.png")

    groups = ("bsds_test", "established", "external_chromatic")
    methods = ("gmax", "mix3-hard", "mix3-soft", "oracle-7")
    x = np.arange(len(groups))
    width = 0.19
    plt.figure(figsize=(8.0, 4.4))
    for index, method in enumerate(methods):
        values = [natural["evaluation"][group]["pooled"][method]["psnr_db"] for group in groups]
        plt.bar(x + (index - 1.5) * width, values, width, label=method)
    plt.xticks(x, ("BSDS test", "Established", "Chromatic controls"))
    plt.ylabel("PSNR (dB)")
    plt.title("Natural held-out reconstruction")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    _save(output / "natural-quality.png")

    categories = synthetic["categories"]
    names = sorted(categories)
    values = [categories[name]["mean_mix_minus_gmax_db"] for name in names]
    minimum = [categories[name]["minimum_mix_minus_gmax_db"] for name in names]
    plt.figure(figsize=(8.4, 5.4))
    positions = np.arange(len(names))
    plt.barh(positions, values, color=["#1b9e77" if value >= 0 else "#d95f02" for value in values])
    plt.scatter(minimum, positions, color="black", s=16, label="worst case")
    plt.axvline(0.0, color="black", linewidth=1)
    plt.yticks(positions, names)
    plt.xlabel("MIX3 hard minus GMAX PSNR (dB)")
    plt.title("Analytical safety suite")
    plt.legend()
    plt.grid(axis="x", alpha=0.2)
    _save(output / "synthetic-deltas.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.natural, args.synthetic, args.output)


if __name__ == "__main__":
    main()

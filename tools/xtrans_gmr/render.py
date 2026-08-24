#!/usr/bin/env python3
"""Render compact figures for the phase-conditioned GMR report."""

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
    plt.savefig(path, dpi=160, metadata={"Software": "RawTherapee GMR experiment"})
    plt.close()


def component_curve(result: dict, output: Path) -> None:
    rows = result["component_screen"]
    x = [row["component_count"] for row in rows]
    plt.figure(figsize=(6.4, 4.1))
    plt.semilogx(x, [row["selected"]["mmse"]["psnr_db"] for row in rows], "o-", label="posterior MMSE", base=2)
    plt.semilogx(x, [row["selected"]["oracle_7"]["psnr_db"] for row in rows], "o--", label="phase-aware 7x7 oracle", base=2)
    plt.axhline(38.303, color="tab:gray", linestyle=":", label="FULL-GMM validation")
    plt.xticks(x, [str(value) for value in x])
    plt.xlabel("components per phase")
    plt.ylabel("validation PSNR (dB)")
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "component-count.png")


def stability(result: dict, output: Path) -> None:
    checkpoints = result["em_checkpoints"]["rows"]
    sources = result["training_source_curve"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    iterations = [row["iterations"] for row in checkpoints]
    axes[0].plot(iterations, [row["validation"]["mmse"]["psnr_db"] for row in checkpoints], "o-", color="tab:blue")
    axes[0].set_xlabel("EM iterations")
    axes[0].set_ylabel("validation PSNR (dB)", color="tab:blue")
    twin = axes[0].twinx()
    twin.plot(iterations, [row["model"]["mean_training_lower_bound"] for row in checkpoints], "s--", color="tab:orange")
    twin.set_ylabel("mean training log likelihood", color="tab:orange")
    counts = [row["sources"] for row in sources]
    axes[1].plot(counts, [row["validation"]["psnr_db"] for row in sources], "o-", label="validation")
    axes[1].plot(counts, [row["test"]["psnr_db"] for row in sources], "o-", label="test")
    axes[1].set_xlabel("BSDS training sources")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(output / "training-stability.png")


def deltas(result: dict, previous: dict, output: Path) -> None:
    test_rows = result["bsds_test"]["gmr"]["rows"]
    names = list(test_rows)
    gmax = []
    full = []
    for name in names:
        value = test_rows[name]["mmse"]["psnr_db"]
        gmax.append(value - previous["sampled_baselines"]["rows"][name]["methods"]["gmax"]["psnr_db"])
        full.append(value - previous["bsds_test"]["rows"][name]["cfa-mmse"]["psnr_db"])
    order = np.argsort(gmax)
    positions = np.arange(len(names))
    plt.figure(figsize=(10.0, 4.5))
    plt.bar(positions - 0.2, np.asarray(gmax)[order], 0.4, label="GMR - GMAX")
    plt.bar(positions + 0.2, np.asarray(full)[order], 0.4, label="GMR - FULL-GMM")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(positions, [names[index].removesuffix(".jpg") for index in order], rotation=65, ha="right")
    plt.ylabel("PSNR delta (dB)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save(output / "bsds-deltas.png")


def external_deltas(result: dict, previous: dict, output: Path) -> None:
    rows = result["external"]["gmr"]["rows"]
    names = list(rows)
    values = [
        rows[name]["mmse"]["psnr_db"]
        - previous["external_reference"][name]["methods"]["gmax"]["psnr_db"]
        for name in names
    ]
    order = np.argsort(values)
    colors = ["tab:red" if values[index] < -0.5 else "tab:blue" for index in order]
    plt.figure(figsize=(10.0, 4.4))
    plt.bar(np.arange(len(names)), np.asarray(values)[order], color=colors)
    plt.axhline(-0.5, color="tab:red", linestyle="--", label="safety limit")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(names)), [names[index] for index in order], rotation=65, ha="right")
    plt.ylabel("GMR - GMAX PSNR (dB)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save(output / "external-deltas.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    component_curve(result, args.output)
    stability(result, args.output)
    deltas(result, previous, args.output)
    external_deltas(result, previous, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

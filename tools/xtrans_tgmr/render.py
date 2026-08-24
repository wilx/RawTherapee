#!/usr/bin/env python3
"""Render deterministic figures for the Student-t GMR report."""

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


def render(results: Path, output: Path) -> None:
    data = json.loads(results.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    sweep = data["stage_a"]["validation_sweep"]
    labels = [str(row["degrees_of_freedom"]) for row in sweep]
    quality = [row["methods"]["mmse"]["psnr_db"] for row in sweep]
    entropy = [row["posterior"]["entropy_mean"] for row in sweep]
    figure, left = plt.subplots(figsize=(7.2, 4.2))
    right = left.twinx()
    left.plot(labels, quality, "o-", color="#2458a6", label="validation PSNR")
    right.plot(labels, entropy, "s--", color="#b64926", label="posterior entropy")
    left.set_xlabel("degrees of freedom")
    left.set_ylabel("PSNR (dB)", color="#2458a6")
    right.set_ylabel("mean entropy (nats)", color="#b64926")
    left.grid(alpha=0.25)
    figure.suptitle("Frozen experts: Student-t robustness curve")
    _save(output / "robustness-curve.png")

    checkpoints = data["ecm_checkpoints"]
    x = [row["ecm_iterations"] for row in checkpoints]
    y = [row["validation"]["methods"]["mmse"]["psnr_db"] for row in checkpoints]
    plt.figure(figsize=(6.8, 4.1))
    plt.plot(x, y, "o-", color="#2f7f4f")
    plt.xlabel("Student-t ECM iterations")
    plt.ylabel("validation PSNR (dB)")
    plt.title("True Student-t mixture continuation")
    plt.grid(alpha=0.25)
    _save(output / "ecm-checkpoints.png")

    gmr = data["stage_a"]["external_gaussian"]["rows"]
    resp = data["external"]["responsibility_only"]["rows"]
    trained = data["external"]["trained_t"]["rows"]
    baselines = data["external"]["delta_vs_gmax_db"]
    names = sorted(trained)
    # Recover GMAX from the selected trained delta, avoiding another duplicate
    # baseline table in the canonical result.
    gmax = {
        name: trained[name]["mmse"]["psnr_db"] - baselines[name]
        for name in names
    }
    x = np.arange(len(names))
    width = 0.26
    plt.figure(figsize=(12.0, 5.1))
    plt.bar(x - width, [gmr[n]["mmse"]["psnr_db"] - gmax[n] for n in names], width, label="Gaussian GMR")
    plt.bar(x, [resp[n]["mmse"]["psnr_db"] - gmax[n] for n in names], width, label="t responsibilities")
    plt.bar(x + width, [trained[n]["mmse"]["psnr_db"] - gmax[n] for n in names], width, label="trained t-GMR")
    plt.axhline(-0.5, color="#a02020", linestyle="--", linewidth=1, label="safety limit")
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.xticks(x, names, rotation=55, ha="right", fontsize=8)
    plt.ylabel("delta versus GMAX (dB)")
    plt.title("Untouched external chromatic crops")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    _save(output / "external-safety.png")

    synthetic = data["synthetic"]
    names = [row["scene"] for row in synthetic]
    x = np.arange(len(names))
    width = 0.24
    plt.figure(figsize=(9.0, 4.5))
    for offset, key, label in (
        (-width, "gmax", "GMAX"),
        (0.0, "gaussian_gmr", "Gaussian GMR"),
        (width, "trained_t", "trained t-GMR"),
    ):
        plt.bar(
            x + offset,
            [row["phase_range_db"][key] for row in synthetic],
            width,
            label=label,
        )
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("18-phase PSNR spread (dB)")
    plt.title("Phase sensitivity remains a separate limitation")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    _save(output / "phase-spread.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.results, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render compact plots from an external PnP experiment JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, metadata={"Software": "RawTherapee research tooling"})
    plt.close()


def render(results: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    iterations = (1, 2, 3, 5, 10, 20)
    plt.figure(figsize=(7.2, 4.4))
    trajectories = {
        "GMAX, exact": results["validation"]["long_trajectories"]["exact"],
        "GMAX, soft rho=0.1": results["validation"]["long_trajectories"]["soft-rho-0.1"],
        "MLRI, exact": results["validation"]["initialization_controls"]["mlri-exact"],
        "MLRI, soft rho=0.1": results["validation"]["initialization_controls"]["mlri-soft-rho-0.1"],
    }
    for label, row in trajectories.items():
        values = [row["iterations"][str(i)]["pooled"]["psnr_db"] for i in iterations]
        plt.plot(iterations, values, marker="o", label=label)
    plt.axhline(
        results["validation"]["baselines"]["mlri"]["pooled"]["psnr_db"],
        color="0.35", linestyle="--", label="MLRI initializer",
    )
    plt.xlabel("PnP iteration")
    plt.ylabel("Pooled validation PSNR (dB)")
    plt.title("PnP-BM3D quality peaks after 2–3 iterations")
    plt.xticks(iterations)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _save(output / "convergence.png")

    plt.figure(figsize=(7.2, 4.4))
    sigmas = [0.002, 0.005, 0.01, 0.02, 0.04]
    feasibility = results["validation"]["denoiser_feasibility"]
    for name, label in (("gmax_post", "GMAX + BM3D"), ("mlri_post", "MLRI + BM3D")):
        values = [feasibility[str(sigma)][name]["pooled"]["psnr_db"] for sigma in sigmas]
        plt.plot(range(len(sigmas)), values, marker="o", label=label)
    plt.xlabel("BM3D strength sigma")
    plt.ylabel("Pooled validation PSNR (dB)")
    plt.title("One-shot denoiser-strength control")
    plt.xticks(range(len(sigmas)), [str(value) for value in sigmas])
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "denoiser-strength.png")

    test = results["test"]
    baseline = test["baselines"]["mlri"]["by_source"]
    pnp = test["pnp_selected_initializer"]["by_source"]
    labels = list(baseline)
    differences = [pnp[label]["psnr_db"] - baseline[label]["psnr_db"] for label in labels]
    colors = ["#3a923a" if value >= 0.0 else "#b74b4b" for value in differences]
    plt.figure(figsize=(9.0, 4.6))
    plt.bar(range(len(labels)), differences, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("PnP − MLRI PSNR (dB)")
    plt.title("Untouched BSDS test: 16/20 sources improve, pooled gain 0.046 dB")
    plt.grid(axis="y", alpha=0.25)
    _save(output / "test-source-deltas.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    render(json.loads(arguments.results.read_text()), arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render deterministic figures for the X-Trans GMM feasibility report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rawtherapee-xtrans-gmm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#3465a4", "#cc0000", "#4e9a06", "#75507b")


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        path, dpi=150,
        metadata={"Software": "RawTherapee X-Trans GMM experiment"},
    )
    plt.close()


def _component_curve(result: dict, output: Path) -> None:
    rows = [
        row for row in result["configuration_screen"]
        if row["patch_size"] == 5 and row["dc_mode"] == "observed-rgb"
    ]
    x = [row["model"]["component_count"] for row in rows]
    posterior = [row["selected"]["methods"]["cfa-mmse"]["psnr_db"] for row in rows]
    oracle = [row["selected"]["methods"]["oracle-7"]["psnr_db"] for row in rows]
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogx(x, posterior, "o-", color=COLORS[0], label="CFA posterior MMSE")
    plt.semilogx(x, oracle, "o-", color=COLORS[1], label="7x7 component oracle")
    plt.xticks(x, [str(value) for value in x])
    plt.xlabel("Gaussian components")
    plt.ylabel("Validation PSNR (dB)")
    plt.title("Mixture capacity at fixed 5x5 support")
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "component-count.png")


def _training_curve(supplement: dict, output: Path) -> None:
    rows = supplement["training_size_curve"]
    x = [row["source_count"] for row in rows]
    validation = [row["validation"]["psnr_db"] for row in rows]
    test = [row["test"]["psnr_db"] for row in rows]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, validation, "o-", color=COLORS[0], label="validation")
    plt.plot(x, test, "o-", color=COLORS[1], label="untouched test")
    plt.xlabel("BSDS training sources")
    plt.ylabel("PSNR (dB)")
    plt.title("Full-covariance K=16 training-size curve")
    plt.grid(alpha=0.25)
    plt.legend()
    _save(output / "training-size.png")


def _delta_plot(result: dict, output: Path, external: bool) -> None:
    if external:
        rows = []
        for name, baseline in result["external_reference"].items():
            value = result["external_chromatic"]["rows"][name]["cfa-mmse"]["psnr_db"]
            rows.append((name, value - baseline["methods"]["gmax"]["psnr_db"]))
        filename = "external-deltas.png"
        title = "External chromatic: GMM minus GMAX"
    else:
        rows = []
        for name, baseline in result["sampled_baselines"]["rows"].items():
            value = result["bsds_test"]["rows"][name]["cfa-mmse"]["psnr_db"]
            rows.append((name.removesuffix(".jpg"), value - baseline["methods"]["gmax"]["psnr_db"]))
        filename = "bsds-deltas.png"
        title = "Untouched BSDS: GMM minus GMAX"
    rows.sort(key=lambda row: row[1])
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [COLORS[1] if value < 0.0 else COLORS[2] for value in values]
    plt.figure(figsize=(9.4, 5.0))
    plt.bar(range(len(rows)), values, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(range(len(rows)), names, rotation=65, ha="right", fontsize=8)
    plt.ylabel("PSNR delta (dB)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    _save(output / filename)


def _synthetic_plot(result: dict, output: Path) -> None:
    rows = result["synthetic_phase"]["scenes"]
    names = [row["scene"] for row in rows]
    values = [
        row["methods"]["gmm-mmse"]["psnr_db"] - row["methods"]["gmax"]["psnr_db"]
        for row in rows
    ]
    colors = [COLORS[1] if value < 0.0 else COLORS[2] for value in values]
    plt.figure(figsize=(8.2, 4.3))
    plt.bar(range(len(rows)), values, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(range(len(rows)), names, rotation=25, ha="right")
    plt.ylabel("PSNR delta (dB)")
    plt.title("All-phase analytical controls: GMM minus GMAX")
    plt.grid(axis="y", alpha=0.25)
    _save(output / "synthetic-deltas.png")


def _calibration(result: dict, output: Path) -> None:
    bins = result["bsds_test"]["calibration"]["maximum_responsibility"]
    rms = [row["actual_rgb_rms"] for row in bins]
    agreement = [100.0 * row["component_agreement_with_oracle"] for row in bins]
    x = np.arange(1, len(bins) + 1)
    figure, left = plt.subplots(figsize=(7.2, 4.2))
    right = left.twinx()
    left.plot(x, rms, "o-", color=COLORS[0], label="RGB RMS")
    right.plot(x, agreement, "o-", color=COLORS[1], label="oracle agreement")
    left.set_xlabel("Maximum-responsibility quintile (low to high)")
    left.set_ylabel("RGB RMS", color=COLORS[0])
    right.set_ylabel("MAP/oracle agreement (%)", color=COLORS[1])
    left.set_title("Posterior confidence is useful but not component-accurate")
    left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output / "posterior-calibration.png", dpi=150,
        metadata={"Software": "RawTherapee X-Trans GMM experiment"},
    )
    plt.close(figure)


def _component_montage(model_path: Path, output: Path) -> None:
    with np.load(model_path, allow_pickle=False) as values:
        means = np.asarray(values["means"], dtype=np.float64)
        covariances = np.asarray(values["covariances"], dtype=np.float64)
        weights = np.asarray(values["weights"], dtype=np.float64)
        patch_size = int(values["patch_size"])
    components = means.shape[0]
    mean_images = means.reshape(components, 3, patch_size, patch_size).transpose(0, 2, 3, 1)
    eigen_images = []
    for covariance in covariances:
        values, vectors = np.linalg.eigh(covariance)
        vector = vectors[:, -1]
        if float(np.sum(vector)) < 0.0:
            vector = -vector
        eigen_images.append(vector.reshape(3, patch_size, patch_size).transpose(1, 2, 0))
    eigen_images = np.asarray(eigen_images)
    mean_scale = float(np.quantile(np.abs(mean_images), 0.995))
    eigen_scale = float(np.quantile(np.abs(eigen_images), 0.995))
    mean_display = np.clip(0.5 + mean_images / (2.0 * mean_scale), 0.0, 1.0)
    eigen_display = np.clip(0.5 + eigen_images / (2.0 * eigen_scale), 0.0, 1.0)
    figure, axes = plt.subplots(4, 8, figsize=(12.0, 6.4))
    for component in range(components):
        row = component // 4
        column = 2 * (component % 4)
        axes[row, column].imshow(mean_display[component], interpolation="nearest")
        axes[row, column].set_title(f"K{component} mean\nw={weights[component]:.3f}", fontsize=8)
        axes[row, column + 1].imshow(eigen_display[component], interpolation="nearest")
        axes[row, column + 1].set_title("leading mode", fontsize=8)
        axes[row, column].axis("off")
        axes[row, column + 1].axis("off")
    figure.suptitle("Selected 7x7/K=16 normalized RGB component structure")
    figure.tight_layout()
    figure.savefig(
        output / "components.png", dpi=150,
        metadata={"Software": "RawTherapee X-Trans GMM experiment"},
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = json.loads(arguments.results.read_text(encoding="utf-8"))
    supplement = json.loads(arguments.supplement.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    _component_curve(result, arguments.output)
    _training_curve(supplement, arguments.output)
    _delta_plot(result, arguments.output, external=False)
    _delta_plot(result, arguments.output, external=True)
    _synthetic_plot(result, arguments.output)
    _calibration(result, arguments.output)
    _component_montage(arguments.model, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate deterministic X-Trans structured sparse-recovery diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import canonical_json_bytes, sha256_file
from .analysis import (
    alias_dictionary,
    build_recovery_characterization,
    coefficient_relative_error,
    competing_support,
    descriptor,
    orthogonal_matching_pursuit,
)


BACKGROUND = (248, 248, 248)
TEXT = (24, 24, 24)
BLUE = (36, 111, 181)
ORANGE = (230, 126, 34)
GREEN = (39, 145, 84)
RED = (190, 38, 43)

NATURAL_SOURCES = (
    {
        "category": "portrait_and_chromatic_edges",
        "filename": "astronaut.png",
        "license": "public domain; no known copyright restrictions",
        "name": "scikit-image astronaut",
        "sha256": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    },
    {
        "category": "saturated_objects_and_texture",
        "filename": "coffee.png",
        "license": "CC0, photographer Rachel Michetti",
        "name": "scikit-image coffee",
        "sha256": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    },
    {
        "category": "fine_colored_texture",
        "filename": "hubble_deep_field.jpg",
        "license": "NASA public domain",
        "name": "scikit-image Hubble Deep Field",
        "sha256": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
    },
    {
        "category": "smooth_regions_edges_and_fine_detail",
        "filename": "rocket.jpg",
        "license": "SpaceX public domain",
        "name": "scikit-image rocket",
        "sha256": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
    },
)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _find_skimage_data() -> Path:
    try:
        import skimage.data
    except ImportError as error:
        raise RuntimeError("scikit-image 0.26.0 is required for the pinned RGB corpus") from error
    import skimage

    if skimage.__version__ != "0.26.0":
        raise RuntimeError(f"expected scikit-image 0.26.0, found {skimage.__version__}")
    return Path(skimage.data.data_dir)


def load_natural_sources() -> list[dict[str, object]]:
    data_dir = _find_skimage_data()
    sources = []
    for binding in NATURAL_SOURCES:
        path = data_dir / str(binding["filename"])
        if not path.is_file():
            raise RuntimeError(f"missing pinned natural image: {binding['filename']}")
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise RuntimeError(f"natural image digest mismatch: {binding['filename']}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        source = dict(binding)
        source["rgb"] = rgb
        sources.append(source)
    return sources


def _canvas(title: str, subtitle: str, width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill=TEXT, font=_font())
    draw.text((12, 30), subtitle, fill=(70, 70, 70), font=_font())
    return image, draw


def _line_plot(
    title: str,
    subtitle: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    *,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> Image.Image:
    image, draw = _canvas(title, subtitle, 900, 500)
    left, top, right, bottom = 88, 70, 850, 430
    draw.rectangle((left, top, right, bottom), outline=(50, 50, 50))
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        value = y_min + tick * (y_max - y_min) / 5
        draw.line((left, y, right, y), fill=(220, 220, 220))
        draw.text((24, y - 5), f"{value:.2f}", fill=TEXT, font=_font())
    minimum_x = min(x_values)
    maximum_x = max(x_values)
    denominator_x = max(maximum_x - minimum_x, 1e-12)
    for index, (name, values, color) in enumerate(series):
        points = []
        for x, value in zip(x_values, values):
            px = left + (x - minimum_x) / denominator_x * (right - left)
            py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.text((left + 12 + index * 190, 450), name, fill=color, font=_font())
    for index, value in enumerate(x_values):
        if len(x_values) <= 12 or index % 2 == 0:
            x = left + (value - minimum_x) / denominator_x * (right - left)
            draw.text((x - 8, bottom + 8), f"{value:g}", fill=TEXT, font=_font())
    return image


def _oracle_plot(rows: Sequence[Mapping[str, object]]) -> Image.Image:
    x = [float(row["support_size"]) for row in rows]
    return _line_plot(
        "Oracle support recoverability",
        "Fraction of sampled supports that are full column rank",
        x,
        (("full-rank fraction", [float(row["full_rank_fraction"]) for row in rows], BLUE),),
    )


def _blind_plot(rows: Sequence[Mapping[str, object]]) -> Image.Image:
    x = [float(row["support_size"]) for row in rows]
    return _line_plot(
        "Blind support recovery",
        "Exact support; full 18-replica OMP versus one-observed-peak control",
        x,
        (
            ("replica-aware OMP", [float(row["replica_aware"]["exact"]) for row in rows], BLUE),
            ("single-peak control", [float(row["single_peak"]["exact"]) for row in rows], ORANGE),
            ("oracle full rank", [float(row["oracle_full_rank_fraction"]) for row in rows], GREEN),
        ),
    )


def _noise_plot(rows: Sequence[Mapping[str, object]]) -> Image.Image:
    finite_snr = (20.0, 30.0, 40.0, 60.0, 80.0)
    series = []
    colors = (BLUE, GREEN, ORANGE, RED, (125, 80, 160))
    for support_size, color in zip((1, 2, 3, 4, 5), colors):
        lookup = {
            (80.0 if row["snr_db"] == "infinite" else float(row["snr_db"])): float(row["exact_support_rate"])
            for row in rows
            if int(row["support_size"]) == support_size
        }
        series.append((f"K={support_size}", [lookup[value] for value in finite_snr], color))
    return _line_plot(
        "Replica-aware recovery versus observation SNR",
        "The 80 dB endpoint represents exact arithmetic",
        finite_snr,
        series,
    )


def _bar_plot(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    maximum: float,
) -> Image.Image:
    image, draw = _canvas(title, subtitle, 900, 130 + len(labels) * 54)
    left, width = 250, 570
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 68 + index * 54
        draw.text((12, y + 9), label, fill=TEXT, font=_font())
        draw.rectangle((left, y, left + width, y + 28), outline=(50, 50, 50))
        extent = int(round(min(1.0, value / maximum) * (width - 2))) if maximum else 0
        draw.rectangle((left + 1, y + 1, left + 1 + extent, y + 27), fill=BLUE)
        draw.text((left + width + 12, y + 9), f"{value:.4g}", fill=TEXT, font=_font())
    return image


def _confidence_plot(payload: Mapping[str, object]) -> Image.Image:
    populations = payload["populations"]
    labels = ("correct", "incorrect", "deficient")
    values = [
        0.0 if populations[name]["gap_median"] is None else float(populations[name]["gap_median"])
        for name in labels
    ]
    return _bar_plot(
        "Competing-support residual gap",
        "Median alternative-fit gap; larger means the selected support is easier to distinguish",
        labels,
        values,
        maximum=max(values) if max(values) else 1.0,
    )


def _natural_plot(payload: Mapping[str, object]) -> Image.Image:
    summaries = payload["by_dominant_frequency_band"]["high"]
    labels = ("L top-1", "L top-3", "C1 top-1", "C1 top-3", "C2 top-1", "C2 top-3", "mixed top-5")
    values = (
        summaries["luma"]["top_energy_fraction"]["1"],
        summaries["luma"]["top_energy_fraction"]["3"],
        summaries["c1_r_minus_b"]["top_energy_fraction"]["1"],
        summaries["c1_r_minus_b"]["top_energy_fraction"]["3"],
        summaries["c2_green_opponent"]["top_energy_fraction"]["1"],
        summaries["c2_green_opponent"]["top_energy_fraction"]["3"],
        summaries["mixed"]["top_energy_fraction"]["5"],
    )
    return _bar_plot(
        "High-frequency natural alias-family energy concentration",
        "Families whose strongest coefficient is at radius >= 0.25 cycles/pixel",
        labels,
        [float(value) for value in values],
        maximum=1.0,
    )


def _natural_effective_plot(payload: Mapping[str, object]) -> Image.Image:
    summaries = payload["summaries"]
    labels = ("L median", "L p90", "C1 median", "C1 p90", "C2 median", "C2 p90", "mixed median", "mixed p90")
    values = []
    for name in ("luma", "c1_r_minus_b", "c2_green_opponent", "mixed"):
        values.extend(
            (
                float(summaries[name]["effective_sparsity"]["energy_weighted_median"]),
                float(summaries[name]["effective_sparsity"]["energy_weighted_p90"]),
            )
        )
    return _bar_plot(
        "Natural-image effective sparsity",
        "Energy-weighted participation ratio within 18-component direction families (54 mixed)",
        labels,
        values,
        maximum=max(values),
    )


def _example_fit(payload: dict[str, object]) -> Image.Image:
    atoms = alias_dictionary()
    support = (2, 19, 47)
    coefficients = np.asarray((1.0 + 0.2j, -0.55 + 0.35j, 0.25 - 0.4j))
    observation = atoms[:, support] @ coefficients
    best = orthogonal_matching_pursuit(observation, 3, dictionary=atoms)
    alternative = competing_support(observation, best.support, 3, dictionary=atoms)
    best_observation = atoms[:, best.support] @ best.coefficients
    alternative_observation = atoms[:, alternative.support] @ alternative.coefficients
    payload["example_replica_fit"] = {
        "best_relative_residual": best.relative_residual,
        "best_support": [descriptor(index) for index in best.support],
        "coefficient_relative_error": coefficient_relative_error(
            support, coefficients, best.support, best.coefficients
        ),
        "competing_relative_residual": alternative.relative_residual,
        "competing_support": [descriptor(index) for index in alternative.support],
        "truth_support": [descriptor(index) for index in support],
    }
    image, draw = _canvas(
        "Representative three-atom replica fit",
        "Magnitude of 18 observed bins, selected fit, and best forced-different interpretation",
        1000,
        460,
    )
    left, top, right, bottom = 70, 70, 960, 390
    maximum = max(float(np.max(np.abs(observation))), 1e-12)
    series = (
        ("observed", observation, BLUE),
        ("best", best_observation, GREEN),
        ("competitor", alternative_observation, ORANGE),
    )
    for series_index, (name, values, color) in enumerate(series):
        points = []
        for index, value in enumerate(np.abs(values)):
            x = left + index / 17.0 * (right - left)
            y = bottom - float(value) / maximum * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
        draw.text((left + series_index * 170, 420), name, fill=color, font=_font())
    draw.rectangle((left, top, right, bottom), outline=TEXT)
    return image


def _synthetic_oracle_plot(images: Mapping[str, np.ndarray]) -> Image.Image:
    scenes = ("vertical_red_green", "diagonal_red_green", "abc_saturated_edges", "abc_frequency_sweep")
    tile = 192
    image, draw = _canvas(
        "Oracle-support image reconstruction",
        "Each row: ground truth | oracle | blind replica OMP | 8x oracle error",
        tile * 4 + 220,
        70 + tile * len(scenes),
    )
    for row, name in enumerate(scenes):
        truth = np.moveaxis(images[f"{name}_ground_truth"], 0, -1)
        recovered = np.moveaxis(images[f"{name}_oracle"], 0, -1)
        blind = np.moveaxis(images[f"{name}_blind_replica"], 0, -1)
        error = np.clip(np.abs(recovered - truth) * 8.0, 0.0, 1.0)
        y = 64 + row * tile
        draw.text((8, y + 80), name, fill=TEXT, font=_font())
        for column, values in enumerate((truth, recovered, blind, error)):
            rendered = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255.0 + 0.5), "RGB")
            rendered = rendered.resize((tile, tile), Image.Resampling.NEAREST)
            image.paste(rendered, (220 + column * tile, y))
    return image


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def generate(output_dir: Path, force: bool) -> tuple[Path, str]:
    output_dir = output_dir.resolve()
    plot_names = (
        "oracle-recoverability.png",
        "blind-support-recovery.png",
        "recovery-versus-snr.png",
        "confidence-gap.png",
        "natural-energy-concentration.png",
        "natural-effective-sparsity.png",
        "example-replica-fit.png",
        "synthetic-oracle-reconstruction.png",
    )
    manifest_name = "recovery-characterization.json"
    filenames = (*plot_names, manifest_name)
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing output: {existing[0]}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtrans-sparse-alias-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        payload, oracle_images = build_recovery_characterization(load_natural_sources())
        images = {
            "oracle-recoverability.png": _oracle_plot(payload["oracle_support"]),
            "blind-support-recovery.png": _blind_plot(payload["blind_support"]),
            "recovery-versus-snr.png": _noise_plot(payload["noise"]),
            "confidence-gap.png": _confidence_plot(payload["confidence"]),
            "natural-energy-concentration.png": _natural_plot(payload["natural_sparsity"]),
            "natural-effective-sparsity.png": _natural_effective_plot(payload["natural_sparsity"]),
            "example-replica-fit.png": _example_fit(payload),
            "synthetic-oracle-reconstruction.png": _synthetic_oracle_plot(oracle_images),
        }
        for name, image in images.items():
            _save(image, staging / name)
        payload["plots"] = [
            {
                "bytes": (staging / name).stat().st_size,
                "filename": name,
                "sha256": sha256_file(staging / name),
            }
            for name in sorted(images)
        ]
        manifest_bytes = canonical_json_bytes(payload)
        (staging / manifest_name).write_bytes(manifest_bytes)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in plot_names:
            os.replace(staging / name, output_dir / name)
        os.replace(staging / manifest_name, output_dir / manifest_name)
    manifest = output_dir / manifest_name
    return manifest, sha256_file(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        manifest, digest = generate(arguments.output_dir, arguments.force)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"manifest={manifest}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

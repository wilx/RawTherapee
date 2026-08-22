#!/usr/bin/env python3
"""Generate the canonical X-Trans alias characterization and plots."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analysis import (
    CARRIER_INDICES,
    COLOR_NAMES,
    DIRECTIONS,
    ORTHONORMAL_DIRECTION_NAMES,
    build_characterization,
    canonical_json_bytes,
    centered_frequency,
    complex_confusion_table,
    mask_spectra,
    sha256_file,
)


BACKGROUND = (248, 248, 248)
TEXT = (24, 24, 24)


def _color(value: float) -> tuple[int, int, int]:
    value = min(1.0, max(0.0, value))
    stops = (
        (0.00, (18, 12, 68)),
        (0.25, (30, 67, 174)),
        (0.50, (32, 174, 190)),
        (0.75, (247, 218, 72)),
        (1.00, (180, 20, 28)),
    )
    for (left_position, left), (right_position, right) in zip(stops, stops[1:]):
        if value <= right_position:
            fraction = (value - left_position) / (right_position - left_position)
            return tuple(
                int(round(a + fraction * (b - a))) for a, b in zip(left, right)
            )
    return stops[-1][1]


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_heatmap(
    values: np.ndarray,
    title: str,
    subtitle: str,
    minimum: float | None = None,
    maximum: float | None = None,
    cell: int = 18,
    annotations: list[list[str]] | None = None,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    x_axis_label: str = "fx cycles/pixel",
    y_axis_label: str = "fy",
) -> Image.Image:
    values = np.asarray(values, dtype=np.float64)
    height, width = values.shape
    minimum = float(np.min(values)) if minimum is None else minimum
    maximum = float(np.max(values)) if maximum is None else maximum
    left = 92
    top = 72
    right = 100
    bottom = 64
    image = Image.new("RGB", (left + width * cell + right, top + height * cell + bottom), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((12, 10), title, fill=TEXT, font=font)
    draw.text((12, 30), subtitle, fill=(70, 70, 70), font=font)
    denominator = maximum - minimum
    for y in range(height):
        for x in range(width):
            normalized = 0.5 if denominator == 0.0 else (values[y, x] - minimum) / denominator
            box = (
                left + x * cell,
                top + y * cell,
                left + (x + 1) * cell - 1,
                top + (y + 1) * cell - 1,
            )
            draw.rectangle(box, fill=_color(normalized))
            if annotations is not None:
                label = annotations[y][x]
                draw.text((box[0] + 2, box[1] + 2), label, fill=(255, 255, 255), font=font)
    draw.rectangle(
        (left - 1, top - 1, left + width * cell, top + height * cell),
        outline=(30, 30, 30),
    )
    if x_labels is not None and len(x_labels) != width:
        raise ValueError("x label count does not match heatmap width")
    if y_labels is not None and len(y_labels) != height:
        raise ValueError("y label count does not match heatmap height")
    tick_positions = (
        range(width)
        if x_labels is not None
        else sorted(set((0, width // 4, width // 2, 3 * width // 4, width - 1)))
    )
    shifted_bins = np.fft.fftshift(np.arange(width))
    for position in tick_positions:
        label = (
            x_labels[position]
            if x_labels is not None
            else f"{centered_frequency(int(shifted_bins[position]), width):+.2f}"
        )
        draw.text(
            (left + position * cell - 12, top + height * cell + 8),
            label,
            fill=TEXT,
            font=font,
        )
    tick_positions_y = (
        range(height)
        if y_labels is not None
        else sorted(set((0, height // 4, height // 2, 3 * height // 4, height - 1)))
    )
    shifted_y = np.fft.fftshift(np.arange(height))
    for position in tick_positions_y:
        label = (
            y_labels[position]
            if y_labels is not None
            else f"{centered_frequency(int(shifted_y[position]), height):+.2f}"
        )
        draw.text(
            (12, top + position * cell + 3),
            label,
            fill=TEXT,
            font=font,
        )
    bar_x = left + width * cell + 28
    for row in range(height * cell):
        normalized = 1.0 - row / max(1, height * cell - 1)
        draw.line((bar_x, top + row, bar_x + 18, top + row), fill=_color(normalized))
    draw.text((bar_x + 24, top - 4), f"{maximum:.3g}", fill=TEXT, font=font)
    draw.text((bar_x + 24, top + height * cell - 10), f"{minimum:.3g}", fill=TEXT, font=font)
    draw.text((left + width * cell // 2 - 45, top + height * cell + 35), x_axis_label, fill=TEXT, font=font)
    draw.text((12, top - 18), y_axis_label, fill=TEXT, font=font)
    return image


def _coefficient_label(value: complex) -> str:
    real = int(round(float(value.real) * 36.0))
    imag = int(round(float(value.imag) * 36.0 / math.sqrt(3.0)))
    if real == 0 and imag == 0:
        return "0"
    if imag == 0:
        return f"{real}/36"
    return f"{real:+d}{imag:+d}s3i"


def _mask_plot(color: int) -> Image.Image:
    spectrum = mask_spectra()[color]
    magnitude = np.fft.fftshift(np.abs(spectrum))
    shifted = np.fft.fftshift(spectrum)
    annotations = [
        [_coefficient_label(complex(shifted[y, x])) for x in range(6)]
        for y in range(6)
    ]
    return _draw_heatmap(
        magnitude,
        f"X-Trans {COLOR_NAMES[color]} mask spectrum",
        "Cell text is (a + b*sqrt(3)i)/36; color is magnitude",
        minimum=0.0,
        maximum=5.0 / 9.0,
        cell=72,
        annotations=annotations,
    )


def _frequency_response_plot() -> Image.Image:
    size = 96
    source = (7, 11)
    spectra = mask_spectra()
    panels = []
    for name in ORTHONORMAL_DIRECTION_NAMES:
        observed = np.zeros((size, size), dtype=np.float64)
        for kx, ky in CARRIER_INDICES:
            ox = (source[0] + kx * size // 6) % size
            oy = (source[1] + ky * size // 6) % size
            observed[oy, ox] = abs(np.dot(spectra[:, ky, kx], DIRECTIONS[name]))
        log_magnitude = np.log10(np.maximum(observed, 1e-6))
        panels.append(
            _draw_heatmap(
                np.fft.fftshift(log_magnitude),
                f"Sampled spectrum: {name}",
                "Input bin (7,11); display is log10 magnitude",
                minimum=-6.0,
                maximum=0.0,
                cell=5,
            )
        )
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), BACKGROUND)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def _complex_confusion_plot(
    table: list[dict[str, object]] | None = None,
    title: str = "Complex single-frequency confusion",
) -> Image.Image:
    names = ORTHONORMAL_DIRECTION_NAMES
    table = complex_confusion_table() if table is None else table
    values = np.zeros((3, 3), dtype=np.float64)
    annotations = [["" for _ in names] for _ in names]
    for row in table:
        y = names.index(str(row["source"]))
        x = names.index(str(row["competitor"]))
        values[y, x] = float(row["similarity"])
        annotations[y][x] = f"{values[y, x]:.3f}"
    return _draw_heatmap(
        values,
        title,
        "Rows source, columns competitor; maximum over carrier displacement",
        minimum=0.0,
        maximum=1.0,
        cell=110,
        annotations=annotations,
        x_labels=("L", "C1", "C2"),
        y_labels=("L", "C1", "C2"),
        x_axis_label="competitor",
        y_axis_label="source",
    )


def _nullity_plot() -> Image.Image:
    values = np.full((24, 24), 36.0, dtype=np.float64)
    return _draw_heatmap(
        np.fft.fftshift(values),
        "Full RGB alias-family nullity",
        "18 observations x 54 source coefficients: rank 18, nullity 36 everywhere",
        minimum=0.0,
        maximum=36.0,
        cell=18,
    )


def _edge_plot(edge_cases: dict[str, object]) -> Image.Image:
    names = list(edge_cases)
    width = 1000
    height = 90 + len(names) * 44
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((12, 10), "Energy in rank-deficient active alias families", fill=TEXT, font=font)
    draw.text((12, 30), "Channel means removed; ground-truth coefficients, activity threshold 1e-8", fill=(70, 70, 70), font=font)
    bar_left = 260
    bar_width = 650
    for index, name in enumerate(names):
        y = 64 + index * 44
        value = float(edge_cases[name]["rank_deficient_energy_fraction"])
        draw.text((12, y + 8), name, fill=TEXT, font=font)
        draw.rectangle((bar_left, y, bar_left + bar_width, y + 26), outline=(40, 40, 40))
        draw.rectangle(
            (bar_left + 1, y + 1, bar_left + 1 + int(round(value * (bar_width - 2))), y + 25),
            fill=_color(value),
        )
        draw.text((bar_left + bar_width + 14, y + 8), f"{value:.4f}", fill=TEXT, font=font)
    return image


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def generate(output_dir: Path, force: bool) -> tuple[Path, str]:
    output_dir = output_dir.resolve()
    filenames = (
        "mask-spectrum-red.png",
        "mask-spectrum-green.png",
        "mask-spectrum-blue.png",
        "single-frequency-response.png",
        "complex-confusion.png",
        "bayer-complex-confusion.png",
        "real-luma-chroma-confusion.png",
        "real-chroma-confusion.png",
        "full-alias-nullity.png",
        "edge-family-ambiguity.png",
        "characterization.json",
    )
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing output: {existing[0]}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtrans-alias-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        payload, maps = build_characterization()
        images = {
            "mask-spectrum-red.png": _mask_plot(0),
            "mask-spectrum-green.png": _mask_plot(1),
            "mask-spectrum-blue.png": _mask_plot(2),
            "single-frequency-response.png": _frequency_response_plot(),
            "complex-confusion.png": _complex_confusion_plot(),
            "bayer-complex-confusion.png": _complex_confusion_plot(
                payload["bayer_context"]["complex_confusion"],
                "Bayer complex single-frequency confusion",
            ),
            "real-luma-chroma-confusion.png": _draw_heatmap(
                np.fft.fftshift(maps["luma_chroma"]),
                "Worst phase-sensitive luma/chroma confusion",
                "24x24 FFT grid; four source and competitor phases",
                minimum=0.0,
                maximum=1.0,
                cell=18,
            ),
            "real-chroma-confusion.png": _draw_heatmap(
                np.fft.fftshift(maps["chroma_chroma"]),
                "Worst phase-sensitive chroma/chroma confusion",
                "C1 source against C2; four source and competitor phases",
                minimum=0.0,
                maximum=1.0,
                cell=18,
            ),
            "full-alias-nullity.png": _nullity_plot(),
            "edge-family-ambiguity.png": _edge_plot(payload["edge_cases"]),
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
        (staging / "characterization.json").write_bytes(manifest_bytes)

        output_dir.mkdir(parents=True, exist_ok=True)
        # Publish the manifest last as the completion marker.
        for name in filenames[:-1]:
            os.replace(staging / name, output_dir / name)
        os.replace(staging / "characterization.json", output_dir / "characterization.json")
    manifest = output_dir / "characterization.json"
    return manifest, sha256_file(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        manifest, digest = generate(arguments.output_dir, arguments.force)
    except (FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"manifest={manifest}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

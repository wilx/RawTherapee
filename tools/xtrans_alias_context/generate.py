#!/usr/bin/env python3
"""Generate deterministic windowed X-Trans alias-context diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import canonical_json_bytes, sha256_file
from tools.xtrans_sparse_alias.generate import load_natural_sources
from .analysis import build_context_characterization


BACKGROUND = (248, 248, 248)
TEXT = (24, 24, 24)
BLUE = (36, 111, 181)
ORANGE = (230, 126, 34)
GREEN = (39, 145, 84)
RED = (190, 38, 43)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str, width: int = 940, height: int = 520) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((14, 10), title, fill=TEXT, font=_font())
    draw.text((14, 30), subtitle, fill=(70, 70, 70), font=_font())
    return image, draw


def _bar_plot(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    *,
    maximum: float | None = None,
) -> Image.Image:
    image, draw = _canvas(title, subtitle, height=150 + len(labels) * 58)
    left, width = 270, 560
    all_values = [float(value) for _name, values, _color in series for value in values]
    scale = maximum if maximum is not None else max(all_values, default=1.0)
    scale = max(scale, 1e-12)
    band = 38 / max(1, len(series))
    for row, label in enumerate(labels):
        y = 68 + row * 58
        draw.text((12, y + 11), label, fill=TEXT, font=_font())
        draw.rectangle((left, y, left + width, y + 42), outline=(80, 80, 80))
        for index, (name, values, color) in enumerate(series):
            value = float(values[row])
            top = y + 2 + int(index * band)
            bottom = y + 2 + int((index + 1) * band) - 1
            extent = int(round(min(1.0, max(0.0, value / scale)) * (width - 3)))
            draw.rectangle((left + 2, top, left + 2 + extent, bottom), fill=color)
            draw.text((left + width + 10, top), f"{value:.4f}", fill=color, font=_font())
    for index, (name, _values, color) in enumerate(series):
        draw.text((left + index * 190, 72 + len(labels) * 58), name, fill=color, font=_font())
    return image


def _proposal_plot(payload: Mapping[str, object]) -> Image.Image:
    scenes = payload["synthetic"]["scenes"]
    names = list(scenes)
    return _bar_plot(
        "Truth support retained by local proposal generation",
        "Exact top-K ground-truth support present among distinct local hypotheses",
        names,
        tuple(
            (
                f"top {count}",
                [float(scenes[name]["proposals"][str(count)]["truth_in_top_n"]) for name in names],
                color,
            )
            for count, color in ((1, BLUE), (3, GREEN), (5, ORANGE), (8, RED))
        ),
        maximum=1.0,
    )


def _context_plot(payload: Mapping[str, object]) -> Image.Image:
    scenes = payload["synthetic"]["scenes"]
    names = list(scenes)
    modes = (("local_only", "local", BLUE), ("support_only", "support only", ORANGE), ("phase_only", "phase only", GREEN), ("full", "all terms", RED))
    maximum = max(
        float(scenes[name]["context_modes"][mode]["coefficient_error_mean"])
        for name in names for mode, _label, _color in modes
    )
    return _bar_plot(
        "Contextual coefficient error by synthetic scene",
        "Lower is better; support-only is the generic continuity control",
        names,
        tuple(
            (
                label,
                [float(scenes[name]["context_modes"][mode]["coefficient_error_mean"]) for name in names],
                color,
            )
            for mode, label, color in modes
        ),
        maximum=maximum,
    )


def _natural_structure_plot(payload: Mapping[str, object]) -> Image.Image:
    structures = payload["natural"]["structures"]
    names = list(structures)
    maximum = max(
        float(structures[name][key]) for name in names for key in ("local_error", "context_error")
    )
    return _bar_plot(
        "Natural-window recovery by approximate structure",
        "Authenticated linear-RGB inputs; lower coefficient error is better",
        names,
        (
            ("local", [float(structures[name]["local_error"]) for name in names], BLUE),
            ("context", [float(structures[name]["context_error"]) for name in names], RED),
        ),
        maximum=maximum,
    )


def _confidence_plot(payload: Mapping[str, object]) -> Image.Image:
    curve = payload["natural"]["confidence_calibration"]["curve"]
    labels = [f"top {100 * float(row['retained_fraction']):g}%" for row in curve]
    return _bar_plot(
        "Natural confidence calibration",
        "Families retained from highest contextual confidence; lower error is better",
        labels,
        (("coefficient error", [float(row["coefficient_error_mean"]) for row in curve], BLUE),),
    )


def _window_plot(payload: Mapping[str, object]) -> Image.Image:
    rows = payload["ablation"]["window_shift"]
    labels = [f"{row['window_size']} px / shift {row['shift']}" for row in rows]
    maximum = max(float(row[key]) for row in rows for key in ("local_error", "context_error"))
    return _bar_plot(
        "Window size and shift diagnostic",
        "Saturated red/gray edge; non-CFA-aligned shift 7 is included",
        labels,
        (
            ("local", [float(row["local_error"]) for row in rows], BLUE),
            ("context", [float(row["context_error"]) for row in rows], RED),
        ),
        maximum=maximum,
    )


def _confidence_maps(grids: Mapping[str, np.ndarray]) -> Image.Image:
    names = list(grids)
    tile = 150
    image, draw = _canvas(
        "Spatial contextual-confidence maps",
        "Blue is ambiguous, yellow/red is high confidence; one 3x3 grid per synthetic scene",
        width=40 + tile * len(names),
        height=240,
    )
    maximum = max(float(np.max(grid)) for grid in grids.values())
    minimum = min(float(np.min(grid)) for grid in grids.values())
    scale = max(maximum - minimum, 1e-12)
    for index, (name, grid) in enumerate(grids.items()):
        normalized = (grid - minimum) / scale
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.uint8(np.clip(2.0 * normalized - 0.2, 0.0, 1.0) * 255)
        rgb[..., 1] = np.uint8(np.clip(1.5 * normalized, 0.0, 1.0) * 220)
        rgb[..., 2] = np.uint8(np.clip(1.0 - normalized, 0.0, 1.0) * 255)
        rendered = Image.fromarray(rgb, "RGB").resize((tile, tile), Image.Resampling.NEAREST)
        x = 20 + index * tile
        image.paste(rendered, (x, 62))
        draw.text((x + 4, 212), name, fill=TEXT, font=_font())
    return image


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def render_artifacts(
    payload: dict[str, object],
    confidence_grids: Mapping[str, np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    images = {
        "proposal-survival.png": _proposal_plot(payload),
        "context-ablation.png": _context_plot(payload),
        "natural-structure-recovery.png": _natural_structure_plot(payload),
        "confidence-calibration.png": _confidence_plot(payload),
        "window-shift-ablation.png": _window_plot(payload),
        "spatial-confidence-maps.png": _confidence_maps(confidence_grids),
    }
    for name, image in images.items():
        _save(image, output_dir / name)
    payload["plots"] = [
        {
            "bytes": (output_dir / name).stat().st_size,
            "filename": name,
            "sha256": sha256_file(output_dir / name),
        }
        for name in sorted(images)
    ]
    return payload, tuple(images)


def generate(output_dir: Path, force: bool) -> tuple[Path, str]:
    output_dir = output_dir.resolve()
    manifest_name = "context-characterization.json"
    plot_names = (
        "proposal-survival.png",
        "context-ablation.png",
        "natural-structure-recovery.png",
        "confidence-calibration.png",
        "window-shift-ablation.png",
        "spatial-confidence-maps.png",
    )
    existing = [output_dir / name for name in (*plot_names, manifest_name) if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing output: {existing[0]}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtrans-alias-context-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        payload, grids = build_context_characterization(load_natural_sources())
        payload, generated = render_artifacts(payload, grids, staging)
        (staging / manifest_name).write_bytes(canonical_json_bytes(payload))
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in generated:
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

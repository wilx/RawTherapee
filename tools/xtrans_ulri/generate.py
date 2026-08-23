#!/usr/bin/env python3
"""Run the ULRI pass-count and structural-comparison experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_mlri_green_fusion.dataset import (
    dataset_manifest as base_dataset_manifest,
    load_cases,
)
from tools.xtrans_mlri_internal.dataset import mosaic
from tools.xtrans_mlri_internal.generate import srgb_encode
from tools.xtrans_ulri.analysis import MARGIN, method_metrics, patch_oracle, pooled_metrics


METHODS = (
    "markesteijn", "corrected-final",
    "ulri-slow0", "ulri-slow1", "ulri-slow2", "ulri-slow3",
)
ULRI_METHODS = tuple(name for name in METHODS if name.startswith("ulri-"))
TIMING_RE = re.compile(r"^(\S+)\t([0-9.eE+-]+)$")
REFERENCE_MANIFEST_SHA256 = "2441cb88637a5c50375093d11e97b6cb46912d68201b338c3f8d4a5c4e7114c3"
PARITY_MAXIMUM_BOUNDS = {
    "ulri-slow0": 0.031,
    "ulri-slow1": 0.031,
    "ulri-slow2": 0.047,
    "ulri-slow3": 0.062,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "infinity" if float(value) > 0 else "-infinity"
        if math.isnan(float(value)):
            raise ValueError("NaN cannot be serialized")
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _read_rgb(path: Path, height: int, width: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != 3 * height * width:
        raise RuntimeError(f"incorrect RGB payload size: {path}")
    result = values.reshape(3, height, width).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite RGB payload: {path}")
    return result


def _write_mosaic(path: Path, value: np.ndarray) -> None:
    checked = np.asarray(value, dtype="<f4")
    if checked.ndim != 2 or not np.isfinite(checked).all():
        raise ValueError("mosaic must be a finite scalar image")
    path.write_bytes(checked.tobytes(order="C"))


def _run(
    runner: Path, truth: np.ndarray, origin_x: int, origin_y: int, directory: Path
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    scalar, cfa = mosaic(truth, origin_x, origin_y)
    input_path = directory / "input.f32le"
    _write_mosaic(input_path, scalar)
    process = subprocess.run(
        [
            str(runner), "run", str(input_path), str(directory),
            str(truth.shape[2]), str(truth.shape[1]), str(origin_x), str(origin_y),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    timings = {}
    for line in process.stdout.splitlines():
        match = TIMING_RE.match(line)
        if match:
            timings[match.group(1)] = float(match.group(2))
    if set(timings) != set(METHODS):
        raise RuntimeError(f"runner timing contract differs: {process.stdout}")
    outputs = {
        method: _read_rgb(directory / f"{method}.f32le", truth.shape[1], truth.shape[2])
        for method in METHODS
    }
    return outputs, timings, cfa


def _reference_parity(runner: Path, golden: Path, temporary: Path) -> dict[str, object]:
    manifest_path = golden / "manifest.json"
    if _sha256(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise RuntimeError("ULRI reference manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    rows = []
    maxima = defaultdict(float)
    squared = defaultdict(float)
    counts = defaultdict(int)
    interior_maxima = defaultdict(float)
    interior_squared = defaultdict(float)
    interior_counts = defaultdict(int)
    for entry in manifest["cases"]:
        case_dir = temporary / f"reference-{entry['name']}"
        case_dir.mkdir()
        width, height = int(entry["width"]), int(entry["height"])
        process = subprocess.run(
            [
                str(runner), "run", str(golden / entry["mosaic"]["file"]),
                str(case_dir), str(width), str(height), "0", "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        del process
        per_method = {}
        for output in entry["outputs"]:
            method = f"ulri-slow{output['slow']}"
            native = _read_rgb(case_dir / f"{method}.f32le", height, width)
            reference = _read_rgb(golden / output["file"], height, width)
            difference = native - reference
            interior = difference[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
            maximum = float(np.max(np.abs(difference)))
            rms = float(np.sqrt(np.mean(difference**2)))
            interior_maximum = float(np.max(np.abs(interior)))
            interior_rms = float(np.sqrt(np.mean(interior**2)))
            maxima[method] = max(maxima[method], maximum)
            squared[method] += float(np.sum(difference**2))
            counts[method] += difference.size
            interior_maxima[method] = max(
                interior_maxima[method], interior_maximum
            )
            interior_squared[method] += float(np.sum(interior**2))
            interior_counts[method] += interior.size
            per_method[method] = {
                "interior_maximum_abs": interior_maximum,
                "interior_rms": interior_rms,
                "maximum_abs": maximum,
                "rms": rms,
            }
        rows.append({"case": entry["name"], "methods": per_method})
    aggregate = {
        method: {
            "interior_maximum_abs": interior_maxima[method],
            "interior_rms": math.sqrt(
                interior_squared[method] / interior_counts[method]
            ),
            "maximum_abs": maxima[method],
            "rms": math.sqrt(squared[method] / counts[method]),
        }
        for method in ULRI_METHODS
    }
    for method, values in aggregate.items():
        if (values["maximum_abs"] > PARITY_MAXIMUM_BOUNDS[method]
                or values["rms"] > 0.002):
            raise RuntimeError(f"{method} exceeds frozen Octave parity bound: {values}")
    return {
        "aggregate": aggregate,
        "cases": rows,
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "note": (
            "differences include final uint16 quantization and Octave/C++ reduction "
            "order; the largest full-frame values are confined to the 16-pixel "
            "boundary region"
        ),
        "required_bounds": {
            "maximum_abs": PARITY_MAXIMUM_BOUNDS,
            "rms": 0.002,
        },
    }


def _case_record(source: Mapping[str, object], outputs, timings, cfa) -> dict[str, object]:
    truth = np.asarray(source["truth"], dtype=np.float64)
    return {
        "case_id": str(source["case_id"]),
        "cfa": cfa,
        "family": str(source["family"]),
        "kind": str(source["kind"]),
        "origin": [int(source["origin_x"]), int(source["origin_y"])],
        "outputs": outputs,
        "source_id": str(source["source_id"]),
        "split": str(source["split"]),
        "timings": timings,
        "truth": truth,
    }


def _aggregate(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result = {method: pooled_metrics(cases, method) for method in METHODS}
    corrected = result["corrected-final"]["psnr_db"]
    for method in METHODS:
        result[method]["delta_psnr_from_corrected_db"] = (
            result[method]["psnr_db"] - corrected
        )
    return result


def _groups(cases: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    natural = [case for case in cases if case["kind"] == "natural"]
    result = {
        "all": list(cases),
        "natural": natural,
        "natural-sparse": [
            case for case in natural
            if case["source_id"] in ("hubble", "nasa-hydra-starfield")
        ],
        "natural-coherent": [
            case for case in natural
            if case["source_id"] in ("brick", "grass", "gravel", "page")
        ],
        "synthetic-sparse": [
            case for case in cases
            if case["kind"] in ("synthetic-failure", "synthetic-transition")
        ],
        "synthetic-coherent": [
            case for case in cases
            if case["kind"] == "synthetic-success"
            and case["source_id"] not in (
                "control-frequency_sweep", "control-saturated_quadrants"
            )
        ],
        "stress": [
            case for case in cases
            if case["source_id"] in (
                "control-frequency_sweep", "control-saturated_quadrants"
            )
        ],
    }
    return {name: rows for name, rows in result.items() if rows}


def _oracles(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    image_counts = Counter()
    image_cases = []
    patch_cases = []
    for case in cases:
        best = min(
            ULRI_METHODS,
            key=lambda method: method_metrics(
                case["truth"], case["outputs"][method], case["cfa"]
            )["rgb_rms"],
        )
        image_counts[best] += 1
        image_case = dict(case)
        image_case["outputs"] = {"oracle": case["outputs"][best]}
        image_cases.append(image_case)
        patch, selected = patch_oracle(
            case["truth"], [case["outputs"][method] for method in ULRI_METHODS], 7
        )
        patch_case = dict(case)
        patch_case["outputs"] = {"oracle": patch}
        patch_cases.append(patch_case)
        case["best_ulri_image"] = best
        case["patch_selection_fraction"] = {
            method: float(np.mean(selected == index))
            for index, method in enumerate(ULRI_METHODS)
        }
    return {
        "image": {
            "metrics": pooled_metrics(image_cases, "oracle"),
            "selection_count": dict(sorted(image_counts.items())),
        },
        "patch7": {
            "metrics": pooled_metrics(patch_cases, "oracle"),
            "window": 7,
        },
    }


def _natural_sources(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = defaultdict(list)
    for case in cases:
        if case["kind"] == "natural":
            grouped[case["source_id"]].append(case)
    return {name: _aggregate(rows) for name, rows in sorted(grouped.items())}


def _transitions(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = []
    for case in cases:
        if any(token in case["family"] for token in ("transition", "thickness")):
            rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "source_id": case["source_id"],
                "methods": {
                    method: method_metrics(
                        case["truth"], case["outputs"][method], case["cfa"]
                    )
                    for method in METHODS
                },
            })
    return {"cases": rows}


def _timings(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        method: {
            "median_seconds": float(np.median([case["timings"][method] for case in cases])),
            "sum_seconds": float(sum(case["timings"][method] for case in cases)),
        }
        for method in METHODS
    }


def _panel(case: Mapping[str, object], path: Path) -> None:
    order = ("truth",) + METHODS
    images = {"truth": np.asarray(case["truth"]), **case["outputs"]}
    scale = 2
    height, width = images["truth"].shape[1:]
    label_height = 22
    canvas = Image.new(
        "RGB", (len(order) * width * scale, 3 * height * scale + 3 * label_height)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    maximum = max(
        float(np.max(np.abs(images[name] - images["truth"])))
        for name in METHODS
    )
    difference_maximum = max(
        float(np.max(np.abs(images[name] - images["corrected-final"])))
        for name in ULRI_METHODS
    )
    for column, name in enumerate(order):
        rgb = np.moveaxis(np.clip(srgb_encode(images[name]), 0, 1), 0, -1)
        rendered = Image.fromarray(np.round(rgb * 255).astype(np.uint8)).resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        x = column * width * scale
        canvas.paste(rendered, (x, label_height))
        draw.text((x + 3, 4), name, fill="white", font=font)
        if name == "truth":
            error = np.zeros_like(images["truth"])
        else:
            error = np.abs(images[name] - images["truth"]) / max(maximum, 1e-12)
        error_rgb = np.moveaxis(np.clip(error, 0, 1), 0, -1)
        error_image = Image.fromarray(np.round(error_rgb * 255).astype(np.uint8)).resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        canvas.paste(error_image, (x, height * scale + 2 * label_height))
        if name in ULRI_METHODS:
            difference = (
                0.5 + (images[name] - images["corrected-final"])
                / (2 * max(difference_maximum, 1e-12))
            )
        else:
            difference = np.full_like(images["truth"], 0.5)
        difference_rgb = np.moveaxis(np.clip(difference, 0, 1), 0, -1)
        difference_image = Image.fromarray(
            np.round(difference_rgb * 255).astype(np.uint8)
        ).resize((width * scale, height * scale), Image.Resampling.NEAREST)
        canvas.paste(difference_image, (x, 2 * height * scale + 3 * label_height))
    draw.text((3, height * scale + label_height + 4),
              f"absolute error, shared max={maximum:.5f}", fill="white", font=font)
    draw.text(
        (3, 2 * height * scale + 2 * label_height + 4),
        "signed ULRI - corrected-final, gray=zero, shared "
        f"+/-{difference_maximum:.5f}",
        fill="white", font=font,
    )
    canvas.save(path, optimize=True)


def _diagnostics(cases: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    wanted = {
        "brick": "brick",
        "hubble": "hubble",
        "nasa-hydra-starfield": "hydra",
        "page": "page",
        "control-saturated_red_gray": "saturated-edge",
        "transition-coherence-continuous-line": "continuous-line",
        "white-impulse": "impulse",
    }
    result = []
    for source_id, label in wanted.items():
        case = next((row for row in cases if row["source_id"] == source_id), None)
        if case is None:
            case = next((row for row in cases if source_id in row["case_id"]), None)
        if case is None:
            continue
        path = output / f"map-{label}.png"
        _panel(case, path)
        result.append(path)
    return result


def _json_case(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "best_ulri_image": case["best_ulri_image"],
        "case_id": case["case_id"],
        "family": case["family"],
        "kind": case["kind"],
        "methods": {
            method: method_metrics(
                case["truth"], case["outputs"][method], case["cfa"]
            )
            for method in METHODS
        },
        "origin": case["origin"],
        "patch_selection_fraction": case["patch_selection_fraction"],
        "source_id": case["source_id"],
        "split": case["split"],
        "timings": case["timings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--starfield", type=Path, required=True)
    parser.add_argument("--reference-golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runner = args.runner.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output exists: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=".xtrans-ulri-", dir=output.parent))
    work = Path(tempfile.mkdtemp(prefix="xtrans-ulri-work-"))
    try:
        parity = _reference_parity(runner, args.reference_golden.resolve(), work)
        sources = load_cases(args.starfield.resolve())
        cases = []
        for index, source in enumerate(sources):
            case_dir = work / f"case-{index:03d}"
            case_dir.mkdir()
            outputs, timings, cfa = _run(
                runner, np.asarray(source["truth"], dtype=np.float64),
                int(source["origin_x"]), int(source["origin_y"]), case_dir,
            )
            cases.append(_case_record(source, outputs, timings, cfa))
        groups = _groups(cases)
        oracle = _oracles(cases)
        results = {
            "case_count": len(cases),
            "cases": [_json_case(case) for case in cases],
            "groups": {name: _aggregate(rows) for name, rows in groups.items()},
            "methods": list(METHODS),
            "natural_sources": _natural_sources(cases),
            "oracles": oracle,
            "reference_parity": parity,
            "timings": _timings(cases),
            "transitions": _transitions(cases),
        }
        dataset = base_dataset_manifest(sources)
        dataset.update({
            "format": "rawtherapee-xtrans-ulri-dataset-v1",
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        })
        (temporary_output / "dataset.json").write_bytes(canonical_json_bytes(dataset))
        (temporary_output / "results.json").write_bytes(
            canonical_json_bytes(_json_safe(results))
        )
        images = _diagnostics(cases, temporary_output)
        manifest = {
            "dataset": {"file": "dataset.json", "sha256": _sha256(temporary_output / "dataset.json")},
            "format": "rawtherapee-xtrans-ulri-artifacts-v1",
            "images": [
                {"file": path.name, "sha256": _sha256(path)} for path in sorted(images)
            ],
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "results": {"file": "results.json", "sha256": _sha256(temporary_output / "results.json")},
        }
        (temporary_output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            shutil.rmtree(output)
        temporary_output.rename(output)
        print(f"wrote {output}")
        print(f"manifest SHA-256: {_sha256(output / 'manifest.json')}")
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

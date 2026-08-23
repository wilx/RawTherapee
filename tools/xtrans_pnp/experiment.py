#!/usr/bin/env python3
"""Run the staged X-Trans PnP-BM3D feasibility gates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

for _name in (
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_lmmse.experiment import _run_baselines
from tools.xtrans_lmmse.model import predict_region
from tools.xtrans_lmmse_mixture.dataset import (
    discover_expanded,
    external_chromatic_sources,
    load_rgb,
    manifest as lmmse_manifest,
)
from tools.xtrans_lmmse_mixture.training import survey_and_train_globals

from .denoiser import BM3DDenoiser, authenticate_bm3d
from .operator import xtrans_operator
from .solver import pnp_admm, pnp_pgm


SIGMAS = (0.002, 0.005, 0.01, 0.02, 0.04)
RHOS = (0.1, 0.3, 1.0, 3.0, 10.0)
ITERATIONS = (1, 2, 3, 5, 10, 20)
MARGIN = 12
MINIMUM_ITERATIVE_GAIN_DB = 0.01
MINIMUM_POPULATION_GAIN_DB = 0.05


def metric(output: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    checked = np.asarray(output, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    if checked.shape != reference.shape or checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("metric arrays must be matching CHW RGB")
    selected = np.zeros(reference.shape[1:], dtype=bool)
    margin = min(MARGIN, max(0, min(selected.shape) // 4))
    if margin:
        selected[margin:-margin, margin:-margin] = True
    else:
        selected[:] = True
    difference = checked[:, selected] - reference[:, selected]
    absolute = np.abs(difference)
    squared = difference * difference
    mse = float(np.mean(squared))
    return {
        "count": int(difference.size),
        "maximum_abs": float(np.max(absolute)),
        "mse": mse,
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": float(np.sum(squared)),
    }


def pooled(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot pool an empty metric list")
    count = sum(int(row["count"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    mse = sse / count
    return {
        "case_count": len(rows),
        "count": count,
        "maximum_abs": max(float(row["maximum_abs"]) for row in rows),
        "mse": mse,
        "p95_abs_mean": float(np.mean([float(row["p95_abs"]) for row in rows])),
        "p99_abs_mean": float(np.mean([float(row["p99_abs"]) for row in rows])),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-30))),
        "sse": sse,
    }


def _region(shape: tuple[int, int], size: int, index: int = 0) -> tuple[int, int, int, int]:
    height, width = shape
    if size > min(height, width):
        raise ValueError("crop exceeds source")
    # Three deterministic offsets prevent all validation crops from sampling
    # only the exact image center while retaining six-pixel phase alignment.
    fractions = ((0.50, 0.50), (0.36, 0.58), (0.64, 0.42))
    fx, fy = fractions[index % len(fractions)]
    x0 = int(round((width - size) * fx)) // 6 * 6
    y0 = int(round((height - size) * fy)) // 6 * 6
    x0 = min(max(x0, 0), width - size)
    y0 = min(max(y0, 0), height - size)
    return x0, y0, x0 + size, y0 + size


def _case(
    row: dict[str, object], bank, runner: Path, crop_size: int, index: int,
) -> dict[str, object]:
    full = load_rgb(row)
    region = _region(full.shape[1:], crop_size, index)
    x0, y0, x1, y1 = region
    truth = np.ascontiguousarray(full[:, y0:y1, x0:x1])
    gmax = predict_region(bank, full, region)
    with tempfile.TemporaryDirectory(prefix="xtrans-pnp-baseline-") as temporary:
        baselines, timings = _run_baselines(
            runner, truth, x0 % 6, y0 % 6, Path(temporary)
        )
    operator = xtrans_operator(crop_size, crop_size, x0 % 6, y0 % 6)
    measured = operator.forward(truth)
    return {
        "id": str(row["filename"]),
        "region": list(region),
        "truth": truth,
        "operator": operator,
        "measured": measured,
        "initializers": {
            "zero-samples": operator.adjoint(measured),
            "gmax": gmax,
            "markesteijn": baselines["markesteijn"],
            "mlri": baselines["corrected-final"],
        },
        "baseline_seconds": timings,
    }


def _metric_by_case(cases: list[dict[str, object]], outputs: list[np.ndarray]) -> dict[str, object]:
    rows = []
    by_source = {}
    for case, output in zip(cases, outputs):
        row = metric(output, case["truth"])
        rows.append(row)
        by_source[str(case["id"])] = row
    return {"by_source": by_source, "pooled": pooled(rows)}


def _baseline_table(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        name: _metric_by_case(cases, [case["initializers"][name] for case in cases])
        for name in ("zero-samples", "gmax", "markesteijn", "mlri")
    }


def _denoiser_feasibility(cases: list[dict[str, object]]) -> tuple[dict[str, object], float]:
    rows = {}
    for sigma in SIGMAS:
        clean = []
        gmax = []
        mlri = []
        for case in cases:
            denoiser = BM3DDenoiser("rgb")
            truth = case["truth"]
            operator = case["operator"]
            measured = case["measured"]
            clean.append(denoiser(truth, sigma))
            gmax.append(operator.project(denoiser(case["initializers"]["gmax"], sigma), measured))
            mlri.append(operator.project(denoiser(case["initializers"]["mlri"], sigma), measured))
        rows[str(sigma)] = {
            "clean_bias": _metric_by_case(cases, clean),
            "gmax_post": _metric_by_case(cases, gmax),
            "mlri_post": _metric_by_case(cases, mlri),
        }
        print(
            f"sigma={sigma:g} clean={rows[str(sigma)]['clean_bias']['pooled']['psnr_db']:.3f} "
            f"gmax-post={rows[str(sigma)]['gmax_post']['pooled']['psnr_db']:.3f}",
            flush=True,
        )
    selected = max(
        SIGMAS,
        key=lambda value: float(rows[str(value)]["gmax_post"]["pooled"]["psnr_db"]),
    )
    return rows, selected


def _short_sweep(cases: list[dict[str, object]], sigma: float) -> dict[str, object]:
    settings = {}
    for rho in RHOS:
        snapshots = {iteration: [] for iteration in (1, 2, 3, 5)}
        records = []
        denoiser_seconds = 0.0
        for case in cases:
            denoiser = BM3DDenoiser("rgb")
            result = pnp_admm(
                case["operator"], case["measured"], case["initializers"]["gmax"],
                denoiser, sigma, rho, 5, exact_projection=False,
                capture_iterations=(1, 2, 3, 5),
            )
            denoiser_seconds += denoiser.seconds
            records.append([record.__dict__ for record in result.records])
            for iteration in snapshots:
                snapshots[iteration].append(result.snapshots[iteration])
        key = f"soft-rho-{rho:g}"
        settings[key] = {
            "denoiser_seconds": denoiser_seconds,
            "iterations": {
                str(iteration): _metric_by_case(cases, outputs)
                for iteration, outputs in snapshots.items()
            },
            "records": records,
            "rho": rho,
        }
        print(
            f"{key} iter5={settings[key]['iterations']['5']['pooled']['psnr_db']:.3f}",
            flush=True,
        )
    # With hard projection on both x and z, the observed components and dual
    # remain exactly consistent and rho cancels.  Run it once rather than five
    # redundant times, and record that mathematical independence explicitly.
    snapshots = {iteration: [] for iteration in (1, 2, 3, 5)}
    records = []
    denoiser_seconds = 0.0
    for case in cases:
        denoiser = BM3DDenoiser("rgb")
        result = pnp_admm(
            case["operator"], case["measured"], case["initializers"]["gmax"],
            denoiser, sigma, 1.0, 5, exact_projection=True,
            capture_iterations=(1, 2, 3, 5),
        )
        denoiser_seconds += denoiser.seconds
        records.append([record.__dict__ for record in result.records])
        for iteration in snapshots:
            snapshots[iteration].append(result.snapshots[iteration])
    settings["exact"] = {
        "denoiser_seconds": denoiser_seconds,
        "iterations": {
            str(iteration): _metric_by_case(cases, outputs)
            for iteration, outputs in snapshots.items()
        },
        "records": records,
        "rho": "independent after exact x/z projection",
    }
    return settings


def _long_trajectories(
    cases: list[dict[str, object]], sigma: float, short: dict[str, object]
) -> dict[str, object]:
    best_soft = max(
        (key for key in short if key.startswith("soft-")),
        key=lambda key: max(
            float(short[key]["iterations"][str(i)]["pooled"]["psnr_db"])
            for i in (1, 2, 3, 5)
        ),
    )
    rho = float(short[best_soft]["rho"])
    result_rows = {}
    for mode, exact, selected_rho in ((best_soft, False, rho), ("exact", True, 1.0)):
        snapshots = {iteration: [] for iteration in ITERATIONS}
        records = []
        total_seconds = 0.0
        for case in cases:
            denoiser = BM3DDenoiser("rgb")
            result = pnp_admm(
                case["operator"], case["measured"], case["initializers"]["gmax"],
                denoiser, sigma, selected_rho, 20, exact_projection=exact,
                capture_iterations=ITERATIONS,
            )
            total_seconds += denoiser.seconds
            records.append([record.__dict__ for record in result.records])
            for iteration in ITERATIONS:
                snapshots[iteration].append(result.snapshots[iteration])
        result_rows[mode] = {
            "denoiser_seconds": total_seconds,
            "iterations": {
                str(iteration): _metric_by_case(cases, outputs)
                for iteration, outputs in snapshots.items()
            },
            "records": records,
            "rho": selected_rho if not exact else "independent",
        }
    return result_rows


def _select(long_rows: dict[str, object]) -> dict[str, object]:
    candidates = []
    for mode, row in long_rows.items():
        for iteration in ITERATIONS:
            candidates.append((
                float(row["iterations"][str(iteration)]["pooled"]["mse"]),
                mode,
                iteration,
            ))
    _, mode, iteration = min(candidates)
    row = long_rows[mode]
    return {
        "exact_projection": mode == "exact",
        "iteration": iteration,
        "mode": mode,
        "rho": row["rho"],
        "validation": row["iterations"][str(iteration)],
    }


def _initialization_controls(
    cases: list[dict[str, object]], sigma: float, soft_rho: float,
) -> dict[str, object]:
    """Measure whether inverse coupling depends on the complete initializer."""

    rows = {}
    for initializer in ("mlri", "markesteijn", "zero-samples"):
        modes = (("exact", True, 1.0),)
        limit = 5
        captures = (1, 2, 3, 5)
        if initializer == "mlri":
            modes = (("exact", True, 1.0), (f"soft-rho-{soft_rho:g}", False, soft_rho))
            limit = 20
            captures = ITERATIONS
        for mode, exact, rho in modes:
            snapshots = {iteration: [] for iteration in captures}
            records = []
            seconds = 0.0
            for case in cases:
                denoiser = BM3DDenoiser("rgb")
                result = pnp_admm(
                    case["operator"], case["measured"],
                    case["initializers"][initializer], denoiser, sigma, rho, limit,
                    exact_projection=exact, capture_iterations=captures,
                )
                seconds += denoiser.seconds
                records.append([record.__dict__ for record in result.records])
                for iteration in captures:
                    snapshots[iteration].append(result.snapshots[iteration])
            rows[f"{initializer}-{mode}"] = {
                "bm3d_seconds": seconds,
                "initializer": initializer,
                "exact_projection": exact,
                "rho": rho if not exact else "independent",
                "iterations": {
                    str(iteration): _metric_by_case(cases, outputs)
                    for iteration, outputs in snapshots.items()
                },
                "records": records,
            }
    return rows


def _select_with_initializers(
    gmax_long: dict[str, object], controls: dict[str, object]
) -> dict[str, object]:
    candidates = []
    for mode, row in gmax_long.items():
        for iteration in ITERATIONS:
            candidates.append((
                float(row["iterations"][str(iteration)]["pooled"]["mse"]),
                "gmax", mode, iteration, row,
            ))
    for key, row in controls.items():
        if row["initializer"] != "mlri":
            continue
        for iteration, value in row["iterations"].items():
            candidates.append((
                float(value["pooled"]["mse"]), "mlri", key,
                int(iteration), row,
            ))
    _, initializer, mode, iteration, row = min(candidates)
    return {
        "exact_projection": bool(row.get("exact_projection", mode == "exact")),
        "initializer": initializer,
        "iteration": iteration,
        "mode": mode,
        "rho": row["rho"],
        "validation": row["iterations"][str(iteration)],
    }


def _selected_outputs(
    cases: list[dict[str, object]], sigma: float, selected: dict[str, object],
    initializer: str = "gmax", color_mode: str = "rgb",
) -> tuple[list[np.ndarray], float, list[list[dict[str, object]]]]:
    outputs = []
    seconds = 0.0
    all_records = []
    for case in cases:
        denoiser = BM3DDenoiser(color_mode)
        result = pnp_admm(
            case["operator"], case["measured"], case["initializers"][initializer],
            denoiser, sigma,
            1.0 if selected["exact_projection"] else float(selected["rho"]),
            int(selected["iteration"]),
            exact_projection=bool(selected["exact_projection"]),
        )
        outputs.append(result.image)
        seconds += denoiser.seconds
        all_records.append([record.__dict__ for record in result.records])
    return outputs, seconds, all_records


def _population(
    rows: list[dict[str, object]], bank, runner: Path, crop_size: int,
    sigma: float, selected: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cases = [_case(row, bank, runner, crop_size, index) for index, row in enumerate(rows)]
    baselines = _baseline_table(cases)
    initializer = str(selected["initializer"])
    outputs, seconds, records = _selected_outputs(
        cases, sigma, selected, initializer=initializer
    )
    post = []
    for case in cases:
        denoiser = BM3DDenoiser("rgb")
        post.append(case["operator"].project(
            denoiser(case["initializers"][initializer], sigma), case["measured"]
        ))
    return ({
        "baselines": baselines,
        "bm3d_post_selected_initializer": _metric_by_case(cases, post),
        "pnp_selected_initializer": _metric_by_case(cases, outputs),
        "pnp_records": records,
        "pnp_bm3d_seconds": seconds,
        "regions": {str(case["id"]): case["region"] for case in cases},
    }, cases)


def run(
    output: Path, bsds_root: Path, starfield: Path, runner: Path,
    crop_size: int, validation_count: int, test_count: int,
) -> dict[str, object]:
    started = time.monotonic()
    splits = discover_expanded(bsds_root)
    external = external_chromatic_sources()
    globals_by_size, _, _ = survey_and_train_globals(splits["train"])
    bank = globals_by_size[200]
    validation_cases = [
        _case(row, bank, runner, crop_size, index)
        for index, row in enumerate(splits["val"][:validation_count])
    ]
    baselines = _baseline_table(validation_cases)
    feasibility, sigma = _denoiser_feasibility(validation_cases)
    short = _short_sweep(validation_cases, sigma)
    long_rows = _long_trajectories(validation_cases, sigma, short)
    best_soft_key = max(
        (key for key in short if key.startswith("soft-")),
        key=lambda key: max(
            float(short[key]["iterations"][str(i)]["pooled"]["psnr_db"])
            for i in (1, 2, 3, 5)
        ),
    )
    initialization_controls = _initialization_controls(
        validation_cases, sigma, float(short[best_soft_key]["rho"])
    )
    selected = _select_with_initializers(long_rows, initialization_controls)
    validation_pnp = selected["validation"]["pooled"]
    initializer = str(selected["initializer"])
    validation_initializer = baselines[initializer]["pooled"]
    post_key = f"{initializer}_post"
    best_post = feasibility[str(sigma)][post_key]["pooled"]
    gain_over_initializer = (
        float(validation_pnp["psnr_db"]) - float(validation_initializer["psnr_db"])
    )
    gain_over_post = float(validation_pnp["psnr_db"]) - float(best_post["psnr_db"])

    # Explicit color-space and solver controls at the validation-selected
    # parameter point.  These do not participate in selection.
    luma_outputs, luma_seconds, _ = _selected_outputs(
        validation_cases, sigma, selected, initializer=initializer,
        color_mode="linear-luma"
    )
    pgm_outputs = []
    pgm_seconds = 0.0
    for case in validation_cases:
        denoiser = BM3DDenoiser("rgb")
        result = pnp_pgm(
            case["operator"], case["measured"], case["initializers"][initializer],
            denoiser, sigma, 1.0, int(selected["iteration"]),
            exact_projection=bool(selected["exact_projection"]),
        )
        pgm_outputs.append(result.image)
        pgm_seconds += denoiser.seconds

    gate_a = gain_over_initializer > 0.0
    gate_b = gain_over_post >= MINIMUM_ITERATIVE_GAIN_DB
    result: dict[str, object] = {
        "bayer_reference": {
            "artifact": "/tmp/xtrans-pnp-bayer-reference.json (external, not trusted at runtime)",
            "result": "equations/interface control passed: 35.987 dB initializer to 37.129 dB at iteration 10",
        },
        "bm3d": authenticate_bm3d(),
        "corpus": lmmse_manifest(splits, external),
        "format": "rawtherapee-xtrans-pnp-bm3d-experiment-v1",
        "formulation": {
            "hard_projection": "x=P_CFA(z-u), z=P_CFA(D_sigma(x+u)), u=u+x-z",
            "soft_update": "unmeasured x=z-u; measured x=(y+rho*(z-u))/(1+rho)",
            "interpretation": "PnP fixed-point/consensus iteration; no explicit minimized energy claimed",
        },
        "gates": {
            "A_prior_useful": gate_a,
            "B_iterative_value": gate_b,
            "B_minimum_gain_db": MINIMUM_ITERATIVE_GAIN_DB,
            "gain_over_selected_initializer_db": gain_over_initializer,
            "gain_over_one_shot_db": gain_over_post,
        },
        "parameter_sweep": {
            "sigma": list(SIGMAS),
            "rho": list(RHOS),
            "iterations": list(ITERATIONS),
            "strategy": "sigma selected by projected one-shot GMAX; rho then swept conditionally; full trajectories for best soft rho and exact projection",
        },
        "selected": {**selected, "sigma": sigma},
        "validation": {
            "baselines": baselines,
            "denoiser_feasibility": feasibility,
            "long_trajectories": long_rows,
            "initialization_controls": initialization_controls,
            "short_sweep": short,
            "controls": {
                "linear_luma": {
                    "metric": _metric_by_case(validation_cases, luma_outputs),
                    "bm3d_seconds": luma_seconds,
                },
                "pgm": {
                    "metric": _metric_by_case(validation_cases, pgm_outputs),
                    "bm3d_seconds": pgm_seconds,
                },
            },
            "regions": {str(case["id"]): case["region"] for case in validation_cases},
        },
    }

    if not gate_a:
        result["decision"] = "NO-GO — denoiser prior"
        result["stopped_after"] = "Gate A"
    elif not gate_b:
        result["decision"] = "NO-GO — inverse coupling"
        result["stopped_after"] = "Gate B"
    else:
        population, _ = _population(
            splits["test"][:test_count], bank, runner, crop_size, sigma, selected
        )
        result["test"] = population
        strongest_name = max(
            ("gmax", "markesteijn", "mlri"),
            key=lambda name: float(population["baselines"][name]["pooled"]["psnr_db"]),
        )
        strongest = population["baselines"][strongest_name]["pooled"]
        pnp = population["pnp_selected_initializer"]["pooled"]
        population_gain = float(pnp["psnr_db"]) - float(strongest["psnr_db"])
        tail_reduction = 1.0 - float(pnp["p99_abs_mean"]) / float(strongest["p99_abs_mean"])
        gate_c = population_gain >= MINIMUM_POPULATION_GAIN_DB or tail_reduction >= 0.05
        result["gates"].update({
            "C_population": gate_c,
            "C_strongest_baseline": strongest_name,
            "C_gain_db": population_gain,
            "C_p99_reduction_fraction": tail_reduction,
        })
        if not gate_c:
            result["decision"] = "NO-GO — quality"
            result["stopped_after"] = "Gate C"
        else:
            # Full external/star/synthetic safety evaluation intentionally
            # follows only after the population gate, per the experiment plan.
            result["decision"] = "PARTIAL — population gate passed; safety evaluation required"
            result["stopped_after"] = "Gate C passed"
            result["deferred_safety_inputs"] = {
                "external_chromatic": [str(row["id"]) for row in external],
                "starfield": str(starfield.name),
                "reason": "implemented gate ordering prevents tuning or spending on safety after an earlier failure",
            }
    result["elapsed_seconds"] = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_bytes(canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    parser.add_argument(
        "--runner", type=Path,
        default=Path("build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests"),
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=6)
    parser.add_argument("--test-count", type=int, default=20)
    arguments = parser.parse_args()
    result = run(
        arguments.output, arguments.bsds_root, arguments.starfield,
        arguments.runner, arguments.crop_size, arguments.validation_count,
        arguments.test_count,
    )
    print(json.dumps({
        "decision": result["decision"],
        "elapsed_seconds": result["elapsed_seconds"],
        "gates": result["gates"],
        "selected": result["selected"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

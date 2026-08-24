#!/usr/bin/env python3
"""Freeze and evaluate the validation-selected Student-t GMR reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.xtrans_alias.analysis import canonical_json_bytes
from tools.xtrans_gmm.dataset import (
    TEST_GRID_SIDE,
    corpus_manifest,
    established_samples,
    evaluation_samples,
    external_samples,
)
from tools.xtrans_gmr.experiment import (
    _metric,
    _synthetic_samples,
    _truth,
    load_model,
)
from tools.xtrans_gmr.model import model_summary, prepare_gmr_cache
from tools.xtrans_tgmr.experiment import (
    FROZEN_TAU,
    FROZEN_TEMPERATURE,
    _synthetic_scenes,
)
from tools.xtrans_tgmr.model import conditional_student_t_predict

from .evaluation import (
    pooled_metric,
    quality_retention,
    summarize_batch,
    summarize_shortlist,
)
from .model import (
    coarse_component_order,
    shortlist_student_t_predict,
    truncate_exact_posterior,
)


NU = 3.0


def _phase_range(summary: dict[str, object]) -> float:
    values = [
        float(row["candidate"]["psnr_db"])
        for row in summary["rows"].values()
    ]
    return max(values) - min(values)


def _dense_batch(model, samples):
    return conditional_student_t_predict(
        prepare_gmr_cache(model, FROZEN_TAU),
        samples,
        degrees_of_freedom=NU,
        temperature=FROZEN_TEMPERATURE,
    )


def _shortlist_batch(model, samples, support: int, retained: int):
    cache = prepare_gmr_cache(model, FROZEN_TAU)
    full = conditional_student_t_predict(
        cache,
        samples,
        degrees_of_freedom=NU,
        temperature=FROZEN_TEMPERATURE,
    )
    order = coarse_component_order(
        cache, samples, degrees_of_freedom=NU, support=support
    )
    return shortlist_student_t_predict(
        cache,
        full,
        samples,
        degrees_of_freedom=NU,
        support=support,
        retained=retained,
        component_order=order,
    ), full


def _baselines(parent: dict[str, object], full_gmm: dict[str, object]):
    external_gmax = pooled_metric(full_gmm["external_reference"], "gmax")
    return {
        "test": full_gmm["sampled_baselines"]["pooled"]["gmax"],
        "external": external_gmax,
        "stars": {
            name: row["methods"]["gmax"]
            for name, row in full_gmm["established_reference"].items()
        },
        "synthetic": {
            row["scene"]: row["gmax"] for row in parent["synthetic"]
        },
    }


def _references(parent: dict[str, object]):
    return {
        "test": parent["bsds_test"]["trained_t"]["methods"]["mmse"],
        "external": parent["external"]["trained_t"]["methods"]["mmse"],
        "stars": {
            name: row["trained_t"]["methods"]["mmse"]
            for name, row in parent["stars"].items()
        },
        "synthetic": {
            row["scene"]: row["trained_t"] for row in parent["synthetic"]
        },
    }


def _compact_candidate(
    summaries: dict[str, dict[str, object]],
    baselines: dict[str, object],
    references: dict[str, object],
    full_gmm: dict[str, object],
) -> dict[str, object]:
    test = summaries["bsds-test"]
    external = summaries["external"]
    external_deltas = {
        name: float(row["candidate"]["psnr_db"])
        - float(full_gmm["external_reference"][name]["methods"]["gmax"]["psnr_db"])
        for name, row in external["rows"].items()
    }
    star_rows = {}
    for name in ("hubble", "hubble-bright", "nasa-hydra-starfield", "nasa-hydra-starfield-bright"):
        metric = summaries[name]["metric"]
        star_rows[name] = {
            "metric": metric,
            "delta_vs_gmax_db": (
                float(metric["psnr_db"])
                - float(baselines["stars"][name]["psnr_db"])
            ),
            "quality_retention": quality_retention(
                float(baselines["stars"][name]["mse"]),
                float(references["stars"][name]["mse"]),
                float(metric["mse"]),
            ),
        }
    synthetic_rows = {}
    for name in references["synthetic"]:
        summary = summaries[f"synthetic:{name}"]
        metric = summary["metric"]
        synthetic_rows[name] = {
            "metric": metric,
            "delta_vs_gmax_db": (
                float(metric["psnr_db"])
                - float(baselines["synthetic"][name]["psnr_db"])
            ),
            "delta_vs_tgmr64_db": (
                float(metric["psnr_db"])
                - float(references["synthetic"][name]["psnr_db"])
            ),
            "phase_range_db": _phase_range(summary),
        }
    native_exact = all(bool(summary["native_center_exact"]) for summary in summaries.values())
    result = {
        "test": {
            "metric": test["metric"],
            "quality_retention": quality_retention(
                float(baselines["test"]["mse"]),
                float(references["test"]["mse"]),
                float(test["metric"]["mse"]),
            ),
            "delta_vs_gmax_db": (
                float(test["metric"]["psnr_db"])
                - float(baselines["test"]["psnr_db"])
            ),
            "delta_vs_tgmr64_db": (
                float(test["metric"]["psnr_db"])
                - float(references["test"]["psnr_db"])
            ),
            "posterior": test["posterior"],
        },
        "external": {
            "metric": external["metric"],
            "quality_retention": quality_retention(
                float(baselines["external"]["mse"]),
                float(references["external"]["mse"]),
                float(external["metric"]["mse"]),
            ),
            "deltas_vs_gmax_db": external_deltas,
            "worst_delta_vs_gmax_db": min(external_deltas.values()),
        },
        "stars": star_rows,
        "synthetic": synthetic_rows,
        "native_center_exact": native_exact,
    }
    if "shortlist" in test:
        result["test"]["shortlist"] = test["shortlist"]
    return result


def _all_summaries(model, groups, shortlist=None):
    rows = {}
    for name, samples in groups.items():
        if shortlist is None:
            rows[name] = summarize_batch(samples, _dense_batch(model, samples))
        else:
            support, retained = shortlist
            batch, _ = _shortlist_batch(model, samples, support, retained)
            rows[name] = summarize_shortlist(samples, batch)
        print(f"candidate group={name}", flush=True)
    return rows


def _posterior_sparsity(full_model, groups):
    result = {}
    for name, samples in groups.items():
        full = _dense_batch(full_model, samples)
        truth = _truth(samples)
        rows = []
        for retained in (1, 2, 4, 8, 16, 32):
            truncated = truncate_exact_posterior(full, samples, 7, retained)
            rows.append({
                "retained": retained,
                "metric": _metric(truncated.mmse_rgb - truth),
                "posterior_mass_mean": float(np.mean(truncated.retained_exact_mass)),
                "posterior_mass_p05": float(np.quantile(
                    truncated.retained_exact_mass, 0.05
                )),
                "posterior_mass_minimum": float(np.min(
                    truncated.retained_exact_mass
                )),
            })
        result[name] = rows
    return result


def run(
    output: Path,
    screen_path: Path,
    models_directory: Path,
    full_model_path: Path,
    bsds_root: Path,
    starfield: Path,
    parent_path: Path,
    full_gmm_path: Path,
) -> dict[str, object]:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    full_gmm = json.loads(full_gmm_path.read_text(encoding="utf-8"))
    _, splits = corpus_manifest(bsds_root)
    groups = {
        "bsds-test": evaluation_samples(
            splits["test"], 7, "bsds-test", TEST_GRID_SIDE
        ),
        "external": external_samples(7),
    }
    groups.update(established_samples(starfield, 7))
    groups.update({
        f"synthetic:{name}": _synthetic_samples(scene)
        for name, scene in _synthetic_scenes().items()
    })
    baselines = _baselines(parent, full_gmm)
    references = _references(parent)
    component_paths = {
        8: models_directory / "k8-nu3-i30.npz",
        16: models_directory / "k16-nu3-i30.npz",
        32: models_directory / "k32-nu3-i30.npz",
        64: full_model_path,
    }
    component_reports = {}
    for count, path in component_paths.items():
        model = load_model(path)
        component_reports[str(count)] = {
            "model": model_summary(model),
            "evaluation": _compact_candidate(
                _all_summaries(model, groups), baselines, references, full_gmm
            ),
        }

    full_model = load_model(full_model_path)
    posterior = _posterior_sparsity(full_model, groups)

    selection = screen["validation_selection"]
    if selection["factor"] is not None:
        raise RuntimeError(
            "post-hoc factor validation passed; true MtFA must be evaluated before final freeze"
        )
    selected_combined = selection.get("combined")
    if selected_combined is not None:
        count = int(selected_combined["component_count"])
        support = int(selected_combined["support"])
        retained = int(selected_combined["retained"])
        selected_model = load_model(component_paths[count])
        selected_kind = "component-plus-shortlist"
        selected_summaries = _all_summaries(
            selected_model, groups, shortlist=(support, retained)
        )
        selected_contract = {
            "component_count": count,
            "coarse_support": support,
            "shortlist": retained,
        }
    else:
        selected = selection["component"]
        if selected is None:
            raise RuntimeError("no validation-selected reduced candidate")
        count = int(selected["component_count"])
        selected_model = load_model(component_paths[count])
        selected_kind = "component-only"
        selected_summaries = _all_summaries(selected_model, groups)
        selected_contract = {"component_count": count}
    selected_report = _compact_candidate(
        selected_summaries, baselines, references, full_gmm
    )
    bright = (
        "hubble-bright", "nasa-hydra-starfield-bright"
    )
    safety = {
        "quality_retention_at_least_0_9": (
            float(selected_report["test"]["quality_retention"]) >= 0.9
        ),
        "test_at_least_3db_over_gmax": (
            float(selected_report["test"]["delta_vs_gmax_db"]) >= 3.0
        ),
        "external": float(
            selected_report["external"]["worst_delta_vs_gmax_db"]
        ) >= -0.5,
        "bright_stars": all(
            float(selected_report["stars"][name]["delta_vs_gmax_db"]) >= 0.0
            for name in bright
        ),
        "analytical": min(
            float(row["delta_vs_gmax_db"])
            for row in selected_report["synthetic"].values()
        ) >= -2.0,
        "native_center_exact": bool(selected_report["native_center_exact"]),
    }
    safety["all"] = all(safety.values())

    result = {
        "format": "rawtherapee-xtrans-tgmr-reduction-final-v1",
        "parents": {
            "screen_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
            "tgmr_results_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "full_gmm_results_sha256": hashlib.sha256(full_gmm_path.read_bytes()).hexdigest(),
        },
        "component_count_safety": component_reports,
        "posterior_sparsity": posterior,
        "selected": {
            "kind": selected_kind,
            "contract": selected_contract,
            "evaluation": selected_report,
            "safety": safety,
        },
        "deferred": {
            "true_factor_training_not_triggered": selection["factor"] is None,
            "native_cpp_benchmark_triggered": bool(safety["all"]),
            "epll": True,
            "production_method": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--bsds-root", type=Path, default=Path("/tmp/BSDS500"))
    parser.add_argument(
        "--starfield", type=Path,
        default=Path("/tmp/xtrans-danger-sources/grail_free_air_stars1.tif"),
    )
    parser.add_argument(
        "--parent", type=Path,
        default=Path("devnotes/images/xtrans-tgmr/results.json"),
    )
    parser.add_argument(
        "--full-gmm", type=Path,
        default=Path("devnotes/images/xtrans-gmm/results.json"),
    )
    arguments = parser.parse_args()
    result = run(
        arguments.output,
        arguments.screen,
        arguments.models,
        arguments.full_model,
        arguments.bsds_root,
        arguments.starfield,
        arguments.parent,
        arguments.full_gmm,
    )
    print(json.dumps(result["selected"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

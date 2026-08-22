"""Observable danger features, asymmetric models, and safety metrics.

Ground truth is used only to construct labels and evaluate reconstructions.
Every detector feature is derived from the mosaic, CFA, Markesteijn output, or
MLRI output and is therefore available at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from tools.xtrans_hybrid.analysis import (
    FEATURE_DECIMALS,
    MARGIN,
    LinearModel,
    Normalizer,
    TreeModel,
    feature_maps,
    fit_logistic,
    fit_tree,
)
from tools.xtrans_oracle.analysis import check_rgb, opponent


EPSILON = 1e-12
BLOCK_SIZE = 7
FROZEN_ALPHA = 0.851
FROZEN_RESULTS_SHA256 = (
    "a1e18f4936518ae164cdb867e986518afb6212163ec73c36695b508d83bf4266"
)


def _kurtosis(values: np.ndarray) -> float:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    centered = flattened - float(np.mean(flattened))
    variance = float(np.mean(centered * centered))
    if variance <= EPSILON:
        return 0.0
    return float(np.mean(centered ** 4) / (variance * variance) - 3.0)


def _top_energy_fraction(values: np.ndarray, count: int) -> float:
    energy = np.asarray(values, dtype=np.float64).ravel() ** 2
    total = float(np.sum(energy))
    if total <= EPSILON:
        return 0.0
    count = min(count, energy.size)
    return float(np.sum(np.partition(energy, -count)[-count:]) / total)


def _coherence(values: np.ndarray) -> tuple[float, float, float]:
    checked = np.asarray(values, dtype=np.float64)
    gy, gx = np.gradient(checked)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    discriminant = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4 * jxy * jxy))
    coherence = discriminant / (trace + EPSILON)
    laplacian = ndimage.laplace(checked, mode="reflect")
    directional_to_point = trace / (float(np.mean(laplacian * laplacian)) + EPSILON)
    return coherence, trace, directional_to_point


def _component_features(chroma: np.ndarray) -> dict[str, float]:
    values = np.asarray(chroma, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    maximum = float(np.max(values))
    threshold = max(median + 3.0 * 1.4826 * mad, median + 0.25 * (maximum - median))
    high = values > threshold
    count = int(np.sum(high))
    if not count:
        return {
            "high_fraction": 0.0,
            "component_count": 0.0,
            "largest_component_fraction": 0.0,
            "isolated_fraction": 0.0,
            "mean_neighbor_fraction": 0.0,
        }
    labels, components = ndimage.label(high, structure=np.ones((3, 3), dtype=np.int8))
    sizes = np.bincount(labels.ravel())[1:]
    neighbor_count = ndimage.convolve(
        high.astype(np.float64), np.ones((3, 3), dtype=np.float64), mode="constant"
    ) - high
    isolated = int(np.sum(high & (neighbor_count == 0)))
    return {
        "high_fraction": count / high.size,
        "component_count": float(components),
        "largest_component_fraction": float(np.max(sizes) / count),
        "isolated_fraction": isolated / count,
        "mean_neighbor_fraction": float(np.mean(neighbor_count[high]) / 8.0),
    }


def impulse_features(markesteijn: np.ndarray, mlri: np.ndarray) -> dict[str, float]:
    """Return patch features specialized for sparse chromatic disagreement."""

    mark = check_rgb(markesteijn)
    candidate = check_rgb(mlri)
    difference = candidate - mark
    components = opponent(difference)
    rgb_magnitude = np.sqrt(np.mean(difference * difference, axis=0))
    chroma = np.sqrt(components[1] ** 2 + components[2] ** 2)
    median = float(np.median(chroma))
    mad = float(np.median(np.abs(chroma - median)))
    maximum = float(np.max(chroma))
    peak_index = np.unravel_index(int(np.argmax(chroma)), chroma.shape)
    surround = chroma.copy()
    surround[peak_index] = np.nan
    surround_rms = math.sqrt(float(np.nanmean(surround * surround)))
    highpass = np.stack(
        [ndimage.laplace(difference[channel], mode="reflect") for channel in range(3)]
    )
    coherence, directional_energy, directional_to_point = _coherence(chroma)
    component = _component_features(chroma)
    luma_energy = float(np.sum(components[0] ** 2))
    c1_energy = float(np.sum(components[1] ** 2))
    c2_energy = float(np.sum(components[2] ** 2))
    chroma_energy = c1_energy + c2_energy
    values = {
        "impulse_rgb_max": float(np.max(rgb_magnitude)),
        "impulse_rgb_rms": math.sqrt(float(np.mean(rgb_magnitude * rgb_magnitude))),
        "impulse_rgb_kurtosis": _kurtosis(difference),
        "impulse_highpass_kurtosis": _kurtosis(highpass),
        "impulse_chroma_max": maximum,
        "impulse_chroma_median": median,
        "impulse_chroma_mad": mad,
        "impulse_chroma_peak_to_mad": maximum / (1.4826 * mad + 1e-6),
        "impulse_chroma_peak_to_surround_rms": maximum / (surround_rms + 1e-6),
        "impulse_chroma_top1_energy_fraction": _top_energy_fraction(chroma, 1),
        "impulse_chroma_top2_energy_fraction": _top_energy_fraction(chroma, 2),
        "impulse_chroma_top4_energy_fraction": _top_energy_fraction(chroma, 4),
        "impulse_luma_energy": luma_energy,
        "impulse_chroma_energy": chroma_energy,
        "impulse_c1_energy_fraction": c1_energy / (chroma_energy + EPSILON),
        "impulse_chroma_to_luma_energy": chroma_energy / (luma_energy + 1e-9),
        "impulse_structure_coherence": coherence,
        "impulse_directional_energy": directional_energy,
        "impulse_directional_to_point_energy": directional_to_point,
        **{f"impulse_{name}": value for name, value in component.items()},
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite impulsiveness feature")
    return values


@dataclass(frozen=True)
class FrozenBlender:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray

    def alpha_field(self, case: Mapping[str, object]) -> np.ndarray:
        maps, _ = feature_maps(
            case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"],
            window=BLOCK_SIZE,
        )
        missing = set(self.feature_names) - set(maps)
        if missing:
            raise ValueError(f"frozen blender features are missing: {sorted(missing)}")
        matrix = np.round(
            np.column_stack([maps[name].ravel() for name in self.feature_names]),
            decimals=FEATURE_DECIMALS,
        )
        score = ((matrix - self.mean) / self.scale) @ self.coefficients + self.intercept
        alpha = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
        alpha = np.round(np.clip(alpha, 0, 1), decimals=3).reshape(
            np.asarray(case["mosaic"]).shape
        )
        return ndimage.uniform_filter(alpha, size=7, mode="reflect")


def load_frozen_blender(results_path: Path) -> FrozenBlender:
    import hashlib
    import json

    data = results_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FROZEN_RESULTS_SHA256:
        raise ValueError(f"frozen blender results digest changed: {digest}")
    payload = json.loads(data)
    selected = payload["selected"]
    if selected["block_size"] != 7 or selected["selected_spatial_variant"] != "per_pixel_box_7":
        raise ValueError("frozen blender selection changed")
    model = selected["model"]
    if model["family"] != "logistic-blend" or model["feature_groups"] != list("ABCDE"):
        raise ValueError("frozen blender model changed")
    parameters = model["parameters"]
    names = tuple(parameters["coefficients"])
    return FrozenBlender(
        names,
        np.asarray([parameters["coefficients"][name] for name in names]),
        float(parameters["intercept"]),
        np.asarray([parameters["normalization_mean"][name] for name in names]),
        np.asarray([parameters["normalization_scale"][name] for name in names]),
    )


def attach_frozen_blend(cases: Sequence[dict[str, object]], blender: FrozenBlender) -> None:
    for case in cases:
        alpha = blender.alpha_field(case)
        case["blend_alpha"] = alpha
        case["blend"] = np.asarray(case["markesteijn"]) + alpha[None] * (
            np.asarray(case["mlri"]) - np.asarray(case["markesteijn"])
        )


@dataclass(frozen=True)
class DangerTable:
    features: np.ndarray
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    mark_sse: np.ndarray
    mlri_sse: np.ndarray
    fixed_sse: np.ndarray
    blend_sse: np.ndarray
    pixels: np.ndarray
    source_ids: np.ndarray
    crop_ids: np.ndarray
    case_indices: np.ndarray
    bounds: np.ndarray
    splits: np.ndarray
    kinds: np.ndarray

    @property
    def regression(self) -> np.ndarray:
        return np.round(
            (self.blend_sse - self.mark_sse) / (3.0 * self.pixels),
            decimals=12,
        )

    def subset(self, mask: np.ndarray) -> "DangerTable":
        checked = np.asarray(mask, dtype=bool)
        return DangerTable(
            self.features[checked], self.feature_names, self.feature_groups,
            self.mark_sse[checked], self.mlri_sse[checked], self.fixed_sse[checked],
            self.blend_sse[checked], self.pixels[checked], self.source_ids[checked],
            self.crop_ids[checked], self.case_indices[checked], self.bounds[checked],
            self.splits[checked], self.kinds[checked],
        )


def build_danger_table(cases: Sequence[Mapping[str, object]]) -> DangerTable:
    rows = []
    names: tuple[str, ...] | None = None
    groups: Mapping[str, tuple[str, ...]] | None = None
    for case_index, case in enumerate(cases):
        maps, map_groups = feature_maps(
            case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"],
            window=BLOCK_SIZE,
        )
        core_names = tuple(
            name for group in "ABCDE" for name in map_groups[group]
        )
        truth = check_rgb(case["truth"])
        mark = check_rgb(case["markesteijn"])
        mlri = check_rgb(case["mlri"])
        blend = check_rgb(case["blend"])
        fixed = mark + FROZEN_ALPHA * (mlri - mark)
        for y in range(MARGIN, truth.shape[1] - MARGIN, BLOCK_SIZE):
            y2 = min(truth.shape[1] - MARGIN, y + BLOCK_SIZE)
            for x in range(MARGIN, truth.shape[2] - MARGIN, BLOCK_SIZE):
                x2 = min(truth.shape[2] - MARGIN, x + BLOCK_SIZE)
                center_y = (y + y2 - 1) // 2
                center_x = (x + x2 - 1) // 2
                extra = impulse_features(mark[:, y:y2, x:x2], mlri[:, y:y2, x:x2])
                case_names = core_names + tuple(extra)
                case_groups = {
                    "core": core_names,
                    "impulse": tuple(extra),
                }
                if names is None:
                    names, groups = case_names, case_groups
                elif names != case_names or groups != case_groups:
                    raise RuntimeError("danger feature contract changed between patches")
                feature = [maps[name][center_y, center_x] for name in core_names]
                feature.extend(extra.values())

                def sse(candidate: np.ndarray) -> float:
                    error = candidate[:, y:y2, x:x2] - truth[:, y:y2, x:x2]
                    # Native OpenMP scheduling can change a reconstructed
                    # float in its last bit. Canonicalizing patch SSE far
                    # below the experiment's smallest danger threshold keeps
                    # model artifacts reproducible without quantizing images
                    # or detector observables.
                    return float(np.round(np.sum(error * error), decimals=10))

                rows.append((
                    feature, sse(mark), sse(mlri), sse(fixed), sse(blend),
                    (y2 - y) * (x2 - x), str(case["source_id"]),
                    str(case["crop_id"]), case_index, (y, y2, x, x2),
                    str(case["split"]), str(case.get("kind", "natural")),
                ))
    if not rows or names is None or groups is None:
        raise ValueError("danger table has no patches")
    return DangerTable(
        np.round(np.asarray([row[0] for row in rows]), FEATURE_DECIMALS),
        names, groups,
        np.asarray([row[1] for row in rows]),
        np.asarray([row[2] for row in rows]),
        np.asarray([row[3] for row in rows]),
        np.asarray([row[4] for row in rows]),
        np.asarray([row[5] for row in rows], dtype=np.int32),
        np.asarray([row[6] for row in rows]),
        np.asarray([row[7] for row in rows]),
        np.asarray([row[8] for row in rows], dtype=np.int32),
        np.asarray([row[9] for row in rows], dtype=np.int32),
        np.asarray([row[10] for row in rows]),
        np.asarray([row[11] for row in rows]),
    )


class RiskModel:
    name: str

    def risk(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def serialize(self, names: Sequence[str]) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class RuleRisk(RiskModel):
    name: str
    feature: int
    direction: float
    center: float
    scale: float

    def risk(self, features: np.ndarray) -> np.ndarray:
        score = self.direction * (np.asarray(features)[:, self.feature] - self.center) / self.scale
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))

    def serialize(self, names: Sequence[str]) -> dict[str, object]:
        return {
            "family": "single-feature-rule", "feature": names[self.feature],
            "direction": self.direction, "center": self.center, "scale": self.scale,
        }


@dataclass(frozen=True)
class LogisticRisk(RiskModel):
    name: str
    model: LinearModel

    def risk(self, features: np.ndarray) -> np.ndarray:
        return self.model.alpha(features)

    def serialize(self, names: Sequence[str]) -> dict[str, object]:
        selected = [names[index] for index in self.model.feature_indices]
        return {
            "family": "weighted-logistic", "regularization": self.model.regularization,
            "intercept": self.model.intercept,
            "coefficients": dict(zip(selected, map(float, self.model.coefficients))),
            "normalization_mean": dict(zip(selected, map(float, self.model.normalizer.mean))),
            "normalization_scale": dict(zip(selected, map(float, self.model.normalizer.scale))),
        }


@dataclass(frozen=True)
class TreeRisk(RiskModel):
    name: str
    model: TreeModel

    def risk(self, features: np.ndarray) -> np.ndarray:
        return self.model.alpha(features)

    def serialize(self, names: Sequence[str]) -> dict[str, object]:
        return {
            "family": "weighted-shallow-tree", "depth": self.model.depth,
            "tree": self.model.root.to_dict(self.model.feature_names),
        }


@dataclass(frozen=True)
class BoostedStumpsRisk(RiskModel):
    name: str
    normalizer: Normalizer
    feature_indices: np.ndarray
    intercept: float
    stumps: tuple[tuple[int, float, float, float], ...]
    learning_rate: float

    def risk(self, features: np.ndarray) -> np.ndarray:
        selected = self.normalizer.apply(np.asarray(features)[:, self.feature_indices])
        score = np.full(selected.shape[0], self.intercept)
        for feature, threshold, left, right in self.stumps:
            score += self.learning_rate * np.where(
                selected[:, feature] <= threshold, left, right
            )
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))

    def serialize(self, names: Sequence[str]) -> dict[str, object]:
        selected_names = [names[index] for index in self.feature_indices]
        return {
            "family": "weighted-logistic-boosted-stumps",
            "intercept": self.intercept, "learning_rate": self.learning_rate,
            "iterations": len(self.stumps),
            "stumps": [
                {
                    "feature": selected_names[feature], "threshold": threshold,
                    "left": left, "right": right,
                }
                for feature, threshold, left, right in self.stumps
            ],
        }


def asymmetric_weights(regression: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = np.asarray(labels, dtype=bool)
    severity = np.maximum(np.asarray(regression, dtype=np.float64), 0.0)
    scale = float(np.percentile(severity[positive], 75)) if positive.any() else 1.0
    relative = np.clip(severity / max(scale, EPSILON), 0, 10)
    return 1.0 + positive * (20.0 + 8.0 * relative)


def _auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if weights is None:
        weights = np.ones(scores.size)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if not positive.size or not negative.size:
        return 0.5
    numerator = 0.0
    denominator = 0.0
    for index in positive:
        pair = weights[index] * weights[negative]
        numerator += float(np.sum(pair * (scores[index] > scores[negative])))
        numerator += 0.5 * float(np.sum(pair * (scores[index] == scores[negative])))
        denominator += float(np.sum(pair))
    return numerator / denominator


def fit_rule(
    train: DangerTable, validation: DangerTable, labels: np.ndarray, weights: np.ndarray,
    *, name: str = "rule",
) -> RuleRisk:
    impulse_indices = [
        index for index, name in enumerate(train.feature_names)
        if name.startswith("impulse_")
    ]
    best: tuple[float, RuleRisk] | None = None
    validation_labels = validation.regression > float(np.min(train.regression[labels]))
    for index in impulse_indices:
        values = train.features[:, index]
        direction = 1.0 if np.average(values[labels], weights=weights[labels]) >= np.average(values[~labels], weights=weights[~labels]) else -1.0
        center = float(np.median(values))
        scale = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-6)
        model = RuleRisk(name, index, direction, center, scale)
        score = _auc(validation_labels, model.risk(validation.features))
        if best is None or score > best[0]:
            best = (score, model)
    assert best is not None
    return best[1]


def fit_boosted_stumps(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    iterations: int = 24,
    learning_rate: float = 0.15,
) -> BoostedStumpsRisk:
    selected_raw = np.asarray(features, dtype=np.float64)[:, indices]
    normalizer = Normalizer.fit(selected_raw)
    selected = normalizer.apply(selected_raw)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(np.clip(np.average(target, weights=weights), 1e-5, 1 - 1e-5))
    intercept = math.log(mean / (1 - mean))
    score = np.full(target.size, intercept)
    stumps = []
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
        residual = target - probability
        best = None
        for feature in range(selected.shape[1]):
            values = selected[:, feature]
            for threshold in np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9))):
                left_mask = values <= threshold
                if np.sum(left_mask) < 16 or np.sum(~left_mask) < 16:
                    continue
                left = float(np.average(residual[left_mask], weights=weights[left_mask]))
                right = float(np.average(residual[~left_mask], weights=weights[~left_mask]))
                prediction = np.where(left_mask, left, right)
                loss = float(np.sum(weights * (residual - prediction) ** 2))
                candidate = (loss, feature, float(threshold), left, right, prediction)
                if best is None or candidate[:5] < best[:5]:
                    best = candidate
        if best is None:
            break
        stumps.append(best[1:5])
        score += learning_rate * best[5]
    return BoostedStumpsRisk(
        "boosted-stumps", normalizer, np.asarray(indices, dtype=np.int32),
        intercept, tuple(stumps), learning_rate,
    )


def fit_models(
    train: DangerTable, validation: DangerTable, tau: float,
    groups: Sequence[str] = ("core", "impulse"),
) -> list[RiskModel]:
    labels = train.regression > tau
    if not labels.any() or labels.all():
        raise ValueError("danger threshold does not produce both classes")
    weights = asymmetric_weights(train.regression, labels)
    selected_names = {
        name for group in groups for name in train.feature_groups[group]
    }
    all_indices = np.asarray([
        index for index, name in enumerate(train.feature_names)
        if name in selected_names
    ], dtype=np.int32)
    suffix = "-plus-".join(groups)
    models: list[RiskModel] = []
    if "impulse" in groups:
        models.append(fit_rule(
            train, validation, labels, weights, name=f"rule-{suffix}"
        ))
    for regularization in (1e-4, 1e-2, 1.0):
        models.append(LogisticRisk(
            f"weighted-logistic-{regularization:g}-{suffix}",
            fit_logistic(
                train.features, labels.astype(np.float64), weights, all_indices,
                regularization,
            ),
        ))
    for depth in (2, 3):
        models.append(TreeRisk(
            f"weighted-tree-depth-{depth}-{suffix}",
            fit_tree(
                train.features, labels.astype(np.float64), weights, all_indices,
                train.feature_names, depth, classification=False,
            ),
        ))
    boosted = fit_boosted_stumps(
        train.features, labels.astype(np.float64), weights, all_indices
    )
    models.append(BoostedStumpsRisk(
        f"boosted-stumps-{suffix}", boosted.normalizer,
        boosted.feature_indices, boosted.intercept, boosted.stumps,
        boosted.learning_rate,
    ))
    return models


def method_metrics(table: DangerTable, sse: np.ndarray) -> dict[str, object]:
    values = np.maximum(np.asarray(sse, dtype=np.float64), 0.0)
    patch_rms = np.sqrt(values / (3.0 * table.pixels))
    mse = float(np.sum(values) / np.sum(3 * table.pixels))
    return {
        "mse": mse,
        "psnr_db": math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse),
        "patch_rms": {
            "p95": float(np.percentile(patch_rms, 95)),
            "p99": float(np.percentile(patch_rms, 99)),
            "maximum": float(np.max(patch_rms)),
        },
    }


def safety_metrics(
    table: DangerTable, danger: np.ndarray, severe_tau: float,
) -> dict[str, object]:
    fallback = np.asarray(danger, dtype=bool)
    output_sse = np.where(fallback, table.mark_sse, table.blend_sse)
    severe = table.regression > severe_tau
    actual_regression = (
        np.sqrt(output_sse / (3.0 * table.pixels))
        - np.sqrt(table.mark_sse / (3.0 * table.pixels))
    )
    top_recalls = {}
    order = np.argsort(table.regression)[::-1]
    for percentage in (1, 5, 10):
        count = max(1, math.ceil(table.regression.size * percentage / 100.0))
        worst = order[:count]
        top_recalls[f"top_{percentage}_percent"] = float(np.mean(fallback[worst]))
    mark = method_metrics(table, table.mark_sse)
    blend = method_metrics(table, table.blend_sse)
    output = method_metrics(table, output_sse)
    mark_gain = mark["mse"] - blend["mse"]
    retained = (mark["mse"] - output["mse"]) / mark_gain if mark_gain > EPSILON else 0.0
    return {
        "fallback_fraction": float(np.mean(fallback)),
        "severe_count": int(np.sum(severe)),
        "severe_recall": float(np.mean(fallback[severe])) if severe.any() else 1.0,
        "severe_precision": float(np.sum(fallback & severe) / max(1, np.sum(fallback))),
        "catastrophic_patches_missed": int(np.sum(severe & ~fallback)),
        "top_regression_recall": top_recalls,
        "aggregate_gain_retained_fraction": float(retained),
        "markesteijn": mark,
        "ungated_blender": blend,
        "gated": output,
        "maximum_paired_rms_regression": float(np.max(actual_regression)),
        "p99_paired_rms_regression": float(np.percentile(actual_regression, 99)),
    }


def threshold_curve(
    table: DangerTable, risk: np.ndarray, severe_tau: float,
) -> list[dict[str, object]]:
    scores = np.asarray(risk, dtype=np.float64)
    thresholds = np.unique(np.concatenate((
        np.asarray((
            np.nextafter(float(np.min(scores)), -math.inf),
            np.nextafter(float(np.max(scores)), math.inf),
        )),
        np.quantile(scores, np.linspace(0, 1, 101)),
    )))
    return [
        {"threshold": float(threshold), **safety_metrics(table, scores >= threshold, severe_tau)}
        for threshold in thresholds
    ]


def calibration(table: DangerTable, risk: np.ndarray, severe_tau: float) -> dict[str, object]:
    scores = np.asarray(risk, dtype=np.float64)
    rows = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (scores >= lower) & (scores <= upper if upper == 1 else scores < upper)
        rows.append({
            "range": [float(lower), float(upper)],
            "count": int(np.sum(mask)),
            "predicted_risk": None if not mask.any() else float(np.mean(scores[mask])),
            "danger_frequency": None if not mask.any() else float(np.mean(table.regression[mask] > severe_tau)),
            "mean_regression_mse": None if not mask.any() else float(np.mean(table.regression[mask])),
        })
    return {"bins": rows, "roc_auc": _auc(table.regression > severe_tau, scores)}


def select_operating_point(
    curve: Sequence[Mapping[str, object]],
    *,
    minimum_recall: float = 0.90,
    minimum_gain_retained: float = 0.85,
) -> Mapping[str, object]:
    eligible = [
        row for row in curve
        if float(row["severe_recall"]) >= minimum_recall
        and float(row["aggregate_gain_retained_fraction"]) >= minimum_gain_retained
    ]
    if not eligible:
        eligible = [row for row in curve if float(row["severe_recall"]) >= minimum_recall]
    if not eligible:
        eligible = list(curve)
    return min(
        eligible,
        key=lambda row: (
            float(row["fallback_fraction"]),
            -float(row["aggregate_gain_retained_fraction"]),
            -float(row["severe_recall"]),
        ),
    )

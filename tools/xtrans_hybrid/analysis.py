"""Leakage-safe features and compact deterministic hybrid models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, optimize

from tools.xtrans_oracle.analysis import OPPONENT_BASIS, check_rgb, opponent


EPSILON = 1e-12
FEATURE_DECIMALS = 6
MARGIN = 16


def _mean(values: np.ndarray, window: int) -> np.ndarray:
    return ndimage.uniform_filter(values, size=window, mode="reflect")


def _rms(values: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, _mean(values * values, window)))


def _shift(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    radius = max(abs(dy), abs(dx))
    padded = np.pad(values, radius, mode="reflect")
    y = radius + dy
    x = radius + dx
    return padded[y : y + values.shape[0], x : x + values.shape[1]]


def _gradients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(values)
    return gx, gy


def _structure(values: np.ndarray, window: int) -> dict[str, np.ndarray]:
    gx, gy = _gradients(values)
    jxx = _mean(gx * gx, window)
    jyy = _mean(gy * gy, window)
    jxy = _mean(gx * gy, window)
    trace = jxx + jyy
    discriminant = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4 * jxy * jxy))
    eigen_small = 0.5 * (trace - discriminant)
    magnitude = np.sqrt(gx * gx + gy * gy)
    laplacian = ndimage.laplace(values, mode="reflect")
    return {
        "gradient_rms": np.sqrt(np.maximum(0.0, trace)),
        "coherence": discriminant / (trace + EPSILON),
        "orientation_cos2": (jxx - jyy) / (trace + EPSILON),
        "orientation_sin2": 2 * jxy / (trace + EPSILON),
        "corner_energy": np.maximum(0.0, eigen_small),
        "laplacian_rms": _rms(laplacian, window),
        "gradient_variance": np.maximum(
            0.0, _mean(magnitude * magnitude, window) - _mean(magnitude, window) ** 2
        ),
    }


def _phase_spread(values: np.ndarray, window: int) -> np.ndarray:
    height, width = values.shape
    y, x = np.mgrid[:height, :width]
    means = []
    for phase_y in range(3):
        for phase_x in range(3):
            mask = ((y % 3 == phase_y) & (x % 3 == phase_x)).astype(np.float64)
            numerator = _mean(values * mask, window)
            denominator = _mean(mask, window)
            means.append(numerator / np.maximum(denominator, EPSILON))
    stacked = np.stack(means)
    return np.std(stacked, axis=0)


def feature_maps(
    mosaic: np.ndarray,
    cfa: np.ndarray,
    markesteijn: np.ndarray,
    mlri: np.ndarray,
    *,
    window: int,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Return observable feature maps; ground truth is intentionally absent."""

    markesteijn = check_rgb(markesteijn)
    mlri = check_rgb(mlri)
    mosaic = np.asarray(mosaic, dtype=np.float64)
    cfa = np.asarray(cfa, dtype=np.int8)
    if mosaic.shape != markesteijn.shape[1:] or cfa.shape != mosaic.shape:
        raise ValueError("feature inputs differ in shape")
    if window < 3 or window % 2 == 0:
        raise ValueError("feature window must be odd and at least three")
    if not np.isfinite(mosaic).all():
        raise ValueError("mosaic contains a non-finite value")

    maps: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = {name: [] for name in "ABCDEFG"}

    def add(group: str, name: str, values: np.ndarray) -> None:
        checked = np.asarray(values, dtype=np.float64)
        if checked.shape != mosaic.shape or not np.isfinite(checked).all():
            raise ValueError(f"invalid feature map {name}")
        maps[name] = checked
        groups[group].append(name)

    mark_components = opponent(markesteijn)
    mlri_components = opponent(mlri)
    difference = mlri - markesteijn
    difference_components = opponent(difference)
    pixel_abs = np.mean(np.abs(difference), axis=0)
    pixel_energy = np.mean(difference * difference, axis=0)
    disagreement_rms = _rms(np.sqrt(pixel_energy), window)
    add("A", "disagreement_mean_abs", _mean(pixel_abs, window))
    add("A", "disagreement_rgb_rms", disagreement_rms)
    add("A", "disagreement_max_abs", ndimage.maximum_filter(np.max(np.abs(difference), axis=0), window, mode="reflect"))
    add("A", "disagreement_variance", np.maximum(0.0, _mean(pixel_energy, window) - _mean(np.mean(difference, axis=0), window) ** 2))
    d_structure = _structure(difference_components[0], window)
    for name in ("gradient_rms", "coherence", "orientation_cos2", "orientation_sin2"):
        add("A", f"disagreement_{name}", d_structure[name])
    d_l = _rms(difference_components[0], window)
    d_c1 = _rms(difference_components[1], window)
    d_c2 = _rms(difference_components[2], window)
    add("A", "disagreement_l_rms", d_l)
    add("A", "disagreement_c1_rms", d_c1)
    add("A", "disagreement_c2_rms", d_c2)
    add("A", "disagreement_chroma_luma_ratio", np.sqrt(d_c1 * d_c1 + d_c2 * d_c2) / (d_l + 1e-6))
    add("A", "disagreement_fraction_gt_005", _mean((np.sqrt(pixel_energy) > 0.005).astype(np.float64), window))
    add("A", "disagreement_fraction_gt_020", _mean((np.sqrt(pixel_energy) > 0.020).astype(np.float64), window))

    horizontal = 0.5 * (_shift(mosaic, 0, 6) - _shift(mosaic, 0, -6))
    vertical = 0.5 * (_shift(mosaic, 6, 0) - _shift(mosaic, -6, 0))
    diagonal = 0.5 * (_shift(mosaic, 6, 6) - _shift(mosaic, -6, -6))
    antidiagonal = 0.5 * (_shift(mosaic, 6, -6) - _shift(mosaic, -6, 6))
    h_rms, v_rms = _rms(horizontal, window), _rms(vertical, window)
    d_rms, a_rms = _rms(diagonal, window), _rms(antidiagonal, window)
    add("B", "mosaic_horizontal_rms", h_rms)
    add("B", "mosaic_vertical_rms", v_rms)
    add("B", "mosaic_diagonal_rms", d_rms)
    add("B", "mosaic_antidiagonal_rms", a_rms)
    add("B", "mosaic_axis_anisotropy", np.abs(h_rms - v_rms) / (h_rms + v_rms + EPSILON))
    add("B", "mosaic_diagonal_anisotropy", np.abs(d_rms - a_rms) / (d_rms + a_rms + EPSILON))

    structures = {}
    for prefix, components in (("mark", mark_components), ("mlri", mlri_components)):
        structure = _structure(components[0], window)
        structures[prefix] = structure
        for name in ("gradient_rms", "coherence", "orientation_cos2", "orientation_sin2"):
            add("C", f"{prefix}_{name}", structure[name])
    add("C", "candidate_gradient_rms_difference", structures["mlri"]["gradient_rms"] - structures["mark"]["gradient_rms"])
    mark_gx, mark_gy = _gradients(mark_components[0])
    mlri_gx, mlri_gy = _gradients(mlri_components[0])
    direction = (mark_gx * mlri_gx + mark_gy * mlri_gy) / (
        np.sqrt((mark_gx * mark_gx + mark_gy * mark_gy) * (mlri_gx * mlri_gx + mlri_gy * mlri_gy)) + EPSILON
    )
    add("C", "candidate_gradient_direction_agreement", _mean(direction, window))

    chroma_maps = {}
    for prefix, components in (("mark", mark_components), ("mlri", mlri_components)):
        chroma = np.sqrt(components[1] ** 2 + components[2] ** 2)
        chroma_maps[prefix] = chroma
        chroma_mean = _mean(chroma, window)
        add("D", f"{prefix}_chroma_mean", chroma_mean)
        add("D", f"{prefix}_chroma_luma_ratio", chroma_mean / (_mean(np.abs(components[0]), window) + 1e-6))
        add("D", f"{prefix}_chroma_gradient_rms", _structure(chroma, window)["gradient_rms"])
    add("D", "candidate_chroma_difference", _mean(chroma_maps["mlri"] - chroma_maps["mark"], window))

    for prefix in ("mark", "mlri"):
        add("E", f"{prefix}_laplacian_rms", structures[prefix]["laplacian_rms"])
        add("E", f"{prefix}_gradient_variance", structures[prefix]["gradient_variance"])
        add("E", f"{prefix}_corner_energy", structures[prefix]["corner_energy"])
    add("E", "candidate_laplacian_difference", structures["mlri"]["laplacian_rms"] - structures["mark"]["laplacian_rms"])

    sampled_mark = np.zeros_like(mosaic)
    sampled_mlri = np.zeros_like(mosaic)
    for channel in range(3):
        mask = cfa == channel
        sampled_mark[mask] = markesteijn[channel][mask] - mosaic[mask]
        sampled_mlri[mask] = mlri[channel][mask] - mosaic[mask]
    add("F", "mark_native_sample_rms", _rms(sampled_mark, window))
    add("F", "mlri_native_sample_rms", _rms(sampled_mlri, window))

    disagreement_chroma = np.sqrt(difference_components[1] ** 2 + difference_components[2] ** 2)
    add("G", "phase_disagreement_chroma_spread", _phase_spread(disagreement_chroma, window))
    add("G", "phase_disagreement_luma_spread", _phase_spread(difference_components[0], window))
    add("G", "phase_mosaic_spread", _phase_spread(mosaic, window))

    return maps, {name: tuple(values) for name, values in groups.items()}


@dataclass(frozen=True)
class PatchTable:
    features: np.ndarray
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    mark_sse: np.ndarray
    mlri_sse: np.ndarray
    cross: np.ndarray
    quadratic: np.ndarray
    component_mark_sse: np.ndarray
    component_cross: np.ndarray
    component_quadratic: np.ndarray
    pixels: np.ndarray
    source_ids: np.ndarray
    crop_ids: np.ndarray
    case_indices: np.ndarray
    bounds: np.ndarray
    block_size: int

    @property
    def labels(self) -> np.ndarray:
        return (self.mlri_sse < self.mark_sse).astype(np.int8)

    @property
    def margins(self) -> np.ndarray:
        return self.mark_sse - self.mlri_sse

    @property
    def oracle_alpha(self) -> np.ndarray:
        return np.clip(
            np.divide(-self.cross, self.quadratic, out=np.zeros_like(self.cross), where=self.quadratic > 0),
            0,
            1,
        )

    def subset(self, mask: np.ndarray) -> "PatchTable":
        checked = np.asarray(mask, dtype=bool)
        return PatchTable(
            self.features[checked], self.feature_names, self.feature_groups,
            self.mark_sse[checked], self.mlri_sse[checked], self.cross[checked],
            self.quadratic[checked], self.component_mark_sse[checked],
            self.component_cross[checked], self.component_quadratic[checked],
            self.pixels[checked], self.source_ids[checked], self.crop_ids[checked],
            self.case_indices[checked], self.bounds[checked], self.block_size,
        )


def build_patch_table(cases: Sequence[Mapping[str, object]], block_size: int) -> PatchTable:
    rows = []
    names: tuple[str, ...] | None = None
    groups: Mapping[str, tuple[str, ...]] | None = None
    for case_index, case in enumerate(cases):
        maps, case_groups = feature_maps(
            case["mosaic"], case["cfa"], case["markesteijn"], case["mlri"],
            window=block_size,
        )
        case_names = tuple(maps)
        if names is None:
            names, groups = case_names, case_groups
        elif names != case_names or groups != case_groups:
            raise RuntimeError("feature contract changed between cases")
        truth = check_rgb(case["truth"])
        mark = check_rgb(case["markesteijn"])
        mlri = check_rgb(case["mlri"])
        truth_components = opponent(truth)
        mark_components = opponent(mark)
        mlri_components = opponent(mlri)
        for y in range(MARGIN, truth.shape[1] - MARGIN, block_size):
            y2 = min(truth.shape[1] - MARGIN, y + block_size)
            for x in range(MARGIN, truth.shape[2] - MARGIN, block_size):
                x2 = min(truth.shape[2] - MARGIN, x + block_size)
                center_y = (y + y2 - 1) // 2
                center_x = (x + x2 - 1) // 2
                mark_error = mark[:, y:y2, x:x2] - truth[:, y:y2, x:x2]
                delta = mlri[:, y:y2, x:x2] - mark[:, y:y2, x:x2]
                component_mark_error = mark_components[:, y:y2, x:x2] - truth_components[:, y:y2, x:x2]
                component_delta = mlri_components[:, y:y2, x:x2] - mark_components[:, y:y2, x:x2]
                rows.append((
                    [maps[name][center_y, center_x] for name in case_names],
                    float(np.sum(mark_error * mark_error)),
                    float(np.sum((mark_error + delta) ** 2)),
                    float(np.sum(mark_error * delta)),
                    float(np.sum(delta * delta)),
                    np.sum(component_mark_error * component_mark_error, axis=(1, 2)),
                    np.sum(component_mark_error * component_delta, axis=(1, 2)),
                    np.sum(component_delta * component_delta, axis=(1, 2)),
                    (y2 - y) * (x2 - x),
                    str(case["source_id"]), str(case["crop_id"]), case_index,
                    (y, y2, x, x2),
                ))
    assert names is not None and groups is not None
    return PatchTable(
        # Candidate images originate as float32.  Canonicalizing derived
        # observables far below that precision prevents reduction-order noise
        # in the last float64 bits from changing fitted model artifacts.
        features=np.round(
            np.asarray([row[0] for row in rows], dtype=np.float64),
            decimals=FEATURE_DECIMALS,
        ),
        feature_names=names,
        feature_groups=groups,
        mark_sse=np.asarray([row[1] for row in rows]),
        mlri_sse=np.asarray([row[2] for row in rows]),
        cross=np.asarray([row[3] for row in rows]),
        quadratic=np.asarray([row[4] for row in rows]),
        component_mark_sse=np.asarray([row[5] for row in rows]),
        component_cross=np.asarray([row[6] for row in rows]),
        component_quadratic=np.asarray([row[7] for row in rows]),
        pixels=np.asarray([row[8] for row in rows], dtype=np.int32),
        source_ids=np.asarray([row[9] for row in rows]),
        crop_ids=np.asarray([row[10] for row in rows]),
        case_indices=np.asarray([row[11] for row in rows], dtype=np.int32),
        bounds=np.asarray([row[12] for row in rows], dtype=np.int32),
        block_size=block_size,
    )


def feature_indices(table: PatchTable, group_names: Sequence[str]) -> np.ndarray:
    selected = {name for group in group_names for name in table.feature_groups[group]}
    return np.asarray([index for index, name in enumerate(table.feature_names) if name in selected], dtype=np.int32)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        checked = np.asarray(values, dtype=np.float64)
        mean = np.round(checked.mean(axis=0), decimals=12)
        scale = np.round(checked.std(axis=0), decimals=12)
        scale[scale < 1e-12] = 1.0
        return cls(mean, scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale


@dataclass(frozen=True)
class LinearModel:
    kind: str
    coefficients: np.ndarray
    intercept: float
    normalizer: Normalizer
    feature_indices: np.ndarray
    regularization: float

    def score(self, features: np.ndarray) -> np.ndarray:
        selected = np.asarray(features, dtype=np.float64)[:, self.feature_indices]
        normalized = self.normalizer.apply(selected)
        return normalized @ self.coefficients + self.intercept

    def alpha(self, features: np.ndarray) -> np.ndarray:
        score = self.score(features)
        if self.kind == "logistic":
            score = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
        return np.clip(score, 0, 1)


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    regularization: float,
) -> LinearModel:
    selected = np.asarray(features, dtype=np.float64)[:, indices]
    normalizer = Normalizer.fit(selected)
    normalized = normalizer.apply(selected)
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    sqrt_weights = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), EPSILON))
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = np.asarray(target, dtype=np.float64) * sqrt_weights
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0
    solution = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return LinearModel("linear", solution[1:], float(solution[0]), normalizer, indices, regularization)


def fit_logistic(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    regularization: float,
    *,
    soft_target: bool = False,
) -> LinearModel:
    selected = np.asarray(features, dtype=np.float64)[:, indices]
    normalizer = Normalizer.fit(selected)
    normalized = normalizer.apply(selected)
    target = np.asarray(target, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), EPSILON)
    weights = weights / weights.mean()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        score = normalized @ coefficients + intercept
        probability = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
        if soft_target:
            residual = probability - target
            loss = 0.5 * np.mean(weights * residual * residual)
            derivative = weights * residual * probability * (1 - probability) / target.size
        else:
            loss = np.mean(weights * (np.logaddexp(0, score) - target * score))
            derivative = weights * (probability - target) / target.size
        loss += 0.5 * regularization * float(np.dot(coefficients, coefficients))
        gradient = np.empty_like(parameters)
        gradient[0] = np.sum(derivative)
        gradient[1:] = normalized.T @ derivative + regularization * coefficients
        return float(loss), gradient

    initial = np.zeros(normalized.shape[1] + 1)
    mean_target = float(np.clip(np.average(target, weights=weights), 1e-5, 1 - 1e-5))
    initial[0] = math.log(mean_target / (1 - mean_target))
    fitted = optimize.minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-11, "maxls": 50},
    )
    if not fitted.success:
        raise RuntimeError(f"logistic fit failed: {fitted.message}")
    return LinearModel(
        "logistic", fitted.x[1:], float(fitted.x[0]), normalizer, indices,
        regularization,
    )


@dataclass(frozen=True)
class TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.feature < 0:
            return np.full(features.shape[0], self.value)
        mask = features[:, self.feature] <= self.threshold
        result = np.empty(features.shape[0])
        result[mask] = self.left.predict(features[mask])
        result[~mask] = self.right.predict(features[~mask])
        return result

    def to_dict(self, names: Sequence[str]) -> dict[str, object]:
        if self.feature < 0:
            return {"value": self.value}
        return {
            "feature": names[self.feature], "threshold": self.threshold,
            "left": self.left.to_dict(names), "right": self.right.to_dict(names),
        }


@dataclass(frozen=True)
class TreeModel:
    root: TreeNode
    feature_indices: np.ndarray
    feature_names: tuple[str, ...]
    depth: int
    classification: bool

    def alpha(self, features: np.ndarray) -> np.ndarray:
        return np.clip(self.root.predict(np.asarray(features)[:, self.feature_indices]), 0, 1)


def fit_tree(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    names: Sequence[str],
    depth: int,
    *,
    classification: bool,
) -> TreeModel:
    selected = np.asarray(features, dtype=np.float64)[:, indices]
    target = np.asarray(target, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), EPSILON)

    def leaf_value(rows: np.ndarray) -> float:
        average = float(np.average(target[rows], weights=weights[rows]))
        return float(average >= 0.5) if classification else average

    def leaf_cost(rows: np.ndarray) -> float:
        value = leaf_value(rows)
        return float(np.sum(weights[rows] * (target[rows] - value) ** 2))

    def build(rows: np.ndarray, remaining: int) -> TreeNode:
        value = leaf_value(rows)
        if remaining == 0 or rows.size < 32:
            return TreeNode(value)
        best = None
        parent_cost = leaf_cost(rows)
        for feature in range(selected.shape[1]):
            values = selected[rows, feature]
            thresholds = np.unique(np.quantile(values, np.linspace(0.08, 0.92, 15)))
            for threshold in thresholds:
                left = rows[values <= threshold]
                right = rows[values > threshold]
                if left.size < 16 or right.size < 16:
                    continue
                cost = leaf_cost(left) + leaf_cost(right)
                if best is None or cost < best[0]:
                    best = (cost, feature, float(threshold), left, right)
        if best is None or best[0] >= parent_cost - 1e-12:
            return TreeNode(value)
        return TreeNode(
            value, best[1], best[2], build(best[3], remaining - 1),
            build(best[4], remaining - 1),
        )

    root = build(np.arange(selected.shape[0]), depth)
    return TreeModel(root, indices, tuple(names[index] for index in indices), depth, classification)


def sse_for_alpha(table: PatchTable, alpha: np.ndarray) -> np.ndarray:
    checked = np.clip(np.asarray(alpha, dtype=np.float64), 0, 1)
    return table.mark_sse + 2 * checked * table.cross + checked * checked * table.quadratic


def quality_metrics(table: PatchTable, alpha: np.ndarray) -> dict[str, object]:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0, 1)
    sse = np.maximum(0.0, sse_for_alpha(table, alpha))
    value_count = int(3 * np.sum(table.pixels))
    mse = float(np.sum(sse) / value_count)
    patch_rms = np.sqrt(sse / (3 * table.pixels))
    component_sse = (
        table.component_mark_sse
        + 2 * alpha[:, None] * table.component_cross
        + alpha[:, None] ** 2 * table.component_quadratic
    )
    component_rms = np.sqrt(np.maximum(0.0, np.sum(component_sse, axis=0) / np.sum(table.pixels)))
    return {
        "alpha_mean": float(np.mean(alpha)),
        "coverage_nonzero": float(np.mean(alpha > 1e-9)),
        "coverage_mlri_majority": float(np.mean(alpha >= 0.5)),
        "component_rms": dict(zip(("L", "C1", "C2"), map(float, component_rms))),
        "mse": mse,
        "patch_rms": {
            "median": float(np.percentile(patch_rms, 50)),
            "p75": float(np.percentile(patch_rms, 75)),
            "p90": float(np.percentile(patch_rms, 90)),
            "p95": float(np.percentile(patch_rms, 95)),
            "p99": float(np.percentile(patch_rms, 99)),
            "maximum": float(np.max(patch_rms)),
        },
        "psnr_db": math.inf if mse == 0 else float(10 * math.log10(1 / mse)),
    }


def classification_metrics(table: PatchTable, alpha: np.ndarray) -> dict[str, float]:
    scores = np.asarray(alpha, dtype=np.float64)
    prediction = scores >= 0.5
    labels = table.labels.astype(bool)
    positive = labels
    negative = ~labels
    true_positive = np.mean(prediction[positive]) if positive.any() else 0.0
    true_negative = np.mean(~prediction[negative]) if negative.any() else 0.0
    weights = np.abs(table.margins)
    weighted_error = float(np.sum(weights * (prediction != labels)) / max(np.sum(weights), EPSILON))
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    positive_count = int(np.sum(positive))
    negative_count = int(np.sum(negative))
    auc = 0.5
    if positive_count and negative_count:
        auc = float((np.sum(ranks[positive]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))
    precision = float(np.sum(prediction & positive) / max(1, np.sum(prediction)))
    recall = float(np.sum(prediction & positive) / max(1, positive_count))
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(0.5 * (true_positive + true_negative)),
        "margin_weighted_error": weighted_error,
        "mlri_precision": precision,
        "mlri_recall": recall,
        "roc_auc": auc,
    }


def calibration_metrics(table: PatchTable, alpha: np.ndarray) -> dict[str, object]:
    predicted = np.clip(np.asarray(alpha, dtype=np.float64), 0, 1)
    target = table.oracle_alpha
    correlation = 0.0
    if np.std(predicted) > 0 and np.std(target) > 0:
        correlation = float(np.corrcoef(predicted, target)[0, 1])
    bins = []
    edges = np.linspace(0, 1, 11)
    for index in range(10):
        mask = (predicted >= edges[index]) & (
            predicted <= edges[index + 1] if index == 9 else predicted < edges[index + 1]
        )
        bins.append({
            "count": int(np.sum(mask)),
            "predicted_mean": None if not mask.any() else float(np.mean(predicted[mask])),
            "oracle_mean": None if not mask.any() else float(np.mean(target[mask])),
        })
    return {
        "bins": bins,
        "correlation": correlation,
        "mae": float(np.mean(np.abs(predicted - target))),
        "oracle_alpha_distribution": {
            "at_zero": float(np.mean(target == 0)),
            "interior": float(np.mean((target > 0) & (target < 1))),
            "at_one": float(np.mean(target == 1)),
            "mean": float(np.mean(target)),
        },
    }


def apply_block_alpha(cases: Sequence[Mapping[str, object]], table: PatchTable, alpha: np.ndarray) -> list[np.ndarray]:
    fields = [np.zeros(np.asarray(case["mosaic"]).shape, dtype=np.float64) for case in cases]
    for index, value in enumerate(np.asarray(alpha)):
        y, y2, x, x2 = table.bounds[index]
        fields[int(table.case_indices[index])][y:y2, x:x2] = value
    return fields


def dense_quality_metrics(
    cases: Sequence[Mapping[str, object]],
    alpha_fields: Sequence[np.ndarray],
    *,
    tail_block_size: int,
) -> dict[str, object]:
    total_sse = 0.0
    total_values = 0
    component_sse = np.zeros(3)
    patch_rms = []
    for case, field in zip(cases, alpha_fields):
        truth = check_rgb(case["truth"])
        mark = check_rgb(case["markesteijn"])
        mlri = check_rgb(case["mlri"])
        alpha = np.clip(np.asarray(field, dtype=np.float64), 0, 1)
        if alpha.shape != truth.shape[1:]:
            raise ValueError("alpha field differs from case shape")
        reconstruction = mark + alpha[None] * (mlri - mark)
        error = reconstruction - truth
        component_error = opponent(error)
        interior = error[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        component_interior = component_error[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        total_sse += float(np.sum(interior * interior))
        component_sse += np.sum(component_interior * component_interior, axis=(1, 2))
        total_values += interior.size
        for y in range(MARGIN, truth.shape[1] - MARGIN, tail_block_size):
            y2 = min(truth.shape[1] - MARGIN, y + tail_block_size)
            for x in range(MARGIN, truth.shape[2] - MARGIN, tail_block_size):
                x2 = min(truth.shape[2] - MARGIN, x + tail_block_size)
                block = error[:, y:y2, x:x2]
                patch_rms.append(math.sqrt(float(np.mean(block * block))))
    mse = total_sse / total_values
    return {
        "alpha_mean": float(np.mean(np.concatenate([field[MARGIN:-MARGIN, MARGIN:-MARGIN].ravel() for field in alpha_fields]))),
        "component_rms": dict(zip(("L", "C1", "C2"), map(float, np.sqrt(component_sse / (total_values / 3))))),
        "mse": mse,
        "patch_rms": {
            "median": float(np.percentile(patch_rms, 50)),
            "p75": float(np.percentile(patch_rms, 75)),
            "p90": float(np.percentile(patch_rms, 90)),
            "p95": float(np.percentile(patch_rms, 95)),
            "p99": float(np.percentile(patch_rms, 99)),
            "maximum": float(np.max(patch_rms)),
        },
        "psnr_db": math.inf if mse == 0 else float(10 * math.log10(1 / mse)),
    }


def alpha_fragmentation(alpha_fields: Sequence[np.ndarray]) -> dict[str, float]:
    boundary_changes = 0
    possible = 0
    positive = 0
    samples = 0
    regions = 0
    for field in alpha_fields:
        labels = np.asarray(field)[MARGIN:-MARGIN, MARGIN:-MARGIN] >= 0.5
        boundary_changes += int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
        boundary_changes += int(np.count_nonzero(labels[1:, :] != labels[:-1, :]))
        possible += labels.shape[0] * (labels.shape[1] - 1)
        possible += (labels.shape[0] - 1) * labels.shape[1]
        positive += int(np.count_nonzero(labels))
        samples += labels.size
        _, count = ndimage.label(labels)
        _, inverse_count = ndimage.label(~labels)
        regions += int(count + inverse_count)
    probability = positive / samples if samples else 0.0
    entropy = 0.0
    for value in (probability, 1 - probability):
        if value > 0:
            entropy -= value * math.log2(value)
    return {
        "binary_label_entropy_bits": entropy,
        "boundary_fraction": boundary_changes / possible if possible else 0.0,
        "connected_regions": regions,
        "mlri_majority_fraction": probability,
    }

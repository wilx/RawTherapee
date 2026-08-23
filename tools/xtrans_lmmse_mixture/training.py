"""Streaming covariance fitting for global and three-bank LMMSE models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.xtrans_lmmse.model import (
    FilterBank,
    PhaseStatistics,
    _dc_transform,
    collect_phase_samples,
    derive_filter_bank,
)
from tools.xtrans_lmmse.population import _Accumulator

from .dataset import TRAIN_COUNTS, load_rgb
from .model import (
    CLASS_COUNT,
    FeatureNormalization,
    MixtureBank,
    MixtureDefinition,
    classify_features,
    normalize_features,
    observable_features,
)


SUPPORT = 11
RIDGE_RATIO = 1e-3
SAMPLES_PER_PHASE_PER_SOURCE = 192
MINIMUM_CLASS_PHASE_SAMPLES = 1024


@dataclass
class _Moment:
    count: int
    x_mean: np.ndarray
    target_mean: np.ndarray
    x_m2: np.ndarray
    target_x_m2: np.ndarray


class _PhaseAccumulator:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.row: _Moment | None = None

    def add(self, x_values: np.ndarray, targets: np.ndarray) -> None:
        x = np.asarray(x_values, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_count or target.shape != (x.shape[0], 3):
            raise ValueError("invalid class covariance samples")
        if x.shape[0] == 0:
            return
        x_mean = np.mean(x, axis=0)
        target_mean = np.mean(target, axis=0)
        centered_x = x - x_mean
        centered_target = target - target_mean
        incoming = _Moment(
            count=x.shape[0],
            x_mean=x_mean,
            target_mean=target_mean,
            x_m2=centered_x.T @ centered_x,
            target_x_m2=centered_target.T @ centered_x,
        )
        current = self.row
        if current is None:
            self.row = incoming
            return
        total = current.count + incoming.count
        ratio = current.count * incoming.count / total
        x_delta = incoming.x_mean - current.x_mean
        target_delta = incoming.target_mean - current.target_mean
        self.row = _Moment(
            count=total,
            x_mean=current.x_mean + x_delta * incoming.count / total,
            target_mean=current.target_mean + target_delta * incoming.count / total,
            x_m2=current.x_m2 + incoming.x_m2 + ratio * np.outer(x_delta, x_delta),
            target_x_m2=(
                current.target_x_m2
                + incoming.target_x_m2
                + ratio * np.outer(target_delta, x_delta)
            ),
        )

    def statistics(self) -> PhaseStatistics:
        row = self.row
        if row is None or row.count < 2:
            raise RuntimeError("insufficient class covariance samples")
        covariance = row.x_m2 / (row.count - 1)
        covariance = (covariance + covariance.T) * 0.5
        cross = row.target_x_m2 / (row.count - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        return PhaseStatistics(
            count=row.count,
            x_mean=row.x_mean.copy(),
            target_mean=row.target_mean.copy(),
            covariance=covariance,
            cross_covariance=cross,
            minimum_eigenvalue=float(eigenvalues[0]),
            maximum_eigenvalue=float(eigenvalues[-1]),
        )


def _sample_rows(source: dict[str, object]):
    truth = load_rgb(source)
    radius = SUPPORT // 2
    return collect_phase_samples(
        truth,
        SUPPORT,
        (radius, radius, truth.shape[2] - radius, truth.shape[1] - radius),
        maximum_per_phase=SAMPLES_PER_PHASE_PER_SOURCE,
    )


def survey_and_train_globals(
    training_rows: list[dict[str, object]],
) -> tuple[dict[int, FilterBank], FeatureNormalization, dict[str, np.ndarray]]:
    if len(training_rows) < max(TRAIN_COUNTS):
        raise ValueError("expanded training split is incomplete")
    accumulator = _Accumulator(SUPPORT * SUPPORT)
    raw_features: dict[str, list[list[np.ndarray]]] = {
        name: [[] for _ in range(18)]
        for name in ("raw_variance", "highpass_variance", "chroma_variation", "anisotropy")
    }
    globals_by_size = {}
    for source_index, source in enumerate(training_rows[: max(TRAIN_COUNTS)]):
        observations, targets = _sample_rows(source)
        accumulator.add(observations, targets, support=SUPPORT, dc_mode="m2")
        for phase in range(18):
            features = observable_features(observations[phase], phase, SUPPORT)
            for name in raw_features:
                raw_features[name][phase].append(features[name])
        count = source_index + 1
        if count in TRAIN_COUNTS:
            globals_by_size[count] = derive_filter_bank(
                accumulator.statistics(), SUPPORT, RIDGE_RATIO,
                basis="rgb", dc_mode="m2",
            )
            print(f"global training_sources={count}", flush=True)
    concatenated = {
        name: [np.concatenate(values) for values in by_phase]
        for name, by_phase in raw_features.items()
    }
    normalization = FeatureNormalization(
        raw_variance_median=np.asarray([
            np.median(values) for values in concatenated["raw_variance"]
        ]),
        highpass_variance_median=np.asarray([
            np.median(values) for values in concatenated["highpass_variance"]
        ]),
        chroma_variation_median=np.asarray([
            np.median(values) for values in concatenated["chroma_variation"]
        ]),
    )
    normalized: dict[str, list[np.ndarray]] = {
        name: [] for name in ("raw_variance", "highpass_variance", "chroma_variation", "anisotropy")
    }
    for phase in range(18):
        phase_features = {name: concatenated[name][phase] for name in concatenated}
        values = normalize_features(phase_features, phase, normalization)
        for name in normalized:
            normalized[name].append(values[name])
    survey = {name: np.concatenate(values) for name, values in normalized.items()}
    return globals_by_size, normalization, survey


def definition_from_quantiles(
    name: str,
    partition: str,
    activity_feature: str,
    survey: dict[str, np.ndarray],
    activity_quantile: float,
    anisotropy_quantile: float,
) -> MixtureDefinition:
    if partition not in ("set1", "set2"):
        raise ValueError("partition must be set1 or set2")
    activity = survey[activity_feature]
    threshold = float(np.quantile(activity, activity_quantile))
    active = activity >= threshold
    anisotropy_values = survey["anisotropy"][active]
    anisotropy_threshold = float(np.quantile(anisotropy_values, anisotropy_quantile))
    activity_width = max(float(np.quantile(activity, 0.75) - np.quantile(activity, 0.25)) * 0.10, 0.02)
    anisotropy_width = max(float(np.quantile(anisotropy_values, 0.75) - np.quantile(anisotropy_values, 0.25)) * 0.10, 0.01)
    class_names = (
        ("smooth", "edge", "texture")
        if partition == "set1"
        else ("low-chroma", "edge", "high-chroma")
    )
    return MixtureDefinition(
        name=name,
        partition=partition,
        activity_feature=activity_feature,
        activity_threshold=threshold,
        anisotropy_threshold=anisotropy_threshold,
        activity_width=activity_width,
        anisotropy_width=anisotropy_width,
        class_names=class_names,
        activity_quantile=activity_quantile,
        anisotropy_quantile=anisotropy_quantile,
    )


def initial_definitions(survey: dict[str, np.ndarray]) -> list[MixtureDefinition]:
    return [
        definition_from_quantiles(
            "set1-raw", "set1", "raw_variance", survey, 0.40, 0.60
        ),
        definition_from_quantiles(
            "set1-highpass", "set1", "highpass_variance", survey, 0.40, 0.60
        ),
        definition_from_quantiles(
            "set2-chroma", "set2", "chroma_variation", survey, 0.40, 0.60
        ),
    ]


def sensitivity_definitions(
    selected: MixtureDefinition, survey: dict[str, np.ndarray]
) -> list[MixtureDefinition]:
    coordinates = (
        (selected.activity_quantile - 0.08, selected.anisotropy_quantile),
        (selected.activity_quantile + 0.08, selected.anisotropy_quantile),
        (selected.activity_quantile, selected.anisotropy_quantile - 0.10),
        (selected.activity_quantile, selected.anisotropy_quantile + 0.10),
    )
    return [
        definition_from_quantiles(
            f"{selected.name}-sensitivity-{index}",
            selected.partition,
            selected.activity_feature,
            survey,
            float(np.clip(activity_q, 0.1, 0.9)),
            float(np.clip(anisotropy_q, 0.1, 0.9)),
        )
        for index, (activity_q, anisotropy_q) in enumerate(coordinates)
    ]


def _derive_class_bank(
    statistics: tuple[PhaseStatistics, ...], global_bank: FilterBank
) -> tuple[FilterBank, np.ndarray]:
    specialized = derive_filter_bank(
        statistics, SUPPORT, RIDGE_RATIO, basis="rgb", dc_mode="m2"
    )
    fallback = specialized.sample_counts < MINIMUM_CLASS_PHASE_SAMPLES
    if not np.any(fallback):
        return specialized, fallback
    arrays = {
        "x_means": specialized.x_means.copy(),
        "target_means": specialized.target_means.copy(),
        "weights": specialized.weights.copy(),
        "condition_numbers": specialized.condition_numbers.copy(),
        "minimum_eigenvalues": specialized.minimum_eigenvalues.copy(),
        "filter_norms": specialized.filter_norms.copy(),
        "sample_counts": specialized.sample_counts.copy(),
    }
    for name in (
        "x_means", "target_means", "weights", "condition_numbers",
        "minimum_eigenvalues", "filter_norms",
    ):
        arrays[name][fallback] = getattr(global_bank, name)[fallback]
    return FilterBank(
        support=SUPPORT,
        ridge_ratio=RIDGE_RATIO,
        basis="rgb",
        dc_mode="m2",
        **arrays,
    ), fallback


def train_mixtures(
    training_rows: list[dict[str, object]],
    definitions: list[MixtureDefinition],
    normalization: FeatureNormalization,
    global_bank: FilterBank,
) -> tuple[dict[str, MixtureBank], dict[str, object]]:
    accumulators = {
        definition.name: [
            [_PhaseAccumulator(SUPPORT * SUPPORT) for _ in range(18)]
            for _ in range(CLASS_COUNT)
        ]
        for definition in definitions
    }
    for source_index, source in enumerate(training_rows[: max(TRAIN_COUNTS)]):
        observations, targets = _sample_rows(source)
        for phase in range(18):
            raw = observable_features(observations[phase], phase, SUPPORT)
            features = normalize_features(raw, phase, normalization)
            transformed_x, transformed_target, _ = _dc_transform(
                observations[phase], targets[phase], phase, "m2", SUPPORT
            )
            if transformed_target is None:
                raise RuntimeError("missing transformed targets")
            for definition in definitions:
                labels = classify_features(features, definition)
                for class_index in range(CLASS_COUNT):
                    selected = labels == class_index
                    accumulators[definition.name][class_index][phase].add(
                        transformed_x[selected], transformed_target[selected]
                    )
        if (source_index + 1) % 10 == 0 or source_index + 1 == max(TRAIN_COUNTS):
            print(
                f"mixture candidates={len(definitions)} training_sources={source_index + 1}",
                flush=True,
            )
    mixtures = {}
    diagnostics = {}
    for definition in definitions:
        banks = []
        fallback_rows = []
        class_rows = []
        for class_index in range(CLASS_COUNT):
            statistics = tuple(
                accumulator.statistics()
                for accumulator in accumulators[definition.name][class_index]
            )
            bank, fallback = _derive_class_bank(statistics, global_bank)
            banks.append(bank)
            fallback_rows.append(fallback)
            phase_rows = []
            for phase, row in enumerate(statistics):
                eigenvalues = np.linalg.eigvalsh(row.covariance)
                maximum = max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
                phase_rows.append({
                    "condition_number_regularized": float(bank.condition_numbers[phase]),
                    "condition_number_unregularized": (
                        float(eigenvalues[-1] / max(eigenvalues[0], np.finfo(np.float64).tiny))
                    ),
                    "effective_rank_1e-6": int(np.count_nonzero(eigenvalues > maximum * 1e-6)),
                    "fallback": bool(fallback[phase]),
                    "filter_norm_maximum": float(np.max(bank.filter_norms[phase])),
                    "minimum_eigenvalue": float(eigenvalues[0]),
                    "phase": phase,
                    "sample_count": row.count,
                    "top_eigenvalues": [float(value) for value in eigenvalues[-5:][::-1]],
                })
            class_rows.append({
                "class_index": class_index,
                "class_name": definition.class_names[class_index],
                "phases": phase_rows,
                "sample_count": int(sum(row.count for row in statistics)),
            })
        mixture = MixtureBank(
            definition=definition,
            normalization=normalization,
            banks=tuple(banks),
            fallback_to_global=np.asarray(fallback_rows, dtype=bool),
        )
        mixtures[definition.name] = mixture
        diagnostics[definition.name] = {
            "classes": class_rows,
            "fallback_count": int(np.count_nonzero(mixture.fallback_to_global)),
        }
    return mixtures, diagnostics


def definition_json(definition: MixtureDefinition) -> dict[str, object]:
    return {
        "activity_feature": definition.activity_feature,
        "activity_quantile": definition.activity_quantile,
        "activity_threshold": definition.activity_threshold,
        "activity_width": definition.activity_width,
        "anisotropy_quantile": definition.anisotropy_quantile,
        "anisotropy_threshold": definition.anisotropy_threshold,
        "anisotropy_width": definition.anisotropy_width,
        "class_names": list(definition.class_names),
        "name": definition.name,
        "partition": definition.partition,
    }


def normalization_json(normalization: FeatureNormalization) -> dict[str, object]:
    return {
        "chroma_variation_median": normalization.chroma_variation_median.tolist(),
        "highpass_variance_median": normalization.highpass_variance_median.tolist(),
        "raw_variance_median": normalization.raw_variance_median.tolist(),
    }


def survey_summary(survey: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "count": int(values.size),
            "maximum": float(np.max(values)),
            "minimum": float(np.min(values)),
            "quantiles": {
                str(q): float(np.quantile(values, q))
                for q in (0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99)
            },
        }
        for name, values in survey.items()
    }

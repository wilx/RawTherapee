"""Numerical primitives for the X-Trans demosaicer oracle experiment."""

from __future__ import annotations

from collections import deque
import itertools
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


METHODS = (
    "markesteijn",
    "mlri-final",
    "triangulated-rgb",
    "triangulated-chroma",
    "global-b",
    "sparse-alias",
)

PATCH_SIZES = (1, 3, 7, 15, 31)

# Rows map RGB to the orthonormal L/C1/C2 basis used by the preceding alias
# experiments and by the experiment specification.
OPPONENT_BASIS = np.asarray(
    (
        (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
        (1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)),
        (1 / math.sqrt(6), -2 / math.sqrt(6), 1 / math.sqrt(6)),
    ),
    dtype=np.float64,
)


def check_rgb(image: np.ndarray) -> np.ndarray:
    checked = np.asarray(image, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[0] != 3:
        raise ValueError("RGB image must have shape (3,height,width)")
    if checked.shape[1] < 1 or checked.shape[2] < 1:
        raise ValueError("RGB image dimensions must be nonzero")
    if not np.isfinite(checked).all():
        raise ValueError("RGB image must contain only finite values")
    return checked


def opponent(image: np.ndarray) -> np.ndarray:
    return np.einsum("dc,cyx->dyx", OPPONENT_BASIS, check_rgb(image))


def from_opponent(image: np.ndarray) -> np.ndarray:
    checked = check_rgb(image)
    return np.einsum("cd,dyx->cyx", OPPONENT_BASIS.T, checked)


def evaluation_slices(shape: Sequence[int], margin: int) -> tuple[slice, slice]:
    height, width = int(shape[-2]), int(shape[-1])
    if margin < 0 or 2 * margin >= min(height, width):
        raise ValueError("comparison margin leaves no interior")
    return slice(margin, height - margin), slice(margin, width - margin)


def mse_psnr(error: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean(np.asarray(error, dtype=np.float64) ** 2))
    return mse, math.inf if mse == 0.0 else float(10 * math.log10(1 / mse))


def method_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    cfa: np.ndarray,
    *,
    margin: int,
) -> dict[str, object]:
    truth = check_rgb(truth)
    reconstruction = check_rgb(reconstruction)
    if reconstruction.shape != truth.shape:
        raise ValueError("truth and reconstruction shapes differ")
    cfa = np.asarray(cfa, dtype=np.int8)
    if cfa.shape != truth.shape[1:]:
        raise ValueError("CFA shape differs from image")
    ys, xs = evaluation_slices(truth.shape, margin)
    error = reconstruction[:, ys, xs] - truth[:, ys, xs]
    opponent_error = opponent(reconstruction)[:, ys, xs] - opponent(truth)[:, ys, xs]
    mse, psnr = mse_psnr(error)
    channel_rms = [float(np.sqrt(np.mean(error[channel] ** 2))) for channel in range(3)]
    component_rms = [
        float(np.sqrt(np.mean(opponent_error[channel] ** 2)))
        for channel in range(3)
    ]
    sampled_error = []
    interpolation_error = []
    interior_cfa = cfa[ys, xs]
    for channel in range(3):
        sampled_error.append(error[channel][interior_cfa == channel])
        interpolation_error.append(error[channel][interior_cfa != channel])
    sampled = np.concatenate(sampled_error)
    interpolated = np.concatenate(interpolation_error)
    interpolation_mse, interpolation_psnr = mse_psnr(interpolated)
    return {
        "channel_rms": dict(zip(("R", "G", "B"), channel_rms)),
        "component_rms": dict(zip(("L", "C1", "C2"), component_rms)),
        "interpolation_only_psnr_db": interpolation_psnr,
        "interpolation_only_rms": math.sqrt(interpolation_mse),
        "native_sample_maximum": float(np.max(np.abs(sampled))),
        "native_sample_rms": float(np.sqrt(np.mean(sampled**2))),
        "psnr_db": psnr,
        "rgb_rms": math.sqrt(mse),
    }


def normalized_correlation(left: np.ndarray, right: np.ndarray, *, centered: bool) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size != b.size or not a.size:
        raise ValueError("correlation vectors must have the same nonzero size")
    if centered:
        a = a - a.mean()
        b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def _target_indices(target: str) -> tuple[int, ...]:
    targets = {
        "rgb": (0, 1, 2),
        "luminance": (0,),
        "chroma": (1, 2),
        "c1": (1,),
        "c2": (2,),
    }
    try:
        return targets[target]
    except KeyError as error:
        raise ValueError(f"unknown selection target {target}") from error


def block_oracle(
    truth: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    methods: Sequence[str],
    block_size: int,
    *,
    margin: int,
    target: str = "rgb",
) -> tuple[np.ndarray, np.ndarray]:
    """Select one method for every nonoverlapping, globally anchored block."""

    truth = check_rgb(truth)
    if block_size < 1:
        raise ValueError("block size must be positive")
    if not methods:
        raise ValueError("oracle requires at least one method")
    checked = {name: check_rgb(outputs[name]) for name in methods}
    if any(image.shape != truth.shape for image in checked.values()):
        raise ValueError("oracle inputs differ in shape")
    target_truth = truth if target == "rgb" else opponent(truth)
    target_outputs = {
        name: image if target == "rgb" else opponent(image)
        for name, image in checked.items()
    }
    components = _target_indices(target)
    selected = np.asarray(checked[methods[0]], dtype=np.float64).copy()
    labels = np.full(truth.shape[1:], -1, dtype=np.int16)
    ys, xs = evaluation_slices(truth.shape, margin)
    if block_size == 1:
        energies = np.stack(
            [
                np.sum(
                    (target_outputs[name][components, ys, xs]
                     - target_truth[components, ys, xs]) ** 2,
                    axis=0,
                )
                for name in methods
            ]
        )
        interior_labels = np.argmin(energies, axis=0).astype(np.int16)
        labels[ys, xs] = interior_labels
        stacked = np.stack([checked[name][:, ys, xs] for name in methods])
        selected[:, ys, xs] = np.take_along_axis(
            stacked,
            interior_labels[None, None, :, :],
            axis=0,
        )[0]
        return selected, labels
    for y in range(ys.start, ys.stop, block_size):
        y2 = min(ys.stop, y + block_size)
        for x in range(xs.start, xs.stop, block_size):
            x2 = min(xs.stop, x + block_size)
            errors = []
            for name in methods:
                difference = (
                    target_outputs[name][components, y:y2, x:x2]
                    - target_truth[components, y:y2, x:x2]
                )
                errors.append(float(np.sum(difference * difference)))
            winner = int(np.argmin(errors))
            selected[:, y:y2, x:x2] = checked[methods[winner]][:, y:y2, x:x2]
            labels[y:y2, x:x2] = winner
    return selected, labels


def independent_component_oracle(
    truth: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    methods: Sequence[str],
    block_size: int,
    *,
    margin: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    truth_components = opponent(truth)
    output_components = {name: opponent(outputs[name]) for name in methods}
    selected = output_components[methods[0]].copy()
    labels = []
    for component, target in enumerate(("luminance", "c1", "c2")):
        _, component_labels = block_oracle(
            truth, outputs, methods, block_size, margin=margin, target=target
        )
        labels.append(component_labels)
        ys, xs = evaluation_slices(truth.shape, margin)
        for method_index, name in enumerate(methods):
            mask = component_labels[ys, xs] == method_index
            plane = selected[component, ys, xs]
            plane[mask] = output_components[name][component, ys, xs][mask]
        # Keep the border from the first method; it is excluded from metrics.
    return from_opponent(selected), labels


def convex_blend_oracle(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    block_size: int,
    *,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    truth = check_rgb(truth)
    left = check_rgb(left)
    right = check_rgb(right)
    if left.shape != truth.shape or right.shape != truth.shape:
        raise ValueError("blend inputs differ in shape")
    blended = right.copy()
    alpha_map = np.full(truth.shape[1:], np.nan, dtype=np.float64)
    ys, xs = evaluation_slices(truth.shape, margin)
    if block_size == 1:
        delta = left[:, ys, xs] - right[:, ys, xs]
        target = truth[:, ys, xs] - right[:, ys, xs]
        denominator = np.sum(delta * delta, axis=0)
        numerator = np.sum(delta * target, axis=0)
        alpha = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        alpha = np.clip(alpha, 0, 1)
        alpha_map[ys, xs] = alpha
        blended[:, ys, xs] = alpha[None] * left[:, ys, xs] + (1 - alpha[None]) * right[:, ys, xs]
        return blended, alpha_map
    for y in range(ys.start, ys.stop, block_size):
        y2 = min(ys.stop, y + block_size)
        for x in range(xs.start, xs.stop, block_size):
            x2 = min(xs.stop, x + block_size)
            delta = left[:, y:y2, x:x2] - right[:, y:y2, x:x2]
            target = truth[:, y:y2, x:x2] - right[:, y:y2, x:x2]
            denominator = float(np.sum(delta * delta))
            alpha = 0.0 if denominator == 0 else float(np.sum(delta * target) / denominator)
            alpha = min(1.0, max(0.0, alpha))
            blended[:, y:y2, x:x2] = (
                alpha * left[:, y:y2, x:x2]
                + (1 - alpha) * right[:, y:y2, x:x2]
            )
            alpha_map[y:y2, x:x2] = alpha
    return blended, alpha_map


def winner_statistics(labels: np.ndarray, method_count: int, *, margin: int) -> dict[str, object]:
    checked = np.asarray(labels)
    ys, xs = evaluation_slices((3, *checked.shape), margin)
    interior = checked[ys, xs]
    if np.any(interior < 0) or np.any(interior >= method_count):
        raise ValueError("winner map contains an invalid label")
    counts = np.bincount(interior.ravel(), minlength=method_count)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    normalized_entropy = entropy / math.log2(method_count) if method_count > 1 else 0.0
    horizontal = int(np.count_nonzero(interior[:, 1:] != interior[:, :-1]))
    vertical = int(np.count_nonzero(interior[1:, :] != interior[:-1, :]))
    boundary_edges = horizontal + vertical
    possible_edges = interior.shape[0] * max(0, interior.shape[1] - 1) + max(
        0, interior.shape[0] - 1
    ) * interior.shape[1]

    visited = np.zeros(interior.shape, dtype=bool)
    regions = 0
    for start_y in range(interior.shape[0]):
        for start_x in range(interior.shape[1]):
            if visited[start_y, start_x]:
                continue
            regions += 1
            label = interior[start_y, start_x]
            queue = deque(((start_y, start_x),))
            visited[start_y, start_x] = True
            while queue:
                y, x = queue.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < interior.shape[0]
                        and 0 <= nx < interior.shape[1]
                        and not visited[ny, nx]
                        and interior[ny, nx] == label
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return {
        "boundary_fraction": boundary_edges / possible_edges if possible_edges else 0.0,
        "connected_regions": regions,
        "label_entropy_bits": entropy,
        "normalized_entropy": normalized_entropy,
        "selection_fractions": [float(value) for value in probabilities],
    }


def label_consistency(
    left: np.ndarray, right: np.ndarray, *, margin: int
) -> float:
    if left.shape != right.shape:
        raise ValueError("winner map shapes differ")
    ys, xs = evaluation_slices((3, *left.shape), margin)
    return float(np.mean(left[ys, xs] == right[ys, xs]))


def radial_frequency_summary(error: np.ndarray) -> dict[str, float]:
    checked = check_rgb(error)
    transformed = opponent(checked)
    height, width = checked.shape[1:]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    bands = {
        "low": radius < 0.125,
        "middle": (radius >= 0.125) & (radius < 0.30),
        "high": radius >= 0.30,
    }
    result = {}
    names = ("L", "C1", "C2")
    for component, name in enumerate(names):
        spectrum = np.fft.fft2(transformed[component])
        energy = np.abs(spectrum) ** 2
        total = float(np.sum(energy))
        for band, mask in bands.items():
            result[f"{name}_{band}_fraction"] = (
                float(np.sum(energy[mask]) / total) if total else 0.0
            )
    return result


def patch_rms_values(error: np.ndarray, block_size: int) -> np.ndarray:
    checked = check_rgb(error)
    values = []
    for y in range(0, checked.shape[1], block_size):
        for x in range(0, checked.shape[2], block_size):
            block = checked[:, y : y + block_size, x : x + block_size]
            values.append(math.sqrt(float(np.mean(block * block))))
    return np.asarray(values)


def percentiles(values: Iterable[float]) -> dict[str, float]:
    checked = np.asarray(tuple(values), dtype=np.float64)
    if not checked.size:
        raise ValueError("percentile input is empty")
    return {
        "median": float(np.percentile(checked, 50)),
        "p75": float(np.percentile(checked, 75)),
        "p90": float(np.percentile(checked, 90)),
        "p95": float(np.percentile(checked, 95)),
        "maximum": float(np.max(checked)),
    }


def combinations_by_size(methods: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    return itertools.combinations(methods, size)

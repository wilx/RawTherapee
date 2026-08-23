"""Bennett et al. two-color likelihood and controlled oracle diagnostics.

The likelihood follows equations (5)--(12) of:

    E. P. Bennett et al., "Video and Image Bayesian Demosaicing with a
    Two Color Image Prior", ECCV 2006, DOI 10.1007/11744023_40.

In particular, all neighboring CFA samples are modeled as noisy observations
of the *target pixel's* single blend coordinate.  They are not assigned their
own blend coordinates.  This distinction is easy to miss when adapting the
paper to an irregular CFA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


BENNETT_LAMBDA = 6.0
BENNETT_NORMALIZED_CHANNEL_THRESHOLD = 2.0 / 255.0


@dataclass(frozen=True)
class BayesianAlphaResult:
    alpha: np.ndarray
    unconstrained_alpha: np.ndarray
    likelihood_alpha: np.ndarray
    posterior_variance: np.ndarray
    log_likelihood_margin: np.ndarray
    identifiable: np.ndarray
    selected_candidate: np.ndarray


def endpoints_from_fit(fit, endpoint: str = "extrema") -> tuple[np.ndarray, np.ndarray]:
    """Return HWC endpoint colors from a color-line fit."""
    if endpoint == "extrema":
        low = fit.minimum_projection
        high = fit.maximum_projection
    elif endpoint == "two-means":
        low = fit.two_means_low
        high = fit.two_means_high
    else:
        raise ValueError("unknown endpoint definition")
    first = fit.mean + low[..., None] * fit.direction
    second = fit.mean + high[..., None] * fit.direction
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("non-finite endpoints")
    return first, second


def projected_truth_alpha(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Project CHW truth onto the bounded line segment J--K."""
    checked = np.moveaxis(np.asarray(truth, dtype=np.float64), 0, -1)
    direction = second - first
    denominator = np.sum(direction * direction, axis=-1)
    numerator = np.sum((checked - first) * direction, axis=-1)
    alpha = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, 0.5, dtype=np.float64),
        where=denominator > 1e-20,
    )
    return np.clip(alpha, 0.0, 1.0)


def _shift_reflect(values: np.ndarray, dy: int, dx: int, radius: int) -> np.ndarray:
    height, width = values.shape[:2]
    padding = ((radius, radius), (radius, radius)) + ((0, 0),) * (values.ndim - 2)
    padded = np.pad(values, padding, mode="reflect")
    return padded[
        radius + dy : radius + dy + height,
        radius + dx : radius + dx + width,
        ...,
    ]


def _sample_endpoint(endpoint: np.ndarray, channels: np.ndarray) -> np.ndarray:
    return np.take_along_axis(endpoint, channels[..., None], axis=-1)[..., 0]


def direct_alpha(
    scalar: np.ndarray,
    cfa: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    channel_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Equation (2): recover alpha from only the target CFA component."""
    observed = np.asarray(scalar, dtype=np.float64)
    channels = np.asarray(cfa, dtype=np.int64)
    low = _sample_endpoint(first, channels)
    delta = _sample_endpoint(second - first, channels)
    identifiable = np.abs(delta) >= channel_threshold
    alpha = np.divide(
        observed - low,
        delta,
        out=np.full(observed.shape, 0.5, dtype=np.float64),
        where=identifiable,
    )
    return np.clip(alpha, 0.0, 1.0), identifiable


def _noise_by_channel(noise_sigma: float | tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(noise_sigma, dtype=np.float64)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,) or np.any(values <= 0.0) or not np.isfinite(values).all():
        raise ValueError("noise sigma must contain three positive finite values")
    return values


def bennett_alpha(
    scalar: np.ndarray,
    cfa: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    support_radius: int = 2,
    spatial_lambda: float = BENNETT_LAMBDA,
    noise_sigma: float | tuple[float, float, float] = 2.0 / 255.0,
    channel_threshold: float = BENNETT_NORMALIZED_CHANNEL_THRESHOLD,
    eta: float = 1.0,
) -> BayesianAlphaResult:
    """Evaluate the single-image product-of-Gaussians model (eqs. 5--12).

    ``eta=1`` disables the endpoint impulses and isolates the measurement
    likelihood.  The paper states only ``eta < 1`` and does not publish the
    value, so callers must make the sensitivity choice explicit.
    """
    observed = np.asarray(scalar, dtype=np.float64)
    channels = np.asarray(cfa, dtype=np.int64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        observed.shape != channels.shape
        or first.shape != (*observed.shape, 3)
        or second.shape != first.shape
        or np.any((channels < 0) | (channels > 2))
        or support_radius < 0
        or spatial_lambda < 0.0
        or not 0.0 < eta <= 1.0
    ):
        raise ValueError("invalid Bennett likelihood inputs")
    if not all(np.isfinite(value).all() for value in (observed, first, second)):
        raise ValueError("non-finite Bennett likelihood input")
    sigma_channel = _noise_by_channel(noise_sigma)

    precision = np.zeros(observed.shape, dtype=np.float64)
    weighted_alpha = np.zeros(observed.shape, dtype=np.float64)
    constant = np.zeros(observed.shape, dtype=np.float64)
    endpoint_delta = second - first
    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            neighbor = _shift_reflect(observed, dy, dx, support_radius)
            neighbor_cfa = _shift_reflect(channels, dy, dx, support_radius)
            low = _sample_endpoint(first, neighbor_cfa)
            delta = _sample_endpoint(endpoint_delta, neighbor_cfa)
            sigma = sigma_channel[neighbor_cfa] * (
                1.0 + spatial_lambda * math.hypot(dx, dy)
            )
            usable = np.abs(delta) >= channel_threshold
            inverse_variance = np.where(usable, 1.0 / (sigma * sigma), 0.0)
            residual = neighbor - low
            precision += inverse_variance * delta * delta
            weighted_alpha += inverse_variance * delta * residual
            constant += inverse_variance * residual * residual

    identifiable = precision > 0.0
    unconstrained = np.divide(
        weighted_alpha,
        precision,
        out=np.full(observed.shape, 0.5, dtype=np.float64),
        where=identifiable,
    )
    likelihood_alpha = np.clip(unconstrained, 0.0, 1.0)

    # Expanded weighted least-squares log likelihood.  The additive Gaussian
    # normalization is common to all three candidates and cancels.
    def log_likelihood(alpha: np.ndarray | float) -> np.ndarray:
        return -0.5 * (
            constant - 2.0 * np.asarray(alpha) * weighted_alpha
            + np.asarray(alpha) * np.asarray(alpha) * precision
        )

    candidates = np.stack(
        (
            np.zeros(observed.shape),
            likelihood_alpha,
            np.ones(observed.shape),
        ),
        axis=-1,
    )
    log_scores = np.stack(
        (
            log_likelihood(0.0),
            log_likelihood(likelihood_alpha) + math.log(eta),
            log_likelihood(1.0),
        ),
        axis=-1,
    )
    selected = np.argmax(log_scores, axis=-1)
    alpha = np.take_along_axis(candidates, selected[..., None], axis=-1)[..., 0]
    sorted_scores = np.sort(log_scores, axis=-1)
    margin = sorted_scores[..., -1] - sorted_scores[..., -2]
    posterior_variance = np.divide(
        1.0,
        precision,
        out=np.full(observed.shape, np.inf, dtype=np.float64),
        where=identifiable,
    )
    alpha = np.where(identifiable, alpha, 0.5)
    return BayesianAlphaResult(
        alpha=alpha,
        unconstrained_alpha=unconstrained,
        likelihood_alpha=likelihood_alpha,
        posterior_variance=posterior_variance,
        log_likelihood_margin=margin,
        identifiable=identifiable,
        selected_candidate=selected,
    )


def reconstruct(
    first: np.ndarray,
    second: np.ndarray,
    alpha: np.ndarray,
    *,
    scalar: np.ndarray | None = None,
    cfa: np.ndarray | None = None,
    reinject_samples: bool = True,
) -> np.ndarray:
    """Reconstruct CHW RGB and optionally apply Bennett's sample forcing."""
    result = first + np.asarray(alpha, dtype=np.float64)[..., None] * (second - first)
    if reinject_samples:
        if scalar is None or cfa is None:
            raise ValueError("sample reinjection requires scalar and CFA")
        for channel in range(3):
            mask = np.asarray(cfa) == channel
            result[..., channel][mask] = np.asarray(scalar)[mask]
    if not np.isfinite(result).all():
        raise ValueError("non-finite reconstruction")
    return np.moveaxis(result, -1, 0)


def oracle_neighbor_alpha(
    truth_alpha: np.ndarray,
    *,
    support_radius: int,
    spatial_lambda: float = BENNETT_LAMBDA,
) -> np.ndarray:
    """A2: infer the target from neighboring true alphas, excluding itself."""
    values = np.asarray(truth_alpha, dtype=np.float64)
    numerator = np.zeros(values.shape, dtype=np.float64)
    denominator = 0.0
    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            if dx == 0 and dy == 0:
                continue
            weight = 1.0 / (1.0 + spatial_lambda * math.hypot(dx, dy)) ** 2
            numerator += weight * _shift_reflect(values, dy, dx, support_radius)
            denominator += weight
    if denominator == 0.0:
        return values.copy()
    return np.clip(numerator / denominator, 0.0, 1.0)


def oracle_neighbor_alpha_from_rgb(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    support_radius: int,
    spatial_lambda: float = BENNETT_LAMBDA,
) -> np.ndarray:
    """A2 with every neighbor projected onto the *target's* endpoint line."""
    truth_hwc = np.moveaxis(np.asarray(truth, dtype=np.float64), 0, -1)
    direction = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    denominator_line = np.sum(direction * direction, axis=-1)
    numerator = np.zeros(denominator_line.shape, dtype=np.float64)
    denominator_weight = 0.0
    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = _shift_reflect(truth_hwc, dy, dx, support_radius)
            alpha = np.divide(
                np.sum((neighbor - first) * direction, axis=-1),
                denominator_line,
                out=np.full(denominator_line.shape, 0.5),
                where=denominator_line > 1e-20,
            )
            weight = 1.0 / (1.0 + spatial_lambda * math.hypot(dx, dy)) ** 2
            numerator += weight * alpha
            denominator_weight += weight
    if denominator_weight == 0.0:
        return projected_truth_alpha(truth, first, second)
    return np.clip(numerator / denominator_weight, 0.0, 1.0)


def oracle_plane_alpha(
    truth_alpha: np.ndarray,
    *,
    support_radius: int,
    spatial_lambda: float = BENNETT_LAMBDA,
) -> np.ndarray:
    """A3: weighted spatial-plane intercept from neighboring true alphas.

    For a complete symmetric square support the fitted intercept is
    algebraically the same as the A2 weighted mean.  This implementation keeps
    the explicit plane solve so the equivalence is tested rather than assumed.
    """
    values = np.asarray(truth_alpha, dtype=np.float64)
    normal = np.zeros((3, 3), dtype=np.float64)
    shifted: list[tuple[np.ndarray, np.ndarray, float]] = []
    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            if dx == 0 and dy == 0:
                continue
            design = np.asarray((1.0, float(dx), float(dy)))
            weight = 1.0 / (1.0 + spatial_lambda * math.hypot(dx, dy)) ** 2
            normal += weight * np.outer(design, design)
            shifted.append((_shift_reflect(values, dy, dx, support_radius), design, weight))
    if not shifted:
        return values.copy()
    inverse = np.linalg.inv(normal)
    right = np.zeros((*values.shape, 3), dtype=np.float64)
    for neighbor, design, weight in shifted:
        right += weight * neighbor[..., None] * design
    coefficients = np.einsum("ij,hwj->hwi", inverse, right, optimize=True)
    return np.clip(coefficients[..., 0], 0.0, 1.0)


def oracle_plane_alpha_from_rgb(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    support_radius: int,
    spatial_lambda: float = BENNETT_LAMBDA,
) -> np.ndarray:
    """A3 spatial plane after projecting neighbors onto each target line."""
    truth_hwc = np.moveaxis(np.asarray(truth, dtype=np.float64), 0, -1)
    direction = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    denominator_line = np.sum(direction * direction, axis=-1)
    normal = np.zeros((3, 3), dtype=np.float64)
    right = np.zeros((*denominator_line.shape, 3), dtype=np.float64)
    count = 0
    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = _shift_reflect(truth_hwc, dy, dx, support_radius)
            alpha = np.divide(
                np.sum((neighbor - first) * direction, axis=-1),
                denominator_line,
                out=np.full(denominator_line.shape, 0.5),
                where=denominator_line > 1e-20,
            )
            design = np.asarray((1.0, float(dx), float(dy)))
            weight = 1.0 / (1.0 + spatial_lambda * math.hypot(dx, dy)) ** 2
            normal += weight * np.outer(design, design)
            right += weight * alpha[..., None] * design
            count += 1
    if count == 0:
        return projected_truth_alpha(truth, first, second)
    coefficients = np.einsum("ij,hwj->hwi", np.linalg.inv(normal), right, optimize=True)
    return np.clip(coefficients[..., 0], 0.0, 1.0)

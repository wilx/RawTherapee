"""Independent Phase 9 raw-domain wrapper for reviewed X-Trans RTNN weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

from .demosaicnet_reference import (
    CANONICAL_XTRANS,
    DemosaicNetXTransReference,
    deterministic_reference_execution,
    reference_from_loaded_rtnn,
)
from .rtnn_reader import LoadedRTNN, read_rtnn


RawDomain = Literal["linear", "gamma22"]
TRANSFORM_MATRICES = (
    (1, 0, 0, 1),
    (0, -1, 1, 0),
    (-1, 0, 0, -1),
    (0, 1, -1, 0),
    (-1, 0, 0, 1),
    (1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, -1, -1, 0),
)


class RawWrapperError(RuntimeError):
    """The raw mosaic cannot be processed under the Phase 9 contract."""


@dataclass(frozen=True)
class CfaTransform:
    a: int
    b: int
    c: int
    d: int
    ox: int
    oy: int


@dataclass(frozen=True)
class CanonicalView:
    transform: CfaTransform
    minimum_x: int
    minimum_y: int
    width: int
    height: int
    inverse_a: int
    inverse_b: int
    inverse_c: int
    inverse_d: int

    @classmethod
    def create(
        cls,
        transform: CfaTransform,
        actual_width: int,
        actual_height: int,
    ) -> "CanonicalView":
        if actual_width <= 0 or actual_height <= 0:
            raise RawWrapperError("raw image dimensions are empty")
        determinant = transform.a * transform.d - transform.b * transform.c
        if determinant not in (-1, 1):
            raise RawWrapperError("CFA transform is singular")
        corners = (
            (0, 0),
            (actual_width - 1, 0),
            (0, actual_height - 1),
            (actual_width - 1, actual_height - 1),
        )
        mapped = tuple(
            (
                transform.a * x + transform.b * y + transform.ox,
                transform.c * x + transform.d * y + transform.oy,
            )
            for x, y in corners
        )
        minimum_x = min(point[0] for point in mapped)
        maximum_x = max(point[0] for point in mapped)
        minimum_y = min(point[1] for point in mapped)
        maximum_y = max(point[1] for point in mapped)
        return cls(
            transform=transform,
            minimum_x=minimum_x,
            minimum_y=minimum_y,
            width=maximum_x - minimum_x + 1,
            height=maximum_y - minimum_y + 1,
            inverse_a=determinant * transform.d,
            inverse_b=-determinant * transform.b,
            inverse_c=-determinant * transform.c,
            inverse_d=determinant * transform.a,
        )

    def actual_to_canonical(self, x: int, y: int) -> tuple[int, int]:
        transform = self.transform
        return (
            transform.a * x + transform.b * y + transform.ox - self.minimum_x,
            transform.c * x + transform.d * y + transform.oy - self.minimum_y,
        )

    def canonical_to_actual(self, u: int, v: int) -> tuple[int, int]:
        transform = self.transform
        x = u + self.minimum_x - transform.ox
        y = v + self.minimum_y - transform.oy
        return (
            self.inverse_a * x + self.inverse_b * y,
            self.inverse_c * x + self.inverse_d * y,
        )

    def canonical_colour(self, u: int, v: int) -> int:
        return CANONICAL_XTRANS[
            (v + self.minimum_y) % 6
        ][(u + self.minimum_x) % 6]


def cfa_from_transform(transform: CfaTransform) -> tuple[tuple[int, ...], ...]:
    """Construct the actual 6x6 CFA represented by a canonical transform."""

    return tuple(
        tuple(
            CANONICAL_XTRANS[
                (transform.c * x + transform.d * y + transform.oy) % 6
            ][(transform.a * x + transform.b * y + transform.ox) % 6]
            for x in range(6)
        )
        for y in range(6)
    )


def unique_xtrans_cfas() -> tuple[tuple[tuple[int, ...], ...], ...]:
    result = []
    for a, b, c, d in TRANSFORM_MATRICES:
        for oy in range(6):
            for ox in range(6):
                cfa = cfa_from_transform(CfaTransform(a, b, c, d, ox, oy))
                if cfa not in result:
                    result.append(cfa)
    if len(result) != 18:
        raise AssertionError(f"expected 18 unique X-Trans matrices; found {len(result)}")
    return tuple(result)


def find_canonical_transform(
    actual: Sequence[Sequence[int]],
) -> CfaTransform:
    if len(actual) != 6 or any(len(row) != 6 for row in actual):
        raise RawWrapperError("X-Trans CFA must be a 6x6 matrix")
    for a, b, c, d in TRANSFORM_MATRICES:
        for oy in range(6):
            for ox in range(6):
                candidate = CfaTransform(a, b, c, d, ox, oy)
                if cfa_from_transform(candidate) == tuple(tuple(row) for row in actual):
                    return candidate
    raise RawWrapperError(
        "X-Trans CFA is not a supported phase or orientation of the canonical pattern"
    )


def _reflect101_indices(size: int, halo: int = 12) -> torch.Tensor:
    if size <= 0:
        raise RawWrapperError("cannot reflect an empty dimension")
    if size == 1:
        return torch.zeros(size + 2 * halo, dtype=torch.int64)
    period = 2 * (size - 1)
    coordinates = torch.arange(-halo, size + halo, dtype=torch.int64)
    folded = torch.remainder(coordinates, period)
    return torch.where(folded < size, folded, period - folded)


def _canonical_sparse(
    raw_data: np.ndarray,
    view: CanonicalView,
    domain: RawDomain,
) -> torch.Tensor:
    raw_float = raw_data.astype(np.float32, copy=False)
    if not bool(np.isfinite(raw_float).all()):
        raise RawWrapperError("raw mosaic contains NaN or infinity")
    normalized = np.clip(raw_float / 65535.0, 0.0, 1.0)
    if domain == "gamma22":
        normalized = np.power(normalized, np.float32(1.0 / 2.2)).astype(
            np.float32, copy=False
        )
    elif domain != "linear":
        raise RawWrapperError(f"unsupported raw domain: {domain}")

    sparse = torch.zeros((1, 3, view.height, view.width), dtype=torch.float32)
    for y in range(raw_data.shape[0]):
        for x in range(raw_data.shape[1]):
            u, v = view.actual_to_canonical(x, y)
            sparse[0, view.canonical_colour(u, v), v, u] = float(normalized[y, x])
    return sparse


def run_raw_wrapper_loaded(
    loaded: LoadedRTNN,
    raw_data: np.ndarray,
    actual_cfa: Sequence[Sequence[int]],
    domain: RawDomain,
) -> np.ndarray:
    """Return full-size normalized CHW RGB from an untiled reference run."""

    return run_raw_wrapper_model(
        reference_from_loaded_rtnn(loaded), raw_data, actual_cfa, domain
    )


def run_raw_wrapper_model(
    model: DemosaicNetXTransReference,
    raw_data: np.ndarray,
    actual_cfa: Sequence[Sequence[int]],
    domain: RawDomain,
) -> np.ndarray:
    if raw_data.ndim != 2:
        raise RawWrapperError("raw mosaic must be a two-dimensional scalar array")
    transform = find_canonical_transform(actual_cfa)
    view = CanonicalView.create(transform, raw_data.shape[1], raw_data.shape[0])
    sparse = _canonical_sparse(raw_data, view, domain)
    rows = _reflect101_indices(view.height)
    columns = _reflect101_indices(view.width)
    padded = sparse.index_select(2, rows).index_select(3, columns)

    with deterministic_reference_execution(), torch.inference_mode():
        canonical = torch.clamp(model(padded), 0.0, 1.0)
        if domain == "gamma22":
            canonical = torch.pow(canonical, 2.2)
    if tuple(canonical.shape) != (1, 3, view.height, view.width):
        raise RawWrapperError("reference output does not restore the full image size")
    if not bool(torch.isfinite(canonical).all()):
        raise RawWrapperError("reference output contains NaN or infinity")

    canonical_array = canonical[0].contiguous().cpu().numpy()
    actual = np.empty((3, raw_data.shape[0], raw_data.shape[1]), dtype=np.float32)
    for v in range(view.height):
        for u in range(view.width):
            x, y = view.canonical_to_actual(u, v)
            actual[:, y, x] = canonical_array[:, v, u]
    return actual


def run_raw_wrapper(
    path: str | Path,
    raw_data: np.ndarray,
    actual_cfa: Sequence[Sequence[int]],
    domain: RawDomain,
) -> np.ndarray:
    return run_raw_wrapper_loaded(read_rtnn(path), raw_data, actual_cfa, domain)

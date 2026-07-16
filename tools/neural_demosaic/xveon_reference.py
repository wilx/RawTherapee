"""Independent reference for the hidden X-veon ONNX raw wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np


MODEL_BYTES = 15_536_134
MODEL_SHA256 = "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500"
PATCH_SIZE = 288
OVERLAP = 48
STRIDE = 240
PATTERN = np.asarray(
    (
        (0, 2, 1, 2, 0, 1),
        (1, 1, 0, 1, 1, 2),
        (1, 1, 2, 1, 1, 0),
        (2, 0, 1, 0, 2, 1),
        (1, 1, 2, 1, 1, 0),
        (1, 1, 0, 1, 1, 2),
    ),
    dtype=np.uint8,
)
TRANSFORMS = (
    (1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
    (-1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0),
)


class XVeonReferenceError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transformed_cfa(matrix: tuple[int, int, int, int], ox: int, oy: int) -> np.ndarray:
    a, b, c, d = matrix
    result = np.empty((6, 6), dtype=np.uint8)
    for y in range(6):
        for x in range(6):
            result[y, x] = PATTERN[(c * x + d * y + oy) % 6, (a * x + b * y + ox) % 6]
    return result


def unique_cfas() -> tuple[np.ndarray, ...]:
    result: dict[bytes, np.ndarray] = {}
    for matrix in TRANSFORMS:
        for oy in range(6):
            for ox in range(6):
                cfa = transformed_cfa(matrix, ox, oy)
                result.setdefault(cfa.tobytes(), cfa)
    return tuple(result.values())


def find_transform(actual: np.ndarray) -> tuple[int, int, int, int, int, int]:
    actual = np.asarray(actual, dtype=np.uint8)
    if actual.shape != (6, 6):
        raise XVeonReferenceError("CFA must be 6x6")
    for a, b, c, d in TRANSFORMS:
        for oy in range(6):
            for ox in range(6):
                if np.array_equal(actual, transformed_cfa((a, b, c, d), ox, oy)):
                    return a, b, c, d, ox, oy
    raise XVeonReferenceError("CFA is not an X-veon canonical phase/orientation")


class _View:
    def __init__(self, transform: tuple[int, int, int, int, int, int], width: int, height: int):
        self.a, self.b, self.c, self.d, self.ox, self.oy = transform
        determinant = self.a * self.d - self.b * self.c
        self.ia, self.ib = determinant * self.d, -determinant * self.b
        self.ic, self.id = -determinant * self.c, determinant * self.a
        corners = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
        coordinates = tuple((self.a * x + self.b * y + self.ox, self.c * x + self.d * y + self.oy) for x, y in corners)
        self.min_x = min(point[0] for point in coordinates)
        self.min_y = min(point[1] for point in coordinates)
        self.width = max(point[0] for point in coordinates) - self.min_x + 1
        self.height = max(point[1] for point in coordinates) - self.min_y + 1

    def canonical_to_actual(self, u: int, v: int) -> tuple[int, int]:
        cx = u + self.min_x - self.ox
        cy = v + self.min_y - self.oy
        return self.ia * cx + self.ib * cy, self.ic * cx + self.id * cy


def _padded_extent(extent: int) -> int:
    steps = 0 if extent <= OVERLAP else (extent - OVERLAP + STRIDE - 1) // STRIDE
    return steps * STRIDE + PATCH_SIZE


def run_wrapper(
    raw: np.ndarray,
    cfa: np.ndarray,
    run_tile: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Return normalized HWC RGB using the reviewed 288/48/240 contract."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not raw.size:
        raise XVeonReferenceError("raw input must be a nonempty 2D array")
    if not np.isfinite(raw).all():
        raise XVeonReferenceError("raw input contains non-finite values")
    height, width = raw.shape
    view = _View(find_transform(cfa), width, height)
    pad_left, pad_top = view.min_x % 6, view.min_y % 6
    aligned_width, aligned_height = view.width + pad_left, view.height + pad_top
    w_pad, h_pad = _padded_extent(aligned_width), _padded_extent(aligned_height)
    aligned = np.zeros((h_pad, w_pad), dtype=np.float32)
    for py in range(aligned_height):
        v = min(pad_top - 1 - py if py < pad_top else py - pad_top, view.height - 1)
        for px in range(aligned_width):
            u = min(pad_left - 1 - px if px < pad_left else px - pad_left, view.width - 1)
            x, y = view.canonical_to_actual(u, v)
            aligned[py, px] = raw[y, x] / np.float32(65535.0)

    masks = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    rows, columns = np.indices((PATCH_SIZE, PATCH_SIZE))
    channels = PATTERN[rows % 6, columns % 6]
    for channel in range(3):
        masks[channel] = channels == channel
    ramp = np.ones(PATCH_SIZE, dtype=np.float32)
    for index in range(OVERLAP):
        ramp[index] = np.float32(index / OVERLAP)
        ramp[PATCH_SIZE - 1 - index] = np.float32(index / OVERLAP)
    blend = np.outer(ramp, ramp).astype(np.float32)
    output = np.zeros((3, h_pad, w_pad), dtype=np.float32)
    weights = np.zeros((h_pad, w_pad), dtype=np.float32)
    for ty in range(0, h_pad - PATCH_SIZE + 1, STRIDE):
        for tx in range(0, w_pad - PATCH_SIZE + 1, STRIDE):
            tile = np.empty((1, 4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            tile[0, 0] = aligned[ty : ty + PATCH_SIZE, tx : tx + PATCH_SIZE]
            tile[0, 1:] = masks
            predicted = np.asarray(run_tile(tile), dtype=np.float32)
            if predicted.shape == (1, 3, PATCH_SIZE, PATCH_SIZE):
                predicted = predicted[0]
            if predicted.shape != (3, PATCH_SIZE, PATCH_SIZE) or not np.isfinite(predicted).all():
                raise XVeonReferenceError("tile output contract differs or contains non-finite values")
            output[:, ty : ty + PATCH_SIZE, tx : tx + PATCH_SIZE] += predicted * blend
            weights[ty : ty + PATCH_SIZE, tx : tx + PATCH_SIZE] += blend
    np.divide(output, weights[None], out=output, where=weights[None] > np.float32(1e-8))

    result = np.empty((height, width, 3), dtype=np.float32)
    for v in range(view.height):
        for u in range(view.width):
            x, y = view.canonical_to_actual(u, v)
            result[y, x] = output[:, v + pad_top, u + pad_left]
    return result


def load_session(path: Path):
    """Load only the pinned model with CPU ONNX Runtime 1.27.0."""
    import onnxruntime as ort

    if ort.__version__ != "1.27.0":
        raise XVeonReferenceError(f"onnxruntime {ort.__version__} is installed; 1.27.0 is required")
    if path.stat().st_size != MODEL_BYTES or file_sha256(path) != MODEL_SHA256:
        raise XVeonReferenceError("X-veon model identity differs")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "input" or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 4, 288, 288]:
        raise XVeonReferenceError("X-veon input contract differs")
    if len(outputs) != 1 or outputs[0].name != "output" or outputs[0].type != "tensor(float)" or outputs[0].shape != [1, 3, 288, 288]:
        raise XVeonReferenceError("X-veon output contract differs")
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("epoch") != "399" or metadata.get("base_width") != "32" or metadata.get("best_val_psnr") != "45.78":
        raise XVeonReferenceError("X-veon model metadata differs")
    return session

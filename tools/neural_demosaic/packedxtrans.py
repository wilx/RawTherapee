"""Independent PackedXTransNet checkpoint contract and fixed-tile reference."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .inspect_checkpoint import CheckpointError, ValidatedCheckpoint, load_validated_checkpoint
from .schema import PACKED_XTRANS_V1


TILE_SIZE = 288
MODEL_WIDTH = 32
MODEL_DEPTH = 8
TRAINABLE_PARAMETERS = 158_683
XTRANS_PATTERN = np.array(
    [
        [0, 2, 1, 2, 0, 1],
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
    ],
    dtype=np.int32,
)


def _tent(size: int) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.float32)
    line = 1 - (coordinates - (size - 1) / 2).abs() / ((size + 1) / 2)
    return torch.outer(line, line)[None, None]


def expected_fixed_buffers() -> dict[str, torch.Tensor]:
    masks = torch.from_numpy(
        np.stack([XTRANS_PATTERN == channel for channel in range(3)]).astype(np.float32)
    )
    return {
        "baseline.masks": masks,
        "baseline.kern_g": _tent(5),
        "baseline.kern_rb": _tent(7),
    }


def validate_fixed_buffers(validated: ValidatedCheckpoint) -> None:
    payloads = {
        spec.name: payload
        for spec, payload in zip(PACKED_XTRANS_V1.tensors, validated.tensor_payloads, strict=True)
    }
    for name, expected in expected_fixed_buffers().items():
        actual = np.frombuffer(payloads[name], dtype="<f4").reshape(tuple(expected.shape))
        expected_array = expected.numpy().astype("<f4", copy=False)
        if actual.tobytes(order="C") != expected_array.tobytes(order="C"):
            raise CheckpointError(f"checkpoint fixed buffer {name} differs from the reviewed definition")


def load_packed_checkpoint(path: str | Path) -> ValidatedCheckpoint:
    validated = load_validated_checkpoint(path, PACKED_XTRANS_V1)
    validate_fixed_buffers(validated)
    return validated


class _ChromaBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name, value in expected_fixed_buffers().items():
            self.register_buffer(name.split(".")[-1], value)

    @staticmethod
    def _normalized_convolution(value: torch.Tensor, mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padding = kernel.shape[-1] // 2
        numerator = F.conv2d(value * mask, kernel, padding=padding)
        denominator = F.conv2d(mask.expand_as(value), kernel, padding=padding)
        return numerator / denominator.clamp_min(1e-8)

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        height, width = mosaic.shape[-2:]
        masks = self.masks.repeat(1, height // 6 + 1, width // 6 + 1)[:, :height, :width, None]
        masks = masks.permute(0, 3, 1, 2)
        green = self._normalized_convolution(mosaic, masks[1], self.kern_g)
        difference = mosaic - green
        red = green + self._normalized_convolution(difference, masks[0], self.kern_rb)
        blue = green + self._normalized_convolution(difference, masks[2], self.kern_rb)
        return torch.cat((red, green, blue), dim=1)


class _ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(MODEL_WIDTH, MODEL_WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(MODEL_WIDTH, MODEL_WIDTH, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(F.relu(self.conv1(value)))


class PackedXTransReference(nn.Module):
    """Reviewed graph reconstructed without importing upstream executable code."""

    def __init__(self) -> None:
        super().__init__()
        self.baseline = _ChromaBaseline()
        self.stem = nn.Conv2d(10, MODEL_WIDTH, 3, padding=1)
        self.body = nn.Sequential(*[_ResidualBlock() for _ in range(MODEL_DEPTH)])
        self.head = nn.Conv2d(MODEL_WIDTH, 27, 3, padding=1)

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        if mosaic.ndim != 4 or mosaic.shape[1] != 1:
            raise ValueError("PackedXTransNet input must be NCHW with one channel")
        height, width = mosaic.shape[-2:]
        if height % 6 or width % 6:
            raise ValueError("PackedXTransNet dimensions must be divisible by six")
        packed = F.pixel_unshuffle(mosaic, 3)
        yy = torch.arange(height // 3, device=packed.device)
        xx = torch.arange(width // 3, device=packed.device)
        phase = ((yy[:, None] + xx[None, :]) % 2).to(packed.dtype)
        features = torch.cat((packed, phase.expand(mosaic.shape[0], 1, -1, -1)), dim=1)
        residual = F.pixel_shuffle(self.head(self.body(self.stem(features))), 3)
        return self.baseline(mosaic) + residual


def state_dict_from_checkpoint(validated: ValidatedCheckpoint) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for spec, payload in zip(PACKED_XTRANS_V1.tensors, validated.tensor_payloads, strict=True):
        array = np.frombuffer(payload, dtype="<f4").reshape(spec.shape).copy()
        result[spec.name] = torch.from_numpy(array)
    return result


def reference_from_checkpoint(path: str | Path) -> PackedXTransReference:
    validated = load_packed_checkpoint(path)
    model = PackedXTransReference()
    model.load_state_dict(state_dict_from_checkpoint(validated), strict=True)
    model.eval()
    return model


def tensor_payload_map(validated: ValidatedCheckpoint) -> Mapping[str, bytes]:
    return {
        spec.name: payload
        for spec, payload in zip(PACKED_XTRANS_V1.tensors, validated.tensor_payloads, strict=True)
    }


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

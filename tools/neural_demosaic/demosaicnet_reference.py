"""Local PyTorch reference for the pinned Gharbi X-Trans network graph."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .inspect_checkpoint import load_validated_checkpoint
from .rtnn_reader import LoadedRTNN, read_rtnn
from .semantic_schema import (
    SemanticSchema,
    load_semantic_schema,
    validate_inspection_manifest,
)


CANONICAL_XTRANS = (
    (1, 2, 1, 1, 0, 1),
    (0, 1, 0, 2, 1, 2),
    (1, 2, 1, 1, 0, 1),
    (1, 0, 1, 1, 2, 1),
    (2, 1, 2, 0, 1, 0),
    (1, 0, 1, 1, 2, 1),
)


class DemosaicNetXTransReference(nn.Module):
    """Exact local definition of upstream XTransDemosaick(11, 64, pad=False)."""

    def __init__(self) -> None:
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        for index in range(1, 12):
            input_channels = 3 if index == 1 else 64
            layers[f"conv{index}"] = nn.Conv2d(input_channels, 64, 3, padding=0)
            layers[f"relu{index}"] = nn.ReLU(inplace=True)

        self.main_processor = nn.Sequential(layers)
        self.fullres_processor = nn.Sequential(
            OrderedDict(
                (
                    ("post_conv", nn.Conv2d(67, 64, 3, padding=0)),
                    ("post_relu", nn.ReLU(inplace=True)),
                    ("output", nn.Conv2d(64, 3, 1)),
                )
            )
        )

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        features = self.main_processor(mosaic)
        cropped = _crop_like(mosaic, features)
        packed = torch.cat((cropped, features), dim=1)
        return self.fullres_processor(packed)


@dataclass(frozen=True)
class ReferenceActivation:
    """One named execution checkpoint from the fixed reference graph."""

    name: str
    tensor: torch.Tensor


def reference_forward_with_activations(
    model: DemosaicNetXTransReference,
    mosaic: torch.Tensor,
) -> tuple[torch.Tensor, tuple[ReferenceActivation, ...]]:
    """Execute the reference graph while exposing stable diagnostic points."""

    activations = []
    features = mosaic

    for index in range(1, 12):
        convolution = model.main_processor.get_submodule(f"conv{index}")
        activation = model.main_processor.get_submodule(f"relu{index}")
        features = activation(convolution(features))
        activations.append(
            ReferenceActivation(f"main_processor.relu{index}", features)
        )

    cropped = _crop_like(mosaic, features)
    packed = torch.cat((cropped, features), dim=1)
    activations.append(ReferenceActivation("fullres_processor.input_concat", packed))

    post_convolution = model.fullres_processor.get_submodule("post_conv")
    post_activation = model.fullres_processor.get_submodule("post_relu")
    post = post_activation(post_convolution(packed))
    activations.append(ReferenceActivation("fullres_processor.post_relu", post))

    output_layer = model.fullres_processor.get_submodule("output")
    return output_layer(post), tuple(activations)


def _crop_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    crop_height = source.shape[-2] - target.shape[-2]
    crop_width = source.shape[-1] - target.shape[-1]

    if crop_height < 0 or crop_width < 0:
        raise ValueError("target is larger than source")

    top = crop_height // 2
    bottom = crop_height - top
    left = crop_width // 2
    right = crop_width - left
    height_end = source.shape[-2] - bottom
    width_end = source.shape[-1] - right
    return source[..., top:height_end, left:width_end]


def _tensor_from_bytes(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    array = np.frombuffer(payload, dtype="<f4").copy()
    tensor = torch.from_numpy(array)
    return tensor.reshape(shape)


def _state_dict_from_payloads(
    schema: SemanticSchema,
    payloads: Sequence[bytes],
) -> OrderedDict[str, torch.Tensor]:
    if len(payloads) != len(schema.tensors):
        raise ValueError(
            f"received {len(payloads)} payloads; expected {len(schema.tensors)}"
        )

    state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for spec, payload in zip(schema.tensors, payloads, strict=True):
        expected_bytes = spec.element_count * 4

        if len(payload) != expected_bytes:
            raise ValueError(
                f"tensor {spec.id} has {len(payload)} bytes; expected {expected_bytes}"
            )

        state[spec.source_name] = _tensor_from_bytes(payload, spec.shape)

    return state


def _model_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> DemosaicNetXTransReference:
    model = DemosaicNetXTransReference()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def reference_from_checkpoint(path: str | Path) -> DemosaicNetXTransReference:
    """Build the reference graph from Phase 1 authenticated checkpoint bytes."""

    schema = load_semantic_schema()
    checkpoint = load_validated_checkpoint(path)
    validate_inspection_manifest(schema, checkpoint.manifest)
    return _model_from_state_dict(
        _state_dict_from_payloads(schema, checkpoint.tensor_payloads)
    )


def reference_from_loaded_rtnn(model: LoadedRTNN) -> DemosaicNetXTransReference:
    """Build the reference graph from a fully authenticated RTNN result."""

    schema = load_semantic_schema()
    payloads = tuple(model.tensor(spec.id).data for spec in schema.tensors)
    return _model_from_state_dict(_state_dict_from_payloads(schema, payloads))


def reference_from_rtnn(path: str | Path) -> DemosaicNetXTransReference:
    """Read RTNN and construct the reference graph without loading a checkpoint."""

    return reference_from_loaded_rtnn(read_rtnn(path))


def _xtrans_mask(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros((3, height, width), dtype=torch.float32)

    for row in range(height):
        for column in range(width):
            channel = CANONICAL_XTRANS[row % 6][column % 6]
            mask[channel, row, column] = 1.0

    return mask


def _sparse_from_scalar(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("sparse source values must be a two-dimensional tensor")

    height, width = values.shape
    return (_xtrans_mask(height, width) * values.unsqueeze(0)).unsqueeze(0)


def reference_inputs() -> tuple[tuple[str, torch.Tensor], ...]:
    """Return the fixed Phase 4 sparse inputs without exporting binary fixtures."""

    inputs = []
    inputs.append(("zero-even-32x32", torch.zeros((1, 3, 32, 32))))

    constant_mask = _xtrans_mask(31, 35)
    channel_values = torch.tensor((0.25, 0.5, 0.75), dtype=torch.float32)
    constant = constant_mask * channel_values[:, None, None]
    inputs.append(("constant-odd-31x35", constant.unsqueeze(0)))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x52544E4E)
    random_values = torch.rand((37, 38), generator=generator, dtype=torch.float32)
    inputs.append(("random-37x38", _sparse_from_scalar(random_values)))

    impulse_sites = {
        "red": (0, 18, 22),
        "green": (1, 18, 18),
        "blue": (2, 18, 19),
    }

    for name, (channel, row, column) in impulse_sites.items():
        impulse = torch.zeros((1, 3, 36, 36), dtype=torch.float32)

        if CANONICAL_XTRANS[row % 6][column % 6] != channel:
            raise AssertionError(f"{name} impulse is not on its CFA colour")

        impulse[0, channel, row, column] = 1.0
        inputs.append((f"impulse-{name}-36x36", impulse))

    rows = torch.arange(35, dtype=torch.int64)[:, None]
    columns = torch.arange(36, dtype=torch.int64)[None, :]
    saturated_values = ((rows + columns) % 2).to(torch.float32)
    inputs.append(("alternating-saturated-35x36", _sparse_from_scalar(saturated_values)))
    return tuple(inputs)


@contextmanager
def deterministic_reference_execution() -> Iterator[None]:
    """Temporarily force the deterministic CPU settings used by parity tests."""

    previous_threads = torch.get_num_threads()
    previous_mkldnn = torch.backends.mkldnn.enabled
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.set_num_threads(previous_threads)

"""Pinned tensor schema for the published Gharbi X-Trans checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


INSPECTION_MANIFEST_FORMAT = "rawtherapee-neural-checkpoint-inspection-v1"
GHARBI_XTRANS_V1_MANIFEST_SHA256 = (
    "371a3e20bac66877238e44d36e349078953c0b6c4e256299f64d66bbd8b72848"
)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    layout: str

    @property
    def element_count(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class CheckpointSchema:
    model_id: str
    architecture_name: str
    architecture_depth: int
    architecture_width: int
    convolution_padding: int
    upstream_repository: str
    upstream_revision: str
    upstream_checkpoint_path: str
    upstream_license: str
    expected_file_size: int
    expected_sha256: str
    tensors: tuple[TensorSpec, ...]

    @property
    def parameter_count(self) -> int:
        return sum(tensor.element_count for tensor in self.tensors)

    @property
    def payload_bytes(self) -> int:
        return self.parameter_count * 4


def _main_processor_specs() -> tuple[TensorSpec, ...]:
    result: list[TensorSpec] = []

    for index in range(1, 12):
        input_channels = 3 if index == 1 else 64
        prefix = f"main_processor.conv{index}"
        result.extend(
            (
                TensorSpec(f"{prefix}.weight", (64, input_channels, 3, 3), "OIHW"),
                TensorSpec(f"{prefix}.bias", (64,), "vector"),
            )
        )

    return tuple(result)


GHARBI_XTRANS_V1 = CheckpointSchema(
    model_id="demosaicnet-xtrans-v1",
    architecture_name="XTransDemosaick",
    architecture_depth=11,
    architecture_width=64,
    convolution_padding=0,
    upstream_repository="https://github.com/mgharbi/demosaicnet",
    upstream_revision="959e9d1630976b421d5af5e35b2e2a01f5630e5c",
    upstream_checkpoint_path="demosaicnet/data/xtrans.pth",
    upstream_license="MIT",
    expected_file_size=1_644_547,
    expected_sha256="3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7",
    tensors=_main_processor_specs()
    + (
        TensorSpec("fullres_processor.post_conv.weight", (64, 67, 3, 3), "OIHW"),
        TensorSpec("fullres_processor.post_conv.bias", (64,), "vector"),
        TensorSpec("fullres_processor.output.weight", (3, 64, 1, 1), "OIHW"),
        TensorSpec("fullres_processor.output.bias", (3,), "vector"),
    ),
)


assert len(GHARBI_XTRANS_V1.tensors) == 26
assert GHARBI_XTRANS_V1.parameter_count == 409_923
assert GHARBI_XTRANS_V1.payload_bytes == 1_639_692

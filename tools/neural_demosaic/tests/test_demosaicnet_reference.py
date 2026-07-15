from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as functional

from tools.neural_demosaic.convert_checkpoint import convert_checkpoint
from tools.neural_demosaic.demosaicnet_reference import (
    CANONICAL_XTRANS,
    DemosaicNetXTransReference,
    deterministic_reference_execution,
    reference_from_checkpoint,
    reference_from_rtnn,
    reference_forward_with_activations,
    reference_inputs,
)
from tools.neural_demosaic.inspect_checkpoint import load_validated_checkpoint
from tools.neural_demosaic.rtnn_reader import parse_rtnn
from tools.neural_demosaic.semantic_schema import load_semantic_schema


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )


def test_reference_graph_has_the_exact_semantic_state_dict() -> None:
    model = DemosaicNetXTransReference()
    schema = load_semantic_schema()
    state = model.state_dict()

    assert list(state) == [tensor.source_name for tensor in schema.tensors]

    for spec in schema.tensors:
        assert tuple(state[spec.source_name].shape) == spec.shape
        assert state[spec.source_name].dtype is torch.float32


def test_reference_inputs_are_fixed_sparse_xtrans_planes() -> None:
    cases = reference_inputs()
    assert [(name, tuple(value.shape)) for name, value in cases] == [
        ("zero-even-32x32", (1, 3, 32, 32)),
        ("constant-odd-31x35", (1, 3, 31, 35)),
        ("random-37x38", (1, 3, 37, 38)),
        ("impulse-red-36x36", (1, 3, 36, 36)),
        ("impulse-green-36x36", (1, 3, 36, 36)),
        ("impulse-blue-36x36", (1, 3, 36, 36)),
        ("alternating-saturated-35x36", (1, 3, 35, 36)),
    ]

    for _, value in cases:
        assert value.dtype is torch.float32
        assert value.device.type == "cpu"
        height, width = value.shape[-2:]
        mask = torch.zeros_like(value)

        for row in range(height):
            for column in range(width):
                channel = CANONICAL_XTRANS[row % 6][column % 6]
                mask[0, channel, row, column] = 1.0

        assert torch.count_nonzero(value * (1.0 - mask)) == 0

    first_random = dict(reference_inputs())["random-37x38"]
    second_random = dict(reference_inputs())["random-37x38"]
    assert torch.equal(first_random, second_random)


def test_reference_graph_output_shrinks_by_24_and_is_finite() -> None:
    model = DemosaicNetXTransReference().eval()
    input_tensor = dict(reference_inputs())["constant-odd-31x35"]

    with deterministic_reference_execution(), torch.inference_mode():
        output = model(input_tensor)

    assert output.shape == (1, 3, 7, 11)
    assert bool(torch.isfinite(output).all())


def test_activation_trace_matches_ordinary_reference_execution() -> None:
    model = DemosaicNetXTransReference().eval()
    input_tensor = dict(reference_inputs())["random-37x38"]

    with deterministic_reference_execution(), torch.inference_mode():
        traced, activations = reference_forward_with_activations(model, input_tensor)
        ordinary = model(input_tensor)

    assert torch.equal(traced, ordinary)
    assert [activation.name for activation in activations] == [
        *(f"main_processor.relu{index}" for index in range(1, 12)),
        "fullres_processor.input_concat",
        "fullres_processor.post_relu",
    ]
    assert [tuple(activation.tensor.shape) for activation in activations][-3:] == [
        (1, 64, 15, 16),
        (1, 67, 15, 16),
        (1, 64, 13, 14),
    ]


def test_reference_forward_matches_an_independent_functional_graph() -> None:
    model = DemosaicNetXTransReference().eval()
    input_tensor = dict(reference_inputs())["constant-odd-31x35"]
    state = model.state_dict()

    with deterministic_reference_execution(), torch.inference_mode():
        expected = input_tensor

        for index in range(1, 12):
            expected = functional.relu(
                functional.conv2d(
                    expected,
                    state[f"main_processor.conv{index}.weight"],
                    state[f"main_processor.conv{index}.bias"],
                )
            )

        cropped = input_tensor[..., 11:-11, 11:-11]
        expected = functional.relu(
            functional.conv2d(
                torch.cat((cropped, expected), dim=1),
                state["fullres_processor.post_conv.weight"],
                state["fullres_processor.post_conv.bias"],
            )
        )
        expected = functional.conv2d(
            expected,
            state["fullres_processor.output.weight"],
            state["fullres_processor.output.bias"],
        )
        actual = model(input_tensor)

    assert torch.equal(actual, expected)


def test_deterministic_context_restores_global_cpu_settings() -> None:
    previous_threads = torch.get_num_threads()
    previous_mkldnn = torch.backends.mkldnn.enabled
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    with deterministic_reference_execution():
        assert torch.get_num_threads() == 1
        assert not torch.backends.mkldnn.enabled
        assert torch.are_deterministic_algorithms_enabled()

    assert torch.get_num_threads() == previous_threads
    assert torch.backends.mkldnn.enabled == previous_mkldnn
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == previous_warn_only


def test_checkpoint_and_rtnn_reference_are_bit_identical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = os.environ.get("GHARBI_XTRANS_CHECKPOINT")

    if not checkpoint_path:
        pytest.skip("GHARBI_XTRANS_CHECKPOINT is not set")

    conversion = convert_checkpoint(checkpoint_path)
    rtnn_path = tmp_path / "model.rtnn"
    rtnn_path.write_bytes(conversion.artifact.data)
    checkpoint_model = reference_from_checkpoint(checkpoint_path)
    validated = load_validated_checkpoint(checkpoint_path)
    loaded_rtnn = parse_rtnn(conversion.artifact.data)
    schema = load_semantic_schema()
    imported_upstream = False
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        nonlocal imported_upstream

        if name == "demosaicnet" or name.startswith("demosaicnet."):
            imported_upstream = True
            raise AssertionError("upstream executable model code must not be imported")

        return original_import(name, *args, **kwargs)

    def forbidden_load(*args, **kwargs):
        pytest.fail("RTNN-backed construction must not call torch.load")

    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setattr(torch, "load", forbidden_load)
    rtnn_model = reference_from_rtnn(rtnn_path)
    assert not imported_upstream
    assert not any(
        name == "demosaicnet" or name.startswith("demosaicnet.")
        for name in sys.modules
    )

    checkpoint_state = checkpoint_model.state_dict()
    rtnn_state = rtnn_model.state_dict()

    for spec, checkpoint_payload in zip(
        schema.tensors,
        validated.tensor_payloads,
        strict=True,
    ):
        assert checkpoint_payload == loaded_rtnn.tensor(spec.id).data
        assert tensor_bytes(checkpoint_state[spec.source_name]) == checkpoint_payload
        assert tensor_bytes(rtnn_state[spec.source_name]) == checkpoint_payload

    with deterministic_reference_execution(), torch.inference_mode():
        for name, input_tensor in reference_inputs():
            checkpoint_output = checkpoint_model(input_tensor)
            rtnn_output = rtnn_model(input_tensor)
            assert checkpoint_output.shape[-2:] == (
                input_tensor.shape[-2] - 24,
                input_tensor.shape[-1] - 24,
            ), name
            assert bool(torch.isfinite(checkpoint_output).all()), name
            assert bool(torch.isfinite(rtnn_output).all()), name
            assert torch.equal(checkpoint_output, rtnn_output), name

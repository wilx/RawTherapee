#!/usr/bin/env python3
"""Convert the authenticated PackedXTransNet checkpoint to canonical ONNX."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .inspect_checkpoint import CheckpointError, canonical_manifest_bytes
from .packedxtrans import TILE_SIZE, XTRANS_PATTERN, load_packed_checkpoint, tensor_payload_map
from .schema import PACKED_XTRANS_V1


FORMAT = "rawtherapee-packedxtrans-conversion-manifest-v1"
OPSET = 18
IR_VERSION = 10


def _array(payloads: dict[str, bytes], name: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(payloads[name], dtype="<f4").reshape(shape).copy()


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype="<f4", order="C"), name=name)


def build_model(checkpoint: str | Path) -> tuple[bytes, dict[str, Any]]:
    validated = load_packed_checkpoint(checkpoint)
    payloads = dict(tensor_payload_map(validated))
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    masks6 = _array(payloads, "baseline.masks", (3, 6, 6))
    masks = np.tile(masks6, (1, TILE_SIZE // 6, TILE_SIZE // 6))[:, None]
    initializers.extend(
        (
            _initializer("mask_r", masks[0:1]),
            _initializer("mask_g", masks[1:2]),
            _initializer("mask_b", masks[2:3]),
            _initializer("kernel_g", _array(payloads, "baseline.kern_g", (1, 1, 5, 5))),
            _initializer("kernel_rb", _array(payloads, "baseline.kern_rb", (1, 1, 7, 7))),
            _initializer("epsilon", np.array(1e-8, dtype=np.float32)),
        )
    )

    def node(operation: str, inputs: list[str], outputs: list[str], name: str, **attributes: Any) -> None:
        nodes.append(helper.make_node(operation, inputs, outputs, name=name, **attributes))

    node("Mul", ["input", "mask_g"], ["green_masked"], "baseline.green.mask")
    node("Conv", ["green_masked", "kernel_g"], ["green_num"], "baseline.green.numerator", pads=[2, 2, 2, 2])
    node("Conv", ["mask_g", "kernel_g"], ["green_den_raw"], "baseline.green.denominator", pads=[2, 2, 2, 2])
    node("Max", ["green_den_raw", "epsilon"], ["green_den"], "baseline.green.floor")
    node("Div", ["green_num", "green_den"], ["green"], "baseline.green.divide")
    node("Sub", ["input", "green"], ["chroma_source"], "baseline.chroma.source")
    colors: list[str] = []
    for color, mask in (("red", "mask_r"), ("blue", "mask_b")):
        node("Mul", ["chroma_source", mask], [f"{color}_masked"], f"baseline.{color}.mask")
        node("Conv", [f"{color}_masked", "kernel_rb"], [f"{color}_num"], f"baseline.{color}.numerator", pads=[3, 3, 3, 3])
        node("Conv", [mask, "kernel_rb"], [f"{color}_den_raw"], f"baseline.{color}.denominator", pads=[3, 3, 3, 3])
        node("Max", [f"{color}_den_raw", "epsilon"], [f"{color}_den"], f"baseline.{color}.floor")
        node("Div", [f"{color}_num", f"{color}_den"], [f"{color}_difference"], f"baseline.{color}.divide")
        node("Add", ["green", f"{color}_difference"], [color], f"baseline.{color}.add")
        colors.append(color)
    node("Concat", [colors[0], "green", colors[1]], ["baseline"], "baseline.concat", axis=1)

    phase_size = TILE_SIZE // 3
    yy, xx = np.meshgrid(np.arange(phase_size), np.arange(phase_size), indexing="ij")
    phase = ((yy + xx) % 2).astype(np.float32)[None, None]
    initializers.append(_initializer("phase", phase))
    node("SpaceToDepth", ["input"], ["packed"], "network.pack", blocksize=3)
    node("Concat", ["packed", "phase"], ["features_0"], "network.phase", axis=1)

    def convolution(prefix: str, source: str, destination: str, node_name: str) -> None:
        weight_name = f"{prefix}.weight"
        bias_name = f"{prefix}.bias"
        spec_by_name = {spec.name: spec for spec in PACKED_XTRANS_V1.tensors}
        initializers.append(_initializer(weight_name, _array(payloads, weight_name, spec_by_name[weight_name].shape)))
        initializers.append(_initializer(bias_name, _array(payloads, bias_name, spec_by_name[bias_name].shape)))
        node("Conv", [source, weight_name, bias_name], [destination], node_name, pads=[1, 1, 1, 1])

    convolution("stem", "features_0", "features_1", "network.stem")
    source = "features_1"
    for index in range(8):
        convolution(f"body.{index}.conv1", source, f"body_{index}_conv1", f"network.body.{index}.conv1")
        node("Relu", [f"body_{index}_conv1"], [f"body_{index}_relu"], f"network.body.{index}.relu")
        convolution(f"body.{index}.conv2", f"body_{index}_relu", f"body_{index}_conv2", f"network.body.{index}.conv2")
        destination = f"body_{index}_out"
        node("Add", [source, f"body_{index}_conv2"], [destination], f"network.body.{index}.residual")
        source = destination
    convolution("head", source, "packed_residual", "network.head")
    node("DepthToSpace", ["packed_residual"], ["residual"], "network.unpack", blocksize=3, mode="CRD")
    node("Add", ["baseline", "residual"], ["output"], "network.output")

    graph = helper.make_graph(
        nodes,
        "packedxtransnet-xtrans-v1",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, TILE_SIZE, TILE_SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, TILE_SIZE, TILE_SIZE])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="RawTherapee PackedXTransNet converter",
        producer_version="1",
        domain="org.rawtherapee.development",
        model_version=1,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    metadata = {
        "architecture": "PackedXTransNet-width32-depth8",
        "checkpoint_sha256": PACKED_XTRANS_V1.expected_sha256,
        "license": PACKED_XTRANS_V1.upstream_license,
        "model_id": PACKED_XTRANS_V1.model_id,
        "upstream_revision": PACKED_XTRANS_V1.upstream_revision,
    }
    for key in sorted(metadata):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = metadata[key]
    onnx.checker.check_model(model, full_check=True)
    model_bytes = model.SerializeToString(deterministic=True)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    tensors = []
    for spec, entry in zip(PACKED_XTRANS_V1.tensors, validated.manifest["tensors"], strict=True):
        tensors.append(
            {
                "element_count": spec.element_count,
                "layout": spec.layout,
                "name": spec.name,
                "sha256": entry["sha256"],
                "shape": list(spec.shape),
            }
        )
    manifest = {
        "format": FORMAT,
        "graph": {
            "input": {"dtype": "float32", "name": "input", "shape": [1, 1, TILE_SIZE, TILE_SIZE]},
            "ir_version": IR_VERSION,
            "opset": OPSET,
            "output": {"dtype": "float32", "name": "output", "shape": [1, 3, TILE_SIZE, TILE_SIZE]},
        },
        "model": {
            "id": PACKED_XTRANS_V1.model_id,
            "license": PACKED_XTRANS_V1.upstream_license,
            "upstream_repository": PACKED_XTRANS_V1.upstream_repository,
            "upstream_revision": PACKED_XTRANS_V1.upstream_revision,
        },
        "onnx": {"sha256": model_sha, "size_bytes": len(model_bytes)},
        "source": {
            "path": PACKED_XTRANS_V1.upstream_checkpoint_path,
            "sha256": PACKED_XTRANS_V1.expected_sha256,
            "size_bytes": PACKED_XTRANS_V1.expected_file_size,
        },
        "summary": {
            "state_value_count": PACKED_XTRANS_V1.parameter_count,
            "tensor_count": len(PACKED_XTRANS_V1.tensors),
            "trainable_parameter_count": 158_683,
        },
        "tensors": tensors,
    }
    return model_bytes, manifest


def _temporary(path: Path, payload: bytes) -> str:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.unlink(name)
        raise
    return name


def convert(checkpoint: str | Path, output: str | Path, *, force: bool = False) -> dict[str, Any]:
    output_path = Path(output)
    manifest_path = Path(str(output_path) + ".json")
    if not output_path.parent.is_dir():
        raise CheckpointError(f"output directory does not exist: {output_path.parent}")
    if not force and (output_path.exists() or manifest_path.exists()):
        raise CheckpointError("output ONNX or companion manifest already exists")
    model_bytes, manifest = build_model(checkpoint)
    manifest_bytes = canonical_manifest_bytes(manifest)
    model_temporary = _temporary(output_path, model_bytes)
    manifest_temporary = _temporary(manifest_path, manifest_bytes)
    try:
        if not force and (output_path.exists() or manifest_path.exists()):
            raise CheckpointError("output ONNX or companion manifest already exists")
        os.replace(manifest_temporary, manifest_path)
        manifest_temporary = ""
        os.replace(model_temporary, output_path)
        model_temporary = ""
    finally:
        for name in (model_temporary, manifest_temporary):
            if name:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass
    return manifest


def _arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _arguments(arguments)
    try:
        manifest = convert(options.checkpoint, options.output, force=options.force)
    except CheckpointError as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest["onnx"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

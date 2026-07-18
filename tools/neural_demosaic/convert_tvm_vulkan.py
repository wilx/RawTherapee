#!/usr/bin/env python3
"""Compile authenticated X-Trans ONNX models to portable TVM Vulkan modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import onnx
from onnx import numpy_helper, TensorProto

FORMAT = "rawtherapee-tvm-vulkan-conversion-manifest-v1"
TVM_VERSION = "0.25.0"
TVM_COMMIT = "c7ba0735a4f346c67b761e1fde38a68a60be8adb"
TVM_ARCHIVE = {
    "name": "apache-tvm-src-v0.25.0.tar.gz",
    "sha256": "ea7c3248e2a8ca91969fa487247ed94db22f1dbfadf7b9408fede76c8f16e56d",
    "size_bytes": 80_764_445,
}
TVM_FFI_COMMIT = "59da4c0b82af0d499dae34bd89ef010f64d3ff45"

VULKAN_1_1 = (1 << 22) | (1 << 12)
SPIRV_1_3 = 0x00010300
PORTABLE_TARGET: dict[str, int | str] = {
    "kind": "vulkan",
    "max_block_size_x": 128,
    "max_block_size_y": 128,
    "max_block_size_z": 64,
    "max_num_threads": 128,
    "max_per_stage_descriptor_storage_buffer": 4,
    "max_push_constants_size": 128,
    "max_shared_memory_per_block": 16_384,
    "max_spirv_version": SPIRV_1_3,
    "max_storage_buffer_range": 128 * 1024 * 1024,
    "max_threads_per_block": 128,
    "max_uniform_buffer_range": 16_384,
    "supported_subgroup_operations": 0,
    "supports_16bit_buffer": 0,
    "supports_8bit_buffer": 0,
    "supports_cooperative_matrix": 0,
    "supports_dedicated_allocation": 0,
    "supports_float16": 0,
    "supports_float32": 1,
    "supports_float64": 0,
    "supports_int16": 0,
    "supports_int32": 1,
    "supports_int64": 0,
    "supports_int8": 0,
    "supports_integer_dot_product": 0,
    "supports_push_descriptor": 0,
    "supports_storage_buffer_storage_class": 1,
    "thread_warp_size": 1,
    "vulkan_api_version": VULKAN_1_1,
}
PORTABLE_HOST: dict[str, str] = {
    "kind": "llvm",
    "mcpu": "x86-64",
    "mtriple": "x86_64-linux-gnu",
}


class TVMConversionError(RuntimeError):
    """A stable, user-facing conversion failure."""


def canonical_manifest_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    size_bytes: int
    sha256: str
    license: str
    upstream_revision: str
    opset: int
    operators: tuple[tuple[str, int], ...]


MODEL_SPECS = {
    "xveon": ModelSpec(
        key="xveon",
        model_id="xveon-xtrans-v1",
        input_shape=(1, 4, 288, 288),
        output_shape=(1, 3, 288, 288),
        size_bytes=15_536_134,
        sha256="45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500",
        license="NOASSERTION",
        upstream_revision="2e6b96c63559aa3909b0c7c1bc45dfd4b5dfe680",
        opset=17,
        operators=(("Add", 1), ("Cast", 3), ("Concat", 4), ("Conv", 19),
                   ("ConvTranspose", 4), ("Expand", 1), ("MaxPool", 4),
                   ("Relu", 18), ("Slice", 1)),
    ),
    "packedxtransnet": ModelSpec(
        key="packedxtransnet",
        model_id="packedxtransnet-xtrans-v1",
        input_shape=(1, 1, 288, 288),
        output_shape=(1, 3, 288, 288),
        size_bytes=1_673_648,
        sha256="ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c",
        license="CC-BY-NC-4.0",
        upstream_revision="9c3cc5ab841c9afd2ed0bb702468950481043d06",
        opset=18,
        operators=(("Add", 11), ("Concat", 2), ("Conv", 24),
                   ("DepthToSpace", 1), ("Div", 3), ("Max", 3), ("Mul", 3),
                   ("Relu", 8), ("SpaceToDepth", 1), ("Sub", 1)),
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_contract(value: onnx.ValueInfoProto) -> tuple[str, tuple[int, ...], int]:
    tensor = value.type.tensor_type
    dimensions: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise TVMConversionError(f"{value.name} must have a fixed positive shape")
        dimensions.append(dimension.dim_value)
    return value.name, tuple(dimensions), tensor.elem_type


def authenticate_onnx(path: str | Path, spec: ModelSpec) -> onnx.ModelProto:
    """Authenticate and validate a fixed ONNX graph before TVM sees it."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise TVMConversionError(f"cannot stat ONNX input: {error}") from error
    if size != spec.size_bytes:
        raise TVMConversionError(
            f"{spec.key} ONNX size differs: expected {spec.size_bytes}, found {size}"
        )
    digest = file_sha256(source)
    if digest != spec.sha256:
        raise TVMConversionError(
            f"{spec.key} ONNX SHA-256 differs: expected {spec.sha256}, found {digest}"
        )
    try:
        model = onnx.load(source, load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:  # ONNX exposes several exception classes.
        raise TVMConversionError(f"invalid authenticated ONNX graph: {error}") from error
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise TVMConversionError("ONNX graph must have exactly one input and one output")
    input_contract = _tensor_contract(model.graph.input[0])
    output_contract = _tensor_contract(model.graph.output[0])
    if input_contract != ("input", spec.input_shape, TensorProto.FLOAT):
        raise TVMConversionError(f"unexpected ONNX input contract: {input_contract}")
    if output_contract != ("output", spec.output_shape, TensorProto.FLOAT):
        raise TVMConversionError(f"unexpected ONNX output contract: {output_contract}")
    if any(initializer.data_location == TensorProto.EXTERNAL for initializer in model.graph.initializer):
        raise TVMConversionError("external ONNX tensor data is forbidden")
    opsets = [(entry.domain, entry.version) for entry in model.opset_import]
    if opsets != [("", spec.opset)]:
        raise TVMConversionError(f"unexpected ONNX opset contract: {opsets}")
    operators: dict[str, int] = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    if tuple(sorted(operators.items())) != spec.operators:
        raise TVMConversionError(
            "unexpected ONNX operator contract: "
            + ", ".join(f"{name}={count}" for name, count in sorted(operators.items()))
        )
    return model


def promote_float16_graph_to_float32(model: onnx.ModelProto) -> int:
    """Promote authenticated FP16 storage and casts for the portable profile.

    X-veon's published ONNX graph deliberately performs its hidden layers in
    float16.  Phase 13's cross-vendor Vulkan artifact instead has a strict
    float32/int32 contract.  Promotion happens only after complete source-file
    authentication and is recorded in the companion manifest.
    """
    promoted = 0
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT16:
            continue
        values = numpy_helper.to_array(initializer).astype("float32")
        replacement = numpy_helper.from_array(values, initializer.name)
        model.graph.initializer[index].CopyFrom(replacement)
        promoted += 1
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                attribute.i = TensorProto.FLOAT
                promoted += 1
    for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if tensor.elem_type == TensorProto.FLOAT16:
            tensor.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model, full_check=True)
    return promoted


def portable_target_manifest() -> dict[str, Any]:
    return {
        "host": dict(PORTABLE_HOST),
        "profile": "linux-x86_64-vulkan-1.1-fp32-portable-v1",
        "target": dict(PORTABLE_TARGET),
    }


def _walk_modules(module: Any) -> list[Any]:
    result = [module]
    for imported in getattr(module, "imports", []):
        result.extend(_walk_modules(imported))
    return result


def _spirv_records(executable: Any, workspace: Path) -> list[dict[str, Any]]:
    """Validate and identify each SPIR-V kernel independently of the module."""
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise TVMConversionError("spirv-as and spirv-val are required")
    records: list[dict[str, Any]] = []
    for module in _walk_modules(executable.mod):
        if getattr(module, "kind", "") != "vulkan":
            continue
        source = str(module.inspect_source("spv"))
        kernels = ["; SPIR-V" + part for part in source.split("; SPIR-V")[1:]]
        for index, kernel in enumerate(kernels):
            capabilities = sorted(
                line.strip() for line in kernel.splitlines() if "OpCapability" in line
            )
            forbidden = [
                capability
                for capability in capabilities
                if not capability.endswith(" Shader")
            ]
            if forbidden:
                raise TVMConversionError(
                    "portable SPIR-V uses forbidden capabilities: " + ", ".join(forbidden)
                )
            entry_points = re.findall(r'OpEntryPoint\s+GLCompute\s+%\S+\s+"([^"]+)"', kernel)
            local_sizes = re.findall(
                r"OpExecutionMode\s+%\S+\s+LocalSize\s+(\d+)\s+(\d+)\s+(\d+)",
                kernel,
            )
            if len(entry_points) != 1 or len(local_sizes) != 1:
                raise TVMConversionError("each SPIR-V module must contain one compute entry point")
            workgroup = tuple(int(value) for value in local_sizes[0])
            if (
                workgroup[0] > int(PORTABLE_TARGET["max_block_size_x"])
                or workgroup[1] > int(PORTABLE_TARGET["max_block_size_y"])
                or workgroup[2] > int(PORTABLE_TARGET["max_block_size_z"])
                or workgroup[0] * workgroup[1] * workgroup[2]
                > int(PORTABLE_TARGET["max_num_threads"])
            ):
                raise TVMConversionError(f"SPIR-V workgroup exceeds portable limits: {workgroup}")
            forbidden_types = re.findall(r"OpType(?:Int\s+(?:8|16|64)|Float\s+(?:16|64))\b", kernel)
            if forbidden_types:
                raise TVMConversionError(
                    "portable SPIR-V uses forbidden scalar types: "
                    + ", ".join(sorted(set(forbidden_types)))
                )
            bindings = {
                int(value)
                for value in re.findall(r"OpDecorate\s+%\S+\s+Binding\s+(\d+)", kernel)
            }
            if len(bindings) > int(PORTABLE_TARGET["max_per_stage_descriptor_storage_buffer"]):
                raise TVMConversionError(
                    f"SPIR-V kernel {entry_points[0]} uses {len(bindings)} storage bindings, "
                    "portable limit is 4"
                )
            assembly = workspace / f"kernel-{len(records):04d}.spvasm"
            binary = workspace / f"kernel-{len(records):04d}.spv"
            assembly.write_text(kernel, encoding="utf-8")
            assembled = subprocess.run(
                [spirv_as, "--target-env", "spv1.3", str(assembly), "-o", str(binary)],
                check=False,
                capture_output=True,
                text=True,
            )
            if assembled.returncode:
                raise TVMConversionError(f"spirv-as rejected {entry_points[0]}: {assembled.stderr.strip()}")
            validated = subprocess.run(
                [spirv_val, "--target-env", "vulkan1.1", str(binary)],
                check=False,
                capture_output=True,
                text=True,
            )
            if validated.returncode:
                raise TVMConversionError(f"spirv-val rejected {entry_points[0]}: {validated.stderr.strip()}")
            records.append(
                {
                    "binary_sha256": file_sha256(binary),
                    "binary_size_bytes": binary.stat().st_size,
                    "capabilities": capabilities,
                    "entry_point": entry_points[0],
                    "index": index,
                    "storage_bindings": sorted(bindings),
                    "workgroup_size": list(workgroup),
                }
            )
    if not records:
        raise TVMConversionError("TVM executable contains no Vulkan module")
    return records


def _load_tvm(tvm_source: Path, tvm_build: Path) -> Any:
    if not (tvm_source / "python" / "tvm").is_dir():
        raise TVMConversionError("TVM source directory does not contain python/tvm")
    library_dir = tvm_build / "lib" if (tvm_build / "lib").is_dir() else tvm_build
    compiler = library_dir / "libtvm_compiler.so"
    runtime = library_dir / "libtvm_runtime.so"
    if not compiler.is_file() or not runtime.is_file():
        raise TVMConversionError("TVM build directory lacks compiler/runtime libraries")
    sys.path.insert(0, str(tvm_source / "python"))
    os.environ["TVM_LIBRARY_PATH"] = str(library_dir)
    try:
        import tvm  # pylint: disable=import-outside-toplevel
    except Exception as error:
        raise TVMConversionError(f"cannot import pinned TVM compiler: {error}") from error
    # The official source archive lacks Git metadata, so setuptools-scm leaves
    # the Python package at its release fallback even though the C++ libraries
    # are explicitly built with TVM_VERSION=0.25.0.
    if tvm.__version__ not in {TVM_VERSION, "0.25.dev0"}:
        raise TVMConversionError(f"expected TVM {TVM_VERSION}, found {tvm.__version__}")
    return tvm


def _register_portable_tir_pipeline(tvm: Any) -> str:
    """Register the stock S-TIR pipeline followed by mandatory int32 narrowing.

    ``INDEX_DEFAULT_I64=OFF`` controls newly created index expressions, but the
    imported Relax graph can still carry int64 loop and thread indices into
    Vulkan code generation.  Vulkan's portable profile deliberately rejects
    the Int64 capability, so apply TVM's stronger, checked narrowing pass to
    the fully lowered host/device module before final code generation.
    """
    from tvm import tirx  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.pipeline import (  # pylint: disable=import-outside-toplevel
        default_s_tir_pipeline,
    )

    pipeline_name = "rawtherapee-vulkan-int32-v1"

    def pipeline_factory() -> tuple[Any, Any, Any]:
        stock, finalize_host, finalize_device = default_s_tir_pipeline()
        pipeline = tvm.transform.Sequential(
            [stock, tirx.transform.ForceNarrowIndexToInt32()]
        )
        return pipeline, finalize_host, finalize_device

    tirx.register_tir_pipeline(pipeline_name, pipeline_factory)
    return pipeline_name


def build_module(
    source: str | Path,
    spec: ModelSpec,
    tvm_source: Path,
    tvm_build: Path,
    output: Path,
    workspace: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = authenticate_onnx(source, spec)
    promoted_float16_items = promote_float16_graph_to_float32(model)
    workspace.mkdir(parents=True, exist_ok=True)
    tvm = _load_tvm(tvm_source, tvm_build)
    from tvm.relax.frontend.onnx import from_onnx  # pylint: disable=import-outside-toplevel

    target = tvm.target.Target(PORTABLE_TARGET, host=tvm.target.Target(PORTABLE_HOST))
    module = from_onnx(
        model,
        shape_dict={"input": list(spec.input_shape)},
        dtype_dict={"input": "float32"},
        keep_params_in_input=False,
    )
    tir_pipeline = _register_portable_tir_pipeline(tvm)
    # Limit Relax fusion depth so generated kernels stay within Vulkan 1.1's
    # portable minimum of four storage-buffer descriptors per shader stage.
    with tvm.transform.PassContext(opt_level=3, config={"relax.FuseOps.max_depth": 1}):
        executable = tvm.compile(module, target=target, tir_pipeline=tir_pipeline)
    spirv = _spirv_records(executable, workspace)
    library_dir = tvm_build / "lib" if (tvm_build / "lib").is_dir() else tvm_build
    executable.export_library(
        str(output),
        workspace_dir=str(workspace),
        options=[
            "-Wl,--build-id=none",
            "-Wl,--no-undefined",
            f"-L{library_dir}",
            "-Wl,--no-as-needed",
            "-ltvm_runtime",
            "-ltvm_ffi",
            f"-ffile-prefix-map={workspace}=.",
            f"-fdebug-prefix-map={workspace}=.",
        ],
    )
    if not output.is_file():
        raise TVMConversionError("TVM did not emit the requested module")
    target_manifest = portable_target_manifest()
    target_manifest["source_graph_float16_items_promoted"] = promoted_float16_items
    return target_manifest, spirv


def _manifest(spec: ModelSpec, module: Path, target: dict[str, Any], spirv: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "model": {
            "id": spec.model_id,
            "input": {"dtype": "float32", "name": "input", "shape": list(spec.input_shape)},
            "license": spec.license,
            "output": {"dtype": "float32", "name": "output", "shape": list(spec.output_shape)},
            "source_onnx_sha256": spec.sha256,
            "source_onnx_size_bytes": spec.size_bytes,
            "upstream_revision": spec.upstream_revision,
        },
        "module": {
            "sha256": file_sha256(module),
            "size_bytes": module.stat().st_size,
        },
        "spirv_modules": spirv,
        "target": target,
        "tvm": {
            "ffi_revision": TVM_FFI_COMMIT,
            "release_archive": dict(TVM_ARCHIVE),
            "release_commit": TVM_COMMIT,
            "version": TVM_VERSION,
        },
    }


def convert(
    source: str | Path,
    model: str,
    output: str | Path,
    tvm_source: str | Path,
    tvm_build: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    destination = Path(output)
    manifest_path = Path(str(destination) + ".json")
    if not force and (destination.exists() or manifest_path.exists()):
        raise TVMConversionError("output or companion manifest already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="rt-tvm-phase13-", dir=destination.parent))
    temporary_module = work / destination.name
    temporary_manifest = work / (destination.name + ".json")
    try:
        target, spirv = build_module(
            source,
            spec,
            Path(tvm_source),
            Path(tvm_build),
            temporary_module,
            work / "export",
        )
        manifest = _manifest(spec, temporary_module, target, spirv)
        temporary_manifest.write_bytes(canonical_manifest_bytes(manifest))
        os.chmod(temporary_module, 0o644)
        os.chmod(temporary_manifest, 0o644)
        os.replace(temporary_manifest, manifest_path)
        os.replace(temporary_module, destination)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tvm-source", type=Path, required=True)
    parser.add_argument("--tvm-build", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = convert(
            args.source,
            args.model,
            args.output,
            args.tvm_source,
            args.tvm_build,
            force=args.force,
        )
    except (TVMConversionError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["module"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

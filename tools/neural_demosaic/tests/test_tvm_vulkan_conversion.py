from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import pytest

from tools.neural_demosaic.convert_tvm_vulkan import (
    DEFAULT_TARGET_PROFILE,
    DIAGNOSTIC_VULKAN12_TARGET,
    FORMAT,
    MODEL_SPECS,
    PORTABLE_TARGET,
    SPIRV_1_3,
    SPIRV_1_5,
    TARGET_PROFILES,
    TVMConversionError,
    VULKAN_1_1,
    VULKAN_1_2,
    VULKAN12_TVM_PATCH_MARKER,
    VULKAN12_TVM_PATCH_SHA256,
    authenticate_onnx,
    convert,
    portable_target_manifest,
    selected_target_profile,
    verify_compiler_profile,
)
from tools.neural_demosaic.inspect_checkpoint import canonical_manifest_bytes


EXPECTED_ARTIFACTS = {
    "xveon": {
        "filename": "xveon-tvm-vulkan-linux-x86_64.so",
        "module_size": 32_594_896,
        "module_sha256": "8648e3741a98345c8bc76b9e1c853a3b4ef58155b65ed726f9c2fda0226c206d",
        "manifest_sha256": "ddd39d1c3e68fe01f24ead4955ee61f2aedbb9350b82ce2c4bdc75ec5f556baf",
    },
    "packedxtransnet": {
        "filename": "packedxtransnet-tvm-vulkan-linux-x86_64.so",
        "module_size": 3_097_488,
        "module_sha256": "2975665b5f36ffb29e9b0c9dec62f69ffc916605189f41ae11484e19d18cfc6e",
        "manifest_sha256": "b84eece056d97ef0218bf0dcf705027709f4564f01db7d9a3fcbcb78a9d7842f",
    },
}

EXPECTED_VULKAN12_ARTIFACTS = {
    "xveon": {
        "filename": "xveon-tvm-vulkan12-linux-x86_64.so",
        "module_size": 32_594_896,
        "module_sha256": "43a1af9a0171844befe6e6d7714049abeb80cb6c3820e3503ef0186408a033d9",
        "manifest_sha256": "c27bff81782e09736e96353a0c7176e88501237ce6bb45c743a62d0a1b1eb1b7",
    },
    "packedxtransnet": {
        "filename": "packedxtransnet-tvm-vulkan12-linux-x86_64.so",
        "module_size": 3_097_488,
        "module_sha256": "aa618b4d2b0cdcd4f7d054e0d6d778e5afa1cfda0fa5ee9786fe8c5e1d79bbd1",
        "manifest_sha256": "f87bffed467da589e7155198ba10d408ceaa144039ef5382d25142da0bb51e71",
    },
}


def external_model(name: str) -> Path:
    variable = "XVEON_XTRANS_ONNX" if name == "xveon" else "PACKED_XTRANS_ONNX"
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set")
    return Path(value)


@pytest.mark.parametrize("name", sorted(EXPECTED_ARTIFACTS))
def test_reviewed_generated_artifact_identity(name: str) -> None:
    root_value = os.environ.get("TVM_PHASE13_ARTIFACT_DIR")
    if not root_value:
        pytest.skip("TVM_PHASE13_ARTIFACT_DIR is not set")
    expected = EXPECTED_ARTIFACTS[name]
    module = Path(root_value) / str(expected["filename"])
    manifest_path = Path(str(module) + ".json")
    assert module.stat().st_size == expected["module_size"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == expected["module_sha256"]
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == expected["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == canonical_manifest_bytes(manifest)
    assert manifest["module"] == {
        "sha256": expected["module_sha256"],
        "size_bytes": expected["module_size"],
    }


@pytest.mark.parametrize("name", sorted(EXPECTED_VULKAN12_ARTIFACTS))
def test_diagnostic_vulkan12_artifact_identity(name: str) -> None:
    root_value = os.environ.get("TVM_PHASE13_VULKAN12_ARTIFACT_DIR")
    if not root_value:
        pytest.skip("TVM_PHASE13_VULKAN12_ARTIFACT_DIR is not set")
    expected = EXPECTED_VULKAN12_ARTIFACTS[name]
    module = Path(root_value) / str(expected["filename"])
    manifest_path = Path(str(module) + ".json")
    assert module.stat().st_size == expected["module_size"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == expected["module_sha256"]
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == expected["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == canonical_manifest_bytes(manifest)
    assert manifest["module"] == {
        "sha256": expected["module_sha256"],
        "size_bytes": expected["module_size"],
    }
    target = dict(manifest["target"])
    promoted = target.pop("source_graph_float16_items_promoted")
    assert promoted == (48 if name == "xveon" else 0)
    assert target == portable_target_manifest("diagnostic-vulkan12")


@pytest.mark.parametrize("name", sorted(MODEL_SPECS))
def test_authenticates_reviewed_onnx_contract(name: str) -> None:
    model = authenticate_onnx(external_model(name), MODEL_SPECS[name])
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


def test_authentication_precedes_deserialization(tmp_path: Path) -> None:
    source = tmp_path / "bad.onnx"
    source.write_bytes(b"not protobuf")
    spec = replace(MODEL_SPECS["xveon"], size_bytes=12, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(TVMConversionError, match="invalid authenticated"):
        authenticate_onnx(source, spec)
    with pytest.raises(TVMConversionError, match="size differs"):
        authenticate_onnx(source, MODEL_SPECS["xveon"])


def test_rejects_wrong_graph_contract(tmp_path: Path) -> None:
    source_model = onnx.load(external_model("packedxtransnet"), load_external_data=False)
    source_model.graph.input[0].name = "wrong"
    for node in source_model.graph.node:
        for index, name in enumerate(node.input):
            if name == "input":
                node.input[index] = "wrong"
    payload = source_model.SerializeToString(deterministic=True)
    source = tmp_path / "wrong.onnx"
    source.write_bytes(payload)
    spec = replace(
        MODEL_SPECS["packedxtransnet"],
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(TVMConversionError, match="input contract"):
        authenticate_onnx(source, spec)


def test_rejects_unreviewed_operator_contract(tmp_path: Path) -> None:
    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 288, 288])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 288, 288])
    repeats = helper.make_tensor("repeats", TensorProto.INT64, [4], [1, 3, 1, 1])
    graph = helper.make_graph(
        [helper.make_node("Tile", ["input", "repeats"], ["output"])],
        "unreviewed",
        [input_value],
        [output_value],
        [repeats],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    payload = model.SerializeToString(deterministic=True)
    source = tmp_path / "unreviewed.onnx"
    source.write_bytes(payload)
    spec = replace(
        MODEL_SPECS["packedxtransnet"],
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(TVMConversionError, match="operator contract"):
        authenticate_onnx(source, spec)


def test_portable_target_is_conservative_and_canonical() -> None:
    assert PORTABLE_TARGET["vulkan_api_version"] == VULKAN_1_1
    assert PORTABLE_TARGET["max_spirv_version"] == SPIRV_1_3
    assert PORTABLE_TARGET["supports_float32"] == 1
    assert PORTABLE_TARGET["supports_int32"] == 1
    for feature in (
        "supports_float16",
        "supports_float64",
        "supports_int8",
        "supports_int16",
        "supports_int64",
        "supports_integer_dot_product",
        "supports_cooperative_matrix",
        "supports_push_descriptor",
    ):
        assert PORTABLE_TARGET[feature] == 0
    assert PORTABLE_TARGET["max_num_threads"] == 128
    assert PORTABLE_TARGET["max_per_stage_descriptor_storage_buffer"] == 4
    assert PORTABLE_TARGET["max_shared_memory_per_block"] == 16_384
    assert PORTABLE_TARGET["max_storage_buffer_range"] == 128 * 1024 * 1024
    manifest = {"format": FORMAT, "target": portable_target_manifest()}
    payload = canonical_manifest_bytes(manifest)
    assert payload == canonical_manifest_bytes(json.loads(payload))
    assert b"/tmp/" not in payload and b"timestamp" not in payload


def test_vulkan12_profile_changes_only_environment_versions() -> None:
    assert DEFAULT_TARGET_PROFILE == "portable-vulkan11"
    assert sorted(TARGET_PROFILES) == ["diagnostic-vulkan12", "portable-vulkan11"]
    assert DIAGNOSTIC_VULKAN12_TARGET["vulkan_api_version"] == VULKAN_1_2
    assert DIAGNOSTIC_VULKAN12_TARGET["max_spirv_version"] == SPIRV_1_5
    differences = {
        key
        for key in PORTABLE_TARGET
        if PORTABLE_TARGET[key] != DIAGNOSTIC_VULKAN12_TARGET[key]
    }
    assert differences == {"max_spirv_version", "vulkan_api_version"}
    diagnostic = portable_target_manifest("diagnostic-vulkan12")
    assert diagnostic["profile"] == "linux-x86_64-vulkan-1.2-spirv-1.5-fp32-diagnostic-v1"
    assert diagnostic["target"] == DIAGNOSTIC_VULKAN12_TARGET
    assert diagnostic["compiler_patch"] == {
        "file": "apache-tvm-0.25.0-vulkan12-spirv15.patch",
        "sha256": VULKAN12_TVM_PATCH_SHA256,
    }
    assert canonical_manifest_bytes(diagnostic) == canonical_manifest_bytes(
        json.loads(canonical_manifest_bytes(diagnostic))
    )


def test_vulkan12_compiler_patch_is_authenticated(tmp_path: Path) -> None:
    portable = selected_target_profile(DEFAULT_TARGET_PROFILE)
    diagnostic = selected_target_profile("diagnostic-vulkan12")
    verify_compiler_profile(tmp_path, portable)
    with pytest.raises(TVMConversionError, match="lacks"):
        verify_compiler_profile(tmp_path, diagnostic)
    marker = tmp_path / VULKAN12_TVM_PATCH_MARKER
    marker.write_text("wrong\n", encoding="ascii")
    with pytest.raises(TVMConversionError, match="marker mismatch"):
        verify_compiler_profile(tmp_path, diagnostic)
    marker.write_text(VULKAN12_TVM_PATCH_SHA256 + "\n", encoding="ascii")
    verify_compiler_profile(tmp_path, diagnostic)


def test_rejects_unknown_target_profile() -> None:
    with pytest.raises(TVMConversionError, match="unknown target profile"):
        selected_target_profile("future-vulkan")
    with pytest.raises(TVMConversionError, match="unknown target profile"):
        convert(
            "unused",
            "xveon",
            "unused",
            "unused",
            "unused",
            target_profile="future-vulkan",
        )


def test_output_replacement_is_explicit_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "model.so"
    output.write_bytes(b"old module")
    with pytest.raises(TVMConversionError, match="already exists"):
        convert("unused", "xveon", output, "unused", "unused")

    def fake_build(
        source,
        spec,
        tvm_source,
        tvm_build,
        temporary,
        workspace,
        *,
        target_profile,
    ):
        del source, spec, tvm_source, tvm_build, workspace
        assert target_profile == DEFAULT_TARGET_PROFILE
        temporary.write_bytes(b"new module")
        return portable_target_manifest(), [{"entry_point": "synthetic"}]

    monkeypatch.setattr("tools.neural_demosaic.convert_tvm_vulkan.build_module", fake_build)
    manifest = convert("unused", "xveon", output, "unused", "unused", force=True)
    assert output.read_bytes() == b"new module"
    assert Path(str(output) + ".json").read_bytes() == canonical_manifest_bytes(manifest)
    assert not list(tmp_path.glob("rt-tvm-phase13-*"))

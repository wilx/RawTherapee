set(NEURALMODEL_TEST_SOURCES
    demosaicnet_inference_tests.cc
    neuralmodel_tests.cc
    rtnn_inspection.cc
    xtrans_demosaicnet_tests.cc
    xtrans_packed_tests.cc
    xtrans_xveon_tests.cc
)

add_executable(rawtherapee-neuralmodel-tests ${NEURALMODEL_TEST_SOURCES})
target_link_libraries(rawtherapee-neuralmodel-tests PRIVATE rtengine rtengine-neuralmodel-inference)
target_include_directories(rawtherapee-neuralmodel-tests PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(
    rawtherapee-neuralmodel-tests
    PRIVATE
        RT_NEURAL_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tools/neural_demosaic/golden/demosaicnet-xtrans-v1"
        RT_NEURAL_TRACE_DIR="${PROJECT_SOURCE_DIR}/tools/neural_demosaic/native_trace"
        RT_NEURAL_RAW_WRAPPER_DIR="${PROJECT_SOURCE_DIR}/tools/neural_demosaic/golden/demosaicnet-xtrans-raw-wrapper-v1"
)
if(WITH_ONNXRUNTIME)
    target_compile_definitions(rawtherapee-neuralmodel-tests PRIVATE RT_TEST_WITH_ONNXRUNTIME)
endif()
if(WITH_MIGRAPHX)
    target_compile_definitions(rawtherapee-neuralmodel-tests PRIVATE RT_TEST_WITH_MIGRAPHX)
endif()
if(WITH_TVM_VULKAN)
    target_compile_definitions(rawtherapee-neuralmodel-tests PRIVATE RT_TEST_WITH_TVM_VULKAN)
endif()
rt_setup_target(rawtherapee-neuralmodel-tests)

add_executable(
    rawtherapee-rtnn-inspect
    rtnn_inspect.cc
    rtnn_inspection.cc
)
target_link_libraries(rawtherapee-rtnn-inspect PRIVATE rtengine-neuralmodel-loader)
target_include_directories(rawtherapee-rtnn-inspect PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-rtnn-inspect)

if(WITH_ONNXRUNTIME OR WITH_MIGRAPHX OR WITH_TVM_VULKAN)
    add_executable(rawtherapee-xveon-tile xveon_tile.cc)
    target_link_libraries(rawtherapee-xveon-tile PRIVATE rtengine-xveon-onnx)
    target_include_directories(rawtherapee-xveon-tile PRIVATE ${PROJECT_SOURCE_DIR})
    rt_setup_target(rawtherapee-xveon-tile)

    add_executable(rawtherapee-xveon-wrapper xveon_wrapper.cc)
    target_link_libraries(rawtherapee-xveon-wrapper PRIVATE rtengine)
    target_include_directories(rawtherapee-xveon-wrapper PRIVATE ${PROJECT_SOURCE_DIR})
    rt_setup_target(rawtherapee-xveon-wrapper)

    if(WITH_TVM_VULKAN AND UNIX AND NOT APPLE)
        set_target_properties(
            rawtherapee-neuralmodel-tests
            rawtherapee-xveon-tile
            rawtherapee-xveon-wrapper
            PROPERTIES BUILD_RPATH "\$ORIGIN;${TVM_RUNTIME_LIBRARY_DIR}")
    endif()
endif()

function(add_neuralmodel_test TEST_NAME TEST_MODE)
    add_test(
        NAME ${TEST_NAME}
        COMMAND rawtherapee-neuralmodel-tests ${TEST_MODE}
    )
    set_tests_properties(
        ${TEST_NAME}
        PROPERTIES
            LABELS "rtengine;neural-model"
            TIMEOUT 90
    )
endfunction()

add_neuralmodel_test(rtengine.neural_model.synthetic_valid synthetic-valid)
add_neuralmodel_test(rtengine.neural_model.header_corruption header-corruption)
add_neuralmodel_test(rtengine.neural_model.directory_corruption directory-corruption)
add_neuralmodel_test(rtengine.neural_model.payload_and_digest_corruption payload-corruption)
add_neuralmodel_test(rtengine.neural_model.limits_and_file_io limits-io)
add_neuralmodel_test(rtengine.neural_model.reviewed_artifact reviewed-artifact)
add_neuralmodel_test(rtengine.neural_model.inspection_parity inspection-parity)
add_neuralmodel_test(rtengine.neural_model.inference_synthetic inference-synthetic)
add_neuralmodel_test(rtengine.neural_model.inference_errors inference-errors)
add_neuralmodel_test(rtengine.neural_model.inference_golden inference-golden)
add_neuralmodel_test(rtengine.neural_model.inference_trace inference-trace)
add_neuralmodel_test(rtengine.neural_model.xtrans_raw_contract xtrans-raw-contract)
add_neuralmodel_test(rtengine.neural_model.xtrans_raw_reviewed xtrans-raw-reviewed)
add_neuralmodel_test(rtengine.neural_model.xveon_mock xveon-mock)
add_neuralmodel_test(rtengine.neural_model.xveon_loader xveon-loader)
add_neuralmodel_test(rtengine.neural_model.xveon_reviewed xveon-reviewed)
add_neuralmodel_test(rtengine.neural_model.xveon_migraphx xveon-migraphx)
add_neuralmodel_test(rtengine.neural_model.xveon_tvm_vulkan xveon-tvm-vulkan)
add_neuralmodel_test(rtengine.neural_model.packed_mock packed-mock)
add_neuralmodel_test(rtengine.neural_model.packed_loader packed-loader)
add_neuralmodel_test(rtengine.neural_model.packed_reviewed packed-reviewed)
add_neuralmodel_test(rtengine.neural_model.packed_migraphx packed-migraphx)
add_neuralmodel_test(rtengine.neural_model.packed_tvm_vulkan packed-tvm-vulkan)

set_tests_properties(
    rtengine.neural_model.reviewed_artifact
    rtengine.neural_model.inspection_parity
    rtengine.neural_model.inference_golden
    rtengine.neural_model.inference_trace
    rtengine.neural_model.xtrans_raw_reviewed
    rtengine.neural_model.xveon_reviewed
    rtengine.neural_model.xveon_migraphx
    rtengine.neural_model.xveon_tvm_vulkan
    rtengine.neural_model.packed_reviewed
    rtengine.neural_model.packed_migraphx
    rtengine.neural_model.packed_tvm_vulkan
    PROPERTIES SKIP_RETURN_CODE 77
)

set_tests_properties(
    rtengine.neural_model.xveon_migraphx
    rtengine.neural_model.packed_migraphx
    rtengine.neural_model.xveon_tvm_vulkan
    rtengine.neural_model.packed_tvm_vulkan
    PROPERTIES TIMEOUT 180)

# The portable direct convolution is intentionally unoptimized in Debug. The
# seven reviewed raw-wrapper cases still run there, but need a wider timeout
# than the ordinary loader and optimized-inference cases.
set_tests_properties(
    rtengine.neural_model.xtrans_raw_contract
    rtengine.neural_model.xtrans_raw_reviewed
    PROPERTIES TIMEOUT 300
)

# The benchmark is deliberately not registered with CTest. It has no stable
# machine-independent timing threshold and is run explicitly by developers.

set(NEURALMODEL_TEST_SOURCES
    demosaicnet_inference_tests.cc
    neuralmodel_tests.cc
    rtnn_inspection.cc
    xtrans_demosaicnet_tests.cc
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
rt_setup_target(rawtherapee-neuralmodel-tests)

add_executable(
    rawtherapee-rtnn-inspect
    rtnn_inspect.cc
    rtnn_inspection.cc
)
target_link_libraries(rawtherapee-rtnn-inspect PRIVATE rtengine-neuralmodel-loader)
target_include_directories(rawtherapee-rtnn-inspect PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-rtnn-inspect)

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

set_tests_properties(
    rtengine.neural_model.reviewed_artifact
    rtengine.neural_model.inspection_parity
    rtengine.neural_model.inference_golden
    rtengine.neural_model.inference_trace
    rtengine.neural_model.xtrans_raw_reviewed
    PROPERTIES SKIP_RETURN_CODE 77
)

# The portable direct convolution is intentionally unoptimized in Debug. The
# seven reviewed raw-wrapper cases still run there, but need a wider timeout
# than the ordinary loader and optimized-inference cases.
set_tests_properties(
    rtengine.neural_model.xtrans_raw_reviewed
    PROPERTIES TIMEOUT 300
)

# The benchmark is deliberately not registered with CTest. It has no stable
# machine-independent timing threshold and is run explicitly by developers.

set(NEURALMODEL_TEST_SOURCES
    neuralmodel_tests.cc
    rtnn_inspection.cc
)

add_executable(rawtherapee-neuralmodel-tests ${NEURALMODEL_TEST_SOURCES})
target_link_libraries(rawtherapee-neuralmodel-tests PRIVATE rtengine-neuralmodel-loader)
target_include_directories(rawtherapee-neuralmodel-tests PRIVATE ${PROJECT_SOURCE_DIR})
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

set_tests_properties(
    rtengine.neural_model.reviewed_artifact
    rtengine.neural_model.inspection_parity
    PROPERTIES SKIP_RETURN_CODE 77
)

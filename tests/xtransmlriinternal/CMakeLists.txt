add_executable(
    rawtherapee-xtrans-mlri-internal-tests
    mlri_internal_tests.cc
    ${PROJECT_SOURCE_DIR}/rtgui/alignedmalloc.cc
    ${PROJECT_SOURCE_DIR}/rtgui/editcallbacks.cc
    ${PROJECT_SOURCE_DIR}/rtgui/multilangmgr.cc
    ${PROJECT_SOURCE_DIR}/rtgui/options.cc
    ${PROJECT_SOURCE_DIR}/rtgui/paramsedited.cc
    ${PROJECT_SOURCE_DIR}/rtgui/pathutils.cc
    ${PROJECT_SOURCE_DIR}/rtgui/threadutils.cc)
target_link_libraries(rawtherapee-xtrans-mlri-internal-tests PRIVATE rtengine)
target_include_directories(
    rawtherapee-xtrans-mlri-internal-tests
    BEFORE PRIVATE
        ${CMAKE_BINARY_DIR}/rtgui
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/rtgui)
target_compile_definitions(rawtherapee-xtrans-mlri-internal-tests PRIVATE CLIVERSION)
rt_setup_target(rawtherapee-xtrans-mlri-internal-tests)

add_test(
    NAME rtengine.xtrans_mlri_internal.contract
    COMMAND rawtherapee-xtrans-mlri-internal-tests contract)
set_tests_properties(
    rtengine.xtrans_mlri_internal.contract
    PROPERTIES LABELS "rtengine;xtrans-mlri-internal;xtrans-mlri-regression" TIMEOUT 300)

add_test(
    NAME rtengine.xtrans_mlri_internal.synthetic
    COMMAND rawtherapee-xtrans-mlri-internal-tests synthetic)
set_tests_properties(
    rtengine.xtrans_mlri_internal.synthetic
    PROPERTIES LABELS "rtengine;xtrans-mlri-internal;xtrans-mlri-regression" TIMEOUT 600)

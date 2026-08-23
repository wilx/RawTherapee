add_executable(
    rawtherapee-xtrans-ulri-tests
    ulri_tests.cc
    ${PROJECT_SOURCE_DIR}/rtgui/alignedmalloc.cc
    ${PROJECT_SOURCE_DIR}/rtgui/editcallbacks.cc
    ${PROJECT_SOURCE_DIR}/rtgui/multilangmgr.cc
    ${PROJECT_SOURCE_DIR}/rtgui/options.cc
    ${PROJECT_SOURCE_DIR}/rtgui/paramsedited.cc
    ${PROJECT_SOURCE_DIR}/rtgui/pathutils.cc
    ${PROJECT_SOURCE_DIR}/rtgui/threadutils.cc)
target_link_libraries(rawtherapee-xtrans-ulri-tests PRIVATE rtengine)
target_include_directories(
    rawtherapee-xtrans-ulri-tests
    BEFORE PRIVATE
        ${CMAKE_BINARY_DIR}/rtgui
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/rtgui)
target_compile_definitions(rawtherapee-xtrans-ulri-tests PRIVATE CLIVERSION)
rt_setup_target(rawtherapee-xtrans-ulri-tests)

add_test(
    NAME rtengine.xtrans_ulri.contract
    COMMAND rawtherapee-xtrans-ulri-tests contract)
set_tests_properties(
    rtengine.xtrans_ulri.contract
    PROPERTIES LABELS "rtengine;xtrans-ulri" TIMEOUT 300)

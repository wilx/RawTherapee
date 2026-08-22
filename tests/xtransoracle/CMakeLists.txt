add_executable(
    rawtherapee-xtrans-oracle-runner
    oracle_runner.cc
    ${PROJECT_SOURCE_DIR}/rtgui/alignedmalloc.cc
    ${PROJECT_SOURCE_DIR}/rtgui/editcallbacks.cc
    ${PROJECT_SOURCE_DIR}/rtgui/multilangmgr.cc
    ${PROJECT_SOURCE_DIR}/rtgui/options.cc
    ${PROJECT_SOURCE_DIR}/rtgui/paramsedited.cc
    ${PROJECT_SOURCE_DIR}/rtgui/pathutils.cc
    ${PROJECT_SOURCE_DIR}/rtgui/threadutils.cc)
target_link_libraries(rawtherapee-xtrans-oracle-runner PRIVATE rtengine)
target_include_directories(
    rawtherapee-xtrans-oracle-runner
    BEFORE PRIVATE
        ${CMAKE_BINARY_DIR}/rtgui
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/rtgui)
target_compile_definitions(rawtherapee-xtrans-oracle-runner PRIVATE CLIVERSION)
rt_setup_target(rawtherapee-xtrans-oracle-runner)

foreach(mode IN ITEMS contract synthetic)
    add_test(
        NAME rtengine.xtrans_oracle.${mode}
        COMMAND rawtherapee-xtrans-oracle-runner ${mode})
    set_tests_properties(
        rtengine.xtrans_oracle.${mode}
        PROPERTIES LABELS "rtengine;xtrans-oracle" TIMEOUT 600)
endforeach()

add_executable(rawtherapee-xtrans-triangulation-tests triangulation_tests.cc)
target_link_libraries(rawtherapee-xtrans-triangulation-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-triangulation-tests PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-xtrans-triangulation-tests)

add_test(
    NAME rtengine.xtrans_triangulation.contract
    COMMAND rawtherapee-xtrans-triangulation-tests contract)
set_tests_properties(
    rtengine.xtrans_triangulation.contract
    PROPERTIES LABELS "rtengine;xtrans-triangulation" TIMEOUT 120)

add_test(
    NAME rtengine.xtrans_triangulation.synthetic
    COMMAND rawtherapee-xtrans-triangulation-tests synthetic)
set_tests_properties(
    rtengine.xtrans_triangulation.synthetic
    PROPERTIES LABELS "rtengine;xtrans-triangulation" TIMEOUT 120)

add_test(
    NAME rtengine.xtrans_triangulation.orientation
    COMMAND rawtherapee-xtrans-triangulation-tests orientation)
set_tests_properties(
    rtengine.xtrans_triangulation.orientation
    PROPERTIES LABELS "rtengine;xtrans-triangulation" TIMEOUT 120)

add_test(
    NAME rtengine.xtrans_triangulation.artifacts
    COMMAND rawtherapee-xtrans-triangulation-tests artifacts)
set_tests_properties(
    rtengine.xtrans_triangulation.artifacts
    PROPERTIES LABELS "rtengine;xtrans-triangulation" TIMEOUT 120)

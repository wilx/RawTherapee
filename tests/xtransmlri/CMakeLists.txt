add_executable(rawtherapee-xtrans-mlri-tests mlri_tests.cc)
target_link_libraries(rawtherapee-xtrans-mlri-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-mlri-tests PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(
    rawtherapee-xtrans-mlri-tests
    PRIVATE RT_MLRI_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tools/xtrans_mlri/golden")
rt_setup_target(rawtherapee-xtrans-mlri-tests)

add_test(
    NAME rtengine.xtrans_mlri.contract
    COMMAND rawtherapee-xtrans-mlri-tests contract)
set_tests_properties(
    rtengine.xtrans_mlri.contract
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 300)

add_test(
    NAME rtengine.xtrans_mlri.golden
    COMMAND rawtherapee-xtrans-mlri-tests golden)
set_tests_properties(
    rtengine.xtrans_mlri.golden
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 900)

add_test(
    NAME rtengine.xtrans_mlri.corrected_variant
    COMMAND rawtherapee-xtrans-mlri-tests corrected)
set_tests_properties(
    rtengine.xtrans_mlri.corrected_variant
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 300)

add_test(
    NAME rtengine.xtrans_mlri.paper_core_variants
    COMMAND rawtherapee-xtrans-mlri-tests paper-core)
set_tests_properties(
    rtengine.xtrans_mlri.paper_core_variants
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 300)

add_test(
    NAME rtengine.xtrans_mlri.tiled_seam
    COMMAND rawtherapee-xtrans-mlri-tests tiled-seam)
set_tests_properties(
    rtengine.xtrans_mlri.tiled_seam
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 900)

add_test(
    NAME rtengine.xtrans_mlri.orientation_parity
    COMMAND rawtherapee-xtrans-mlri-tests orientation-parity)
set_tests_properties(
    rtengine.xtrans_mlri.orientation_parity
    PROPERTIES LABELS "rtengine;xtrans-mlri" TIMEOUT 900)

add_executable(rawtherapee-xtrans-global-tests global_tests.cc)
target_link_libraries(rawtherapee-xtrans-global-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-global-tests PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-xtrans-global-tests)

foreach(mode IN ITEMS contract synthetic orientation tiled control)
    add_test(
        NAME rtengine.xtrans_global.${mode}
        COMMAND rawtherapee-xtrans-global-tests ${mode})
    set_tests_properties(
        rtengine.xtrans_global.${mode}
        PROPERTIES LABELS "rtengine;xtrans-global" TIMEOUT 600)
endforeach()

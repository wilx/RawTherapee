add_executable(rawtherapee-xtrans-alias-tests alias_tests.cc)
target_link_libraries(rawtherapee-xtrans-alias-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-alias-tests PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-xtrans-alias-tests)

foreach(mode IN ITEMS contract spectra replicas variants linearity)
    add_test(
        NAME rtengine.xtrans_alias.${mode}
        COMMAND rawtherapee-xtrans-alias-tests ${mode})
    set_tests_properties(
        rtengine.xtrans_alias.${mode}
        PROPERTIES LABELS "rtengine;xtrans-alias" TIMEOUT 120)
endforeach()

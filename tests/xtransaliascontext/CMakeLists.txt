add_executable(rawtherapee-xtrans-alias-context-tests alias_context_tests.cc)
target_link_libraries(rawtherapee-xtrans-alias-context-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-alias-context-tests PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-xtrans-alias-context-tests)

foreach(mode IN ITEMS contract phase null_context selection transition)
    add_test(
        NAME rtengine.xtrans_alias_context.${mode}
        COMMAND rawtherapee-xtrans-alias-context-tests ${mode})
    set_tests_properties(
        rtengine.xtrans_alias_context.${mode}
        PROPERTIES LABELS "rtengine;xtrans-alias-context" TIMEOUT 120)
endforeach()

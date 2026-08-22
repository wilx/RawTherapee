add_executable(rawtherapee-xtrans-sparse-alias-tests sparse_alias_tests.cc)
target_link_libraries(rawtherapee-xtrans-sparse-alias-tests PRIVATE rtengine)
target_include_directories(rawtherapee-xtrans-sparse-alias-tests PRIVATE ${PROJECT_SOURCE_DIR})
rt_setup_target(rawtherapee-xtrans-sparse-alias-tests)

foreach(mode IN ITEMS contract oracle omp ambiguity grouped)
    add_test(
        NAME rtengine.xtrans_sparse_alias.${mode}
        COMMAND rawtherapee-xtrans-sparse-alias-tests ${mode})
    set_tests_properties(
        rtengine.xtrans_sparse_alias.${mode}
        PROPERTIES LABELS "rtengine;xtrans-sparse-alias" TIMEOUT 120)
endforeach()

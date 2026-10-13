add_executable(ninfer_ple_table_test "${CMAKE_CURRENT_LIST_DIR}/test_ple_table.cpp")
ninfer_test_includes(ninfer_ple_table_test)
target_link_libraries(ninfer_ple_table_test PRIVATE ninfer_model_loading)
add_test(NAME ninfer_ple_table_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_ple_table.py"
    $<TARGET_FILE:ninfer_ple_table_test>)
set_tests_properties(ninfer_ple_table_test PROPERTIES TIMEOUT 30)

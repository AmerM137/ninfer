add_executable(ninfer_ple_table_test "${CMAKE_CURRENT_LIST_DIR}/test_ple_table.cpp")
ninfer_test_includes(ninfer_ple_table_test)
target_link_libraries(ninfer_ple_table_test PRIVATE ninfer_model_loading)
add_test(NAME ninfer_ple_table_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_ple_table.py"
    $<TARGET_FILE:ninfer_ple_table_test>)
set_tests_properties(ninfer_ple_table_test PROPERTIES TIMEOUT 30)

add_executable(ninfer_ple_prepare_test "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp")
ninfer_test_includes(ninfer_ple_prepare_test)
target_link_libraries(ninfer_ple_prepare_test PRIVATE ninfer_model_runtime)
target_link_options(ninfer_ple_prepare_test PRIVATE "LINKER:--wrap=cudaEventRecord")
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_ple_prepare_${mode}_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_prepare.py"
      $<TARGET_FILE:ninfer_ple_prepare_test> --${mode})
  set_tests_properties(ninfer_ple_prepare_${mode}_test PROPERTIES TIMEOUT 30)
endforeach()
set_tests_properties(ninfer_ple_prepare_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(ninfer_ple_prepare_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

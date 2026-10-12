ninfer_add_op_test(ninfer_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixture.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_complete.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_route.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_routed.cpp"
  LIBRARIES ninfer_ops)

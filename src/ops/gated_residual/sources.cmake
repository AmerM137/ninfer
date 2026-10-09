target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_residual.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu")

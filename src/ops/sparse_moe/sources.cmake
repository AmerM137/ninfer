target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/decode/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/token_experts.cu"
)

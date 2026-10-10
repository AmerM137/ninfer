target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n1664_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n12288_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n16384_k2560.cu"
)

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/conv.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/conv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/state_extract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state_extract.cu"
)

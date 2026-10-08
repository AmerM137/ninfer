target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/record.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/record.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cu"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/recurrence.cu"
)

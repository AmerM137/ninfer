target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/workspace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/state_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/kv_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/buffers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/resources.cpp")

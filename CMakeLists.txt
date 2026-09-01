cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

if(MSVC)
  # UTF-8 source and execution charset (non-ASCII string literals keep their
  # UTF-8 bytes), section headroom for the template-heavy op/target TUs, the
  # standard-conforming preprocessor CCCL requires (it #errors on the
  # traditional one), and the windows.h definitions that keep its unguarded
  # min/max macros from breaking std::min/std::max in the ported platform
  # files. Each flag must reach both the host-compiler path and nvcc's
  # host-compilation path. This block also covers clang-cl (MSVC is true for
  # it); /Zc:preprocessor is cl-only there — clang's preprocessor is already
  # conforming and clang-cl warns "argument unused" on the flag — while the
  # -Xcompiler lines always target cl, nvcc's only supported Windows host
  # compiler, regardless of who compiles the C/C++ TUs.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

if(WIN32)
  # The shared FFmpeg runtime must sit beside every binary that loads it: the
  # two apps and the media-linked test executables. swresample never appears
  # on a link line (it is a private dependency of avformat behind the DLL
  # boundary) but is required at load time, so it is staged as well.
  # Configure-time copy: rerun CMake after upgrading the FFmpeg package.
  list(GET FFMPEG_LIBRARY_DIRS 0 NINFER_FFMPEG_LIBDIR)
  file(GLOB NINFER_FFMPEG_DLLS
    "${NINFER_FFMPEG_LIBDIR}/../bin/avformat-*.dll"
    "${NINFER_FFMPEG_LIBDIR}/../bin/avcodec-*.dll"
    "${NINFER_FFMPEG_LIBDIR}/../bin/avutil-*.dll"
    "${NINFER_FFMPEG_LIBDIR}/../bin/swscale-*.dll"
    "${NINFER_FFMPEG_LIBDIR}/../bin/swresample-*.dll")
  if(NOT NINFER_FFMPEG_DLLS)
    message(WARNING "no FFmpeg DLLs found under ${NINFER_FFMPEG_LIBDIR}/../bin; "
      "ninfer.exe, ninfer-serve.exe, and media-linked tests will fail to start")
  endif()
  if(NINFER_BUILD_APPS)
    file(COPY ${NINFER_FFMPEG_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/apps")
  endif()
  if(BUILD_TESTING)
    file(COPY ${NINFER_FFMPEG_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/tests")
  endif()
  if(NINFER_BUILD_BENCHMARKS)
    file(COPY ${NINFER_FFMPEG_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/bench")
  endif()
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

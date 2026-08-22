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
option(NINFER_USE_CCACHE "Compile through ccache when it is installed" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake scans every C++20 target for module dependencies by default under Ninja.
# NInfer declares no modules (no `export module`, no `import`), so the scan adds
# a dependency-scan edge per TU and, worse, puts -fmodules-ts/-fmodule-mapper on
# every C++ compile line -- flags ccache refuses to cache, which makes all of the
# C++ objects uncacheable. Turn it off unless a target ever ships a module.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling

# Compile through ccache when it is present. Every object here is cacheable --
# nvcc included -- so a rebuilt tree costs the link steps rather than a full
# compile. Found-not-required: a box without ccache configures unchanged, and
# an explicit -DCMAKE_<LANG>_COMPILER_LAUNCHER on the command line wins over
# this. Set -DNINFER_USE_CCACHE=OFF to force uncached compiles, which is what
# you want when timing a cold build.
if(NINFER_USE_CCACHE)
  find_program(NINFER_CCACHE_PROGRAM ccache)
  if(NINFER_CCACHE_PROGRAM)
    foreach(lang C CXX CUDA)
      if(NOT CMAKE_${lang}_COMPILER_LAUNCHER)
        set(CMAKE_${lang}_COMPILER_LAUNCHER "${NINFER_CCACHE_PROGRAM}")
      endif()
    endforeach()
    message(STATUS "ccache: compiling through ${NINFER_CCACHE_PROGRAM}")
  else()
    message(STATUS "ccache: not found, compiling directly")
  endif()
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The supported generator is Ninja. It parallelises by default (no --parallel
# needed) and it is the only generator that honours the link job pool below,
# which keeps the memory-hungry CUDA device link from running alongside the
# executable links. A default `cmake -B build` picks the platform default
# instead -- Unix Makefiles, which builds single-threaded unless -j is passed --
# so warn rather than let a 5x slower build look normal.
if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
else()
  message(WARNING
    "NInfer expects the Ninja generator; this build tree uses "
    "'${CMAKE_GENERATOR}'. Builds will be serial unless you pass --parallel, "
    "and the cuda_link job pool that serialises the CUDA device link is "
    "inactive. Reconfigure with -G Ninja (see README.md), or set the "
    "CMAKE_GENERATOR environment variable to Ninja.")
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

# Windows Port Plan (2026-08-15)

Status: temporary working plan. This document records the port design, work items, and
completion criteria for making NInfer build and run natively on 64-bit Windows. It is removed
when the port lands and its stable content is integrated into `README.md` and `AGENTS.md`.

Reviewed against the repository on 2026-08-17: the qwen3.8-27b identities, the fp8/NVFP4 op
wave, and the post-plan behavior fixes are accounted for in the text below.

Executed on 2026-08-17 (first bring-up session). Phase 0 is complete: dependencies are
provisioned and `cmake -S . -B build -G Ninja` **configures cleanly on Windows with no CMake
changes at all**. The first build reached 120/423 targets and stopped on two failures — the
expected `src/artifact/reader.cpp` POSIX includes, and a new blocker in `src/ops` that
section 4.4 had previously cleared (see 4.4.1). Sections 3, 4.1, 7, and 9 are updated to match
what was actually observed rather than what was projected.

## 1. Objective and scope

Deliver: `ninfer.exe` and `ninfer-serve.exe` built from this tree on 64-bit Windows with MSVC
and the CUDA Toolkit, executing the full product surface (Text, Vision, MTP/DFlash, prefix
reuse, CLI, OpenAI/Anthropic serving) from the same `.ninfer` artifacts, compiled for `sm_120a`
as today.

- The product contract in `AGENTS.md` gains Windows as a supported host platform alongside
  64-bit Linux. No target identity, kernel, artifact format, or scheduling change: the port is
  host-platform work only. **One known exception is now open** — see 4.4.1.
- Published performance numbers remain Linux/RTX 5090 results. Windows receives its own
  measured row at the end (WDDM), not a reinterpretation of the existing numbers.
- Docker stays Linux-only. The WSL/WSL2 path is explicitly out of scope: this is a native
  Windows port.
- Out of scope: TCC mode, multi-GPU, GPU-installed drivers, Windows packaging/installers.

## 2. Verified environment (target machine)

| Item | State |
|---|---|
| GPU | RTX 5090 32 GB, WDDM, driver 610.88 (UMD 13.3) |
| CUDA | 13.3 toolkit at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` (nvcc reports 13.3.33; satisfies the existing >= 13.1 gate) |
| Host compiler | MSVC 14.51 / `cl` 19.51.36256.0 via Visual Studio 2026 (Community) at `C:\Program Files\Microsoft Visual Studio\18`; clang 22 / clang-cl available but **not** used (nvcc on Windows officially supports only `cl` as host compiler) |
| Build tools | CMake 4.3.2, Ninja (scoop), pkgconf 3.0.5 (scoop, shimmed as `pkg-config.exe`) |
| Python | 3.14.6 (mamba env) |
| Python packages | `torch`/`safetensors`/`numpy` for the conversion tooling — confirm they are installed in this env before choosing the conversion option in section 9 (3.14.6 is fine for the tooling; the 3.11 convention in `AGENTS.md` is not a hard floor) |
| FFmpeg | **Provisioned.** BtbN `n6.1.2` win64 **LGPL shared** prebuilt at `C:\libs\ffmpeg-n6.1.2`. avformat 60.16.100, avcodec 60.31.102, avutil 58.29.100, swscale 7.5.100, swresample 4.12.100 |
| libcurl | **Provisioned.** Built from source with MSVC + CMake/Ninja, static, Schannel TLS, installed to `C:\libs\curl`. 8.21.0 era |

The build environment is entered by launching the shell from the **x64 Native Tools Command
Prompt for VS 2026** (`cl`, `nvcc`, and the SDK `LIB`/`INCLUDE` paths must all be present;
`nvcc` shells out to `cl`, so this is not optional).

### 2.1 Dependency provisioning decisions and rationale

**FFmpeg is a prebuilt download, never a source build.** Building FFmpeg on Windows (via
vcpkg or otherwise) is a multi-hour source build with an msys2/nasm toolchain beneath it;
a prebuilt package is a download and an unzip.

Three qualifiers on the package, all of which matter:

1. **`shared`, not `static`.** These builds are cross-compiled with mingw-w64. Their static
   `.a` archives cannot be linked by MSVC; the shared variant's `lib/*.lib` import libraries
   can, because only a plain C ABI crosses the DLL boundary. Verified safe for this tree:
   `src/media/decode/decode.cpp` frees every FFmpeg allocation through `av_free`/`av_freep`/
   `av_frame_free`/`av_packet_free`/`avio_context_free`/`sws_freeContext` — nothing crosses
   the CRT heap — and it feeds data through a custom in-memory `AVIOContext`
   (`decode.cpp:175-186`) rather than a path or `FILE*`, so no CRT file handle crosses either.
2. **`lgpl`, not `gpl`.** Confirmed: `LICENSE.txt` is LGPL v3 and the configure line carries
   `--enable-version3` with `--disable-libx264 --disable-libx265 --disable-libxvid`, and no
   `--enable-gpl` or `--enable-nonfree`. This matches the LGPL shared FFmpeg that distro
   packages provide on Linux and keeps `ninfer.exe` distributable.
3. **Version must match the Linux reference.** `Dockerfile:34-36` pins
   `libavcodec60 / libavformat60 / libavutil58` — Ubuntu 24.04, FFmpeg 6.1, swscale 7. The
   first package tried was FFmpeg 9.0.1 (swscale 10); it clears the `>=60/>=60/>=58/>=7` gate
   easily but spans the swscale rework, and `decode.cpp:294-298` runs every decoded frame
   through `sws_scale` for YUV→RGB. A conversion-rounding difference there changes decoded
   pixels, which changes patch embeddings, which changes vision tokens, which changes
   generated text — silently failing the section 6 cross-platform check while looking like a
   port bug. n6.1.2 is the same release branch as the Linux reference, so the question is
   closed rather than reasoned about.

**libcurl is built from source, because building it is trivial.** With Schannel it has zero
external dependencies and configures and builds in about a minute with MSVC. Static is
curl's CMake default and is what is used here: nothing extra to deploy at runtime.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=C:/libs/curl -DBUILD_SHARED_LIBS=OFF -DBUILD_CURL_EXE=OFF \
  -DCURL_USE_SCHANNEL=ON -DCURL_USE_LIBPSL=OFF
cmake --build build --parallel
cmake --install build
```

The configure summary must report `Enabled SSL backends: Schannel`. Without it,
`CURLOPT_SSL_VERIFYPEER` at `acquire.cpp:235` has nothing behind it and every `https://` media
fetch fails at runtime rather than at build time.

**vcpkg was evaluated and rejected.** It offers no CMake integration benefit that this tree
collects (section 4.1's manual finder path would have been written anyway), while adding a
multi-hour FFmpeg source build and a registry-baseline concept to pin. Two assumptions in the
earlier draft of this plan were also wrong and would have failed at Phase 3: the default
`x64-windows` triplet is **dynamic**, not static, and vcpkg's `curl[ssl]` meta-feature selects
**Schannel** on Windows, not OpenSSL — which would have invalidated the CA-bundle mitigation
that the old section 7 depended on.

### 2.2 Deployment set

Five FFmpeg DLLs must sit beside `ninfer.exe` and `ninfer-serve.exe`:
`avcodec-60.dll`, `avformat-60.dll`, `avutil-58.dll`, `swscale-7.dll`, `swresample-4.dll`.

`swresample` never appears on the link line — it is `Requires.private` of libavformat, hidden
behind the DLL — but it is required at load time. libcurl is static and Schannel-backed, so it
contributes nothing to deploy.

## 3. Fixed design decisions

1. **Single source tree, no Windows-only copy of any hot path.** Platform branches are
   confined to the file list in section 4 and guarded by `_WIN32`. No feature is dropped on
   Windows: every identity, schedule, and option behaves as on Linux. The NVFP4 TMA item in
   4.4.1 is the one open challenge to this decision.
2. **Artifact direct I/O stays unbuffered on Windows** (`FILE_FLAG_NO_BUFFERING`), matching
   the Linux `O_DIRECT` contract. Fallback: if the volume's byte sector size exceeds 4096,
   direct reads degrade to buffered reads (correct, slower load) instead of failing. The
   4096-byte alignment contract in `Reader::direct_io_alignment` is unchanged.
3. **Pinned staging is explicitly 4096-byte aligned.** `PinnedHostBuffer` keeps its API;
   its internals become aligned allocation + `cudaHostRegister` instead of `cudaMallocHost`
   (whose documented alignment is only 256 bytes). The Linux O_DIRECT path currently relies
   on the driver returning page-aligned pinned memory; this makes the guarantee explicit and
   identical on both platforms. **See open decision 9.4** — the scope of this change is under
   review, because it alters a working Linux path for a Windows-only guarantee.
4. **Dependencies are provisioned locally and discovered through pkg-config on both
   platforms.** No vcpkg, no package manager, no CMake platform branch. FFmpeg ships
   relocatable `.pc` files in its prebuilt package (`prefix=${pcfiledir}/../..`); curl
   generates one at install. Both are located by `-DCMAKE_PREFIX_PATH`, which `FindPkgConfig`
   automatically extends into the pkg-config search path. **Verified working** — see 4.1.
5. **`SetConsoleCtrlHandler` on Windows** for serve shutdown (Ctrl+C, Ctrl+Break, console
   close, shutdown). POSIX keeps `std::signal` for SIGINT/SIGTERM.
6. **Windows is documented as a supported platform** in `README.md` (requirements + build
   section) and `AGENTS.md` (local environment), as part of this task.
7. **No behavioral contract changes.** `.ninfer` framing, artifact bytes, sampling, serving
   schemas, and request-log JSONL are byte-identical across platforms. Two platform
   differences are accepted and must be documented rather than hidden: TLS trust roots
   (system CA bundle on Linux, Windows certificate store via Schannel), and the reduced curl
   feature set (no zlib/brotli/zstd, so no compressed transfer encodings; no nghttp2, so
   HTTP/1.1 only). Neither affects media bytes, which are not usefully compressible and are
   fetched over a protocol set pinned to `http,https` at `acquire.cpp:227-228`.

## 4. Work items

### 4.1 CMake (`CMakeLists.txt`, `src/CMakeLists.txt`) — discovery resolved, link unproven

**Dependency discovery is closed; linking is not yet demonstrated.** The unmodified tree
configures on Windows:

```
-- Found PkgConfig: C:/Users/A149/scoop/shims/pkg-config.exe (found version "3.0.5")
-- Checking for modules 'libavformat>=60;libavcodec>=60;libavutil>=58;libswscale>=7'
--   Found libavformat, version 60.16.100
--   Found libavcodec, version 60.31.102
--   Found libavutil, version 58.29.100
--   Found libswscale, version 7.5.100
-- Checking for module 'libcurl>=7.85'
--   Found libcurl, version 8.21.0-DEV
-- Configuring done / Generating done
```

`pkg_check_modules(... IMPORTED_TARGET)` resolves GNU-style `-lavformat` through
`find_library`, which under MSVC finds `avformat.lib` in the `-L` directory and picks the
Windows SDK libraries out of the `LIB` environment variable. The `PkgConfig::FFMPEG` and
`PkgConfig::LIBCURL` targets referenced at `src/CMakeLists.txt:275,282` work as-is. No
`find_path`/`find_library` fallback, no imported-target renaming, and no `PkgConfig` guard is
required. The version gates are enforced identically on both platforms.

**Scope of that claim.** Only discovery and version gating are verified. CMake resolves the
GNU-style `-l` names into real files at generate time, and `CMakeCache.txt` records no
`_LINK_LIBRARIES` entry, so `link.exe` has not yet been handed `avformat.lib` or curl's
`wldap32 / bcrypt / advapi32 / crypt32 / secur32 / ws2_32 / iphlpapi` set — the 2026-08-17
build stopped at the compile step. Treat link resolution as pending until Phase 1 produces the
first executable. The failure mode to watch for: when an `-l` name does not resolve, CMake
passes the raw GNU-style flag through, and `link.exe` does not understand it.

The `Threads` probe reports `CMAKE_HAVE_LIBC_PTHREAD - Failed` and finds no `pthread` library
before concluding `Found Threads: TRUE`. That is normal Windows behavior — CMake falls through
to the Win32 implementation — and needs no action.

Remaining CMake work, none of it dependency-related:

- Add `/utf-8` for MSVC (source and execution charset) so non-ASCII string literals in test
  sources (e.g. `tests/targets/qwen3_6/test_frontend.cpp`,
  `tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp`) keep their UTF-8 bytes instead of
  being decoded in the machine code page; the flag must reach both the `cl` and nvcc host
  paths (`$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>` and `-Xcompiler=/utf-8` for
  `COMPILE_LANGUAGE:CUDA`).
- Add `/bigobj` on the same paths. Not yet observed, but the template-heavy op and target TUs
  are the classic trigger, and discovering it mid-Phase-1 costs a rebuild.
- Define `NOMINMAX` and `WIN32_LEAN_AND_MEAN` for MSVC targets. **Neither speculative nor
  optional**: `windows.h` defines unguarded `min`/`max` macros, and every file that gains it
  under 4.2 and 4.4 already uses something those macros break — `reader.cpp:40,197,235,236`
  (`std::numeric_limits<>::max()`), `src/serve/request_log.cpp:386,551`,
  `src/product/load_progress/load_progress.cpp:132`, and
  `src/product/media_acquire/acquire.cpp:45,194,196`. Set it once at the CMake level rather
  than rediscovering it file by file as each port item lands.
- Link `ws2_32` for `ninfer_serve` on Windows (httplib's `#pragma comment(lib, "ws2_32.lib")`
  covers MSVC; add it explicitly so the dependency is visible). `ninfer_media_acquire` does
  **not** need it added: curl's `.pc` already carries `-lws2_32` along with `wldap32`,
  `bcrypt`, `advapi32`, `crypt32`, `secur32`, and `iphlpapi`.
- **Op-test FP flags do not apply on MSVC.** `tests/CMakeLists.txt:34-38` gates
  `-fno-fast-math` and `-ffp-contract=off` behind `CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"`,
  so on MSVC every `ninfer_add_op_test` host reference compiles under MSVC's *default* FP
  model instead of the pinned one the Linux reference uses, while Phase 5 demands full ctest
  parity. Whether that default actually contracts `a*b+c` into an FMA on x86-64 is **not
  established here and must not be assumed** — probe it at decision time with a trivial TU and
  a disassembly check before choosing a remedy. If contraction is live, `/fp:strict` on the
  op-test host references is the candidate that matches `-ffp-contract=off`. Either way the
  action item stands: decide the MSVC behavior deliberately rather than inheriting it by
  omission, because tolerance failures here will present as port bugs. Note also that clang-cl
  reports `Clang` and would take the existing branch — verify it accepts those GCC-style
  spellings under the MSVC driver rather than warning and ignoring them.
- Nothing else changes: `CMAKE_CUDA_ARCHITECTURES=120a` gate, CUDA >= 13.1 gate, RDC
  settings, `-lineinfo`, Ninja job pool, and the `sm_120a`-only policy all apply as-is on
  Windows — all verified by the passing configure.
- Artifacts are `ninfer.exe` / `ninfer-serve.exe` (CMake appends the suffix); target names
  are unchanged.

### 4.2 Artifact reader (`src/artifact/reader.cpp`) — the central item

Confirmed as the first build failure: `reader.cpp(19): fatal error C1083: Cannot open include
file: 'sys/mman.h'`. `ninfer_artifact` depends only on `ninfer_core`
(`src/CMakeLists.txt:45`) and never reaches `ninfer_ops`, so this work is fully unblocked by
the 4.4.1 issue — iterate with `cmake --build build --target ninfer_artifact`.

`MappedFile` is split into a POSIX implementation and a Win32 implementation behind the same
class:

| Current (POSIX) | Windows |
|---|---|
| `open(O_RDONLY \| O_CLOEXEC \| O_DIRECT)` | two handles: (a) plain `CreateFileW(path, FILE_READ_DATA, FILE_SHARE_READ, ..., OPEN_EXISTING, 0, 0)` for the mapping; (b) `CreateFileW(..., FILE_FLAG_NO_BUFFERING, ...)` for direct reads. `O_CLOEXEC` has no analogue (handles close at process exit). `std::filesystem::path::c_str()` is already `const wchar_t*` on Windows |
| `fstat` + `st_size` | `GetFileSizeEx` |
| `mmap(PROT_READ, MAP_PRIVATE)` / `munmap` | `CreateFileMapping(handle, PAGE_READONLY, sizeHigh, sizeLow, nullptr)` + `MapViewOfFile` / `UnmapViewOfFile` (size passed as 64-bit high/low pair; a 17–22 GB artifact maps fine in a 64-bit process) |
| `pread` with `EINTR` retry | `ReadFile` on the unbuffered handle, offset supplied through an `OVERLAPPED` structure on a **synchronous** handle (positional, no file-pointer mutation, keeps `Reader::read_direct`'s `const` honest). The existing 4096-alignment pre-checks on offset, size, and destination are kept verbatim: they match `FILE_FLAG_NO_BUFFERING` sector rules |
| `errno` | `GetLastError()` → `std::system_error(err, std::generic_category(), ...)`, matching the existing POSIX `generic_category` style |
| `off_t` / `ssize_t` limit checks (`reader.cpp:235-240`) | Neither type exists in MSVC. Re-express against `LONGLONG` / `DWORD` — `ReadFile`'s byte count is a `DWORD`, which is a tighter bound than `ssize_t` and must be enforced |

Behavior notes:

- **Concurrency is not a concern, and the plan should not leave the read mechanism open.**
  `read_direct` has exactly one call site (`src/artifact/materializer.cpp:200`) inside a
  single-threaded slot loop, so the file-pointer race that would force `FILE_FLAG_OVERLAPPED`
  does not arise. The synchronous-handle-plus-`OVERLAPPED`-offset form above is the choice.
  If concurrent loading is ever added, note that a synchronous handle serializes silently
  rather than failing.
- The mapping serves the JSON directory and small resource objects only; tensor payloads are
  always consumed through `read_direct`, exactly as on Linux. Page-cache cost of the mapping
  is negligible (metadata is small).
- At open, query the volume sector size with
  `GetFileInformationByHandleEx(handle, FileStorageInfo, ...)` and read
  `FILE_STORAGE_INFO.LogicalBytesPerSector`. (An earlier draft named a
  `GetVolumeInformation` `lpFileSystemSectorSize` parameter; no such parameter exists.
  `GetDiskFreeSpaceW`'s `lpBytesPerSector` is the alternative but requires a root path, where
  `FileStorageInfo` takes the handle already in hand.) If the sector size is <= 4096, use the
  unbuffered handle; if larger, fall back to the plain handle with buffered `ReadFile` (no
  alignment requirements). Log the fallback at most once. The `direct_io_alignment` public
  constant is unchanged.
- **Short reads at EOF are load-bearing and must be verified explicitly.**
  `materializer.cpp:195-204` requests `align_up(remaining, 4096)` bytes and accepts
  `bytes_read >= min(request, remaining)` — it depends on the final read of a range returning
  a short count rather than an error. Unbuffered `ReadFile` does return the true byte count
  for a trailing partial sector and succeeds, matching `O_DIRECT` `pread`, but this is the one
  place where the port can silently corrupt a load. Make it a stated Phase 1 check.
- Error text keeps the same `system_error` message shape ("open <path>", "direct artifact
  read", ...) so existing error-path tests stay meaningful.

### 4.3 Pinned staging (`src/core/arena.{h,cu}`)

- `PinnedHostBuffer` keeps its public API. Internals:
  - allocate with 4096-byte alignment (`_aligned_malloc` on Windows, `posix_memalign` on
    POSIX),
  - pin with `cudaHostRegister(ptr, size, cudaHostRegisterDefault)`,
  - release with `cudaHostUnregister(ptr)` + the original aligned free (`_aligned_free` /
    `free`) — `cudaFreeHost` is only valid for `cudaMallocHost` memory and must not be used
    on registered user memory (move semantics unchanged; keep the `noexcept` log-on-failure
    shape of the existing `free_pinned` helper).
- Document the 4096-byte alignment guarantee in the header; `Reader::read_direct` keeps its
  runtime check as defense, but the materializer's slot buffers now satisfy it by
  construction on both platforms.
- Audit existing `PinnedHostBuffer` consumers (materializer slots, family program staging)
  for any mapped-pinned assumption — `cudaHostRegister` memory is interchangeable for
  `cudaMemcpyAsync`, but verify no consumer calls `cudaHostGetDevicePointer` or another
  mapped-pinned API (audit result: none exists in the tree; consumers use `cudaMemcpyAsync`
  + event sync only).
- **Scope is under review (open decision 9.4).** Only the materializer's slot buffers
  (`materializer.cpp:181,198`) need the alignment. `PinnedHostBuffer` is also used for small
  per-request ingress buffers (`src/targets/qwen3_6/impl/runtime/program.h:281-289`), which
  would gain page-granular pinning and a slower registration path on both platforms for no
  benefit.

### 4.4 Small product-surface port items

| File | Change |
|---|---|
| `src/product/load_progress/load_progress.cpp` | `isatty(STDERR_FILENO)` → `GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode) != 0` on Windows |
| `src/serve/request_log.cpp` | (a) `getpid()` → `GetCurrentProcessId()` (instance id string format unchanged). (b) **Add `std::ios::binary` to the stream open at line 637.** `output_.open(path_, std::ios::out \| std::ios::app)` is a text-mode stream; MSVC translates `\n` → `\r\n` on write, so every JSONL line would diverge byte-wise from Linux and silently break decision 3.7. This is the only file-writing stream in `src/` and `apps/`, so the fix is that one flag |
| `src/serve/console_log.cpp` | `localtime_r` → `localtime_s(&tm, &time_t)` on Windows (MSVC argument order, verified) |
| `apps/serve/main.cpp` | Windows: `SetConsoleCtrlHandler` for `CTRL_C_EVENT`, `CTRL_BREAK_EVENT`, `CTRL_CLOSE_EVENT`, `CTRL_SHUTDOWN_EVENT` calling `HttpServer::stop()`; POSIX keeps `std::signal(SIGINT/SIGTERM)` |
| `src/product/media_acquire/acquire.cpp` | `#include <winsock2.h>` **then `<ws2tcpip.h>`** on Windows, both before any `windows.h`; one-time `WSAStartup` before `getaddrinfo` (with `WSACleanup` on process exit). `inet_ntop`, `getaddrinfo`/`freeaddrinfo`, `gai_strerror`, and the `IN6_IS_ADDR_*` macros used at `acquire.cpp:100-104,150-166` are declared in **ws2tcpip.h**, not winsock2.h. `ntohl` is in winsock2. The address logic is unchanged. No explicit `ws2_32` link needed — curl's `.pc` supplies it |
| `tests/test_request_log.cpp` | `getpid()` → `GetCurrentProcessId()` on Windows |
| `tests/test_gdn_replay_records.cpp` | `std::aligned_alloc` at line 17 **does not exist in MSVC** — the UCRT deliberately omits it, and the `std::free` deleter at line 15 would be wrong for `_aligned_malloc` anyway. Replace with `operator new(bytes, std::align_val_t{256})` and a matching sized aligned delete, which is portable |

`apps/cli`, `src/text`, `src/runtime`, `src/targets`, `src/core` (besides 4.3), and
`third_party/*` (httplib 0.18.3 is natively Win32-capable; utf8proc is plain C; nlohmann is
header-only) need no changes; the build itself is the verification. **`src/ops` is no longer
on this list** — see 4.4.1.

The `__restrict__` qualifiers throughout `src/ops` appear only in `__device__`/`__global__`
declarations, which nvcc handles before the host compiler sees them; they are not a problem.
Every `asm volatile` in the tree is `__device__` code. No host-side inline assembly and no
`__builtin_` usage exists in the added qwen3.8 fp8/NVFP4 files.

#### 4.4.1 NVFP4 TMA kernels do not compile with MSVC as nvcc's host compiler — **blocker**

`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu` and
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu` both fail:

```
nvfp4_w4a4_tma.cuh(185): error C2719: 'descriptors': formal parameter with
                         requested alignment of 128 won't be aligned
tmpxft_..._nvfp4_w4a4_tma.cudafe1.stub.c(44): error C2719: 'unnamed-parameter': ...
```

Root cause is in CUDA's own header, `cuda.h:3746-3755`:

```c
/* Tensor map descriptor. Requires compiler support for aligning to 128 bytes. */
typedef struct CUtensorMap_st {
    alignas(128)
    cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;
```

`nvfp4_w4a4_tma.cuh:185` passes four of these by value as
`const __grid_constant__ Nvfp4W4a4TmaDescriptors descriptors`. nvcc emits a host-side launch
stub (`cudafe1.stub.c`) declaring the same parameter by value, and MSVC rejects any formal
parameter aligned beyond the stack-slot guarantee. GCC copies over-aligned by-value parameters
into an aligned slot; MSVC refuses.

Two things that do **not** fix it:

- Removing `alignas(128)` from `Nvfp4W4a4TmaDescriptors` (`nvfp4_w4a4_tma.cuh:17`) changes
  nothing — each `CUtensorMap` member carries the alignment from NVIDIA's header. Any by-value
  kernel parameter containing a tensor map hits this under MSVC.
- Skipping the target. `ninfer_nvfp4_tma` is only two files, but `src/CMakeLists.txt:265`
  links it into `ninfer_ops`, and the launchers have **five** caller families:
  `ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu:90`,
  `ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu:44`,
  `ops/linear/nvfp4/nvfp4_w4a4.cu:59`,
  `ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu:64`, and
  `ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:125`. The last two are the plain
  linear and MLP gate-up paths — the largest NVFP4 GEMMs in the tree — which matters for
  weighing option 2's per-launch cost below. Everything depends on it.

Candidate approaches, in the order they should be tried:

1. **Opaque byte parameter.** Pass a `unsigned char[sizeof(Nvfp4W4a4TmaDescriptors)]` wrapper
   by value (alignment 1) and reinterpret inside the kernel. Keeps `__grid_constant__` and
   constant-bank residency, and touches only the parameter declaration. Risk: with the
   declared alignment stripped, nvcc may place the parameter at an offset that does not
   satisfy `CUtensorMap`'s own `alignas(128)`, and reinterpreting storage at such an address
   is undefined behavior — it fails as a silent wrong-address TMA load, not as an error.
   **Validate with a device-side assert on the descriptor address, and assert 128, not 64.**
   Note the two numbers are different requirements: `cuda.h` documents the driver contract as
   "tensorMap address must be aligned to 64 bytes" (lines 24806, 24975, 25163), while the type
   itself declares `alignas(128)` (line 3750). 128 is the conservative bound and the one that
   matches the type; an assert on 64 would pass on an address that still violates it.
   Cheapest to try and preserves the kernel contract.
2. **Pass by pointer.** Stage descriptors in device or `__constant__` memory and take
   `const Nvfp4W4a4TmaDescriptors*`. The device code is already pointer-based —
   `nvfp4_tma_load_2d` at `nvfp4_w4a4_tma.cuh:170` takes `const CUtensorMap*` — so the edit is
   `&descriptors.a_codes` → `&descriptors->a_codes` plus an upload per launch. Certain to
   work, but adds an H2D transfer and gives up what `__grid_constant__` buys on a hot GEMM
   path.

   **The invariant that makes a per-launch upload safe, and which must not be broken.** Every
   TMA route is prefill-only: four call sites gate on `tokens >= 1024 && (tokens %
   kTmaBlockM) == 0` (`nvfp4_attn_input_w4a4.cu:88`, `nvfp4_gdn_input_w4a4.cu:42`,
   `nvfp4_w4a4.cu:57`, `nvfp4_linear_add_w4a4.cu:62`) and the swiglu route is selected only at
   `tokens == kPrimaryT` (1024) in `nvfp4_linear_swiglu_plan.cpp:39`. Decode runs at
   `tokens == 1`, so no TMA path is reachable inside a graph-captured round — the upload is
   eager, outside capture, and therefore harmless. If a TMA path is ever admitted into a
   captured round, a per-launch host descriptor plus a captured H2D node would read stale host
   memory at replay. Option 2 makes that a correctness trap rather than a performance
   question; record the invariant next to the code if this option is chosen.
3. **clang-cl as nvcc's host compiler for that one target.** clang-cl accepts over-aligned
   by-value parameters, and `ninfer_nvfp4_tma` is already an isolated non-RDC archive
   (`src/CMakeLists.txt:51-59`), so a per-target `-ccbin` is structurally plausible. Not
   recommended: nvcc on Windows officially supports only `cl` as host compiler. Note that a
   project using clang-cl for its C++ sources while nvcc still uses `cl` for `.cu` files —
   the common llama.cpp arrangement — would **not** avoid this error, because the failure is
   in the stub nvcc generates for a `.cu` file.

Deferred by decision on 2026-08-17; the reader work in 4.2 proceeds in parallel. This is a
kernel-adjacent change and therefore an exception to section 8 that must be decided
explicitly, not settled at the keyboard.

**This is not a Windows-only change once it is fixed.** Options 1 and 2 both edit kernel
source that Linux compiles and runs, so whichever is chosen pulls NVFP4 correctness into the
Phase 5 Linux regression pass, and option 2's per-launch H2D upload additionally reopens
section 8's "no Linux performance re-baselining". Only option 3 would confine the change to
the Windows build, and it is the one not recommended.

The 4.4 audit that cleared `src/ops` scanned for platform includes, host inline assembly, and
`__builtin_` usage. A host-compiler ABI constraint reached through nvcc's generated stub was
not in that scan's vocabulary — widen it when it is re-run.

### 4.5 Python tooling

- Add a tiny shared helper (e.g. in `tools/bench/`) that resolves `build/apps/ninfer-serve`,
  `build/apps/ninfer`, and `build/bench/ninfer_bench` with a `.exe` suffix on `os.name ==
  "nt"`; use it in `tools/bench/run_serve_corpus.py`, `tools/bench/run_serve_concurrency.py`,
  `tools/bench/run_ninfer_bench_matrix.py`, `tools/smoke/serve_thinking_preservation.py`, and
  any `eval/` driver that spawns a binary. The `os.access(serve, os.X_OK)` checks in the
  corpus runners can stay — Windows CPython ignores `X_OK` entirely, so they pass (vacuously)
  rather than failing.
- **The runners hard-kill the server on Windows.** `run_serve_corpus.py:212` and
  `tools/smoke/serve_thinking_preservation.py:343` call `Popen.terminate()`, which on Windows
  is `TerminateProcess` — no console control event is delivered, so the new
  `SetConsoleCtrlHandler` path never runs and the request log gets no clean close. Phase 4 and
  Phase 5 exit criteria both depend on this. Send `CTRL_BREAK_EVENT` via
  `CREATE_NEW_PROCESS_GROUP` + `os.kill(pid, signal.CTRL_BREAK_EVENT)` on Windows, or accept
  the hard kill and state the consequence in the phase criteria.
- `shlex.quote` / `shlex.join` (`run_ninfer_bench_matrix.py:72`,
  `run_serve_concurrency.py:1040`) emit POSIX quoting. If those strings are recorded in report
  artifacts as reproduction command lines, they are wrong on Windows.
- `tools/artifact/migrate_v1_to_v2.py`: `os.O_DIRECTORY` does not exist on Windows. v1 is a
  dead format, so raise a clear "v1 migration is Linux-only" error on Windows rather than
  porting it.
- Conversion tools (`tools/convert/*`) need no code changes (torch/safetensors/numpy are
  cross-platform); the `.ninfer` output is byte-identical on Windows.
- The `build/apps/...` paths in these helpers assume a single-config generator. Ninja is the
  documented choice; a Visual Studio generator would place binaries under `build/apps/Release/`
  and break them. State the assumption rather than supporting both.

### 4.6 Documentation and contract

- `README.md` Requirements: "64-bit Linux" → "64-bit Linux or Windows"; add the Windows
  dependency set (MSVC + Windows SDK, CUDA Toolkit >= 13.1, CMake >= 3.28, Ninja, pkgconf,
  and the FFmpeg/libcurl provisioning recipe from 2.1 with the LGPL-shared and version-match
  qualifiers spelled out).
- `README.md` Build section: note the Linux and Windows command lines side by side. The
  Windows line carries the dependency prefixes as a cache variable, not an environment
  variable:

  ```
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="C:/libs/ffmpeg-n6.1.2;C:/libs/curl"
  ```

  `FindPkgConfig` extends the pkg-config search path with `<prefix>/lib/pkgconfig` for every
  `CMAKE_PREFIX_PATH` entry (default behavior for `cmake_minimum_required` >= 3.1). Because it
  is a cache variable it survives reconfigures and is visible in `CMakeCache.txt`, unlike an
  exported `PKG_CONFIG_PATH`. If `PKG_CONFIG_PATH` is used directly instead, note that pkgconf
  splits it on **`;`** under Windows, not `:` — a colon-separated value silently finds nothing.
- Note that the build shell must be launched from the x64 Native Tools prompt (see section 2).
- Document the deployment set from 2.2 and the accepted platform differences from decision 3.7
  (TLS trust store, curl feature set).
- `AGENTS.md` local-environment table: add the Windows toolchain row (MSVC/VS path, CUDA 13.3,
  `C:\libs` dependency prefixes) alongside the Linux conventions.
- `docs/README.md` needs no change unless the build section of the README moves.
- Docker section stays Linux-only, unchanged.

## 5. Phases and completion criteria

Each phase ends green on the target machine before the next starts. Phase 5 additionally
requires a Linux regression pass, because 4.3 changes a shared load path.

**Phase 0 — Dependency provisioning — COMPLETE (2026-08-17)**

- FFmpeg n6.1.2 LGPL-shared prebuilt at `C:\libs\ffmpeg-n6.1.2`; libcurl built from source
  (static, Schannel) at `C:\libs\curl`; pkgconf 3.0.5 via scoop. Details and rationale in 2.1.
- Done: the unmodified tree configures with all five version gates satisfied (4.1). This
  additionally retired the "vcpkg vs. dev kits" open decision and the entire projected Windows
  CMake dependency branch.

**Phase 1 — Core engine + artifact reader**

- Re-run the platform-API scan against the current tree, **widened** past includes / host
  inline assembly / `__builtin_` to cover host-compiler ABI constraints surfaced through
  nvcc's generated stubs (the 4.4.1 class).
- Reader port (4.2), pinned-buffer change (4.3), MSVC flags (4.1), test fixes
  (4.4: request-log test, `aligned_alloc`).
- Resolve 4.4.1 (NVFP4 TMA / C2719). `ninfer_ops` cannot build until this lands. `4.2` does
  not depend on it — `cmake --build build --target ninfer_artifact` is the working loop.
- Done when: `ninfer_core`, `ninfer_artifact`, `ninfer_ops`, `ninfer_engine`,
  `ninfer_text` build for MSVC + CUDA 13.3 + `120a`; `ninfer_artifact_reader_test` and
  `ninfer_artifact_materialization_test` pass against a real `.ninfer` artifact; load path
  reads via the unbuffered handle (no fallback), the trailing short read at EOF is confirmed
  to behave as `O_DIRECT` `pread` does, and materialization statistics (`file_bytes`,
  `peak_staging_bytes`, `upload_seconds`) are sane.

**Phase 2 — CLI + serve**

- Product-surface items in 4.4 (load progress, console log, serve main, request log),
  `ninfer.exe` and `ninfer-serve.exe` build.
- Done when: CLI text generation from an artifact completes with correct output; serve starts,
  binds, and handles a non-stream and a streaming OpenAI Chat Completions request; Ctrl+C
  shuts down cleanly with a complete request log; console close is best-effort — Windows
  terminates the process on a short deadline after the `CTRL_CLOSE_EVENT` handler returns, so
  an in-flight request-log line may be truncated; request-log JSONL is written, well-formed,
  and **LF-only** — grep the file for CR bytes, since a text-mode stream would produce CRLF
  and break decision 3.7 without any visible symptom (see the 4.4 row).

**Phase 3 — Media path**

- `ninfer_media_decode` (FFmpeg) and `ninfer_media_acquire` (curl + Winsock) build; the five
  DLLs from 2.2 are deployed beside the executables.
- Done when: a real image prompt and a real video prompt produce the same output as the Linux
  reference for the same artifact and seed; an `http://` media URL fetch works (including the
  private-address guard); an `https://` fetch succeeds against the Windows certificate store.

**Phase 4 — Python tooling + docs**

- 4.5 and 4.6 land.
- Done when: `run_serve_corpus.py` smoke on Windows drives `ninfer-serve.exe` end to end and
  shuts it down through a console control event (or the hard-kill consequence is stated);
  `git diff --check` clean; README/AGENTS.md consistent with the delivered commands.

**Phase 5 — Verification and Windows performance row**

- Full `ctest` on Windows (same test selection as Linux; GPU-limited tests use SKIP_RETURN
  code 77 as today). The op-test FP-flag gap in 4.1 must be resolved before this is meaningful.
- Single-request benchmark parity check: rerun the published single-request serving method
  against `ninfer-serve.exe` — the same runner (`run_serve_corpus.py`), fixtures, seeds, and
  sampling profile documented in `docs/performance.md` (MTP0 = the four Long NIAH prompts;
  MTP3 = the speculative-decode corpus of three long-reasoning and twelve cross-scenario
  fixtures) — on the `qwen3.6-27b/groupwise-int` artifact (the registered
  `qwen3.8-27b/groupwise-int` profile is outside the published benchmark campaign), reported
  in `docs/performance.md` as a new Windows/WDDM row next to the Linux rows under the same
  method. Numbers are added only if the run is clean, and a row covering fewer fixtures than
  the full published corpus must state its coverage. `ninfer_bench` CLI numbers are a
  different measurement level and must not substitute for the serving rows.
- Linux regression pass. 4.3 changes a shared load path; 4.1 no longer does. **4.4.1 does
  too, unless it is resolved by option 3** — options 1 and 2 edit NVFP4 kernel source that
  Linux compiles and runs. Minimum scope: clean Linux build, `ninfer_artifact_reader_test` +
  `ninfer_artifact_materialization_test`, and a load run whose `file_bytes` /
  `peak_staging_bytes` / `upload_seconds` match the pre-change baseline. Extended scope when
  4.4.1 lands as option 1 or 2: NVFP4 op-test correctness on Linux across all five caller
  families listed in 4.4.1 — attention input, GDN input, linear, linear-add, and
  linear-swiglu — plus an NVFP4 performance check if option 2 was chosen, weighted toward the
  linear and linear-swiglu paths where the per-launch H2D upload lands hardest.
- Done when: all ctest targets pass or their platform limitation is stated; the performance
  row (or a stated reason it could not be run) exists; the Linux regression pass is green.

## 6. Verification principles

- Artifact correctness: the same `.ninfer` file must produce the same generated text at
  temperature 0 (greedy) on both platforms for a fixed prompt and seed. This is the
  end-to-end numerical check; per-op oracles are unchanged by the port.
- Media correctness depends on decoder version parity, not just on code. The Windows FFmpeg
  must stay on the same release branch as the Linux reference (`Dockerfile:34-36`); see 2.1.
  A decoder-version drift would fail the check above while looking like a port defect.
- Load path: verify the unbuffered read is actually used (log or trace the sector-size
  branch once), and that `file_bytes` / `h2d_bytes` / `peak_staging_bytes` match the Linux
  formulas. (These are the actual `MaterializationStats` field names,
  `src/artifact/materializer.h:21-30`; there is no `artifact_bytes_read`.)
- Serving: the OpenAI/Anthropic schema tests and request-log JSONL shape are platform
  contracts — run them, don't reinterpret them.
- WDDM is not a correctness risk for pinned H2D or CUDA Graphs (no mapped pinned memory in
  the codebase), but load-time H2D throughput will differ; that difference belongs in the
  Windows performance row, not in the design.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **MATERIALIZED** — MSVC host-compiler constraint in a never-compiled-on-Windows file | 4.4.1. The earlier stance ("no such patterns found in audit") was wrong; the other half held — it surfaced as a direct compile error, not a latent bug. The audit's vocabulary is widened in Phase 1 |
| `cudaHostRegister` behaves differently under WDDM than `cudaMallocHost` (registration failure, or H2D bandwidth) | Phase 1 measures load time on the real 17 GB artifact; if registration is problematic, fall back to `cudaMallocHost` + a runtime alignment check that throws the existing "not 4096-byte aligned" error only when a driver returns unaligned memory. Open decision 9.4 may narrow the change to the materializer slots regardless |
| Volume sector size > 4096 breaks `FILE_FLAG_NO_BUFFERING` | Sector-size probe at open via `GetFileInformationByHandleEx(FileStorageInfo)` with buffered fallback (decision 2, 4.2) |
| Trailing short read at EOF behaves differently from `O_DIRECT` `pread` | Explicit Phase 1 check; `materializer.cpp:195-204` depends on it |
| Op tests run without FP-contraction control on MSVC, producing tolerance failures that look like port bugs | 4.1; resolve before Phase 5's full `ctest` |
| Windows FFmpeg drifts from the Linux reference version, silently changing decoded pixels | 2.1 and section 6; pinned to the 6.1 branch matching `Dockerfile:34-36` |
| GPU contention on the target machine (other workloads visible at exploration time) | All GPU-bound verification runs on an idle GPU; benchmark runs wait for a quiet window |

Retired risks: vcpkg FFmpeg/curl build failures (vcpkg not used); curl CA-bundle discovery
(Schannel uses the Windows certificate store, so `CURL_CA_BUNDLE`/`SSL_CERT_FILE` are not
involved — this is a documented platform difference under decision 3.7, not a risk).

## 8. Explicitly not doing

- No WSL wrapper, no Cygwin/MSYS runtime dependency, no MinGW toolchain. (The FFmpeg DLLs are
  mingw-built upstream, but nothing mingw is required at build or run time; only a C ABI
  crosses that boundary — see 2.1.)
- No Windows service, autostart, installer, or packaging.
- No changes to kernels, schedules, artifact format, public Engine API, or serving schemas —
  **with one open exception**, 4.4.1, which cannot be avoided and must be decided explicitly.
- No Linux performance re-baselining; existing published numbers are untouched. Revisit only
  if 4.4.1 is resolved by option 2 (pointer parameter), which adds an H2D upload per NVFP4
  launch on both platforms.
- No port of `tools/hbm_bandwidth_probe.cu`-style maintainer probes: `tools/` is not built by
  any CMake target and no default target needs them.
- No port of the opt-in `bench/` targets (`NINFER_BUILD_BENCHMARKS`, off by default): they are
  not part of the default build or of the port verification, same stance as the maintainer
  probes.

## 9. Open decisions

1. ~~**Dependency provisioning**: vcpkg vs. pinned prebuilt dev kits.~~ **Resolved
   2026-08-17**: prebuilt LGPL-shared FFmpeg + source-built static Schannel curl, discovered
   through pkg-config on both platforms, no CMake branch. See 2.1 and 4.1.
2. **NVFP4 TMA / C2719 approach** (4.4.1): opaque byte parameter, pointer parameter, or
   clang-cl host compiler. Blocks `ninfer_ops` and therefore all of Phase 1's build criteria.
   Deferred 2026-08-17; must be decided before Phase 1 closes.
3. **Verification artifact**: download `qwen3_6_27b.ninfer` from the published HF artifact,
   or run `tools/convert` from a local BF16 checkpoint on this machine. Download is cheaper;
   conversion additionally proves the Windows conversion toolchain. The choice is scoped to
   the `qwen3.6-27b/groupwise-int` artifact that Phase 5 measures, so the conversion option
   exercises `tools/convert/qwen3_6_27b` plus the shared `tools/convert/qwen3_6` and
   `tools/convert/common` family code. It does **not** touch `tools/convert/qwen3_8_27b`,
   which is a separate per-target package that converting this artifact never reaches.
4. **`PinnedHostBuffer` change scope** (decision 3.3, 4.3): apply `cudaHostRegister` to every
   consumer on both platforms, or introduce an explicitly-aligned staging type used only by
   the materializer's direct-I/O slots. The narrow option leaves the working Linux allocation
   path untouched and shrinks the Phase 5 regression surface; the broad option makes one
   guarantee hold everywhere. Decide before 4.3 is implemented.
5. **Console and command-line text encoding**: `apps/cli/main.cpp:234` takes ANSI-encoded
   `char** argv`, so non-ASCII prompts passed on the command line are lossy, and UTF-8 model
   output written to `std::cout`/`std::cerr` renders as mojibake unless the console code page
   is UTF-8. `/utf-8` (4.1) fixes source literals only, not runtime I/O. Either add
   `SetConsoleOutputCP(CP_UTF8)`/`SetConsoleCP(CP_UTF8)` plus a `GetCommandLineW`-based argv
   path, or scope non-ASCII console input/rendering out explicitly. Decision 3.7's
   byte-identical claim holds for files and pipes regardless.
6. **Documentation timing**: land the README/AGENTS.md Windows sections in Phase 4 with the
   working build (recommended, matches the "active authority" lifecycle), or defer docs until
   Phase 5 performance numbers exist.

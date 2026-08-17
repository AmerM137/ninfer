# Windows Port Plan (2026-08-15)

Status: temporary working plan. This document records the port design, work items, and
completion criteria for making NInfer build and run natively on 64-bit Windows. It is removed
when the port lands and its stable content is integrated into `README.md` and `AGENTS.md`.
Reviewed against the repository on 2026-08-17: the qwen3.8-27b identities, the fp8/NVFP4 op
wave, and the post-plan behavior fixes are accounted for in the text below.

## 1. Objective and scope

Deliver: `ninfer.exe` and `ninfer-serve.exe` built from this tree on 64-bit Windows with MSVC
and the CUDA Toolkit, executing the full product surface (Text, Vision, MTP/DFlash, prefix
reuse, CLI, OpenAI/Anthropic serving) from the same `.ninfer` artifacts, compiled for `sm_120a`
as today.

- The product contract in `AGENTS.md` gains Windows as a supported host platform alongside
  64-bit Linux. No target identity, kernel, artifact format, or scheduling change: the port is
  host-platform work only.
- Published performance numbers remain Linux/RTX 5090 results. Windows receives its own
  measured row at the end (WDDM), not a reinterpretation of the existing numbers.
- Docker stays Linux-only. The WSL/WSL2 path is explicitly out of scope: this is a native
  Windows port.
- Out of scope: TCC mode, multi-GPU, GPU-installed drivers, Windows packaging/installers.

## 2. Verified environment (target machine)

| Item | State |
|---|---|
| GPU | RTX 5090 32 GB, WDDM, driver 610.88 (UMD 13.3) |
| CUDA | 13.3 toolkit at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` (satisfies the existing >= 13.1 gate) |
| Host compiler | MSVC 14.51 via Visual Studio 2026 (Community) at `C:\Program Files\Microsoft Visual Studio\18`; clang 22 / clang-cl available as fallback |
| Build tools | CMake 4.3.2, Ninja (scoop) |
| Python | 3.14.6 (mamba env) |
| Python packages | `torch`/`safetensors`/`numpy` for the conversion tooling — confirm they are installed in this env before choosing the conversion option in section 9 (3.14.6 is fine for the tooling; the 3.11 convention in `AGENTS.md` is not a hard floor) |
| Missing | FFmpeg development libraries, libcurl development libraries, pkg-config (not needed) |

## 3. Fixed design decisions

1. **Single source tree, no Windows-only copy of any hot path.** Platform branches are
   confined to the file list in section 4 and guarded by `_WIN32`. No feature is dropped on
   Windows: every identity, schedule, and option behaves as on Linux.
2. **Artifact direct I/O stays unbuffered on Windows** (`FILE_FLAG_NO_BUFFERING`), matching
   the Linux `O_DIRECT` contract. Fallback: if the volume's byte sector size exceeds 4096,
   direct reads degrade to buffered reads (correct, slower load) instead of failing. The
   4096-byte alignment contract in `Reader::direct_io_alignment` is unchanged.
3. **Pinned staging is explicitly 4096-byte aligned.** `PinnedHostBuffer` keeps its API;
   its internals become aligned allocation + `cudaHostRegister` instead of `cudaMallocHost`
   (whose documented alignment is only 256 bytes). The Linux O_DIRECT path currently relies
   on the driver returning page-aligned pinned memory; this makes the guarantee explicit and
   identical on both platforms.
4. **Dependencies via vcpkg** on Windows (pending confirmation, see section 9): one toolchain
   file provides FFmpeg (avformat/avcodec/avutil/swscale) and libcurl with OpenSSL. Linux
   keeps pkg-config; the two paths are symmetric CMake branches, no pkg-config on Windows.
   Pin the triplet and exact port/feature set in Phase 0: the vcpkg default `x64-windows`
   triplet builds FFmpeg, curl, and OpenSSL as **static** libraries (larger `.exe`, no
   runtime DLLs), and the `curl` port's default `ssl` feature uses the OpenSSL SSL backend —
   the section 7 CA-bundle mitigation depends on that.
5. **`SetConsoleCtrlHandler` on Windows** for serve shutdown (Ctrl+C, Ctrl+Break, console
   close, shutdown). POSIX keeps `std::signal` for SIGINT/SIGTERM.
6. **Windows is documented as a supported platform** in `README.md` (requirements + build
   section) and `AGENTS.md` (local environment), as part of this task.
7. **No behavioral contract changes.** `.ninfer` framing, artifact bytes, sampling, serving
   schemas, and request-log JSONL are byte-identical across platforms.

## 4. Work items

### 4.1 CMake (`CMakeLists.txt`, `src/CMakeLists.txt`)

- Guard the `PkgConfig`/`pkg_check_modules` block on non-Windows.
- Windows branch:
  - vcpkg toolchain: `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`
    (portable: vcpkg provides `avformat`, `avcodec`, `avutil`, `swscale`, `curl[ssl]`).
  - For manual dev kits, fall back to `find_path`/`find_library` for the four FFmpeg libraries
    and `libcurl`, building an imported interface target `NINFER::ffmpeg` and consuming
    `CURL::libcurl` (vcpkg provides both natively; the `find_*` path is the escape hatch).
  - Replace `PkgConfig::FFMPEG` / `PkgConfig::LIBCURL` link references with the imported
    targets on both platforms (Linux side: wrap the existing pkg-config imported targets in
    the same names, so the target lists do not branch).
- Link `ws2_32` for `ninfer_media_acquire` and `ninfer_serve` on Windows (httplib's
  `#pragma comment(lib, "ws2_32.lib")` covers MSVC; add it explicitly so clang-cl works).
- Add `/utf-8` for MSVC (source and execution charset) so non-ASCII string literals in test
  sources (e.g. `tests/targets/qwen3_6/test_frontend.cpp`,
  `tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp`) keep their UTF-8 bytes instead of
  being decoded in the machine code page; the flag must reach both the `cl` and nvcc host
  paths (`$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>` and `-Xcompiler=/utf-8` for
  `COMPILE_LANGUAGE:CUDA`).
- Nothing else changes: `CMAKE_CUDA_ARCHITECTURES=120a` gate, CUDA >= 13.1 gate, RDC
  settings, `-lineinfo`, Ninja job pool, and the `sm_120a`-only policy all apply as-is on
  Windows.
- Artifacts are `ninfer.exe` / `ninfer-serve.exe` (CMake appends the suffix); target names
  are unchanged.

### 4.2 Artifact reader (`src/artifact/reader.cpp`) — the central item

`MappedFile` is split into a POSIX implementation and a Win32 implementation behind the same
class:

| Current (POSIX) | Windows |
|---|---|
| `open(O_RDONLY \| O_CLOEXEC \| O_DIRECT)` | two handles: (a) plain `CreateFileW(path, FILE_READ_DATA, FILE_SHARE_READ, ..., OPEN_EXISTING, 0, 0)` for the mapping; (b) `CreateFileW(..., FILE_FLAG_NO_BUFFERING, ...)` for direct reads. `O_CLOEXEC` has no analogue (handles close at process exit). |
| `fstat` + `st_size` | `GetFileSizeEx` |
| `mmap(PROT_READ, MAP_PRIVATE)` / `munmap` | `CreateFileMapping(handle, PAGE_READONLY, sizeHigh, sizeLow, nullptr)` + `MapViewOfFile` / `UnmapViewOfFile` (size passed as 64-bit high/low pair; a 17–22 GB artifact maps fine in a 64-bit process) |
| `pread` with `EINTR` retry | `ReadFile` on the unbuffered handle — synchronous via `SetFilePointerEx`, or `FILE_FLAG_OVERLAPPED` + `OVERLAPPED` (offset in the structure) + `GetOverlappedResult` (the canonical pattern; `ReadFileEx` requires an overlapped handle and is not a fit) — and the existing 4096-alignment pre-checks on offset, size, and destination kept verbatim: they match `FILE_FLAG_NO_BUFFERING` sector rules |
| `errno` | `GetLastError()` → `std::system_error(err, std::generic_category(), ...)`, matching the existing POSIX `generic_category` style |

Behavior notes:

- The mapping serves the JSON directory and small resource objects only; tensor payloads are
  always consumed through `read_direct`, exactly as on Linux. Page-cache cost of the mapping
  is negligible (metadata is small).
- At open, query the volume sector size (`GetVolumeInformation`
  `lpFileSystemSectorSize`). If it is <= 4096, use the unbuffered handle; if larger, fall
  back to the plain handle with buffered `ReadFile` (no alignment requirements). Log the
  fallback at most once. The `direct_io_alignment` public constant is unchanged.
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

### 4.4 Small product-surface port items

| File | Change |
|---|---|
| `src/product/load_progress/load_progress.cpp` | `isatty(STDERR_FILENO)` → `GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode) != 0` on Windows |
| `src/serve/request_log.cpp` | `getpid()` → `GetCurrentProcessId()` (instance id string format unchanged) |
| `src/serve/console_log.cpp` | `localtime_r` → `localtime_s(&tm, &time_t)` on Windows |
| `apps/serve/main.cpp` | Windows: `SetConsoleCtrlHandler` for `CTRL_C_EVENT`, `CTRL_BREAK_EVENT`, `CTRL_CLOSE_EVENT`, `CTRL_SHUTDOWN_EVENT` calling `HttpServer::stop()`; POSIX keeps `std::signal(SIGINT/SIGTERM)` |
| `src/product/media_acquire/acquire.cpp` | `#include <winsock2.h>` on Windows; one-time `WSAStartup` before `getaddrinfo` (with `WSACleanup` on process exit); link `ws2_32`. `inet_ntop`, `ntohl`, `IN6_IS_ADDR_*`, `gai_strerror` are all in winsock2 — the address logic is unchanged |
| `tests/test_request_log.cpp` | `getpid()` → `GetCurrentProcessId()` on Windows |

`apps/cli`, `src/text`, `src/runtime`, `src/targets`, `src/ops`, `src/core` (besides
4.3), `third_party/*` (httplib 0.18.3 is natively Win32-capable; utf8proc is plain C;
nlohmann is header-only) need no changes; the build itself is the verification.

Known MSVC compatibility already checked and clean: integer `std::from_chars`,
`std::filesystem` with 20+ GB paths, `std::aligned_alloc(256, multiple-of-256)` in
`tests/test_gdn_replay_records.cpp` (size is aligned by construction), C++20 features used.
Re-verified on 2026-08-17 across the qwen3.8 fp8/NVFP4 op wave: no new platform includes, no
host-side inline assembly, and no `__builtin_` usage in the added files; every `asm volatile`
in the tree is `__device__` code. The tree moves while this plan runs, so Phase 1 re-runs the
scan before the first Windows build.

### 4.5 Python tooling

- Add a tiny shared helper (e.g. in `tools/bench/`) that resolves `build/apps/ninfer-serve`,
  `build/apps/ninfer`, and `build/bench/ninfer_bench` with a `.exe` suffix on `os.name ==
  "nt"`; use it in `tools/bench/run_serve_corpus.py`, `tools/bench/run_serve_concurrency.py`,
  `tools/bench/run_ninfer_bench_matrix.py`, `tools/smoke/serve_thinking_preservation.py`, and
  any `eval/` driver that spawns a binary. The `os.access(serve, os.X_OK)` checks in the
  corpus runners pass for `.exe` on Windows (CPython treats `.exe` as executable), so they
  can stay as-is.
- `tools/artifact/migrate_v1_to_v2.py`: `os.O_DIRECTORY` does not exist on Windows. v1 is a
  dead format, so raise a clear "v1 migration is Linux-only" error on Windows rather than
  porting it.
- Conversion tools (`tools/convert/*`) need no code changes (torch/safetensors/numpy are
  cross-platform); the `.ninfer` output is byte-identical on Windows.

### 4.6 Documentation and contract

- `README.md` Requirements: "64-bit Linux" → "64-bit Linux or Windows"; add the Windows
  dependency set (MSVC + Windows SDK, CUDA Toolkit >= 13.1, CMake >= 3.28, FFmpeg dev
  libraries, libcurl dev libraries; vcpkg command line) and a Windows build section with
  `.exe` output names.
- `README.md` Build section: note the Linux and Windows command lines side by side.
- `AGENTS.md` local-environment table: add the Windows toolchain row (MSVC/VS path, CUDA 13.3,
  vcpkg root) alongside the Linux conventions.
- `docs/README.md` needs no change unless the build section of the README moves.
- Docker section stays Linux-only, unchanged.

## 5. Phases and completion criteria

Each phase ends green on the target machine before the next starts. Phase 5 additionally
requires a Linux regression pass, because 4.1 and 4.3 change shared build and load paths.

**Phase 0 — Dependency provisioning (external, blocking)**

- vcpkg installed and bootstrapped; `vcpkg install ffmpeg curl` (default features; exact
  feature set pinned when executed) produces the FFmpeg + curl dev libraries.
- Done when: a trivial CMake project on this machine consumes `CURL::libcurl` and the four
  FFmpeg libraries through the vcpkg toolchain.

**Phase 1 — Core engine + artifact reader**

- Re-run the platform-API scan (POSIX includes, host inline assembly, `__builtin_`) against
  the current tree before the first Windows build (the 4.4 audit predates the qwen3.8 op
  wave).
- CMake platform branch (4.1), reader port (4.2), pinned-buffer change (4.3), test fixes
  (4.4: request-log test).
- Done when: `ninfer_core`, `ninfer_artifact`, `ninfer_ops`, `ninfer_engine`,
  `ninfer_text` build for MSVC + CUDA 13.3 + `120a`; `ninfer_artifact_reader_test` and
  `ninfer_artifact_materialization_test` pass against a real `.ninfer` artifact; load path
  reads via the unbuffered handle (no fallback) and materialization statistics
  (`file_bytes`, `peak_staging_bytes`, `upload_seconds`) are sane.

**Phase 2 — CLI + serve**

- Product-surface items in 4.4 (load progress, console log, serve main, request log),
  `ninfer.exe` and `ninfer-serve.exe` build.
- Done when: CLI text generation from an artifact completes with correct output; serve starts,
  binds, and handles a non-stream and a streaming OpenAI Chat Completions request; Ctrl+C
  shuts down cleanly with a complete request log; console close is best-effort — Windows
  terminates the process on a short deadline after the `CTRL_CLOSE_EVENT` handler returns, so
  an in-flight request-log line may be truncated; request-log JSONL is written and
  well-formed.

**Phase 3 — Media path**

- `ninfer_media_decode` (FFmpeg) and `ninfer_media_acquire` (curl + Winsock) build. With the
  static vcpkg triplet (the default) there are no runtime DLLs to deploy; if the pinned
  triplet produces shared libraries, deploy the FFmpeg/curl/OpenSSL DLLs next to the
  executables or on PATH.
- Done when: a real image prompt and a real video prompt produce the same output as the Linux
  reference for the same artifact and seed; an `http://` media URL fetch works (including the
  private-address guard).

**Phase 4 — Python tooling + docs**

- 4.5 and 4.6 land.
- Done when: `run_serve_corpus.py` smoke on Windows drives `ninfer-serve.exe` end to end;
  `git diff --check` clean; README/AGENTS.md consistent with the delivered commands.

**Phase 5 — Verification and Windows performance row**

- Full `ctest` on Windows (same test selection as Linux; GPU-limited tests use SKIP_RETURN
  code 77 as today).
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
- Linux regression pass (4.1 and 4.3 change shared paths): clean Linux build,
  `ninfer_artifact_reader_test` + `ninfer_artifact_materialization_test`, and a load run
  whose `file_bytes` / `peak_staging_bytes` / `upload_seconds` match the pre-change baseline.
- Done when: all ctest targets pass or their platform limitation is stated; the performance
  row (or a stated reason it could not be run) exists; the Linux regression pass is green.

## 6. Verification principles

- Artifact correctness: the same `.ninfer` file must produce the same generated text at
  temperature 0 (greedy) on both platforms for a fixed prompt and seed. This is the
  end-to-end numerical check; per-op oracles are unchanged by the port.
- Load path: verify the unbuffered read is actually used (log or trace the sector-size
  branch once), and that `artifact_bytes_read` / `peak_staging_bytes` match the Linux
  formulas.
- Serving: the OpenAI/Anthropic schema tests and request-log JSONL shape are platform
  contracts — run them, don't reinterpret them.
- WDDM is not a correctness risk for pinned H2D or CUDA Graphs (no mapped pinned memory in
  the codebase), but load-time H2D throughput will differ; that difference belongs in the
  Windows performance row, not in the design.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `cudaHostRegister` behaves differently under WDDM than `cudaMallocHost` (registration failure, or H2D bandwidth) | Phase 1 measures load time on the real 17 GB artifact; if registration is problematic, fall back to `cudaMallocHost` + a runtime alignment check that throws the existing "not 4096-byte aligned" error only when a driver returns unaligned memory |
| Volume sector size > 4096 breaks `FILE_FLAG_NO_BUFFERING` | Sector-size probe at open with buffered fallback (decision 2) |
| vcpkg FFmpeg/curl build failures or version mismatch (need avformat >= 60 etc.) | Pin the vcpkg port versions; the `find_*` escape hatch in 4.1 covers a prebuilt dev kit |
| curl CA-bundle discovery for `https://` media fetches on Windows | vcpkg OpenSSL ships `cert.pem`; document setting `SSL_CERT_FILE` (or `CURL_CA_BUNDLE`) once in the README Windows section |
| MSVC C++20 conformance surprise in a never-compiled-on-Windows file | The build is the test; no such patterns found in audit, and any failure is a direct compile error, not a latent bug |
| GPU contention on the target machine (other workloads visible at exploration time) | All GPU-bound verification runs on an idle GPU; benchmark runs wait for a quiet window |

## 8. Explicitly not doing

- No WSL wrapper, no Cygwin/MSYS runtime dependency, no MinGW toolchain.
- No Windows service, autostart, installer, or packaging.
- No changes to kernels, schedules, artifact format, public Engine API, or serving schemas.
- No Linux performance re-baselining; existing published numbers are untouched.
- No port of `tools/hbm_bandwidth_probe.cu`-style maintainer probes: `tools/` is not built by
  any CMake target and no default target needs them.
- No port of the opt-in `bench/` targets (`NINFER_BUILD_BENCHMARKS`, off by default): they are
  not part of the default build or of the port verification, same stance as the maintainer
  probes.

## 9. Open decisions (resolve before Phase 0)

1. **Dependency provisioning**: vcpkg (recommended) vs. pinned prebuilt dev kits (BtbN
   FFmpeg + curl Windows dev kits). Affects 4.1 and the README build section.
2. **Verification artifact**: download `qwen3_6_27b.ninfer` from the published HF artifact,
   or run `tools/convert` from a local BF16 checkpoint on this machine. Download is cheaper;
   conversion additionally proves the Windows conversion toolchain. The choice is scoped to
   the `qwen3.6-27b/groupwise-int` artifact that Phase 5 measures; the conversion option
   additionally exercises the qwen3.8-27b converter, which postdates this plan.
3. **Documentation timing**: land the README/AGENTS.md Windows sections in Phase 4 with the
   working build (recommended, matches the "active authority" lifecycle), or defer docs until
   Phase 5 performance numbers exist.

# Windows Port — Review Findings and Remaining Work (2026-08-17)

Status: temporary working document, companion to
`docs/windows-port/plan.md`. It records the review of the unstaged
working tree on `docs/windows-port-bringup` (the Phase 1/2 code implementation) against that
plan: bugs to fix, decisions the implementation made that the plan left open, and the
verification items still owed. Remove it when the items are resolved and folded into the plan
doc or the code. File/line references are against the working tree as of this date.

All "verified on the target machine" claims below were established with small MSVC-built
probe programs (cl 19.51.36256, x64) run on this machine on 2026-08-17, or — for the build
and test claims in section 4 — by building and running the existing `build/` tree directly.

Headline state, ahead of what the plan doc records: **the full build is green** —
`ninfer.exe`, `ninfer-serve.exe`, `ninfer_ops`, and `ninfer_nvfp4_tma` all build from the
current working tree, the five FFmpeg DLLs are staged beside the executables, and **ctest
passes 83/89** (5 skips are the artifact-gated `_real`/load-plan tests — resolvable now that
`qwen3_6_27b.ninfer` exists at `C:\Users\A149\models\ninfer\Qwen3.6-27B-Ninfer\`, see
decision 9.3; 1 failure was a pre-existing test-fixture dependency, section 1.3).

**Fix status (2026-08-17, same day):** every actionable item below has been applied to the
working tree — 1.1 (`GetConsoleMode` output pointer), 1.2 (`CreateFileMappingW` `NULL`
sentinel), 1.3 (frontend test: `NINFER_QWEN3_6_27B_HF` env override + skip-77 + top-level
catch; FFmpeg DLLs now staged into `build/tests/` and `build/apps/` by CMake at configure
time), 3.1 (`WSAStartup` throws `Error` instead of `std::exit`), 3.2 (TMA trap printf gated
to one thread per block), 3.4 (LF-only byte assertion in `test_request_log.cpp`), and the
2.2 comment now records the disassembly probe result. 2.3 gained a regression test
(`tests/test_media_acquire_path.cpp`, covering inside/nested/`..hidden`/outside/dot-dot
escape). Still open by design: the section 2 ratification decisions (9.4 scope, splitting
the containment fix into its own commit), and everything in "Still owed".

## 1. Defects to fix before the port lands

### 1.1 `GetConsoleMode(handle, nullptr)` crashes on a real console — `load_progress.cpp:71`

```cpp
const bool interactive =
    ::GetConsoleMode(::GetStdHandle(STD_ERROR_HANDLE), /*lpMode=*/nullptr) != 0;
```

`lpMode` is not optional. **Verified on the target machine**: `GetConsoleMode(h, nullptr)`
on a real console handle (opened via `CONOUT$`) raises an access violation — the process
dies, it does not return `FALSE`. With a redirected (pipe) stderr the call fails early on
the handle check *before* dereferencing `lpMode`, so it "works" and returns `FALSE`. The
failure polarity is the worst one: `ninfer.exe` / `ninfer-serve.exe` crash at the first
progress line precisely in interactive console use, and pass silently in any piped/CI run.

Fix shape (verified working in the same probe — returns `TRUE`, mode `0x3`):

```cpp
DWORD mode = 0;
const bool interactive =
    ::GetConsoleMode(::GetStdHandle(STD_ERROR_HANDLE), &mode) != 0;
```

### 1.2 `CreateFileMappingW` failure checked against the wrong sentinel — `reader.cpp:226-229`

`CreateFileMappingW` returns **`NULL`** on failure, not `INVALID_HANDLE_VALUE` (unlike
`CreateFileW`; documented on its API page). Consequences of the current
`mapping_handle_ == INVALID_HANDLE_VALUE` check:

- A real mapping failure slips past the check and is only caught one call later when
  `MapViewOfFile(NULL, ...)` fails — the thrown `system_error` then carries
  `MapViewOfFile`'s error (`ERROR_INVALID_HANDLE`) instead of the true cause.
- `close_all()` uses `INVALID_HANDLE_VALUE` as the "not open" sentinel for
  `mapping_handle_`, so on that path it calls `CloseHandle(NULL)` (harmless failed call,
  but wrong).

Fix: check `mapping_handle_ == nullptr` after `CreateFileMappingW`, and use `nullptr` as
the not-open sentinel for `mapping_handle_` (keep `INVALID_HANDLE_VALUE` for the two
`CreateFileW` handles, whose convention that is).

### 1.3 `ninfer_qwen3_6_frontend_test` fails with `0xc0000409` — a fixture dependency the port surfaces, not a port bug

The only ctest failure on Windows. Diagnosed by temporarily wrapping the test's `main` in a
try/catch (reverted): the test throws
`failed to open test resource: /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json`.

`tests/targets/qwen3_6/test_frontend.cpp:93-97` reads the official HF checkpoint's
`tokenizer.json` / `tokenizer_config.json` / `generation_config.json` from a **hardcoded
absolute path in the maintainer's Linux home directory** — no environment override, no
skip-77 when absent. The test can never run on any machine without that exact path; it is
machine-specific, not Windows-specific. Two compounding factors made it look like a port bug:

- `main()` has no top-level catch, so the throw reaches `std::terminate`, which MSVC
  implements as fail-fast: exit code `0xc0000409` (`STATUS_STACK_BUFFER_OVERRUN`) with
  **zero output** — buffered stdout/stderr is discarded. On Linux the same miss would at
  least print `terminate called after throwing ... what(): failed to open test resource: ...`.
- The exe links the media path, so it also needs the FFmpeg DLLs at load time; `build/tests/`
  had none (only `build/apps/` is populated by the build). The five DLLs were copied into
  `build/tests/` during this review to unblock the diagnosis — the DLL deployment for
  media-linked *test* executables needs a real answer (a CMake copy step or a documented
  PATH requirement), or every media-linked test dies at load on a clean Windows checkout.

Recommended fixes, all cheap: (a) route the three files through an environment variable with
skip-77 when unset/missing, matching the `NINFER_QWEN3_6_27B_WEIGHTS` pattern in
`test_load_plan.cpp`; (b) add a top-level try/catch in the test's `main` so a missing
resource fails with a message instead of an opaque fail-fast; (c) extend the apps-directory
DLL copy to the tests directory. The same hardcoded path also appears in
`tests/targets/qwen3_6_27b/test_convert.py`, `test_reference_frontend.py`, and
`test_official_resources.py` (outside ctest, same problem when they run on this machine).

## 2. Decisions the diff made that the plan left open — ratify and record

### 2.1 Open decision 9.4 (`PinnedHostBuffer` scope) was resolved as the broad option, silently

`arena.cu:261-278` applies aligned-malloc + `cudaHostRegister` to **every**
`PinnedHostBuffer` consumer on **both** platforms — including the small per-request ingress
buffers (`src/targets/qwen3_6/impl/runtime/program.h:281-289`) that plan §4.3/§9.4 called
out as gaining page-granular pinning and a slower registration path for no benefit. That may
be the right call (one 4096-alignment guarantee everywhere, documented in `arena.h:89-93`),
but it is the larger Phase 5 Linux regression surface and the plan said "decide before 4.3
is implemented". Either record the broad decision and its rationale in plan §9.4, or narrow
the change to the materializer's slot buffers.

### 2.2 Op-test FP flags: `/fp:strict` chosen; the plan's demanded probe is now done

`tests/CMakeLists.txt:36-44` settles plan §4.1's open action with `/fp:strict` and a comment
arguing SSE2 has no FMA instruction. **Verified on the target machine** by disassembling
`a * b + c` (float and double) under cl 19.51:

| Flags | Result |
|---|---|
| default (`/O2`, SSE2 baseline, `/fp:precise`) | `mulss`+`addss` / `mulsd`+`addsd` — no contraction |
| `/O2 /arch:AVX2 /fp:precise` | `vmulss`+`vaddss` — **still no contraction** |
| `/O2 /arch:AVX2 /fp:strict` | same, no contraction |

So MSVC's default FP model matches the pinned GCC/Clang `-ffp-contract=off` behavior even
under `/arch:AVX2` on this compiler version; contraction would require `/fp:fast` or an
explicit `/fp:contract`. `/fp:strict` is correct and conservative. Two small follow-ups:
soften the comment's claim that `/arch:AVX2` alone would contract (it would not, on this
compiler — `/fp:fast` / `/fp:contract` would), and note in plan §4.1 that the probe was run
and what it showed.

### 2.3 Media-root containment fix is a cross-platform behavior change — `acquire.cpp:323`

`relative.native().starts_with("..")` → `*relative.begin() == ".."` changes an
access-control check on **Linux too**, and rode in with the port. **Verified on the target
machine** (`std::filesystem` under MSVC):

| Case | relative() | old check | new check |
|---|---|---|---|
| inside root | `sub\file.png` | accept | accept |
| genuine escape | `..\other\file.png` | reject | reject |
| entry named `..hidden` | `..hidden` | **falsely reject** | accept |
| root itself | `.` | accept | accept |

The new check is a real fix (the old one rejected legitimate entries whose names start with
`..`), but it deserves its own commit with a test covering both the escape and the
`..hidden` cases, rather than being an unremarked hunk inside the Windows port.

## 3. Smaller items

1. **`WSAStartup` failure calls `std::exit(1)` from library code** — `acquire.cpp:44-53`.
   Everything around it throws typed `Error`s; `throw Error(ErrorKind::RemoteUnavailable,
   ...)` would keep serve alive when only media acquisition is broken. (The
   `ensure_winsock()` call itself is correctly placed: `resolve_public`'s `getaddrinfo` runs
   before curl ever initializes Winsock.)
2. **TMA alignment-trap printf runs per-thread** — `nvfp4_w4a4_tma.cuh:41-47`. On a
   misaligned descriptor every thread in the grid printfs before `__trap()`. Gating the
   printf on `threadIdx.x == 0` keeps the diagnostic readable. Cosmetic; the guard itself is
   exactly what plan §4.4.1 option 1 required (128-byte assert, trap not silence).
3. **`CTRL_LOGOFF_EVENT` handled beyond the plan's four events** — `apps/serve/main.cpp:41`.
   Sensible (a console session logoff should stop the server), just undocumented; add it to
   plan §3.5 / §4.4 when recording results.
4. **`test_request_log.cpp` does not assert LF-only bytes.** Phase 2's exit criterion greps
   the JSONL for CR bytes; a cheap byte-level check in the test (read the file back in
   binary, assert no `0x0D`) would encode the §3.7 contract permanently instead of relying
   on a one-time manual grep.
5. **Plan §4.4 table is missing four items the port actually discovered.** Record them so
   the "small product-surface port items" list stays truthful: `/Zc:preprocessor` for MSVC
   (CCCL rejects the traditional preprocessor), `UTF8PROC_STATIC` on `ninfer_text` (C2491),
   the `api_impl.h` explicit-specialization defaulted-move workaround (MSVC LNK2019 — the
   tree's only occurrence of that pattern; other `= default` uses are non-template and
   unaffected), and `constexpr std::sqrt` → `const` in three test files (GCC constexpr
   extension MSVC rejects).

## 4. Verification ledger

### Closed by this review (target machine, 2026-08-17)

- **Full build is green.** All 423 targets, including `ninfer_ops`, `ninfer_nvfp4_tma`,
  `ninfer.exe`, and `ninfer-serve.exe`, are built in `build/` from the current working tree
  (Ninja reports no work to do). The five FFmpeg DLLs from plan §2.2 — including
  `swresample-4.dll`, which never appears on a link line — are staged beside the executables
  in `build/apps/`. `dumpbin /dependents` on a media-linked test confirms the import set is
  exactly `avformat-60 / avcodec-60 / swscale-7 / avutil-58` plus the MSVC runtime.
- **ctest: 83 passed, 5 skipped, 1 failed (89 total, 85 s, idle RTX 5090).** The skips are
  the artifact-gated tests (`qwen3_6_27b_prefix_real`, `qwen3_6_27b_load_plan`,
  `qwen3_6_35b_a3b_real`, `_dflash_real`, `_dflash_load_plan`) — correct skip-77 behavior,
  no `.ninfer` artifact on this machine yet. The failure is section 1.3.
- **NVFP4 TMA fix verified live, not just compiled.** `ninfer_linear_nvfp4_a4_test` runs at
  `tokens = 1024`, which satisfies the TMA gate (`tokens >= 1024 && tokens % kTmaBlockM ==
  0`), and `ninfer_linear_swiglu_nvfp4_test` selects the TMA route at `kPrimaryT` — both
  pass, along with `attn_input_proj`, `gdn_input_proj`, and `linear_add` NVFP4 tests: all
  five caller families from plan §4.4.1. The device-side 128-byte alignment guard never
  trapped, so the option-1 risk (nvcc placing the alignment-1 wrapper at a misaligned
  constant-bank offset) does not materialize in practice on this toolchain.
- **`cudaHostRegister` works under WDDM** — `ninfer_artifact_materialization_test` passes,
  retiring the registration-failure half of the plan §7 risk row. Load throughput on the
  real ~17 GB artifact remains open.
- **The unbuffered read path is actually used**: running the reader and materialization
  tests directly produces no "direct I/O unavailable" fallback line on stderr, and C: reports
  a 512-byte logical sector — the `FILE_FLAG_NO_BUFFERING` handle is live.
- **Trailing short read at EOF under `FILE_FLAG_NO_BUFFERING`** (plan §4.2's "the one place
  the port can silently corrupt a load"): on C: (NTFS, logical sector 512), a synchronous
  positional `ReadFile` of 4096 bytes whose range ends 1808 bytes past the last sector
  boundary returns `TRUE` with `bytes_read = 1808` and correct content — matching `O_DIRECT`
  `pread`. A read starting at/past EOF fails with `ERROR_HANDLE_EOF` (the materializer never
  issues one). Re-verify only if artifacts move to a volume with a different sector size —
  the reader's `FileStorageInfo` gate covers > 4096 already.
- **MSVC FP contraction** — see 2.2.
- **`GetConsoleMode` null-pointer behavior** — see 1.1 (crash confirmed, fix shape confirmed).
- **No leftover POSIX-only usage**: every `unistd.h` / `sys/*` / `arpa,netdb` /
  `localtime_r` / `getpid` / `isatty` / `std::aligned_alloc` occurrence in
  `src`/`apps`/`tests` is inside an `#else` (POSIX) branch. `curl_global_init` is called
  (`acquire.cpp:236`). Wider sweeps also clean: no `dirent/poll/termios/sys-time` headers,
  no `mkstemp/fork/popen/strcasecmp/setenv/...` calls, no `av_err2str` (decode.cpp uses
  `av_strerror` into a local array, FFmpeg headers correctly `extern "C"`-wrapped), no
  remaining `cudaMallocHost`, `ssize_t`/`off_t` confined to the POSIX reader branch, no
  hardcoded `/tmp`-style paths in C++ sources, and `PinnedHostBuffer`'s move-assign frees
  through the new unregister path.
- **Plan §7's GPU-contention row is a practical reality, not a hypothetical**: a resident
  `llama-server` held 27.3 of 32.6 GB VRAM at review time and had to be closed before the
  suite could run. Benchmark runs need the same discipline.

### Closed after the fixes landed (target machine, 2026-08-17, real artifact)

- **Real-artifact integration passes**: with `NINFER_QWEN3_6_27B_WEIGHTS` pointed at
  `qwen3_6_27b.ninfer` (16.7 GB), `ninfer_qwen3_6_27b_prefix_real_test` **passes in 6.4 s**
  on Windows. `load_plan` still skips — it additionally wants
  `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` (`qwen3_6_27b_nvfp4.ninfer`, not yet downloaded).
  **Update (second session): `qwen3_6_27b_nvfp4.ninfer` is now downloaded and
  `ninfer_qwen3_6_27b_load_plan_test` passes**; `prefix_real` re-passes (9.75 s). The only
  remaining skips are the 35B-A3B artifact tests (artifact not local) and the frontend test
  (wants the HF checkpoint via `NINFER_QWEN3_6_27B_HF`).
- **WDDM load throughput is a non-issue** (retires the last §7 half-risk): CLI load of the
  real artifact reports weights materialized in 2.94 s (~5.2 GB/s through the unbuffered
  reader) and host-to-device in 1.87 s (~8.2 GB/s under WDDM); 15.25 GiB resident.
- **Phase 2 exit criteria all pass.** CLI: greedy generation from the real artifact answers
  correctly (`--no-thinking` → "Paris"; with thinking, reasoning streams to stderr as
  designed). Serve: binds, `/v1/models` reports `qwen3.6-27b`, non-stream and streaming
  (201 SSE events + `[DONE]`) chat completions succeed, `CTRL_BREAK_EVENT` shuts down
  cleanly with exit 0 through the new `SetConsoleCtrlHandler` path, and the request-log
  JSONL is complete and **LF-only** (byte-checked): `server_start`, two
  `request_start`/`request_done` pairs, `throughput`. (There is no `server_stop` event in
  the log contract — `write_server_start` is the only lifecycle writer.)
- **All five NVFP4/input-proj caller-family tests re-pass** after the trap-printf change.
- **Phase 3 media smoke passes.** A locally generated solid-red PNG through `--vision` is
  decoded (mingw-built FFmpeg DLLs + swscale) and answered correctly ("red"); an
  `https://raw.githubusercontent.com/...` image URL is fetched through Schannel against the
  Windows certificate store, resolved via the new Winsock path, and recognized ("Python").
  What remains of Phase 3 is only the Linux-reference comparison (same artifact/seed, same
  decoded pixels) and a video prompt.

### Closed in the second session (2026-08-17, later the same day)

- **Full ctest is green: 90/90, 0 failures** (6 skips: the artifact-gated tests, env vars
  unset in that shell, plus the frontend test now skipping-77 correctly instead of crashing;
  the 90th test is the new `test_media_acquire_path`).
- **Phase 4 tooling (§4.5) landed.** New `tools/bench/hostexec.py`: `binary_path()` appends
  `.exe` on Windows, `SERVER_POPEN_FLAGS` (`CREATE_NEW_PROCESS_GROUP`) +
  `request_stop()` (CTRL_BREAK_EVENT / SIGTERM) replace bare `Popen.terminate()`, and
  `format_command()` renders reproduction lines with `list2cmdline` on Windows instead of
  POSIX `shlex` quoting. Wired into `run_serve_corpus.py`, `run_serve_concurrency.py`,
  `run_ninfer_bench_matrix.py`, and `serve_thinking_preservation.py`;
  `migrate_v1_to_v2.py` now fails fast on Windows with a "Linux-only" parser error
  (`os.O_DIRECTORY`; v1 is a dead format). All runners' argument parsing verified on the
  target machine.
- **Phase 4 docs (§4.6) landed.** README Requirements/Build gained the Windows platform,
  toolchain, and the FFmpeg (LGPL-shared, 6.1-branch) / libcurl (Schannel) provisioning
  recipe with the `CMAKE_PREFIX_PATH` configure line; AGENTS.md gained the Windows host
  platform in the product contract and a Windows toolchain row in the local-environment
  table.
- **Open decision 9.5 resolved and implemented**: `ninfer::product::ConsoleUtf8Scope`
  (`src/product/console_unicode/`, new `ninfer_product_console_unicode` library) enters both
  apps' `main` — UTF-8 console code pages (restored on normal exit) plus
  `GetCommandLineW`/`CommandLineToArgvW`-based UTF-8 argv; passthrough off Windows.
  Verified: a non-ASCII argument round-trips through `ninfer.exe` as UTF-8 bytes
  (`0xC3 0xB6` for `ö`) where the ANSI argv delivered the ACP byte.
- **Video-prompt smoke passes** (the Windows-side remainder of Phase 3): the repo fixture
  `examples/cli/messages/video_temporal.json` (`temporal_events.mp4`) through
  `--vision --greedy --no-thinking` answers `红色圆形；3；是；9` — the four demanded items,
  correct UTF-8, exit 0 — through the mingw FFmpeg DLLs.
- The plan doc's §4.1 probe note, §4.4 four discovered items, `CTRL_LOGOFF_EVENT`, and open
  decisions 9.2–9.6 are now recorded/struck in `docs/windows-port/plan.md`.

### New finding (2026-08-17, second session)

- **`serve_thinking_preservation.py` fails one assertion on Windows**: the Responses-API
  continuation (`previous_response_id`, thinking inherited) takes `append_frontier` where
  the smoke expects `restore_response_checkpoint`; the other nine requests and all chat-side
  reuse-path assertions (including both `restore_turn_checkpoint` hits) pass. The reuse
  choice is a deterministic function of whether the parent's 16 greedy output tokens
  re-tokenize identically across the response boundary in the re-rendered continuation
  prompt — i.e., this is the plan-§6 cross-platform greedy-parity question surfacing through
  a smoke, not a tooling defect (the runner drove all 10 requests end to end). Settle it
  with the Linux-side run: same artifact, compare the parent response's generated tokens; if
  Linux generates the same 16 tokens and still takes `restore_response_checkpoint`, this is
  a real port divergence to chase, otherwise the smoke's expectation is content-coupled.

### Interim Windows performance measurements (2026-08-18, campaign paused at 28/95)

The Phase 5 serving campaign (`run_serve_corpus.py`, qwen3.6-27b/groupwise-int, published
method: persistent server, serial requests, five fixed seeds, stochastic sampling) was **cut
short after 28 of 95 requests** to free the machine; it resumes from
`out/serve-corpus-windows/run.jsonl` with the same command. Two consequences to carry into
the final write-up: the `long_decode_aime26_15` point has only 3 of 5 seeds so far, and the
resume restarts the server mid-MTP3-block, so if the split-run numbers look off, repeat the
affected measurements in one session. One tree fix was needed to run at all:
`SERVER_LOG_SCHEMA_VERSION` in `run_serve_corpus.py` bumped 9 → 10 — commit `3b607545`
bumped the server's request-log schema (additive media-preparation fields) without updating
the runner's pin; pre-existing, not a Windows issue.

Completed so far (mean ± sample stddev; MTP0 block complete at 5 seeds per point):

| MTP0 prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 3,537.9 ± 9.0 | 2,180.2 ± 5.5 | 83.5 ± 0.2 |
| 64,512 | 2,832.7 ± 13.0 | 22,856.3 ± 105.1 | 75.6 ± 0.5 |
| 130,048 | 2,299.2 ± 5.9 | 56,729.2 ± 147.3 | 68.8 ± 0.2 |
| 260,096 | 1,673.2 ± 1.3 | 155,779.5 ± 118.6 | 58.6 ± 0.1 |

| MTP3 fixture | Seeds | Decode tok/s | MTP acceptance | Tokens/round |
|---|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5/5 | 214.0 ± 2.7 | 79% ± 1% | 3.37 ± 0.04 |
| `long_decode_aime26_15` | 3/5 | 192.6 ± 1.4 | 73% ± 1% | 3.18 ± 0.03 |

Every point is **above** the published Linux row for this artifact (e.g. MTP0 decode
83.5 vs 77.6 at 8K, 58.6 vs 54.8 at 256K; MTP3 aime26_01 214.0 vs 175.4). Do not read this
as "Windows is faster than Linux": the published Linux rows were measured at an older commit
and toolchain (CUDA 13.1-era campaigns), so the delta conflates tree and toolchain evolution
with the platform. The honest cross-platform comparison is the plan-§5 Linux regression run
at this commit. These interim values stay out of `docs/performance.md` until the campaign is
complete and clean.

### clang-cl bring-up (2026-08-18, third session)

The tree now also builds and passes with **clang-cl 22.1.8** (`C:\libs\llvm\bin`) as the
C/C++ host compiler, nvcc keeping `cl` as its host compiler (`-ccbin`; nvcc on Windows
supports only `cl`), lld-link as the linker. Same VS x64 Native Tools environment. Configure:

```
cmake -S . -B build-clangcl -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ^
  -DCMAKE_C_COMPILER=C:/libs/llvm/bin/clang-cl.exe ^
  -DCMAKE_CXX_COMPILER=C:/libs/llvm/bin/clang-cl.exe ^
  -DCMAKE_CUDA_HOST_COMPILER="C:/.../VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_PREFIX_PATH="C:/libs/ffmpeg-n6.1.2;C:/libs/curl"
```

Evidence (target machine, 2026-08-18):

- **Full build green, zero source changes** — 353/353 steps in `build-clangcl/`. Only two
  CMake edits were needed, both flag-routing:
  - `tests/CMakeLists.txt`: the op-test FP branch now excludes clang-cl from the GNU-spelling
    branch (`CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"`) and gives it `/fp:strict`. **This is
    load-bearing under clang-cl, not hygiene**: probed on clang-cl 22.1.8 — it
    warns-and-ignores `-fno-fast-math`/`-ffp-contract=off`, and unlike cl 19.51 its default
    `/fp:precise` DOES contract `a*b+c` into `vfmadd` under `/arch:AVX2`; `/fp:strict`
    disables it (`mulss`+`addss` at the SSE2 baseline either way). This retires plan §4.1's
    "verify clang-cl accepts those GCC-style spellings" action: it does not.
  - Top-level `CMakeLists.txt`: `/Zc:preprocessor` gated to real cl via
    `$<CXX_COMPILER_ID:MSVC>` (clang's preprocessor is already conforming; clang-cl warns
    "argument unused"). `/utf-8` and `/bigobj` are accepted by clang-cl and stay unconditional.
- **ctest 90/90, 0 failures** with the 27B artifact env vars set — including
  `prefix_real` (10.1 s), `load_plan`, reader, and materialization. 4 skips: the 35B-A3B
  artifacts and the HF-checkpoint frontend test (fixtures not local, correct skip-77).
- **Both apps smoke-pass**: CLI greedy `--no-thinking` → "Paris", exit 0; `ninfer-serve.exe`
  binds, `/v1/models` reports `qwen3.6-27b`, a chat completion answers "Paris" with
  `finish_reason: stop`.
- Warning delta vs cl: benign CRT `getenv` deprecation notes in test TUs, and one
  pre-existing `-Wswitch` in `w8_pair_plan.cpp:193` (the `ExactConcat*` values are
  intentionally handled by the post-switch throw; `homogeneous_schedule` maps them away).
  Not port work.

- **`ninfer_bench` smoke passes** (`run_ninfer_bench_matrix.py --preset smoke`, first-ever
  Windows compile of the opt-in `bench/` targets — the port had scoped them out): 3/3 cases,
  schema-v11 reports valid, load 2.87 s / upload 1.9 s (matching the MSVC-recorded 2.94 s /
  1.87 s). One staging gap found and fixed in top-level `CMakeLists.txt`: the FFmpeg DLL
  copy covered `apps/` and `tests/` but not `bench/`, so every bench case died at load with
  `0xC0000135` (DLL not found) when `NINFER_BUILD_BENCHMARKS=ON`. Toolchain-independent —
  the MSVC build has the same gap when benchmarks are enabled.

Status: clang-cl is a **verified-working alternative host compiler**, not the documented
default — README/AGENTS.md continue to name MSVC. The MSVC `build/` tree is untouched.

### Still owed (updated after the evidence above)

- Cross-platform greedy parity (plan §6): the same prompt/seed against the Linux reference
  build — needs a Linux-side run to compare against. Same for image/video output parity
  (Phase 3's remaining half; the Windows-side video smoke above passes) and the
  thinking-preservation divergence in "New finding".
- Phase 5: the Windows performance row (`run_serve_corpus.py`, now Windows-capable); the
  Linux regression pass — whose scope is fixed by the choices above: arena (both platforms),
  NVFP4 TMA kernel source (both platforms, all five caller families' op tests),
  `acquire.cpp` containment check (both platforms), `api_impl.h` (both platforms,
  semantically identical).
- Ratification leftovers by design: splitting the containment fix into its own commit when
  the port lands.

## 5. Reference comparison: `natpate/ninfer-windows` (2026-08-18)

A second Windows port of this project exists publicly:
[`natpate/ninfer-windows`](https://github.com/natpate/ninfer-windows), a fork of the same
upstream (`Neroued/ninfer`) that shares our base commit `b2b96bae`, so the two ports diff
exactly. Its README states its Windows compatibility layer is itself ported from
[`Don-Chad/ninfer-3090`](https://github.com/Don-Chad/ninfer-3090) (minus that fork's `sm_86`
retargeting) — a third lineage, unexamined here. Its fork diff is ~15 Windows commits plus two
unrelated features (llama.cpp WebUI serving, `meta.n_ctx` on `/v1/models`); it touches 12
source files against our 20 plus tooling. Nothing below changes the "Still owed" list.

### 5.1 Independently convergent decisions

Same diagnosis and materially the same fix in both trees, which is the useful signal: the
`localtime_r` / `getpid` / `isatty` substitutions, `winsock2.h`+`ws2tcpip.h` ordering with
`WSAStartup` before the first `getaddrinfo`, `relative.native().starts_with("..")` being
MSVC-hostile, `UTF8PROC_STATIC` (C2491), `NOMINMAX`/`WIN32_LEAN_AND_MEAN`, `/Zc:preprocessor`
for CCCL, `mmap` -> `CreateFileMappingW`+`MapViewOfFile`, `O_DIRECT`/`pread` ->
`FILE_FLAG_NO_BUFFERING` + positional `ReadFile` under the same 4096-byte contract, `ws2_32`,
and both MSVC blockers (C2719 on the TMA descriptor, LNK2019 on the `qwen3_6` plan moves).
On the last one both ports ended at *explicit move bodies*; that fork reached it after three
reverted attempts (`702ebfd8` -> `ad469bee` -> `398c69d6` -> `f637218b`) and had to edit the
exported `runtime.h` on the way, which our `api_impl.h`-only change avoids.

Two non-findings worth recording so they are not re-investigated: that fork's `d7cc6552`
("pair_row in the SwiGLU TMA epilogue", described there as breaking every platform) fixes a
`parent_row` identifier its *own* earlier commit introduced — upstream `b2b96bae` is clean,
verified. And its `727d6fdc` tried `/d2FH4-` to make MSVC accept the over-aligned by-value
kernel parameter, then reverted it in `86e85fcc`; that avenue is closed for both ports.

### 5.2 Where this tree is ahead

- **Console/Unicode**: absent there — no `SetConsoleOutputCP`, no `CommandLineToArgvW`
  anywhere in its `src`/`apps`. Non-ASCII prompts, paths, and the Phase 3 CJK video answer
  would be ACP-mangled. Covered here by `ninfer::product::ConsoleUtf8Scope` (decision 9.5).
- **Interactivity probe**: it uses `_isatty(_fileno(stderr))`, true for any character device
  (`NUL` included); the `GetConsoleMode` form (1.1) is a real console probe.
- **Request-log line endings**: its `request_log.cpp` still opens the JSONL in text mode, so
  Windows logs get CRLF and silently break the section 3.7 byte contract the runners and
  `tools/` consumers rely on. Fixed here, plus the byte-level assertion (3.4).
- **Direct-I/O robustness**: it opens `FILE_FLAG_NO_BUFFERING` unconditionally and throws if
  the open fails, with no sector-size gate; a volume with a logical sector above 4096, or
  without unbuffered support, fails the load outright. Our `FileStorageInfo` probe degrades to
  buffered reads with a one-time warning instead.
- **Pinned staging alignment**: it leaves `cudaMallocHost` in place, so the 4096-byte
  direct-I/O destination alignment stays an unstated allocator assumption. Decision 9.4 turns
  it into a construction guarantee (`arena.h:89-93`).
- **Tests**: that port has no Windows test story — `constexpr ... std::sqrt` is still present
  in four of its `tests/ops/*` files (the section 4.4 GCC-extension item), and its
  `tests/CMakeLists.txt` is untouched, so the op tests keep only the GCC-spelled FP flags.
  Here: 90/90 ctest under cl and clang-cl, `/fp:strict` on both (load-bearing for clang-cl,
  see the clang-cl section), the frontend-test fixture gate (1.3), and
  `test_media_acquire_path` (2.3).
- **Runtime DLL deployment**: it documents "keep the `vcpkg_installed/` tree next to the
  executables"; here CMake stages the five FFmpeg DLLs into `apps/`, `tests/`, and `bench/` at
  configure time, including `swresample`, which never appears on a link line.
- **Tooling**: `tools/bench`, `tools/smoke`, and `migrate_v1_to_v2.py` are untouched there
  (POSIX-only); section 4.5's `hostexec.py` covers `.exe` suffixing,
  `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` shutdown, and `list2cmdline` reproduction
  lines.
- **Alternative host compiler**: clang-cl is verified here; that port is cl-only.
- **Reviewability**: its `nvfp4_w4a4_tma.cuh` is committed CRLF in the index (`git ls-files
  --eol` reports `i/crlf`), which is why its descriptor commit shows 763 changed lines in one
  header — an upstreaming obstacle this tree does not have.

### 5.3 The one real design fork: the C2719 TMA descriptor

That port keeps the descriptor by value on POSIX and passes it **by device pointer on
Windows** (`NINFER_NVFP4_TMA_DESCRIPTOR_PARAM`, commit `48035f7b`): per launch a
`cudaMallocAsync`, a 512-byte `cudaMemcpyAsync` from a stack local, and a `cudaFreeAsync` in
the block guard's destructor. Two consequences, both arguing for option 1 as implemented here
(plan section 4.4.1):

1. **Stream-ordering hazard.** The guard frees with `cudaFreeAsync(device, nullptr)` — the
   legacy default stream — immediately after the launch, while the consuming kernel runs on a
   stream created `cudaStreamNonBlocking` (`src/core/device.cu:66`). A non-blocking stream
   does not synchronize with the legacy stream, so the descriptor block can be recycled while
   the TMA unit is still reading it: precisely the silent wrong-address load that the pointer
   scheme was adopted to avoid. The alignment-1 wrapper here keeps the map in parameter space,
   so there is no separate lifetime to order.
2. **Per-launch cost** on every NVFP4 GEMM (alloc + H2D from unpinned memory + free), versus
   zero added work here.

The trade-off accepted in exchange is the one already recorded: the 128-byte guarantee is a
device-side runtime check rather than a static property, and it has never trapped across all
five caller families on this toolchain.

### 5.4 Worth borrowing (not applied)

1. **Static CUDA runtime.** That port links `CUDA::cudart_static` with
   `CUDA_RUNTIME_LIBRARY Static` and a global `/NODEFAULTLIB:LIBCMT` (the static runtime is
   built `/MT` and emits a conflicting `LIBCMT` default-lib directive under our `/MD`), so the
   CUDA Toolkit becomes a build-time dependency only. This tree links shared `cudart`, i.e.
   the executables need `cudart64_13.dll` from the toolkit `bin` on `PATH` — fine on a
   developer machine, disqualifying for any redistributable build. Proposed shape,
   deliberately opt-in so the documented default is unchanged:

   ```cmake
   option(NINFER_CUDART_STATIC
     "Link the CUDA runtime statically (Windows only; no cudart DLL at run time)" OFF)
   # after find_package(CUDAToolkit), before add_subdirectory(src):
   if(NINFER_CUDART_STATIC)
     set(NINFER_CUDART_TARGET CUDA::cudart_static)
     set(CMAKE_CUDA_RUNTIME_LIBRARY Static)   # initializes the per-target property
     if(MSVC)
       add_link_options(/NODEFAULTLIB:LIBCMT) # keep one CRT: /MD everywhere
     endif()
   else()
     set(NINFER_CUDART_TARGET CUDA::cudart)
   endif()
   ```

   The six explicit `CUDA::cudart` link sites (`src/CMakeLists.txt:35,59,356`,
   `tests/CMakeLists.txt:138,155`, `bench/CMakeLists.txt:74`) then route through
   `${NINFER_CUDART_TARGET}`. Validating it costs a full CUDA recompile (the `-cudart` flag
   reaches the compile line, so no object is reusable), then `dumpbin /dependents` on both
   apps to confirm no `cudart64_*.dll` import, a ctest pass, and the two app smokes. Not
   started.
2. **vcpkg manifest** (`vcpkg.json` with a pinned `builtin-baseline`) as an *alternative*
   provisioning path to the pkgconf recipe in README. Keep ours as the documented default:
   vcpkg does not pin FFmpeg to the 6.1 branch, which is exactly the decoded-pixel parity
   assumption behind the Phase 3 comparison still owed.
3. **A plain `docs/windows.md`** linked from `docs/README.md` as the user-facing build guide —
   where the README recipe should end up when the port lands.
4. **`.gitignore` entries** that fork added and this tree lacks: `vcpkg_installed/`,
   `models/`, `benchmark_results/`, and a `dist/*` carve-out.
5. **Portable zip release** — out of scope for the port itself, and gated on item 1.

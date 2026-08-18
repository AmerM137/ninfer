# Windows New-User Setup — Notes and Open Questions (2026-08-18)

Status: temporary working document, companion to
`docs/windows-port/plan.md` and
`docs/windows-port/review.md`. It records what setting up a *fresh* Windows machine to
compile this tree currently costs, where a new user gets stuck, and the options for shortening
that path. **No code changes have been made for any of this** — §4 is a menu, not a plan.
Remove or fold this document in once §4 is decided.

Everything in §1–§3 is stated against the verified reference machine (this one) and was
re-checked on 2026-08-18.

## 1. Gates before setup matters at all

- `CMakeLists.txt` rejects any `CMAKE_CUDA_ARCHITECTURES` other than `120a` with a
  `FATAL_ERROR` before compiler detection, so the hardware prerequisite is literally an
  **RTX 5090**.
- NVIDIA driver supporting CUDA 13.1+, Windows 11 x64.
- No install target, no packaged binary: the apps are run from the build tree.

The addressable audience is therefore small, and every instruction below assumes someone
willing to build C++/CUDA from source.

## 2. Toolchain and dependencies

### 2.1 Tools

| Piece | Reference machine | New user gets it from |
|---|---|---|
| Host compiler | VS 18 Community, cl 14.51.36231 | VS installer, **Desktop development with C++** (nvcc supports only `cl` on Windows) |
| CUDA | Toolkit 13.3 | NVIDIA installer (13.1 is the floor) |
| CMake | 4.3.2 (scoop) | `scoop install cmake` / winget; 3.28 is the floor |
| Ninja | scoop | `scoop install ninja` |
| pkgconf | 3.0.5, scoop shims provide both `pkg-config` and `pkgconf` | `scoop install pkgconf` |
| Python 3 | on PATH | only for `-DBUILD_TESTING=ON` (`find_package(Python3 REQUIRED)`, `tests/CMakeLists.txt:1`) |

Note for the README: our Windows section says pkgconf must be "shimmed as `pkg-config.exe`".
That is stricter than reality — CMake 4.3's `FindPkgConfig` searches `pkg-config`,
`pkg-config.bat`, then `pkgconf` (`FindPkgConfig.cmake:498-518`), so the plain `pkgconf.exe`
name is found.

### 2.2 FFmpeg — easy

BtbN `ffmpeg-n6.1.2-*-win64-lgpl-shared`, unzipped to a prefix (`C:\libs\ffmpeg-n6.1.2` here).
Why it is painless, verified in the installed package:

- The `.pc` files are relocatable: `prefix=${pcfiledir}/../..`, so any unpack path works with
  no editing.
- `lib/` ships MSVC import libraries (`avformat.lib`, `avcodec.lib`, `avutil.lib`,
  `swscale.lib`, `swresample.lib`) alongside the mingw `.dll.a` set, so `link.exe` resolves
  the pkg-config `-l` entries.
- `bin/` holds the five runtime DLLs that our configure step stages into `build/apps`,
  `build/tests`, and `build/bench`.

Both package qualifiers still matter for reasons recorded in the README: **shared** because
MSVC cannot link the mingw static archives, and the **6.1 branch** because the Linux reference
decoders are 6.1 and decoded pixels feed vision output parity.

### 2.3 libcurl — the wall

We resolve curl only through `pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET
libcurl>=7.85)`, and essentially no prebuilt Windows curl ships a `libcurl.pc`. So the user
builds it. What produced the working install here, modulo the version tag:

```bat
git clone --branch curl-8_21_0 --depth 1 https://github.com/curl/curl C:\src\curl
cmake -S C:\src\curl -B C:\src\curl\build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF -DCURL_USE_SCHANNEL=ON -DCURL_USE_LIBPSL=OFF ^
  -DCURL_ZLIB=OFF -DBUILD_CURL_EXE=OFF -DCMAKE_INSTALL_PREFIX=C:/libs/curl
cmake --build C:\src\curl\build --target install
```

Resulting install, as inspected: static `lib/libcurl.lib`, `Cflags: -DCURL_STATICLIB`, and
`Libs: … -lbcrypt -lcrypt32 -lsecur32 -lws2_32 -liphlpapi -lwldap32 -ladvapi32` — Schannel TLS
against the Windows certificate store, no CA bundle deployed, no curl DLL.

Two things to know:

- Unlike FFmpeg, the generated `libcurl.pc` hardcodes `prefix=C:/libs/curl`, so that directory
  cannot be moved after install without editing the `.pc`.
- The copy on this machine reports `8.21.0-DEV`, i.e. it was built from git master. Any
  documented recipe should name a release tag.

## 3. The build, and where new users get stuck

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="C:/libs/ffmpeg-n6.1.2;C:/libs/curl"
cmake --build build --parallel
```

Run from the **x64 Native Tools Command Prompt**. `CMAKE_PREFIX_PATH` extends pkgconf's search
path automatically, so no `PKG_CONFIG_PATH` juggling is needed. Outputs are
`build\apps\ninfer.exe` and `build\apps\ninfer-serve.exe` with the FFmpeg DLLs already beside
them; `cudart64_13.dll` resolves from the toolkit's `bin` on PATH.

Failure modes, ranked by how likely a newcomer hits them:

1. **Not in the dev prompt.** nvcc shells out to `cl` and needs `INCLUDE`/`LIB`; a plain shell
   produces cryptic failures.
2. **No pkg-config on the machine.** Nothing in a normal VS + CUDA setup pulls pkgconf in, and
   most Windows C++ developers have never installed it.
3. **Reaching for a prebuilt curl**, which has no `.pc` — configure fails. Mildly absurd
   detail: the curl install here *also* ships `lib/cmake/CURL`, which plain
   `find_package(CURL)` would consume happily; we simply never look.
4. **Wrong FFmpeg package.** A static mingw build will not link under MSVC (loud, fine). The
   quiet one: `libavformat>=60` accepts 7.x/8.x, and our DLL staging glob is version-agnostic,
   so a user can get a **fully working build on FFmpeg 8** whose decoded pixels differ from
   the Linux reference — exactly the parity assumption the Phase 3 comparison rests on. There
   is no upper bound.
5. **Visual Studio generator.** Unsupported by our tooling (per-config output layout); Ninja
   only, and the README says so.
6. **`BUILD_TESTING=ON` extras.** Python 3, plus the artifact env vars
   (`NINFER_QWEN3_6_27B_WEIGHTS`, `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS`,
   `NINFER_QWEN3_6_27B_HF`) or the affected tests skip-77.

Rough budget: ~30 minutes of tool installs, ~10 minutes to build curl, then one long first
CUDA compile.

## 4. Options to shorten the on-ramp — undecided, nothing applied

1. **Fall back to `find_package(CURL)` (and `find_package(FFMPEG)`) when pkg-config finds
   nothing**, which is what `natpate/ninfer-windows` does on Windows (see
   `docs/windows-port/review.md` §5.4). Removes failure modes 2 and 3 outright: any
   CMake-installed curl then works without pkgconf. Cheapest change with the largest effect.
2. **Pin the FFmpeg major** (`libavformat>=60 libavformat<61`, or an explicit version check
   with a clear error message) so failure mode 4 cannot happen silently. Small, and it is a
   correctness fix for the parity claim, not just ergonomics.
3. **`CMakePresets.json`** with a `windows` preset encoding the generator, build type, and
   prefix paths, so the documented invocation becomes `cmake --preset windows`.
4. **vcpkg manifest** as an alternative provisioning path (their `vcpkg.json`). Removes the
   curl build entirely, at the cost of the FFmpeg 6.1 pin — which is why it should be an
   alternative, never the documented default.
5. **Static CUDA runtime** (`NINFER_CUDART_STATIC`, sketched in
   `docs/windows-port/review.md` §5.4). Not an on-ramp item — it matters for redistributing
   binaries, not for compiling.

Open questions to settle when this is picked back up:

- Does the README's Windows section stay a "build curl yourself" recipe, or become
  "pkgconf *or* CMake package config" once option 1 exists?
- Is the FFmpeg upper bound acceptable to upstream, given the Linux path also carries the
  6.1-vs-newer parity question and currently has no bound either?
- Do options 1–3 belong in the Windows port series, or in a follow-up branch after the port
  lands? They all touch the shared (Linux-affecting) dependency-resolution code.

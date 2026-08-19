# Debugging

NInfer runs as one process across two address spaces: host C++ driving the CLI, server, artifact
reader, and scheduler, and CUDA kernels operating on VRAM. No single debugger covers both, so the
tooling splits along that line. This guide records which tool answers which question, and the
build configurations each one needs.

Debug builds are not a supported deployment configuration. Everything below is a local
investigation workflow; the published performance and accuracy numbers come from the Release
build described in the [project README](../README.md).

## Choosing a tool

| Symptom | First tool |
|---|---|
| Host crash, bad state, wrong control flow | Visual Studio debugger |
| Kernel fault, `CUDA_CHECK` failure, silent corruption | `compute-sanitizer` |
| Wrong logits or tokens with no error reported | per-layer tensor dumps against a reference |
| Kernel-local numerical error already isolated to one Op | in-kernel `printf`, then Nsight Visual Studio Edition |
| Slower than expected | Nsight Systems, then Nsight Compute |

For device-side work the interactive debugger is rarely the first move. `compute-sanitizer` and
in-kernel `printf` locate most memory and indexing defects, and numerical bisection locates the
rest; source-level device debugging is worth its cost only once a single kernel is implicated.

## Build configurations

Presets live in `CMakeUserPresets.json`, which is machine-local and not tracked, because they
carry absolute dependency prefixes. The presets below inherit the `windows` preset described in
the README; adapt the paths to the local FFmpeg and libcurl prefixes.

```json
{
  "name": "windows-debug",
  "inherits": "windows",
  "binaryDir": "${sourceDir}/build-debug",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "RelWithDebInfo",
    "BUILD_TESTING": "ON"
  }
}
```

`RelWithDebInfo` is the working default. It emits full PDBs while keeping host optimization, which
matters because a `Debug` build of this template-heavy C++20 and CUDA source is slow enough that a
full model load and decode becomes impractical. Drop `CMAKE_BUILD_TYPE` to `Debug` only when a
specific host translation unit resists stepping.

Device code is already compiled with `-lineinfo` in `src/CMakeLists.txt`, so sanitizer and profiler
reports resolve to kernel source lines in every configuration, Release included. Source-level
device *debugging* additionally needs `-G`:

```json
{
  "name": "windows-cuda-debug",
  "inherits": "windows-debug",
  "binaryDir": "${sourceDir}/build-cuda-debug",
  "cacheVariables": {
    "CMAKE_CUDA_FLAGS": "-G"
  }
}
```

`-G` disables device optimization entirely and changes register allocation and scheduling. Kernels
become several times slower, and timing-dependent defects frequently stop reproducing under it.
`nvcc` reports that `-G` overrides the project's `-lineinfo`; that is expected and harmless.
Reserve this configuration for a specific kernel under a specific breakpoint.

## Host-side debugging

The repository builds with Ninja, a single-config generator, so there is no solution file. Visual
Studio reads `CMakePresets.json` and `CMakeUserPresets.json` directly through **File > Open >
Folder**; select the `windows-debug` preset in the configuration dropdown and breakpoints, watches,
and call stacks work normally.

To debug an already-built executable without opening the folder:

```bat
devenv /debugexe build-debug\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
```

For the server, start `ninfer-serve` normally and use **Debug > Attach to Process**, which avoids
restarting a process that has already loaded weights.

## Reporting device errors at the right call site

Kernel launches are asynchronous, so a fault surfaces at whichever later synchronizing call
observes it — typically an unrelated copy or a stream synchronize — rather than at the launch that
caused it. Serialize launches while investigating:

```bat
set CUDA_LAUNCH_BLOCKING=1
```

Every launch then synchronizes, and the `CUDA_CHECK` macro from `src/core/device.h` fires at the
responsible line. Unset it afterwards; it changes both performance and the interleaving that some
defects depend on.

## compute-sanitizer

`compute-sanitizer` ships with the CUDA toolkit and is the device-side equivalent of a memory
sanitizer. It needs no special build — the Release or `windows-debug` binary works, and
`-lineinfo` already gives it source lines.

```bat
compute-sanitizer --tool memcheck  build-debug\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
compute-sanitizer --tool racecheck build-debug\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
compute-sanitizer --tool initcheck build-debug\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
compute-sanitizer --tool synccheck build-debug\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
```

- `memcheck` — out-of-bounds and misaligned global-memory access;
- `racecheck` — shared-memory races;
- `initcheck` — reads of uninitialized device memory;
- `synccheck` — invalid `__syncthreads` and warp-synchronization use.

Expect a large slowdown, so keep the prompt short and cap the decode length. `racecheck` and
`initcheck` matter most for attention, paged-KV, and reduction code, where the failure mode is a
slightly wrong value rather than a fault.

The same tools run over a single test binary, which is usually the faster loop:

```bat
compute-sanitizer --tool memcheck build-debug\tests\<test-binary>.exe
```

## In-kernel printf

`printf` is available in device code and is an appropriate tool, not a fallback. Guard it by thread
and block index or the output volume makes it useless:

```cuda
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("layer=%d acc=%f\n", layer, static_cast<float>(acc));
}
```

Output flushes at synchronization points, so pair it with `CUDA_LAUNCH_BLOCKING=1` when ordering
against host logging matters.

## Nsight Visual Studio Edition

Nsight Visual Studio Edition installs with the CUDA toolkit and is the only source-level CUDA
debugger on Windows; `cuda-gdb` is Linux-only. It provides breakpoints inside `__global__` and
`__device__` code, register and shared-memory inspection, and a focus selector for choosing which
block and thread the inspection windows follow. Use the `windows-cuda-debug` configuration.

On a single-GPU workstation, debugging kernels on the display adapter can trip the Windows
display-driver timeout and reset the device. That is a constraint of the configuration rather than
a fault in the build.

## Numerical debugging

Most defects in an inference engine produce plausible but wrong output rather than a crash, and no
debugger localizes that efficiently across a full decoder stack. The reliable procedure is
bisection:

1. Dump the hidden state after each layer to a file and compare against a reference implementation
   executing the same artifact weights. The first layer whose maximum absolute deviation grows
   sharply contains the defect.
2. Reproduce the failing computation as an Op-level test under `tests/ops/`, which compares against
   an independent oracle at supported shapes. [`tests/README.md`](../tests/README.md) describes the
   suite organization, and [Op development](maintainer/op-development.md) defines the Op contract
   and tolerance rules the new test must satisfy.
3. Concentrate on the recurring causes: accumulation order and dtype in reduced-precision paths,
   RoPE and MRoPE index construction, attention masking boundaries, paged-KV page mapping, and
   epsilon placement in normalization.

A defect worth a regression test belongs in the suite permanently, not only for the duration of the
investigation.

## Profiling

Profiling answers a different question than debugging, but uses adjacent tooling and the same
`-lineinfo` build.

```bat
nsys profile --output ninfer-timeline build\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
ncu --set full --kernel-name <kernel> --output ninfer-kernel build\apps\ninfer.exe models\qwen3_6_27b.ninfer --prompt "hello"
```

Start with Nsight Systems, which shows the whole-application timeline and distinguishes GPU-bound
execution from host stalls and synchronization gaps. Move to Nsight Compute only after the timeline
identifies which kernel dominates; it reports occupancy, memory throughput, and stall reasons for
one kernel at high cost. Measured results and their reproduction commands are recorded in
[performance](performance.md); [`bench/README.md`](../bench/README.md) covers the benchmark harness.

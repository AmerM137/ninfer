# Quick benchmark notes

Running notes from `tools/bench/quick_serve_bench.py` and other fast shakeout runs. Newest first.

**These are not published results.** The quick bench is a 2–5 minute functional shakeout with
indicative numbers — single repetition, n=2 timed requests per mode, prefix reuse left on for the
mode stage. The published serving methodology lives in [`docs/performance.md`](docs/performance.md)
and is produced by `tools/bench/run_serve_corpus.py`; concurrency by `run_serve_concurrency.py`;
the Engine route by `ninfer_bench`. Nothing here is a substitute for those, and numbers from this
file should not be quoted as campaign results.

Each entry records the commit it measured so a later regression has something to compare against.

---

## 2026-08-21 — Windows quick serve bench, MSVC, RTX 5090

Measured the binaries built from `e4676446` (`fix(qwen3.6): make plan moves compiler-generated`).
Later commits present during measurement were documentation-only. Windows 11, Release / Ninja /
CUDA 13.3, RTX 5090 32,607 MiB, driver 610.88, cl 19.51.36256. The complete build and its 90-test
suite passed; six opt-in checkpoint/artifact tests skipped.

```powershell
python tools/bench/quick_serve_bench.py `
  --artifact "$HOME\models\ninfer\Qwen3.6-27B-NInfer\qwen3_6_27b.ninfer" `
  --serve "build\apps\ninfer-serve.exe"

python tools/bench/quick_serve_bench.py `
  --artifact "$HOME\models\ninfer\Qwen3.6-35B-A3B-NInfer\qwen3_6_35b_a3b.ninfer" `
  --serve "build\apps\ninfer-serve.exe" `
  --mode mtp0 --mode mtp3 --mode dflash7
```

Both artifacts were `groupwise-int`. The tables use the final complete MSVC run for 35B-A3B;
two earlier decode-only repetitions agreed within 0.4%.

### Prefill

Default stage: prefix reuse off, `--max-context 64768`, 1 output token, 2 reps + discarded warmup.

| model | fixture | prompt tok | prefill s | tok/s | sd | server TTFT ms |
|---|---|---|---|---|---|---|
| qwen3.6-27B | long_niah_8k | 7,678 | 2.27 | 3,388.7 | 5.4 | 2,276 |
| qwen3.6-27B | long_niah_64k | 64,510 | 23.59 | 2,734.4 | 7.7 | 23,675 |
| qwen3.6-35B-A3B | long_niah_8k | 7,678 | 0.49 | **15,726.2** | 5.8 | 498 |
| qwen3.6-35B-A3B | long_niah_64k | 64,510 | 6.01 | **10,726.3** | 3.2 | 6,096 |

### Decode

Mode stage on the server default 8192-token context; means over the 2 timed requests. `vs mtp0`
is within-model and within-build.

| model | mode | decode tok/s | vs mtp0 | prefill ms | TTFT ms | reuse TTFT | tok/round | accept |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27B | mtp0 | 82.2 | 1.00x | 61 | 61 | 16 | – | – |
| qwen3.6-27B | mtp3 | 210.5 | 2.56x | 63 | 64 | 19 | 3.41 | 80% |
| qwen3.6-35B-A3B | mtp0 | 329.7 | 1.00x | 24 | 24 | 6 | – | – |
| qwen3.6-35B-A3B | mtp3 | 677.9 | 2.06x | 26 | 26 | 8 | 3.34 | 78% |
| qwen3.6-35B-A3B | dflash7 | **707.4** | **2.15x** | 26 | 27 | 6 | 5.03 | 58% |

### What stood out

- **35B-A3B remains much faster than dense 27B on Windows**: 4.6x / 3.9x prefill at the 8k / 64k
  fixtures, 4.0x mtp0 decode, and 3.2x mtp3 decode.
- **dflash7 beats mtp3 by 4.4% on Windows** (707.4 vs 677.9), smaller than the Linux quick run's
  14% gap. Repeated Windows runs were stable (707.4–709.5 dflash7, 677.9–679.3 mtp3), so this is
  larger than local timing noise, but the quick runs do not isolate an OS, driver, or scheduling
  cause.
- **The Windows/Linux ordering is consistent, but Linux is faster in this shakeout.** Windows MSVC
  trails the corresponding Linux GCC rows by roughly 3–6% for prefill/mtp0/mtp3 and 13% for
  dflash7. These are separate machines and software stacks, not a controlled compiler comparison.
- **Prefix reuse works in every Windows mode**, reducing the repeated-prompt TTFT to 6–19 ms.

### Caveats specific to this run

- The general quick-benchmark caveats at the top of this file apply: n=2 timed requests, fixed
  greedy prompts, default KV dtype, one GPU, no concurrency, and no pre-change baseline.
- Windows measured qwen3.6-27B and qwen3.6-35B-A3B only; no qwen3.8-27B Windows row was run.
- The serving path prepared two fewer tokens than each fixture's CLI-basis manifest declaration,
  as expected and documented; rates use the actually prepared counts.

The request-log JSONL and server stderr remain in the printed
`%LOCALAPPDATA%\Temp\quick_serve_bench_*` directories on this machine; they were not copied into
the repository.

---

## 2026-08-21 — quick serve bench, three artifacts, RTX 5090

Measured `e4676446` (`fix(qwen3.6): make plan moves compiler-generated`). Tree moved to
`b859474e` during the session, but that commit is docs-only — no source, no CMake — so the
binary built from `e4676446` is what these numbers describe.

Release / Ninja / GCC 16.2.1 / CUDA 13.3, RTX 5090 32,607 MiB, driver 610.57.04. Build at that
commit: `280/280` exit 0; with `BUILD_TESTING=ON`, `197/197` exit 0, zero warnings. Test binaries
were built but **not executed**.

```bash
python3 tools/bench/quick_serve_bench.py --artifact <artifact>
python3 tools/bench/quick_serve_bench.py --artifact <35b-a3b> \
  --no-prefill --mode mtp0 --mode mtp3 --mode dflash7
```

Artifacts from the local model directory: `qwen3_6_27b.ninfer`, `qwen3_8_27b.ninfer`,
`qwen3_6_35b_a3b.ninfer` (all `groupwise-int`).

### Prefill

Default stage: prefix reuse off, `--max-context 64768`, 1 output token, 2 reps + discarded warmup.

| model | fixture | prompt tok | prefill s | tok/s | sd | server TTFT ms |
|---|---|---|---|---|---|---|
| qwen3.6-27B | long_niah_8k | 7,678 | 2.17 | 3,542.1 | 0.1 | 2,184 |
| qwen3.8-27B | long_niah_8k | 7,716 | 2.22 | 3,468.3 | 0.3 | 2,240 |
| qwen3.6-35B-A3B | long_niah_8k | 7,678 | 0.46 | **16,741.2** | 8.8 | 473 |
| qwen3.6-27B | long_niah_64k | 64,510 | 22.81 | 2,827.6 | 1.6 | 22,906 |
| qwen3.8-27B | long_niah_64k | 64,548 | 22.90 | 2,818.7 | 1.7 | 22,990 |
| qwen3.6-35B-A3B | long_niah_64k | 64,510 | 5.75 | **11,213.4** | 2.5 | 5,842 |

### Decode

Mode stage on the server default 8192-token context; means over the 2 timed requests. `vs mtp0`
is within-model.

| model | mode | decode tok/s | vs mtp0 | prefill ms | TTFT ms | reuse TTFT | tok/round | accept |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27B | mtp0 | 86.1 | 1.00x | 54 | 54 | 13 | – | – |
| qwen3.6-27B | mtp3 | 217.2 | 2.52x | 56 | 56 | 15 | 3.41 | 80% |
| qwen3.8-27B | mtp0 | 84.5 | 1.00x | 83 | 84 | 14 | – | – |
| qwen3.8-27B | mtp3 | 200.1 | 2.37x | 85 | 86 | 16 | 3.15 | 72% |
| qwen3.6-35B-A3B | mtp0 | 351.0 | 1.00x | 19 | 20 | 4 | – | – |
| qwen3.6-35B-A3B | mtp3 | 715.8 | 2.04x | 20 | 21 | 5 | 3.34 | 78% |
| qwen3.6-35B-A3B | dflash7 | **816.1** | **2.32x** | 20 | 20 | 4 | 5.03 | 58% |

Mode-stage `prefill ms` is a latency, not a rate — at 31–84 token prompts the pass costs ~20–85 ms
regardless of length. The prefill table above is where speed is measured.

### What stood out

- **The A3B dominates**: ~4x prefill and ~4.1x mtp0 decode against the dense 27Bs. MoE routing cuts
  FLOPs per token as well as bytes read, so prefill benefits nearly as much as decode despite being
  compute-bound rather than bandwidth-bound. 64k TTFT drops from ~23 s to 5.8 s.
- **dflash7 beats mtp3 by 14% on the A3B** (816.1 vs 715.8) while accepting a *lower* fraction,
  58% vs 78%. The longer proposal block (k=7 vs 3) gives 5.03 tokens per round against 3.34. Tokens
  per round is what pays; acceptance rate alone is the wrong thing to optimize.
- **The mtp3 multiplier shrinks as the base model gets faster** — 2.52x on qwen3.6-27B, 2.04x on the
  A3B, from near-identical acceptance (80% / 78%). Speculation trades bandwidth-bound decode for
  compute-bound verify, and on a 3B-active model the verify pass is proportionally more of the cost.
  The A3B's *absolute* gain is still much the largest (+365 tok/s vs +131).
- **qwen3.6-27B vs qwen3.8-27B**: prefill within 2%, mtp0 decode within 2%. The measured difference
  is MTP acceptance — 80% vs 72%, 3.41 vs 3.15 tok/round — but that rests on n=2, see below.
- **mtp0 decode is flat across all four requests on every model** (e.g. 86.0/86.1/86.1/86.1). Plan
  objects are constructed and destroyed per request, so the deleter path added in `e4676446` shows
  no per-request cost or instability.
- **Reproducibility is excellent.** The A3B mtp0/mtp3 stages ran twice: 351.0 vs 351.0 and 715.6 vs
  715.8 tok/s, per-request acceptance identical to the decimal. Requests are greedy, so the token
  stream is deterministic and only timing varies.
- **Prefix reuse works** on every model: TTFT falls to 4–16 ms on the repeated prompt.

### Caveats specific to this run

- **No baseline.** No pre-commit run exists on this box, so these characterize current behavior;
  they do **not** attribute anything to `e4676446`. A paired comparison needs `HEAD~1` built and
  re-run.
- **n=2 timed requests per mode**, 3 fixed prompts. Acceptance swings hard by prompt content —
  dflash7 ranged 36% → 56% → 60%, qwen3.6-27B 69% → 76% → 85%. Mean acceptance gaps of a few points
  are suggestive, not established. The 14% dflash7-over-mtp3 gap is comfortably larger than observed
  run-to-run noise; the 8-point 3.6-vs-3.8 acceptance gap is not similarly safe.
- **Fixture token counts diverge from the manifest**, expected and documented: the serving path
  prepares 2 fewer tokens than the CLI basis (7,678 / 64,510 for the 3.6 family). qwen3.8 diverges
  further (7,716 / 64,548) from its different chat template. Rates come from the actually-prepared
  count, so they are correct; only the fixture labels shift.
- **`req2` on qwen3.8 stopped early** — 483 tokens (mtp0), 519 (mtp3) against the 1024 cap. Decode
  rate is per-token so it stays valid, but that request covers less work than its 3.6 counterpart.
- **KV dtype left at the server default.** The published NIAH rows declare `kv_dtype: int8`.
- **Single GPU, single stream, no concurrency.** Says nothing about behavior under concurrent load.

Verbatim tool output was kept under `profiles/bench/quick-serve-2026-08-21/` (untracked). The
per-request JSONL the tool wrote to `/tmp/quick_serve_bench_*` was not preserved.

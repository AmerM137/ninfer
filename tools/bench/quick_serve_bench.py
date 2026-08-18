"""Quick serving benchmark: one artifact, speculative modes side by side.

Starts ninfer-serve once per requested mode, sends the same fixed greedy
prompts, and reads prefill/decode rates from the request-log JSONL. Runs in
about 2-5 minutes and is meant as a fast functional shakeout with indicative
numbers -- single repetition, short prompts, prefix reuse left ON (one request
deliberately repeats the warm prompt to show the reuse TTFT). It is not the
published serving methodology; that is `run_serve_corpus.py`.

Examples:

    python tools/bench/quick_serve_bench.py --artifact C:/models/qwen3_8_27b.ninfer
    python tools/bench/quick_serve_bench.py --artifact ... --mode mtp3 \
        --serve build-clangcl/apps/ninfer-serve.exe
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import hostexec  # noqa: E402

MODEL_ID = "quick-bench"

# The request log is a versioned contract: pin it so a schema bump fails loudly
# here instead of quietly reading absent fields as zero rates.
SERVER_LOG_ARTIFACT_TYPE = "ninfer_serve_request_log"
SERVER_LOG_SCHEMA_VERSION = 10

# mode name -> (serve --spec backend, --draft-tokens); mirrors run_serve_corpus.py.
SPECULATIVE_MODES = {
    "mtp0": ("none", 0),
    "mtp3": ("mtp", 3),
    "dflash7": ("dflash", 7),
}
DEFAULT_MODES = ("mtp0", "mtp3")

# Fixed in-distribution prompts (prose / code / math) so acceptance rates see
# some content variety. The first prompt doubles as the warm request, and the
# first timed request repeats it to exercise prefix reuse.
PROMPTS = (
    "Explain how speculative decoding works in LLM inference, covering draft "
    "generation, verification, and acceptance, with a worked example.",
    "Write a Python implementation of an LRU cache with O(1) operations and "
    "explain each design decision.",
    "Prove that the sum of the first n odd numbers is n squared, then use the "
    "result to compute the sum of the first 40 odd numbers.",
)


class QuickBenchError(RuntimeError):
    pass


def wait_ready(process: subprocess.Popen, port: int, stderr_path: Path) -> None:
    deadline = time.monotonic() + 600
    while True:
        if process.poll() is not None:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise QuickBenchError(
                f"server died during startup (rc={process.returncode}):\n{tail}"
            )
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise QuickBenchError("server never became ready")
        time.sleep(0.5)


def chat(port: int, prompt: str, max_tokens: int) -> None:
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status != 200:
        raise QuickBenchError(f"request failed {resp.status}: {data[:400]!r}")


def run_mode(args: argparse.Namespace, name: str) -> list[dict]:
    backend, draft_tokens = SPECULATIVE_MODES[name]
    work_dir = Path(tempfile.mkdtemp(prefix=f"quick_serve_bench_{name}_"))
    log_path = work_dir / "request_log.jsonl"
    stderr_path = work_dir / "serve_stderr.txt"
    # Kept, not cleaned: the request log and server stderr are the evidence for
    # every number printed below, so name the directory instead of orphaning it.
    print(f"\n== {name} == logs: {work_dir}", flush=True)
    command = [
        str(args.serve), str(args.artifact),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", MODEL_ID,
        "--device", str(args.device),
        "--request-log-jsonl", str(log_path),
        "--greedy",
    ]
    if backend != "none":
        command += ["--spec", backend, "--draft-tokens", str(draft_tokens),
                    "--lm-head-draft"]
    with stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(command, cwd=REPO_ROOT,
                                   stdout=subprocess.DEVNULL, stderr=stderr_file,
                                   creationflags=hostexec.SERVER_POPEN_FLAGS)
        try:
            wait_ready(process, args.port, stderr_path)
            chat(args.port, PROMPTS[0], args.max_tokens)  # warm, excluded
            for prompt in (PROMPTS[0], *PROMPTS[1:]):  # repeat = prefix reuse
                chat(args.port, prompt, args.max_tokens)
            hostexec.request_stop(process)
            process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=20)

    records = [json.loads(line)
               for line in log_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        identity = (record.get("artifact_type"), record.get("schema_version"))
        expected = (SERVER_LOG_ARTIFACT_TYPE, SERVER_LOG_SCHEMA_VERSION)
        if identity != expected:
            raise QuickBenchError(
                f"unexpected serving log identity {identity!r}; expected {expected!r}")
    done = [r for r in records if r.get("event") == "request_done"]
    # warm + reuse (both PROMPTS[0]) + the remaining prompts
    if len(done) != len(PROMPTS) + 1:
        raise QuickBenchError(
            f"expected {len(PROMPTS) + 1} request_done records, got {len(done)}")
    return done


def request_metrics(record: dict) -> dict:
    result = record.get("result", {})
    timings = record.get("timings_seconds", {})
    spec = record.get("speculative", {})
    generated = result.get("completion_tokens", 0)
    metrics = {
        "prompt_tokens": result.get("prompt_tokens"),
        "generated": generated,
        "prefill_tok_s": (result.get("computed_prefill_tokens", 0) / timings["prefill"]
                          if timings.get("prefill") else 0.0),
        # prefill emits the first token; the remaining generated-1 are decode
        "decode_tok_s": ((generated - 1) / timings["decode"]
                         if timings.get("decode") and generated > 1 else 0.0),
        "ttft_ms": timings.get("ttft", 0.0) * 1000.0,
        "tok_per_round": None,
        "acceptance": None,
    }
    rounds = spec.get("rounds", 0)
    drafted = spec.get("drafted_tokens", 0)
    if spec.get("backend") not in (None, "none") and rounds:
        metrics["tok_per_round"] = 1.0 + spec.get("accepted_tokens", 0) / rounds
        metrics["acceptance"] = (100.0 * spec.get("accepted_tokens", 0) / drafted
                                 if drafted else 0.0)
    return metrics


def report_mode(name: str, done: list[dict]) -> dict:
    timed = []
    for i, record in enumerate(done):
        m = request_metrics(record)
        if i == 0:
            tag = "warm"
        elif i == 1:
            tag = "reuse"  # repeats the warm prompt: TTFT shows prefix reuse
        else:
            tag = f"req{i - 1}"
            timed.append(m)
        line = (f"{tag}: prompt={m['prompt_tokens']} gen={m['generated']} "
                f"prefill={m['prefill_tok_s']:.1f} tok/s "
                f"decode={m['decode_tok_s']:.1f} tok/s ttft={m['ttft_ms']:.0f} ms")
        if m["tok_per_round"] is not None:
            line += (f" | mtp {m['tok_per_round']:.2f} tok/round"
                     f" ({m['acceptance']:.1f}%)")
        print(line, flush=True)
    reuse = request_metrics(done[1])
    return {
        "decode_tok_s": statistics.fmean(m["decode_tok_s"] for m in timed),
        "prefill_tok_s": statistics.fmean(m["prefill_tok_s"] for m in timed),
        "ttft_ms": statistics.fmean(m["ttft_ms"] for m in timed),
        "reuse_ttft_ms": reuse["ttft_ms"],
        "tok_per_round": (statistics.fmean(m["tok_per_round"] for m in timed)
                          if timed[0]["tok_per_round"] is not None else None),
        "acceptance": (statistics.fmean(m["acceptance"] for m in timed)
                       if timed[0]["acceptance"] is not None else None),
    }


def print_summary(summaries: dict[str, dict]) -> None:
    baseline = next(iter(summaries.values()))["decode_tok_s"]
    print("\n== summary (means over timed requests) ==", flush=True)
    header = (f"{'mode':<9}{'prefill tok/s':>14}{'decode tok/s':>14}{'vs first':>10}"
              f"{'ttft ms':>9}{'reuse ttft':>11}{'tok/round':>11}{'accept':>8}")
    print(header, flush=True)
    for name, s in summaries.items():
        speedup = s["decode_tok_s"] / baseline if baseline else 0.0
        per_round = f"{s['tok_per_round']:.2f}" if s["tok_per_round"] is not None else "-"
        accept = f"{s['acceptance']:.0f}%" if s["acceptance"] is not None else "-"
        print(f"{name:<9}{s['prefill_tok_s']:>14.1f}{s['decode_tok_s']:>14.1f}"
              f"{speedup:>9.2f}x{s['ttft_ms']:>9.0f}{s['reuse_ttft_ms']:>11.0f}"
              f"{per_round:>11}{accept:>8}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True,
                        help=".ninfer artifact to serve")
    parser.add_argument("--serve", type=Path,
                        default=hostexec.binary_path("build/apps/ninfer-serve"),
                        help="ninfer-serve binary (default: build/apps)")
    parser.add_argument("--mode", action="append", choices=sorted(SPECULATIVE_MODES),
                        default=None,
                        help=f"speculative mode; repeatable (default: {DEFAULT_MODES})")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max_tokens per request (default: 1024)")
    parser.add_argument("--port", type=int, default=18432)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    args.artifact = args.artifact.expanduser().resolve()
    if not args.artifact.is_file():
        parser.error(f"artifact not found: {args.artifact}")
    if not args.serve.is_file():
        parser.error(f"serve binary not found: {args.serve} (build it first)")
    modes = args.mode or list(DEFAULT_MODES)

    summaries: dict[str, dict] = {}
    failures = 0
    for name in modes:
        try:
            summaries[name] = report_mode(name, run_mode(args, name))
        except QuickBenchError as error:
            failures += 1
            print(f"\n== {name} ==\nFAILED: {error}", file=sys.stderr, flush=True)
    if summaries:
        print_summary(summaries)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import contextlib
import hashlib
import http.client
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import hostexec  # noqa: E402

MODEL_ID = "quick-bench"

# The request log is a versioned contract: pin it so a schema bump fails loudly
# here instead of quietly reading absent fields as zero rates.
SERVER_LOG_ARTIFACT_TYPE = "ninfer_serve_request_log"
SERVER_LOG_SCHEMA_VERSION = 10

MANIFEST_PATH = REPO_ROOT / "examples/cli/manifest.json"

# Prefill points come from the committed CLI fixtures. Their declared
# prompt_tokens are exact Engine preparation counts (manifest
# prompt_token_basis), so a rate carries a real length instead of an estimate,
# and they are the same fixtures the published campaign measures. 7,680 and
# 64,512 tokens are the two cheapest rungs of that NIAH ladder; long_niah_128k
# and long_niah_256k are selectable and cost roughly one and three minutes per
# repetition.
DEFAULT_PREFILL_FIXTURES = ("long_niah_8k", "long_niah_64k")

# One output token by default: the request is then prefill plus a single decode
# step, so timings_seconds.prefill is the measurement and nothing else competes
# with it. --prefill-max-tokens raises it, which is how a point can be taken
# under the published campaign's 128-token NIAH shape instead.
DEFAULT_PREFILL_MAX_TOKENS = 1

# max_context is fixed at startup, so the prefill stage runs its own server
# sized for the largest selected fixture. That keeps the shakeout modes on the
# server default (8192), where their numbers stay comparable with earlier runs.
PREFILL_CONTEXT_HEADROOM = 256

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


def chat(port: int, messages: list[dict], max_tokens: int) -> None:
    body = json.dumps({
        "model": MODEL_ID,
        "messages": messages,
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


def server_command(args: argparse.Namespace, log_path: Path, backend: str,
                   draft_tokens: int, max_context: int | None = None,
                   measurement: bool = False) -> list[str]:
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
    if max_context is not None:
        command += ["--max-context", str(max_context)]
    if args.kv_dtype is not None:
        command += ["--kv-dtype", args.kv_dtype]
    if measurement:
        # The two flags run_serve_corpus.py uses for every published row, so a
        # rate from this tool is comparable with those: no prefix-cache
        # bookkeeping in the measured path, and no periodic stats thread
        # competing with it.
        command += ["--no-prefix-reuse", "--log-stats-interval-ms", "0"]
    return command


def work_directory(label: str) -> Path:
    # Kept, not cleaned: the request log and server stderr are the evidence for
    # every number printed below, so name the directory instead of orphaning it.
    work_dir = Path(tempfile.mkdtemp(prefix=f"quick_serve_bench_{label}_"))
    print(f"logs: {work_dir}", flush=True)
    return work_dir


# Start the server, hand control back once /health answers, then stop it through
# hostexec.request_stop (CTRL_BREAK_EVENT on Windows) so the request log this
# tool reads afterwards ends with its final records instead of being cut off.
@contextlib.contextmanager
def running_server(args: argparse.Namespace, command: list[str],
                   stderr_path: Path) -> Iterator[None]:
    with stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(command, cwd=REPO_ROOT,
                                   stdout=subprocess.DEVNULL, stderr=stderr_file,
                                   creationflags=hostexec.SERVER_POPEN_FLAGS)
        try:
            wait_ready(process, args.port, stderr_path)
            yield
            hostexec.request_stop(process)
            process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=20)


def read_log(log_path: Path) -> list[dict]:
    records = [json.loads(line)
               for line in log_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        identity = (record.get("artifact_type"), record.get("schema_version"))
        expected = (SERVER_LOG_ARTIFACT_TYPE, SERVER_LOG_SCHEMA_VERSION)
        if identity != expected:
            raise QuickBenchError(
                f"unexpected serving log identity {identity!r}; expected {expected!r}")
    return records


def user_turn(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def run_mode(args: argparse.Namespace, name: str) -> list[dict]:
    backend, draft_tokens = SPECULATIVE_MODES[name]
    print(f"{chr(10)}== {name} ==", flush=True)
    work_dir = work_directory(name)
    log_path = work_dir / "request_log.jsonl"
    command = server_command(args, log_path, backend, draft_tokens)
    with running_server(args, command, work_dir / "serve_stderr.txt"):
        chat(args.port, user_turn(PROMPTS[0]), args.max_tokens)  # warm, excluded
        for prompt in (PROMPTS[0], *PROMPTS[1:]):  # repeat = prefix reuse
            chat(args.port, user_turn(prompt), args.max_tokens)

    done = [r for r in read_log(log_path) if r.get("event") == "request_done"]
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
        # Latency, not a rate: a short prefill pass costs about the same whatever
        # its exact length, so tok/s here would report the prompt length. The
        # prefill stage measures the rate, on prompts long enough to carry one.
        "prefill_ms": (timings.get("prefill") or 0.0) * 1000.0,
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
                f"prefill={m['prefill_ms']:.0f} ms "
                f"decode={m['decode_tok_s']:.1f} tok/s ttft={m['ttft_ms']:.0f} ms")
        if m["tok_per_round"] is not None:
            line += (f" | mtp {m['tok_per_round']:.2f} tok/round"
                     f" ({m['acceptance']:.1f}%)")
        print(line, flush=True)
    reuse = request_metrics(done[1])
    return {
        "decode_tok_s": statistics.fmean(m["decode_tok_s"] for m in timed),
        "prefill_ms": statistics.fmean(m["prefill_ms"] for m in timed),
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
    header = (f"{'mode':<9}{'prefill ms':>12}{'decode tok/s':>14}{'vs first':>10}"
              f"{'ttft ms':>9}{'reuse ttft':>11}{'tok/round':>11}{'accept':>8}")
    print(header, flush=True)
    for name, s in summaries.items():
        speedup = s["decode_tok_s"] / baseline if baseline else 0.0
        per_round = f"{s['tok_per_round']:.2f}" if s["tok_per_round"] is not None else "-"
        accept = f"{s['acceptance']:.0f}%" if s["acceptance"] is not None else "-"
        print(f"{name:<9}{s['prefill_ms']:>12.0f}{s['decode_tok_s']:>14.1f}"
              f"{speedup:>9.2f}x{s['ttft_ms']:>9.0f}{s['reuse_ttft_ms']:>11.0f}"
              f"{per_round:>11}{accept:>8}", flush=True)


class PrefillFixture(NamedTuple):
    name: str
    messages: list[dict]
    prompt_tokens: int


def load_prefill_fixtures(names: Sequence[str]) -> list[PrefillFixture]:
    """Load committed manifest fixtures, refusing any whose bytes have drifted.

    The digest check is the point of using committed fixtures: the declared
    prompt_tokens labels every rate reported for a fixture, so an edited messages
    file must not pass silently under the old length.
    """
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuickBenchError(f"failed to read {MANIFEST_PATH}: {error}") from error
    cases = {case["name"]: case for case in manifest["cases"]}
    fixtures = []
    for name in names:
        case = cases.get(name)
        if case is None:
            raise QuickBenchError(f"unknown fixture {name!r} in {MANIFEST_PATH}")
        messages_path = MANIFEST_PATH.parent / case["messages"]
        try:
            raw = messages_path.read_bytes()
        except OSError as error:
            raise QuickBenchError(f"failed to read {messages_path}: {error}") from error
        expected_digest = case.get("messages_sha256")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise QuickBenchError(
                f"fixture {name!r} does not match its manifest digest: {messages_path}")
        fixtures.append(PrefillFixture(name, json.loads(raw.decode("utf-8")),
                                       int(case["prompt_tokens"])))
    return fixtures


def prefill_metrics(record: dict) -> dict:
    result = record.get("result", {})
    timings = record.get("timings_seconds", {})
    computed = result.get("computed_prefill_tokens", 0)
    prefill_seconds = timings.get("prefill") or 0.0
    prepare_seconds = timings.get("prepare", 0.0)
    vision_seconds = timings.get("vision", 0.0)
    return {
        "prompt_tokens": result.get("prompt_tokens", 0),
        "computed_prefill_tokens": computed,
        "prefix_cache_hit_tokens": result.get("prefix_cache_hit_tokens", 0),
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": computed / prefill_seconds if prefill_seconds else 0.0,
        "prepare_ms": prepare_seconds * 1000.0,
        # run_serve_corpus.py's definition: prepare + vision + prefill, not the
        # log's ttft field, so the two tools' TTFT columns mean the same thing.
        "server_ttft_ms": (prepare_seconds + vision_seconds + prefill_seconds) * 1000.0,
    }


def run_prefill(args: argparse.Namespace,
                fixtures: list[PrefillFixture]) -> list[dict]:
    """Measure prefill on prompts long enough for the rate to mean something.

    Each request asks for a single output token, so timings_seconds.prefill is
    the whole measurement. This stage runs its own server with max_context sized
    for the largest fixture, because max_context is fixed at startup and the
    shakeout modes are deliberately left on the server default.
    """
    print(f"{chr(10)}== prefill ==", flush=True)
    work_dir = work_directory("prefill")
    log_path = work_dir / "request_log.jsonl"
    max_context = args.prefill_max_context or (
        max(f.prompt_tokens for f in fixtures) + PREFILL_CONTEXT_HEADROOM)
    backend, draft_tokens = SPECULATIVE_MODES[args.prefill_mode]
    command = server_command(args, log_path, backend, draft_tokens, max_context,
                             measurement=True)
    print(f"server: max-context={max_context} mode={args.prefill_mode} "
          f"kv-dtype={args.kv_dtype or 'default'} "
          f"repetitions={args.prefill_reps} max-tokens={args.prefill_max_tokens} "
          f"prefix-reuse=off warmup=1", flush=True)

    plan = [(fixture, repetition)
            for fixture in fixtures
            for repetition in range(1, args.prefill_reps + 1)]
    with running_server(args, command, work_dir / "serve_stderr.txt"):
        # One excluded warm-up on the first fixture, as run_serve_corpus.py does
        # before its own measurements: the first prefill on a freshly loaded
        # server measures graph capture and cold clocks as well as prefill, and
        # came out several percent slow against the published rows without it.
        chat(args.port, fixtures[0].messages, args.prefill_max_tokens)
        for fixture, _ in plan:
            # Verbatim, not marked per repetition: prefix reuse is off on this
            # server, so the measured prompt_tokens is the fixture's declared
            # length and the label on the rate is exact.
            chat(args.port, fixture.messages, args.prefill_max_tokens)

    done = [r for r in read_log(log_path) if r.get("event") == "request_done"]
    if len(done) != len(plan) + 1:
        raise QuickBenchError(
            f"expected {len(plan) + 1} request_done records, got {len(done)}")
    done = done[1:]  # drop the warm-up

    rows = []
    for fixture in fixtures:
        samples = [prefill_metrics(record)
                   for (planned, _), record in zip(plan, done)
                   if planned.name == fixture.name]
        contaminated = [s for s in samples if s["prefix_cache_hit_tokens"]]
        if contaminated:
            # Refused rather than reported: a cache hit means the measured
            # seconds cover fewer tokens than the label claims, which reads as a
            # prefill speedup that never happened.
            raise QuickBenchError(
                f"{fixture.name}: {len(contaminated)} of {len(samples)} repetitions "
                "were served from the prefix cache; prefill rate would be overstated")
        measured_tokens = samples[0]["prompt_tokens"]
        if measured_tokens != fixture.prompt_tokens:
            print(f"NOTE: {fixture.name} prepared {measured_tokens} tokens, "
                  f"manifest declares {fixture.prompt_tokens}", flush=True)
        row = {
            "fixture": fixture.name,
            "declared_tokens": fixture.prompt_tokens,
            "prompt_tokens": measured_tokens,
            "reps": len(samples),
            "prefill_seconds": statistics.fmean(s["prefill_seconds"] for s in samples),
            "prefill_tok_s": statistics.fmean(s["prefill_tok_s"] for s in samples),
            "prefill_tok_s_sd": (statistics.stdev(s["prefill_tok_s"] for s in samples)
                                 if len(samples) > 1 else 0.0),
            "prepare_ms": statistics.fmean(s["prepare_ms"] for s in samples),
            "server_ttft_ms": statistics.fmean(s["server_ttft_ms"] for s in samples),
        }
        rows.append(row)
        print(f"{fixture.name}: prompt={row['prompt_tokens']} "
              f"prefill={row['prefill_seconds']:.2f} s "
              f"{row['prefill_tok_s']:.1f} tok/s "
              f"(sd {row['prefill_tok_s_sd']:.1f}, n={row['reps']}) "
              f"server-ttft={row['server_ttft_ms']:.0f} ms", flush=True)
    return rows


def print_prefill_summary(rows: list[dict]) -> None:
    print(f"{chr(10)}== prefill summary (one output token per request) ==", flush=True)
    header = (f"{'fixture':<18}{'prompt tok':>11}{'reps':>6}{'prefill s':>11}"
              f"{'prefill tok/s':>15}{'sd':>8}{'prepare ms':>12}"
              f"{'server ttft ms':>16}")
    print(header, flush=True)
    for row in rows:
        print(f"{row['fixture']:<18}{row['prompt_tokens']:>11}{row['reps']:>6}"
              f"{row['prefill_seconds']:>11.2f}{row['prefill_tok_s']:>15.1f}"
              f"{row['prefill_tok_s_sd']:>8.1f}{row['prepare_ms']:>12.1f}"
              f"{row['server_ttft_ms']:>16.0f}", flush=True)


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
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default=None,
                        help="server --kv-dtype (default: the server default)")
    parser.add_argument("--prefill-fixture", action="append", default=None,
                        metavar="NAME",
                        help="examples/cli manifest case for the prefill stage; "
                             "repeatable (default: "
                             f"{' '.join(DEFAULT_PREFILL_FIXTURES)})")
    parser.add_argument("--prefill-reps", type=int, default=2,
                        help="prefill repetitions per fixture (default: 2)")
    parser.add_argument("--prefill-max-tokens", type=int,
                        default=DEFAULT_PREFILL_MAX_TOKENS, metavar="N",
                        help="output tokens per prefill request "
                             f"(default: {DEFAULT_PREFILL_MAX_TOKENS}; the published "
                             "NIAH rows use 128)")
    parser.add_argument("--prefill-max-context", type=int, default=None,
                        metavar="N",
                        help="server --max-context for the prefill stage "
                             "(default: largest fixture plus headroom; pass 262144 "
                             "to match the published campaign exactly)")
    parser.add_argument("--prefill-mode", choices=sorted(SPECULATIVE_MODES),
                        default="mtp0",
                        help="speculative mode for the prefill stage "
                             "(default: mtp0; prefill is not a speculative path)")
    parser.add_argument("--no-prefill", action="store_true",
                        help="skip the prefill stage")
    args = parser.parse_args(argv)

    args.artifact = args.artifact.expanduser().resolve()
    if not args.artifact.is_file():
        parser.error(f"artifact not found: {args.artifact}")
    if not args.serve.is_file():
        parser.error(f"serve binary not found: {args.serve} (build it first)")
    modes = args.mode or list(DEFAULT_MODES)
    if args.prefill_reps < 1:
        parser.error("--prefill-reps must be at least 1")

    prefill_rows: list[dict] = []
    failures = 0
    if not args.no_prefill:
        try:
            fixtures = load_prefill_fixtures(
                args.prefill_fixture or list(DEFAULT_PREFILL_FIXTURES))
            prefill_rows = run_prefill(args, fixtures)
        except QuickBenchError as error:
            failures += 1
            print(f"{chr(10)}== prefill =={chr(10)}FAILED: {error}",
                  file=sys.stderr, flush=True)

    summaries: dict[str, dict] = {}
    for name in modes:
        try:
            summaries[name] = report_mode(name, run_mode(args, name))
        except QuickBenchError as error:
            failures += 1
            print(f"\n== {name} ==\nFAILED: {error}", file=sys.stderr, flush=True)
    if prefill_rows:
        print_prefill_summary(prefill_rows)
    if summaries:
        print_summary(summaries)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

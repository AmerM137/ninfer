# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Op qualification and CUDA implementation rules
are defined in [Op development](../docs/maintainer/op-development.md); product execution and
ownership contracts are defined in [Engine architecture](../docs/maintainer/engine-architecture.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- `runtime/` — scheduler fairness, cache resource policy, capacity and context-cost calculations;
- `cpu/` — topology/ISA, bounded task admission, bound collectives, publication and shutdown;
- `bench/ttft/` — HTTP measurements, campaign orchestration and workload construction;
- root C++ tests — core storage, public API, serving protocols, logging, benchmark reports and
  causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.
`ops/op_tester.h` and `ops/op_check.h` own reusable device/guard and comparison mechanics.
`ops/linear_common.h` shares extent checks, explicit sampling recipes, payload preservation and
0/1/77 entry handling; each projection family keeps its own fixtures and complete oracle.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

Family-specific case declarations, independent oracles and numerical criteria live alongside the
Op suites. `ops/quantized_weight.h` owns packed-weight fixtures and independent logical decoding;
`ops/direct_bf16_weight.h` owns direct BF16 fixtures. GDN record/fold tests preserve their specified
snapshot reference separately from projection and recurrence mathematical qualification.

For segmented projection, run the original 2560-input suite and the migrated two/multi-output suites:
`ctest --test-dir build -R '^ninfer_linear_split(_two_output|_multi_output)?_test$' --output-on-failure`.
`ninfer_qsa_sparse_gqa_test --kv-cache bf16|fp8|int8|nvfp4|k8v4` selects one cache profile;
the default checks all five. QSA tail, prepare, indexer and sparse-GQA partial-initialization cases
also check their permitted read footprints under Compute Sanitizer initcheck.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

CPU foundation tests link only the C++ `ninfer_cpu` target and need no GPU:
`ctest --test-dir build -R '^ninfer_cpu_.*_test$' --output-on-failure`.
The memory test also exercises CPU Tensor/arena computation. Frontend and HTTP transport tests
cover the task-pool consumers, including restricted media affinity and HTTP queue saturation.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

`ninfer_softmax_attention_test --causal-only` runs both D256 geometries and all five KV types;
`--kv-dtype bf16|int8|fp8|nvfp4|k8v4` selects the same complete causal suite for one type.
The default invocation also runs packed and context attention. Large causal oracle calculations
use at most eight CPU workers while preserving each output's FP64 accumulation order; small cases
remain serial. The runner reports elapsed time per KV type.

Run the RMSNorm and GDN control families together:

```bash
cmake --build build -j --target \
  ninfer_rmsnorm_test ninfer_gated_rmsnorm_test \
  ninfer_gdn_control_test ninfer_gdn_control_proj_test ninfer_gdn_norm_control_proj_test
ctest --test-dir build --output-on-failure \
  -R '^ninfer_(rmsnorm|gated_rmsnorm|gdn_control|gdn_control_proj|gdn_norm_control_proj)_test$'
```

Numerical criteria live in [test_gated.cpp](ops/rmsnorm/test_gated.cpp) and
[GDN criteria.h](ops/gdn_control/criteria.h). `NINFER_OP_REPORT_STATS=1` with verbose CTest output
reports the same suite-owned criteria used by the normal verdict.

Run the HC residual family:

```bash
cmake --build build -j --target \
  ninfer_hc_expand_norm_test ninfer_hc_grouped_rmsnorm_test \
  ninfer_hc_combine_test ninfer_hc_combine_norm_test ninfer_hc_mix_test
ctest --test-dir build --output-on-failure \
  -R '^ninfer_hc_(expand_norm|grouped_rmsnorm|combine|combine_norm|mix)_test$'
```

HC suites qualify the public [gated residual forms](../include/ninfer/ops/gated_residual.h),
including Mix, using [family-owned criteria](ops/gated_residual/criteria.h). Exact copying and
mathematical outputs retain their respective references and the public BF16 residual boundary.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_q8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

For a change confined to one Q8 geometry, use the same public conformance cases with
`./build/tests/ninfer_linear_q8_a16_test --shape N K`. The default CTest invocation still covers
all registered Q8 geometries.

For a change confined to one BF16 geometry, use
`./build/tests/ninfer_linear_bf16_a16_test --shape N K`. The filter covers all registered BF16
geometries, including the original projections and selector matrix. The default runs all cases.

Linear (including both BF16 fixture recipes), LinearAdd (including FP8/NVFP4) and LinearSwiGLU
common implementations each compile once into a test support library. Their libraries and CXX Op test executables receive `-fno-fast-math` and `-ffp-contract=off`
on GNU/Clang. The remaining organization, entry-point and oracle-build work is tracked in the
[Linear test reorganization requirements](../docs/maintainer/linear-test-reorganization-plan.md).

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert tests/bench/ttft \
  tests/test_serve_corpus.py
```

The Python suites exercise conversion, encoded output and measurement tools without model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Real-artifact targets
carry the `real` label and `RUN_SERIAL` so CTest runs them alone. Run directly invoked GPU
integration tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as
`vision`, `late-instructions` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The `attention` scenario checks the selected KV type, chunked prefill, concurrent Graph decode
across a resource tier, prefix continuation, and workspace bounds:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=attention NINFER_TEST_KV_DTYPE=fp8 \
NINFER_TEST_SPECULATIVE=mtp NINFER_TEST_BATCH=2 \
  ./build/tests/ninfer_qwen3_5_prefix_real_test
```

KV choices are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`; backend choices are `none`, `mtp`,
`dflash`, and `dflash2`, requiring an artifact with the selected component. Batch defaults to 2;
`NINFER_TEST_DRAFT_TOKENS` overrides the default MTP3 or DFlash7 block. The DFlash2-specific
integration executable also accepts all five KV names as its fifth positional argument and rejects
unknown names.

Continuation and pressure recovery have dedicated entries:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_agent_continuation_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
NINFER_TEST_BACKEND=dflash2 \
  ctest --test-dir build -R ninfer_qwen3_5_preemption_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ./build/tests/ninfer_qwen3_5_native_transactions_test dflash2
```

The agent entry checks multi-turn continuation and branching. The preemption entry exercises
Snapshot/Replay recovery and cancellation while paused or replaying; `NINFER_PREEMPTION_REAL_SCENARIO`
selects `all`, `snapshot`, `replay`, `cancel-paused` or `cancel-replay`. Native transaction tests cover
physical state/KV ownership, binding, capture, reclamation and abort; their positional backend is
`none`, `mtp`, `dflash` or `dflash2`. Each backend requires an artifact containing that component.
Public-HTTP latency and output gaps are measured separately by the
[TTFT campaign](../tools/bench/ttft/README.md).

`ninfer_qwen3_5_tools_real_test [none|mtp|dflash|dflash2] [graph|eager|basic|snapshot|replay|cancel] [concurrency]`
uses `NINFER_TEST_ARTIFACT` for strict tools, thinking, raw continuation, mixed batches, and
call/result prefix reuse, and required-tool → JSON continuation with committed constraint observations.
`basic` checks default constraints with open/complex schemas, continuation,
streaming and mixed strict/basic/free rows. Snapshot/Replay modes force resource pressure and
validate the completed argument value after recovery; `cancel` interrupts the paused request.
`NINFER_TEST_TOOL_REPORT` appends schema/output/timing JSONL.
`python3 tests/models/qwen3_5/test_tool_schema.py` checks the native Qwen grammar and decoder against
`jsonschema` (dependencies in `tests/text/requirements.txt`), including string pattern/Unicode-length
intersections. The JSON Schema oracle also covers finite-value filtering, reference/union siblings,
closed-object and positional-array intersections, draft-07 tuples, recursive conjunctions,
numeric endpoints and JSON publication rounding. HTTP parsing tests check schema-number precision
before protocol adapters serialize the schema.

`ninfer_qwen3_5_grammar_real_test [none|mtp|dflash|dflash2] [graph|eager] [concurrency] [vision]` uses
`NINFER_TEST_DRAFT_TOKENS` to override the default draft count of three and
`NINFER_TEST_ARTIFACT` to check GBNF/JSON/schema/choice/regex content, sampling, thinking, continuation, prefix reuse
and mixed batches. `ninfer_regex_choice_test` checks literal-set prefix masks and regex edge cases;
`python3 tests/text/test_regex_choice.py` compares regex membership with independent fullmatch semantics.
Set `NINFER_TEST_CONSTRAINT=grammar` or `json_schema` on the preemption test to
check matcher continuity through Snapshot/Replay and cancellation. `ninfer_grammar_test` and
`ninfer_json_schema_test` cover CPU language semantics. `ninfer_json_schema_oracle_test` compares
supported schemas with the independent Python `jsonschema` validator; install its dependency with
`python3 -m pip install -r tests/text/requirements.txt` in the selected test environment.
The sampling and speculative Op tests qualify masks against independent mathematical oracles.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080 --vision
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Structural refactoring preserves existing cases, seeds, input distributions, oracles and numerical
criteria. Treat changes to test data or acceptance as explicit verification changes, identifying
the regression risk they protect and their execution cost.

Test each Op's workspace query against its own execution and interval contract. Shared allocation
helpers do not promise equal capacities across different Ops.

Stateful integration fixtures establish the prerequisites of the path they test. Isolate unrelated
cache history when qualifying restoration; share history deliberately when testing retention or
resource pressure. Enabling prefix reuse does not guarantee a cache hit after arbitrary preceding
work: availability and selection follow the [context-cache policy](../docs/maintainer/resource-scheduling-and-context-cache.md).

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The DFlash prefill regression checks actual KV contents after a StateImage fork and a conflicting
decode binding, including shortened chunks and oversized local/full KV appends. It uses native
Program storage and the production prefill route; select the draft component stored in the artifact:

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash2
NINFER_TEST_ARTIFACT=out/qwen3_6_35b_a3b.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test requires a DFlash2 companion artifact and covers speculative transactions,
page-boundary commit/continuation and retained/fresh prefix behavior. K>=7 exercises a stop inside
a licensed block; K=15 adds oversized prefill, local-ring wrap and the context-capacity tail.
Optional Vision enables image/video and prefix-restore cases. Zero extra Device StateImage slots
exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash2_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16`, `int8`,
`fp8`, `nvfp4`, or `k8v4`), Vision enabled, and extra Device StateImage slots. Defaults are
`15 1 1 8 bf16 0 3`. Run GPU integration tests serially. The individual Op suites remain the
numerical/state-transition oracle; the fixed Engine fixture does not define bit parity across
arbitrary floating-point routes.

## Expert sources and replicas

`ninfer_weight_bank_test` checks local matrix/plane addressing for INT, NVFP4 and block-FP8.
`ninfer_expert_cache_test` checks encoded bytes, variable entry sizes, bounded upload batches, required-set
changes, fragmentation, delayed consumers, Graph replay, independent prefetch readiness and failed
submission cleanup. The byte oracle is the immutable source representation, not another cache.
`ninfer_artifact_cpu_materialization_test` runs with CUDA devices hidden and verifies aligned Host
backing and retained resource lifetimes. The materialization suite also checks dual-destination I/O,
registered/ordinary Host regions and registered backing moves. Registered transfer views and pitched
copies are also exercised by `ninfer_arena_test`.

The opt-in loader test accepts `--host-experts` to put Main experts in registered RAM while retaining MTP and
fixed tensors on GPU. This checks storage and binding, not mixed MoE execution:

```bash
build/tests/ninfer_qwen3_5_loading_real_test --artifact out/qwen3_6_35b_a3b.ninfer \
  --vision --speculative mtp --host-experts
```

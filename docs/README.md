# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

Published artifact downloads are listed in the [project README](../README.md).
[Versioned model cards](../model-cards/) own each artifact's identity, provenance and evaluation results.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for continuation, retention, incremental resources, preemption and recovery |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear tuning and measurement](maintainer/linear-tuning.md) | logical metrics, timing, tuning ranges, dispatch tradeoffs and performance reports |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

## Planned work

Qwen3.8-Flash-Next uses the Qwen4Exp architecture. Its converter and many supporting Ops are
implemented; Engine integration remains pending. The execution design owns the current support
status and remaining requirements. Concrete Op contracts live in `include/ninfer/ops/`.

| Design | Responsibility |
|---|---|
| [Continuation state and context snapshots](maintainer/context-state-redesign.md) | planned StateImage layout, write ownership, KV prefix references, migration and Qwen3.5 integration steps |
| [Qwen4Exp model and weights](maintainer/qwen4_exp-model.md) | architecture, quantization, merged matrices, source binding and residency budgets |
| [Qwen4Exp execution and state](maintainer/qwen4_exp-execution.md) | layer flow, cross-Op handoffs, transactions, state lifetime and remaining integration |
| [Qwen4Exp GPU MoE](maintainer/qwen4_exp-moe-gpu.md) | NVFP4/block-FP8 expert computation, GPU interfaces, fixed Graph topology and workload tuning plan |
| [Qwen4Exp PLE preparation](maintainer/qwen4_exp-ple-io.md) | lookup parameters and inputs, disk preparation, independent H2D slots and cancellation |
| [Expert weight cache](maintainer/expert-weight-cache.md) | affine expert sources, explicit placement, GPU replica cache and pending mixed execution |
| [Linear test maintenance](maintainer/linear-test-reorganization-plan.md) | pending consolidation of test runners and explicit case contracts |

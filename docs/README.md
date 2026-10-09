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

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

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
| [Expert weight cache design](maintainer/expert-weight-cache.md) | planned Qwen3.5-35B-A3B loading modes, complete Host backing for offload, GPU replica storage, whole-layer prefill and CPU/GPU decode |
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
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

## Planned Qwen4Exp support

Qwen3.8-Flash-Next is a Qwen4Exp checkpoint. Start with the support plan for shared conventions,
weight decisions and current progress, then read the relevant execution design. The model is not
yet integrated into the Engine; the converter is implemented and its 20 pure Linear problems
are supported and qualified. A1 sigmoid-gated RMSNorm and A2 BF16 `[96,2560]` control projection
and A3 HC expansion, grouped norm and residual combination are implemented and mathematically
qualified. Their current contracts live in
[rmsnorm.h](../include/ninfer/ops/rmsnorm.h),
[gdn_control.h](../include/ninfer/ops/gdn_control.h), and
[gated_residual.h](../include/ninfer/ops/gated_residual.h). Proposed storage and remaining Op interfaces
stay in these design documents until implemented; the support plan tracks remaining layer/model
integration.

| Design | Responsibility |
|---|---|
| [Support plan and model weights](maintainer/qwen3_8-flash-next-model.md) | entry point, architecture, quantization, merged matrices, artifact/residency, cross-Op handoffs and remaining work |
| [Weight conversion](maintainer/qwen3_8-flash-next-converter.md) | source mapping, logical bindings, ModelOpt import, NF4/MTP encoding, artifact writing and the local command |
| [Main workspace tasks](maintainer/qwen3_8-flash-next-main-tasks.md) | implementation order, shared storage/state/I/O work, secondary-workspace dependencies and the PLE header plan |
| [Independent Op tasks](maintainer/qwen3_8-flash-next-independent-ops.md) | completed A1/A2/A3 scope, remaining tasks in two batches, fixed PLE interfaces, file ownership and integration boundaries |
| [HC / Gated Residual](maintainer/qwen3_8-flash-next-hc.md) | grouped normalization, mixing, residual fusion, interfaces and Main/MTP boundary features |
| [GDN](maintainer/qwen3_8-flash-next-gdn.md) | full layer flow, projection/convolution/control extensions, sigmoid output gating and ReplaySSM integration |
| [QSA](maintainer/qwen3_8-flash-next-qsa.md) | preparation, indexer, sparse GQA, selection, candidate records and tail state |
| [Shared KV page extension](maintainer/qwen3_8-flash-next-paged-kv.md) | planned plane row spans, capacity, coverage, COW and offload; current storage remains in the stable Paged KV reference |
| [PLE](maintainer/qwen3_8-flash-next-ple.md) | NF4 SSD binding, async row reads, prefetch, gated convolution, HC handoff and accepted-prefix state |

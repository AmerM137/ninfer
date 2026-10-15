#pragma once

#include "core/layout.h"
#include "models/qwen4_exp/execution/profile.h"
#include "models/common/moe/plan.h"
#include "models/common/qwen/vision/vision.h"

namespace ninfer::models::qwen4_exp::execution {
struct WorkspacePlan {
    std::size_t prefill = 0, ordinary = 0, verify = 0, mtp = 0, finish = 0;
    std::size_t general_bytes = 0, bytes = 0;
    std::optional<qwen::VisionWorkspacePlan> vision;
};

[[nodiscard]] WorkspacePlan plan_workspace(const ModelProfile&, const EngineOptions&,
                                           std::uint32_t chunk, std::int32_t sm_count,
                                           const moe::Plan*);
} // namespace ninfer::models::qwen4_exp::execution

namespace ninfer::models::qwen4_exp::execution::workspace {
struct Roots {
    Tensor residual, normalized, input, output, injection;
};

template <class Arena>
Roots roots(Arena& a, std::int32_t m) {
    return {a.alloc(DType::BF16, {2560, 4, m}), a.alloc(DType::BF16, {2560, 4, m}),
            a.alloc(DType::BF16, {2560, m}), a.alloc(DType::BF16, {2560, m}),
            a.alloc(DType::FP32, {4, m})};
}

struct GdnBuffers {
    Tensor qkv, z, query, key, value, g, beta, recurrent, normalized;
};

template <class Arena>
GdnBuffers gdn(Arena& a, std::int32_t m, bool prefill) {
    return {prefill ? a.alloc(DType::BF16, {10240, m}) : Tensor{},
            a.alloc(DType::BF16, {6144, m}),
            a.alloc(DType::BF16, {128, 16, m}),
            a.alloc(DType::BF16, {128, 16, m}),
            a.alloc(DType::BF16, {128, 48, m}),
            a.alloc(DType::FP32, {48, m}),
            a.alloc(DType::FP32, {48, m}),
            a.alloc(DType::BF16, {6144, m}),
            a.alloc(DType::BF16, {6144, m})};
}

struct QsaBuffers {
    Tensor query, gate, key, value, index_query, index_key, selection, counts, output;
    Tensor raw_records, coordinate_records;
};

template <class Arena>
QsaBuffers qsa(Arena& a, std::int32_t m, bool prefill) {
    return {a.alloc(DType::BF16, {256, 24, m}),
            a.alloc(DType::BF16, {256, 24, m}),
            a.alloc(DType::BF16, {256, 2, m}),
            a.alloc(DType::BF16, {256, 2, m}),
            a.alloc(DType::BF16, {128, 4, m}),
            a.alloc(DType::BF16, {128, m}),
            a.alloc(DType::I32, {512, m}),
            a.alloc(DType::I32, {m}),
            a.alloc(DType::BF16, {256, 24, m}),
            prefill ? a.alloc(DType::BF16, {128, m}) : Tensor{},
            prefill ? a.alloc(DType::I32, {3, m}) : Tensor{}};
}

struct PleBuffers {
    Tensor projection, gated, records;
};

template <class Arena>
PleBuffers ple(Arena& a, std::int32_t m, bool prefill) {
    return {a.alloc(DType::BF16, {12800, m}), a.alloc(DType::BF16, {2560, 4, m}),
            prefill ? a.alloc(DType::BF16, {10240, m}) : Tensor{}};
}

struct ScoreBuffers {
    Tensor logits, targets, logprobs;
};

template <class Arena>
ScoreBuffers score(Arena& a, std::int32_t vocabulary, std::int32_t columns) {
    return {a.alloc(DType::BF16, {vocabulary, columns}), a.alloc(DType::I32, {columns}),
            a.alloc(DType::FP32, {columns})};
}

struct MtpStemBuffers {
    Tensor embedding, normalized_embedding, normalized_hidden, projected_embedding;
};

template <class Arena>
MtpStemBuffers mtp_stem(Arena& a, std::int32_t m) {
    return {a.alloc(DType::BF16, {2560, m}), a.alloc(DType::BF16, {2560, m}),
            a.alloc(DType::BF16, {10240, m}), a.alloc(DType::BF16, {2560, m})};
}
} // namespace ninfer::models::qwen4_exp::execution::workspace

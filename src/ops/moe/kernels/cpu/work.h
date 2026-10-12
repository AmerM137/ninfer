#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cpu {

inline constexpr int hidden = 2048, intermediate = 512, experts = 256, top_k = 8;
inline constexpr int max_token_block = 8;

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

struct MatrixBank {
    Matrix base;
    std::size_t code_stride, high_stride, scale_stride;

    [[nodiscard]] Matrix matrix(int expert) const noexcept {
        return {base.codes + expert * code_stride,
                base.high ? base.high + expert * high_stride : nullptr,
                reinterpret_cast<const std::uint16_t*>(
                    reinterpret_cast<const std::byte*>(base.scales) + expert * scale_stride)};
    }
};

struct Group {
    Matrix gate_up, down;
    int count;
    int tokens[max_token_block], slots[max_token_block];
    float coefficients[max_token_block];
};

struct Task {
    int group, first, count;
};

struct Scratch {
    Group* groups;
    Task* tasks;
    float* input;
    float* activation;
};

struct Work {
    const std::uint16_t* input;
    const std::int32_t* ids;
    const float* coefficients;
    MatrixBank gate_up, down;
    ExpertSubsetKind subset;
    const std::int32_t* mask;
    float* output;
    int tokens, token_block, max_columns, down_bits;
    Scratch scratch;
    int group_count = 0, task_count = 0;
    bool has_four_columns = false;
};

inline bool participates(ExpertSubsetKind kind, const std::int32_t* mask, int expert) noexcept {
    switch (kind) {
    case ExpertSubsetKind::All:
        return true;
    case ExpertSubsetKind::None:
        return false;
    case ExpertSubsetKind::MaskOnes:
        return mask[expert] == 1;
    case ExpertSubsetKind::MaskZeros:
        return mask[expert] == 0;
    }
    return false;
}

void prepare_chunk(Work&, int token_begin, int columns) noexcept;
void run_avx2(void*, const CpuWorkerContext&) noexcept;
void run_avx512(void*, const CpuWorkerContext&) noexcept;
void run_avx512_bf16(void*, const CpuWorkerContext&) noexcept;

} // namespace ninfer::ops::detail::moe_cpu

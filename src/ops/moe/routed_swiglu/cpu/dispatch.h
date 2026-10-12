#pragma once

#include "core/layout.h"
#include "ops/moe/kernels/cpu/work.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_routed_cpu {

struct Plan {
    CpuWork run;
    int max_columns;
    bool prepare_first;
};

struct ShapePolicy {
    int token_block, down_bits;
    Plan (*select)(CpuFeatures);
};

enum class Isa { Avx2, Avx512, Avx512Bf16 };
Isa select_isa(CpuFeatures features);
const ShapePolicy& resolve_shape(const RoutedSwiGLUProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
void validate_capacity(ExpertSubsetKind subset, int first, int last);

template <class Arena>
moe_cpu::Scratch allocate(Arena& arena, const ShapePolicy& shape, int tokens) {
    using namespace moe_cpu;
    const int columns = std::min(shape.token_block, tokens);
    return {static_cast<Group*>(arena.alloc_bytes(sizeof(Group) * top_k * columns).data),
            static_cast<Task*>(arena.alloc_bytes(sizeof(Task) * top_k * columns).data),
            static_cast<float*>(arena.alloc(DType::FP32, {hidden, columns}).data),
            static_cast<float*>(arena.alloc(DType::FP32, {intermediate, top_k * columns}).data)};
}

inline std::size_t capacity(const ShapePolicy& shape, ExpertSubsetKind subset, int tokens) {
    if (subset == ExpertSubsetKind::None) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, shape, tokens);
    return layout.peak_bytes(1);
}

} // namespace ninfer::ops::detail::moe_routed_cpu

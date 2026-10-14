#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select(int tokens, bool) {
    Plan plan;
    const int columns = tokens <= 16 ? 8 : tokens <= 32 ? 16 : tokens <= 64 ? 32 : 64;
    plan.block_scaled = {.format                = QType::FP8_E4M3FN_BLOCK128X128_BF16,
                         .gate_columns          = tokens >= 257 ? 128 : columns,
                         .down_columns          = columns,
                         .vector                = tokens <= 2,
                         .atomic_output         = tokens >= 128,
                         .single_block_grouping = tokens <= 64};
    return plan;
}
} // namespace

const ShapePolicy& fp8_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{3, 17, 33, 65, 128, 257};
    static const ShapePolicy policy{select, boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed

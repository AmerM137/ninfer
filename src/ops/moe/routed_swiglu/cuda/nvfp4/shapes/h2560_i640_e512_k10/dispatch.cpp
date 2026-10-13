#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select(int tokens, bool) {
    Plan plan;
    const int columns = tokens <= 16 ? 8 : tokens <= 128 ? 32 : 64;
    plan.nvfp4        = {tokens >= 3584 ? 128 : columns, columns, tokens == 1, tokens >= 128,
                  tokens <= 64};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2, 17, 65, 128, 129, 3584};
    static const ShapePolicy policy{select, boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed

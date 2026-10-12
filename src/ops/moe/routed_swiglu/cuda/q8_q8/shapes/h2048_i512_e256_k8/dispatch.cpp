#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {
constexpr int kGroupedFirst = 20;
constexpr int kWideFirst    = 384;
constexpr std::array kAllocationBoundaries{kGroupedFirst, kWideFirst};

Plan select(int tokens, bool shared) {
    Plan plan;
    plan.grouped = tokens >= kGroupedFirst;
    plan.token   = {tokens == 1   ? (shared ? 9 : 8)
                    : tokens <= 5 ? 1
                                  : (shared ? 9 : 8),
                  tokens <= 8 ? 1 : 4, tokens == 1, false};
    plan.group = {tokens >= kWideFirst ? 64 : 32, false, 7, 4, 32, 64};
    return plan;
}
} // namespace

const ShapePolicy& q8_q8_h2048_i512_e256_k8() {
    static const ShapePolicy policy{select, kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed

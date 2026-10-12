#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include <array>

namespace ninfer::ops::detail::moe_routed {
namespace {
constexpr int kGroupedFirst = 47;
constexpr std::array kAllocationBoundaries{kGroupedFirst, 768};

Plan select(int tokens, bool shared) {
    Plan plan;
    plan.grouped = tokens >= kGroupedFirst;
    plan.token   = {tokens == 1 ? (shared ? 9 : 8) : (shared ? 3 : 4),
                  tokens <= 2    ? 1
                    : tokens <= 11 ? 2
                                   : 4,
                  tokens == 1, false};
    plan.group          = {tokens >= 768 ? 64 : 32, false, 7, 4, 32, 64};
    plan.group.adaptive = plan.grouped && tokens <= 128;
    if (plan.group.adaptive) { plan.token = {shared ? 3 : 4, 4, false, false}; }
    return plan;
}
} // namespace

const ShapePolicy& q4_q6_h2048_i512_e256_k8() {
    static const ShapePolicy policy{select, kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
